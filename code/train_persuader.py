"""Train the RL persuader (GPU PPO) on the diffusion environment.

Usage
-----

# 1. Pipeline validation / debug (offline mock LLM, no API needed):
python -m code.train_persuader --mode mock

# 2. Live training (requires .env with LLM_BASE_URL / LLM_API_KEY /
#    LLM_MODEL filled in; see .env.example):
python -m code.train_persuader --mode live

The mock backend validates the full training loop (render -> target ->
diffuse -> reward -> PPO update on GPU) but message content is fixed,
so learning signal is limited. Live mode renders real LLM messages per
strategy, giving the policy a genuine persuasion signal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from .config import GLOBAL_LLM_CONFIG, GLOBAL_EMBEDDING_CONFIG  # noqa: F401
from .rl_env import ElitePersuaderEnv
from .scenario import LLMScenario
from .strategy import N_STRATEGIES
from .torch_trainer import (
    PPOConfig,
    PPOTrainer,
    evaluate_heuristic_baseline,
    evaluate_random_baseline,
)


def build_env(cfg: PPOConfig, observer_inference: bool = False,
              target_selection: str = "learned",
              obs_mode: str = "matrix",
              judge_fn=None,
              align: str = "none",
              goal_delta: bool = False,
              mechanism: str = "paper",
              mech_beta_init: bool = False,
              mech_target_positive_only: bool = False,
              reward_adoption_bonus: float = 0.0) -> ElitePersuaderEnv:
    scenario = LLMScenario(
        agent_num=cfg.agent_num,
        # Observer-side inference (Prompt 5) is not used by the RL
        # reward (ground truth); disabling it saves 3 LLM calls per
        # adoption during training. Re-enable for analysis runs.
        use_post_evidence=observer_inference,
        use_profile_evidence=observer_inference,
    )
    # Track B: goal alignment a(m) + goal-direction injection.
    if align and align != "none":
        scenario.goal_align = align
    scenario.goal_delta = bool(goal_delta) or bool(scenario.goal_delta)
    return ElitePersuaderEnv(
        scenario=scenario,
        n_periods=cfg.n_periods,
        topology=cfg.topology,
        topic=cfg.topic,
        target_selection=target_selection,
        obs_mode=obs_mode,
        reward_cascade_discount=cfg.reward_cascade_discount,
        message_cost=cfg.message_cost,
        shaping_accept_bonus=cfg.shaping_accept_bonus,
        persuade_all_neighbors=cfg.persuade_all_neighbors,
        n_persuaders=cfg.n_persuaders,
        judge_fn=judge_fn,
        mechanism=mechanism,
        mech_beta_init=mech_beta_init,
        mech_target_positive_only=mech_target_positive_only,
        reward_adoption_bonus=reward_adoption_bonus,
    )


def check_llm() -> bool:
    """Fail fast with a helpful message when credentials are missing."""
    if GLOBAL_LLM_CONFIG.has_credentials():
        return True
    print(
        "[!] LLM credentials not configured.\n"
        "    Copy .env.example to .env and fill in:\n"
        "      LLM_BASE_URL=...   (OpenAI-compatible endpoint)\n"
        "      LLM_API_KEY=...    (your key)\n"
        "      LLM_MODEL=...      (model name)\n"
        "    The .env file is gitignored; never commit it.",
        file=sys.stderr,
    )
    return False


def check_embeddings() -> bool:
    """Verify the embedding API in live mode (falls back to hash)."""
    from .tools import embedding_backend

    backend = embedding_backend()
    if backend.startswith("api:"):
        print(f"[+] embedding backend: {backend}")
        return True
    print(
        "[!] embedding API not configured; using deterministic hash "
        "fallback (NOT numerically compatible with the API backend).\n"
        "    Set EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL"
        " in .env for real embeddings.",
        file=sys.stderr,
    )
    return False


def ping_llm() -> bool:
    """One cheap live call to verify the endpoint works."""
    from . import LLM_module

    try:
        reply = LLM_module.call_llm(
            "Reply with the single word: ok",
            temperature=0.0,
            max_tokens=64,
            cache=False,
        )
        print(f"[+] LLM endpoint reachable (reply: {reply[:20]!r})")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[!] LLM ping failed: {exc}", file=sys.stderr)
        return False


def reconstruct_history(log_path: str | Path) -> list[dict]:
    """Rebuild the per-iteration training records from a stdout run log.

    `train_persuader` writes `training_log.json` only when training
    finishes; an interrupted process therefore loses its in-memory
    history. This parses the human-readable log lines emitted by
    `PPOTrainer.train` (logged every `log_every` iters) plus the greedy
    eval lines, so a resumed run can produce a complete curve.
    """
    # Preferred source: a `training_log.json` written incrementally by a
    # run that was interrupted (it carries the full per-iteration record,
    # including entropy/KL). Fall back to parsing the human-readable log.
    path = Path(log_path)
    text = path.read_text()
    if path.suffix == ".json":
        try:
            prior = json.loads(text).get("history", [])
            if prior:
                return prior
        except (json.JSONDecodeError, AttributeError):
            pass

    iter_re = re.compile(
        r"iter (\d+)/\d+ reward=(-?[\d.]+) final_stance=(\S+) "
        r"adopted=(-?[\d.]+) pi_loss=(-?[\d.]+) v_loss=(-?[\d.eE+]+)"
        r"(?: ent=([\d.]+))?.*?\(([\d.]+)s\)"
    )
    eval_re = re.compile(r"eval @ (\d+): greedy mean reward (-?[\d.]+)")

    def _f(tok: str):
        try:
            return float(tok)
        except (ValueError, TypeError):
            return None

    records: dict[int, dict] = {}
    for line in text.splitlines():
        m = iter_re.search(line)
        if m:
            it = int(m.group(1))
            records[it] = {
                "iteration": it,
                "mean_episode_reward": float(m.group(2)),
                "mean_final_stance": _f(m.group(3)),
                "mean_adopted": float(m.group(4)),
                "train_pi_loss": float(m.group(5)),
                "train_v_loss": float(m.group(6)),
                "train_entropy": _f(m.group(7)),
                "train_kl": None,
                "elapsed_s": float(m.group(8)),
                "reconstructed_from_log": True,
            }
            continue
        m = eval_re.search(line)
        if m:
            it = int(m.group(1))
            records.setdefault(
                it,
                {
                    "iteration": it,
                    "reconstructed_from_log": True,
                },
            )
            records[it]["eval_greedy_reward"] = float(m.group(2))
    return [records[k] for k in sorted(records)]


def warmup_render_cache(cfg: "PPOConfig", workers: int | None = None) -> None:
    """Pre-render every (strategy x profile) message into .llm_cache.

    With 30 agents x 180 strategies = 5,400 renders (~1-2s each),
    warmup takes a few minutes with parallel workers but makes ALL
    subsequent training renders instant cache hits. With a fixed
    profile seed the profile pool is finite, so coverage is complete.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from . import LLM_module
    from .profiles import generate_synthetic_profiles
    from .strategy import all_strategies

    profiles = generate_synthetic_profiles(
        cfg.agent_num, seed=1234  # must match rl_env default profile_seed
    )
    strategies = all_strategies()
    topic = cfg.topic
    n_workers = workers or GLOBAL_LLM_CONFIG.max_workers

    jobs = [
        (spec, profile) for profile in profiles for spec in strategies
    ]
    print(
        f"[*] warming render cache: {len(jobs)} (strategy x profile) "
        f"messages, {n_workers} workers"
    )
    t0 = time.time()

    def render_one(job):
        spec, profile = job
        try:
            message = LLM_module.render_strategy_message(
                spec, target_profile=profile, topic=topic,
                stance_direction="opposing",
            )
            if isinstance(message, str):
                return (message, profile)
            return None
        except Exception:  # noqa: BLE001
            return None

    done = ok = 0
    rendered: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(render_one, job) for job in jobs]
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if res is not None:
                ok += 1
                rendered.append(res)
            if done % 250 == 0:
                print(
                    f"    {done}/{len(jobs)} rendered "
                    f"({time.time() - t0:.0f}s elapsed)"
                )
    print(
        f"[+] render cache warmup done: {ok}/{len(jobs)} ok in "
        f"{time.time() - t0:.0f}s"
    )

    # warm the persuasiveness judge cache too (message x profile);
    # without this every training step pays one uncached LLM call
    print(f"[*] warming judge cache: {len(rendered)} pairs")
    t1 = time.time()

    def judge_one(item):
        message, profile = item
        try:
            LLM_module.judge_persuasiveness(message, profile)
            return True
        except Exception:  # noqa: BLE001
            return False

    done = ok = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(judge_one, item) for item in rendered]
        for fut in as_completed(futures):
            done += 1
            if fut.result():
                ok += 1
            if done % 500 == 0:
                print(
                    f"    {done}/{len(rendered)} judged "
                    f"({time.time() - t1:.0f}s elapsed)"
                )
    print(
        f"[+] judge cache warmup done: {ok}/{len(rendered)} ok in "
        f"{time.time() - t1:.0f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["mock", "live"], default="mock",
        help="mock: offline deterministic LLM (pipeline debug); "
             "live: real LLM API via .env",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--episodes-per-iter", type=int, default=None)
    parser.add_argument(
        "--rollout-workers", type=int, default=1,
        help="parallel rollout threads (each runs full episodes; "
             "LLM renders dominate wall-clock, so >1 helps a lot)",
    )
    parser.add_argument(
        "--warmup-cache", action="store_true",
        help="pre-render all (strategy x profile) messages into the "
             "on-disk LLM cache before training (parallel); training "
             "renders then become instant cache hits",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume training from <output-dir>/policy.pt instead of "
             "starting fresh (weights + optimizer state + iteration)",
    )
    parser.add_argument(
        "--resume-log", type=str, default=None,
        help="path to a previous run's stdout log; its per-iteration "
             "records are prepended to training_log.json so a resumed "
             "run keeps the full curve",
    )
    parser.add_argument("--agent-num", type=int, default=None)
    parser.add_argument("--n-persuaders", type=int, default=None,
                        help="RL-controlled seed agents (shared policy)")
    parser.add_argument("--topology", type=str, default=None,
                        help="watts_strogatz / barabasi_albert / "
                             "erdos_renyi / complete")
    parser.add_argument("--target-selection", type=str,
                        default="learned",
                        help="learned (compound target+strategy action) "
                             "or random_neighbor (random own-neighbor "
                             "target; policy chooses strategy only)")
    parser.add_argument("--obs-mode", type=str, default="matrix",
                        help="matrix / summary / pair "
                             "(pair requires random_neighbor)")
    parser.add_argument("--n-periods", type=int, default=None)
    parser.add_argument("--judge", choices=["absolute", "bt"],
                        default="absolute",
                        help="persuasiveness reward model: the absolute "
                             "LLM score, or a Bradley-Terry pairwise judge")
    parser.add_argument("--bt-table", type=str, default=None,
                        help="path to a BT reward-model table (built if "
                             "absent); defaults to <output-dir>/bt_judge.json")
    parser.add_argument("--align", choices=["none", "semdiff", "nli", "judge"],
                        default="none",
                        help="Track B: goal-alignment backend a(m) "
                             "(semdiff reuses the embedding backend)")
    parser.add_argument("--goal-delta", action="store_true",
                        help="Track B: inject a(m) into the stance delta "
                             "(s_eff = s_src·(2a−1)); default off")
    parser.add_argument("--bt-opponents", type=int, default=10,
                        help="pairwise comparisons per message when "
                             "building the BT table")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu (default: auto)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--with-observer", action="store_true",
        help="enable observer-side inference (Prompt 5) during training "
             "(3 extra LLM calls per adoption; only for analysis runs)",
    )
    parser.add_argument(
        "--no-factorized-policy", action="store_true",
        help="use a flat policy head instead of the factorized "
             "(target x appeal x tone x emphasis x intensity) heads",
    )
    parser.add_argument(
        "--entropy-coef-start", type=float, default=None,
        help="entropy bonus at iteration 1, linearly annealed to "
             "--entropy-coef-end (default 0.05 -> 0.005)",
    )
    parser.add_argument(
        "--entropy-coef-end", type=float, default=None,
        help="entropy bonus at the final iteration (see --entropy-coef-start)",
    )
    parser.add_argument(
        "--paired-eval-every", type=int, default=None,
        help="run a paired greedy/random/heuristic evaluation (shared "
             "seeds, lower variance) every N iterations (0=off)",
    )
    parser.add_argument(
        "--paired-eval-episodes", type=int, default=None,
        help="episodes per paired evaluation",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--mechanism", choices=["paper", "cascade", "exposure"],
        default="paper",
        help="transition dynamics: the paper's original sim (default), "
             "or the colleague's two Melodie mechanisms ported in "
             "code/mech_sim.py — 'cascade' = stance-aware within-period "
             "BFS, 'exposure' = stance-less persistent queue with an "
             "LLM accept/reject gate",
    )
    parser.add_argument(
        "--mech-beta-init", action="store_true",
        help="mechanism modes only: initialize crowd stances from the "
             "colleague's Beta mixture (60%% Beta(6,2) support / 40%% "
             "-Beta(8,2) oppose) instead of the constant +0.65",
    )
    parser.add_argument(
        "--mech-positive-only", action="store_true",
        help="mechanism modes only: restrict targets to positive-stance "
             "agents (paper-mode rule; default off = faithful to the "
             "original code, which has no positivity gate)",
    )
    parser.add_argument(
        "--reward-adoption-bonus", type=float, default=0.0,
        help="flat reward per persuader-sourced adoption; recommended "
             ">0 for exposure mode where stance deltas are zero",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="record the IMMEDIATE variables (per-decision "
             "persuasiveness/goal-align/reward, per-period population "
             "state, per-hop diffusion events, episode summaries) as "
             "JSONL under <output-dir>/trace for code/live_monitor.py. "
             "Purely observational — dynamics and rewards are unchanged.",
    )
    parser.add_argument(
        "--trace-dir", type=str, default=None,
        help="override the trace directory (default <output-dir>/trace)",
    )
    parser.add_argument(
        "--trace-episodes", type=int, default=0,
        help="rollout episodes traced per iteration (0 = all); a cap "
             "bounds trace size on long campaigns",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("code.agent").setLevel(logging.WARNING)

    # LLM backend selection --------------------------------------------
    if args.mode == "mock":
        from . import offline

        offline.install()
        print("[+] mock LLM backend installed (offline pipeline mode)")
    else:
        if not check_llm():
            return 2
        if not ping_llm():
            return 2
        check_embeddings()  # warn-only: hash fallback keeps training alive

    # config ------------------------------------------------------------
    cfg = PPOConfig()
    if args.iterations is not None:
        cfg.iterations = args.iterations
    if args.episodes_per_iter is not None:
        cfg.episodes_per_iteration = args.episodes_per_iter
    if args.rollout_workers is not None:
        cfg.rollout_workers = max(1, args.rollout_workers)
    if args.agent_num is not None:
        cfg.agent_num = args.agent_num
    if args.n_persuaders is not None:
        cfg.n_persuaders = args.n_persuaders
    if args.topology is not None:
        cfg.topology = args.topology
    if args.n_periods is not None:
        cfg.n_periods = args.n_periods
    if args.device is not None:
        cfg.device = args.device
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_factorized_policy:
        cfg.factorized_policy = False
    if args.entropy_coef_start is not None:
        cfg.entropy_coef_start = args.entropy_coef_start
    if args.entropy_coef_end is not None:
        cfg.entropy_coef_end = args.entropy_coef_end
    if args.paired_eval_every is not None:
        cfg.paired_eval_every = args.paired_eval_every
    if args.paired_eval_episodes is not None:
        cfg.paired_eval_episodes = args.paired_eval_episodes
    if args.mode == "mock":
        cfg.output_dir = str(Path(cfg.output_dir) / "mock")

    # instrumentation (after output_dir is final, so the default trace
    # directory lands next to the run's other artifacts)
    if args.trace or args.trace_dir:
        cfg.trace_dir = args.trace_dir or str(Path(cfg.output_dir) / "trace")
    cfg.trace_episodes = max(0, int(args.trace_episodes))

    print(
        f"[*] PPO config: agents={cfg.agent_num} persuaders={cfg.n_persuaders} "
        f"topology={cfg.topology} periods={cfg.n_periods} "
        f"target_selection={args.target_selection} "
        f"iters={cfg.iterations} x {cfg.episodes_per_iteration} episodes "
        f"workers={cfg.rollout_workers}"
    )
    print(f"[*] strategy space: {N_STRATEGIES} arms "
          f"(PERSUASION_STRATEGY_SPACE="
          f"{os.environ.get('PERSUASION_STRATEGY_SPACE', 'v1')}), "
          f"judge={args.judge}")
    if cfg.trace_dir:
        print(f"[*] tracing immediate variables -> {cfg.trace_dir} "
              f"({cfg.trace_episodes or 'all'} episodes/iteration)")

    # reward model: Bradley-Terry pairwise judge (optional) ------------
    judge_fn = None
    if args.judge == "bt" and args.mode == "live":
        from .bt_judge import BTJudge, build_table
        from .profiles import generate_synthetic_profiles
        from .strategy import all_strategies

        table_path = Path(args.bt_table) if args.bt_table else (
            Path(cfg.output_dir) / "bt_judge.json"
        )
        # Profile order must match ElitePersuaderEnv.profile_seed (default
        # 1234): the identity index keys on the profile *index*, so a
        # different ordering would silently pair a target with the wrong
        # reader. Keep this seed in lockstep with env default.
        profiles = generate_synthetic_profiles(cfg.agent_num, seed=1234)
        strategies = all_strategies()
        if table_path.is_file():
            judge_fn = BTJudge.load(table_path)
            print(f"[+] loaded BT judge ({len(judge_fn.table)} entries, "
                  f"SD {judge_fn.stats.get('score_sd')}) from {table_path}")
        else:
            judge_fn = build_table(
                strategies, profiles, topic=cfg.topic,
                n_opponents=args.bt_opponents,
                workers=cfg.rollout_workers,
            )
            judge_fn.save(table_path)
            print(f"[+] built BT judge ({judge_fn.stats}) -> {table_path}")

        # Track C: enforce the identity index. A table written before the
        # index existed is migrated in place (cached renders); then full
        # coverage is asserted so a partial table fails LOUD instead of
        # silently falling back to the saturated content/absolute judge
        # at training time.
        expected = len(strategies) * len(profiles)
        if not judge_fn.table_id:
            n_idx = judge_fn.attach_ids(strategies, profiles, topic=cfg.topic)
            print(f"[+] BT judge: derived identity index "
                  f"{n_idx}/{expected} cells (cached renders)")
            judge_fn.save(table_path)
        if len(judge_fn.table_id) < expected:
            missing = expected - len(judge_fn.table_id)
            raise SystemExit(
                f"FATAL: BT judge identity index incomplete — "
                f"{len(judge_fn.table_id)}/{expected} cells "
                f"({missing} missing). Rebuild the table (delete "
                f"{table_path} and re-run) so training cannot silently "
                f"fall back to the saturated content/absolute judge."
            )
        print(f"[+] BT judge identity index: "
              f"{len(judge_fn.table_id)}/{expected} cells (full coverage)")

    # cache warmup: pre-render the full (strategy x profile) matrix ----
    if args.warmup_cache and args.mode == "live":
        warmup_render_cache(cfg, workers=cfg.rollout_workers)

    env = build_env(
        cfg,
        observer_inference=args.with_observer,
        target_selection=args.target_selection,
        obs_mode=args.obs_mode,
        judge_fn=judge_fn,
        align=args.align,
        goal_delta=args.goal_delta,
        mechanism=args.mechanism,
        mech_beta_init=args.mech_beta_init,
        mech_target_positive_only=args.mech_positive_only,
        reward_adoption_bonus=args.reward_adoption_bonus,
    )
    trainer = PPOTrainer(
        env,
        cfg,
        env_factory=lambda: build_env(
            cfg,
            target_selection=args.target_selection,
            obs_mode=args.obs_mode,
            judge_fn=judge_fn,
            align=args.align,
            goal_delta=args.goal_delta,
            mechanism=args.mechanism,
            mech_beta_init=args.mech_beta_init,
            mech_target_positive_only=args.mech_positive_only,
            reward_adoption_bonus=args.reward_adoption_bonus,
        ),
    )
    print(f"[*] training on device: {trainer.device}; "
          f"action_space={env.action_space.n} obs={env.observation_space.shape}")

    # resume from a checkpoint if requested ----------------------------
    resume_from = None
    prior_history = None
    if args.resume:
        resume_from = Path(cfg.output_dir) / "policy.pt"
        if not resume_from.is_file():
            print(f"[!] --resume set but no checkpoint at {resume_from}",
                  file=sys.stderr)
            return 2
        if args.resume_log:
            prior_history = reconstruct_history(args.resume_log)
            print(f"[*] reconstructed {len(prior_history)} prior iteration "
                  f"records from {args.resume_log}")

    policy, history = trainer.train(
        resume_from=resume_from, prior_history=prior_history
    )
    print(f"[*] training complete: {len(history)} iterations recorded")

    # final comparison ---------------------------------------------------
    eval_score = trainer.evaluate(episodes=cfg.eval_episodes)
    random_score = evaluate_random_baseline(
        env, episodes=cfg.eval_episodes, workers=cfg.rollout_workers
    )
    heuristic_score = evaluate_heuristic_baseline(
        env, episodes=cfg.eval_episodes, workers=cfg.rollout_workers
    )

    results = {
        "mode": args.mode,
        "config": vars(cfg),
        "final_greedy_reward": eval_score,
        "random_baseline_reward": random_score,
        "heuristic_baseline_reward": heuristic_score,
        "improvement_over_random": eval_score - random_score,
        "improvement_over_heuristic": eval_score - heuristic_score,
    }
    out = Path(cfg.output_dir) / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n===== RESULTS =====")
    print(f"learned policy (greedy): {eval_score:.4f}")
    print(f"random baseline        : {random_score:.4f}")
    print(f"heuristic baseline     : {heuristic_score:.4f}")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
