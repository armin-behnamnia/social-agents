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
              obs_mode: str = "matrix") -> ElitePersuaderEnv:
    scenario = LLMScenario(
        agent_num=cfg.agent_num,
        # Observer-side inference (Prompt 5) is not used by the RL
        # reward (ground truth); disabling it saves 3 LLM calls per
        # adoption during training. Re-enable for analysis runs.
        use_post_evidence=observer_inference,
        use_profile_evidence=observer_inference,
    )
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
            max_tokens=8,
            cache=False,
        )
        print(f"[+] LLM endpoint reachable (reply: {reply[:20]!r})")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[!] LLM ping failed: {exc}", file=sys.stderr)
        return False


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
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu (default: auto)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--with-observer", action="store_true",
        help="enable observer-side inference (Prompt 5) during training "
             "(3 extra LLM calls per adoption; only for analysis runs)",
    )
    parser.add_argument("--verbose", action="store_true")
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
    if args.mode == "mock":
        cfg.output_dir = str(Path(cfg.output_dir) / "mock")

    print(
        f"[*] PPO config: agents={cfg.agent_num} persuaders={cfg.n_persuaders} "
        f"topology={cfg.topology} periods={cfg.n_periods} "
        f"target_selection={args.target_selection} "
        f"iters={cfg.iterations} x {cfg.episodes_per_iteration} episodes "
        f"workers={cfg.rollout_workers}"
    )

    # cache warmup: pre-render the full (strategy x profile) matrix ----
    if args.warmup_cache and args.mode == "live":
        warmup_render_cache(cfg, workers=cfg.rollout_workers)

    env = build_env(
        cfg,
        observer_inference=args.with_observer,
        target_selection=args.target_selection,
        obs_mode=args.obs_mode,
    )
    trainer = PPOTrainer(
        env,
        cfg,
        env_factory=lambda: build_env(
            cfg,
            target_selection=args.target_selection,
            obs_mode=args.obs_mode,
        ),
    )
    print(f"[*] training on device: {trainer.device}; "
          f"action_space={env.action_space.n} obs={env.observation_space.shape}")

    policy, history = trainer.train()

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
