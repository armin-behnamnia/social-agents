"""Paired-seed final evaluation for a trained persuader policy.

Motivation
----------
The per-run evals in `torch_trainer` use *different* episode seeds for
the learned policy (900_000+k), the random baseline (700_000+k) and the
heuristic baseline (600_000+k). Episode-to-episode reward variance is
large (per-episode sigma ~0.5), so independent seeds make the
policy-vs-baseline gap noisy. This script evaluates all three policies on
the *same* episode seeds and reports paired differences (mean delta with
standard error and a paired t-statistic), which is a much lower-variance
comparison on the same compute budget.

It is read-only w.r.t. training: it loads `<run-dir>/policy.pt` and runs
episodes. Random/heuristic operate in the env's native action space, so
the script works for both `target_selection=learned` (compound 5400
actions) and `random_neighbor` (strategy-only 180 actions).

Usage
-----
python -m code.eval_final --run-dir output/ppo5p_rn/live_run1 \
    --target-selection random_neighbor --obs-mode pair \
    --episodes 30 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .rl_env import ElitePersuaderEnv
from .scenario import LLMScenario
from .strategy import N_STRATEGIES
from .torch_trainer import PPOConfig, PPOTrainer

logger = logging.getLogger("code.eval_final")


# ----------------------------------------------------------------------
# policies: each returns the env action(s) for the current period
# ----------------------------------------------------------------------

class GreedyPolicy:
    def __init__(self, policy: torch.nn.Module):
        self.policy = policy.cpu().eval()

    def actions(self, env: ElitePersuaderEnv, info: dict):
        mask = info.get("action_mask")
        mask_t = (
            torch.as_tensor(mask, dtype=torch.bool)
            if mask is not None else None
        )
        n = int(getattr(env, "n_persuaders", 1))
        out = []
        for i in range(n):
            obs_i = env.observation(actor=i)
            obs_t = torch.as_tensor(obs_i, dtype=torch.float32)
            out.append(self.policy.act(obs_t, mask=mask_t, greedy=True))
        return out[0] if n == 1 else out


class RandomStrategyPolicy:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def actions(self, env: ElitePersuaderEnv, info: dict):
        mask = info.get("action_mask")
        valid = None
        if mask is not None and not np.all(mask):
            valid = np.flatnonzero(mask)
        n = int(getattr(env, "n_persuaders", 1))
        out = []
        for _ in range(n):
            if valid is not None:
                out.append(int(self.rng.choice(valid)))
            else:
                out.append(int(self.rng.integers(env.action_space.n)))
        return out[0] if n == 1 else out


class HeuristicPolicy:
    """Fixed mid-range strategy (the 'fixed message bank' analogue)."""

    def __init__(self, strategy_idx: int = N_STRATEGIES // 2):
        self.strategy_idx = strategy_idx

    def actions(self, env: ElitePersuaderEnv, info: dict):
        n = int(getattr(env, "n_persuaders", 1))
        return self.strategy_idx if n == 1 else [self.strategy_idx] * n


# ----------------------------------------------------------------------

def run_episode(env: ElitePersuaderEnv, seed: int, policy) -> float:
    _, info = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        action = policy.actions(env, info)
        _, reward, terminated, truncated, info = env.step(action)
        total += float(np.sum(np.atleast_1d(reward)))
        done = terminated or truncated
    return total


def mean_se(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return mean, se


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="policy.pt")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=500_000)
    p.add_argument("--agent-num", type=int, default=50)
    p.add_argument("--n-persuaders", type=int, default=5)
    p.add_argument("--topology", type=str, default="erdos_renyi")
    p.add_argument("--n-periods", type=int, default=10)
    p.add_argument("--target-selection", type=str, default="random_neighbor")
    p.add_argument("--obs-mode", type=str, default="pair")
    p.add_argument(
        "--observer", action="store_true",
        help="enable observer-side inference (Prompt 5) during eval "
             "(3 extra LLM calls per adoption; does not affect reward, so "
             "off by default to match training and cut eval cost)",
    )
    p.add_argument("--out", type=str, default="results_final.json")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # the diffusion engine logs every adoption at INFO; noise for a
    # read-only evaluation
    logging.getLogger("code.agent").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = PPOConfig(
        agent_num=args.agent_num,
        n_persuaders=args.n_persuaders,
        topology=args.topology,
        n_periods=args.n_periods,
        rollout_workers=args.workers,
    )

    def build() -> ElitePersuaderEnv:
        scenario = LLMScenario(
            agent_num=cfg.agent_num,
            use_post_evidence=args.observer,
            use_profile_evidence=args.observer,
        )
        return ElitePersuaderEnv(
            scenario=scenario,
            n_periods=cfg.n_periods,
            topology=cfg.topology,
            topic=cfg.topic,
            target_selection=args.target_selection,
            obs_mode=args.obs_mode,
            reward_cascade_discount=cfg.reward_cascade_discount,
            message_cost=cfg.message_cost,
            shaping_accept_bonus=cfg.shaping_accept_bonus,
            n_persuaders=cfg.n_persuaders,
        )

    env = build()
    ckpt = Path(args.run_dir) / args.checkpoint
    policy = PPOTrainer.load_policy(ckpt, device="cpu")
    logger.info("loaded checkpoint %s", ckpt)

    n_workers = max(1, args.workers)

    def run_all(k: int) -> tuple[float, float, float]:
        """Run the three policies on the same episode seed (paired)."""
        seed = args.seed_base + k
        greedy = run_episode(env.clone(), seed, GreedyPolicy(policy))
        rand = run_episode(
            env.clone(), seed, RandomStrategyPolicy(seed + 7_919)
        )
        heur = run_episode(env.clone(), seed, HeuristicPolicy())
        return greedy, rand, heur

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        rows = list(pool.map(run_all, range(args.episodes)))

    rows = np.asarray(rows, dtype=float)  # (episodes, 3)
    greedy, rand, heur = rows[:, 0], rows[:, 1], rows[:, 2]

    g_mean, g_se = mean_se(greedy)
    r_mean, r_se = mean_se(rand)
    h_mean, h_se = mean_se(heur)

    d_rand = greedy - rand
    d_heur = greedy - heur
    dr_mean, dr_se = mean_se(d_rand)
    dh_mean, dh_se = mean_se(d_heur)

    def tstat(d):
        d = np.asarray(d, dtype=float)
        if d.std(ddof=1) == 0:
            return float("inf")
        return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))

    result = {
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "paired_seeds": [args.seed_base + k for k in range(args.episodes)],
        "config": {
            "agent_num": args.agent_num,
            "n_persuaders": args.n_persuaders,
            "topology": args.topology,
            "n_periods": args.n_periods,
            "target_selection": args.target_selection,
            "obs_mode": args.obs_mode,
        },
        "policies": {
            "greedy": {"mean": g_mean, "se": g_se, "per_episode": greedy.tolist()},
            "random": {"mean": r_mean, "se": r_se, "per_episode": rand.tolist()},
            "heuristic": {"mean": h_mean, "se": h_se, "per_episode": heur.tolist()},
        },
        "paired_delta_vs_random": {"mean": dr_mean, "se": dr_se, "t": tstat(d_rand)},
        "paired_delta_vs_heuristic": {"mean": dh_mean, "se": dh_se, "t": tstat(d_heur)},
    }

    out = Path(args.run_dir) / args.out
    out.write_text(json.dumps(result, indent=2))

    print("\n===== PAIRED EVALUATION =====")
    print(f"episodes: {args.episodes} (paired seeds)")
    print(f"greedy    : {g_mean:.4f} +/- {g_se:.4f} (SE)")
    print(f"random    : {r_mean:.4f} +/- {r_se:.4f} (SE)")
    print(f"heuristic : {h_mean:.4f} +/- {h_se:.4f} (SE)")
    print(f"delta vs random    : {dr_mean:+.4f} +/- {dr_se:.4f} "
          f"(t={tstat(d_rand):.2f})")
    print(f"delta vs heuristic : {dh_mean:+.4f} +/- {dh_se:.4f} "
          f"(t={tstat(d_heur):.2f})")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
