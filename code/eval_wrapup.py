"""Wrap-up evaluation for interrupted runs: loads the last checkpoint
and computes the formal results.json (learned greedy vs random vs
heuristic baselines) that train_persuader only writes after the final
iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .rl_env import ElitePersuaderEnv
from .scenario import LLMScenario
from .torch_trainer import (
    PPOConfig,
    PPOTrainer,
    evaluate_heuristic_baseline,
    evaluate_random_baseline,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--agent-num", type=int, default=50)
    p.add_argument("--n-persuaders", type=int, default=5)
    p.add_argument("--topology", type=str, default="erdos_renyi")
    p.add_argument("--n-periods", type=int, default=10)
    p.add_argument("--target-selection", type=str, default="learned")
    p.add_argument("--obs-mode", type=str, default="matrix")
    args = p.parse_args()

    cfg = PPOConfig(
        agent_num=args.agent_num,
        n_persuaders=args.n_persuaders,
        topology=args.topology,
        n_periods=args.n_periods,
        rollout_workers=args.workers,
    )
    scenario = LLMScenario(agent_num=cfg.agent_num)

    def build():
        return ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=cfg.agent_num),
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
    trainer = PPOTrainer(env, cfg, env_factory=build)
    ckpt = Path(args.run_dir) / "policy.pt"
    loaded = PPOTrainer.load_policy(ckpt)
    trainer.policy.load_state_dict(loaded.state_dict())
    print(f"[+] loaded checkpoint {ckpt}")

    greedy = trainer.evaluate(episodes=args.episodes)
    print(f"[+] greedy eval ({args.episodes} eps): {greedy:.4f}")

    rand = evaluate_random_baseline(
        env, episodes=args.episodes, workers=args.workers
    )
    print(f"[+] random baseline: {rand:.4f}")

    heur = evaluate_heuristic_baseline(
        env, episodes=args.episodes, workers=args.workers
    )
    print(f"[+] heuristic baseline: {heur:.4f}")

    results = {
        "mode": "live",
        "note": "wrapped up from interrupted run via eval_wrapup.py",
        "config": vars(cfg),
        "final_greedy_reward": greedy,
        "random_baseline_reward": rand,
        "heuristic_baseline_reward": heur,
        "improvement_over_random": greedy - rand,
        "improvement_over_heuristic": greedy - heur,
    }
    out = Path(args.run_dir) / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[+] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
