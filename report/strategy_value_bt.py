#!/usr/bin/env python
"""Fixed-strategy reward landscape under the Bradley-Terry judge.

Runs every action in the active strategy space with a constant-strategy
policy over shared seeds and reports the spread — i.e. the "best
constant" the learned policy has to beat, and the learnable headroom.

Pure lookup (the BT table is in memory), so this makes no LLM calls.

    PERSUASION_STRATEGY_SPACE=v2 PYTHONPATH=. python report/strategy_value_bt.py
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from code.bt_judge import BTJudge
from code.eval_final import HeuristicPolicy, run_episode
from code.rl_env import ElitePersuaderEnv
from code.scenario import LLMScenario
from code.strategy import N_STRATEGIES, describe, index_to_strategy

HERE = Path(__file__).resolve().parent
TABLE = "output/ppo_v2_bt/bt_judge.json"
OUT = HERE / "strategy_value_bt.json"
N_SEEDS = 8
WORKERS = 48
SEED0 = 500_000


def main() -> None:
    judge = BTJudge.load(TABLE)
    env = ElitePersuaderEnv(
        scenario=LLMScenario(
            agent_num=50, use_post_evidence=False, use_profile_evidence=False
        ),
        n_periods=10, topology="erdos_renyi", topic="the policy",
        target_selection="random_neighbor", obs_mode="pair", n_persuaders=5,
        judge_fn=judge,
    )
    run_episode(env.clone(), SEED0, HeuristicPolicy(0))  # warm caches

    def run(job):
        s, k = job
        return s, k, float(
            run_episode(env.clone(), SEED0 + k, HeuristicPolicy(strategy_idx=s))
        )

    jobs = [(s, k) for s in range(N_STRATEGIES) for k in range(N_SEEDS)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(run, jobs))

    per = {}
    for s, k, r in rows:
        per.setdefault(s, []).append(r)
    means = {s: float(np.mean(v)) for s, v in per.items()}
    vals = np.array([means[s] for s in range(N_STRATEGIES)])
    order = sorted(range(N_STRATEGIES), key=lambda s: means[s], reverse=True)

    out = {
        "judge": "bt", "space": "v2", "n_strategies": N_STRATEGIES,
        "n_seeds": N_SEEDS,
        "mean": {str(s): round(means[s], 4) for s in range(N_STRATEGIES)},
        "sd_by_strategy": {str(s): round(float(np.std(per[s], ddof=1)), 4)
                           for s in range(N_STRATEGIES)},
        "stats": {
            "mean": round(float(vals.mean()), 4),
            "sd": round(float(vals.std(ddof=1)), 4),
            "range": round(float(vals.max() - vals.min()), 4),
            "p10_p90": round(float(np.percentile(vals, 90) - np.percentile(vals, 10)), 4),
            "best_idx": int(order[0]), "best": round(means[order[0]], 4),
            "best_spec": describe(index_to_strategy(order[0])),
            "worst_idx": int(order[-1]), "worst": round(means[order[-1]], 4),
            "worst_spec": describe(index_to_strategy(order[-1])),
            "per_episode_se": round(float(np.mean([
                np.std(per[s], ddof=1) / np.sqrt(N_SEEDS) for s in per
            ])), 4),
        },
        "top10": [{"idx": s, "spec": describe(index_to_strategy(s)),
                   "mean": round(means[s], 4)} for s in order[:10]],
        "bottom5": [{"idx": s, "spec": describe(index_to_strategy(s)),
                     "mean": round(means[s], 4)} for s in order[-5:]],
    }
    OUT.write_text(json.dumps(out, indent=1))
    st = out["stats"]
    print(f"[+] BT landscape: mean {st['mean']} sd {st['sd']} range {st['range']} "
          f"p10-90 {st['p10_p90']} se {st['per_episode_se']}")
    print(f"    best  {st['best_idx']:>2} {st['best']:>6}  {st['best_spec']}")
    print(f"    worst {st['worst_idx']:>2} {st['worst']:>6}  {st['worst_spec']}")
    print("[+] wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
