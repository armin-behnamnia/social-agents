#!/usr/bin/env python
"""Measure the per-strategy reward landscape through the network.

For every strategy index s in [0,180) run N_SEEDS episodes with all five
persuaders forced to s, on a fixed block of paired seeds (common random
numbers). Incremental: writes report/strategy_value.json after every
block so partial coverage is always usable. Also records the trained
greedy policy's strategy distribution over a few episodes.

Usage: PYTHONPATH=. python report/strategy_value.py
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from code.rl_env import ElitePersuaderEnv
from code.scenario import LLMScenario
from code.strategy import N_STRATEGIES
from code.torch_trainer import PPOConfig, PPOTrainer
from code.eval_final import run_episode, HeuristicPolicy, GreedyPolicy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "strategy_value.json"
SEED_BASE = 500_000
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
BLOCK = int(sys.argv[3]) if len(sys.argv) > 3 else 6
STRATEGIES = list(range(N_STRATEGIES))

cfg = PPOConfig(agent_num=50, n_persuaders=5, topology="erdos_renyi",
                n_periods=10, rollout_workers=WORKERS)


def build():
    sc = LLMScenario(agent_num=cfg.agent_num, use_post_evidence=False,
                     use_profile_evidence=False)
    return ElitePersuaderEnv(
        scenario=sc, n_periods=cfg.n_periods, topology=cfg.topology,
        topic=cfg.topic, target_selection="random_neighbor", obs_mode="pair",
        reward_cascade_discount=cfg.reward_cascade_discount,
        message_cost=cfg.message_cost,
        shaping_accept_bonus=cfg.shaping_accept_bonus,
        n_persuaders=cfg.n_persuaders)


base = build()
run_episode(base.clone(), SEED_BASE, HeuristicPolicy(90))  # warm one

mat = np.full((N_STRATEGIES, N_SEEDS), np.nan)
done_strats: set[int] = set()


def dump():
    have = [s for s in STRATEGIES if s in done_strats]
    sub = mat[have]
    mean = np.nanmean(sub, axis=1)
    se = np.nanstd(sub, axis=1, ddof=1) / np.sqrt(N_SEEDS)
    heur_row = next((i for i, s in enumerate(have) if s == 90), None)
    pv = ((sub - sub[heur_row].mean(axis=0)).mean(axis=1)
          if heur_row is not None else np.zeros(len(have)))
    payload = {
        "n_seeds": N_SEEDS, "seed_base": SEED_BASE,
        "strategies": have,
        "mean": np.round(mean, 4).tolist(),
        "se": np.round(se, 4).tolist(),
        "paired_vs_heuristic": np.round(pv, 4).tolist(),
        "heuristic_idx": 90, "trained_idx": 168,
        "coverage": f"{len(have)}/{N_STRATEGIES}",
        "stats": {
            "n": len(have),
            "best_idx": int(have[int(np.argmax(mean))]),
            "best_mean": float(mean.max()),
            "heur_mean": float(mean[have.index(90)]) if 90 in have else None,
            "worst_idx": int(have[int(np.argmin(mean))]),
            "worst_mean": float(mean.min()),
            "spread_all": float(mean.max() - mean.min()),
            "spread_trim": float(np.percentile(mean, 90) - np.percentile(mean, 10)),
            "sd_mean": float(mean.std(ddof=1)),
        },
    }
    # preserve a prior trained-spread block if present
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text())
            for k in ("trained_top", "trained_distinct", "trained_actions",
                      "mean_score", "corr_score_reward"):
                if k in old:
                    payload[k] = old[k]
        except Exception:
            pass
    OUT.write_text(json.dumps(payload, indent=1))


def run(job):
    s, k = job
    return s, k, float(run_episode(base.clone(), SEED_BASE + k,
                                   HeuristicPolicy(strategy_idx=s)))


t0 = time.time()
for b0 in range(0, len(STRATEGIES), BLOCK):
    block = STRATEGIES[b0:b0 + BLOCK]
    jobs = [(s, k) for s in block for k in range(N_SEEDS)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run, j) for j in jobs]
        for f in as_completed(futs):
            s, k, r = f.result()
            mat[s, k] = r
    done_strats.update(block)
    dump()
    el = time.time() - t0
    print(f"[{len(done_strats)}/{N_STRATEGIES}] {el:.0f}s "
          f"({el/max(1,len(done_strats)):.1f}s/strategy)", flush=True)

# --------------------------------------------------- trained policy spread
try:
    policy = PPOTrainer.load_policy(
        ROOT / "output/ppo5p_rn/live_run1/policy.pt", device="cpu")
    greedy = GreedyPolicy(policy)

    def run_greedy(k):
        env = base.clone()
        hist = {}
        _, info = env.reset(seed=SEED_BASE + k)
        done = False
        while not done:
            acts = greedy.actions(env, info)
            for a in (acts if isinstance(acts, list) else [acts]):
                hist[int(a)] = hist.get(int(a), 0) + 1
            _, _, term, trunc, info = env.step(acts)
            done = term or trunc
        return hist

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        hist_list = list(pool.map(run_greedy, range(24)))
    th: dict[int, int] = {}
    for h in hist_list:
        for k, v in h.items():
            th[k] = th.get(k, 0) + v
    dumped = json.loads(OUT.read_text())
    dumped["trained_top"] = sorted(th.items(), key=lambda kv: -kv[1])[:8]
    dumped["trained_distinct"] = len(th)
    dumped["trained_actions"] = int(sum(th.values()))
    OUT.write_text(json.dumps(dumped, indent=1))
    print("[+] trained distinct strategies:", len(th), "top:", dumped["trained_top"][:4])
except Exception as exc:  # noqa: BLE001
    print("[!] trained-spread step skipped:", exc)

# ------------------------------------------------------- judge-score link
try:
    jp = json.loads((HERE / "judge_probe.json").read_text())
    ms = np.zeros(N_STRATEGIES)
    per = jp["per_profile"]
    for _, sc in per.items():
        ms += np.asarray(sc, dtype=float)
    ms /= max(1, len(per))
    d = json.loads(OUT.read_text())
    have = d["strategies"]
    m = np.asarray(d["mean"])
    d["mean_score"] = np.round(ms, 4).tolist()   # full 180
    d["corr_score_reward"] = float(np.corrcoef(ms[have], m)[0, 1])
    OUT.write_text(json.dumps(d, indent=1))
    print("[+] corr(score,reward) =", round(d["corr_score_reward"], 3))
except Exception as exc:  # noqa: BLE001
    print("[!] score-link step skipped:", exc)

print("[+] wrote", OUT, "coverage", json.loads(OUT.read_text())["coverage"])
