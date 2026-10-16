#!/usr/bin/env python
"""Goal-direction A/B analysis (Track B5 / item 15).

Compares matched goal-delta arms against the no-goal baseline
(``output/ppo_v2_bt/live_run2``) at shared paired-eval iterations. Every run
uses the identical protocol (v2 space, BT readout, 50 agents / 5 persuaders /
erdos_renyi / 10 periods, ``--obs-mode pair``, ``--target-selection
random_neighbor``, seed 0, paired eval on 50 shared seeds every 20 iters);
the arms differ only in the goal-alignment backend:

- ``baseline``      — no injection
- ``goal_semdiff``  — ``--goal-delta --align semdiff``  (embedding backend)
- ``goal_judge``    — ``--goal-delta --align judge``    (LLM backend)

Reward is not directly comparable across differing dynamics, so the readout
is (a) within-arm learning (greedy vs random on the same seeds), (b) the goal
metrics the plan asks for (mean_final_stance, adopted), and (c) the ``a(m)``
trajectory — whether the policy drives its messages more on-goal.

Writes ``report/ab_goal.json`` (consumed by ``report/build_report.py``).
Runnable at any time: an in-progress arm is reported as ``complete: false``.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# name -> (run dir, log path, is_goal_arm, backend)
ARMS = {
    "baseline": (ROOT / "output/ppo_v2_bt/live_run2",
                 ROOT / "output/ppo_v2_bt/train2.log", False, None),
    "goal_semdiff": (ROOT / "output/ppo_v2_goal/live_goal",
                     ROOT / "output/ppo_v2_goal/goal.log", True, "semdiff"),
    "goal_judge": (ROOT / "output/ppo_v2_goal/live_goal_judge",
                   ROOT / "output/ppo_v2_goal/goal_judge.log", True, "judge"),
}

_PAIRED = re.compile(
    r"paired eval @ (\d+): greedy ([\d.]+) \| random ([\d.]+) "
    r"\(d\+([\d.]+) t=([\d.]+)\) \| heuristic ([\d.]+)"
)


def paired_evals(log_path: Path) -> list[dict]:
    out = []
    if not log_path.exists():
        return out
    for line in log_path.read_text(errors="replace").splitlines():
        m = _PAIRED.search(line)
        if m:
            out.append({
                "iter": int(m.group(1)), "greedy": float(m.group(2)),
                "random": float(m.group(3)), "d": float(m.group(4)),
                "t": float(m.group(5)), "heuristic": float(m.group(6)),
            })
    by = {e["iter"]: e for e in out}  # resumed runs repeat lines
    return [by[k] for k in sorted(by)]


def history(run_dir: Path) -> list[dict]:
    p = run_dir / "training_log.json"
    if not p.exists():
        return []
    h = json.loads(p.read_text()).get("history", [])
    keep = ("iteration", "mean_episode_reward", "mean_final_stance",
            "mean_adopted", "train_goal_align")
    return [{k: r.get(k) for k in keep} for r in h]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def arm(name: str) -> dict:
    run_dir, log, is_goal, backend = ARMS[name]
    h = history(run_dir)
    tail = h[len(h) // 2:] if h else []
    return {
        "name": name,
        "backend": backend,
        "goal_delta": is_goal,
        "complete": (run_dir / "results.json").exists(),
        "paired": paired_evals(log),
        "n_iters": len(h),
        "reward_tail": _mean([r["mean_episode_reward"] for r in tail]),
        "final_stance_tail": _mean([r["mean_final_stance"] for r in tail]),
        "adopted_tail": _mean([r["mean_adopted"] for r in tail]),
        "goal_align_tail": _mean([r["train_goal_align"] for r in tail]),
        "goal_align_series": [
            None if r.get("train_goal_align") is None
            else round(r["train_goal_align"], 4) for r in h],
    }


def main() -> int:
    arms = {n: arm(n) for n in ARMS}
    base_by = {e["iter"]: e for e in arms["baseline"]["paired"]}
    matched = {}
    for n, a in arms.items():
        if not a["goal_delta"]:
            continue
        rows = []
        for e in a["paired"]:
            b = base_by.get(e["iter"])
            if b:
                rows.append({
                    "iter": e["iter"],
                    "b_greedy": b["greedy"], "g_greedy": e["greedy"],
                    "b_d": b["d"], "g_d": e["d"], "g_t": e["t"],
                })
        matched[n] = rows

    payload = {
        "complete": all(a["complete"] for a in arms.values()),
        "protocol": {
            "space": "v2", "judge": "bt", "agents": 50, "persuaders": 5,
            "topology": "erdos_renyi", "periods": 10, "obs_mode": "pair",
            "target_selection": "random_neighbor", "paired_eval_every": 20,
            "paired_eval_episodes": 50, "seed": 0,
        },
        "arms": arms,
        "matched": matched,
    }
    out = ROOT / "report/ab_goal.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[+] wrote {out}  complete={payload['complete']}")
    for n, rows in matched.items():
        print(f"  [{n}]")
        for m in rows:
            print(f"    iter {m['iter']:>3}: baseline {m['b_greedy']:.3f} "
                  f"(d+{m['b_d']:.3f}) | arm {m['g_greedy']:.3f} "
                  f"(d+{m['g_d']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
