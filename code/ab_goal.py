#!/usr/bin/env python
"""Goal-direction A/B analysis (Track B5 / item 15).

Compares the matched goal-delta arm against the no-goal baseline
(``output/ppo_v2_bt/live_run2``) at shared paired-eval iterations. Both runs
use the identical protocol (v2 space, BT readout, 50 agents / 5 persuaders /
erdos_renyi / 10 periods, ``--obs-mode pair``, ``--target-selection
random_neighbor``, seed 0, paired eval on 50 shared seeds every 20 iters);
the *only* difference is ``--goal-delta --align semdiff``.

Reward is not directly comparable across the two dynamics, so the readout is
(a) within-arm learning (greedy vs random on the same seeds), (b) the goal
metrics the plan asks for (mean_final_stance, adopted), and (c) for the goal
arm only, the ``a(m)`` trajectory — whether the policy drives its messages
more on-goal under the reward.

Writes ``report/ab_goal.json`` (consumed by ``report/build_report.py``).
Runnable at any time: an in-progress goal arm is reported as ``complete:
false`` with whatever checkpoints exist so far.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

BASELINE = ROOT / "output/ppo_v2_bt/live_run2"
BASELINE_LOG = ROOT / "output/ppo_v2_bt/train2.log"
GOAL = ROOT / "output/ppo_v2_goal/live_goal"
GOAL_LOG = ROOT / "output/ppo_v2_goal/goal.log"

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
    # de-dup (resumed runs repeat lines), keep the last per iteration
    by = {e["iter"]: e for e in out}
    return [by[k] for k in sorted(by)]


def history(run_dir: Path) -> list[dict]:
    p = run_dir / "training_log.json"
    if not p.exists():
        return []
    h = json.loads(p.read_text()).get("history", [])
    keep = ("iteration", "mean_episode_reward", "mean_final_stance",
            "mean_adopted", "train_goal_align", "train_judge_src",
            "train_strategy_entropy")
    return [{k: r.get(k) for k in keep} for r in h]


def results(run_dir: Path) -> dict | None:
    p = run_dir / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def arm(name: str, run_dir: Path, log: Path) -> dict:
    ev = paired_evals(log)
    h = history(run_dir)
    tail = h[len(h) // 2:] if h else []
    return {
        "name": name,
        "seed": 0,
        "goal_delta": name == "goal",
        "paired": ev,
        "n_iters_recorded": len(h),
        "reward_tail": _mean([r["mean_episode_reward"] for r in tail]),
        "final_stance_tail": _mean([r["mean_final_stance"] for r in tail]),
        "adopted_tail": _mean([r["mean_adopted"] for r in tail]),
        "goal_align_tail": _mean([r["train_goal_align"] for r in tail]),
        "goal_align_series": [
            None if r.get("train_goal_align") is None
            else round(r["train_goal_align"], 4) for r in h
        ],
        "reward_series": [
            None if r.get("mean_episode_reward") is None
            else round(r["mean_episode_reward"], 4) for r in h
        ],
        "results": results(run_dir),
    }


def main() -> int:
    base = arm("baseline", BASELINE, BASELINE_LOG)
    goal = arm("goal", GOAL, GOAL_LOG)
    base_by = {e["iter"]: e for e in base["paired"]}
    matched = []
    for e in goal["paired"]:
        b = base_by.get(e["iter"])
        if b:
            matched.append({
                "iter": e["iter"],
                "b_greedy": b["greedy"], "g_greedy": e["greedy"],
                "b_d": b["d"], "g_d": e["d"],
                "b_t": b["t"], "g_t": e["t"],
            })
    complete = (GOAL / "results.json").exists()
    payload = {
        "complete": complete,
        "protocol": {
            "space": "v2", "judge": "bt", "agents": 50, "persuaders": 5,
            "topology": "erdos_renyi", "periods": 10, "obs_mode": "pair",
            "target_selection": "random_neighbor", "paired_eval_every": 20,
            "paired_eval_episodes": 50, "seed": 0,
            "goal_arm": "--goal-delta --align semdiff",
        },
        "baseline": base,
        "goal": goal,
        "matched": matched,
        "summary": {
            "n_matched_checkpoints": len(matched),
            "baseline_learns": (matched and
                                all(m["b_d"] > 0 for m in matched)),
            "goal_learns": (matched and
                            all(m["g_d"] > 0 for m in matched)),
            "goal_align_tail": goal["goal_align_tail"],
        },
    }
    out = ROOT / "report/ab_goal.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[+] wrote {out}  complete={complete}  "
          f"matched={len(matched)} checkpoints")
    for m in matched:
        print(f"    iter {m['iter']:>3}: baseline greedy {m['b_greedy']:.3f} "
              f"(d+{m['b_d']:.3f}) | goal greedy {m['g_greedy']:.3f} "
              f"(d+{m['g_d']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
