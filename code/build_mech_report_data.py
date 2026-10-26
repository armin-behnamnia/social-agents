"""Collect the final C7/C8 mechanism-campaign numbers into one small JSON.

The report needs a handful of figures from each mechanism arm: the reward and
adoption curves (from the trainer's `training_log.json`), the paired-eval
significance series, the baselines from `results.json`, and the exposure
count *by period* from the trace. The trace is large (hundreds of MB of
`events.jsonl`), so this runs once here and writes
`report/mech_runs.json`; `report/build_report.py` then reads that file and
stays fast, exactly like `report/{ab_goal,goal_backends}.json` do for
campaign 6.

Usage:  .venv/bin/python -m code.build_mech_report_data
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ARMS = {
    "cascade": "C7 cascade (stance-ful)",
    "exposure": "C8 exposure (stance-less)",
}


def _period_counts(events: Path) -> tuple[list[int], int]:
    """Count exposures per period across the whole trace."""
    counts = [0] * 10
    total = 0
    with events.open(errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = e.get("period")
            if p is None:
                continue
            total += 1
            if 0 <= p < len(counts):
                counts[p] += 1
    return counts, total


def collect() -> dict:
    out: dict = {"source": "code.build_mech_report_data", "arms": {}}
    for arm, title in ARMS.items():
        live = ROOT / "output" / f"ppo_mech_{arm}" / "live"
        log = json.loads((live / "training_log.json").read_text())
        hist = [x for x in log["history"]
                if isinstance(x, dict) and "elapsed_s" in x]
        evals = [x for x in log["history"]
                 if isinstance(x, dict) and "eval_greedy_reward" in x]

        rewards = [x["mean_episode_reward"] for x in hist]
        adopted = [x["mean_adopted"] for x in hist]
        t_rand = [x["paired_eval"]["vs_random"]["t"] for x in evals]
        t_heur = [x["paired_eval"]["vs_heuristic"]["t"] for x in evals]

        def _judge(v):
            return v["mean"] if isinstance(v, dict) else v
        by_period, total = _period_counts(live / "trace" / "events.jsonl")

        results = json.loads((live / "results.json").read_text())
        out["arms"][arm] = {
            "title": title,
            "iterations": hist[-1]["iteration"],
            "elapsed_h": round(hist[-1]["elapsed_s"] / 3600.0, 1),
            "reward_first": round(rewards[0], 3),
            "reward_last": round(rewards[-1], 3),
            "reward_last6_mean": round(st.mean(rewards[-6:]), 3),
            "adopted_first": adopted[0],
            "adopted_last": adopted[-1],
            "adopted_min": min(adopted),
            "adopted_max": max(adopted),
            "train_judge_first": round(hist[0]["train_judge"]["mean"], 3),
            "train_judge_last": round(hist[-1]["train_judge"]["mean"], 3),
            "eval_iters": [x["iteration"] for x in evals],
            "eval_greedy": [round(x["eval_greedy_reward"], 3) for x in evals],
            "eval_judge": [round(_judge(x["eval_judge"]), 3) for x in evals],
            "eval_strategy_entropy": [round(x["eval_strategy_entropy"], 2)
                                      for x in evals],
            "t_random": [round(t, 2) for t in t_rand],
            "t_heuristic": [round(t, 2) for t in t_heur],
            "t_random_mean": round(st.mean(t_rand), 2),
            "t_heuristic_mean": round(st.mean(t_heur), 2),
            "t_heuristic_min": round(min(t_heur), 2),
            "exposures_by_period": by_period,
            "exposures_total": total,
            "results": {
                "greedy": results["final_greedy_reward"],
                "random": results["random_baseline_reward"],
                "heuristic": results["heuristic_baseline_reward"],
            },
        }
    return out


def main() -> int:
    data = collect()
    dest = ROOT / "report" / "mech_runs.json"
    dest.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[+] wrote {dest}")
    for arm, a in data["arms"].items():
        print(f"    {arm:9s} iters {a['iterations']} reward {a['reward_first']}"
              f"->{a['reward_last']} adopted {a['adopted_min']:.1f}.."
              f"{a['adopted_max']:.1f} t_rand mean {a['t_random_mean']:+.2f}"
              f" exposures/period {a['exposures_by_period']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
