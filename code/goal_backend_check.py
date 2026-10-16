#!/usr/bin/env python
"""Compare goal-alignment backends on the *actual* rendered messages.

The hand-built directional sanity (a hand-written anti post vs pro post)
passes for every backend — but the messages the policy actually renders are
strategy-templated, and the embedding backends misjudge them. This script
renders a sample of the real (v2) strategy messages and scores each with:

- ``semdiff``      — cosine to the proposition pair
- ``semdiff+seed`` — cosine to the seed post vs a pro-policy post
- ``judge``        — the cached LLM goal-axis prompt

so the report can show *which* backend actually separates on-goal from
off-goal for the messages that matter.

Writes ``report/goal_backends.json``.

Usage::

    PYTHONPATH=. .venv/bin/python -m code.goal_backend_check --n 12
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SEED = ("I believe this policy is harmful for our community. The evidence is "
        "clear and people deserve to know the truth about the risks it "
        "creates.")
PRO = ("This policy is beneficial and should be supported. It helps our "
       "community and the evidence favors it.")
# short, unambiguous posts for the directional-sanity check (the judge is
# unreliable on long hedged sentences)
SHORT_ANTI = "This policy is harmful for our community."
SHORT_PRO = "This policy is beneficial and should be supported."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12,
                    help="number of strategies to sample")
    ap.add_argument("--profile-seed", type=int, default=1234)
    ap.add_argument("--out", default="report/goal_backends.json")
    args = ap.parse_args()

    import os
    os.environ.setdefault("PERSUASION_STRATEGY_SPACE", "v2")
    from code import LLM_module
    from code.goal_align import _cos, get_align_fn
    from code.profiles import generate_synthetic_profiles
    from code.strategy_v2 import N_STRATEGIES, index_to_strategy, describe
    from code.tools import text2embedding

    prof = generate_synthetic_profiles(50, seed=args.profile_seed)[0]
    sem = get_align_fn("semdiff")
    judge = get_align_fn("judge")

    g = np.asarray(text2embedding(SEED), dtype=np.float64)
    n = np.asarray(text2embedding(PRO), dtype=np.float64)

    def a_seed(message: str) -> float:
        m = np.asarray(text2embedding(message), dtype=np.float64)
        return float(0.5 * (_cos(m, g) - _cos(m, n) + 1.0))

    # even spread across the 72-arm space
    idx = sorted({round(i * (N_STRATEGIES - 1) / (args.n - 1))
                  for i in range(args.n)})
    rows = []
    for i in idx:
        s = index_to_strategy(i)
        msg = LLM_module.render_strategy_message(s, prof, topic="the policy")
        rows.append({
            "index": i, "strategy": describe(s), "message": msg[:200],
            "semdiff": round(sem(msg), 3),
            "semdiff_seed": round(a_seed(msg), 3),
            "judge": round(judge(msg), 3),
        })

    def mean(key):
        return round(float(np.mean([r[key] for r in rows])), 3)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_strategies": len(rows),
        "proposition_anchor": True,
        "directional_sanity": {
            "semdiff_seed": {"anti": round(a_seed(SEED), 3),
                             "pro": round(a_seed(PRO), 3)},
            "semdiff": {"anti": round(sem(SHORT_ANTI), 3),
                        "pro": round(sem(SHORT_PRO), 3)},
            "judge": {"anti": round(judge(SHORT_ANTI), 3),
                      "pro": round(judge(SHORT_PRO), 3)},
        },
        "means": {"semdiff": mean("semdiff"),
                  "semdiff_seed": mean("semdiff_seed"),
                  "judge": mean("judge")},
        "rows": rows,
    }
    out = ROOT / args.out
    out.write_text(json.dumps(payload, indent=2))
    print(f"[+] wrote {out}")
    print(f"[*] mean a(m) over {len(rows)} real strategy messages:")
    print(f"      semdiff={payload['means']['semdiff']}  "
          f"semdiff+seed={payload['means']['semdiff_seed']}  "
          f"judge={payload['means']['judge']}   (0.5 = neutral)")
    print(f"[*] sanity (anti/pro): "
          f"judge {payload['directional_sanity']['judge']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
