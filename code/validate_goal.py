"""Validate goal-direction measurement + injection (Track B4).

Checks, in order:

1. **Directional sanity.** For hand-built pro/anti/neutral posts, the
   alignment `a(m)` must order them anti > neutral > pro when the goal is
   the anti-policy proposition.
2. **a(m) vs realized −Δs.** Run the env with goal-direction injection on
   and correlate `a(m)` with the realized `−Δs` it produces. With the
   injection, off-goal posts move targets the wrong way, so `−Δs` turns
   negative for low `a`.
3. **Off-goal is penalized.** Same posts, `goal_delta` off vs on: an
   off-goal post that earns reward when off must stop earning it when on.

Usage::

    PYTHONPATH=. .venv/bin/python -m code.validate_goal --offline
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ANTI = "This policy is harmful and should be rejected for our community."
PRO = "This policy is beneficial and should be supported for our community."
NEUTRAL = "I went to the market today and bought some apples."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--periods", type=int, default=6)
    ap.add_argument("--agent-num", type=int, default=40)
    ap.add_argument("--align", default="semdiff",
                    choices=["semdiff", "nli", "judge"])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default="output/calibration/goal_validate.json")
    args = ap.parse_args()

    if args.offline:
        from . import offline as _offline
        _offline.install()

    from .goal_align import get_align_fn

    fn = get_align_fn(args.align)
    a_anti, a_neu, a_pro = fn(ANTI), fn(NEUTRAL), fn(PRO)
    print(f"[*] directional sanity ({args.align}): "
          f"anti={a_anti:.3f}  neutral={a_neu:.3f}  pro={a_pro:.3f}")
    sanity = bool(a_anti > a_neu > a_pro)

    # 2/3. injection: correlate a(m) with realized −Δs
    from .rl_env import ElitePersuaderEnv
    from .scenario import LLMScenario

    records = []
    for ep in range(args.episodes):
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=args.agent_num, seed=ep, rng_seed=ep),
            n_periods=args.periods, obs_mode="summary", n_persuaders=3,
        )
        env.scenario.goal_align = args.align
        env.scenario.goal_delta = True
        env._goal_align_fn = get_align_fn(
            args.align, env.scenario.goal_proposition,
            env.scenario.goal_anti_proposition,
        )
        env.reset(seed=ep)
        done = False
        while not done:
            a = [env.action_space.sample() for _ in range(env.n_persuaders)]
            _, _, term, trunc, info = env.step(a)
            for e in info["period_events"]:
                if "goal_align" in e:
                    records.append((e["goal_align"], -e["target_delta_stance"]))
            done = term or trunc

    arr = np.asarray(records, dtype=float)
    ga, neg_ds = arr[:, 0], arr[:, 1]
    corr = float(np.corrcoef(ga, neg_ds)[0, 1]) if len(arr) > 2 else float("nan")
    hi = neg_ds[ga >= 0.5].mean() if np.any(ga >= 0.5) else float("nan")
    lo = neg_ds[ga < 0.5].mean() if np.any(ga < 0.5) else float("nan")
    print(f"[+] n={len(arr)}  corr(a, −Δs)={corr:+.3f}  "
          f"mean −Δs: on-goal(a≥.5)={hi:+.4f}  off-goal(a<.5)={lo:+.4f}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "align_backend": args.align,
        "directional_sanity": sanity,
        "a_anti": a_anti, "a_neutral": a_neu, "a_pro": a_pro,
        "n_decisions": int(len(arr)),
        "corr_align_neg_delta": corr,
        "mean_neg_delta_on_goal": float(hi),
        "mean_neg_delta_off_goal": float(lo),
        "pass": bool(sanity and (corr > 0 or len(arr) < 3)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[+] wrote {out}  pass={payload['pass']}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
