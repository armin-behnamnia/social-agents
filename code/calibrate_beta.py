"""Calibrate the stance step β (Track A2).

``stance_influence_rate`` (β = 0.4) sets the reward scale
(``reward ≈ β·p·gap``), the convergence speed, and time-to-adopt, but it
is inherited rather than derived. This script sweeps β against the
qualitative dynamics the paper reports:

    early mean ΔS (t ∈ 1..3)   ≈ −0.168
    late  mean ΔS (t ≥ 16)     ≈ −0.107
    late  mean  S (plateau)    ≈ −0.55

(from ``Agentic_Information_diffusion.pdf``; signs in our convention: the
crowd starts supportive at +0.65 and the seed argues the opposing view).

For each β it runs frozen crowd simulations (single seed, matching the
paper's setup — no RL policy involved), records the mean-stance
trajectory, and scores the three targets. The chosen β is the minimiser,
reported with the full sweep curve — not as a bare constant.

Usage::

    PYTHONPATH=. .venv/bin/python -m code.calibrate_beta --offline
    PYTHONPATH=. .venv/bin/python -m code.calibrate_beta --betas 0.1 1.0 19
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "1.0"

# paper targets (deltas negative: stance falls toward the opposing seed)
TARGET_EARLY_DS = -0.168
TARGET_LATE_DS = -0.107
TARGET_PLATEAU_S = -0.55


def _simulate_beta(beta: float, *, episodes: int, periods: int, agent_num: int,
                   seed0: int, kappa: float = 1.0) -> dict:
    from .profiles import generate_synthetic_profiles
    from .scenario import LLMScenario
    from .simulation import Simulation

    profiles = generate_synthetic_profiles(agent_num, seed=1234)
    traj = np.zeros((episodes, periods + 1))
    first_adopt = []
    for ep in range(episodes):
        scenario = LLMScenario(
            agent_num=agent_num, seed=seed0 + ep,
            stance_influence_rate=beta,
            # κ scales the four equal-weight α's jointly: the influence
            # *scale* (shape stays equal-weight; Track A1 showed the α's
            # are not separately identifiable).
            alpha_profile=0.25 * kappa, alpha_content=0.25 * kappa,
            alpha_stance=0.25 * kappa, alpha_degree=0.25 * kappa,
            use_post_evidence=False, use_profile_evidence=False,
        )
        sim = Simulation(scenario=scenario, profiles=profiles, seed_ids=[0])
        traj[ep, 0] = float(sim.stances().mean())
        adopt_at = None
        for t in range(1, periods + 1):
            sim.step()
            traj[ep, t] = float(sim.stances().mean())
            if adopt_at is None and sum(
                a.preference_state == 1 for a in sim.agents
            ) > 1:
                adopt_at = t
        first_adopt.append(adopt_at if adopt_at is not None else np.nan)

    mean_traj = traj.mean(0)            # (periods+1,)
    deltas = np.diff(mean_traj)          # per-period ΔS
    early = float(deltas[0:3].mean()) if periods >= 3 else float("nan")
    late_idx = deltas[15:] if periods > 16 else deltas[len(deltas) // 2:]
    late = float(late_idx.mean()) if len(late_idx) else float("nan")
    plateau = float(mean_traj[-min(5, len(mean_traj)):].mean())
    adopt_med = float(np.nanmedian(first_adopt)) if np.any(
        ~np.isnan(first_adopt)) else float("nan")
    return {
        "beta": beta,
        "kappa": kappa,
        "rho": beta * kappa,
        "early_dS": early,
        "late_dS": late,
        "plateau_S": plateau,
        "periods_to_adopt": adopt_med,
        "mean_trajectory": [float(v) for v in mean_traj],
    }


def _score(row: dict) -> float:
    """Normalised distance to the three paper targets (lower is better)."""
    return float(
        ((row["early_dS"] - TARGET_EARLY_DS) / abs(TARGET_EARLY_DS)) ** 2
        + ((row["late_dS"] - TARGET_LATE_DS) / abs(TARGET_LATE_DS)) ** 2
        + ((row["plateau_S"] - TARGET_PLATEAU_S) / abs(TARGET_PLATEAU_S)) ** 2
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--betas", type=float, nargs=3,
                    metavar=("LO", "HI", "N"), default=[0.1, 1.0, 19])
    ap.add_argument("--p-scales", type=float, nargs="+", default=[1.0],
                    help="κ grid scaling the four α jointly (joint with β)")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--periods", type=int, default=24)
    ap.add_argument("--agent-num", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", type=str, default="output/calibration/beta.json")
    ap.add_argument("--plot", type=str,
                    default="output/calibration/beta_sweep.png")
    args = ap.parse_args()

    if args.offline:
        from . import offline as _offline
        _offline.install()

    lo, hi, n = args.betas
    betas = np.linspace(lo, hi, int(n))
    kappas = list(args.p_scales)
    print(f"[*] β×κ sweep: {len(betas)} β × {len(kappas)} κ × "
          f"{args.episodes} episodes × {args.periods} periods "
          f"(offline={args.offline})", flush=True)

    rows = []
    for kappa in kappas:
        for b in betas:
            r = _simulate_beta(
                float(b), episodes=args.episodes, periods=args.periods,
                agent_num=args.agent_num, seed0=args.seed0, kappa=float(kappa),
            )
            r["score"] = _score(r)
            rows.append(r)
            print(f"    β={b:.3f} κ={kappa:.2f} (ρ={r['rho']:.2f})  "
                  f"earlyΔS={r['early_dS']:+.4f}  "
                  f"lateΔS={r['late_dS']:+.4f}  plateauS={r['plateau_S']:+.4f}  "
                  f"adopt@{r['periods_to_adopt']}  score={r['score']:.4f}",
                  flush=True)

    best = min(rows, key=lambda r: r["score"])
    print(f"[+] best β={best['beta']:.3f} κ={best['kappa']:.2f} "
          f"(ρ=β·κ={best['rho']:.3f})  score {best['score']:.4f}")
    print(f"    earlyΔS={best['early_dS']:+.4f} (target {TARGET_EARLY_DS})  "
          f"lateΔS={best['late_dS']:+.4f} (target {TARGET_LATE_DS})  "
          f"plateauS={best['plateau_S']:+.4f} (target {TARGET_PLATEAU_S})")

    spec = {"script": "calibrate_beta", "version": SCRIPT_VERSION,
            "episodes": args.episodes, "periods": args.periods,
            "agent_num": args.agent_num, "seed0": args.seed0,
            "offline": args.offline, "p_scales": kappas, "targets": {
                "early_dS": TARGET_EARLY_DS, "late_dS": TARGET_LATE_DS,
                "plateau_S": TARGET_PLATEAU_S}}
    payload = {
        "spec": spec,
        "data_hash": hashlib.sha256(
            json.dumps({**spec, "betas": [float(b) for b in betas]},
                       sort_keys=True).encode()).hexdigest()[:16],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_config_beta": 0.4,
        "chosen_beta": float(best["beta"]),
        "chosen_kappa": float(best["kappa"]),
        "chosen_rho": float(best["rho"]),
        "note": ("β and the α-scale κ enter only through the product ρ=β·κ "
                 "(Δs = p·β·(s_src−s), p = κ·Σ0.25·sim), so the dynamics pin "
                 "down ρ, not β alone. Pick any (β, κ) with the target ρ."),
        "sweep": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[+] wrote {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        rho = [r["rho"] for r in rows]
        ax[0].plot(rho, [r["early_dS"] for r in rows], ".", alpha=0.7)
        ax[0].axhline(TARGET_EARLY_DS, color="r", ls="--")
        ax[0].set_title("early ΔS (target -0.168)")
        ax[1].plot(rho, [r["late_dS"] for r in rows], ".", alpha=0.7)
        ax[1].axhline(TARGET_LATE_DS, color="r", ls="--")
        ax[1].set_title("late ΔS (target -0.107)")
        ax[2].plot(rho, [r["plateau_S"] for r in rows], ".", alpha=0.7)
        ax[2].axhline(TARGET_PLATEAU_S, color="r", ls="--")
        ax[2].set_title("plateau S (target -0.55)")
        for a in ax:
            a.axvline(best["rho"], color="g", ls=":", label="chosen ρ")
            a.set_xlabel("ρ = β·κ (effective influence strength)")
            a.grid(alpha=0.3)
        fig.tight_layout()
        p = Path(args.plot)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130)
        print(f"[+] wrote {p}")
    except Exception as e:  # noqa: BLE001 - plot is optional
        print(f"[!] plot skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
