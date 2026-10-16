"""Fit / audit the influence-probability weights α (Track A1).

The influence probability is the weighted sum of four similarity terms::

    p = α_p·profile_sim + α_c·content_sim + α_s·stance_sim + α_d·degree_sim

(× the message-persuasiveness multiplier, 1.0 for crowd agents). The four
α's are all 0.25 *by assumption* (``code/config.py``); the paper defines
``p_ij ← InfluenceProbability(...)`` but never expands it.

This script turns that assumption into evidence. It runs frozen crowd
simulations, logs every exposure (the four components are now carried on
each ``diffusion_event``), and reports two views:

1. **Linear recovery.** The accept gate is exactly ``rand < p``, so the
   logged ``influence_probability`` is an exact linear function of the
   components. A non-negative least-squares regression of it on the
   components (no intercept) therefore *recovers* the α's the simulator
   used — a consistency check on the instrumentation, and a hard bound on
   how much information the behavior carries.

2. **Logistic fit.** ``P(accept) = σ(Σ α_k·sim_k + b)`` constrained to
   ``α_k ≥ 0`` — the deliberately mis-specified behavioral model from the
   plan. Report coefficients, bootstrap SEs, decile calibration, and the
   **effective rank / condition number** of the design matrix.

If the design matrix is (near-)collinear, the α's are not separately
identifiable from behavior — the script says so and recommends keeping the
equal-weight prior rather than pretending to a data-derived value.

Usage::

    PYTHONPATH=. .venv/bin/python -m code.fit_alphas --episodes 200
    PYTHONPATH=. .venv/bin/python -m code.fit_alphas --offline   # mock LLM
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "1.0"
FEATURES = ["profile_sim", "content_sim", "stance_sim", "degree_sim"]


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------

def collect_rows(*, episodes: int, periods: int, agent_num: int,
                 seed0: int = 0, offline: bool = False) -> dict:
    """Run frozen crowd simulations and return one row per exposure."""
    if offline:
        from . import offline as _offline
        _offline.install()

    from .profiles import generate_synthetic_profiles
    from .scenario import LLMScenario
    from .simulation import Simulation

    X, y, p_logged = [], [], []
    for ep in range(episodes):
        seed = seed0 + ep
        scenario = LLMScenario(
            agent_num=agent_num, seed=seed,
            # skip the three observer-inference LLM calls (not part of the
            # influence gate); saves time and is irrelevant to α.
            use_post_evidence=False, use_profile_evidence=False,
        )
        sim = Simulation(
            scenario=scenario,
            profiles=generate_synthetic_profiles(agent_num, seed=1234),
            seed_ids=[0],
        )
        sim.run(periods)
        for e in sim.diffusion_events:
            if "profile_sim" not in e:
                continue
            # crowd baseline: persuasiveness multiplier must be 1.0
            if abs(e.get("persuasiveness", 1.0) - 1.0) > 1e-9:
                continue
            X.append([e[f] for f in FEATURES])
            y.append(1.0 if e["behavior"] == "accept" else 0.0)
            p_logged.append(e["influence_probability"])

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p_logged = np.asarray(p_logged, dtype=np.float64)
    return {"X": X, "y": y, "p_logged": p_logged}


def data_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------
# Estimators (numpy only — no scipy/sklearn in this venv)
# ----------------------------------------------------------------------

def nnls(A: np.ndarray, b: np.ndarray, *, iters: int = 20000) -> np.ndarray:
    """min ||A a - b||² s.t. a >= 0, by projected gradient."""
    scale = 1.0 / max(np.linalg.norm(A, 2) ** 2, 1e-12)
    a = np.zeros(A.shape[1])
    for _ in range(iters):
        a = np.maximum(a - scale * (A.T @ (A @ a - b)), 0.0)
    return a


def fit_logistic_nn(X: np.ndarray, y: np.ndarray, *,
                    l2: float = 1e-6, iters: int = 4000,
                    lr: float | None = None) -> tuple[np.ndarray, float]:
    """Non-negative logistic fit: P(y=1|x) = σ(α·x + b), α ≥ 0."""
    n, d = X.shape
    Xb = np.column_stack([X, np.ones(n)])
    lr = lr or (1.0 / (np.linalg.norm(Xb, 2) ** 2 / n + 1e-9))
    w = np.zeros(d + 1)
    for it in range(iters):
        z = Xb @ w
        r = 1.0 / (1.0 + np.exp(-z)) - y
        g = Xb.T @ r / n + l2 * w
        w = w - lr * g
        w[:d] = np.maximum(w[:d], 0.0)  # project α ≥ 0
        if it % 500 == 0 and np.max(np.abs(g)) < 1e-7:
            break
    return w[:d], float(w[d])


def bootstrap_logistic(X, y, *, B: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y)
    coefs = np.zeros((B, X.shape[1] + 1))
    for b in range(B):
        idx = rng.integers(0, n, n)
        a, b0 = fit_logistic_nn(X[idx], y[idx], iters=1200)
        coefs[b] = np.append(a, b0)
    return coefs.mean(0), coefs.std(0)


def effective_rank(X: np.ndarray) -> dict:
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    ent = float(-(p * np.log(p)).sum())
    return {
        "singular_values": [float(v) for v in s],
        "condition_number": float(s[0] / s[-1]) if len(s) else float("inf"),
        "effective_rank": float(np.exp(ent)),
        "n_singular": int(len(s)),
        "per_feature_std": [float(v) for v in X.std(0)],
    }


def decile_calibration(y, p_hat, bins: int = 10):
    order = np.argsort(p_hat)
    out = []
    for k in range(bins):
        idx = order[k * len(order) // bins:(k + 1) * len(order) // bins]
        if len(idx) == 0:
            continue
        out.append({
            "bin": k,
            "n": int(len(idx)),
            "mean_pred": float(p_hat[idx].mean()),
            "empirical": float(y[idx].mean()),
        })
    return out


def log_loss(y, p_hat) -> float:
    p = np.clip(p_hat, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--periods", type=int, default=12)
    ap.add_argument("--agent-num", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--offline", action="store_true",
                    help="install the offline mock LLM (fast, deterministic)")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out", type=str,
                    default="output/calibration/alphas.json")
    args = ap.parse_args()

    spec = {
        "script": "fit_alphas", "version": SCRIPT_VERSION,
        "episodes": args.episodes, "periods": args.periods,
        "agent_num": args.agent_num, "seed0": args.seed0,
        "offline": args.offline, "features": FEATURES,
    }
    print(f"[*] generating α data: {args.episodes} episodes x "
          f"{args.periods} periods (offline={args.offline})", flush=True)
    data = collect_rows(
        episodes=args.episodes, periods=args.periods,
        agent_num=args.agent_num, seed0=args.seed0, offline=args.offline,
    )
    X, y, p_logged = data["X"], data["y"], data["p_logged"]
    n = len(y)
    if n < 50:
        print(f"[!] only {n} rows — increase --episodes")
    print(f"[+] rows: {n}  accept rate: {y.mean():.3f}")

    diag = effective_rank(X)
    print(f"[+] effective rank: {diag['effective_rank']:.2f}/4  "
          f"cond: {diag['condition_number']:.1f}")
    print(f"    per-feature std: "
          f"{[round(v, 3) for v in diag['per_feature_std']]}")

    # 1. linear recovery (exact, since p is the gate)
    a_lin = nnls(X, p_logged)
    resid = float(np.abs(X @ a_lin - p_logged).mean())

    # 2. logistic fit
    a_log, b_log = fit_logistic_nn(X, y)
    _, se = bootstrap_logistic(X, y, B=args.bootstrap, seed=1)
    p_hat = 1.0 / (1.0 + np.exp(-(X @ a_log + b_log)))
    calib = decile_calibration(y, p_hat)
    ll = log_loss(y, p_hat)

    print("[+] linear recovery  α (no intercept): "
          f"{[round(v, 4) for v in a_lin]}  mean|resid|={resid:.2e}")
    print(f"[+] logistic fit     α: {[round(v, 4) for v in a_log]}  "
          f"b={b_log:.3f}  logloss={ll:.4f}")
    print(f"    bootstrap SE     : {[round(v, 4) for v in se[:4]]}  "
          f"(intercept {se[4]:.3f})")

    identifiable = diag["effective_rank"] >= 3.5 and \
        diag["condition_number"] < 100
    recommendation = (
        "identifiable: α recoverable from behavior"
        if identifiable else
        "NOT separately identifiable (near-collinear design): keep the "
        "equal-weight prior (0.25) and report non-identifiability rather "
        "than a fitted value"
    )
    print(f"[=] {recommendation}")

    payload = {
        "spec": spec,
        "data_hash": data_hash({**spec, "n": n}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(n),
        "accept_rate": float(y.mean()),
        "current_config_alpha": 0.25,
        "linear_recovery_alpha": [float(v) for v in a_lin],
        "linear_recovery_mean_abs_resid": resid,
        "logistic_alpha": [float(v) for v in a_log],
        "logistic_intercept": b_log,
        "logistic_bootstrap_se": [float(v) for v in se],
        "logistic_log_loss": ll,
        "identifiability": diag,
        "identifiable": bool(identifiable),
        "recommendation": recommendation,
        "calibration_deciles": calib,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[+] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
