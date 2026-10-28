"""Analyse the finished C7/C8 mechanism campaign from its traces.

Everything here is measured from what the trainer itself recorded, so the
conclusions are checked rather than assumed:

  * `periods.jsonl` — the exact `period_reward_sum` per period (it reproduces
    the logged `mean_episode_reward` to 1e-5), plus `n_accept`,
    `n_persuader_events`, `n_adopted`, `n_events`;
  * `steps.jsonl` — one row per persuader action with its `persuasiveness`
    (BT judge) and `goal_align` (= a(m), the goal-axis judge the reward uses);
  * `events.jsonl` — one row per diffusion event with source, target,
    `target_delta_stance`, `persuasiveness`, `behavior`, `adopted`.

`code/rl_env.py::step` assembles the reward as

    reward = -delta_stance + 0.05*n_accept + B*n_adopt + LAM*sum(a(m)) - cost

with B = 0 (C7) / 1.0 (C8), LAM = 0 (C7) / 0.2 (C8), cost = 0, and a(m) the
`goal_align` field (NOT `persuasiveness` — they are different judges and are
almost uncorrelated). C8's stance term is identically 0, so regressing its
reward on `(n_accept, sum(a(m)))` recovers the weights exactly; that is the
validation that lets us attribute C7's remainder to the stance pull.

Usage:  PYTHONPATH=. .venv/bin/python -m code.analyze_mech_campaign
Writes: report/mech_analysis.json
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARMS = {"cascade": {"title": "C7 cascade (stance-ful)", "bonus": 0.0, "lam": 0.0},
        "exposure": {"title": "C8 exposure (stance-less)", "bonus": 1.0, "lam": 0.2}}
N_AGENTS = 50
N_PERSUADERS = 5
CASCADE_DISCOUNT = 0.5


def _load_stance_terms(arm):
    """Per-episode (Σ_persuader −Δs, Σ_all −Δs) from the diffusion events.

    The reward's stance pull is `−Δs` over events *sourced by* a persuader;
    the cascade credit is `κ·(mean_stance_before − mean_stance_after)`, which
    summed over the episode telescopes to `κ·Σ_all(−Δs)/N_AGENTS` — note the
    Σ must be over ALL events, and the term is added to every persuader, so in
    the team-summed reward (what `period_reward_sum` records) its coefficient
    is `N_PERSUADERS·κ`. Measuring only the recorded period means 1..10 misses
    the period-0 movement and understates it by ~5×.
    """
    ev: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    path = ROOT / "output" / f"ppo_mech_{arm}" / "live" / "trace" / "events.jsonl"
    for line in path.open(errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = (int(e["iteration"]), e["episode"])
        nd = -float(e.get("target_delta_stance") or 0.0)
        if e.get("source_is_persuader"):
            ev[k][0] += nd
        ev[k][1] += nd
    return ev


def _load_periods(arm):
    live = ROOT / "output" / f"ppo_mech_{arm}" / "live"
    ga = defaultdict(float)
    pv = defaultdict(float)
    nstep = defaultdict(int)
    for line in (live / "trace" / "steps.jsonl").open(errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = (int(e["iteration"]), e["episode"], e.get("step"))
        ga[k] += float(e.get("goal_align") or 0.0)
        pv[k] += float(e.get("persuasiveness") or 0.0)
        nstep[k] += 1
    rows = []
    for line in (live / "trace" / "periods.jsonl").open(errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = (int(e["iteration"]), e["episode"], e.get("step"))
        rows.append({
            "iter": k[0], "episode": e["episode"], "period": e.get("period"),
            "reward": float(e.get("period_reward_sum") or 0.0),
            "n_accept": float(e.get("n_accept") or 0.0),
            "sum_a": ga.get(k, 0.0), "sum_judge": pv.get(k, 0.0),
            "n_msgs": nstep.get(k, 0),
            "n_pe": float(e.get("n_persuader_events") or 0.0),
            "n_events": float(e.get("n_events") or 0.0),
            "n_adopted_pop": float(e.get("n_adopted") or 0.0),
            "n_candidates": float(e.get("n_candidates") or 0.0),
            "mean_stance": float(e.get("mean_stance") or 0.0),
        })
    return rows


def _episodes(rows, cfg, ev):
    """Aggregate to one record per episode and decompose the reward."""
    eps: dict[tuple, dict] = defaultdict(lambda: {
        "reward": 0.0, "n_accept": 0.0, "n_adopt": 0.0, "sum_a": 0.0,
        "sum_judge": 0.0, "n_msgs": 0.0, "n_pe": 0.0, "n_events": 0.0,
        "n_adopted_pop": 0.0, "n_periods": 0.0, "n_idle": 0.0,
        "stance_first": None, "stance_last": None})
    for r in rows:
        k = (r["iter"], r["episode"])
        a = eps[k]
        a["reward"] += r["reward"]
        a["n_accept"] += r["n_accept"]
        # exposure: an accepted exposure becomes an adoption, so n_adopt=n_accept
        a["n_adopt"] += r["n_accept"] if cfg["bonus"] else 0.0
        a["sum_a"] += r["sum_a"]
        a["sum_judge"] += r["sum_judge"]
        a["n_msgs"] += r["n_msgs"]
        a["n_pe"] += r["n_pe"]
        a["n_events"] += r["n_events"]
        a["n_periods"] += 1
        if r["n_candidates"] <= 0:
            a["n_idle"] += r["n_msgs"]
        if a["stance_first"] is None:
            a["stance_first"] = r["mean_stance"]
        a["stance_last"] = r["mean_stance"]
    for k, a in eps.items():
        pull, all_neg = ev.get(k, (0.0, 0.0))
        a["t_pull"] = pull
        a["t_cascade"] = N_PERSUADERS * CASCADE_DISCOUNT * all_neg / N_AGENTS
        a["t_accept"] = 0.05 * a["n_accept"]
        a["t_bonus"] = cfg["bonus"] * a["n_adopt"]
        a["t_align"] = cfg["lam"] * a["sum_a"]
        # closure check: everything not named must be ~0
        a["t_resid"] = (a["reward"] - a["t_pull"] - a["t_cascade"]
                        - a["t_accept"] - a["t_bonus"] - a["t_align"])
    return eps


def _var_share(vals: dict[str, list[float]], total_var: float):
    if total_var <= 0:
        return {}
    return {k: round(100 * st.pvariance(v) / total_var, 1)
            for k, v in vals.items()}


def _trend(xs: list[float]):
    pairs = [(i, x) for i, x in enumerate(xs)]
    n = len(pairs)
    if n < 3:
        return None
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((a - mx) * (b - my) for a, b in pairs)
    den = sum((a - mx) ** 2 for a, b in pairs)
    dx = den ** 0.5
    dy = sum((b - my) ** 2 for a, b in pairs) ** 0.5
    return {"slope_per_iter": round(num / den, 6) if den else 0.0,
            "r": round(num / (dx * dy), 3) if dx and dy else None}


def _deciles(samples, key_idx):
    """samples: tuples (persuasiveness, content_sim, adopted, abs_delta)."""
    vals = sorted(s[key_idx] for s in samples)
    edges = [vals[int((len(vals) - 1) * i / 10)] for i in range(1, 10)]
    buckets: dict[int, list] = defaultdict(list)
    for s in samples:
        b = sum(1 for e in edges if s[key_idx] >= e)
        buckets[b].append(s)
    out = []
    for b in sorted(buckets):
        grp = buckets[b]
        out.append({
            "bin": b, "n": len(grp),
            "mean_persuasiveness": round(st.mean(g[0] for g in grp), 4),
            "mean_content_sim": round(st.mean(g[1] for g in grp), 4),
            "adopt_rate": round(st.mean(g[2] for g in grp), 4),
            "mean_abs_delta": round(st.mean(g[3] for g in grp), 4),
        })
    return out


def analyse(arm):
    cfg = ARMS[arm]
    rows = _load_periods(arm)

    # --- recovered algebra ------------------------------------------
    A = np.array([[r["n_accept"], r["sum_a"], 1.0] for r in rows])
    Y = np.array([r["reward"] for r in rows])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    algebra = {
        "regression": "reward ~ c1*n_accept + c2*sum(a(m)) + c3",
        "coef": [round(float(c), 5) for c in coef],
        "r2": round(1 - float(np.sum(resid ** 2)) / ss_tot, 6),
        "max_abs_resid": round(float(np.max(np.abs(resid))), 6),
        "expected": {"n_accept": 0.05 + cfg["bonus"], "sum_a": cfg["lam"],
                     "intercept": 0.0},
    }

    # --- per-episode decomposition ----------------------------------
    eps = _episodes(rows, cfg, _load_stance_terms(arm))
    keys = ["t_pull", "t_cascade", "t_accept", "t_bonus", "t_align"]
    means = {k: st.mean(e[k] for e in eps.values()) for k in keys}
    reward_mean = st.mean(e["reward"] for e in eps.values())
    total_var = st.pvariance([e["reward"] for e in eps.values()])
    variance_shares = _var_share({k: [e[k] for e in eps.values()] for k in keys},
                                 total_var)
    cov_bonus_align = st.covariance(
        [e["t_bonus"] for e in eps.values()],
        [e["t_align"] for e in eps.values()]) if arm == "exposure" else 0.0
    decomposition = {
        "n_episodes": len(eps),
        "mean_reward": round(reward_mean, 4),
        "term_means": {k: round(v, 4) for k, v in means.items()},
        "term_shares_pct": {k: round(100 * v / reward_mean, 1)
                            for k, v in means.items()},
        "term_variance": {k: round(st.pvariance([e[k] for e in eps.values()]), 5)
                          for k in keys},
        "term_variance_shares_pct": variance_shares,
        "reward_variance": round(total_var, 5),
        "cov_bonus_align": round(cov_bonus_align, 5),
        "per_episode": {
            "n_accept": round(st.mean(e["n_accept"] for e in eps.values()), 3),
            "sum_a_m": round(st.mean(e["sum_a"] for e in eps.values()), 3),
            "sum_a_m_sd": round(st.pstdev([e["sum_a"] for e in eps.values()]), 3),
            "sum_judge": round(st.mean(e["sum_judge"] for e in eps.values()), 3),
            "n_msgs": round(st.mean(e["n_msgs"] for e in eps.values()), 2),
            "n_persuader_events": round(st.mean(e["n_pe"] for e in eps.values()), 2),
            "n_events": round(st.mean(e["n_events"] for e in eps.values()), 2),
            "n_messages": round(st.mean(e["n_msgs"] for e in eps.values()), 2),
            "idle_messages": round(st.mean(e["n_idle"] for e in eps.values()), 2),
            "idle_frac": round(st.mean(e["n_idle"] for e in eps.values()) /
                               max(st.mean(e["n_msgs"] for e in eps.values()), 1), 4),
        },
    }

    # --- per-iteration trends of each component ----------------------
    by_iter: dict[int, dict] = defaultdict(
        lambda: {k: 0.0 for k in keys} | {"reward": 0.0, "n": 0})
    for e_key, e in eps.items():
        it = e_key[0]
        by_iter[it]["reward"] += e["reward"]
        for k in keys:
            by_iter[it][k] += e[k]
        by_iter[it]["n"] += 1
    iters = sorted(by_iter)
    series = {k: [by_iter[i][k] / by_iter[i]["n"] for i in iters] for k in keys}
    series["reward"] = [by_iter[i]["reward"] / by_iter[i]["n"] for i in iters]
    trends = {k: _trend(v) for k, v in series.items()}

    # --- direct-exposure quality -> outcome curves -------------------
    # events.jsonl carries `persuasiveness` but not the reward's a(m)
    # (`goal_align` lives only in steps.jsonl), so the curve uses the judge;
    # the persuasiveness-vs-a(m) correlation is measured from steps below.
    samples = []
    for line in (ROOT / "output" / f"ppo_mech_{arm}" / "live" / "trace"
                 / "events.jsonl").open(errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not e.get("source_is_persuader"):
            continue
        samples.append((
            float(e.get("persuasiveness") or 0.0),
            float(e.get("content_sim") or 0.0),
            int(e.get("adopted") or 0),
            abs(float(e.get("target_delta_stance") or 0.0)),
        ))
    curves = {"n_direct_exposures": len(samples),
              "by_persuasiveness": _deciles(samples, 0)}

    # --- judge saturation + the two judges' correlation --------------
    pv_step, ga_step = [], []
    for line in (ROOT / "output" / f"ppo_mech_{arm}" / "live" / "trace"
                 / "steps.jsonl").open(errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("persuasiveness") is not None:
            pv_step.append(float(e["persuasiveness"]))
        if e.get("goal_align") is not None:
            ga_step.append(float(e["goal_align"]))
    sat = {}
    for name, vals in (("persuasiveness", pv_step), ("a_m_goal_align", ga_step)):
        if vals:
            sat[name] = {
                "mean": round(st.mean(vals), 4),
                "sd": round(st.pstdev(vals), 4),
                "frac_at_1.0": round(sum(1 for v in vals if v >= 0.9999) / len(vals), 4),
            }
    corr = None
    if (len(pv_step) == len(ga_step) and len(pv_step) > 2
            and st.pstdev(pv_step) and st.pstdev(ga_step)):
        corr = round(st.covariance(pv_step, ga_step)
                     / (st.pstdev(pv_step) * st.pstdev(ga_step)), 4)

    # --- per-period ------------------------------------------------
    per_period = defaultdict(lambda: {"n": 0, "acc": 0, "ev": 0, "pe": 0})
    for r in rows:
        p = per_period[r["period"]]
        p["n"] += 1
        p["acc"] += r["n_accept"]
        p["ev"] += r["n_events"]
        p["pe"] += r["n_pe"]

    return {
        "title": cfg["title"],
        "reward_weights_used": {"accept_bonus": 0.05, "adoption_bonus": cfg["bonus"],
                                "goal_lambda": cfg["lam"], "message_cost": 0.0},
        "recovered_algebra": algebra,
        "decomposition": decomposition,
        "component_trends": trends,
        "component_series": {"iters": iters, **{k: [round(x, 4) for x in v]
                                                for k, v in series.items()}},
        "quality_curves": curves,
        "alignment_vs_persuasiveness_corr": corr,
        "saturation": sat,
        "per_period": [{"period": p, "n_periods": per_period[p]["n"],
                        "accept_per_period": round(per_period[p]["acc"] / per_period[p]["n"], 3),
                        "events_per_period": round(per_period[p]["ev"] / per_period[p]["n"], 2),
                        "persuader_events_per_period": round(per_period[p]["pe"] / per_period[p]["n"], 3)}
                       for p in sorted(per_period)],
    }


def main() -> int:
    out = {"source": "code.analyze_mech_campaign", "arms": {}}
    for arm in ARMS:
        out["arms"][arm] = analyse(arm)
    (ROOT / "report" / "mech_analysis.json").write_text(
        json.dumps(out, indent=2) + "\n")
    for arm, a in out["arms"].items():
        g = a["recovered_algebra"]
        d = a["decomposition"]
        print(f"===== {a['title']} =====")
        print(f"  algebra: reward ~ {g['coef'][0]}*n_accept + {g['coef'][1]}*sum(a(m))"
              f" + {g['coef'][2]}  (R^2={g['r2']}, max|resid|={g['max_abs_resid']})"
              f"  expected {g['expected']}")
        print(f"  per-episode reward {d['mean_reward']} = {d['term_means']}")
        print(f"  term shares %      : {d['term_shares_pct']}")
        print(f"  variance shares %  : {d['term_variance_shares_pct']}"
              f"  (reward var {d['reward_variance']})")
        print(f"  per-episode detail : {d['per_episode']}")
        print(f"  component trends   : {a['component_trends']}")
        print(f"  corr(persuasiveness, a(m)) = {a['alignment_vs_persuasiveness_corr']}"
              f"   saturation {a['saturation']}")
        print(f"  direct exposures   : {a['quality_curves']['n_direct_exposures']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
