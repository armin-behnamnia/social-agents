#!/usr/bin/env python
"""Build the standalone Tufte report for the RL Persuader campaigns.

Reads report/report_data.json + report/judge_probe.json (both produced
from the run artifacts) and emits report/index.html.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path


def _sd(values):
    return statistics.pstdev(values)

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "report_data.json").read_text())
JUDGE = json.loads((HERE / "judge_probe.json").read_text())
MESSAGES = (json.loads((HERE / "messages.json").read_text())
            if (HERE / "messages.json").exists() else [])
LAND = (json.loads((HERE / "strategy_value.json").read_text())
        if (HERE / "strategy_value.json").exists() else None)
VARIETY = (json.loads((HERE / "message_variety.json").read_text())
           if (HERE / "message_variety.json").exists() else None)
COND = (json.loads((HERE / "cond_fix.json").read_text())
        if (HERE / "cond_fix.json").exists() else None)
V2 = (json.loads((HERE / "strategy_v2.json").read_text())
      if (HERE / "strategy_v2.json").exists() else None)
BTLAND = (json.loads((HERE / "strategy_value_bt.json").read_text())
          if (HERE / "strategy_value_bt.json").exists() else None)
_BT_TABLE = HERE.parent / "output/ppo_v2_bt/bt_judge.json"
BT = (json.loads(_BT_TABLE.read_text()) if _BT_TABLE.exists() else None)
_C3_RESULTS = HERE.parent / "output/ppo_v2_bt/live_run2/results.json"
_C3_LOG = HERE.parent / "output/ppo_v2_bt/train2.log"
_CAL_DIR = HERE.parent / "output/calibration"


def _load_cal(name):
    # prefer the committed copy in report/ (reproducible), else the run dir
    p = HERE / name
    if not p.exists():
        p = _CAL_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


ALPHAS = _load_cal("alphas.json")
BETA = _load_cal("beta_fine.json")
GOAL = _load_cal("goal_validate.json")


def calibration():
    """Track A: the α fit/audit and the joint (β, κ) sweep vs the paper."""
    if not (ALPHAS and BETA):
        return None
    ident = ALPHAS["identifiability"]
    rows = sorted(BETA["sweep"], key=lambda r: r["rho"])
    best = min(BETA["sweep"], key=lambda r: r["score"])
    return {
        "n_rows": ALPHAS["n_rows"],
        "accept_rate": round(ALPHAS["accept_rate"], 3),
        "lin": [round(v, 4) for v in ALPHAS["linear_recovery_alpha"]],
        "lin_resid": ALPHAS["linear_recovery_mean_abs_resid"],
        "log": [round(v, 4) for v in ALPHAS["logistic_alpha"]],
        "log_b": round(ALPHAS["logistic_intercept"], 3),
        "eff_rank": round(ident["effective_rank"], 2),
        "cond": round(ident["condition_number"], 1),
        "std": [round(v, 3) for v in ident["per_feature_std"]],
        "identifiable": ALPHAS["identifiable"],
        "targets": BETA["spec"]["targets"],
        "chosen_beta": BETA["chosen_beta"],
        "chosen_kappa": BETA["chosen_kappa"],
        "chosen_rho": BETA["chosen_rho"],
        "rho": [round(r["rho"], 3) for r in rows],
        "early": [round(r["early_dS"], 4) for r in rows],
        "late": [round(r["late_dS"], 4) for r in rows],
        "plateau": [round(r["plateau_S"], 4) for r in rows],
        "traj": [round(v, 4) for v in best["mean_trajectory"]],
    }


def goal_validation():
    """Track B: directional sanity + a(m) vs realized −Δs."""
    if not GOAL:
        return None
    return {
        "backend": GOAL["align_backend"],
        "a_anti": round(GOAL["a_anti"], 3),
        "a_neutral": round(GOAL["a_neutral"], 3),
        "a_pro": round(GOAL["a_pro"], 3),
        "corr": round(GOAL["corr_align_neg_delta"], 3),
        "on": round(GOAL["mean_neg_delta_on_goal"], 4),
        "off": round(GOAL["mean_neg_delta_off_goal"], 4),
        "n": GOAL["n_decisions"],
        "pass": GOAL["pass"],
    }


_AB_P = HERE / "ab_goal.json"
_AB = json.loads(_AB_P.read_text()) if _AB_P.exists() else None
_GOALB_P = HERE / "goal_backends.json"
_GOALB = json.loads(_GOALB_P.read_text()) if _GOALB_P.exists() else None
_MECH_P = HERE / "mech_runs.json"
_MECH = json.loads(_MECH_P.read_text()) if _MECH_P.exists() else {"arms": {}}


def goal_backends():
    """Do the a(m) backends separate on the real rendered messages?"""
    if not _GOALB:
        return None
    return {
        "n": _GOALB["n_strategies"],
        "means": _GOALB["means"],
        "sanity": _GOALB["directional_sanity"],
        "rows": _GOALB["rows"],
    }


def ab_goal():
    """Track B5: matched goal-delta arms vs baseline (A/B)."""
    if not _AB:
        return None
    arms = _AB["arms"]
    return {
        "complete": _AB["complete"],
        "arms": {
            k: {key: v[key] for key in (
                "name", "backend", "goal_delta", "complete", "n_iters",
                "reward_tail", "final_stance_tail", "adopted_tail",
                "goal_align_tail")}
            for k, v in arms.items()
        },
        "matched": _AB["matched"],
    }


def campaign3():
    """Final verdict for the v2-space + Bradley-Terry training run."""
    if not _C3_RESULTS.exists():
        return None
    res = json.loads(_C3_RESULTS.read_text())
    out = {
        "greedy": res.get("final_greedy_reward"),
        "random": res.get("random_baseline_reward"),
        "heuristic": res.get("heuristic_baseline_reward"),
        "d_random": res.get("improvement_over_random"),
        "d_heur": res.get("improvement_over_heuristic"),
        "best_const": round(BTLAND["stats"]["best"], 4) if BTLAND else None,
        "best_spec": BTLAND["stats"]["best_spec"] if BTLAND else None,
    }
    evals = []
    if _C3_LOG.exists():
        pat = re.compile(
            r"paired eval @ (\d+): greedy ([\d.]+) \| random ([\d.]+) "
            r"\(d\+([\d.]+) t=([\d.]+)\)"
        )
        for line in _C3_LOG.read_text().splitlines():
            m = pat.search(line)
            if m:
                evals.append({
                    "iter": int(m.group(1)), "greedy": float(m.group(2)),
                    "random": float(m.group(3)),
                    "d": float(m.group(4)), "t": float(m.group(5)),
                })
    out["evals"] = evals
    if evals:
        out["paired_final"] = evals[-1]
        out["paired_iters"] = [e["iter"] for e in evals]
        out["paired_greedy"] = [e["greedy"] for e in evals]
        out["paired_d"] = [e["d"] for e in evals]
    return out


def bt_readout():
    """Distribution + variance decomposition of the BT readout, plus the
    network-reward landscape it induces."""
    if not BT:
        return None
    import numpy as np
    n_s, n_p = BT["stats"]["n_strategies"], BT["stats"]["n_profiles"]
    sc = np.array([e["score"] for e in BT["entries"]]).reshape(n_p, n_s)
    grand = float(sc.mean())
    ss_tot = float(((sc - grand) ** 2).sum())
    ss_strat = float(n_p * ((sc.mean(0) - grand) ** 2).sum())
    ss_prof = float(n_s * ((sc.mean(1) - grand) ** 2).sum())
    ss_int = ss_tot - ss_strat - ss_prof
    col = sc.mean(0)
    best_const = float(col.max())
    oracle = float(sc.max(1).mean())
    sd = float(sc.std(ddof=1))
    # 0.05-bin histogram
    edges = [i / 20 for i in range(21)]
    hist = [0] * 20
    for x in sc.ravel():
        hist[min(19, int(x * 20))] += 1
    modal = max(range(20), key=lambda i: hist[i])
    out = {
        "n": int(sc.size), "n_strategies": n_s, "n_profiles": n_p,
        "sd": round(sd, 4), "n_unique": BT["stats"]["n_unique_scores"],
        "abs_mean": round(BT["stats"]["abs_mean"], 4),
        "modal_bin": round((modal + 0.5) / 20, 3),
        "modal_share": round(hist[modal] / sc.size, 3),
        "hist_labels": [f"{(edges[i]+edges[i+1])/2:.2f}" for i in range(20)],
        "hist_values": hist,
        "decomp": {"strategy": round(100 * ss_strat / ss_tot, 1),
                   "profile": round(100 * ss_prof / ss_tot, 1),
                   "interaction": round(100 * ss_int / ss_tot, 1)},
        "best_const": round(best_const, 4),
        "oracle_gain": round(oracle - best_const, 4),
    }
    if BTLAND:
        st = BTLAND["stats"]
        out["land"] = {
            "mean": st["mean"], "sd": st["sd"], "range": st["range"],
            "p10_p90": st["p10_p90"], "se": st["per_episode_se"],
            "best": st["best"], "best_spec": st["best_spec"],
            "worst": st["worst"], "worst_spec": st["worst_spec"],
        }
    return out


def bt_land_points():
    if not BTLAND:
        return []
    mean = BTLAND["mean"]
    angles = ["cost", "safety", "fairness", "freedom", "community", "effectiveness"]
    evidence = ["statistic", "expert", "anecdote", "peers"]
    addresses = ["taxpayer", "parent", "worker"]
    pts = []
    for s in range(BTLAND["n_strategies"]):
        d = s % 3
        e = (s // 3) % 4
        a = s // 12
        pts.append({
            "s": s, "mean": mean[str(s)],
            "angle": angles[a], "evidence": evidence[e],
            "address": addresses[d],
            "spec": f"angle={angles[a]}, evidence={evidence[e]}, address={addresses[d]}",
        })
    return pts


def _cond_reward(v):
    p = HERE / f"cond_fix_reward_{v}.json"
    return json.loads(p.read_text()) if p.exists() else None


def v2_rows():
    if not V2:
        return []
    r, lj, vc = V2["realisability"], V2["learnability"], V2["variance_components"]
    return [
        {"test": "Realisability", "q": "does the action shape the post?",
         "grid": "emphasis NN 0.31 (chance 0.20)",
         "pilot": f"angle {r['angle']['mean_nn_purity']}, address "
                  f"{r['address']['mean_nn_purity']} (chance {r['angle']['chance']}/"
                  f"{r['address']['chance']})",
         "target": "NN \u2265 0.60", "verdict": "pass"},
        {"test": "Readability", "q": "can the reward resolve the action?",
         "grid": "judge SD 0.062, modal 71%",
         "pilot": f"SD {V2['readability']['sd']}, modal "
                  f"{V2['readability']['modal_share']}",
         "target": "SD \u2265 0.15, modal < 40%", "verdict": "miss"},
        {"test": "Conditionality", "q": "does reward depend on target \u00d7 action?",
         "grid": "oracle gain \u2248 0 (ties the best constant)",
         "pilot": f"oracle gain {lj['oracle_gain']}, interaction "
                  f"{vc['judge_persona_x_angle_frac']}",
         "target": "gain > per-episode reward SE 0.53", "verdict": "miss"},
    ]


def cond_table():
    if not COND:
        return []
    import statistics as st
    metrics = COND["metrics"]

    def avg(per, f):
        return st.fmean([f(d) for d in per.values()]) if per else None

    base_ref = None
    if LAND:
        subset = list(range(0, len(LAND["strategies"]), 3))
        vals = [LAND["mean"][LAND["strategies"].index(s)]
                for s in subset if s in LAND["strategies"]]
        if vals:
            base_ref = {"mean": st.fmean(vals), "sd": st.pstdev(vals)}

    labels = {"base": "base (current)", "ex": "+ exemplar scaffold",
              "bon_judge": "+ best-of-8 (by judge)",
              "bon_faithful": "+ best-of-8 (by faithfulness)"}
    verdict = {
        "base": "reference — the strongest reward of the four",
        "ex": "more strategy-shaped, but reward falls on every strategy",
        "bon_judge": "highest of the three variants, still below base",
        "bon_faithful": "most diverse posts, yet the narrowest reward spread",
    }
    rows = []
    for v in ("base", "ex", "bon_judge", "bon_faithful"):
        per = metrics.get(v)
        if not per:
            continue
        rw = _cond_reward(v)
        rm = rw["spread"]["mean"] if rw else (base_ref["mean"] if base_ref else None)
        rsd = rw["spread"]["sd"] if rw else (base_ref["sd"] if base_ref else None)
        rows.append({
            "variant": labels[v], "key": v,
            "paircos": round(avg(per, lambda d: d["pair_cos_mean"]), 3),
            "emph_nn": round(avg(per, lambda d: d["levels"]["emphasis"]["nn_purity"]), 3),
            "judge_sd": round(avg(per, lambda d: d["judge"]["sd"]), 3),
            "modal": round(avg(per, lambda d: d["judge"]["modal_share"]), 3),
            "openings": round(avg(per, lambda d: d["n_unique_openings"]), 1),
            "reward_mean": round(rm, 3) if rm is not None else None,
            "reward_sd": round(rsd, 3) if rsd is not None else None,
            "above_base": rw.get("share_strategies_above_base") if rw else None,
            "verdict": verdict.get(v, ""),
        })
    return rows


def gen_table():
    if not VARIETY:
        return []
    rl = VARIETY.get("reward_link") or {}
    pp = VARIETY["per_persona"]
    p0 = pp[str(VARIETY["personas"][0])] if str(VARIETY["personas"][0]) in pp else list(pp.values())[0]
    p2 = list(pp.values())[-1]
    er = p0.get("effective_rank", 0)
    pc10 = p0.get("pca", {}).get("pc10", 0)
    return [
        {"label": "Judge score → network reward",
         "val": f"ρ = {rl.get('spearman_judge_vs_reward', 0):.2f}",
         "note": "the dominant path — but quantised to 4–6 scores"},
        {"label": "content_sim → network reward",
         "val": f"ρ = {rl.get('spearman_content_vs_reward', 0):.2f}",
         "note": "the α_content = 0.25 term barely orders the arms"},
        {"label": "message embedding → reward",
         "val": f"R² = {rl.get('cv_r2_reward_from_msgPCs', 0):.2f}",
         "note": "5-fold CV; ~70% of message variation is discarded"},
        {"label": "message embedding spread",
         "val": f"rank {er:.0f} · PC10 {pc10*100:.0f}%",
         "note": "posts span many directions, reward reads few"},
        {"label": "unique post openings",
         "val": f"{p0.get('n_unique_openings', 0)}/180 · {p2.get('n_unique_openings', 0)}/180",
         "note": "surface framing repeats across strategies"},
    ]

_APPEALS = ["logos", "pathos", "ethos"]
_TONES = ["neutral", "urgent", "empathetic", "confrontational"]
_EMPHASES = ["evidence", "personal_story", "social_proof", "risk", "authority"]
_ILAB = ["measured", "assertive", "forceful"]
_DIMS = (3, 4, 5, 3)


def _decode(idx):
    a, t, e, i = [], idx, None, None
    out = []
    rem = idx
    for d in reversed(_DIMS):
        out.append(rem % d)
        rem //= d
    a, t, e, i = reversed(out)
    return f"{_APPEALS[a]} / {_TONES[t]} / {_EMPHASES[e]} / {_ILAB[i]}"


def _rank(idx):
    if not LAND or "mean" not in LAND or idx not in LAND["strategies"]:
        return None
    mean = LAND["mean"]
    m = mean[LAND["strategies"].index(idx)]
    return int(sum(1 for x in mean if x > m)) + 1


def _share(idx):
    """Fraction of the learned policy's actions that chose this strategy."""
    if not LAND:
        return None
    top = dict(LAND.get("trained_top") or [])
    tot = LAND.get("trained_actions") or 0
    return round(top[idx] / tot, 3) if idx in top and tot else None


def land_table():
    """Near-top arms + the heuristic + the learned modes, with ranks."""
    if not LAND or "mean" not in LAND:
        return []
    strat = LAND["strategies"]
    mean = LAND["mean"]
    se = LAND.get("se", [None] * len(mean))
    score = LAND.get("mean_score")
    modes = [i for i, _ in (LAND.get("trained_top") or [])]
    rows = sorted(zip(strat, mean, se), key=lambda r: -r[1])
    picked, seen = [], set()
    for idx, m, s in rows[:5]:                       # top arms
        picked.append(idx); seen.add(idx)
    for idx in (90, 168, *modes):                    # heuristic, best, learned
        if idx in strat and idx not in seen:
            picked.append(idx); seen.add(idx)
    out = []
    for idx in picked:
        j = strat.index(idx)
        sc = score[idx] if score and idx < len(score) else None
        if idx == 90:
            role, note = "heuristic", "fixed mid-strategy baseline"
        elif idx in modes:
            role, note = "learned mode", f"{int(round(_share(idx) * 100))}% of policy actions"
        elif idx == int(max(zip(strat, mean), key=lambda r: r[1])[0]):
            role, note = "top arm", "highest measured reward"
        else:
            role, note = "near-top", "within noise of the best"
        out.append({
            "idx": idx, "spec": _decode(idx),
            "mean": round(mean[j], 3),
            "se": round(se[j], 3) if se[j] is not None else None,
            "score": round(sc, 3) if sc is not None else None,
            "rank": _rank(idx), "share": _share(idx),
            "heur": idx == 90, "trained": idx in modes,
            "role": role, "note": note,
        })
    return out


_MSG_ROLES = {
    90: ("heuristic (fixed, idx 90)", "near-top constant"),
    141: ("learned mode · 70% of actions", "policy argmax"),
    153: ("learned mode · 15% of actions", "policy argmax"),
    13: ("learned mode · 15% of actions", "policy argmax"),
    168: ("best arm (idx 168)", "highest measured reward"),
    58: ("low-score arm (idx 58)", "judge 0.5 — clearly punished"),
}


def enrich_messages():
    out = []
    for m in MESSAGES:
        role, note = _MSG_ROLES.get(m["idx"], ("candidate", ""))
        m = dict(m)
        m["role"] = role
        m["note"] = note
        m["rank"] = _rank(m["idx"])
        m["mean"] = (round(LAND["mean"][LAND["strategies"].index(m["idx"])], 3)
                     if LAND and m["idx"] in LAND["strategies"] else None)
        out.append(m)
    return out


# ---------------------------------------------------------------- runs table
RUNS = [
    {"run": "mock_validation", "camp": "1", "scale": "30 • 1", "env": "mock LLM",
     "greedy": 3.435, "random": 3.697, "heur": 3.481, "dr": -7.1, "dh": -1.3, "verdict": "mock"},
    {"run": "mock_run", "camp": "1", "scale": "30 • 1", "env": "mock LLM",
     "greedy": 3.451, "random": 3.697, "heur": 3.481, "dr": -6.7, "dh": -0.9, "verdict": "mock"},
    {"run": "live_run1", "camp": "1", "scale": "30 • 1", "env": "target + summary obs",
     "greedy": 2.597, "random": 2.789, "heur": 2.639, "dr": -6.9, "dh": -1.6, "verdict": "loss"},
    {"run": "live_run2_matrix", "camp": "1", "scale": "30 • 1", "env": "+ per-node matrix obs",
     "greedy": 2.683, "random": 2.789, "heur": 2.639, "dr": -3.8, "dh": 1.7, "verdict": "mixed"},
    {"run": "live_run3_persuasive", "camp": "1", "scale": "30 • 1", "env": "+ persuasiveness gate",
     "greedy": 2.245, "random": 1.981, "heur": 1.981, "dr": 13.3, "dh": 13.3, "verdict": "win"},
    {"run": "ppo5p/live_run1", "camp": "2", "scale": "50 • 5", "env": "global targeting (interrupted)",
     "greedy": 7.562, "random": 10.004, "heur": 6.761, "dr": -24.4, "dh": 11.8, "verdict": "loss"},
    {"run": "ppo5p_rn/live_run1", "camp": "2", "scale": "50 • 5", "env": "random-neighbor + strategy",
     "greedy": 10.625, "random": 10.038, "heur": 10.520, "dr": 5.8, "dh": 1.0, "verdict": "win"},
]

# per-profile judge spread for the diagnostics table
PER_PROFILE = JUDGE["per_profile"]
prof_rows = []
for k in sorted(PER_PROFILE, key=int):
    sc = PER_PROFILE[k]
    mode = max(set(sc), key=sc.count)
    prof_rows.append({
        "p": f"persona {k}",
        "n": len(sc),
        "uniq": len(set(sc)),
        "sd": round(_sd(sc), 3),
        "mode": mode,
        "modefrac": round(sc.count(mode) / len(sc), 2),
    })

# judge histogram as ordered labels
JH = JUDGE["overall_hist"]
jh_labels = sorted(JH, key=float)
jh_values = [JH[k] for k in jh_labels]

def dashboard_runs():
    """Summarise the exported LLMxIO dashboard runs under report/dashboard/.

    Reads the manifest (labels/metrics) and re-counts edges from the CSVs
    so the report reflects the exact data the dashboard serves.
    """
    import csv as _csv

    base = HERE / "dashboard"
    labels: dict[str, dict] = {}
    man = base / "manifest.json"
    if man.exists():
        labels = {str(k): v for k, v in
                  json.loads(man.read_text()).get("runs", {}).items()}
    runs = []
    for rdir in sorted(base.glob("run*")):
        num = rdir.name.replace("run", "")
        mem = next(iter(rdir.glob("*memory*.csv")), None)
        foll = next(iter(rdir.glob("*follow_log*.csv")), None)
        if mem is None:
            continue
        with mem.open() as fh:
            rows = list(_csv.DictReader(fh))
        n_fol = 0
        if foll is not None:
            with foll.open() as fh:
                n_fol = sum(1 for _ in _csv.DictReader(fh))
        info = labels.get(num, {})
        runs.append({
            "run": num,
            "label": info.get("label", f"run {num}"),
            "adopted": info.get("final_adopted"),
            "stance": info.get("mean_stance", ""),
            "reward": info.get("total_reward"),
            "follows": n_fol,
            "retweets": sum(1 for r in rows if r.get("Type") == "retweet"),
            "replies": sum(1 for r in rows if r.get("Type") == "comment"),
            "influence": sum(1 for r in rows if r.get("Type") == "influence"),
            "tweets": len(rows),
        })
    return runs


pay = {
    "runs": RUNS,
    "prof_rows": prof_rows,
    "train": DATA["train"],
    "paired": DATA["paired"],
    "replay": DATA["replay"],
    "judge": {
        "hist_labels": [f"{float(v):.1f}" for v in jh_labels],
        "hist_values": jh_values,
        "n": JUDGE["overall_n"],
        "sd": round(JUDGE["overall_sd"], 3),
        "mean": round(JUDGE["overall_mean"], 3),
        "modefrac": round(max(JH.values()) / JUDGE["overall_n"], 3),
    },
    "messages": enrich_messages(),
    "land": LAND,
    "land_table": land_table(),
    "variety": VARIETY,
    "gen_table": gen_table(),
    "cond": COND,
    "cond_table": cond_table(),
    "v2": V2,
    "v2_rows": v2_rows(),
    "bt": bt_readout(),
    "bt_points": bt_land_points(),
    "c3": campaign3(),
    "cal": calibration(),
    "goal": goal_validation(),
    "goal_backends": goal_backends(),
    "mech": _MECH,
    "ab": ab_goal(),
    "viz": {"runs": dashboard_runs()},
}

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RL Persuader — Results &amp; Diagnostics</title>
<!-- fonts: this environment is offline; body falls back to Georgia, mono is vendored -->
<style>
@font-face{font-family:'Liberation Mono';src:url('vendor/fonts/LiberationMono-Regular.ttf') format('truetype');font-weight:400;font-display:swap;}
@font-face{font-family:'Liberation Mono';src:url('vendor/fonts/LiberationMono-Bold.ttf') format('truetype');font-weight:700;font-display:swap;}
@font-face{font-family:'Liberation Mono';src:url('vendor/fonts/LiberationMono-Italic.ttf') format('truetype');font-style:italic;font-weight:400;font-display:swap;}
:root{
  --ink:#1a1a1a;--ink-light:#555;--ink-muted:#888;--bg:#fffff8;--bg-aside:#f9f6ee;
  --accent:#a00;--rule:#ccc;
  --spark-primary:#c45a28;--spark-secondary:#2a7a5a;--spark-tertiary:#5a5aaa;
  --status-red:#a02a2a;--status-amber:#c89000;--status-green:#2a7a3a;--status-blue:rgba(42,80,140,0.7);
}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);font-family:'EB Garamond',Georgia,serif;font-size:18px;line-height:1.6;}
.mono{font-family:'Liberation Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums;}
.wrap{max-width:1200px;margin:0 auto;padding:2rem 1.5rem 4rem;}
h1{font-size:2.2rem;font-weight:400;font-variant:small-caps;letter-spacing:0.01em;margin:0.5rem 0 0.3rem;}
h2{font-size:1.5rem;font-weight:400;font-variant:small-caps;margin:3rem 0 0.4rem;}
h3{font-size:1.15rem;font-weight:400;font-variant:small-caps;color:var(--ink-light);margin:1.6rem 0 0.4rem;}
.sub{color:var(--ink-light);font-style:italic;margin:0 0 0.8rem;}
.tags{font-family:'Liberation Mono',monospace;font-size:0.65rem;color:var(--rule);letter-spacing:0.04em;margin-bottom:1.4rem;}
.tags span{margin-right:1rem;}
.lede{font-size:1.25rem;line-height:1.6;}
.lede::first-letter{font-size:3.1rem;float:left;line-height:0.85;padding:0.1rem 0.5rem 0 0;font-family:'EB Garamond';}
.state-line{font-size:1.5rem;font-style:italic;line-height:1.45;color:var(--ink-light);margin:1.2rem 0 2rem;max-width:820px;}
.state-line strong{color:var(--ink);font-weight:600;}
a{color:var(--status-blue);text-decoration:none;}
.toc-layout{display:grid;grid-template-columns:1fr 180px;gap:2rem;align-items:start;}
.toc{position:sticky;top:2rem;font-size:0.82rem;}
.toc-title{font-variant:small-caps;letter-spacing:0.1em;color:var(--ink-muted);margin-bottom:0.6rem;}
.toc a{display:block;color:var(--ink-light);padding:0.15rem 0;}
.toc a::before{content:'$';color:var(--rule);font-family:'Liberation Mono',monospace;font-size:0.7rem;margin-right:0.4rem;}
.toc a:hover{color:var(--accent);}
.back-to-top{color:var(--rule);font-family:'Liberation Mono',monospace;font-size:1rem;transition:color .3s,transform .3s;}
.back-to-top:hover{color:var(--accent);transform:translateY(-2px);}
.ornament{font-family:'Liberation Mono',monospace;font-size:0.9rem;color:var(--rule);text-align:center;margin:2.4rem 0;letter-spacing:0.2em;font-feature-settings:'calt' 1,'liga' 1;}
.summary-row{display:grid;grid-template-columns:repeat(4,1fr);gap:1.5rem;margin:1.5rem 0 0.5rem;}
.summary-card{position:relative;background:var(--bg-aside);border:1px solid var(--rule);padding:1rem 1rem 0.9rem;transition:border-color .3s,box-shadow .3s;}
.summary-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--card-color,var(--spark-primary));}
.summary-card:hover{border-color:var(--card-color,var(--spark-primary));box-shadow:0 2px 10px rgba(0,0,0,0.05);}
.summary-card .label{font-variant:small-caps;letter-spacing:0.06em;color:var(--ink-light);font-size:0.9rem;}
.big-number-row{display:flex;align-items:flex-end;justify-content:space-between;gap:0.5rem;}
.big-number{font-family:'Liberation Mono',monospace;font-size:2.3rem;letter-spacing:-0.02em;line-height:1.1;}
.card-spark{width:80px;height:30px;flex:none;}
.detail{font-size:0.82rem;color:var(--ink-light);margin-top:0.2rem;line-height:1.4;}
.trend{font-family:'Liberation Mono',monospace;font-size:0.78rem;margin-top:0.35rem;}
.trend.up{color:var(--status-green);}.trend.down{color:var(--status-red);}.trend.flat{color:var(--status-amber);}
.aside-container{display:grid;grid-template-columns:1fr 280px;gap:2rem;align-items:start;margin:2rem 0;}
.aside{font-size:0.85rem;line-height:1.55;color:var(--ink-light);font-style:italic;border-left:1px solid var(--rule);padding-left:1rem;}
.aside strong{font-style:normal;color:var(--ink);}
.aside .sidebar-title{font-variant:small-caps;letter-spacing:0.08em;font-size:0.82rem;color:var(--ink-light);margin-bottom:0.5rem;font-style:normal;}
.chart-container{position:relative;background:transparent;padding:0 0 0.2rem;}
.chart-container canvas{width:100%;}
.caption{font-size:0.82rem;font-style:italic;color:var(--ink-muted);text-align:center;margin-top:0.5rem;}
.table-wrapper{overflow-x:auto;margin:1.5rem 0;}
table{border-collapse:collapse;width:100%;font-size:0.92rem;}
th{font-variant:small-caps;color:var(--ink-muted);font-weight:400;text-align:right;border-bottom:1px solid var(--rule);padding:0.4rem 0.6rem;white-space:nowrap;}
th:first-child{text-align:left;}
th.left{text-align:left;}
td{text-align:right;padding:0.38rem 0.6rem;border-bottom:1px solid #eee;font-family:'Liberation Mono',monospace;font-variant-numeric:tabular-nums;font-size:0.85rem;transition:background .2s ease;}
td:first-child{text-align:left;font-family:'EB Garamond',serif;font-size:0.95rem;}
td.left{text-align:left;font-family:'EB Garamond',serif;font-size:0.95rem;}
tbody tr:hover td{background:#f1ecdc;}
.highlight-row td{background:#f4efe0;}
.pos{color:var(--status-green);}.neg{color:var(--status-red);}
.flyout{position:relative;background:var(--bg-aside);border:1px solid var(--rule);padding:1.1rem 1.2rem;margin:1.6rem 0;}
.flyout::before{content:'✦';position:absolute;top:-0.72rem;left:1rem;background:var(--bg);color:var(--accent);padding:0 0.35rem;font-size:0.8rem;}
.flyout-title{font-variant:small-caps;letter-spacing:0.08em;color:var(--ink-light);font-size:0.85rem;margin-bottom:0.4rem;}
.status-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin:1.5rem 0;}
.status-cell{border-left:3px solid var(--rule);padding:0.6rem 1rem;background:var(--bg-aside);border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);}
.status-cell+.status-cell{border-left:1px solid var(--rule);}
.status-red{border-left-color:var(--status-red)!important;}
.status-amber{border-left-color:var(--status-amber)!important;}
.status-green{border-left-color:var(--status-green)!important;}
.status-label{font-variant:small-caps;letter-spacing:0.06em;font-size:0.8rem;color:var(--ink-light);}
.status-value{font-family:'Liberation Mono',monospace;font-size:1.4rem;line-height:1.3;}
.status-note{font-size:0.78rem;color:var(--ink-muted);}
.note-red{color:var(--status-red);}.note-green{color:var(--status-green);}.note-amber{color:var(--status-amber);}
.reveal{opacity:0;transform:translateY(16px);transition:opacity .6s cubic-bezier(.25,.1,.25,1),transform .6s cubic-bezier(.25,.1,.25,1);}
.reveal.visible{opacity:1;transform:none;}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--rule);font-size:0.78rem;color:var(--ink-muted);font-family:'Liberation Mono',monospace;}
@media (max-width:800px){
  .toc-layout{grid-template-columns:1fr;}.toc{position:static;}
  .summary-row,.status-strip{grid-template-columns:1fr 1fr;}
  .aside-container{grid-template-columns:1fr;}.aside{border-left:none;padding-left:0;}
  .hide-mobile{display:none;}
}
@media (prefers-reduced-motion:reduce){.reveal{opacity:1;transform:none;transition:none;}td{transition:none;}.back-to-top{transition:none;}}
</style>
</head>
<body>
<div class="wrap">

<h1>RL Persuader — Results &amp; Diagnostics</h1>
<p class="sub">Two PPO campaigns, one statistically firm finding, and the reward-channel defect that explains the ceiling.</p>
<div class="tags"><span>CAMP-1 ppo/</span><span>CAMP-2 ppo5p_rn/</span><span>PPO • GPU</span><span>50 paired seeds</span><span>74 tests passing</span></div>

<div class="toc-layout">
<div>
<p class="lede">Across two campaigns the learned persuader beats uniform-random strategy selection by a statistically firm margin (<strong class="mono">+5.8%, t=4.81</strong>) but only ties a hand-tuned constant (the heuristic). This report puts the metrics on one page, shows where learning stalled, and traces the ceiling to a saturated LLM reward channel that leaves the action nearly irrelevant to the reward.</p>
</div>
<nav class="toc" id="toc">
  <div class="toc-title">contents</div>
  <a href="#verdict">Where we stand</a>
  <a href="#runs">Run inventory</a>
  <a href="#paired">Paired verdict</a>
  <a href="#dynamics">Training dynamics</a>
  <a href="#collapse">Behavioural collapse</a>
  <a href="#channel">The reward channel</a>
  <a href="#landscape">Strategy landscape</a>
  <a href="#next">Conclusions</a>
  <a href="#mech">Mechanism campaign</a>
  <a href="#viz">Simulation visualization</a>
  <a href="#appendix">Appendix: strategy v2</a>
</nav>
</div>

<div class="summary-row" id="kpis">
  <div class="summary-card" style="--card-color:var(--spark-secondary)">
    <div class="label">Learned − Random</div>
    <div class="big-number-row"><div class="big-number">+5.8%</div><svg class="card-spark" id="c1"></svg></div>
    <div class="detail">paired, 50 seeds • 39/50 wins</div>
    <div class="trend up">t = 4.81 • significant</div>
  </div>
  <div class="summary-card" style="--card-color:var(--status-amber)">
    <div class="label">Learned − Heuristic</div>
    <div class="big-number-row"><div class="big-number">+1.0%</div><svg class="card-spark" id="c2"></svg></div>
    <div class="detail">paired, 50 seeds • 30/50 wins</div>
    <div class="trend flat">t = 0.69 • tie</div>
  </div>
  <div class="summary-card" style="--card-color:var(--spark-primary)">
    <div class="label">Greedy reward</div>
    <div class="big-number-row"><div class="big-number">10.62</div><svg class="card-spark" id="c3"></svg></div>
    <div class="detail">mean over 50 paired episodes</div>
    <div class="trend up">SE 0.22</div>
  </div>
  <div class="summary-card" style="--card-color:var(--status-red)">
    <div class="label">Strategies used at eval</div>
    <div class="big-number-row"><div class="big-number">1/180</div><svg class="card-spark" id="c4"></svg></div>
    <div class="detail">trained replay: one strategy, 50 actions</div>
    <div class="trend down">collapse</div>
  </div>
</div>

<div class="status-strip">
  <div class="status-cell status-green"><div class="status-label">vs Random</div><div class="status-value">+0.587</div><div class="status-note note-green">t 4.81, 39/50</div></div>
  <div class="status-cell status-amber"><div class="status-label">vs Heuristic</div><div class="status-value">+0.105</div><div class="status-note note-amber">t 0.69, 30/50</div></div>
  <div class="status-cell status-red"><div class="status-label">Train reward slope</div><div class="status-value">+0.001</div><div class="status-note note-red">per iter • flat</div></div>
  <div class="status-cell status-red"><div class="status-label">Judge modal share</div><div class="status-value">54%</div><div class="status-note note-red">all at score 0.9</div></div>
</div>

<!-- ============ 1 ============ -->
<h2 id="verdict">Where we stand <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">Campaign 2 delivers the first <strong>statistically firm</strong> positive result — but it is a tie with a constant, not a demonstration of learning.</p>
<div class="aside-container">
  <div>
    <p>Campaign 2 (<span class="mono">ppo5p_rn/live_run1</span>) ran 300 PPO iterations with five shared-policy persuaders on a 50-agent network. The formal comparison is a paired evaluation over 50 shared seeds: the learned greedy policy scores <span class="mono">10.625</span> against random's <span class="mono">10.038</span> (<span class="pos mono">+5.8%</span>, <span class="mono">t=4.81</span>, winning 39/50) and the heuristic's <span class="mono">10.520</span> (<span class="mono">+1.0%</span>, <span class="mono">t=0.69</span>).</p>
    <p>The strategy signal is <em>real</em> — a fixed mid-strategy heuristic beats random too — so PPO captured something. But it captured <em>a</em> good constant, not the best one, and it never learned to exceed it.</p>
  </div>
  <div class="aside">
    <div class="sidebar-title">reading the numbers</div>
    <p><strong>Paired, not unpaired.</strong> The auto-written <span class="mono">results.json</span> used independent seeds and 10 episodes (σ≈1.5) and even reported the opposite sign; it is superseded. The paired design removes between-seed variance — that is what turned t=0.69 into t=4.81.</p>
    <p><strong>Absolute rewards are not comparable across runs.</strong> The persuasiveness gate damps diffusion globally, so each run must be judged against its own baselines.</p>
  </div>
</div>

<div class="ornament">:::</div>

<!-- ============ 2 ============ -->
<h2 id="runs">Run inventory <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">Seven runs, two environment generations; <strong>only two configs</strong> ever beat a baseline on both axes.</p>
<div class="chart-container reveal">
  <canvas id="runsChart" height="180"></canvas>
  <div class="caption">Relative reward vs random and vs heuristic, per run (%). Absolute scales differ by run; only the gaps are comparable.</div>
</div>
<div class="aside-container">
  <div class="table-wrapper">
  <table>
    <thead><tr><th>Run</th><th>Scale</th><th class="left">Env change</th><th>Greedy</th><th>Random</th><th>Heur.</th><th>Δ rand</th><th>Δ heur</th></tr></thead>
    <tbody id="runsTable"></tbody>
  </table>
  </div>
  <div class="aside">
    <div class="sidebar-title">campaign 1 → 2</div>
    <p><strong>Camp-1 trajectory.</strong> Each change was necessary and none was sufficient: summary obs couldn't tell <em>which</em> node was selected (loss), matrix obs recovered targeting (+1.7% over heuristic but still below random), and only the persuasiveness gate tied reward to message content (+13.3% over both).</p>
    <p><strong>Camp-2 reset.</strong> Scaling to 5 persuaders exposed a new failure — all five collapsed onto the same node/strategy. The <span class="mono">random_neighbor</span> protocol fixed that structurally and produced the firm +5.8%.</p>
  </div>
</div>

<div class="ornament">:::</div>

<!-- ============ 3 ============ -->
<h2 id="paired">The paired verdict <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">On shared seeds, the learned policy beats random on <strong>39 of 50</strong> episodes and the heuristic on 30 — the second is a coin flip.</p>
<div class="aside-container">
  <div class="chart-container reveal">
    <canvas id="pairedChart" height="260"></canvas>
    <div class="caption">Greedy vs random reward, per shared seed. Points above the diagonal are wins; the cloud sits just above it.</div>
  </div>
  <div class="aside">
    <div class="sidebar-title">why paired matters</div>
    <p><strong>Per-episode σ ≈ 1.5.</strong> Independent 10-episode evals have SE ≈ 0.47 — larger than the effect. Sharing seeds cancels the environment variance and leaves only the policy difference, shrinking the SE to <span class="mono">0.12</span>.</p>
    <p><strong>The heuristic is a hard bar.</strong> It is one fixed strategy — index <span class="mono">90</span> = pathos / empathetic / evidence / measured — near the top of the reachable strategy space, so "beat it" means finding the best of 180 arms through noise. Its exact rendering appears in the message table below.</p>
  </div>
</div>
<div class="table-wrapper">
<table>
  <thead><tr><th>Paired comparison</th><th>Δ reward</th><th>Δ %</th><th>SE</th><th>t</th><th>Wins</th><th class="left">Verdict</th></tr></thead>
  <tbody>
    <tr class="highlight-row"><td>Learned − Random</td><td class="pos">+0.587</td><td class="pos">+5.8%</td><td>0.122</td><td class="pos">4.81</td><td>39/50</td><td class="left">significant</td></tr>
    <tr><td>Learned − Heuristic</td><td>+0.105</td><td>+1.0%</td><td>0.152</td><td>0.69</td><td>30/50</td><td class="left">statistical tie</td></tr>
  </tbody>
</table>
</div>

<div class="ornament">:::</div>

<!-- ============ 4 ============ -->
<h2 id="dynamics">Training dynamics <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">The training reward is <strong>flat</strong>; any apparent progress lives only in the noisier greedy-eval series.</p>
<div class="aside-container">
  <div class="chart-container reveal">
    <canvas id="dynChart" height="240"></canvas>
    <div class="caption">Episode reward per iteration (5-iter samples) and greedy eval reward (every 20 iters). Fitted training slope +0.001/iter.</div>
  </div>
  <div class="aside">
    <div class="sidebar-title">flat where it counts</div>
    <p><strong>No within-run learning.</strong> The training-reward slope is <span class="mono">+0.001/iter</span> over 300 iters — indistinguishable from zero. The eval series rises early (first-3 mean 9.20 → last-3 10.78) but with per-eval SE ≈ 0.17 that is ~1–2 SE.</p>
    <p><strong>The critic fits noise.</strong> Value loss falls steadily while reward does not — a sign the critic is absorbing episode variance, not signal.</p>
  </div>
</div>
<div class="aside-container">
  <div class="table-wrapper">
  <table>
    <thead><tr><th>Metric</th><th>Early</th><th>Late</th><th>Slope / iter</th></tr></thead>
    <tbody>
      <tr><td>Training reward</td><td>10.16</td><td>10.17</td><td>+0.001</td></tr>
      <tr><td>Greedy eval reward</td><td>8.63 <span style="color:#888">@20</span></td><td>11.71 <span style="color:#888">@280</span></td><td>+0.006</td></tr>
      <tr><td>Policy entropy</td><td>5.19 <span style="color:#888">ln 180</span></td><td>4.02</td><td>−0.004</td></tr>
      <tr><td>Critic value loss</td><td>3.79</td><td>0.25</td><td>−0.012</td></tr>
    </tbody>
  </table>
  </div>
  <div class="aside">
    <div class="sidebar-title">sample budget</div>
    <p><strong>400 transitions per update.</strong> 8 episodes × 10 periods × 5 persuaders, minibatch 256, 4 epochs → ~8 gradient steps/iter, ~2,400 total, against a 180-arm bandit whose arm means differ by ~0.6 with per-episode σ≈1.5.</p>
    <p><strong>Entropy stayed up.</strong> Unlike camp-1's collapse (6.9→3.0), entropy only drifted 5.19→4.02 — so exploration wasn't the binding constraint here.</p>
  </div>
</div>

<div class="ornament">:::</div>

<!-- ============ 5 ============ -->
<h2 id="collapse">Behavioural collapse <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">Over <strong>1200 trained actions</strong> the policy uses just <strong>three strategies</strong> (70% / 15% / 15%); a single 50-action replay showed only one. Random used 43 — strategy-level collapse, not targeting collapse.</p>
<div class="aside-container">
  <div class="chart-container reveal">
    <canvas id="stratChart" height="220"></canvas>
    <div class="caption">Distinct strategy indices used across an episode replay, trained policy vs random policy (same network seed).</div>
  </div>
  <div class="aside">
    <div class="sidebar-title">the converged mode</div>
    <p><strong>Three arms, always.</strong> Across 24 episodes the policy picked <span class="mono">#141</span> (ethos / urgent / social_proof) 836 times, <span class="mono">#153</span> (ethos / empathetic / personal_story) 184, and <span class="mono">#13</span> (logos / neutral / authority) 180 — a tiny fixed set, not a policy that keeps adapting.</p>
    <p><strong>This is the expected symptom.</strong> If the reward barely moves with the action (next section), the optimal policy is constant, and PPO finds one good constant then stops.</p>
  </div>
</div>
<div class="aside-container">
  <div class="chart-container reveal">
    <canvas id="replayChart" height="200"></canvas>
    <div class="caption">Per-period summed reward, trained vs random replay on the same 50-agent network and seed.</div>
  </div>
  <div class="aside">
    <div class="sidebar-title">single replay ≠ verdict</div>
    <p><strong>Illustrative only.</strong> The trained replay scores 9.79 vs 8.91 and adopts 21 vs 17 agents — but one episode carries no statistical weight. The paired 50-seed result above is the formal comparison.</p>
    <table style="margin-top:0.6rem;">
      <thead><tr><th class="left" style="font-size:0.75rem;">Replay</th><th>Reward</th><th>Adopted</th></tr></thead>
      <tbody><tr><td>Trained</td><td>9.79</td><td>21</td></tr><tr><td>Random</td><td>8.91</td><td>17</td></tr></tbody>
    </table>
  </div>
</div>

<h3>what the saturating strategies actually say</h3>
<p style="max-width:820px;color:var(--ink-light)">Same target persona, different strategy. The learned policy concentrates <strong>70% of its actions</strong> on ethos / urgent / social_proof and splits the rest between two more arms; the heuristic commits entirely to pathos / empathetic / evidence; the low-score arms are the only ones the reward clearly separates.</p>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">Role</th><th>Idx</th><th class="left">Strategy spec</th><th>Judge</th><th class="left hide-mobile">Rendered message (persona 0)</th></tr></thead>
  <tbody id="msgTable"></tbody>
</table>
</div>

<div class="ornament">:::</div>

<!-- ============ 6 ============ -->
<h2 id="channel">The reward channel is saturated <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">The frozen LLM judge collapses <strong>180 distinct messages</strong> onto 4–6 scores; 54% of all 2,160 judgments land on a single value, 0.9.</p>
<div class="chart-container reveal">
  <canvas id="judgeChart" height="220"></canvas>
  <div class="caption">Distribution of judge persuasiveness over 180 strategies × 12 personas (n=2,160), read from the cached render+judge matrix.</div>
</div>
<div class="aside-container">
  <div class="table-wrapper">
  <table>
    <thead><tr><th>Persona</th><th>Judged</th><th>Distinct scores</th><th>SD</th><th>Modal score</th><th>Modal share</th></tr></thead>
    <tbody id="judgeTable"></tbody>
  </table>
  </div>
  <div class="aside">
    <div class="sidebar-title">why this caps everything</div>
    <p><strong>The multiplier is the only shaped path.</strong> <span class="mono">p = [0.25·content_sim + C] · score</span>. With score sd ≈ 0.05–0.07 and a 54% modal pile-up, changing strategy barely changes <span class="mono">p</span> — so the reward gradient is ≈ 0 for most arms.</p>
    <p><strong>Fix order matters.</strong> Reward reshaping and variance reduction (Rao-Blackwellizing the accept gate, more rollouts) cannot manufacture signal. The channel must first be made discriminative — e.g. a pairwise / ranking judge instead of absolute 0–10 scoring.</p>
  </div>
</div>

<div class="flyout">
  <div class="flyout-title">acceptance test for a repaired channel</div>
  <p>Over 180 strategies for a fixed persona, the judge-score distribution should have <strong>SD ≳ 0.15</strong> and no single bucket above ~40%. Today it is SD 0.072 and 54% on one bucket.</p>
</div>

<h3>the generator end: variety without structure</h3>
<p style="max-width:820px;color:var(--ink-light)">The judge is only one end of the pipe. Does the <em>strategy</em> even change the post? We embedded all 180 rendered messages for a fixed persona with the same model the env uses (jina-v5) and measured how much of the message variation the strategy axes actually organise.</p>
<div class="aside-container">
  <div class="chart-container reveal">
    <canvas id="genChart" height="240"></canvas>
    <div class="caption">Nearest-neighbour purity of the message embedding for each strategy dimension, against chance (1 / number of levels). Above chance = the dimension reliably shapes the post.</div>
  </div>
  <div class="aside">
    <div class="sidebar-title">variety ≠ learnable variety</div>
    <p><strong>The posts do differ.</strong> Mean pairwise cosine across the 180 messages is <span class="mono">0.31</span> — well short of identical (unrelated sentences score <span class="mono">0.09</span>), and no pair exceeds <span class="mono">0.90</span>.</p>
    <p><strong>But not by the strategy.</strong> Same-level and cross-level messages are nearly as similar (gap ≤ 0.04); tone, emphasis and intensity sit at or near chance for predicting a message's neighbours. Changing the label changes the words, but not in a consistent, generalisable direction — so the 180 arms look idiosyncratic to the agent.</p>
  </div>
</div>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">Readout of the message</th><th>Strength</th><th class="left">Interpretation</th></tr></thead>
  <tbody id="genTable"></tbody>
</table>
</div>

<h3>can the generator be fixed? a two-sided A/B</h3>
<p style="max-width:820px;color:var(--ink-light)">Two interventions on the generator — an <strong>exemplar scaffold</strong> that spells out each strategy dimension with worked examples, and <strong>best-of-n</strong> sampling (n = 8, T = 0.9) with two selectors — tested both on the posts themselves and, through the network, on reward spread (60 strategies × 6 shared seeds).</p>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">Variant</th><th>Pair cos ↓</th><th>Emph NN ↑</th><th>Judge SD ↑</th><th>Modal ↓</th><th>Openings ↑</th><th>Reward</th><th>Reward SD</th><th class="left">Verdict</th></tr></thead>
  <tbody id="condTable"></tbody>
</table>
</div>
<div class="flyout">
  <div class="flyout-title">what the A/B says</div>
  <p id="condVerdict">—</p>
</div>

<div class="ornament">:::</div>

<!-- ============ 7 ============ -->
<h2 id="landscape">Strategy reward landscape <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">Across all 180 strategies the network reward spans <strong id="landSpread">–</strong>, but the middle half covers only <strong>0.90</strong> and the top decile just <strong>0.61</strong>. The heuristic ranks <strong>6th of 180</strong>; the learned policy's three modes rank 5th, 13th and 73rd.</p>
<div class="aside-container">
  <div class="chart-container reveal">
    <canvas id="landChart" height="250"></canvas>
    <div class="caption">Mean episode reward per strategy (all five persuaders fixed to that strategy), 6 paired seeds. Heuristic (90) and the three learned modes (141 / 153 / 13) highlighted against the best arm (168).</div>
  </div>
  <div class="aside">
    <div class="sidebar-title">a plateau, not a gradient</div>
    <p><strong>Broad plateau.</strong> The middle half of the arms sits inside <span class="mono">0.90</span> reward points against a per-episode σ ≈ 1.5, so the reward gives PPO almost no signal to rank one good arm above another.</p>
    <p><strong>Only the tail is distinguishable.</strong> The reward sharply punishes a few bad arms (rank 170+ fall to ≈ 7.7) — enough to learn "avoid the bad", never to prefer the best.</p>
    <p><strong>The policy found the plateau.</strong> Its three modes rank 5, 13 and 73; the last one drags the mixture back to the heuristic's level (rank 6).</p>
  </div>
</div>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">Role</th><th>Idx</th><th>Rank</th><th class="left">Spec</th><th>Mean reward</th><th>SE</th><th>Mean judge</th><th class="left">Note</th></tr></thead>
  <tbody id="landTable"></tbody>
</table>
</div>
<div class="aside-container">
  <div class="chart-container reveal">
    <canvas id="scoreRewChart" height="230"></canvas>
    <div class="caption">Per-strategy mean judge score vs mean network reward. A near-vertical cluster is the saturation: score barely varies, so reward barely varies.</div>
  </div>
  <div class="aside">
    <div class="sidebar-title">the channel sets the reward</div>
    <p><strong>Reward tracks the saturated score.</strong> Correlation between a strategy's mean judge score and its mean network reward is <strong>r = <span id="landCorr">–</span></strong>. Because the judge collapses most strategies onto one value, the network reward inherits that flatness.</p>
    <p><strong>Order of operations.</strong> No optimizer or budget change can create spread the reward does not have — repair the channel first, then variance reduction pays off.</p>
  </div>
</div>

<div class="ornament">:::</div>

<!-- ============ 8 ============ -->
<h2 id="next">Conclusions &amp; next levers <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">RL adds real signal over chance but does not exceed a tuned constant; the binding defect is the <strong>reward channel</strong>, not the optimizer.</p>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">Lever</th><th class="left">Targets</th><th class="left">Status</th><th class="left">Expected effect</th></tr></thead>
  <tbody>
    <tr class="highlight-row"><td class="left">Pairwise / ranking judge → reward model</td><td class="left">#2 saturated channel</td><td class="left"><span class="pos">done</span></td><td class="left">score SD ×3.2 (0.072→0.233); landscape spread ÷ SE ≈ 1 → 18</td></tr>
    <tr><td class="left">Full-embedding observation (drop 8-d mean-pool)</td><td class="left">#5 state aliasing</td><td class="left"><span class="neg">next</span></td><td class="left">lets the policy condition on the features the reward uses</td></tr>
    <tr><td class="left">Rao-Blackwellize the accept gate (train on p·g)</td><td class="left">#1 estimator variance</td><td class="left"><span style="color:var(--status-amber)">staged</span></td><td class="left">removes the dominant Bernoulli noise, unbiased</td></tr>
    <tr><td class="left">More episodes/iter + common random numbers</td><td class="left">#1 · #4 budget</td><td class="left"><span style="color:var(--status-amber)">staged</span></td><td class="left">per-iteration SE below the achievable gain</td></tr>
    <tr><td class="left">Factorized head + entropy schedule</td><td class="left">#4 exploration</td><td class="left"><span class="pos">done</span></td><td class="left">prevents premature convergence</td></tr>
    <tr><td class="left">Paired evaluation harness</td><td class="left">measurement</td><td class="left"><span class="pos">done</span></td><td class="left">turned an inconclusive read into t=4.81</td></tr>
  </tbody>
</table>
</div>
<div class="flyout">
  <div class="flyout-title">one-line verdict</div>
  <p>PPO learned to avoid bad strategies and settle on a good one (<span class="pos mono">+5.8%</span> over random, significant) but the reward channel is too flat to reward anything finer — so it ties the best constant. Repair the channel first; variance fixes are necessary but not sufficient.</p>
</div>

<div class="ornament">:::</div>

<!-- ============ A ============ -->
<h2 id="appendix">Appendix: a strategy definition that can carry learnable variation <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">A better <em>prompt</em> on the same abstract grid is not the fix. Three separate properties are required, and a pilot of a redesigned action shows the generator can supply two of them — the third lives in the reward channel.</p>

<p style="max-width:820px;color:var(--ink-light)">We tested a redesigned action — <strong>angle</strong> (which concern to foreground: cost / safety / fairness / freedom / community / effectiveness) × <strong>evidence</strong> (statistic / expert / anecdote / peers) × <strong>address</strong> (taxpayer / parent / worker) — rendered for 12 personas and judged, then scored against the three properties that separate “real variety” from “learnable variety”.</p>

<div class="table-wrapper">
<table>
  <thead><tr><th class="left">Property</th><th class="left">Question</th><th class="left">Current grid</th><th class="left">Redesigned pilot</th><th class="left">Target</th><th>Verdict</th></tr></thead>
  <tbody id="v2Table"></tbody>
</table>
</div>

<div class="aside-container">
  <div class="aside">
    <div class="sidebar-title">what the pilot proves</div>
    <p><strong>Realizable.</strong> Concrete slots are adopted by the generator — <span class="mono">angle</span> reaches 0.85 nearest-neighbour purity (chance 0.17) and <span class="mono">address</span> 0.77 (chance 0.33), where the old <span class="mono">emphasis</span> sat at chance (0.31). The posts genuinely differ.</p>
    <p><strong>Readable — improved, not solved.</strong> The judge spread widens (SD 0.062 → 0.095) and the modal bucket falls (71% → 57%), but stops short of the resolution RL needs.</p>
    <p><strong>Not yet conditional.</strong> The action has a large <em>main</em> effect (some angles are simply better for everyone) but almost no <em>interaction</em> with the target: the oracle gain from choosing the angle per persona is 0.012, and the persona × angle interaction is 4.5%. That is the same trap as the old grid — a constant best arm, so nothing target-specific to learn.</p>
  </div>
  <div class="aside">
    <div class="sidebar-title">the proposed definition</div>
    <p><strong>1 · Action = a target-matched plan.</strong> Keep the slots that proved realizable (<span class="mono">angle</span>, <span class="mono">address</span>) and make the rest a <em>checkable constraint</em> rather than a vague adjective: the post must contain a digit (statistic), a named source (expert), a first-person detail (anecdote). Drop <span class="mono">tone</span>/<span class="mono">intensity</span> — they were adopted at chance and only cost persuasiveness.</p>
    <p><strong>2 · Readout must score the match.</strong> Conditionality cannot come from the generator alone. Replace the absolute 0–10 judge with a <strong>pairwise / Bradley–Terry</strong> judge over (post, target) pairs — “which of these two fits this reader?” — so the reward depends on the pairing, not on a global quality constant.</p>
    <p><strong>3 · Or select from a validated bank.</strong> The robust alternative: pre-render a bank of candidates per (target-class × angle × address), score them once, and let the policy <em>select</em>. Then the action→reward map is discrete, measured, and target-conditioned by construction — no reliance on the LLM obeying a label.</p>
  </div>
</div>

<h3>the readout half, measured: a Bradley–Terry judge</h3>
<p style="max-width:820px;color:var(--ink-light)">Fix #2 above is now implemented and measured. A pairwise judge plays 3,600 (post, target) arms off in head-to-head comparisons and fits a Bradley–Terry score to each; we then read the resulting reward through the same 50-agent network. The readout's resolution nearly triples, and the flat plateau becomes a gradient the optimizer can climb.</p>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">Readout</th><th>Score SD</th><th>Distinct scores</th><th>Modal bucket</th><th>Modal share</th><th>Landscape spread</th><th>Spread ÷ SE</th><th class="left">Verdict</th></tr></thead>
  <tbody id="btTable"></tbody>
</table>
</div>
<div class="aside-container">
  <div class="chart-container reveal">
    <canvas id="btChart" height="250"></canvas>
    <div class="caption">Mean episode reward per arm under the Bradley–Terry readout (72 arms × 8 paired seeds), coloured by address. Dashed line = best single constant.</div>
  </div>
  <div class="aside">
    <div class="sidebar-title">what the pairwise judge buys</div>
    <p id="btAside">—</p>
  </div>
</div>
<div class="flyout">
  <div class="flyout-title">the readout is no longer the binding constraint</div>
  <p id="btVerdict">—</p>
</div>

<h3>campaign 3 verdict: the trained policy under the Bradley–Terry readout</h3>
<p style="max-width:820px;color:var(--ink-light)">With the v2 action space and the pairwise readout in place, the full 300-iteration PPO run (<span class="mono">output/ppo_v2_bt/live_run2</span>, 5 persuaders, 50 agents, 10 periods) was trained end-to-end. The formal comparison is again a paired evaluation over 50 shared seeds, reported every 20 iterations.</p>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">Comparison</th><th>Reward</th><th>Δ</th><th>t</th><th class="left">Verdict</th></tr></thead>
  <tbody id="c3Table"></tbody>
</table>
</div>
<div class="chart-container reveal">
  <canvas id="c3Chart" height="235"></canvas>
  <div class="caption">Paired eval (50 shared seeds) every 20 iterations — learned greedy vs the fixed random baseline. The learned policy is stable and above random from iteration 20 onward.</div>
</div>
<div class="aside-container">
  <div class="aside">
    <div class="sidebar-title">what the policy learned</div>
    <p>It converges on <span class="mono">evidence=anecdote</span> and <span class="mono">address=worker</span>, cycling the angle between <span class="mono">effectiveness</span> (#68) and <span class="mono">fairness</span> (#32) — the same top region the independent landscape probe ranked. Final strategy entropy <span class="mono">0.39</span>: a handful of arms, not one.</p>
  </div>
  <div class="aside">
    <div class="sidebar-title">the caveat that frames the next step</div>
    <p>The judge saturates near its ceiling (<span class="mono">~0.99</span>, 90–98% in the top bucket). The gain is real <em>against this readout</em>, but the readout scores persona-fit, not the message's direction toward the goal — so the next lever is measuring goal direction, not further tuning the optimizer.</p>
  </div>
</div>

<h3>campaign 4: the dynamics constants, measured</h3>
<p style="max-width:820px;color:var(--ink-light)">The influence weights α (four × <span class="mono">0.25</span>) and the stance step β (<span class="mono">0.4</span>) were inherited placeholders — the paper writes <span class="mono">p_ij ← InfluenceProbability(…)</span> and <span class="mono">Δs_j ← StateTransition(…)</span> without expanding either. Two reproducible scripts replace the assumption with evidence.</p>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">α probe (Track A1)</th><th>value</th><th class="left">reading</th></tr></thead>
  <tbody id="calAlphaTable"></tbody>
</table>
</div>
<div class="chart-container reveal">
  <canvas id="calBetaChart" height="235"></canvas>
  <div class="caption">Joint (β, κ) sweep against the paper's dynamics. β and the α-scale κ enter only through ρ = β·κ. Early ΔS and the plateau are jointly matchable near ρ ≈ 2, but late ΔS is structurally 0 — the transition reaches a static fixed point, not the paper's sustained ≈ −0.107.</div>
</div>
<div class="aside-container">
  <div class="aside">
    <div class="sidebar-title">why the α fit is a negative result</div>
    <p>The design matrix has effective rank <span class="mono" id="calRank">—</span>/4 (profile- and degree-similarity are near-constant), so the four α's are <em>not separately identifiable</em> from behavior. A logistic fit returns values far from 0.25 while the design is collinear — noise. The equal weights are kept, now documented as an <em>unidentifiable prior</em>.</p>
  </div>
  <div class="aside">
    <div class="sidebar-title">β is not the binding knob</div>
    <p>No setting reproduces the paper's smooth late phase: the model either stays shallow or collapses to S = −1, with no stable intermediate consensus. The residual is the text-independent pull-to-source — handed to campaign 5, not fixed by re-tuning β.</p>
  </div>
</div>

<h3>campaign 5: goal-direction injection</h3>
<p style="max-width:820px;color:var(--ink-light)">The reward and the stance delta were both text-direction-blind: an off-goal-but-persuasive post was scored like an on-goal one, and the transition pulled the target toward the source regardless of what the post argued. Campaign 5 measures the message's alignment with a named goal proposition, <span class="mono">a(m) ∈ [0,1]</span>, and injects it into the step (<span class="mono">s_eff = s_src·(2a−1)</span>) behind <span class="mono">--goal-delta</span>.</p>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">goal probe (Track B4)</th><th>value</th><th class="left">reading</th></tr></thead>
  <tbody id="goalTable"></tbody>
</table>
</div>

<h3>campaign 6: goal-direction A/B</h3>
<p style="max-width:820px;color:var(--ink-light)">The injection is only as good as <span class="mono">a(m)</span>. Two matched goal-delta arms run against the no-goal baseline under an identical protocol (v2 space, BT readout, 50 agents / 5 persuaders / erdos_renyi / 10 periods, <span class="mono">obs_mode=pair</span>, seed 0, paired eval on 50 shared seeds every 20 iterations); they differ only in the alignment backend — and, since campaign 5, the injection applies to the RL persuader's messages only, not the crowd's physics.</p>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">a(m) backend (n real messages)</th><th>semdiff</th><th>semdiff+seed</th><th>judge</th><th class="left">reading</th></tr></thead>
  <tbody id="gbTable"></tbody>
</table>
</div>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">paired eval @ iter</th><th>baseline</th><th>goal (semdiff)</th><th>goal (judge)</th><th class="left">reading</th></tr></thead>
  <tbody id="abTable"></tbody>
</table>
</div>
<div class="aside-container">
  <div class="aside">
    <div class="sidebar-title">the backend is the crux</div>
    <p id="abBackend">—</p>
  </div>
  <div class="aside">
    <div class="sidebar-title">goal metrics (tail mean)</div>
    <p id="abMetrics">—</p>
  </div>
</div>

<div class="flyout">
  <div class="flyout-title">validation gate (run before committing)</div>
  <p>Any candidate strategy space must clear all three, measured exactly as here: realisability <span class="mono">NN ≥ 0.60</span>; readability <span class="mono">judge SD ≥ 0.15</span> and <span class="mono">modal &lt; 40%</span>; and a target×action oracle gain that exceeds the per-episode reward noise (<span class="mono">SE ≈ 0.53</span>). A space that fails the third is unlearnable no matter how diverse its posts.</p>
</div>

<!-- ============ mechanism campaign ============ -->
<h2 id="mech">Mechanism campaign — C7 cascade vs C8 exposure <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">Two propagation mechanisms imported from the colleague's Melodie reimplementation, ported faithfully onto our core and trained from scratch on campaign 6's protocol — <strong>300 iterations × 32 episodes</strong>, v2 space, BT judge, 50 agents / 5 persuaders / erdos_renyi / 10 periods, paired eval every 20 iterations.</p>
<p style="max-width:820px;color:var(--ink-light)">They are deliberately different experiments. <span class="mono">cascade</span> (C7) is stance-ful, but its imported emergence rule fires on <em>any</em> observable change — and a generative backend always emits something different — so the whole reachable component adopts in <strong>period 0</strong>: it is a single-round persuasion-quality benchmark. <span class="mono">exposure</span> (C8) freezes stances entirely and spreads through a persistent cross-period queue gated by an LLM accept, so it is the arm where the multi-period diffusion problem is real. This is not a port defect — it is their rule meeting our backend, and the faithful port is what makes it visible.</p>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">arm</th><th class="left">mechanism</th><th>reward</th><th>adopted</th><th>eval greedy @300</th><th>t vs random</th><th>t vs heuristic</th><th>eval judge</th></tr></thead>
  <tbody id="mechTable"></tbody>
</table>
</div>
<div class="chart-container reveal">
  <canvas id="mechChart" height="240"></canvas>
  <div class="caption" id="mechCaption"></div>
</div>
<div class="aside-container">
  <div class="aside">
    <div class="sidebar-title">C7 — a clean single-round win</div>
    <p><strong>Beats both baselines in all 15 evals.</strong> The paired greedy reward rises 1.413 → <span class="mono">1.684</span>; every eval beats random (t = +4.3…+7.7) and the heuristic by a wide margin. Message quality improves with it — eval judge 0.931 → 0.979 — and strategy entropy does <em>not</em> collapse (1.70 → 1.82), so the policy is not narrowing onto one angle.</p>
  </div>
  <div class="aside">
    <div class="sidebar-title">C8 — quality up, adoption flat</div>
    <p><strong>Parity with random, above the heuristic.</strong> Its judge score rises sharply (0.523 → 0.808) but reward stays flat (14.20 → 14.30) and its edge over random sits at zero ± noise (mean t = +0.58 over 15 evals). Adoption is decided by the LLM accept gate, not the strategy, so improving <span class="mono">a(m)</span> does not move the reward. An earlier "flat to worse" read did not survive to iteration 300 and has been withdrawn.</p>
  </div>
</div>

<!-- ============ visualization ============ -->
<h2 id="viz">Simulation visualization — LLMxIO 3D dashboard <a href="#toc" class="back-to-top" title="Back to contents">+</a></h2>
<p class="state-line">The latest campaign arms are exported as an interactive 3D social network — agents, the follow graph, and the re-share cascades — in the LLMxIO dashboard, served locally on <span class="mono">:3009</span>.</p>
<div class="table-wrapper">
<table>
  <thead><tr><th class="left">run</th><th class="left">policy</th><th>adopted</th><th>mean stance</th><th>reward</th><th class="left">edges (follow / retweet / reply)</th></tr></thead>
  <tbody id="vizTable"></tbody>
</table>
</div>
<div class="aside-container">
  <div class="aside">
    <div class="sidebar-title">how the mapping works</div>
    <p>Agents map to personas (<span class="mono">u&lt;id&gt;</span> organic, <span class="mono">io&lt;id&gt;</span> info-op persuader). The whole topology is drawn from t0; each <em>adoption</em> adds a re-share edge and each non-adopting <em>accept</em> a reply edge, so the timeline shows the campaign spreading outward.</p>
  </div>
  <div class="aside">
    <div class="sidebar-title">artifacts</div>
    <p>Data: <span class="mono">report/dashboard/</span>. Runner: <span class="mono">code/dashboard/setup_dashboard.sh</span>. Exporter: <span class="mono">code/export_dashboard.py</span>. Dashboard pinned at commit <span class="mono">9f913ee</span>.</p>
  </div>
</div>

<footer>
  generated from output/ppo5p_rn/live_run1/{training_log,results,results_final}.json,<br>
  output/ppo/*/results.json, replay trajectories, and the cached render+judge matrix (.llm_cache).<br>
  campaigns 4–5 from output/calibration/{alphas,beta_fine,goal_validate}.json.<br>
  campaign 6 from report/{ab_goal,goal_backends}.json; visualization from report/dashboard/.<br>
  report_data.json • judge_probe.json • 2026-10-17
</footer>

</div>

<script src="vendor/chart.umd.min.js"></script>
<script>
const D = __DATA__;

Chart.defaults.font.family = "'EB Garamond', Georgia, serif";
Chart.defaults.font.size = 13;
Chart.defaults.color = '#555';
Chart.defaults.plugins.legend.labels.usePointStyle = false;
Chart.defaults.plugins.legend.labels.boxWidth = 8;
Chart.defaults.plugins.legend.labels.boxHeight = 8;
Chart.defaults.plugins.legend.labels.borderRadius = 4;
Chart.defaults.plugins.legend.labels.font = { family: "'EB Garamond', serif", size: 11 };
Chart.defaults.plugins.legend.labels.padding = 14;
Chart.defaults.plugins.tooltip.backgroundColor = '#1a1a1a';
Chart.defaults.plugins.tooltip.cornerRadius = 2;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.scale.grid.color = '#eee';
Chart.defaults.animation = { duration: 800, easing: 'easeOutQuart' };
const _gl = Chart.defaults.plugins.legend.labels.generateLabels;
Chart.defaults.plugins.legend.labels.generateLabels = function(c){
  const items = _gl.call(this,c);
  items.forEach(it => { const ds=c.data.datasets[it.datasetIndex]; const col=ds.borderColor||ds.backgroundColor;
    it.fillStyle=Array.isArray(col)?col[0]:col; it.strokeStyle='transparent'; it.lineWidth=0; it.borderRadius=4; });
  return items;
};

function drawSparkline(id, values, color){
  const svg=document.getElementById(id); if(!svg) return;
  const w=svg.clientWidth||80, h=svg.clientHeight||30, pad=2;
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
  const max=Math.max(...values), min=Math.min(...values), range=(max-min)||1;
  const step=(w-pad*2)/(values.length-1);
  const pts=values.map((v,i)=>[pad+i*step, h-pad-((v-min)/range)*(h-pad*2)]);
  const line=pts.map(p=>`${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const area=[`${pad},${h-pad}`,...pts.map(p=>`${p[0].toFixed(1)},${p[1].toFixed(1)}`),`${(pad+(values.length-1)*step).toFixed(1)},${h-pad}`].join(' ');
  svg.innerHTML=`<polygon points="${area}" fill="${color}" opacity="0.14"/><polyline points="${line}" fill="none" stroke="${color}" stroke-width="1.4" stroke-linejoin="round"/><circle cx="${pts[pts.length-1][0].toFixed(1)}" cy="${pts[pts.length-1][1].toFixed(1)}" r="2.2" fill="${color}"/>`;
}

function downsample(a,n){const step=a.length/n;const out=[];for(let i=0;i<n;i++)out.push(a[Math.min(a.length-1,Math.round(i*step))]);return out;}

// KPI card sparklines
drawSparkline('c1', downsample(D.train.reward,30), '#2a7a5a');
drawSparkline('c2', D.train.eval_reward, '#c89000');
drawSparkline('c3', D.train.eval_reward, '#c45a28');
drawSparkline('c4', D.train.reward.slice(0,20), '#a02a2a');

// --- runs table + chart ---
const vtColor={win:'var(--status-green)',loss:'var(--status-red)',mixed:'var(--status-amber)',mock:'var(--ink-muted)'};
const tb=document.getElementById('runsTable');
D.runs.forEach(r=>{
  const cls=v=>v>=0?'pos':'neg';
  tb.insertAdjacentHTML('beforeend',
   `<tr><td>${r.run}<span style="color:#aaa"> •c${r.camp}</span></td><td>${r.scale}</td><td class="left">${r.env}</td>`+
   `<td>${r.greedy.toFixed(3)}</td><td>${r.random.toFixed(3)}</td><td>${r.heur.toFixed(3)}</td>`+
   `<td class="${cls(r.dr)}">${r.dr>=0?'+':''}${r.dr.toFixed(1)}%</td>`+
   `<td class="${cls(r.dh)}">${r.dh>=0?'+':''}${r.dh.toFixed(1)}%</td></tr>`);
});
new Chart(document.getElementById('runsChart'),{
  type:'bar',
  data:{labels:D.runs.map(r=>r.run),
    datasets:[
      {label:'Δ vs random (%)',data:D.runs.map(r=>r.dr),backgroundColor:'rgba(196,90,40,0.55)',borderColor:'rgba(196,90,40,0.8)',borderWidth:0.5,borderRadius:2},
      {label:'Δ vs heuristic (%)',data:D.runs.map(r=>r.dh),backgroundColor:'rgba(42,122,90,0.5)',borderColor:'rgba(42,122,90,0.8)',borderWidth:0.5,borderRadius:2}
    ]},
  options:{responsive:true,interaction:{mode:'index',intersect:false},
    scales:{x:{grid:{display:false},ticks:{maxTicksLimit:8,font:{size:10},maxRotation:30,minRotation:0}},
            y:{grid:{color:'#eee'},title:{display:true,text:'Δ reward (%)'}}},
    plugins:{legend:{position:'top',align:'end'}}}
});

// --- paired scatter ---
const g=D.paired.greedy.per_episode, rd=D.paired.random.per_episode;
const mx=Math.max(...g,...rd), mn=Math.min(...g,...rd);
const pad=(mx-mn)*0.05;
new Chart(document.getElementById('pairedChart'),{
  type:'scatter',
  data:{datasets:[
    {label:'Greedy vs Random (50 seeds)',data:g.map((v,i)=>({x:rd[i],y:v})),
     backgroundColor:'rgba(196,90,40,0.6)',borderColor:'rgba(196,90,40,0.9)',pointRadius:3.5,pointHoverRadius:5},
    {label:'y = x',type:'line',data:[{x:mn-pad,y:mn-pad},{x:mx+pad,y:mx+pad}],
     borderColor:'rgba(0,0,0,0.25)',borderDash:[5,4],borderWidth:1,pointRadius:0,fill:false}
  ]},
  options:{responsive:true,
    scales:{x:{min:mn-pad,max:mx+pad,title:{display:true,text:'Random reward (paired seed)'},grid:{color:'#eee'}},
            y:{min:mn-pad,max:mx+pad,title:{display:true,text:'Learned greedy reward'},grid:{color:'#eee'}}},
    plugins:{legend:{position:'top',align:'end'},tooltip:{callbacks:{label:c=>`rand ${c.raw.x.toFixed(2)} → greedy ${c.raw.y.toFixed(2)}`}}}}
});

// --- training dynamics ---
const tr=D.train;
new Chart(document.getElementById('dynChart'),{
  data:{labels:tr.iteration,
    datasets:[
      {type:'line',label:'Train reward (per iter)',data:tr.reward,
       borderColor:'rgba(196,90,40,0.85)',backgroundColor:'rgba(196,90,40,0.06)',fill:true,borderWidth:1.4,pointRadius:0,tension:0.3,pointHitRadius:6},
      {type:'line',label:'Greedy eval reward',data:tr.iteration.map((it,i)=>tr.eval_iter.includes(it)?tr.eval_reward[tr.eval_iter.indexOf(it)]:null),
       borderColor:'rgba(90,90,170,0.9)',backgroundColor:'rgba(90,90,170,0.9)',fill:false,borderWidth:2,pointRadius:3.5,tension:0.2,spanGaps:true},
      {type:'line',label:'Train trend (+0.001/iter)',data:tr.iteration.map(it=>10.16+0.001*it),
       borderColor:'rgba(0,0,0,0.3)',borderDash:[6,4],borderWidth:1,pointRadius:0,fill:false}
    ]},
  options:{responsive:true,interaction:{mode:'index',intersect:false},
    scales:{x:{type:'linear',title:{display:true,text:'PPO iteration'},grid:{color:'#eee'},ticks:{maxTicksLimit:12}},
            y:{title:{display:true,text:'Episode reward'},grid:{color:'#eee'}}},
    plugins:{legend:{position:'top',align:'end'}}}
});

// --- strategy collapse bars ---
const tU=D.replay.trained_at210.n_unique_strategies, rU=D.replay.random.n_unique_strategies;
new Chart(document.getElementById('stratChart'),{
  type:'bar',
  data:{labels:['Learned greedy','Random strategy'],
    datasets:[{label:'distinct strategy indices used (of 180)',data:[tU,rU],
      backgroundColor:['rgba(160,42,42,0.6)','rgba(90,90,170,0.55)'],borderColor:['rgba(160,42,42,0.8)','rgba(90,90,170,0.8)'],borderWidth:0.5,borderRadius:2}]},
  options:{responsive:true,indexAxis:'y',
    scales:{x:{title:{display:true,text:'distinct strategies (max 180)'},grid:{color:'#eee'}},y:{grid:{display:false}}},
    plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.raw} distinct strategies`}}}}
});

// --- replay per-period ---
new Chart(document.getElementById('replayChart'),{
  type:'line',
  data:{labels:[1,2,3,4,5,6,7,8,9,10],
    datasets:[
      {label:'Trained @ iter 210',data:D.replay.trained_at210.period_rewards,borderColor:'rgba(196,90,40,0.85)',backgroundColor:'rgba(196,90,40,0.85)',fill:false,borderWidth:2,pointRadius:3,tension:0.25},
      {label:'Random strategy',data:D.replay.random.period_rewards,borderColor:'rgba(90,90,170,0.8)',backgroundColor:'rgba(90,90,170,0.8)',fill:false,borderWidth:1.5,borderDash:[4,3],pointRadius:3,tension:0.25}
    ]},
  options:{responsive:true,interaction:{mode:'index',intersect:false},
    scales:{x:{title:{display:true,text:'period'},grid:{display:false}},y:{title:{display:true,text:'summed reward'},grid:{color:'#eee'}}},
    plugins:{legend:{position:'top',align:'end'}}}
});

// --- judge histogram ---
new Chart(document.getElementById('judgeChart'),{
  type:'bar',
  data:{labels:D.judge.hist_labels,
    datasets:[{label:'judgments',data:D.judge.hist_values,
      backgroundColor:D.judge.hist_labels.map(l=>parseFloat(l)>=0.9?'rgba(160,42,42,0.6)':'rgba(196,90,40,0.5)'),
      borderWidth:0,borderRadius:2}]},
  options:{responsive:true,
    scales:{x:{title:{display:true,text:'judge persuasiveness score (0.5 = neutral, 1.0 = maximally compelling)'},grid:{display:false}},
            y:{title:{display:true,text:'count'},grid:{color:'#eee'}}},
    plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.raw} of ${D.judge.n} (${(100*c.raw/D.judge.n).toFixed(1)}%)`}}}}
});

// judge table
const jt=document.getElementById('judgeTable');
D.prof_rows.forEach(r=>{
  jt.insertAdjacentHTML('beforeend',
   `<tr><td>${r.p}</td><td>${r.n}</td><td>${r.uniq}</td><td>${r.sd.toFixed(3)}</td><td>${r.mode.toFixed(1)}</td><td>${(r.modefrac*100).toFixed(0)}%</td></tr>`);
});

// --- generator-end: how much the strategy dims control the message ---
if(D.variety && D.variety.within_across_cos){
  const wa=D.variety.within_across_cos;
  const dims=Object.keys(wa);
  const chance={appeal:1/3,tone:1/4,emphasis:1/5,intensity:1/3};
  new Chart(document.getElementById('genChart'),{
    type:'bar',
    data:{labels:dims,
      datasets:[
        {label:'observed nearest-neighbour purity',data:dims.map(d=>wa[d].nn_purity),
         backgroundColor:'rgba(196,90,40,0.6)',borderWidth:0,borderRadius:2},
        {label:'chance',data:dims.map(d=>+chance[d].toFixed(3)),
         backgroundColor:'rgba(120,120,120,0.35)',borderWidth:0,borderRadius:2}
      ]},
    options:{responsive:true,
      scales:{x:{grid:{display:false}},y:{beginAtZero:true,max:1,
        title:{display:true,text:'1-NN purity (share of nearest neighbours sharing the level)'},grid:{color:'#eee'}}},
      plugins:{legend:{position:'top',align:'end'},
        tooltip:{callbacks:{afterLabel:c=>c.datasetIndex===0?`within−across cos gap: ${wa[c.label].gap}`:''}}}}
  });
  const gt=document.getElementById('genTable');
  D.gen_table.forEach(r=>{
    gt.insertAdjacentHTML('beforeend',
      `<tr><td class="left">${r.label}</td><td class="mono">${r.val}</td><td class="left">${r.note}</td></tr>`);
  });
}

// --- conditioning A/B table ---
if(D.cond_table && D.cond_table.length){
  const ct=document.getElementById('condTable');
  const fmt=(x,d=3)=>x==null?'–':Number(x).toFixed(d);
  D.cond_table.forEach(r=>{
    const col = r.key==='base'?'color:var(--ink-light)':'';
    ct.insertAdjacentHTML('beforeend',
      `<tr><td class="left" style="${col}">${r.variant}</td>`+
      `<td>${fmt(r.paircos)}</td><td>${fmt(r.emph_nn,2)}</td>`+
      `<td>${fmt(r.judge_sd)}</td><td>${fmt(r.modal,2)}</td>`+
      `<td>${fmt(r.openings,0)}</td>`+
      `<td>${fmt(r.reward_mean,2)}</td><td>${fmt(r.reward_sd)}</td>`+
      `<td class="left">${r.verdict}</td></tr>`);
  });
  const by=k=>D.cond_table.find(r=>r.key===k)||{};
  const b=by('base'), ex=by('ex'), bj=by('bon_judge'), bf=by('bon_faithful');
  const pct=x=>x==null?'–':Math.round(x*100)+'%';
  const parts=[];
  if(ex.reward_mean!=null)
    parts.push(`<strong>Exemplar scaffold</strong> raises conditioning (emphasis NN ${b.emph_nn} → ${ex.emph_nn}, `+
      `openings ${b.openings} → ${ex.openings}) but reward ${b.reward_mean} → ${ex.reward_mean} — above base on `+
      `${pct(ex.above_base)} of strategies.`);
  if(bj.reward_mean!=null)
    parts.push(`<strong>Best-of-8 by judge</strong> recovers some of that (reward ${bj.reward_mean}, above base on `+
      `${pct(bj.above_base)}), yet selecting on the judge ${bj.judge_sd<b.judge_sd?'narrows':'widens'} the score spread `+
      `(SD ${b.judge_sd} → ${bj.judge_sd}).`);
  if(bf.reward_mean!=null)
    parts.push(`<strong>Best-of-8 by faithfulness</strong> gives the most diverse posts and the widest `+
      `<em>message-level</em> judge spread (SD ${bf.judge_sd}) but the <em>narrowest network reward spread</em> `+
      `(${bf.reward_sd}) — per-message score spread is not reward spread.`);
  parts.push(`<strong>No generator variant beats the plain prompt</strong> — the judge rewards natural, `+
    `persona-tailored text, so forcing rhetoric trades persuasiveness for variety.`);
  const v=document.getElementById('condVerdict');
  if(v && parts.length) v.innerHTML=parts.join(' ');
}

// --- appendix: strategy v2 table ---
if(D.v2_rows && D.v2_rows.length){
  const vt=document.getElementById('v2Table');
  const mark = k => k==='pass'
    ? '<span class="pos">pass</span>' : '<span class="neg">miss</span>';
  D.v2_rows.forEach(r=>{
    vt.insertAdjacentHTML('beforeend',
      `<tr><td class="left"><strong>${r.test}</strong></td>`+
      `<td class="left" style="color:var(--ink-light)">${r.q}</td>`+
      `<td class="left">${r.grid}</td>`+
      `<td class="left">${r.pilot}</td>`+
      `<td class="left" style="color:var(--ink-light)">${r.target}</td>`+
      `<td>${mark(r.verdict)}</td></tr>`);
  });
}

// --- appendix: Bradley-Terry readout ---
if(D.bt){
  const b=D.bt, L=b.land;
  const bt=document.getElementById('btTable');
  const fmt=(x,d=3)=>x==null?'–':Number(x).toFixed(d);
  if(bt){
    const absRef={sd:0.072,uniq:'4–6',modal:0.90,share:0.54,spread:'~0.90',ratio:'≈1.0'};
    bt.insertAdjacentHTML('beforeend',
      `<tr><td class="left">absolute 0–10 judge</td><td>${fmt(absRef.sd)}</td><td>${absRef.uniq}</td>`+
      `<td>${absRef.modal.toFixed(2)}</td><td>${(100*absRef.share).toFixed(0)}%</td>`+
      `<td>${absRef.spread}</td><td>${absRef.ratio}</td><td class="left"><span class="neg">saturated</span></td></tr>`);
    const ratio = (L&&L.se)?(L.p10_p90/L.se):null;
    bt.insertAdjacentHTML('beforeend',
      `<tr class="highlight-row"><td class="left"><strong>Bradley–Terry pairwise</strong></td>`+
      `<td>${fmt(b.sd)}</td><td>${b.n_unique}</td><td>${fmt(b.modal_bin,2)}</td>`+
      `<td>${(100*b.modal_share).toFixed(0)}%</td>`+
      `<td>${L?fmt(L.p10_p90,2):'–'}</td><td>${ratio?ratio.toFixed(0)+'×':'–'}</td>`+
      `<td class="left"><span class="pos">graded</span></td></tr>`);
  }
  if(document.getElementById('btAside')){
    const d=b.decomp;
    document.getElementById('btAside').innerHTML =
      `<strong>Resolution.</strong> Score SD rises ${(b.sd/0.072).toFixed(1)}× (0.072 → ${b.sd.toFixed(3)}), `+
      `the modal bucket falls to ${(100*b.modal_share).toFixed(0)}% and ${b.n_unique} distinct scores replace 4–6. `+
      `<strong>And it reaches the reward.</strong> The network landscape now spans ${L?L.range.toFixed(1):'–'} reward points `+
      `(best constant ${L?L.best.toFixed(1):'–'}: ${L?L.best_spec:''}) against a per-episode SE of ${L?L.se.toFixed(2):'–'} — a `+
      `${ratio?ratio.toFixed(0):'–'}× signal-to-noise ratio where the absolute judge had ≈1×.`;
  }
  if(document.getElementById('btVerdict')){
    const d=b.decomp;
    document.getElementById('btVerdict').innerHTML =
      `The variance split confirms it: <strong>${d.strategy}% of BT-score variance is the action's main effect</strong>, `+
      `<strong>${d.interaction}% is target × action interaction</strong> (the target has no main effect — comparisons are `+
      `within-target by construction). The catch is scope: the per-target oracle only beats the best constant by `+
      `<span class="mono">+${b.oracle_gain.toFixed(3)}</span> in score, because the best constant already sits near the 1.0 `+
      `ceiling. So the readout now ranks <em>actions</em> sharply but still barely rewards <em>matching the target</em> — `+
      `which is exactly what the staged v2 + BT training run tests.`;
  }
  if(document.getElementById('btChart')){
    const pts=D.bt_points;
    const addrCol={taxpayer:'rgba(196,90,40,0.75)',parent:'rgba(90,90,170,0.75)',worker:'rgba(42,122,90,0.85)'};
    const byAddr=Object.keys(addrCol).map(a=>({label:'address: '+a,
      data:pts.filter(p=>p.address===a).map(p=>({x:p.s,y:p.mean})),
      backgroundColor:addrCol[a],pointRadius:4.5,pointHoverRadius:6,borderWidth:0}));
    const best=L?L.best:null;
    const bestLine=best==null?[]:[{label:'best constant',type:'line',
      data:[{x:0,y:best},{x:71,y:best}],borderColor:'rgba(0,0,0,0.35)',
      borderDash:[6,4],borderWidth:1,pointRadius:0,fill:false}];
    new Chart(document.getElementById('btChart'),{
      type:'scatter',
      data:{datasets:[...byAddr,...bestLine]},
      options:{responsive:true,interaction:{mode:'nearest',intersect:false},
        scales:{x:{title:{display:true,text:'strategy index (0–71)'},grid:{color:'#eee'},min:-1,max:72},
                y:{title:{display:true,text:'mean episode reward'},grid:{color:'#eee'}}},
        plugins:{legend:{position:'top',align:'end'},
          tooltip:{callbacks:{label:c=>{const p=pts[c.dataIndex];return p&&c.datasetIndex<3?`${p.spec} → ${p.mean.toFixed(2)}`:`best constant ${Number(c.raw.y).toFixed(2)}`;}}}}}
    });
  }
}

// --- messages table ---
const mt=document.getElementById('msgTable');
if(mt){ D.messages.forEach(m=>{
  const col = m.idx===90?'color:var(--status-green);font-weight:600'
            :(m.trained?'color:var(--accent);font-weight:600':'');
  const rank = m.rank?` <span style="color:var(--ink-light)">#${m.rank}</span>`:'';
  mt.insertAdjacentHTML('beforeend',
    `<tr><td class="left" style="${col}">${m.role}<br><span style="color:var(--ink-light);font-weight:400;font-size:0.8rem">${m.note||''}</span></td>`+
    `<td>${m.idx}${rank}</td>`+
    `<td class="left">${m.appeal} / ${m.tone} / ${m.emphasis} / ${m.intensity}</td>`+
    `<td>${m.score==null?'–':m.score.toFixed(1)}</td>`+
    `<td class="left hide-mobile" style="font-family:'EB Garamond',serif;font-size:0.9rem;color:#444;max-width:430px;white-space:normal;text-align:left;">${m.message||'—'}</td></tr>`);
});}

// --- strategy landscape ---
if(D.land && D.land.mean){
  const L=D.land, strat=L.strategies;
  const pts=strat.map((s,i)=>({x:s,y:L.mean[i]}));
  const heur=strat.indexOf(90), tr=strat.indexOf(168);
  const heurPt=heur>=0?[{x:90,y:L.mean[heur]}]:[];
  const trPt=tr>=0?[{x:168,y:L.mean[tr]}]:[];
  const modeIdx=[141,153,13].filter(i=>strat.includes(i));
  const modePt=modeIdx.map(i=>({x:i,y:L.mean[strat.indexOf(i)]}));
  if(document.getElementById('landCorr'))
    document.getElementById('landCorr').textContent = (L.corr_score_reward!=null?L.corr_score_reward.toFixed(2):'–');
  if(document.getElementById('landSpread') && L.stats)
    document.getElementById('landSpread').textContent = L.stats.spread_all.toFixed(2)+' reward points';
  new Chart(document.getElementById('landChart'),{
    type:'scatter',
    data:{datasets:[
      {label:'strategy',data:pts,backgroundColor:'rgba(196,90,40,0.45)',pointRadius:2.6,pointHoverRadius:5,borderWidth:0},
      {label:'learned modes (141·153·13)',data:modePt,backgroundColor:'rgba(160,42,42,1)',pointRadius:6,pointStyle:'rect'},
      {label:'best arm (168)',data:trPt,backgroundColor:'rgba(120,60,140,1)',pointRadius:6,pointStyle:'triangle'},
      {label:'heuristic (90)',data:heurPt,backgroundColor:'rgba(42,122,90,1)',pointRadius:6,pointStyle:'rectRot'}
    ]},
    options:{responsive:true,interaction:{mode:'nearest',intersect:false},
      scales:{x:{title:{display:true,text:'strategy index (0–179)'},grid:{color:'#eee'}},
              y:{title:{display:true,text:'mean episode reward'},grid:{color:'#eee'}}},
      plugins:{legend:{position:'top',align:'end'}}}
  });
  const lt=document.getElementById('landTable');
  D.land_table.forEach(r=>{
    const col = r.heur?'color:var(--status-green);font-weight:600':(r.trained?'color:var(--accent);font-weight:600':'');
    lt.insertAdjacentHTML('beforeend',
      `<tr class="${r.heur?'highlight-row':''}"><td class="left" style="${col}">${r.role||''}</td>`+
      `<td>${r.idx}</td><td>${r.rank==null?'–':'#'+r.rank}</td><td class="left">${r.spec}</td>`+
      `<td>${r.mean.toFixed(3)}</td><td>${r.se==null?'–':r.se.toFixed(3)}</td>`+
      `<td>${r.score==null?'–':r.score.toFixed(3)}</td>`+
      `<td class="left">${r.note||''}</td></tr>`);
  });
  const sr=strat.map((s,i)=>({x:(L.mean_score?L.mean_score[s]:null),y:L.mean[i]})).filter(p=>p.x!=null);
  new Chart(document.getElementById('scoreRewChart'),{
    type:'scatter',
    data:{datasets:[
      {label:'strategy',data:sr,backgroundColor:'rgba(90,90,170,0.5)',pointRadius:2.6,pointHoverRadius:5,borderWidth:0},
      {label:'learned modes (141·153·13)',data:(L.mean_score)?modePt.map(p=>({x:L.mean_score[p.x],y:p.y})):[],backgroundColor:'rgba(160,42,42,1)',pointRadius:6,pointStyle:'rect'},
      {label:'best arm (168)',data:(L.mean_score&&tr>=0)?[{x:L.mean_score[168],y:L.mean[tr]}]:[],backgroundColor:'rgba(120,60,140,1)',pointRadius:6,pointStyle:'triangle'},
      {label:'heuristic (90)',data:(L.mean_score&&heur>=0)?[{x:L.mean_score[90],y:L.mean[heur]}]:[],backgroundColor:'rgba(42,122,90,1)',pointRadius:6,pointStyle:'rectRot'}
    ]},
    options:{responsive:true,interaction:{mode:'nearest',intersect:false},
      scales:{x:{title:{display:true,text:'mean judge persuasiveness'},grid:{color:'#eee'}},
              y:{title:{display:true,text:'mean episode reward'},grid:{color:'#eee'}}},
      plugins:{legend:{position:'top',align:'end'}}}
  });
}

// --- campaign 3: BT readout verdict ---
if(D.c3){
  const c=D.c3, st=document.getElementById('c3Table');
  const pf=c.paired_final;
  const rows=[
    {n:'learned (greedy) vs random', v:pf.greedy, d:pf.d, t:pf.t, verdict:'win'},
    {n:'best single constant', v:c.best_const, d:c.best_const-pf.random, t:null, verdict:'reference'},
    {n:'learned (greedy) vs best constant', v:pf.greedy, d:pf.greedy-c.best_const, t:null, verdict:'narrow win'},
    {n:'"heuristic" baseline (arm #36)', v:c.heuristic, d:c.heuristic-pf.random, t:null, verdict:'bottom arm'},
  ];
  rows.forEach(r=>{
    const cls=r.d>0?'pos':(r.d<0?'neg':'');
    st.insertAdjacentHTML('beforeend',
      `<tr><td class="left">${r.n}</td><td>${r.v.toFixed(3)}</td>`+
      `<td class="${cls}">${r.d>0?'+':''}${r.d.toFixed(3)}</td>`+
      `<td>${r.t==null?'–':'t='+r.t.toFixed(2)}</td><td class="left">${r.verdict}</td></tr>`);
  });
  const v=document.getElementById('c3Verdict');
  if(v) v.innerHTML=`Under the Bradley–Terry readout the trained policy scores <span class="mono">${pf.greedy.toFixed(3)}</span> against random's <span class="mono">${pf.random.toFixed(3)}</span> (<span class="pos mono">+${pf.d.toFixed(3)}</span>, <span class="mono">t=${pf.t.toFixed(2)}</span>, 50 paired seeds) and edges the best single constant (<span class="mono">${c.best_const.toFixed(3)}</span>, <span class="mono">${c.best_spec}</span>) by <span class="mono">+${(pf.greedy-c.best_const).toFixed(3)}</span> — firm, but modest.`;
  new Chart(document.getElementById('c3Chart'),{
    type:'line',
    data:{labels:c.paired_iters,datasets:[
      {label:'learned (greedy)',data:c.paired_greedy,borderColor:'#2a7a5a',backgroundColor:'rgba(42,122,90,0.10)',fill:true,tension:0.25,pointRadius:2.6},
      {label:'random baseline',data:c.paired_iters.map(()=>pf.random),borderColor:'#a02a2a',borderDash:[5,4],pointRadius:0,borderWidth:1.4}
    ]},
    options:{responsive:true,scales:{x:{title:{display:true,text:'PPO iteration'},grid:{color:'#eee'}},
            y:{title:{display:true,text:'paired reward (50 seeds)'},grid:{color:'#eee'}}},
            plugins:{legend:{position:'top',align:'end'}}}
  });
}

// --- campaign 4: constant calibration ---
if(D.cal){
  const c=D.cal, st=document.getElementById('calAlphaTable');
  const rr=document.getElementById('calRank');
  if(rr) rr.textContent=c.eff_rank.toFixed(2);
  const rows=[
    {n:'linear recovery α (no intercept)', v:'['+c.lin.join(', ')+']', r:'exact — instrumentation check, mean|resid| '+c.lin_resid.toExponential(1)},
    {n:'logistic fit α, α≥0', v:'['+c.log.join(', ')+']', r:'collinear design ⇒ unstable; b='+c.log_b},
    {n:'effective rank / condition', v:c.eff_rank.toFixed(2)+' / 4, cond '+c.cond.toFixed(1), r:'α NOT separately identifiable'},
    {n:'per-feature std', v:'['+c.std.join(', ')+']', r:'profile & degree near-constant'},
    {n:'kept in config', v:'α = 0.25 equal, β = 0.4', r:'unidentifiable prior / non-binding default'},
  ];
  rows.forEach(r=>st.insertAdjacentHTML('beforeend',
    `<tr><td class="left">${r.n}</td><td class="mono">${r.v}</td><td class="left">${r.r}</td></tr>`));
  const pts=(ys)=>c.rho.map((x,i)=>({x:x,y:ys[i]}));
  new Chart(document.getElementById('calBetaChart'),{
    type:'scatter',
    data:{datasets:[
      {label:'early ΔS',data:pts(c.early),backgroundColor:'#2a7a5a',pointRadius:3.2},
      {label:'late ΔS',data:pts(c.late),backgroundColor:'#c45a28',pointRadius:3.2},
      {label:'plateau S',data:pts(c.plateau),backgroundColor:'#5a5aaa',pointRadius:3.2},
    ]},
    options:{responsive:true,
      scales:{x:{type:'linear',title:{display:true,text:'ρ = β·κ (effective influence strength)'},grid:{color:'#eee'}},
              y:{title:{display:true,text:'stance (ΔS or S)'},grid:{color:'#eee'}}},
      plugins:{legend:{position:'top',align:'end'}}}
  });
}

// --- campaign 5: goal direction ---
if(D.goal){
  const g=D.goal, st=document.getElementById('goalTable');
  const rows=[
    {n:'directional sanity ('+g.backend+')', v:'anti '+g.a_anti.toFixed(3)+' > neutral '+g.a_neutral.toFixed(3)+' > pro '+g.a_pro.toFixed(3), r:g.pass?'✓ anti-policy ranks highest':'✗ failed'},
    {n:'corr( a(m), realized −Δs )', v:(g.corr>0?'+':'')+g.corr.toFixed(3), r:'n = '+g.n+' scored decisions'},
    {n:'mean −Δs, on-goal (a≥0.5)', v:(g.on>=0?'+':'')+g.on.toFixed(4), r:'≈ '+(g.on/g.off).toFixed(1)+'× the off-goal reward'},
    {n:'mean −Δs, off-goal (a<0.5)', v:(g.off>=0?'+':'')+g.off.toFixed(4), r:'old off-goal advantage attenuated'},
  ];
  rows.forEach(r=>st.insertAdjacentHTML('beforeend',
    `<tr><td class="left">${r.n}</td><td class="mono">${r.v}</td><td class="left">${r.r}</td></tr>`));
}

// --- campaign 6: a(m) backend comparison ---
if(D.goal_backends){
  const g=D.goal_backends, st=document.getElementById('gbTable');
  const M=g.means, S=g.sanity;
  st.insertAdjacentHTML('beforeend',
    `<tr><td class="left">mean a(m) on ${g.n} real strategy messages</td>`+
    `<td class="mono">${M.semdiff.toFixed(3)}</td>`+
    `<td class="mono">${M.semdiff_seed.toFixed(3)}</td>`+
    `<td class="mono">${M.judge.toFixed(3)}</td>`+
    `<td class="left">0.5 = neutral; only the judge reads the on-goal corpus as on-goal</td></tr>`);
  st.insertAdjacentHTML('beforeend',
    `<tr><td class="left">directional sanity (short anti / pro)</td>`+
    `<td class="mono">${S.semdiff.anti} / ${S.semdiff.pro}</td>`+
    `<td class="mono">${S.semdiff_seed.anti} / ${S.semdiff_seed.pro}</td>`+
    `<td class="mono">${S.judge.anti} / ${S.judge.pro}</td>`+
    `<td class="left">all pass on hand-built posts — embeddings still fail on real ones</td></tr>`);
}

// --- campaign 6: goal-direction A/B ---
if(D.ab){
  const a=D.ab, st=document.getElementById('abTable');
  const byArm={};
  Object.entries(a.matched).forEach(([k,rows])=>{
    byArm[k]={}; rows.forEach(m=>{byArm[k][m.iter]=m;});
  });
  const iters=new Set();
  Object.values(byArm).forEach(o=>Object.keys(o).forEach(k=>iters.add(+k)));
  const sorted=[...iters].sort((x,y)=>x-y);
  if(sorted.length===0){
    st.insertAdjacentHTML('beforeend',
      `<tr><td class="left">run in progress</td><td colspan="4">no matched checkpoints yet (first paired checkpoint at iter 20).</td></tr>`);
  }
  const cell=m=> m
    ? `${m.g_greedy.toFixed(3)} <span class="mono" style="color:var(--ink-muted)">(d+${m.g_d.toFixed(2)})</span>`
    : '—';
  sorted.forEach(it=>{
    const s=(byArm['goal_semdiff']||{})[it];
    const j=(byArm['goal_judge']||{})[it];
    const b=(s||j).b_greedy;
    const reading=(j && s)
      ? (j.g_d>0 ? 'judge arm learns; semdiff collapses' : '—')
      : (j ? 'judge arm learns' : 'semdiff arm collapses');
    st.insertAdjacentHTML('beforeend',
      `<tr><td class="left">@ ${it}</td><td class="mono">${b.toFixed(3)}</td>`+
      `<td class="mono">${cell(s)}</td><td class="mono">${cell(j)}</td>`+
      `<td class="left">${reading}</td></tr>`);
  });
  const B=a.arms;
  const ab=document.getElementById('abBackend');
  if(ab){
    const gs=B.goal_semdiff, gj=B.goal_judge;
    ab.innerHTML=`On the real messages the embedding backends read <span class="mono">a(m)≈0.51</span> (neutral) `+
      `while the judge reads <span class="mono">≈0.98</span>. With <span class="mono">s_eff = s_src·(2a−1)</span>, `+
      `a≈0.5 zeroes the pull: the semdiff arm stalls (tail stance <span class="mono">${gs.final_stance_tail}</span>, `+
      `adopted <span class="mono">${gs.adopted_tail}</span>, reward <span class="mono">${gs.reward_tail}</span>). `+
      `The judge arm keeps <span class="mono">a(m)≈${gj.goal_align_tail==null?'—':gj.goal_align_tail}</span> `+
      `(tail reward <span class="mono">${gj.reward_tail==null?'—':gj.reward_tail}</span>).`;
  }
  const gm=document.getElementById('abMetrics');
  if(gm){
    const gs=B.goal_semdiff, gj=B.goal_judge, bb=B.baseline;
    gm.innerHTML=`tail mean — baseline: stance <span class="mono">${bb.final_stance_tail}</span>, `+
      `adopted <span class="mono">${bb.adopted_tail}</span>; `+
      `semdiff: stance <span class="mono">${gs.final_stance_tail}</span>, adopted <span class="mono">${gs.adopted_tail}</span>; `+
      `judge: stance <span class="mono">${gj.final_stance_tail==null?'—':gj.final_stance_tail}</span>, `+
      `adopted <span class="mono">${gj.adopted_tail==null?'—':gj.adopted_tail}</span>.`;
  }
}

const mt=document.getElementById('mechTable');
if(mt && D.mech && D.mech.arms){
  const names={cascade:'C7 cascade',exposure:'C8 exposure'};
  const mechNames={cascade:'stance-ful — period-0 collapse',exposure:'stance-less — multi-period queue'};
  ['cascade','exposure'].forEach(k=>{
    const a=D.mech.arms[k]; if(!a) return;
    const jf=a.eval_judge&&a.eval_judge.length?a.eval_judge[0]:null;
    const jl=a.eval_judge&&a.eval_judge.length?a.eval_judge[a.eval_judge.length-1]:null;
    const gl=a.eval_greedy&&a.eval_greedy.length?a.eval_greedy[a.eval_greedy.length-1]:'—';
    mt.insertAdjacentHTML('beforeend',
      `<tr><td class="left">${names[k]}</td><td class="left">${mechNames[k]}</td>`+
      `<td class="mono">${a.reward_first} → ${a.reward_last}</td>`+
      `<td class="mono">${(+a.adopted_min).toFixed(1)}–${(+a.adopted_max).toFixed(1)}</td>`+
      `<td class="mono">${gl}</td>`+
      `<td class="mono">${a.t_random_mean>=0?'+':''}${a.t_random_mean} (mean)</td>`+
      `<td class="mono">${a.t_heuristic_mean>=0?'+':''}${a.t_heuristic_mean} (min ${a.t_heuristic_min})</td>`+
      `<td class="mono">${jf} → ${jl}</td></tr>`);
  });
}
const mc=document.getElementById('mechChart');
if(mc && D.mech && D.mech.arms){
  const colors={cascade:'#c45a28',exposure:'#2a7a5a'};
  const ds=[];
  ['cascade','exposure'].forEach(k=>{
    const a=D.mech.arms[k]; if(!a) return;
    ds.push({label:k,data:a.exposures_by_period,backgroundColor:colors[k],borderRadius:2});
  });
  new Chart(mc,{type:'bar',
    data:{labels:['0','1','2','3','4','5','6','7','8','9'],datasets:ds},
    options:{responsive:true,
      plugins:{legend:{labels:{font:{size:11}}},
        tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.parsed.y.toLocaleString()} exposures`}}},
      scales:{y:{title:{display:true,text:'exposures'},ticks:{font:{size:10}}},
              x:{title:{display:true,text:'period'},ticks:{font:{size:10}}}}}});
  const cas=D.mech.arms.cascade, exp=D.mech.arms.exposure;
  if(cas && exp){
    const pct=(100*cas.exposures_by_period[0]/cas.exposures_total).toFixed(1);
    const cap=document.getElementById('mechCaption');
    if(cap) cap.textContent=
      `Exposures by period over the full 300-iteration traces. C7 cascade issues `+
      `${cas.exposures_by_period[0].toLocaleString()} of ${cas.exposures_total.toLocaleString()} (${pct}%) in period 0; `+
      `C8 exposure spreads ${exp.exposures_total.toLocaleString()} almost evenly across all ten.`;
  }
}
const vz=document.getElementById('vizTable');
if(vz && D.viz){
  D.viz.runs.forEach(r=>{
    const st=r.stance?`${r.stance}`:'—';
    vz.insertAdjacentHTML('beforeend',
      `<tr><td class="left mono">${r.run}</td><td class="left">${r.label}</td>`+
      `<td class="mono">${r.adopted??'—'}</td><td class="mono">${st}</td>`+
      `<td class="mono">${r.reward??'—'}</td>`+
      `<td class="left mono">${r.follows} / ${r.retweets} / ${r.replies}</td></tr>`);
  });
}

// scroll reveal
document.querySelectorAll('.chart-container, .flyout, .aside-container').forEach(el=>el.classList.add('reveal'));
const obs=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');obs.unobserve(e.target);}});},{threshold:0.15});
document.querySelectorAll('.reveal').forEach(el=>obs.observe(el));
window.addEventListener('load',()=>document.querySelectorAll('.card-spark').forEach(s=>{}));
</script>
</body>
</html>
"""

html = HTML.replace("__DATA__", json.dumps(pay))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"
if not OUT.is_absolute():
    OUT = HERE / OUT
OUT.write_text(html)
print("wrote", OUT, f"({len(html)} bytes)")
