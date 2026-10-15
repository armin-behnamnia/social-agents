#!/usr/bin/env python
"""Pilot for a redesigned, relevance-conditioned strategy space.

Motivation (from cond_fix.py): the current abstract grid
(appeal x tone x emphasis x intensity) barely controls the text, and
forcing rhetorical adherence *lowers* persuasiveness — the judge rewards
natural, persona-tailored posts, not strategy obedience. The reward is
judge(persona-fit) x cos(post, target), so the learnable axis is
TARGET-RELEVANCE, not rhetorical form.

Proposed space  --  action = (angle, evidence, address):
  angle    : which concern to foreground for THIS reader (6)
  evidence : what concrete support to give (4)
  address  : which identity of the reader to speak to (3)

This pilot renders 12 personas x 72 arms and measures three things:
  1. realisability  - NN purity per dimension (does the label shape the post?)
  2. readability    - judge SD / modal share (can the reward see it?)
  3. learnability   - persona x angle interaction and the oracle gain from
                      choosing the angle per target rather than a constant.

Writes report/strategy_v2.json.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from code.config import GLOBAL_LLM_CONFIG as C
from code.LLM_module import call_llm, judge_persuasiveness
from code.profiles import generate_synthetic_profiles
from code.tools import text2embedding

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import message_variety as mv  # noqa: E402

N_PERSONAS = 12
WORKERS = 24
OUT = HERE / "strategy_v2.json"

ANGLES = ["cost", "safety", "fairness", "freedom", "community", "effectiveness"]
EVIDENCE = ["statistic", "expert", "anecdote", "peers"]
ADDRESS = ["taxpayer", "parent", "worker"]

_ANGLE = {
    "cost": "the direct financial cost to the reader ('this will cost you ...')",
    "safety": "the risk to the reader's safety or wellbeing",
    "fairness": "who is treated unequally or left out",
    "freedom": "the choice or autonomy being taken away",
    "community": "what the reader's community stands to lose together",
    "effectiveness": "whether the policy actually achieves its stated goal",
}
_EVID = {
    "statistic": "a specific number or percentage",
    "expert": "a named expert, organization or study",
    "anecdote": "a first-person anecdote with one concrete detail",
    "peers": "what people like the reader are doing ('my neighbours ...')",
}
_ADDR = {
    "taxpayer": "a taxpayer who funds this",
    "parent": "a parent or caregiver",
    "worker": "a worker whose job is affected",
}

PROMPT = """You are writing ONE persuasive social-media post (max 50 words, first person) that takes an opposing stance on {topic} and is maximally relevant to the specific reader below.

Lead with this concern (the reader's angle): {angle}
Support it with: {evidence}
Speak to the reader as: {address}

Reader profile: {profile}

Rules:
- The FIRST sentence must foreground the angle above.
- Include exactly the named support.
- Do not mention the strategy.
Return only the post."""


def render_v2(angle, evidence, address, profile, topic="the policy"):
    return call_llm(PROMPT.format(
        topic=topic, angle=_ANGLE[angle], evidence=_EVID[evidence],
        address=_ADDR[address], profile=profile),
        temperature=0.0, max_tokens=C.generation_max_tokens)


def _eta2(vals, labels):
    """Between-group SS / total SS for one factor."""
    vals = np.asarray(vals, float); labels = np.asarray(labels)
    gm = vals.mean()
    ss_b = sum(np.sum((vals[labels == l] - gm) ** 2).item() *
               (labels == l).sum() ** 0 for l in set(labels.tolist()))
    # proper: n_l * (mean_l - gm)^2
    ss_b = 0.0
    for l in set(labels.tolist()):
        v = vals[labels == l]
        ss_b += len(v) * (v.mean() - gm) ** 2
    ss_t = np.sum((vals - gm) ** 2)
    return float(ss_b / max(1e-12, ss_t))


def _two_way_interaction_ss(vals, fa, fb):
    """SS for the fa x fb interaction (sum of (cell - a - b + g)^2)."""
    vals = np.asarray(vals, float)
    g = vals.mean()
    cells = {}
    for a, b, v in zip(fa, fb, vals):
        cells.setdefault((a, b), []).append(v)
    am = {a: np.mean([v for x, v in zip(fa, vals) if x == a]) for a in set(fa)}
    bm = {b: np.mean([v for x, v in zip(fb, vals) if x == b]) for b in set(fb)}
    ss = 0.0
    for (a, b), vs in cells.items():
        ss += len(vs) * (np.mean(vs) - am[a] - bm[b] + g) ** 2
    return float(ss)


def main():
    profiles = generate_synthetic_profiles(N_PERSONAS, seed=1234)
    arms = [(a, e, d) for a in ANGLES for e in EVIDENCE for d in ADDRESS]
    print(f"[*] {len(profiles)} personas x {len(arms)} arms = "
          f"{len(profiles)*len(arms)} renders ...", flush=True)

    jobs = [(pi, ai) for pi in range(len(profiles)) for ai in range(len(arms))]

    def job(t):
        pi, ai = t
        a, e, d = arms[ai]
        return pi, ai, render_v2(a, e, d, profiles[pi])

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rendered = list(pool.map(job, jobs))

    msg = {}
    for pi, ai, m in rendered:
        msg[(pi, ai)] = m

    print("[*] judging ...", flush=True)

    def jj(t):
        pi, ai = t
        return pi, ai, judge_persuasiveness(msg[(pi, ai)], profiles[pi])

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        J = np.zeros((len(profiles), len(arms)))
        for pi, ai, s in pool.map(jj, jobs):
            J[pi, ai] = s

    # ---- 1. realisability: NN purity per dimension ----
    def level(ai, pos):
        return arms[ai][pos]

    Emap = {pi: mv._unit(text2embedding([msg[(pi, ai)] for ai in range(len(arms))]))
            for pi in range(len(profiles))}
    purity = {}
    for pos, name in enumerate(("angle", "evidence", "address")):
        lv = np.array([level(ai, pos) for ai in range(len(arms))])
        ps = []
        for pi in range(len(profiles)):
            E = Emap[pi]
            S = E @ E.T; np.fill_diagonal(S, np.nan)
            nn = np.nanargmax(S, axis=1)
            ps.append(float((lv[nn] == lv).mean()))
        purity[name] = {"mean_nn_purity": round(float(np.mean(ps)), 3),
                        "chance": round(1.0 / len(set(lv.tolist())), 3)}

    # ---- content-relevance channel: cos(post, target) + reward proxy ----
    P = mv._unit(text2embedding(profiles))
    C = np.array([Emap[pi] @ P[pi] for pi in range(len(profiles))])
    R = J * (0.25 * C + 0.25)          # message-dependent part of the reward

    # ---- 2. readability ----
    flat = J.ravel()
    modal_share = float(np.bincount(np.round(np.clip(flat, 0, 1) * 10).astype(int)).max() / flat.size)
    judge = {"mean": round(float(flat.mean()), 3), "sd": round(float(flat.std(ddof=1)), 3),
             "n_unique": int(len(set(np.round(flat, 3)))), "modal_share": round(modal_share, 3)}

    # ---- 3. learnability: variance components + oracle gain ----
    fa = np.array([arms[ai][0] for ai in range(len(arms)) for _ in range(len(profiles))])
    fe = np.array([arms[ai][1] for ai in range(len(arms)) for _ in range(len(profiles))])
    fd = np.array([arms[ai][2] for ai in range(len(arms)) for _ in range(len(profiles))])
    fp = np.array([pi for ai in range(len(arms)) for pi in range(len(profiles))])
    # note: build in (persona-major) order to match J.ravel()
    fp = np.array([pi for pi in range(len(profiles)) for ai in range(len(arms))])
    fa = np.array([arms[ai][0] for pi in range(len(profiles)) for ai in range(len(arms))])
    fe = np.array([arms[ai][1] for pi in range(len(profiles)) for ai in range(len(arms))])
    fd = np.array([arms[ai][2] for pi in range(len(profiles)) for ai in range(len(arms))])
    vals = J.ravel()
    var = {
        "judge_persona_eta2": round(_eta2(vals, fp), 3),
        "judge_angle_eta2": round(_eta2(vals, fa), 3),
        "judge_evidence_eta2": round(_eta2(vals, fe), 3),
        "judge_address_eta2": round(_eta2(vals, fd), 3),
        "judge_persona_x_angle_frac":
            round(_two_way_interaction_ss(vals, fp, fa) / max(1e-9, float(np.sum((vals - vals.mean()) ** 2))), 3),
    }
    rvals = R.ravel()
    var["reward_persona_eta2"] = round(_eta2(rvals, fp), 3)
    var["reward_angle_eta2"] = round(_eta2(rvals, fa), 3)
    var["reward_evidence_eta2"] = round(_eta2(rvals, fe), 3)
    var["reward_address_eta2"] = round(_eta2(rvals, fd), 3)
    var["reward_persona_x_angle_frac"] = round(
        _two_way_interaction_ss(rvals, fp, fa) / max(1e-9, float(np.sum((rvals - rvals.mean()) ** 2))), 3)

    # oracle gain: per-persona best angle vs best global angle
    # marginalise evidence/address by averaging
    ang_idx = np.array([ANGLES.index(arms[ai][0]) for ai in range(len(arms))])

    def _learn(Mx):
        M = np.zeros((len(profiles), len(ANGLES)))
        for pi in range(len(profiles)):
            for k in range(len(ANGLES)):
                M[pi, k] = Mx[pi, ang_idx == k].mean()
        best_const = M.mean(axis=0).max()
        best_const_k = int(M.mean(axis=0).argmax())
        ba = [ANGLES[int(k)] for k in M.argmax(axis=1)]
        return {
            "best_global_angle": ANGLES[best_const_k],
            "oracle_gain": round(float(M.max(axis=1).mean() - best_const), 3),
            "angle_spread": round(float(M.mean(axis=0).max() - M.mean(axis=0).min()), 3),
            "n_distinct_best_angle": len(set(ba)),
            "best_angle_per_persona": ba,
        }

    learn = _learn(J)
    learn_content = _learn(C)
    learn_reward = _learn(R)

    # base reference (from cond_fix.json / message_variety.json if present)
    base = {}
    for f, key in ((HERE / "cond_fix.json", "metrics"),):
        if f.exists():
            d = json.loads(f.read_text())[key]["base"]
            g = lambda fn: float(np.mean([fn(x) for x in d.values()]))  # noqa: E731
            base = {"judge_sd": round(g(lambda x: x["judge"]["sd"]), 3),
                    "judge_modal": round(g(lambda x: x["judge"]["modal_share"]), 3),
                    "emphasis_nn": round(g(lambda x: x["levels"]["emphasis"]["nn_purity"]), 3),
                    "openings": round(g(lambda x: x["n_unique_openings"]), 1)}

    out = {"n_personas": N_PERSONAS, "arms": [list(a) for a in arms],
           "realisability": purity, "readability": judge, "learnability": learn,
           "learnability_content": learn_content, "learnability_reward": learn_reward,
           "content_mean": round(float(C.mean()), 3),
           "variance_components": var, "base_reference": base}
    OUT.write_text(json.dumps(out, indent=1))

    print("\n=== strategy v2 pilot ===")
    print("realisability (NN purity vs chance):")
    for k, v in purity.items():
        print(f"   {k:<9} {v['mean_nn_purity']:.2f}  (chance {v['chance']:.2f})")
    print(f"readability: judge SD {judge['sd']}, modal {judge['modal_share']}, "
          f"distinct {judge['n_unique']}")
    print("variance components:", var)
    print("learnability (judge)   :", learn)
    print("learnability (content) :", learn_content)
    print("learnability (reward~) :", learn_reward)
    print("base reference:", base)
    print("[+] wrote", OUT)


if __name__ == "__main__":
    main()
