#!/usr/bin/env python
"""Prototype + A/B for the strategy-conditioning fix.

Two interventions on the frozen message generator:

  EX    - exemplar/scaffold prompt: per-dimension guidance with worked
          examples, so the four strategy axes visibly shape the post.
  BoN   - best-of-n: sample n candidates at higher temperature and pick
          one. Two selectors are compared:
            * judge    : maximise the env's own persuasiveness judge
            * faithful : maximise a strategy-adherence judge

Stage 1 (`gen`) compares message-level variety of base / ex / bon_judge /
bon_faithful on 3 personas. Stage 2 (`reward`) A/Bs the *reward spread*
through the network by patching the env's renderer + judge for one variant
over 60 strategies x 50 profiles, against the measured base landscape.

Usage:
  PYTHONPATH=. python report/cond_fix.py gen
  PYTHONPATH=. python report/cond_fix.py reward ex
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from code.config import GLOBAL_LLM_CONFIG as C
from code.LLM_module import call_llm, render_strategy_message, judge_persuasiveness
from code.strategy import all_strategies, N_STRATEGIES
from code.profiles import generate_synthetic_profiles
from code.tools import text2embedding

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import message_variety as mv  # noqa: E402

PERSONAS = [0, 1, 2]
N_CAND = 8
BON_TEMP = 0.9
WORKERS = 24
GEN = HERE / "cond_fix_messages.json"
OUT = HERE / "cond_fix.json"

APPEALS = ["logos", "pathos", "ethos"]
TONES = ["neutral", "urgent", "empathetic", "confrontational"]
EMPHASES = ["evidence", "personal_story", "social_proof", "risk", "authority"]
INTENS = ["measured", "assertive", "forceful"]

_APPEAL_HINT = {
    "logos": "lead with a concrete number, fact or logical claim.",
    "pathos": "lead with a human feeling or someone's lived experience.",
    "ethos": "lead with your own credibility, credentials or values.",
}
_TONE_HINT = {
    "neutral": "calm, even cadence.",
    "urgent": "time pressure; short, clipped clauses.",
    "empathetic": "warm and understanding ('I hear you').",
    "confrontational": "a direct challenge ('let's be honest').",
}
_EMPH_HINT = {
    "evidence": "cite a statistic or study.",
    "personal_story": "recount a first-person anecdote.",
    "social_proof": "'everyone around me is...'.",
    "risk": "name the concrete danger of inaction.",
    "authority": "invoke expert consensus.",
}
_INTENS_HINT = {
    "measured": "qualified, hedged claims.",
    "assertive": "confident, declarative claims.",
    "forceful": "blunt, high-conviction claims.",
}

EX_PROMPT = """You are crafting a persuasive social-media message. Make the strategy UNMISTAKABLE: two messages written for different strategies must not read alike.

Persuasion strategy to follow:
- Rhetorical appeal: {appeal} -> {appeal_hint}
- Tone: {tone} -> {tone_hint}
- Emphasis: {emphasis} -> {emphasis_hint}
- Intensity: {label} -> {intens_hint}

Worked examples (same topic, opposite stance - note how the label changes the framing):
- logos / neutral / evidence / measured: "The 2023 pilot data show costs fell 12% where this was tried; the case is modest but real."
- pathos / empathetic / personal_story / assertive: "I watched my neighbour lose her studio last winter. This policy will do the same to people I love - I am sure of it."
- ethos / confrontational / risk / forceful: "As someone who ran these programs, I will say it plainly: wait, and our community pays the price. No more excuses."

Target audience profile: {profile}

Write ONE short social-media post (max 50 words, first person) that takes an {stance} stance on {topic}, follows the strategy above, and whose opening sentence is dictated by the appeal and tone. Do not mention the strategy explicitly. Return only the message.{suffix}"""

ADHERENCE_PROMPT = """You are auditing whether a message obeys an assigned persuasion strategy.

Assigned strategy:
- Rhetorical appeal: {appeal}
- Tone: {tone}
- Emphasis: {emphasis}
- Intensity: {intensity}

Message: {message}

Rate from 0 to 10 how clearly the message follows ALL FOUR strategy dimensions (10 = unmistakably on-strategy, 0 = generic and off-strategy). Return only the number."""


def _ex_prompt(spec, profile, suffix=""):
    return EX_PROMPT.format(
        appeal=spec.appeal, appeal_hint=_APPEAL_HINT[spec.appeal],
        tone=spec.tone, tone_hint=_TONE_HINT[spec.tone],
        emphasis=spec.emphasis, emphasis_hint=_EMPH_HINT[spec.emphasis],
        label=spec.intensity_label, intens_hint=_INTENS_HINT[spec.intensity_label],
        profile=profile, stance="opposing", topic="the policy", suffix=suffix)


def adherence(message, spec):
    raw = call_llm(ADHERENCE_PROMPT.format(
        appeal=spec.appeal, tone=spec.tone, emphasis=spec.emphasis,
        intensity=spec.intensity_label, message=message),
        temperature=0.0, max_tokens=16)
    m = re.search(r"\d+(?:\.\d+)?", raw or "")
    return min(max(float(m.group()) / 10.0, 0.0), 1.0) if m else 0.5


def _variant_message(variant, s, p, specs, profiles):
    if variant == "base":
        return render_strategy_message(specs[s], target_profile=profiles[p],
                                       topic="the policy", stance_direction="opposing")
    if variant == "ex":
        return call_llm(_ex_prompt(specs[s], profiles[p]),
                        temperature=0.0, max_tokens=C.generation_max_tokens)
    if variant in ("bon_judge", "bon_faithful"):
        cands = [call_llm(_ex_prompt(specs[s], profiles[p], suffix=f"\n[variation {k}]"),
                          temperature=BON_TEMP, max_tokens=C.generation_max_tokens)
                 for k in range(N_CAND)]
        if variant == "bon_judge":
            return max(cands, key=lambda x: judge_persuasiveness(x, profiles[p]))
        return max(cands, key=lambda x: adherence(x, specs[s]))
    raise ValueError(variant)


# --------------------------------------------------------------------- stage 1
def _decode_levels(i):
    out, rem = [], i
    for d in reversed(mv.DIMS):
        out.append(rem % d); rem //= d
    a, t, e, inten = reversed(out)
    return APPEALS[a], TONES[t], EMPHASES[e], INTENS[inten]


def _metrics(msgs, judges):
    E = mv._unit(text2embedding(msgs))
    pc = mv._offdiag_cos_stats(E)
    n = len(msgs)
    S = E @ E.T
    np.fill_diagonal(S, np.nan)
    levels = {}
    for pos, name in enumerate(("appeal", "tone", "emphasis", "intensity")):
        lv = np.array([_decode_levels(i)[pos] for i in range(n)])
        same = lv[:, None] == lv[None, :]
        within = float(np.nanmean(np.where(same, S, np.nan)))
        across = float(np.nanmean(np.where(~same, S, np.nan)))
        nn = np.nanargmax(S, axis=1)
        levels[name] = {"within": round(within, 3), "across": round(across, 3),
                        "gap": round(within - across, 3),
                        "nn_purity": round(float((lv[nn] == lv).mean()), 3)}
    openings = [mv._opening(m, 4) for m in msgs]
    modal = float(np.bincount(np.round(np.clip(judges, 0, 1) * 10).astype(int)).max() / n)
    return {
        "pair_cos_mean": round(pc["mean"], 3),
        "pair_cos_frac_gt_0.90": round(pc["frac_gt_0.90"], 3),
        "effective_rank": round(mv._effective_rank(E), 1),
        "pca": {k: round(v, 3) for k, v in mv._pca_explained(E).items()},
        "levels": levels,
        "judge": {"mean": round(float(judges.mean()), 3),
                  "sd": round(float(judges.std(ddof=1)), 3),
                  "n_unique": int(len(set(np.round(judges, 3)))),
                  "min": round(float(judges.min()), 3),
                  "max": round(float(judges.max()), 3),
                  "modal_share": round(modal, 3)},
        "n_unique_openings": len(set(openings)),
    }


def gen():
    specs = all_strategies()
    profiles = generate_synthetic_profiles(50, seed=1234)
    variants = ["base", "ex", "bon_judge", "bon_faithful"]
    pairs = [(s, p) for s in range(N_STRATEGIES) for p in PERSONAS]

    store = {v: {} for v in variants}
    for v in variants:
        def job(sp, v=v):
            s, p = sp
            return s, p, _variant_message(v, s, p, specs, profiles)
        print(f"[*] {v}: {len(pairs)} messages ...", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for s, p, m in pool.map(job, pairs):
                store[v][f"{s}:{p}"] = m

    def judge_all(msgs, prof):
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            return np.array(list(pool.map(lambda m: judge_persuasiveness(m, prof), msgs)))

    metrics = {}
    for v in variants:
        per = {}
        for p in PERSONAS:
            msgs = [store[v][f"{s}:{p}"] for s in range(N_STRATEGIES)]
            per[str(p)] = _metrics(msgs, judge_all(msgs, profiles[p]))
        metrics[v] = per

    GEN.write_text(json.dumps({"personas": PERSONAS, "n_cand": N_CAND,
                               "bon_temp": BON_TEMP, "store": store}))
    OUT.write_text(json.dumps({"metrics": metrics}, indent=1))
    _report(metrics)
    print("[+] wrote", GEN, "and", OUT)


def _report(metrics):
    print("\n=== message-level A/B (mean over personas) ===")
    print(f"{'variant':<13}{'paircos':>8}{'>0.9':>6}{'effrk':>7}{'toneNN':>8}"
          f"{'emphNN':>8}{'intNN':>7}{'judgeSD':>9}{'modal':>7}{'uniqOpen':>9}")
    for v, per in metrics.items():
        g = lambda f: float(np.mean([f(d) for d in per.values()]))  # noqa: E731
        print(f"{v:<13}{g(lambda d: d['pair_cos_mean']):>8.3f}"
              f"{g(lambda d: d['pair_cos_frac_gt_0.90']):>6.2f}"
              f"{g(lambda d: d['effective_rank']):>7.1f}"
              f"{g(lambda d: d['levels']['tone']['nn_purity']):>8.2f}"
              f"{g(lambda d: d['levels']['emphasis']['nn_purity']):>8.2f}"
              f"{g(lambda d: d['levels']['intensity']['nn_purity']):>7.2f}"
              f"{g(lambda d: d['judge']['sd']):>9.3f}"
              f"{g(lambda d: d['judge']['modal_share']):>7.2f}"
              f"{g(lambda d: d['n_unique_openings']):>9.1f}")


# --------------------------------------------------------------------- stage 2
def reward():
    from code.rl_env import ElitePersuaderEnv
    from code.scenario import LLMScenario
    from code.torch_trainer import PPOConfig
    from code.eval_final import run_episode, HeuristicPolicy
    import code.LLM_module as LM

    variant = sys.argv[2] if len(sys.argv) > 2 else "ex"
    subset = list(range(0, N_STRATEGIES, 3))          # 60 strategies
    n_seeds = 6
    specs = all_strategies()
    profiles = generate_synthetic_profiles(50, seed=1234)
    spec_to_idx = {s: i for i, s in enumerate(specs)}

    pairs = [(s, p) for s in subset for p in range(50)]
    print(f"[*] building {variant} messages for {len(pairs)} (strategy,profile) pairs ...",
          flush=True)

    def make(sp):
        s, p = sp
        return s, profiles[p], _variant_message(variant, s, p, specs, profiles)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(make, pairs))
    msg_lut = {(s, prof): m for s, prof, m in rows}

    print("[*] judging ...", flush=True)

    def jj(r):
        _, prof, m = r
        return (m, prof), judge_persuasiveness(m, prof)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jud_lut = dict(pool.map(jj, rows))

    real_render = LM.render_strategy_message
    real_judge = LM.judge_persuasiveness

    def patched_render(spec, target_profile="", topic="the policy",
                       stance_direction="opposing"):
        key = (spec_to_idx.get(spec), target_profile)
        if key in msg_lut:
            return msg_lut[key]
        return real_render(spec, target_profile=target_profile, topic=topic,
                           stance_direction=stance_direction)

    def patched_judge(message, target_profile):
        key = (message, target_profile)
        if key in jud_lut:
            return jud_lut[key]
        return real_judge(message, target_profile)

    LM.render_strategy_message = patched_render
    LM.judge_persuasiveness = patched_judge

    cfg = PPOConfig(agent_num=50, n_persuaders=5, topology="erdos_renyi",
                    n_periods=10, rollout_workers=WORKERS)
    base_env = ElitePersuaderEnv(
        scenario=LLMScenario(agent_num=50, use_post_evidence=False,
                             use_profile_evidence=False),
        n_periods=10, topology="erdos_renyi", topic=cfg.topic,
        target_selection="random_neighbor", obs_mode="pair", n_persuaders=5)
    run_episode(base_env.clone(), 500000, HeuristicPolicy(90))

    def run(job):
        s, k = job
        return s, k, float(run_episode(base_env.clone(), 500000 + k,
                                       HeuristicPolicy(strategy_idx=s)))

    jobs = [(s, k) for s in subset for k in range(n_seeds)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows2 = list(pool.map(run, jobs))
    mat = {}
    for s, k, r in rows2:
        mat.setdefault(s, []).append(r)
    means = {s: float(np.mean(v)) for s, v in mat.items()}
    vals = np.array(list(means.values()))

    # base reference from the full measured landscape
    ref = json.loads((HERE / "strategy_value.json").read_text())
    rs, rm = ref["strategies"], np.array(ref["mean"])
    base_vals = np.array([rm[rs.index(s)] for s in subset if s in rs])
    share_up = float(np.mean([means[s] > rm[rs.index(s)] for s in subset if s in rs]))

    out = {
        "variant": variant, "subset": subset, "n_seeds": n_seeds,
        "mean_by_strategy": {str(k): round(v, 3) for k, v in means.items()},
        "spread": {
            "mean": round(float(vals.mean()), 3),
            "sd": round(float(vals.std(ddof=1)), 3),
            "range": round(float(vals.max() - vals.min()), 3),
            "p10_p90": round(float(np.percentile(vals, 90) - np.percentile(vals, 10)), 3),
            "best_idx": int(max(means, key=means.get)),
            "worst_idx": int(min(means, key=means.get)),
        },
        "base_reference_subset": {
            "mean": round(float(base_vals.mean()), 3),
            "sd": round(float(base_vals.std(ddof=1)), 3),
            "range": round(float(base_vals.max() - base_vals.min()), 3),
            "p10_p90": round(float(np.percentile(base_vals, 90) - np.percentile(base_vals, 10)), 3),
        },
        "share_strategies_above_base": round(share_up, 3),
    }
    p = HERE / f"cond_fix_reward_{variant}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"[+] {variant}: mean {out['spread']['mean']} sd {out['spread']['sd']} "
          f"range {out['spread']['range']} p10-90 {out['spread']['p10_p90']} | "
          f"base sd {out['base_reference_subset']['sd']} | "
          f"above-base {share_up:.2f} -> {p}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if cmd == "gen":
        gen()
    elif cmd == "reward":
        reward()
