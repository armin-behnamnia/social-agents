#!/usr/bin/env python
"""Measure the *semantic* variety of the 180 generated posts.

Offline (reads the warm .llm_cache for messages; only embeddings hit the
local embedding API). String uniqueness != learnable variety: what matters
is whether varying the strategy moves the message along the only two
reward-relevant axes the env has —

    influence_prob = (0.25*prof + 0.25*content_sim + 0.25*stance + 0.25*deg)
                     * persuasiveness

where  content_sim = cos(post_embedding, target_embedding)   and
       persuasiveness = LLM judge.

Writes report/message_variety.json.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from code.config import GLOBAL_LLM_CONFIG as C
from code.strategy import all_strategies, N_STRATEGIES
from code.profiles import generate_synthetic_profiles
from code.tools import text2embedding
from code.LLM_module import _PERSUASIVENESS_PROMPT, _NUMBER_RE

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".llm_cache"

RENDER_PROMPT = """You are crafting a persuasive social-media message.

Persuasion strategy:
- Rhetorical appeal: {appeal} (logos = facts/logic, pathos = emotion, ethos = credibility)
- Tone: {tone}
- Emphasis: {emphasis} (evidence = data and studies, personal_story = lived experience, social_proof = what others are doing, risk = dangers of inaction, authority = expert consensus)
- Intensity: {label} ({intensity:.2f}/1.00)

Target audience profile: {profile}

Write ONE short social-media post (maximum 50 words, first person) that
takes an {stance} stance on {topic} and follows the strategy
above as closely as possible. Do not mention the strategy explicitly.
Return only the message."""

APPEALS = ["logos", "pathos", "ethos"]
TONES = ["neutral", "urgent", "empathetic", "confrontational"]
EMPHASES = ["evidence", "personal_story", "social_proof", "risk", "authority"]
INTENS = ["measured", "assertive", "forceful"]
DIMS = (3, 4, 5, 3)


def _key(prompt, temperature, max_tokens, system=None):
    return hashlib.sha256(json.dumps({
        "model": C.model, "prompt": prompt, "system": system,
        "temperature": temperature, "max_tokens": max_tokens,
    }, sort_keys=True).encode()).hexdigest()


def _read(k):
    p = CACHE / f"{k}.json"
    return json.loads(p.read_text())["content"] if p.exists() else None


def render(spec, profile, topic="the policy", stance="opposing"):
    prompt = RENDER_PROMPT.format(
        appeal=spec.appeal, tone=spec.tone, emphasis=spec.emphasis,
        label=spec.intensity_label, intensity=spec.intensity,
        profile=profile, stance=stance, topic=topic)
    return _read(_key(prompt, 0.0, C.generation_max_tokens))


def judge(msg, profile):
    raw = _read(_key(_PERSUASIVENESS_PROMPT.format(profile=profile, message=msg),
                     C.inference_temperature, C.inference_max_tokens))
    if raw is None:
        return None
    m = _NUMBER_RE.search(raw)
    return min(max(float(m.group()) / 10.0, 0.0), 1.0) if m else 0.5


def _unit(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


def _offdiag_cos_stats(E):
    S = E @ E.T
    n = S.shape[0]
    off = S[~np.eye(n, dtype=bool)]
    return {
        "mean": float(off.mean()), "median": float(np.median(off)),
        "p10": float(np.percentile(off, 10)), "p90": float(np.percentile(off, 90)),
        "min": float(off.min()),
        "frac_gt_0.90": float((off > 0.90).mean()),
        "frac_gt_0.95": float((off > 0.95).mean()),
    }


def _effective_rank(E):
    """Participation ratio of the eigenvalues of the covariance matrix."""
    Ec = E - E.mean(axis=0, keepdims=True)
    C = Ec.T @ Ec / max(1, E.shape[0] - 1)
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 0, None)
    return float((w.sum() ** 2) / max(1e-12, (w ** 2).sum()))


def _pca_explained(E, ks=(1, 5, 10)):
    Ec = E - E.mean(axis=0, keepdims=True)
    w = np.linalg.svd(Ec, compute_uv=False) ** 2
    w = w / w.sum()
    return {f"pc{k}": float(w[:k].sum()) for k in ks}


def _opening(text, k=4):
    return " ".join(re.findall(r"[a-z']+", (text or "").lower())[:k])


def main():
    specs = all_strategies()
    profiles = generate_synthetic_profiles(50, seed=1234)
    personas = [0, 1, 2]

    per_persona = {}
    E_by_persona = {}
    for p in personas:
        prof = profiles[p]
        msgs = [render(s, prof) for s in specs]
        E = _unit(text2embedding(msgs))          # 180 x 1024, unit norm
        E_by_persona[p] = E
        target = _unit(text2embedding(prof)[None, :])[0]

        content_sim = E @ target                  # cos(msg, target) per strategy
        scores = np.array([judge(m, prof) or 0.5 for m in msgs])
        openings = [_opening(m) for m in msgs]

        # how much can content move influence_prob, out of the 0.25 weight
        content_term = 0.25 * (content_sim + 1.0) / 2.0
        per_persona[p] = {
            "pair_cos": _offdiag_cos_stats(E),
            "effective_rank": _effective_rank(E),
            "pca": _pca_explained(E),
            "content_sim": {
                "mean": float(content_sim.mean()),
                "std": float(content_sim.std(ddof=1)),
                "min": float(content_sim.min()), "max": float(content_sim.max()),
                "range": float(content_sim.max() - content_sim.min()),
            },
            "content_term_span": float(content_term.max() - content_term.min()),
            "score": {
                "mean": float(scores.mean()), "std": float(scores.std(ddof=1)),
                "min": float(scores.min()), "max": float(scores.max()),
                "n_unique": int(len(set(np.round(scores, 3)))),
            },
            "n_unique_messages": len(set(msgs)),
            "n_unique_openings": len(set(openings)),
            "top_openings": sorted(
                {o: openings.count(o) for o in set(openings)}.items(),
                key=lambda kv: -kv[1])[:5],
        }

    # ---- cross-dimension variance decomposition on persona 0 embeddings ----
    E = E_by_persona[personas[0]]

    def eta2(level_of):
        groups = {}
        for i in range(N_STRATEGIES):
            groups.setdefault(level_of(i), []).append(E[i])
        gm = E.mean(axis=0)
        ss_between = sum(len(v) * np.sum((np.mean(v, axis=0) - gm) ** 2)
                         for v in groups.values())
        ss_total = np.sum((E - gm) ** 2)
        return float(ss_between / max(1e-12, ss_total))

    def dim_level(i, pos):
        out, rem = [], i
        for d in reversed(DIMS):
            out.append(rem % d); rem //= d
        return list(reversed(out))[pos]

    decomp = {
        "appeal": eta2(lambda i: APPEALS[dim_level(i, 0)]),
        "tone": eta2(lambda i: TONES[dim_level(i, 1)]),
        "emphasis": eta2(lambda i: EMPHASES[dim_level(i, 2)]),
        "intensity": eta2(lambda i: INTENS[dim_level(i, 3)]),
    }

    # ---- within vs across level cosine, and 1-NN same-level purity ----
    S = E @ E.T
    np.fill_diagonal(S, np.nan)

    def level_of(i, pos):
        return dim_level(i, pos)

    levels_by_dim = {
        "appeal": APPEALS, "tone": TONES,
        "emphasis": EMPHASES, "intensity": INTENS,
    }
    within_across = {}
    for pos, (dim, labels) in enumerate(levels_by_dim.items()):
        lv = np.array([labels[level_of(i, pos)] for i in range(N_STRATEGIES)])
        same = lv[:, None] == lv[None, :]
        within = float(np.nanmean(np.where(same, S, np.nan)))
        across = float(np.nanmean(np.where(~same, S, np.nan)))
        # 1-NN purity: nearest neighbour shares the level?
        nn = np.nanargmax(S, axis=1)
        purity = float((lv[nn] == lv).mean())
        within_across[dim] = {
            "within": round(within, 3), "across": round(across, 3),
            "gap": round(within - across, 3), "nn_purity": round(purity, 3),
        }

    # ---- can the reward be predicted from the message embedding? ----
    reward = None
    judge_mean = None
    sv = HERE / "strategy_value.json"
    if sv.exists():
        d = json.loads(sv.read_text())
        reward = np.asarray(d["mean"], dtype=float)
        judge_mean = np.asarray(d.get("mean_score"), dtype=float)

    def cv_r2(X, y, k=5, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        folds = np.array_split(idx, k)
        preds = np.empty_like(y, dtype=float)
        for f in folds:
            tr = np.setdiff1d(idx, f)
            Xtr = np.c_[np.ones(len(tr)), X[tr]]
            beta = np.linalg.lstsq(Xtr, y[tr], rcond=None)[0]
            preds[f] = np.c_[np.ones(len(f)), X[f]] @ beta
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot)

    reward_link = None
    if reward is not None:
        Ec = E - E.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(Ec, full_matrices=False)
        PCs = U[:, :20] * s[:20]
        content_sim = E @ _unit(text2embedding(profiles[0])[None, :])[0]
        from numpy import corrcoef

        def spear(a, b):
            ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
            return float(corrcoef(ra, rb)[0, 1])

        reward_link = {
            "cv_r2_reward_from_msgPCs": round(cv_r2(PCs, reward), 3),
            "spearman_content_vs_reward": round(spear(content_sim, reward), 3),
            "spearman_judge_vs_reward": (round(spear(judge_mean, reward), 3)
                                         if judge_mean is not None else None),
        }

    # ---- strategy vs persona: which axis moves the message more? ----
    # across-strategy (fixed persona): mean off-diag cos of E
    across_strategy = _offdiag_cos_stats(E)["mean"]
    # across-persona (fixed strategy): mean cos between personas' message for same strategy
    xp = []
    for i in range(N_STRATEGIES):
        vecs = [E_by_persona[p][i] for p in personas]
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                xp.append(float(vecs[a] @ vecs[b]))
    across_persona = float(np.mean(xp))

    # a random-unrelated reference
    ref = _unit(text2embedding([
        "The weather tomorrow will be mild with a chance of afternoon showers.",
        "I finally finished reading a long novel about the sea.",
        "My favorite recipe uses tomatoes, basil and a little olive oil."]))
    ref_cos = float((ref @ ref.T)[~np.eye(3, dtype=bool)].mean())

    out = {
        "n_strategies": N_STRATEGIES,
        "personas": personas,
        "per_persona": per_persona,
        "eta2_by_strategy_dim": decomp,
        "within_across_cos": within_across,
        "reward_link": reward_link,
        "mean_pair_cos_across_strategy": across_strategy,
        "mean_pair_cos_across_persona": across_persona,
        "reference_unrelated_cos": ref_cos,
    }
    (HERE / "message_variety.json").write_text(json.dumps(out, indent=1))

    print("persona |  mean±sd pairwise cos | eff.rank | content_sim range | content span | judge uniq | uniq openings")
    for p in personas:
        d = per_persona[p]
        print(f"   {p}    | {d['pair_cos']['mean']:.3f} ± {d['pair_cos']['std'] if 'std' in d['pair_cos'] else 0:.3f}"
              f"          | {d['effective_rank']:6.2f}  | "
              f"{d['content_sim']['min']:.3f}..{d['content_sim']['max']:.3f} | "
              f"{d['content_term_span']:.4f}      | {d['score']['n_unique']:>2}        | "
              f"{d['n_unique_openings']}/180")
    print("\nfrac of strategy pairs with cos>0.90 :",
          round(per_persona[0]['pair_cos']['frac_gt_0.90'], 3))
    print("PCA explained (persona 0):", {k: round(v, 3) for k, v in per_persona[0]['pca'].items()})
    print("eta^2 by strategy dim:", {k: round(v, 3) for k, v in decomp.items()})
    print("within-across cos / NN purity:",
          {k: v for k, v in within_across.items()})
    print("reward link:", reward_link)
    print("across-strategy cos:", round(across_strategy, 3),
          "| across-persona cos:", round(across_persona, 3),
          "| unrelated ref:", round(ref_cos, 3))
    print("wrote report/message_variety.json")


if __name__ == "__main__":
    main()
