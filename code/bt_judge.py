"""Bradley-Terry pairwise persuasiveness judge (reward model).

Why
---
The absolute 0-10 judge saturates: 180 strategies collapse onto 4-6
distinct scores, 54% landing on one bucket (SD 0.072). A reward that
flat cannot tell strategies apart, so the policy can only learn a
constant.

This module replaces the absolute score with a **pairwise** one. For a
given reader we ask the frozen LLM, for pairs of messages, which is more
persuasive *for that reader*; we fit Bradley-Terry strengths from the
win/loss graph and score each message by its **expected win rate**
against the field. That (a) has far higher resolution (every comparison
contributes) and (b) is target-conditional by construction — the
comparison is always "for THIS reader".

Scores are shifted to the absolute judge's mean (keeping the BT spread)
so the reward stays on a familiar scale and baselines remain comparable.

Usage
-----
    from code.bt_judge import build_table, BTJudge
    table = build_table(strategies, profiles, topic="the policy")
    judge = BTJudge(table)          # callable: (message, profile) -> float
    judge.save("output/bt_judge_v2.json")

The env takes ``judge_fn=judge`` (see ``rl_env._judge_score``); unknown
(message, profile) pairs fall back to the absolute judge.
"""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import LLM_module
from .config import GLOBAL_LLM_CONFIG as C

PAIR_PROMPT = """You are comparing two persuasive social-media posts written for ONE specific reader. Decide which post is more persuasive FOR THIS READER.

Reader profile: {profile}

Post A: {a}

Post B: {b}

Which post is more persuasive for this reader overall - A, B, or EQUAL? Weigh how well each fits this reader's personality, interests and values. Reply with exactly one of: A, B, EQUAL."""

_TOK = re.compile(r"\b(A|B|EQUAL)\b", re.IGNORECASE)


def compare(a: str, b: str, profile: str, *, cache: bool = True) -> int | None:
    """1 if A wins, -1 if B wins, 0 if equal, None if unparseable."""
    raw = LLM_module.call_llm(
        PAIR_PROMPT.format(profile=profile, a=a, b=b),
        temperature=0.0,
        max_tokens=8,
        cache=cache,
    )
    m = _TOK.search(raw or "")
    if not m:
        return None
    tok = m.group(1).upper()
    return {"A": 1, "B": -1, "EQUAL": 0}[tok]


def fit_bt(
    items: list[int],
    edges: list[tuple[int, int, int]],
    *,
    iters: int = 500,
    tol: float = 1e-9,
) -> dict[int, float]:
    """MM fit of Bradley-Terry strengths.

    edges: list of (i, j, outcome) with outcome in {1,-1,0} meaning
    i beat j / j beat i / tie. Ties count as half a win each.
    """
    wins = {i: 0.0 for i in items}
    n = {}
    for i, j, o in edges:
        wins[i] += 1.0 if o > 0 else (0.5 if o == 0 else 0.0)
        wins[j] += 1.0 if o < 0 else (0.5 if o == 0 else 0.0)
        n[(i, j)] = n.get((i, j), 0) + 1

    theta = {i: 1.0 for i in items}
    for _ in range(iters):
        new = {}
        for i in items:
            denom = 0.0
            for (a, b), cnt in n.items():
                if a == i:
                    denom += cnt / (theta[i] + theta[b])
                elif b == i:
                    denom += cnt / (theta[i] + theta[a])
            new[i] = wins[i] / denom if denom > 0 and wins[i] > 0 else theta[i]
        s = sum(new.values()) or 1.0
        new = {i: v * len(items) / s for i, v in new.items()}
        delta = max(abs(new[i] - theta[i]) for i in items)
        theta = new
        if delta < tol:
            break
    return theta


def expected_win_rate(
    theta: dict[int, float], items: list[int]
) -> dict[int, float]:
    out = {}
    for i in items:
        tot, cnt = 0.0, 0
        for j in items:
            if i == j:
                continue
            tot += theta[i] / (theta[i] + theta[j])
            cnt += 1
        out[i] = tot / cnt if cnt else 0.5
    return out


@dataclass
class BTJudge:
    """Callable BT reward model: (message, profile) -> score in [0, 1]."""

    table: dict[tuple[str, str], float] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def __call__(self, message: str, profile: str) -> float | None:
        return self.table.get((message, profile))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stats": self.stats,
            "entries": [
                {"message": m, "profile": p, "score": s}
                for (m, p), s in self.table.items()
            ],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "BTJudge":
        d = json.loads(Path(path).read_text())
        table = {(e["message"], e["profile"]): e["score"] for e in d["entries"]}
        return cls(table=table, stats=d.get("stats", {}))


def build_table(
    strategies,
    profiles: list[str],
    *,
    topic: str = "the policy",
    n_opponents: int = 10,
    seed: int = 0,
    workers: int | None = None,
    progress: bool = True,
) -> BTJudge:
    """Render every (strategy x profile) post and fit BT per profile.

    A random comparison graph of degree ~``n_opponents`` is drawn per
    profile; the LLM judges each edge once; BT strengths are fit and
    converted to expected win rates, shifted to the absolute judge's
    mean.
    """
    n_workers = workers or C.max_workers
    n_strat = len(strategies)

    # 1. render the full (strategy x profile) matrix (cached) --------
    if progress:
        print(f"[*] BT build: rendering {n_strat * len(profiles)} posts "
              f"({n_strat} strategies x {len(profiles)} profiles)", flush=True)
    render_jobs = [(si, pi) for pi in range(len(profiles))
                   for si in range(n_strat)]

    def _render(job):
        si, pi = job
        m = LLM_module.render_strategy_message(
            strategies[si], target_profile=profiles[pi], topic=topic,
            stance_direction="opposing",
        )
        return (si, pi), m

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        msg = dict(pool.map(_render, render_jobs))

    # 2. comparison graph per profile --------------------------------
    rng = random.Random(seed)
    edge_jobs = []  # (pi, si, sj)
    for pi in range(len(profiles)):
        edges = set()
        for si in range(n_strat):
            for _ in range(n_opponents):
                sj = rng.randrange(n_strat)
                if sj != si:
                    edges.add((min(si, sj), max(si, sj)))
        for si, sj in edges:
            edge_jobs.append((pi, si, sj))
    if progress:
        print(f"[*] BT build: {len(edge_jobs)} pairwise comparisons "
              f"(~{len(edge_jobs) // max(1, len(profiles))}/profile)",
              flush=True)

    def _cmp(job):
        pi, si, sj = job
        out = compare(msg[(si, pi)], msg[(sj, pi)], profiles[pi])
        return (pi, si, sj, out)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_cmp, edge_jobs))

    # 3. fit BT per profile + shift to absolute-judge mean -----------
    per_edges: dict[int, list] = {}
    n_bad = 0
    for pi, si, sj, out in results:
        if out is None:
            n_bad += 1
            continue
        per_edges.setdefault(pi, []).append((si, sj, out))

    # absolute judge mean over the same posts (cached) for scale match
    abs_scores = []
    def _abs(job):
        si, pi = job
        return LLM_module.judge_persuasiveness(msg[(si, pi)], profiles[pi])
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        abs_scores = list(pool.map(_abs, render_jobs))
    abs_mean = sum(abs_scores) / len(abs_scores) if abs_scores else 0.8

    items = list(range(n_strat))
    table: dict[tuple[str, str], float] = {}
    win_rates_all = []
    per_profile = {}
    for pi in range(len(profiles)):
        edges = per_edges.get(pi, [])
        theta = fit_bt(items, edges)
        wr = expected_win_rate(theta, items)
        wr_mean = sum(wr.values()) / len(wr)
        for si in items:
            shifted = min(max(wr[si] - wr_mean + abs_mean, 0.0), 1.0)
            table[(msg[(si, pi)], profiles[pi])] = shifted
            win_rates_all.append(wr[si])
        per_profile[pi] = {
            "n_edges": len(edges),
            "best": max(wr, key=wr.get),
            "worst": min(wr, key=wr.get),
        }

    import statistics as st
    scores = list(table.values())
    stats = {
        "space": getattr(strategies[0], "__class__", type(strategies[0])).__name__,
        "n_strategies": n_strat,
        "n_profiles": len(profiles),
        "n_comparisons": len(edge_jobs),
        "n_unparseable": n_bad,
        "abs_mean": round(abs_mean, 4),
        "bt_winrate_mean": round(sum(win_rates_all) / len(win_rates_all), 4),
        "bt_winrate_sd": round(st.pstdev(win_rates_all), 4),
        "score_mean": round(sum(scores) / len(scores), 4),
        "score_sd": round(st.pstdev(scores), 4),
        "n_unique_scores": len({round(s, 3) for s in scores}),
    }
    if progress:
        print(f"[+] BT build done: SD {stats['score_sd']} "
              f"(abs judge mean {stats['abs_mean']}, "
              f"{stats['n_unique_scores']} distinct scores)", flush=True)
    return BTJudge(table=table, stats=stats)
