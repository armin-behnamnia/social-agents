"""Goal-direction measurement a(m) (Track B2).

The stance update `Δs = p·β·(s_src − s)` is text-independent: a target is
pulled toward the source's stance regardless of what the post argues, so a
persuasive-but-off-goal post is rewarded the same as an on-goal one. This
module measures how strongly a rendered message *argues for the goal*,
`a(m) ∈ [0, 1]` (1 = fully on-goal, 0 = fully opposite, 0.5 = neutral).

Three interchangeable backends, cheapest first:

- ``semdiff`` (default): embedding semantic differential, reuses the
  existing embedding backend — near-free::

      a(m) = ½·(cos(m, G_goal) − cos(m, G_anti) + 1)

- ``nli``: frozen local MNLI entailment, `entail(m, G_goal) −
  entail(m, G_anti)`. Requires ``transformers``; if unavailable it falls
  back to ``semdiff`` with a warning (kept offline-safe).
- ``judge``: a single cached LLM goal-axis prompt returning −1..+1.

``get_align_fn(kind, goal, anti)`` returns a cached ``a(m) -> float``.
"""

from __future__ import annotations

import re

import numpy as np

_JUDGE_PROMPT = """You are scoring whether a social-media post argues FOR or AGAINST a proposition.

Proposition (the goal): {goal}

Post: {post}

How does the post relate to the proposition? Reply with a single number from -1 (strongly AGAINST the proposition) to +1 (strongly FOR the proposition), e.g. 0.7. Reply with only the number."""

_NUM = re.compile(r"[-+]?\d*\.?\d+")


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _semdiff_fn(goal: str, anti: str, cache: bool):
    from .tools import text2embedding

    g = np.asarray(text2embedding(goal), dtype=np.float64)
    n = np.asarray(text2embedding(anti), dtype=np.float64)

    def a(message: str) -> float:
        m = np.asarray(text2embedding(message), dtype=np.float64)
        return float(0.5 * (_cos(m, g) - _cos(m, n) + 1.0))

    return a


def _judge_fn(goal: str, anti: str, cache: bool):
    from . import LLM_module

    def a(message: str) -> float:
        raw = LLM_module.call_llm(
            _JUDGE_PROMPT.format(goal=goal, post=message),
            temperature=0.0, max_tokens=8, cache=cache,
        )
        m = _NUM.search(raw or "")
        if not m:
            return 0.5
        x = max(-1.0, min(1.0, float(m.group(0))))
        return float((x + 1.0) / 2.0)

    return a


def _nli_fn(goal: str, anti: str, cache: bool):
    try:
        from transformers import pipeline  # type: ignore

        clf = pipeline("text-classification", model="facebook/bart-large-mnli")

        def entail(premise: str, hypothesis: str) -> float:
            out = clf({"text": premise, "text_pair": hypothesis},
                      top_k=None)
            for d in out:
                if d["label"].lower().startswith("entail"):
                    return float(d["score"])
            return 0.0

        def a(message: str) -> float:
            e_goal = entail(message, goal)
            e_anti = entail(message, anti)
            return float(0.5 * (e_goal - e_anti + 1.0))

        return a
    except Exception as e:  # noqa: BLE001 - offline-safe fallback
        import warnings
        warnings.warn(
            f"goal_align: NLI backend unavailable ({e}); using semdiff"
        )
        return _semdiff_fn(goal, anti, cache)


_CACHE: dict = {}


def get_align_fn(kind: str = "semdiff",
                 goal: str = "This policy is harmful and should be rejected.",
                 anti: str = "This policy is beneficial and should be supported.",
                 cache: bool = True):
    """Return a cached ``a(message) -> float in [0,1]`` or ``None`` for none."""
    kind = (kind or "none").lower()
    if kind in ("none", "", "off"):
        return None
    key = (kind, goal, anti)
    if key in _CACHE:
        return _CACHE[key]
    if kind == "semdiff":
        fn = _semdiff_fn(goal, anti, cache)
    elif kind == "judge":
        fn = _judge_fn(goal, anti, cache)
    elif kind == "nli":
        fn = _nli_fn(goal, anti, cache)
    else:
        raise ValueError(f"unknown goal_align backend: {kind!r}")

    if cache:
        memo: dict[str, float] = {}

        def wrapped(message: str) -> float:
            if message not in memo:
                memo[message] = fn(message)
            return memo[message]
        _CACHE[key] = wrapped
        return wrapped
    _CACHE[key] = fn
    return fn
