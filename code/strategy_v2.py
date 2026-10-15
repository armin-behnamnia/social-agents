"""Strategy space v2 — a relevance-conditioned message plan.

Motivation
----------
The v1 grid ``(appeal, tone, emphasis, intensity)`` barely controls the
rendered text (nearest-neighbour purity ≈ chance) and forcing rhetorical
adherence *lowers* persuasiveness. The reward is
``judge(persona-fit) · cos(post, target)``, so the learnable axis is
**target-relevance**, not rhetorical form.

This space makes each action a concrete, realisable plan that the frozen
LLM can actually instantiate:

    action = (angle, evidence, address)

- ``angle``    : which concern to foreground for THIS reader (6)
- ``evidence`` : what concrete support to give (4)
- ``address``  : which identity of the reader to speak to (3)

N = 6 · 4 · 3 = 72 realizable arms (vs 180 near-chance ones), small
enough for the policy to generalise.

Selected by ``PERSUASION_STRATEGY_SPACE=v2`` (see ``code/strategy.py``),
which re-exports this module's public names in place of the v1 grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

ANGLES = ["cost", "safety", "fairness", "freedom", "community", "effectiveness"]
EVIDENCE_KINDS = ["statistic", "expert", "anecdote", "peers"]
ADDRESSES = ["taxpayer", "parent", "worker"]

STRATEGY_DIMS = (len(ANGLES), len(EVIDENCE_KINDS), len(ADDRESSES))
N_STRATEGIES = STRATEGY_DIMS[0] * STRATEGY_DIMS[1] * STRATEGY_DIMS[2]

ANGLE_DESC = {
    "cost": "the direct financial cost to the reader ('this will cost you ...')",
    "safety": "the risk to the reader's safety or wellbeing",
    "fairness": "who is treated unequally or left out",
    "freedom": "the choice or autonomy being taken away",
    "community": "what the reader's community stands to lose together",
    "effectiveness": "whether the policy actually achieves its stated goal",
}
EVIDENCE_DESC = {
    "statistic": "a specific number or percentage",
    "expert": "a named expert, organization or study",
    "anecdote": "a first-person anecdote with one concrete detail",
    "peers": "what people like the reader are doing ('my neighbours ...')",
}
ADDRESS_DESC = {
    "taxpayer": "a taxpayer who funds this",
    "parent": "a parent or caregiver",
    "worker": "a worker whose job is affected",
}

PROMPT = """You are writing ONE persuasive social-media post (max 50 words, first person) that takes an {stance} stance on {topic} and is maximally relevant to the specific reader below.

Lead with this concern (the reader's angle): {angle}
Support it with: {evidence}
Speak to the reader as: {address}

Reader profile: {profile}

Rules:
- The FIRST sentence must foreground the angle above.
- Include exactly the named support.
- Do not mention the strategy.
Return only the post."""


@dataclass(frozen=True)
class StrategySpec:
    angle: str
    evidence: str
    address: str

    def render_prompt(
        self,
        target_profile: str,
        topic: str = "the policy",
        stance_direction: str = "opposing",
    ) -> str:
        return PROMPT.format(
            stance=stance_direction,
            topic=topic,
            angle=ANGLE_DESC[self.angle],
            evidence=EVIDENCE_DESC[self.evidence],
            address=ADDRESS_DESC[self.address],
            profile=target_profile,
        )


def index_to_strategy(index: int) -> StrategySpec:
    """Decode a Discrete action index into a StrategySpec."""
    if not 0 <= index < N_STRATEGIES:
        raise ValueError(
            f"strategy index {index} out of range [0, {N_STRATEGIES})"
        )
    a, e, d = np_unravel(index, STRATEGY_DIMS)
    return StrategySpec(
        angle=ANGLES[a],
        evidence=EVIDENCE_KINDS[e],
        address=ADDRESSES[d],
    )


def strategy_to_index(spec: StrategySpec) -> int:
    a = ANGLES.index(spec.angle)
    e = EVIDENCE_KINDS.index(spec.evidence)
    d = ADDRESSES.index(spec.address)
    return np_ravel((a, e, d), STRATEGY_DIMS)


def all_strategies() -> list[StrategySpec]:
    return [
        StrategySpec(angle=a, evidence=e, address=d)
        for a, e, d in product(ANGLES, EVIDENCE_KINDS, ADDRESSES)
    ]


def describe(spec: StrategySpec) -> str:
    return (
        f"angle={spec.angle}, evidence={spec.evidence}, "
        f"address={spec.address}"
    )


# ----------------------------------------------------------------------
# shared index helpers (same row-major order as strategy.np_ravel)
# ----------------------------------------------------------------------

def np_unravel(index: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    rem = index
    for d in reversed(dims):
        out.append(rem % d)
        rem //= d
    return tuple(reversed(out))


def np_ravel(idx: tuple[int, ...], dims: tuple[int, ...]) -> int:
    out = 0
    for i, d in zip(idx, dims):
        out = out * d + i
    return out
