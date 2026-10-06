"""Strategy space for approach 2 (strategy-parameterized persuasion).

The RL policy does not emit raw text. It selects a strategy spec:
    (appeal, tone, emphasis, intensity)
which a frozen LLM renders into a natural-language message. This keeps
the action space small (clean credit assignment, cheap training) while
retaining expressive messages.

Action encoding: a single Discrete(N) index over the Cartesian product
of the categorical dimensions x intensity bins. N = 3 * 4 * 5 * 3 = 180.

A learned-target-selection extension is planned but not included here:
the env currently presents one candidate target per step (configurable
selection heuristic) and the policy conditions the strategy on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

APPEALS = ["logos", "pathos", "ethos"]
TONES = ["neutral", "urgent", "empathetic", "confrontational"]
EMPHASES = [
    "evidence",
    "personal_story",
    "social_proof",
    "risk",
    "authority",
]
INTENSITY_BINS = [0.33, 0.66, 1.0]
INTENSITY_LABELS = ["measured", "assertive", "forceful"]

STRATEGY_DIMS = (len(APPEALS), len(TONES), len(EMPHASES), len(INTENSITY_BINS))
N_STRATEGIES = STRATEGY_DIMS[0] * STRATEGY_DIMS[1] * STRATEGY_DIMS[2] * STRATEGY_DIMS[3]


@dataclass(frozen=True)
class StrategySpec:
    appeal: str
    tone: str
    emphasis: str
    intensity: float  # in (0, 1]

    @property
    def intensity_label(self) -> str:
        idx = min(
            range(len(INTENSITY_BINS)),
            key=lambda i: abs(INTENSITY_BINS[i] - self.intensity),
        )
        return INTENSITY_LABELS[idx]


def index_to_strategy(index: int) -> StrategySpec:
    """Decode a Discrete action index into a StrategySpec."""
    if not 0 <= index < N_STRATEGIES:
        raise ValueError(f"strategy index {index} out of range [0, {N_STRATEGIES})")
    a, t, e, i = np_unravel(index, STRATEGY_DIMS)
    return StrategySpec(
        appeal=APPEALS[a],
        tone=TONES[t],
        emphasis=EMPHASES[e],
        intensity=INTENSITY_BINS[i],
    )


def strategy_to_index(spec: StrategySpec) -> int:
    a = APPEALS.index(spec.appeal)
    t = TONES.index(spec.tone)
    e = EMPHASES.index(spec.emphasis)
    i = min(
        range(len(INTENSITY_BINS)),
        key=lambda k: abs(INTENSITY_BINS[k] - spec.intensity),
    )
    return np_ravel((a, t, e, i), STRATEGY_DIMS)


def all_strategies() -> list[StrategySpec]:
    return [
        StrategySpec(appeal=a, tone=t, emphasis=e, intensity=inten)
        for a, t, e, inten in product(APPEALS, TONES, EMPHASES, INTENSITY_BINS)
    ]


def describe(spec: StrategySpec) -> str:
    return (
        f"appeal={spec.appeal}, tone={spec.tone}, "
        f"emphasis={spec.emphasis}, intensity={spec.intensity:.2f}"
    )


# ----------------------------------------------------------------------
# small helpers (avoid a numpy dependency in this module's public API
# for pure lookup use-cases)
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
