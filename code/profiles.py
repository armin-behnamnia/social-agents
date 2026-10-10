"""Persona profile generation.

Two backends:

- `generate_synthetic_profiles(n, seed)`: deterministic, template-based
  personas (name, age ~ Gaussian, balanced gender, interests). Used for
  RL training where hundreds of episode resets need profiles instantly
  and reproducibly, without LLM calls.
- `generate_llm_profiles(n)`: the paper's Prompt 1 via the live LLM.

Both produce the same format: one 50-word-style profile string per
agent. They are interchangeable at the `Simulation(profiles=...)` level.
"""

from __future__ import annotations

import numpy as np

_FIRST_NAMES_M = [
    "James", "Benjamin", "Lucas", "Henry", "Theodore", "Jack", "Owen",
    "Daniel", "Marcus", "Victor", "Felix", "Hugo", "Leo", "Adrian",
    "Nathan", "Elias", "Simon", "David", "Peter", "Adam",
]
_FIRST_NAMES_F = [
    "Clara", "Elena", "Maya", "Sophie", "Isla", "Nora", "Alice",
    "Vivian", "Hazel", "Iris", "Lydia", "Rosa", "June", "Mabel",
    "Freya", "Nina", "Daphne", "Pearl", "Stella", "Cora",
]
_INTERESTS = [
    "history and museums", "hiking and camping", "cooking regional food",
    "jazz and vinyl records", "gardening and botany", "board games",
    "photography", "marathon running", "classical literature",
    "homebrewing coffee", "astronomy", "pottery and crafts",
    "cycling around town", "birdwatching", "independent cinema",
    "podcasts about science", "volunteering at shelters", "chess",
    "restoring furniture", "urban sketching", "folk music",
    "machine tinkering", "yoga and meditation", "local politics",
    "farmer's markets",
]
_OCCUPATIONS = [
    "nurse", "high-school teacher", "software developer", "electrician",
    "librarian", "chef", "accountant", "civil engineer", "journalist",
    "graphic designer", "social worker", "pharmacist", "bus driver",
    "university lecturer", "carpenter", "lab technician", "vet assistant",
    "retail manager", "data analyst", "physiotherapist",
]
_PERSONALITY = [
    "cautious and analytical", "warm and talkative", "skeptical of hype",
    "community-minded", "quietly stubborn", "curious about everything",
    "pragmatic and busy", "idealistic", "dry sense of humor",
    "detail-oriented",
]

_TEMPLATE = (
    "{name} is a {age}-year-old {gender} who works as a {occupation}. "
    "A {personality} {pronoun} enjoys {interest1} and {interest2}, and "
    "prefers discussions grounded in personal experience."
)


def generate_synthetic_profiles(
    n: int, seed: int | None = None
) -> list[str]:
    """Deterministic persona bank for reproducible RL training."""
    rng = np.random.default_rng(seed)
    ages = np.clip(rng.normal(42, 12, size=n), 20, 75).astype(int)

    profiles: list[str] = []
    for i in range(n):
        male = i % 2 == 0  # balanced gender split
        name_pool = _FIRST_NAMES_M if male else _FIRST_NAMES_F
        name = name_pool[int(rng.integers(0, len(name_pool)))]
        interest1 = _INTERESTS[int(rng.integers(0, len(_INTERESTS)))]
        interest2 = _INTERESTS[int(rng.integers(0, len(_INTERESTS)))]
        if interest2 == interest1:
            interest2 = _INTERESTS[
                (int(rng.integers(1, len(_INTERESTS))) + i) % len(_INTERESTS)
            ]
        profiles.append(
            _TEMPLATE.format(
                name=name,
                age=int(ages[i]),
                gender="man" if male else "woman",
                occupation=_OCCUPATIONS[
                    int(rng.integers(0, len(_OCCUPATIONS)))
                ],
                personality=_PERSONALITY[
                    int(rng.integers(0, len(_PERSONALITY)))
                ],
                pronoun="he" if male else "she",
                interest1=interest1,
                interest2=interest2,
            )
        )
    return profiles


def generate_llm_profiles(n: int = 50) -> list[str]:
    """Paper Prompt 1 personas via the live LLM."""
    from .simulation import generate_profiles

    return generate_profiles(n)
