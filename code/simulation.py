"""Simulation engine: population init, seeding, diffusion loop.

Replaces the missing Melodie model runner. Deterministic given a seed
(single shared RNG for acceptance gates), agents' observable content is
produced via `LLM_module` (mockable in tests).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from . import LLM_module
from .agent import LLMSocialAgent
from .network import build_network
from .scenario import LLMScenario

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(
        self,
        scenario: LLMScenario | None = None,
        topology: str = "watts_strogatz",
        profiles: list[str] | None = None,
        seed_post: str | None = None,
        initial_stances: np.ndarray | None = None,
    ):
        self.scenario = scenario or LLMScenario()
        self.rng = np.random.default_rng(self.scenario.rng_seed)
        self.topology = topology
        self.period = 0
        self.diffusion_events: list[dict] = []
        self.message_events: list[dict] = []

        self._init_population(profiles, initial_stances)
        self._init_network()
        self._seed_agents(seed_post)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_population(self, profiles, initial_stances):
        n = self.scenario.agent_num
        self.agents: list[LLMSocialAgent] = []
        for i in range(n):
            agent = LLMSocialAgent(i, scenario=self.scenario, rng=self.rng)
            self.agents.append(agent)

        if profiles is not None:
            if len(profiles) < n:
                raise ValueError(
                    f"need {n} profiles, got {len(profiles)}"
                )
            for agent, profile in zip(self.agents, profiles):
                agent.profile = profile
                agent.embedding = agent._get_embedding(profile)
                agent.profile_embedding = agent.embedding

        if initial_stances is None:
            # Paper setup: all targets start supportive (positive).
            initial_stances = np.full(n, 0.65)
        for agent, stance in zip(self.agents, np.asarray(initial_stances, dtype=float)):
            agent.stance = float(stance)

    def _init_network(self):
        adjacency, degrees, degree_centrality, betweenness = build_network(
            self.scenario.agent_num,
            topology=self.topology,
            seed=self.scenario.rng_seed,
        )
        for agent in self.agents:
            agent._neighbors = set(adjacency.get(agent.id, set()))
            agent.degree = int(degrees.get(agent.id, 0))
            agent.degree_centrality = float(degree_centrality.get(agent.id, 0.0))
            agent.betweenness_centrality = float(betweenness.get(agent.id, 0.0))

    def _seed_agents(self, seed_post):
        """Seed the first agent (id 0) with the opposing viewpoint."""
        seed_agent = self.agents[0]
        seed_agent.isSeed = True
        seed_agent.preference_state = 1
        seed_agent.stance = -1.0
        seed_agent.adoption_time = 0
        if seed_post is None:
            seed_post = (
                "I believe this policy is harmful for our community. "
                "The evidence is clear and people deserve to know the truth."
            )
        seed_agent.post = seed_post
        seed_agent.post_embedding = seed_agent._get_embedding(seed_post)
        seed_agent.message_id = seed_agent.generate_message_id()
        seed_agent.original_message_id = seed_agent.message_id

    # ------------------------------------------------------------------
    # Diffusion
    # ------------------------------------------------------------------

    def step(self) -> dict:
        """Advance the simulation by one period."""
        period = self.period

        for agent in self.agents:
            agent.new_adopter = 0

        # Snapshot of sources: only agents that can propagate this period.
        sources = [a for a in self.agents if a.preference_state == 1 and a.post]

        for source in sources:
            # Active targets: positive, non-adopted neighbors.
            targets = [
                self.agents[j]
                for j in source.neighbors
                if self.agents[j].preference_state == 0
                and self.agents[j].stance > 0
                and self.agents[j].id != source.id
            ]
            for target in targets:
                target.infection_impact(
                    period=period,
                    diffusion_events=self.diffusion_events,
                    message_events=self.message_events,
                    source_agent=source,
                )

        self.period += 1
        return self.summary()

    def run(self, n_periods: int) -> list[dict]:
        summaries = []
        for _ in range(n_periods):
            summaries.append(self.step())
        return summaries

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def stances(self) -> np.ndarray:
        return np.array([a.stance for a in self.agents])

    def summary(self) -> dict:
        stances = self.stances()
        return {
            "period": self.period,
            "mean_stance": float(stances.mean()),
            "std_stance": float(stances.std()),
            "n_adopted": int(sum(a.preference_state == 1 for a in self.agents)),
            "n_positive": int((stances > 0).sum()),
            "n_events": len(self.diffusion_events),
        }

    def save_events(self, output_dir: str | Path | None = None) -> None:
        out = Path(output_dir or self.scenario.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "diffusion_events.json").write_text(
            json.dumps(self.diffusion_events, indent=2)
        )
        (out / "message_events.json").write_text(
            json.dumps(self.message_events, indent=2)
        )


def generate_profiles(n_users: int = 50) -> list[str]:
    """Generate and parse persona profiles via the LLM (Prompt 1)."""
    raw = LLM_module.generate_profile(n_users=n_users)
    profiles = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip optional "User ID 15:" prefixes
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        if line:
            profiles.append(line)
    if not profiles:
        raise ValueError("LLM returned no parsable profiles")
    return profiles
