"""Stage-1 elite-persuader RL environment (approach 2).

One learned persuader operates inside a population of frozen LLM
agents (single-agent RL; the crowd is part of the environment).

Per period, the persuader selects a persuasion strategy
(Discrete index over the strategy space in strategy.py) for the
candidate target presented in the observation. A frozen LLM renders
the strategy into a message delivered by the persuader's agent. The
reward is the realized latent stance shift of the addressed targets
(simulator ground truth), plus small shaping terms.

Observation vector (per step), for the candidate target j:
  [ target_profile_embedding_pca (8d),
    target_degree_centrality (1d),
    target_betweenness_centrality (1d),
    target_exposure_count_norm (1d),
    target_last_inferred_stance_proxy (1d),
    persuader_mean_stance_signal (1d) ]
13 dimensions total; deliberately compact for CPU training.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from . import LLM_module
from .scenario import LLMScenario
from .simulation import Simulation
from .strategy import N_STRATEGIES, index_to_strategy

_PROFILE_EMB_DIM = 8
OBS_DIM = _PROFILE_EMB_DIM + 5


class ElitePersuaderEnv:
    """Gymnasium-style env (duck-typed) for one elite persuader."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: LLMScenario | None = None,
        n_periods: int = 10,
        topology: str = "watts_strogatz",
        candidate_targets_per_period: int = 3,
        reward_cascade_discount: float = 0.0,
        message_cost: float = 0.0,
        shaping_accept_bonus: float = 0.05,
        topic: str = "the policy",
        target_selection: str = "highest_degree",
    ):
        self.scenario = scenario or LLMScenario()
        self.n_periods = n_periods
        self.topology = topology
        self.candidate_targets_per_period = candidate_targets_per_period
        self.reward_cascade_discount = reward_cascade_discount
        self.message_cost = message_cost
        self.shaping_accept_bonus = shaping_accept_bonus
        self.topic = topic
        self.target_selection = target_selection

        self.action_space = spaces.Discrete(N_STRATEGIES)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._sim: Simulation | None = None
        self._step_count = 0
        self._episode_seed: int | None = None
        self._prev_mean_stance: float = 0.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_sim(self) -> Simulation:
        scenario = LLMScenario(**vars(self.scenario))
        scenario.rng_seed = self._episode_seed
        return Simulation(
            scenario=scenario, topology=self.topology, initial_stances=None
        )

    def _candidate_targets(self):
        """Frozen heuristic target selection (not learned)."""
        assert self._sim is not None
        eligible = [
            a
            for a in self._sim.agents
            if a.id != 0 and a.preference_state == 0 and a.stance > 0
        ]
        if not eligible:
            return []
        if self.target_selection == "random":
            return eligible
        if self.target_selection == "highest_degree":
            eligible.sort(key=lambda a: a.degree, reverse=True)
        elif self.target_selection == "highest_betweenness":
            eligible.sort(key=lambda a: a.betweenness_centrality, reverse=True)
        return eligible[: self.candidate_targets_per_period]

    def _target_features(self, agent) -> np.ndarray:
        # deterministic compression of profile embedding: mean-pool to 8d
        if agent.embedding is not None:
            emb = np.asarray(agent.embedding, dtype=np.float32)
            pooled = emb.reshape(_PROFILE_EMB_DIM, -1).mean(axis=1)
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
        else:
            pooled = np.zeros(_PROFILE_EMB_DIM, dtype=np.float32)

        features = [
            agent.degree_centrality,
            agent.betweenness_centrality,
            min(agent.exposure_count / 10.0, 1.0),
            agent.support_probability,  # observable proxy: derived label
            self._prev_mean_stance,
        ]
        return np.concatenate([pooled, np.asarray(features, dtype=np.float32)])

    def _observation(self) -> np.ndarray:
        assert self._sim is not None
        targets = self._candidate_targets()
        if not targets:
            return np.zeros(OBS_DIM, dtype=np.float32)
        # mean over candidate features (the policy conditions on the
        # average target context; a per-candidate head is a future
        # extension with a learned selector)
        feats = np.stack([self._target_features(a) for a in targets])
        return feats.mean(axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._episode_seed = seed if seed is not None else int(
            np.random.default_rng().integers(0, 2**31)
        )
        self._sim = self._make_sim()
        self._step_count = 0
        self._prev_mean_stance = float(self._sim.stances().mean())
        self._last_events_len = len(self._sim.diffusion_events)
        return self._observation(), {}

    def step(self, action):
        assert self._sim is not None, "call reset() first"
        sim = self._sim

        # 1. render the strategy into a message (frozen LLM decoder)
        spec = index_to_strategy(int(action))
        message = LLM_module.render_strategy_message(
            spec,
            target_profile="; ".join(
                a.profile[:80] for a in self._candidate_targets()
            ),
            topic=self.topic,
            stance_direction="opposing",
        )

        # 2. deliver the message from the persuader (seed agent 0)
        persuader = sim.agents[0]
        persuader.post = message
        persuader.post_embedding = persuader._get_embedding(message)

        # 3. advance one diffusion period
        events_before = len(sim.diffusion_events)
        sim.step()
        self._step_count += 1

        # 4. compute reward from ground truth
        events = sim.diffusion_events[events_before:]
        reward = 0.0
        n_accept = 0
        for event in events:
            if event["source_agent"] == 0:
                delta = event["target_delta_stance"]
                reward += -delta  # negative delta = shift toward persuader
                if event["behavior"] == "accept":
                    n_accept += 1
        reward += self.shaping_accept_bonus * n_accept
        reward -= self.message_cost

        # optional cascade credit: discounted population movement
        if self.reward_cascade_discount > 0:
            mean_stance = float(sim.stances().mean())
            reward += self.reward_cascade_discount * (
                self._prev_mean_stance - mean_stance
            )
            self._prev_mean_stance = mean_stance

        obs = self._observation()
        terminated = self._step_count >= self.n_periods
        info = {
            "period": sim.period,
            "mean_stance": float(sim.stances().mean()),
            "n_adopted": int(sum(a.preference_state == 1 for a in sim.agents)),
            "n_persuader_events": len(
                [e for e in events if e["source_agent"] == 0]
            ),
            "n_accept": n_accept,
            "message": message,
            "strategy": spec,
        }
        return obs, float(reward), terminated, False, info
