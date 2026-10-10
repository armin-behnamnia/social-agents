"""Stage-1 elite-persuader RL environment (approach 2).

One learned persuader operates inside a population of frozen LLM
agents (single-agent RL; the crowd is part of the environment).

Per period, the persuader takes a COMPOUND action:

  action = (target_node, strategy_index)

The target node is selected from the currently influenceable set
(positive-stance, non-adopted neighbors of already-active sources) —
this is LEARNED, replacing the frozen highest-degree heuristic, and is
the operationalization of "post content to target nodes in order to
influence the whole network". A frozen LLM renders the strategy into a
message delivered to that target. The reward is the realized latent
stance shift of the addressed targets (simulator ground truth), plus
small shaping terms and an optional cascade term for network-wide
movement.

Legacy mode: `target_selection != "learned"` keeps the original
frozen-heuristic behavior with Discrete(N_STRATEGIES) actions.

Observation modes:
- "matrix" (default, learnable): (agent_num, 14) per-agent feature
  matrix — profile-embedding pool (8d), degree centrality,
  betweenness, exposure, stance proxy, is-candidate, is-adopted.
  Combined with a pointer/attention policy this makes BOTH the target
  choice and the strategy choice learnable (the pooled 13-d summary
  cannot distinguish candidates, so target selection was unlearnable).
- "summary" (legacy): 13-d mean-pooled vector.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from . import LLM_module
from .profiles import generate_synthetic_profiles
from .scenario import LLMScenario
from .simulation import Simulation
from .strategy import N_STRATEGIES, index_to_strategy

_PROFILE_EMB_DIM = 8
OBS_DIM = _PROFILE_EMB_DIM + 5
# matrix-mode per-agent features: emb(8) + deg + betw + exposure +
# stance proxy + is_candidate + is_adopted
PER_AGENT_DIM = _PROFILE_EMB_DIM + 6


class ElitePersuaderEnv:
    """Gymnasium-style env (duck-typed) for one elite persuader.

    target_selection="learned" exposes a compound Discrete action:
        index = target_position * N_STRATEGIES + strategy_index
    over the CURRENT candidate list (per-step action mask in `info`).
    Other modes keep frozen heuristic selection and plain
    Discrete(N_STRATEGIES) actions (legacy behavior).
    """

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
        target_selection: str = "learned",
        synthetic_profiles: bool = True,
        profile_seed: int | None = 1234,
        persuade_all_neighbors: bool = False,
        obs_mode: str = "matrix",
        persuader_persuasiveness: bool = True,
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
        self.synthetic_profiles = synthetic_profiles
        self.profile_seed = profile_seed
        # If True, the rendered message is delivered to the chosen
        # target only (surgical targeting). If False, it becomes the
        # persuader's broadcast post seen by all its neighbors.
        self.persuade_all_neighbors = persuade_all_neighbors
        self.obs_mode = obs_mode
        # LLM-judged persuasiveness multiplies the influence prob of
        # the LEARNED persuader's messages only (crowd agents keep the
        # paper's original dynamics). Gives the RL policy a genuine
        # strategy-dependent learning signal.
        self.persuader_persuasiveness = persuader_persuasiveness
        self._persuasiveness_cache: dict[int, float] = {}

        self._learned_targets = target_selection == "learned"
        n_actions = (
            self.scenario.agent_num * N_STRATEGIES
            if self._learned_targets
            else N_STRATEGIES
        )
        self.action_space = spaces.Discrete(n_actions)
        if self.obs_mode == "matrix":
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.scenario.agent_num, PER_AGENT_DIM),
                dtype=np.float32,
            )
        else:
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
        profiles = None
        if self.synthetic_profiles:
            profiles = generate_synthetic_profiles(
                scenario.agent_num,
                seed=(
                    self.profile_seed
                    if self.profile_seed is not None
                    else self._episode_seed
                ),
            )
        return Simulation(
            scenario=scenario,
            topology=self.topology,
            initial_stances=None,
            profiles=profiles,
        )

    def _candidate_targets(self):
        """Influenceable targets this period.

        Learned mode: eligible agents (positive stance, non-adopted,
        non-seed) that are adjacent to at least one active propagator —
        i.e., the persuader can actually reach them this period. When
        none are reachable (isolated seed), falls back to all eligible
        agents so the episode continues.
        """
        assert self._sim is not None
        eligible = [
            a
            for a in self._sim.agents
            if a.id != 0 and a.preference_state == 0 and a.stance > 0
        ]
        if not eligible:
            return []
        if not self._learned_targets:
            if self.target_selection == "random":
                return eligible
            if self.target_selection == "highest_degree":
                eligible.sort(key=lambda a: a.degree, reverse=True)
            elif self.target_selection == "highest_betweenness":
                eligible.sort(
                    key=lambda a: a.betweenness_centrality, reverse=True
                )
            return eligible[: self.candidate_targets_per_period]

        # learned: reachable = neighbors of active sources
        active_ids = {
            a.id for a in self._sim.agents if a.preference_state == 1
        }
        reachable = [
            a for a in eligible if a.neighbors & active_ids
        ]
        if reachable:
            # stable order: by id so action indices are meaningful
            return sorted(reachable, key=lambda a: a.id)
        return sorted(eligible, key=lambda a: a.id)

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
        if self.obs_mode == "matrix":
            return self._observation_matrix()
        return self._observation_summary()

    def _observation_matrix(self) -> np.ndarray:
        """(n_agents, PER_AGENT_DIM) per-agent feature matrix.

        Row order = agent id, so action index pos k refers to row k of
        this matrix: the policy can attribute credit to the exact node
        it selected. is_candidate marks the currently reachable set.
        """
        targets = self._candidate_targets()
        candidate_ids = {a.id for a in targets}
        rows = []
        for a in self._sim.agents:
            if a.embedding is not None:
                emb = np.asarray(a.embedding, dtype=np.float32)
                pooled = emb.reshape(_PROFILE_EMB_DIM, -1).mean(axis=1)
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled = pooled / norm
            else:
                pooled = np.zeros(_PROFILE_EMB_DIM, dtype=np.float32)
            feats = [
                a.degree_centrality,
                a.betweenness_centrality,
                min(a.exposure_count / 10.0, 1.0),
                a.support_probability,  # observable stance proxy
                1.0 if a.id in candidate_ids else 0.0,
                1.0 if a.preference_state == 1 else 0.0,
            ]
            rows.append(
                np.concatenate([pooled, np.asarray(feats, dtype=np.float32)])
            )
        return np.stack(rows).astype(np.float32)

    def _observation_summary(self) -> np.ndarray:
        targets = self._candidate_targets()
        if not targets:
            return np.zeros(OBS_DIM, dtype=np.float32)
        # mean over candidate features (legacy; cannot distinguish
        # candidates — kept for backward compatibility)
        feats = np.stack([self._target_features(a) for a in targets])
        return feats.mean(axis=0).astype(np.float32)

    def _action_mask(self, targets) -> np.ndarray:
        """Boolean mask over the compound action space."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        if not self._learned_targets:
            mask[:] = True
            return mask
        if not targets:
            return mask
        for agent in targets:
            base = agent.id * N_STRATEGIES
            mask[base : base + N_STRATEGIES] = True
        return mask

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def clone(self) -> "ElitePersuaderEnv":
        """Independent copy for thread-per-episode parallel rollouts.

        Shallow-copies the configuration; the internal Simulation is
        rebuilt on reset() so no mutable state is shared between
        threads.
        """
        import copy

        new = copy.copy(self)
        new._sim = None
        new._step_count = 0
        new._episode_seed = None
        new._prev_mean_stance = 0.0
        return new

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._episode_seed = seed if seed is not None else int(
            np.random.default_rng().integers(0, 2**31)
        )
        self._sim = self._make_sim()
        self._step_count = 0
        self._prev_mean_stance = float(self._sim.stances().mean())
        self._last_events_len = len(self._sim.diffusion_events)
        targets = self._candidate_targets()
        info = {"action_mask": self._action_mask(targets), "candidates": targets}
        return self._observation(), info

    def step(self, action):
        assert self._sim is not None, "call reset() first"
        sim = self._sim

        targets = self._candidate_targets()

        # 0. decode the compound action (learned mode) -------------------
        if self._learned_targets:
            action = int(action)
            agent_id, strategy_idx = divmod(action, N_STRATEGIES)
            chosen = self._sim.agents[agent_id]
            if chosen not in targets:
                # masked-out action: fall back to first candidate
                chosen = targets[0]
        else:
            strategy_idx = int(action)
            chosen = targets[0] if targets else None

        # 1. render the strategy into a message (frozen LLM decoder)
        spec = index_to_strategy(int(strategy_idx))
        target_profile = chosen.profile if chosen is not None else ""
        message = LLM_module.render_strategy_message(
            spec,
            target_profile=target_profile,
            topic=self.topic,
            stance_direction="opposing",
        )

        # 2. deliver the message from the persuader (seed agent 0)
        persuader = sim.agents[0]
        persuader.post = message
        persuader.post_embedding = persuader._get_embedding(message)
        if self.persuader_persuasiveness and chosen is not None:
            score = self._persuasiveness_cache.get((id(chosen), message))
            if score is None:
                try:
                    score = LLM_module.judge_persuasiveness(
                        message, chosen.profile
                    )
                    self._persuasiveness_cache[(id(chosen), message)] = score
                except Exception:  # noqa: BLE001 - judge is optional
                    score = 1.0
            persuader.message_persuasiveness = score if score else 1.0
        else:
            persuader.message_persuasiveness = 1.0

        # 3. advance one diffusion period
        events_before = len(sim.diffusion_events)
        if self.persuade_all_neighbors or chosen is None:
            # broadcast: default diffusion processes persuader's post
            sim.step()
        else:
            # surgical: inject the chosen target into the persuader's
            # address list for this period, then run the normal period
            # (all other active sources still operate)
            original_neighbors = persuader._neighbors
            persuader._neighbors = {chosen.id}
            try:
                sim.step()
            finally:
                persuader._neighbors = original_neighbors
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

        targets_after = self._candidate_targets()
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
            "chosen_target": chosen.id if chosen is not None else None,
            "action_mask": self._action_mask(targets_after),
            "candidates": targets_after,
        }
        return obs, float(reward), terminated, False, info
