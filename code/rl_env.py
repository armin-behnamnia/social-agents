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
- "matrix" (default, learnable): (agent_num, 16) per-agent feature
  matrix — profile-embedding pool (8d), degree centrality,
  betweenness, exposure, stance proxy, is-candidate, is-adopted,
  is-persuader, is-acting. Combined with the compound action this
  makes BOTH the target choice and the strategy choice learnable.
- "summary" (legacy): 13-d mean-pooled vector.
- "pair" (for target_selection="random_neighbor"): 26-d vector of
  [acting-persuader features (13d) ‖ current random-neighbor target
  features (13d)]; the policy only chooses the strategy.

Target selection modes:
- "learned": compound action (target x strategy), action mask.
- "random_neighbor": the env draws a random eligible neighbor of each
  persuader at period start; the action is only the strategy index
  (Discrete(N_STRATEGIES)). Matches the protocol: each agent talks to
  one of its few neighbors decided by chance, content by policy.
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
# stance proxy + is_candidate + is_adopted + is_persuader + is_acting
PER_AGENT_DIM = _PROFILE_EMB_DIM + 8


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
        n_persuaders: int = 1,
        judge_fn=None,
        mechanism: str = "paper",
        mech_beta_init: bool = False,
        mech_target_positive_only: bool = False,
        reward_adoption_bonus: float = 0.0,
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
        # RL persuaders are seeded agents 0..n_persuaders-1; the rest of
        # the population is the frozen crowd. One SHARED policy acts for
        # all of them (parameter sharing): each period every persuader
        # picks its own (target x strategy) — see step().
        if n_persuaders < 1 or n_persuaders >= self.scenario.agent_num:
            raise ValueError(
                f"n_persuaders={n_persuaders} must be in "
                f"[1, {self.scenario.agent_num})"
            )
        self.n_persuaders = int(n_persuaders)
        self.persuader_ids = tuple(range(self.n_persuaders))
        self._persuader_index = {pid: i for i, pid in enumerate(self.persuader_ids)}
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
        # Optional reward model (e.g. a Bradley-Terry pairwise judge).
        # Preferred call is judge_fn.lookup_by_id(strategy_index, profile_id)
        # — a drift-proof identity lookup; a None return falls back to the
        # content call judge_fn(message, profile), then to the absolute LLM
        # judge. Hit/fallback counts surface through ``judge_src`` on the
        # action records so a silent fallback can never go unnoticed.
        self._judge_fn = judge_fn
        self._persuasiveness_cache: dict[int, float] = {}
        self._id_cache: dict[tuple[int, int], float] = {}
        self._judge_hits = 0
        self._judge_fallbacks = 0

        # Imported colleague mechanisms (code/mech_sim.py): "paper" keeps
        # the original Simulation; "cascade"/"exposure" run the two
        # Melodie-ported propagation regimes. Purely additive opt-in.
        if mechanism not in ("paper", "cascade", "exposure"):
            raise ValueError(f"unknown mechanism {mechanism!r}")
        self.mechanism = mechanism
        self.mech_beta_init = bool(mech_beta_init)
        self.mech_target_positive_only = bool(mech_target_positive_only)
        # Flat reward per persuader-sourced adoption. Essential in
        # exposure mode (no stance deltas there); optional elsewhere.
        self.reward_adoption_bonus = float(reward_adoption_bonus)

        # Track B: goal-alignment a(m) for rendered messages (optional).
        from .goal_align import get_align_fn

        self._goal_align_fn = get_align_fn(
            getattr(self.scenario, "goal_align", "none"),
            getattr(self.scenario, "goal_proposition", ""),
            getattr(self.scenario, "goal_anti_proposition", ""),
        )
        self._goal_reward_lambda = float(
            getattr(self.scenario, "goal_reward_lambda", 0.0) or 0.0
        )

        self._learned_targets = target_selection == "learned"
        self._random_neighbor = target_selection == "random_neighbor"
        n_actions = (
            self.scenario.agent_num * N_STRATEGIES
            if self._learned_targets
            else N_STRATEGIES
        )
        self.action_space = spaces.Discrete(n_actions)
        # strategy-only obs for random_neighbor mode: self ‖ target
        self._pair_dim = 2 * OBS_DIM
        if obs_mode == "pair" and not self._random_neighbor:
            raise ValueError(
                'obs_mode="pair" requires target_selection='
                '"random_neighbor"'
            )
        if obs_mode == "pair":
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._pair_dim,),
                dtype=np.float32,
            )
        elif self.obs_mode == "matrix":
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
        self._targets_rng = np.random.default_rng()
        # current random-neighbor assignment (per persuader index)
        self._pending_targets: list = []

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
        if self.mechanism != "paper":
            from .mech_sim import MechanismSim, beta_stance_init

            initial_stances = None
            if self.mech_beta_init:
                # Their population: 60% support ~Beta(6,2), 40% oppose
                # ~-Beta(8,2); the persuader seeds are still stamped -1
                # by _seed_agents afterwards.
                initial_stances = beta_stance_init(
                    scenario.agent_num, seed=scenario.rng_seed
                )
            return MechanismSim(
                scenario=scenario,
                topology=self.topology,
                initial_stances=initial_stances,
                profiles=profiles,
                seed_ids=self.persuader_ids,
                mechanism=self.mechanism,
                target_positive_only=self.mech_target_positive_only,
            )
        return Simulation(
            scenario=scenario,
            topology=self.topology,
            initial_stances=None,
            profiles=profiles,
            seed_ids=self.persuader_ids,
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
        # Mechanism modes influence any non-adopted neighbor (their code
        # has no positivity gate); paper mode keeps its positive-stance
        # rule. `mech_target_positive_only` restores the gate if wanted.
        any_stance = (
            self.mechanism != "paper" and not self.mech_target_positive_only
        )
        eligible = [
            a
            for a in self._sim.agents
            if a.id not in self.persuader_ids
            and a.preference_state == 0
            and (a.stance > 0 or any_stance)
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

    def _draw_pending_targets(self) -> None:
        """random_neighbor mode: draw one eligible neighbor per
        persuader for the upcoming period (uniformly at random among
        its positive-stance, non-adopted, non-persuader neighbors).
        Falls back to any eligible agent when the neighborhood has
        dried up, or to None (persuader idles) if nothing is eligible.
        """
        assert self._sim is not None
        self._pending_targets = []
        any_stance = (
            self.mechanism != "paper" and not self.mech_target_positive_only
        )
        eligible_all = [
            a for a in self._sim.agents
            if a.id not in self.persuader_ids
            and a.preference_state == 0
            and (a.stance > 0 or any_stance)
        ]
        for pid in self.persuader_ids:
            me = self._sim.agents[pid]
            options = [a for a in eligible_all if a.id in me.neighbors]
            if not options:
                options = eligible_all
            chosen = (
                options[int(self._targets_rng.integers(len(options)))]
                if options else None
            )
            self._pending_targets.append(chosen)

    def observation(self, actor: int | None = None) -> np.ndarray:
        """Public observation accessor.

        actor: index into self.persuader_ids of the persuader currently
        deciding; the returned obs marks/contains that persuader's own
        context (is_acting column in matrix mode; self‖target pair in
        pair mode) so the shared policy can condition each actor's
        choice on its own identity.
        """
        return self._observation(actor=actor)

    def _observation(self, actor: int | None = None) -> np.ndarray:
        assert self._sim is not None
        if self.obs_mode == "pair":
            return self._observation_pair(actor)
        if self.obs_mode == "matrix":
            return self._observation_matrix(actor=actor)
        return self._observation_summary()

    def _observation_pair(self, actor: int | None) -> np.ndarray:
        """random_neighbor mode: [persuader features ‖ target features].

        actor is REQUIRED (index into persuader_ids); the target half
        describes this period's randomly drawn neighbor. If the
        persuader idles (no pending target), the target half is zeros.
        """
        idx = actor if actor is not None else 0
        me = self._sim.agents[self.persuader_ids[idx]]
        target = (
            self._pending_targets[idx]
            if idx < len(self._pending_targets)
            else None
        )
        self_feats = self._target_features(me)
        tgt_feats = (
            self._target_features(target)
            if target is not None
            else np.zeros(OBS_DIM, dtype=np.float32)
        )
        return np.concatenate([self_feats, tgt_feats]).astype(np.float32)

    def _observation_matrix(self, actor: int | None = None) -> np.ndarray:
        """(n_agents, PER_AGENT_DIM) per-agent feature matrix.

        Row order = agent id, so action index pos k refers to row k of
        this matrix: the policy can attribute credit to the exact node
        it selected. is_candidate marks the currently reachable set;
        is_persuader / is_acting mark the RL-controlled seeds and the
        one currently deciding.
        """
        targets = self._candidate_targets()
        candidate_ids = {a.id for a in targets}
        acting_id = (
            self.persuader_ids[actor] if actor is not None else None
        )
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
                1.0 if a.id in self.persuader_ids else 0.0,
                1.0 if a.id == acting_id else 0.0,
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
        new._pending_targets = []
        return new

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._episode_seed = seed if seed is not None else int(
            np.random.default_rng().integers(0, 2**31)
        )
        self._sim = self._make_sim()
        self._step_count = 0
        self._prev_mean_stance = float(self._sim.stances().mean())
        self._last_events_len = len(self._sim.diffusion_events)
        self._targets_rng = np.random.default_rng(self._episode_seed)
        if self._random_neighbor:
            self._draw_pending_targets()
        targets = self._candidate_targets()
        info = {
            "action_mask": self._action_mask(targets),
            "candidates": targets,
            "mechanism": self.mechanism,
        }
        return self._observation(), info

    def _decode_action(self, action, targets, persuader_idx: int = 0):
        """Action -> (target agent, strategy index).

        learn: compound action. random_neighbor: strategy only; the
        target is this period's pending random neighbor.
        """
        if self._learned_targets:
            action = int(action)
            agent_id, strategy_idx = divmod(action, N_STRATEGIES)
            chosen = self._sim.agents[agent_id]
            if chosen not in targets:
                # masked-out action: fall back to first candidate
                chosen = targets[0]
        elif self._random_neighbor:
            strategy_idx = int(action)
            chosen = (
                self._pending_targets[persuader_idx]
                if persuader_idx < len(self._pending_targets)
                else None
            )
        else:
            strategy_idx = int(action)
            chosen = targets[0] if targets else None
        return chosen, int(strategy_idx)

    def _judge_score(self, message: str, target, strategy_idx: int | None = None) -> float:
        """Persuasiveness score for a delivered message.

        Preference order: (1) identity lookup keyed on
        ``(strategy_idx, target.profile_id)`` — drift-proof and the
        intended path; (2) content lookup keyed on
        ``(message, target.profile)`` — the legacy path; (3) the absolute
        LLM judge. Returns ``1.0`` when persuasiveness is disabled.

        Returns the score only; the caller reads ``judge_src`` from the
        action record to see which path fired.
        """
        if not self.persuader_persuasiveness or target is None:
            return 1.0

        pid = getattr(target, "profile_id", None)
        # 1. identity lookup (drift-proof) -------------------------------
        if (
            strategy_idx is not None
            and pid is not None
            and self._judge_fn is not None
            and hasattr(self._judge_fn, "lookup_by_id")
        ):
            ikey = (int(strategy_idx), int(pid))
            if ikey in self._id_cache:
                self._judge_hits += 1
                return self._id_cache[ikey]
            try:
                score = self._judge_fn.lookup_by_id(*ikey)
            except Exception:  # noqa: BLE001 - judge is optional
                score = None
            if score is not None:
                self._id_cache[ikey] = float(score)
                self._judge_hits += 1
                return float(score)
            # identity miss -> must fall through, and be counted
            self._judge_fallbacks += 1

        # 2. content lookup (legacy) -------------------------------------
        key = (message, target.profile)
        score = self._persuasiveness_cache.get(key)
        if score is None:
            try:
                if self._judge_fn is not None:
                    score = self._judge_fn(message, target.profile)
                if score is None:
                    score = LLM_module.judge_persuasiveness(
                        message, target.profile
                    )
            except Exception:  # noqa: BLE001 - judge is optional
                score = 1.0
            self._persuasiveness_cache[key] = score
        return 1.0 if score is None else float(score)

    def step(self, action):
        """Advance one diffusion period.

        action: with n_persuaders == 1 a single int (legacy); with
        n_persuaders > 1 a sequence of n_persuaders ints — one compound
        (target x strategy) action per persuader. All persuaders'
        messages are delivered simultaneously (one sim.step), so
        interactions within a period don't affect each other.

        Returns (obs, reward, terminated, truncated, info); reward is a
        float for one persuader or an np.ndarray of per-persuader
        rewards otherwise. The shared cascade term is added to every
        persuader's component (cooperative team reward).
        """
        assert self._sim is not None, "call reset() first"
        sim = self._sim
        single = self.n_persuaders == 1
        if single:
            actions = [int(action)]
        else:
            actions = [int(a) for a in action]
            if len(actions) != self.n_persuaders:
                raise ValueError(
                    f"expected {self.n_persuaders} actions, "
                    f"got {len(actions)}"
                )

        targets = self._candidate_targets()

        # 1. per persuader: decode action, render message, deliver -----
        action_records = []
        for pidx, (pid, act) in enumerate(zip(self.persuader_ids, actions)):
            persuader = sim.agents[pid]
            chosen, strategy_idx = self._decode_action(
                act, targets, persuader_idx=pidx
            )
            spec = index_to_strategy(strategy_idx)
            target_profile = chosen.profile if chosen is not None else ""
            message = LLM_module.render_strategy_message(
                spec,
                target_profile=target_profile,
                topic=self.topic,
                stance_direction="opposing",
            )
            persuader.post = message
            persuader.post_embedding = persuader._get_embedding(message)
            hits_before = self._judge_hits
            fallbacks_before = self._judge_fallbacks
            score = self._judge_score(message, chosen, strategy_idx)
            if self._judge_hits > hits_before:
                judge_src = "id"
            elif self._judge_fallbacks > fallbacks_before:
                judge_src = "text"
            else:
                judge_src = "none"
            persuader.message_persuasiveness = score
            goal_align = None
            if self._goal_align_fn is not None and message:
                goal_align = float(self._goal_align_fn(message))
                persuader.message_goal_align = goal_align
            else:
                # no render this period -> do not carry a stale a(m) into
                # the (persuader-only) goal-direction injection
                persuader.message_goal_align = None
            action_records.append(
                {
                    "persuader": pid,
                    "target": chosen.id if chosen is not None else None,
                    "strategy_index": strategy_idx,
                    "strategy": spec,
                    "message": message,
                    "persuasiveness": score,
                    "judge_src": judge_src,
                    "goal_align": goal_align,
                }
            )

        # 2. advance one diffusion period --------------------------------
        events_before = len(sim.diffusion_events)
        restrict = not self.persuade_all_neighbors
        original_neighbors = {}
        if restrict:
            # surgical: each persuader addresses only its chosen target
            # this period; all other active sources (crowd) still work
            for pid, rec in zip(self.persuader_ids, action_records):
                persuader = sim.agents[pid]
                original_neighbors[pid] = persuader._neighbors
                if rec["target"] is not None:
                    persuader._neighbors = {rec["target"]}
                elif self._random_neighbor:
                    # idle: no eligible neighbor this period -> the
                    # persuader's post reaches nobody
                    persuader._neighbors = set()
        try:
            sim.step()
        finally:
            for pid, nb in original_neighbors.items():
                sim.agents[pid]._neighbors = nb
        self._step_count += 1

        # 3. compute per-persuader rewards from ground truth ------------
        events = sim.diffusion_events[events_before:]
        rewards = np.zeros(self.n_persuaders, dtype=np.float64)
        n_accept = np.zeros(self.n_persuaders, dtype=int)
        n_adopt = np.zeros(self.n_persuaders, dtype=int)
        for event in events:
            src = event["source_agent"]
            if src in self._persuader_index:
                i = self._persuader_index[src]
                delta = event["target_delta_stance"]
                rewards[i] += -delta  # negative delta = toward persuader
                # "emerge" is the cascade mode's acceptance-equivalent
                # event; "accept" covers paper mode and exposure mode.
                if event["behavior"] in ("accept", "emerge"):
                    n_accept[i] += 1
                if event["behavior"] in ("accept", "emerge", "adopt") and event[
                    "adopted"
                ]:
                    n_adopt[i] += 1
        rewards += self.shaping_accept_bonus * n_accept
        rewards += self.reward_adoption_bonus * n_adopt
        rewards -= self.message_cost

        # optional goal-alignment shaping: λ·a(m) per delivered message
        if self._goal_reward_lambda and self._goal_align_fn is not None:
            for rec in action_records:
                i = self._persuader_index.get(rec["persuader"])
                if i is not None and rec.get("goal_align") is not None:
                    rewards[i] += self._goal_reward_lambda * rec["goal_align"]

        # optional cascade credit: discounted population movement,
        # shared by the whole persuader team (cooperative reward)
        if self.reward_cascade_discount > 0:
            mean_stance = float(sim.stances().mean())
            cascade = self.reward_cascade_discount * (
                self._prev_mean_stance - mean_stance
            )
            rewards += cascade
            self._prev_mean_stance = mean_stance

        targets_after = self._candidate_targets()
        terminated = self._step_count >= self.n_periods
        if self._random_neighbor and not terminated:
            self._draw_pending_targets()
        obs = self._observation()
        info = {
            "period": sim.period,
            "mechanism": self.mechanism,
            "mean_stance": float(sim.stances().mean()),
            "n_adopted": int(sum(a.preference_state == 1 for a in sim.agents)),
            "n_persuader_events": len(
                [e for e in events if e["source_agent"] in self._persuader_index]
            ),
            "n_accept": int(n_accept.sum()),
            "persuader_actions": action_records,
            "period_events": events,
            # legacy single-persuader fields
            "message": action_records[0]["message"],
            "strategy": action_records[0]["strategy"],
            "chosen_target": action_records[0]["target"],
            "action_mask": self._action_mask(targets_after),
            "candidates": targets_after,
        }
        out_reward = float(rewards[0]) if single else rewards
        return obs, out_reward, terminated, False, info
