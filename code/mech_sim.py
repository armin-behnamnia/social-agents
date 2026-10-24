"""Faithful ports of the colleague's two Melodie diffusion mechanisms
(ShaghayeghNajari-patch-3, merged under ``code/melodie/`` + root
``agent.py`` + ``code/environment.py``), re-implemented on our hardened
non-Melodie core so ``ElitePersuaderEnv`` can train RL persuaders on
them. Existing scenarios/mechanisms are untouched — this module is
purely additive and opt-in via ``ElitePersuaderEnv(mechanism=...)``.

mechanism="cascade"  (their `agents_infection_stance_aware`):
    Within-period BFS cascade. Period sources = adopted agents with
    stance <= 0. Every exposure pulls the target's latent stance toward
    the source (strength = 4-term influence prob x stance_influence_
    rate), then runs the LLM internalization chain (response message ->
    profile update -> new post); emergence = observable post/profile
    change; an emerged target spreads within the SAME period.

mechanism="exposure" (their `agents_infection`):
    Persistent cross-period source queue. No stance dynamics at all;
    an LLM accept/reject gate (`LLM_module.accept_message`) decides each
    exposure. Accepted exposures run the internalization chain; an
    emerged target adopts (preference_state=1) and becomes a source NEXT
    period iff its stance <= 0. Sources drop out of the queue when no
    uninfected neighbors remain.

DEVIATIONS from the original code (each marked MELFIX in the source):

- MELFIX 1: a source whose post embedding is computed lazily is NO
  LONGER skipped on first use (their `infection_stance_aware` /
  `infection` compute the embedding and then unconditionally
  `return False`, so every first-time source silently does nothing).
- MELFIX 2: emergence sets preference_state=1 in cascade mode too
  (their cascade never updates preference_state — the line is commented
  out — so their population counters and next-period seeding see zero
  adoptions; emergence and adoption are reconciled here).
- MELFIX 3: exactly one diffusion event per exposure attempt (their
  code logs zero-placeholder events on guard-skip paths and duplicates
  accept-path events).
- MELFIX 4: LLM/embedding backends are ours (deepseek API + jina /
  hash-fallback) instead of local Ollama llama3.1 + BERT; prompts are
  the colleague's verbatim (`LLM_module.accept_message`,
  `generate_response_message`, `update_profile`, `generate_information`).
- MELFIX 5: `generate_information` keeps our 2-arg signature (no
  old_post parameter).
- Their `Selective`/`Generative` scenario_type post-selection is out of
  scope: the RL policy renders the persuader's post each period, which
  is exactly the realization of their unimplemented "RL_Generative"
  scenario hint.
- RL hooks kept live (deviation from their code by design): the
  persuader's `message_persuasiveness` multiplier applies inside
  `calculate_influence_prob`, and the Track-B goal-delta injection
  (s_eff = s_src·(2a-1)) applies to cascade stance updates, exactly as
  in paper mode. NOTE: in exposure mode there is NO stance update, so
  goal-delta has nothing to act on — use judge/cascade rewards there.

Population-init difference worth choosing deliberately (plumbed via
``ElitePersuaderEnv(mech_beta_init=...)``): their crowd starts with
Beta-distributed stances (60% support ~ Beta(6,2), 40% oppose ~
-Beta(8,2)) while ours starts uniformly at +0.65. Exposure mode needs
beta init (or another source-stamping rule) to spread at all: its
next-period queue only admits adopters with stance <= 0, and stances
never move — with an all-positive crowd the cascade provably stalls
after the persuaders' direct neighbors (covered by tests).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from . import LLM_module
from .simulation import Simulation

logger = logging.getLogger(__name__)

# Their scenario constants (code/melodie/scenario.py), for reference /
# beta init defaults:
BETA_SUPPORT_RATIO = 0.60
BETA_SUPPORT = (6.0, 2.0)
BETA_OPPOSE = (8.0, 2.0)


def beta_stance_init(
    n: int,
    seed: int | None = None,
    support_ratio: float = BETA_SUPPORT_RATIO,
    support_alpha: float = BETA_SUPPORT[0],
    support_beta: float = BETA_SUPPORT[1],
    oppose_alpha: float = BETA_OPPOSE[0],
    oppose_beta: float = BETA_OPPOSE[1],
) -> np.ndarray:
    """Their `stance.generate_stances`: support in +Beta, oppose in
    -Beta, shuffled. Seeded RNG added for reproducibility (theirs draws
    from the global numpy RNG)."""
    rng = np.random.default_rng(seed)
    n_support = int(round(n * support_ratio))
    n_oppose = n - n_support
    support = rng.beta(support_alpha, support_beta, n_support)
    oppose = -rng.beta(oppose_alpha, oppose_beta, n_oppose)
    stances = np.concatenate([support, oppose])
    rng.shuffle(stances)
    return stances.astype(float)


class MechanismSim(Simulation):
    """Simulation running one of the two imported mechanisms.

    mechanism: "cascade" (stance-aware BFS) or "exposure" (persistent
    queue + LLM accept gate). target_positive_only: restrict targets to
    positive-stance agents (our paper-mode rule; the original code does
    NOT restrict targets by stance — default False = faithful).
    """

    def __init__(
        self,
        *args,
        mechanism: str = "cascade",
        target_positive_only: bool = False,
        **kwargs,
    ):
        if mechanism not in ("cascade", "exposure"):
            raise ValueError(f"unknown mechanism {mechanism!r}")
        self.mechanism = mechanism
        self.target_positive_only = target_positive_only
        # exposure mode: persistent source queue across periods
        self._exposure_queue: deque[int] | None = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # eligibility
    # ------------------------------------------------------------------

    @staticmethod
    def _is_source(agent) -> bool:
        """Their rule (both mechanisms): adopted AND stance <= 0."""
        return agent.preference_state == 1 and agent.stance <= 0

    def _eligible_target(self, source, target, blocked: set[int]) -> bool:
        if target.id == source.id or target.id in blocked:
            return False
        if target.preference_state == 1:
            return False
        if self.target_positive_only and target.stance <= 0:
            return False
        return True

    # ------------------------------------------------------------------
    # Simulation API
    # ------------------------------------------------------------------

    def step(self) -> dict:
        if self.mechanism == "cascade":
            return self._step_cascade()
        return self._step_exposure()

    # ------------------------------------------------------------------
    # mechanism 1: stance-aware within-period BFS cascade
    # ------------------------------------------------------------------

    def _step_cascade(self) -> dict:
        period = self.period
        for agent in self.agents:
            agent.new_adopter = 0

        queue = deque(a.id for a in self.agents if self._is_source(a))
        processed: set[int] = set()
        while queue:
            source_id = queue.popleft()
            if source_id in processed:
                continue
            processed.add(source_id)
            source = self.agents[source_id]
            for tid in sorted(source.neighbors):
                target = self.agents[tid]
                if not self._eligible_target(source, target, processed):
                    continue
                target.exposure_count += 1
                if self._expose_cascade(source, target, period):
                    # emerged targets spread within the SAME period (BFS)
                    queue.append(target.id)

        self.period += 1
        return self.summary()

    def _expose_cascade(self, source, target, period: int) -> bool:
        scenario = self.scenario
        if not str(source.post or "").strip():
            return False
        if source.post_embedding is None:
            source.post_embedding = source._get_embedding(source.post)
        if source.post_embedding is None:
            return False  # MELFIX 1: only skip when genuinely unavailable
        if target.embedding is None and target.profile:
            target.embedding = target._get_embedding(target.profile)
            target.profile_embedding = target.embedding

        influence_prob = float(
            np.clip(target.calculate_influence_prob(source), 0.0, 1.0)
        )

        # Latent stance pull toward the source (their formula; the RL
        # goal-delta hook mirrors paper mode exactly).
        before = after = float(target.stance)
        if scenario.stance_influence_rate > 0:
            goal_align = (
                getattr(source, "message_goal_align", None)
                if scenario.goal_delta
                else None
            )
            before, after = target.update_agent_stance(
                source_agent=source,
                source_influence_strength=influence_prob,
                target_susceptibility_rate=scenario.stance_influence_rate,
                goal_align=goal_align,
            )
        stance_delta = after - before

        # LLM internalization chain (their prompts, our backends).
        internalized = LLM_module.generate_response_message(
            source.post, target.profile, abs(stance_delta)
        )
        target.accept = internalized
        target.accept_embedding = target._get_embedding(internalized)
        new_profile = LLM_module.update_profile(
            received=internalized, profile=target.profile
        )
        generated_post = LLM_module.generate_information(
            received=internalized, profile=target.profile
        )  # MELFIX 5: no old_post arg

        old_post, old_profile = target.post, target.profile
        if not generated_post or not str(generated_post).strip():
            generated_post = old_post
        if new_profile is None:
            new_profile = old_profile

        # message lineage
        target.parent_message_id = source.message_id
        target.original_message_id = (
            source.original_message_id or source.message_id
        )
        target.message_id = target.generate_message_id()

        # emergence = observable change (their rule, both mechanisms)
        post_changed = str(generated_post).strip() != str(old_post or "").strip()
        profile_changed = new_profile != old_profile
        emerged = bool(post_changed or profile_changed)

        target.received_message_history.append(
            {"period": period, "source": source.id, "text": source.post}
        )
        target.behavior_history.append(
            {
                "period": period,
                "source": source.id,
                "behavior": "emerge" if emerged else "no_emerge",
                "influence_probability": influence_prob,
                "stance_delta": stance_delta,
            }
        )

        # Faithful: observable state mutates on every exposure.
        target.post = str(generated_post).strip()
        target.post_embedding = target._get_embedding(target.post)
        target.profile = new_profile
        target.profile_embedding = target._get_embedding(str(new_profile))

        if emerged:
            target.preference_state = 1  # MELFIX 2
            target.influencer = source.id
            target.adoption_time = period
            target.new_adopter = 1
            target.generated_message_history.append(
                {"period": period, "target": None, "text": target.post}
            )

        self._log_event(
            source, target, period, influence_prob, before, after,
            "emerge" if emerged else "no_emerge",
        )
        return emerged

    # ------------------------------------------------------------------
    # mechanism 2: exposure/acceptance diffusion (persistent queue)
    # ------------------------------------------------------------------

    def _step_exposure(self) -> dict:
        period = self.period
        for agent in self.agents:
            agent.new_adopter = 0

        if self._exposure_queue is None:
            self._exposure_queue = deque(
                a.id for a in self.agents if self._is_source(a)
            )

        infected_at_start = {
            a.id for a in self.agents if a.preference_state == 1
        }
        current_sources = list(self._exposure_queue)
        self._exposure_queue.clear()

        for sid in current_sources:
            source = self.agents[sid]
            if not self._is_source(source):
                continue
            for tid in sorted(source.neighbors):
                if tid in infected_at_start or tid == sid:
                    continue
                target = self.agents[tid]
                if self.target_positive_only and target.stance <= 0:
                    continue
                target.exposure_count += 1
                self._expose_exposure(source, target, period)

        # Rebuild the queue for the next period: surviving sources with
        # uninfected neighbors + newly adopted agents with stance <= 0.
        queued: set[int] = set()
        for sid in current_sources:
            source = self.agents[sid]
            if not self._is_source(source):
                continue
            has_uninfected = any(
                self.agents[t].preference_state != 1
                for t in source.neighbors
            )
            if has_uninfected and sid not in queued:
                self._exposure_queue.append(sid)
                queued.add(sid)
        for agent in self.agents:
            if (
                agent.id not in infected_at_start
                and self._is_source(agent)
                and agent.id not in queued
            ):
                self._exposure_queue.append(agent.id)
                queued.add(agent.id)

        self.period += 1
        return self.summary()

    def _expose_exposure(self, source, target, period: int) -> bool:
        if not str(source.post or "").strip():
            return False
        if source.post_embedding is None:
            source.post_embedding = source._get_embedding(source.post)
        if source.post_embedding is None:
            return False  # MELFIX 1
        if target.embedding is None and target.profile:
            target.embedding = target._get_embedding(target.profile)
            target.profile_embedding = target.embedding

        before = after = float(target.stance)  # FROZEN: no stance update
        influence_prob = float(
            np.clip(target.calculate_influence_prob(source), 0.0, 1.0)
        )  # logged for analysis; the gate is the LLM, not this value

        target.received_message_history.append(
            {"period": period, "source": source.id, "text": source.post}
        )
        decision = LLM_module.accept_message(
            source.post, target.profile, target.received_message_history
        )
        accepted = decision == "ACCEPT"
        emerged = False

        if accepted:
            new_profile = LLM_module.update_profile(
                received=source.post, profile=target.profile
            )
            generated_post = LLM_module.generate_information(
                received=source.post, profile=target.profile
            )
            if not generated_post or not str(generated_post).strip():
                generated_post = target.post
            if new_profile is None:
                new_profile = target.profile
            old_post, old_profile = target.post, target.profile
            emerged = bool(
                str(generated_post).strip() != str(old_post or "").strip()
                or new_profile != old_profile
            )
            if emerged:
                target.post = str(generated_post).strip()
                target.post_embedding = target._get_embedding(target.post)
                target.profile = new_profile
                target.profile_embedding = target._get_embedding(
                    str(new_profile)
                )
                target.preference_state = 1
                target.influencer = source.id
                target.adoption_time = period
                target.new_adopter = 1
                target.parent_message_id = source.message_id
                target.original_message_id = (
                    source.original_message_id or source.message_id
                )
                target.message_id = target.generate_message_id()
                target.generated_message_history.append(
                    {"period": period, "target": None, "text": target.post}
                )

        target.behavior_history.append(
            {
                "period": period,
                "source": source.id,
                "behavior": "accept" if accepted else "reject",
                "emerged": emerged,
                "influence_probability": influence_prob,
            }
        )
        self._log_event(
            source, target, period, influence_prob, before, after,
            "accept" if accepted else "reject",
        )
        return emerged

    # ------------------------------------------------------------------
    # shared event schema (identical to paper mode -> env reward + the
    # fit_alphas/audit tooling work unchanged)
    # ------------------------------------------------------------------

    def _log_event(
        self, source, target, period, influence_prob, before, after, behavior
    ) -> None:  # MELFIX 3: exactly one event per exposure
        event = {
            "period": period,
            "source_agent": source.id,
            "target_agent": target.id,
            "target_stance_before": before,
            "target_stance_after": after,
            "target_delta_stance": after - before,
            "source_degree_centrality": source.degree_centrality,
            "source_betweenness_centrality": source.betweenness_centrality,
            "target_degree_centrality": target.degree_centrality,
            "target_betweenness_centrality": target.betweenness_centrality,
            "influence_probability": influence_prob,
            "interaction": "exposure",
            "behavior": behavior,
            "adopted": int(target.preference_state == 1),
            "mechanism": self.mechanism,
        }
        terms = getattr(target, "_last_influence_terms", None)
        if terms:
            event.update(terms)
            event["persuasiveness"] = getattr(
                target, "_last_persuasiveness", 1.0
            )
        if target._last_goal_align is not None:
            event["goal_align"] = float(target._last_goal_align)
        self.diffusion_events.append(event)
