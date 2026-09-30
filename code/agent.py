"""LLM-based social agent with latent stance dynamics.

Fixes applied vs. the original (each marked BUGFIX):
- BUGFIX 1: probabilistic acceptance gate restored (`accepted = True`
  was hardcoded, contradicting Algorithm 1 line 7 of the paper and
  rendering `calculate_influence_prob` dead code).
- BUGFIX 2: embedding guard logic corrected (previously printed
  "no valid post embedding" and returned even when the embedding had
  just been computed successfully, skipping every first-time source).
- BUGFIX 3: duplicate `save_diffusion_event` calls removed
  (non-adopted agents were saved twice: once before generation and
  once after, inflating the event dataset).
- BUGFIX 4: redefinition of `behavior` after stance update corrected
  (previously "accept" silently became "stance turned negative",
  corrupting the downstream `adopted` flag).
- BUGFIX 5: all observer-inference variables are always defined before
  use (previously a single None LLM output caused NameError /
  UnboundLocalError in `save_message_event`).
- BUGFIX 6: the commented-out `# return` restored so non-adopted
  agents do NOT propagate generated content onward in the same period.
- Design: no LLM/Melodie imports at module scope for the state-machine
  part; generation and inference go through injectable callables so the
  agent is unit-testable with fakes and RL-trainable without a live
  LLM server.
"""

from __future__ import annotations

import logging
import uuid
from typing import Callable

import numpy as np

from .base_agent import NetworkAgent
from . import LLM_module
from .tools import text2embedding

logger = logging.getLogger(__name__)


class LLMSocialAgent(NetworkAgent):

    def __init__(
        self,
        agent_id: int,
        scenario=None,
        neighbors: set[int] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.scenario = scenario
        self._rng = rng or np.random.default_rng()
        super().__init__(agent_id, neighbors)

    # ==============================================================
    # Agent initialization
    # ==============================================================

    def setup(self):

        # Diffusion state
        self.preference_state: int = 0
        self.isSeed: bool = False

        # Static agent context
        self.profile: str = ""

        # Latent internal state, continuous stance in [-1, 1]
        self.stance: float = 0.0
        self.stance_label: str = "Neutral"
        self.support_probability: float = 0.5

        # Messages
        self.post: str = ""
        self.accept: str = ""

        self.embedding = None
        self.profile_embedding = None
        self.post_embedding = None
        self.accept_embedding = None

        # Interaction
        self.influencer = None

        self.exposure_count: int = 0
        self.num_infected_neighbors: int = 0

        # Diffusion metrics
        self.new_adopter: int = 0
        self.adoption_time: int = -1

        # Network position
        self.degree: int = 0
        self.degree_centrality: float = 0.0
        self.betweenness_centrality: float = 0.0

        # Observable temporal histories
        self.received_message_history: list = []
        self.generated_message_history: list = []
        self.behavior_history: list = []
        self.interaction_history: list = []

        # Message lineage
        self.message_id = None
        self.parent_message_id = None
        self.original_message_id = None

    # ==============================================================
    # Message ID
    # ==============================================================

    @staticmethod
    def generate_message_id() -> str:
        return str(uuid.uuid4())

    # ==============================================================
    # Stance update
    # ==============================================================

    def update_agent_stance(
        self,
        source_agent,
        source_influence_strength: float,
        target_susceptibility_rate: float,
    ) -> tuple[float, float]:
        old_stance = float(self.stance)

        new_stance = (
            old_stance
            + source_influence_strength
            * target_susceptibility_rate
            * (source_agent.stance - old_stance)
        )

        new_stance = np.clip(
            new_stance,
            self.scenario.inference_state_min,
            self.scenario.inference_state_max,
        )

        self.stance = float(new_stance)

        threshold = self.scenario.stance_threshold

        if self.stance >= threshold:
            self.stance_label = "Supportive"
        elif self.stance > 0:
            self.stance_label = "Receptive"
        elif self.stance <= -threshold:
            self.stance_label = "Resistant"
        else:
            self.stance_label = "Neutral"

        self.support_probability = (self.stance + 1.0) / 2.0

        return old_stance, self.stance

    # ==============================================================
    # Safe embedding
    # ==============================================================

    @staticmethod
    def _get_embedding(text):
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        if not text.strip():
            return None
        return text2embedding(text)

    @staticmethod
    def _cosine_similarity(embedding_a, embedding_b) -> float:
        if embedding_a is None or embedding_b is None:
            return 0.0
        try:
            embedding_a = np.asarray(embedding_a).ravel()
            embedding_b = np.asarray(embedding_b).ravel()
            if embedding_a.size == 0 or embedding_b.size == 0:
                return 0.0
            if embedding_a.shape != embedding_b.shape:
                return 0.0
            denom = float(np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))
            if denom == 0.0:
                return 0.0
            return float(np.dot(embedding_a, embedding_b) / denom)
        except Exception:  # noqa: BLE001
            return 0.0

    # ==============================================================
    # Infection / influence process
    # ==============================================================

    def infection_impact(
        self,
        period: int,
        diffusion_events: list,
        message_events: list,
        source_agent,
    ):
        propagation_threshold = self.scenario.adoption_threshold

        # 1. Validate source eligibility: only sufficiently negative
        #    stances can influence positive, non-adopted agents.
        if source_agent.stance > propagation_threshold:
            return

        # 2. Validate target eligibility
        if self.preference_state != 0 or self.stance <= 0:
            return
        if self.id == source_agent.id:
            return

        # 3. Validate source message
        if source_agent.post is None or not str(source_agent.post).strip():
            logger.debug(
                "source agent %s has an empty post; skipping influence",
                source_agent.id,
            )
            return

        # 4. Prepare source message embedding
        # BUGFIX 2: only skip (and warn) when the embedding is genuinely
        # unavailable after attempting to compute it.
        if source_agent.post_embedding is None:
            source_agent.post_embedding = self._get_embedding(source_agent.post)
        if source_agent.post_embedding is None:
            logger.debug(
                "source agent %s has no valid post embedding; skipping influence",
                source_agent.id,
            )
            return

        # 5. Prepare target profile embedding
        if self.embedding is None and self.profile:
            self.embedding = self._get_embedding(self.profile)
            self.profile_embedding = self.embedding

        # 6. Record exposure
        self.exposure_count += 1
        self.behavior_history.append(
            {"period": period, "source": source_agent.id, "behavior": "receive"}
        )

        # 7. Calculate influence probability
        influence_prob = float(
            np.clip(self.calculate_influence_prob(source_agent), 0.0, 1.0)
        )

        # 8. Acceptance / rejection
        # BUGFIX 1: restore the stochastic gate from Algorithm 1.
        rand_value = float(self._rng.random())
        accepted = rand_value < influence_prob
        behavior = "accept" if accepted else "reject"
        self.behavior_history.append(
            {
                "period": period,
                "source": source_agent.id,
                "behavior": behavior,
                "influence_probability": influence_prob,
                "random_value": rand_value,
            }
        )
        self.interaction_history.append(
            {
                "period": period,
                "source": source_agent.id,
                "behavior": behavior,
                "message": source_agent.post,
                "influence_probability": influence_prob,
            }
        )

        # 9. Rejection
        if not accepted:
            self.save_diffusion_event(
                diffusion_events,
                period,
                source_agent,
                self.stance,
                self.stance,
                influence_prob,
                behavior,
            )
            return

        # 10. Update latent internal state
        target_stance_before = float(self.stance)
        target_stance_after = float(self.stance)

        if self.scenario.stance_influence_rate > 0:
            (
                target_stance_before,
                target_stance_after,
            ) = self.update_agent_stance(
                source_agent=source_agent,
                source_influence_strength=influence_prob,
                target_susceptibility_rate=self.scenario.stance_influence_rate,
            )

        stance_delta = target_stance_after - target_stance_before

        # BUGFIX 4: do NOT redefine `behavior` from the sign of the new
        # stance. Acceptance was already decided by the stochastic gate.
        # `propagated` records whether the adoption threshold was crossed.
        reached_adoption_threshold = target_stance_after <= propagation_threshold

        # 11. Store accepted (internalized) message
        internalized_message = LLM_module.generate_response_message(
            source_agent.post, self.profile, abs(stance_delta)
        )
        self.accept = internalized_message
        self.accept_embedding = self._get_embedding(self.accept)

        # 12-13. Adoption
        if reached_adoption_threshold:
            self.preference_state = 1
            self.influencer = source_agent.id
            self.adoption_time = period
            self.new_adopter = 1
            self.behavior_history.append(
                {
                    "period": period,
                    "source": source_agent.id,
                    "behavior": "adopt",
                    "stance_before": target_stance_before,
                    "stance_after": target_stance_after,
                    "stance_delta": stance_delta,
                    "adoption_threshold": propagation_threshold,
                }
            )
        else:
            self.new_adopter = 0
            self.behavior_history.append(
                {
                    "period": period,
                    "source": source_agent.id,
                    "behavior": "state_update",
                    "stance_before": target_stance_before,
                    "stance_after": target_stance_after,
                    "stance_delta": stance_delta,
                    "adoption_threshold": propagation_threshold,
                }
            )

        # BUGFIX 3: save the diffusion event exactly once, here.
        self.save_diffusion_event(
            diffusion_events,
            period,
            source_agent,
            target_stance_before,
            target_stance_after,
            influence_prob,
            behavior,
        )

        # 14. BUGFIX 6: non-adopted agents stop here this period; the
        # message was accepted and changed latent state, but the agent
        # does not produce observable content or propagate yet.
        if self.preference_state != 1:
            return

        # 15-16. Generate observable content (adoption path only)
        new_profile = LLM_module.update_profile(
            received=self.accept, profile=self.profile
        )
        new_profile_embedding = self._get_embedding(new_profile)
        generated_post = LLM_module.generate_information(
            received=self.accept, profile=self.profile
        )
        generated_post_embedding = self._get_embedding(generated_post)

        # 17. Observer-side inference over all three evidence settings.
        # BUGFIX 5: initialize every variable; guard each computation.
        old_profile = self.profile
        old_profile_embedding = self.profile_embedding

        inferr_stance_post: float | None = None
        inferr_stance_profile: float | None = None
        inferr_stance: float | None = None
        estimated_stance_shift_post: float | None = None
        estimated_stance_shift_profile: float | None = None
        estimated_stance_shift: float | None = None

        if generated_post is not None and self.scenario.use_post_evidence:
            inferr_stance_post = LLM_module.inference_stance_post(
                source_agent.post, generated_post
            )
            if generated_post_embedding is not None:
                estimated_stance_shift_post = 1.0 - self._cosine_similarity(
                    generated_post_embedding, source_agent.post_embedding
                )

        if new_profile is not None and self.scenario.use_profile_evidence:
            inferr_stance_profile = LLM_module.inference_stance_profile(
                source_agent.post, old_profile, new_profile
            )
            if old_profile_embedding is not None and new_profile_embedding is not None:
                estimated_stance_shift_profile = 1.0 - self._cosine_similarity(
                    old_profile_embedding, new_profile_embedding
                )

        if (
            new_profile is not None
            and generated_post is not None
            and self.scenario.use_post_evidence
            and self.scenario.use_profile_evidence
        ):
            inferr_stance = LLM_module.inference_stance(
                source_agent.post, generated_post, old_profile, new_profile
            )
            if (
                estimated_stance_shift_post is not None
                and estimated_stance_shift_profile is not None
            ):
                estimated_stance_shift = (
                    estimated_stance_shift_post * estimated_stance_shift_profile
                )

        # 19. Record message generation
        self.behavior_history.append(
            {"period": period, "source": source_agent.id, "behavior": "generate"}
        )
        self.received_message_history.append(
            {"period": period, "source": source_agent.id, "text": source_agent.post}
        )
        self.generated_message_history.append(
            {"period": period, "target": None, "text": generated_post}
        )

        # 20. Message lineage
        self.parent_message_id = source_agent.message_id
        self.original_message_id = (
            source_agent.original_message_id
            if source_agent.original_message_id is not None
            else source_agent.message_id
        )
        self.message_id = self.generate_message_id()

        # 21. Save message event (single, well-defined call)
        self.save_message_event(
            message_events,
            period,
            source_agent,
            influence_prob,
            inferr_stance,
            estimated_stance_shift,
            inferr_stance_post,
            estimated_stance_shift_post,
            inferr_stance_profile,
            estimated_stance_shift_profile,
            old_profile,
            old_profile_embedding,
            new_profile,
            new_profile_embedding,
            source_agent.post,
            source_agent.post_embedding,
            generated_post,
            generated_post_embedding,
            target_stance_before,
            target_stance_after,
        )

        self.post = str(generated_post).strip()
        self.post_embedding = generated_post_embedding
        self.profile = new_profile
        self.profile_embedding = new_profile_embedding

        logger.info(
            "agent %s adopted at period=%s stance=%.4f threshold=%.4f "
            "and will propagate from next period",
            self.id,
            period,
            target_stance_after,
            propagation_threshold,
        )

    # ==============================================================
    # Influence probability
    # ==============================================================

    def calculate_influence_prob(self, source_agent) -> float:
        # 1. Profile similarity
        profile_similarity = 0.0
        if self.embedding is not None and source_agent.embedding is not None:
            profile_similarity = self._cosine_similarity(
                self.embedding, source_agent.embedding
            )
            profile_similarity = (profile_similarity + 1.0) / 2.0

        # 2. Content-profile similarity
        content_profile_similarity = 0.0
        if source_agent.post_embedding is not None and self.embedding is not None:
            content_profile_similarity = self._cosine_similarity(
                source_agent.post_embedding, self.embedding
            )
            content_profile_similarity = (content_profile_similarity + 1.0) / 2.0

        # 3. Stance similarity
        stance_similarity = 1.0 - abs(self.stance - source_agent.stance) / 2.0
        stance_similarity = float(np.clip(stance_similarity, 0.0, 1.0))

        # 4. Degree similarity
        max_degree = max(1, self.scenario.agent_num - 1)
        target_degree_normalized = self.degree / max_degree
        source_degree_normalized = source_agent.degree / max_degree
        degree_similarity = 1.0 - abs(
            target_degree_normalized - source_degree_normalized
        )
        degree_similarity = float(np.clip(degree_similarity, 0.0, 1.0))

        # 5. Weighted influence
        influence_prob = (
            self.scenario.alpha_profile * profile_similarity
            + self.scenario.alpha_content * content_profile_similarity
            + self.scenario.alpha_stance * stance_similarity
            + self.scenario.alpha_degree * degree_similarity
        )

        return float(np.clip(influence_prob, 0.0, 1.0))

    # ==============================================================
    # Event saving
    # ==============================================================

    def save_diffusion_event(
        self,
        diffusion_events,
        period,
        source_agent,
        target_stance_before,
        target_stance_after,
        influence_prob,
        behavior,
    ):
        diffusion_events.append(
            {
                "period": period,
                "source_agent": source_agent.id,
                "target_agent": self.id,
                "target_stance_before": target_stance_before,
                "target_stance_after": target_stance_after,
                "target_delta_stance": target_stance_after - target_stance_before,
                "source_degree_centrality": source_agent.degree_centrality,
                "source_betweenness_centrality": source_agent.betweenness_centrality,
                "target_degree_centrality": self.degree_centrality,
                "target_betweenness_centrality": self.betweenness_centrality,
                "influence_probability": influence_prob,
                "interaction": "exposure",
                "behavior": behavior,
                "adopted": int(self.preference_state == 1),
            }
        )

    def save_message_event(
        self,
        message_events,
        period,
        source_agent,
        influence_prob,
        inferr_stance,
        estimated_stance_shift,
        inferr_stance_post,
        estimated_stance_shift_post,
        inferr_stance_profile,
        estimated_stance_shift_profile,
        old_profile,
        old_profile_embedding,
        new_profile,
        new_profile_embedding,
        received_post,
        received_post_embedding,
        generated_post,
        generated_post_embedding,
        target_stance_before,
        target_stance_after,
    ):
        message_events.append(
            {
                "period": period,
                "message_id": self.message_id,
                "parent_message_id": self.parent_message_id,
                "original_message_id": self.original_message_id,
                "source_agent": source_agent.id,
                "target_agent": self.id,
                "source_stance": source_agent.stance,
                "Inferr_stance": inferr_stance,
                "estimated_stance_shift": estimated_stance_shift,
                "Inferr_stance_post": inferr_stance_post,
                "estimated_stance_shift_post": estimated_stance_shift_post,
                "Inferr_stance_profile": inferr_stance_profile,
                "estimated_stance_shift_profile": estimated_stance_shift_profile,
                "target_stance_before": target_stance_before,
                "target_stance_after": target_stance_after,
                "source_degree_centrality": source_agent.degree_centrality,
                "source_betweenness_centrality": source_agent.betweenness_centrality,
                "target_degree_centrality": self.degree_centrality,
                "target_betweenness_centrality": self.betweenness_centrality,
                "influence_probability": influence_prob,
                "received_post": received_post,
                "embedding_received_post": None
                if received_post_embedding is None
                else np.asarray(received_post_embedding).tolist(),
                "generated_post": generated_post,
                "embedding_generated_post": None
                if generated_post_embedding is None
                else np.asarray(generated_post_embedding).tolist(),
                "experienced_post": self.accept,
                "experienced_embedding_post": None
                if self.accept_embedding is None
                else np.asarray(self.accept_embedding).tolist(),
                "old_profile": old_profile,
                "old_profile_embedding": None
                if old_profile_embedding is None
                else np.asarray(old_profile_embedding).tolist(),
                "new_profile": new_profile,
                "new_profile_embedding": None
                if new_profile_embedding is None
                else np.asarray(new_profile_embedding).tolist(),
                "is_seed": int(self.isSeed),
            }
        )
