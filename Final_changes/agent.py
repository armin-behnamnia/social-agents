import random
import uuid
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Melodie import NetworkAgent

from source.tools import text2embedding
from source.LLM_module import generate_information, goal_promotion, accept_message, accept_message2
from source.LLM_module import generate_response_message, update_profile, Inference_stance, Inference_stance_post, Inference_stance_profile

if TYPE_CHECKING:
    from Melodie import AgentList
    from source.scenario import LLMScenario


class LLMSocialAgent(NetworkAgent):

    scenario: "LLMScenario"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ==========================================================
        # Observable temporal histories
        # ==========================================================

        self.received_message_history = []
        self.decision_history = []
        self.topic_exposures = {}
        self.dominant_decision_style = "emotional" if random.random() < 0.8 else "rational"
        self.generated_message_history = []
        self.behavior_history = []
        self.interaction_history = []

    # ==============================================================
    # Agent category
    # ==============================================================

    def set_category(self):
        self.category = 0

    # ==============================================================
    # Agent initialization
    # ==============================================================

    def setup(self):

        # ----------------------------------------------------------
        # Diffusion state
        # ----------------------------------------------------------

        self.preference_state: int = 0
        self.isSeed: bool = False
        # Hidden simulator role; never exposed to the observer, prompts, or public tables.
        self.is_covert_source: bool = False
        self.stance_prompt: bool = True  # False under the 'stance_less' condition

        # ----------------------------------------------------------
        # Static agent context
        # ----------------------------------------------------------

        self.profile: str = ""

        # ----------------------------------------------------------
        # Latent internal state
        # Continuous stance: [-1, 1]
        # ----------------------------------------------------------

        self.stance: float = 0.0
        self.stance_label: str = "Neutral"
        self.support_probability: float = 0.5

        # ----------------------------------------------------------
        # Messages
        # ----------------------------------------------------------

        self.post: str = ""
        self.internalized : str = ""
        self.embedding = None
        self.profile_embedding = None
        self.post_embedding = None
        self.internalized_embedding = None

        # ----------------------------------------------------------
        # Interaction
        # ----------------------------------------------------------

        self.influencer = None

        # Cumulative number of exposures
        self.exposure_count: int = 0

        # Number of infected/adopted neighbors
        self.num_infected_neighbors: int = 0

        # ----------------------------------------------------------
        # Diffusion metrics
        # ----------------------------------------------------------

        self.new_adopter: int = 0
        self.adoption_time: int = -1

        # ----------------------------------------------------------
        # Network position
        # Static/contextual network properties
        # ----------------------------------------------------------

        self.degree: int = 0
        self.degree_centrality: float = 0.0
        self.betweenness_centrality: float = 0.0

        # ----------------------------------------------------------
        # Message lineage
        # ----------------------------------------------------------

        self.message_id = None
        self.parent_message_id = None
        self.original_message_id = None

    # ==============================================================
    # Message ID
    # ==============================================================

    def generate_message_id(self):
        return str(uuid.uuid4())

    # ==============================================================
    # Stance update
    # ==============================================================

    def update_agent_stance(
        self,
        source_agent,
        source_influence_strength,
        target_susceptibility_rate,
        accepted=True
    ):

        old_stance = float(self.stance)

        # Design B: every exposure shifts stance toward the source;
        # REJECT uses a smaller fraction (reject_influence_ratio) than ACCEPT.
        ratio = 1.0 if accepted else float(
            getattr(self.scenario, 'reject_influence_ratio', 0.25))
        step = (
            ratio
            * source_influence_strength
            * target_susceptibility_rate
            * (source_agent.stance - old_stance)
        )
        new_stance = old_stance + step

        new_stance = np.clip(
            new_stance,
            self.scenario.inference_state_min,
            self.scenario.inference_state_max
        )

        self.stance = float(new_stance)

        # ----------------------------------------------------------
        # Qualitative label derived from continuous stance
        # ----------------------------------------------------------

        threshold = self.scenario.stance_threshold

        if self.stance >= threshold:
            self.stance_label = "Supportive"

        elif self.stance > 0:
            self.stance_label = "Receptive"

        elif self.stance <= -threshold:
            self.stance_label = "Resistant"

        else:
            self.stance_label = "Neutral"

        # ----------------------------------------------------------
        # Derived support probability
        # ----------------------------------------------------------

        self.support_probability = (
            self.stance + 1.0
        ) / 2.0

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

    # ==============================================================
    # Safe cosine similarity
    # ==============================================================

    @staticmethod
    def _cosine_similarity(embedding_a, embedding_b):

        if embedding_a is None or embedding_b is None:
            return 0.0

        try:

            embedding_a = np.asarray(embedding_a)
            embedding_b = np.asarray(embedding_b)

            if embedding_a.size == 0 or embedding_b.size == 0:
                return 0.0

            similarity = cosine_similarity(
                embedding_a.reshape(1, -1),
                embedding_b.reshape(1, -1)
            )[0, 0]

            return float(similarity)

        except Exception:
            return 0.0

    # ==============================================================
    # Infection / influence process
    # ==============================================================
    def selective_post(self, messages, messages_embeddings):
        similarities = []
        for embedding in messages_embeddings:
            similarity = _cosine_similarity(
                embedding, self.profile_embedding
            )
        similarities.append(similarity)

        best_idx = similarities.index(max(similarities))
        return messages[best_idx], messages_embeddings[best_idx]

    def infection_test(self, agents, period,diffusion_events, message_events, source_agent):
        decision1 = accept_message(source_agent.post, self.profile, self.received_message_history)
        decision2 = accept_message2(source_agent.post, self.profile, self.received_message_history)
        self.received_message_history.append(source_agent.post)
        print("decision1,2: ", decision1, decision2); return False
    def infection(self, agents, period, diffusion_events, message_events, source_agent):
        # Infected nodes cannot be infected again.
        if self.isSeed and self.preference_state == 1:
            return False

        if self.id == source_agent.id:
            return False

        source_post = source_agent.post; #print("source_post: ", source_post )
        if source_post is None or not str(source_post).strip():
            return False

        source_post = str(source_post).strip()

        if source_agent.post_embedding is None:
            source_agent.post_embedding = self._get_embedding(source_post)
            # Continue processing; do not return here.

        old_post = self.post
        old_profile = (
            self.profile.copy()
            if isinstance(self.profile, dict)
            else self.profile
        )
        old_profile_embedding = self.profile_embedding

        # Stance stays fixed.
        target_stance_before = float(self.stance)
        target_stance_after = target_stance_before

        influence_prob = float(np.clip(
            self.calculate_influence_prob(source_agent), 0.0, 1.0
        ))

        topic_key = str(getattr(self.scenario, 'topic', None) or
                        getattr(self.scenario, 'stance_topic', 'default'))
        encounter = self.topic_exposures.get(topic_key, 0) + 1
        dominant = self.dominant_decision_style
        mode = dominant if random.random() < 0.8 else (
            "rational" if dominant == "emotional" else "emotional")
        result = accept_message(
            source_post, self.profile,
            [row for row in self.decision_history if row['topic'] == topic_key],
            topic_exposure_count=encounter, decision_mode=mode,
            stance=target_stance_before if self.stance_prompt else None)
        # Accept structured responses without converting the whole dict to a label.
        # Keep legacy label-only providers compatible; do not invent a rationale.
        if isinstance(result, dict):
            result = dict(result)
            decision_text = str(result.get('decision', '')).strip().upper()
            result.setdefault('reason', '')
            result.setdefault('reason_source', 'llm' if result['reason'] else 'missing')
        elif isinstance(result, str):
            decision_text = result.strip().upper()
            result = dict(decision=decision_text, reason='',
                          reason_source='legacy_missing')
        else:
            raise ValueError(f'Invalid accept_message response: {result!r}')
        if decision_text not in {'ACCEPT', 'REJECT'}:
            raise ValueError(
                'accept_message must return ACCEPT or REJECT, either as a '
                f'label or in the decision field; received {result!r}')
        result['decision'] = decision_text
        self.topic_exposures[topic_key] = encounter
        decision_record = dict(period=period, source=source_agent.id, topic=topic_key,
                               topic_exposure_count=encounter, decision=decision_text,
                               reason=result['reason'], reason_source=result['reason_source'],
                               dominant_decision_style=dominant, decision_mode=mode)
        self.decision_history.append(decision_record)
        was_adopted = self.preference_state == 1

        accepted = decision_text == "ACCEPT"
 
        # Histories are appended centrally after the completed interaction.

        # Reference dynamics: one acceptance gate, then latent response on every exposure.
        # This hidden update is never a predictor feature for the current event.
        target_stance_before, target_stance_after = self.update_agent_stance(
            source_agent=source_agent, source_influence_strength=influence_prob,
            target_susceptibility_rate=self.scenario.stance_influence_rate,
            accepted=accepted)
        stance_delta = target_stance_after - target_stance_before

        if accepted:
            # Only on ACCEPT: internalize the message, then derive profile and post from it.
            self.internalized = generate_response_message(source_post, self.profile, abs(stance_delta))
            self.internalized_embedding = self._get_embedding(self.internalized)
            received_text = self.internalized or source_post
            new_profile = update_profile(
                received=received_text,
                profile=self.profile,
            )

            generated_post = generate_information(
                received=received_text,
                profile=self.profile,
                old_post=self.post,
            )

            if new_profile is None:
                new_profile = old_profile

            if generated_post is None or not str(generated_post).strip():
                generated_post = old_post

            self.profile = new_profile
            self.profile_embedding = self._get_embedding(str(new_profile))

            self.post = (
                str(generated_post).strip()
                if generated_post is not None
                else None
            )
            self.post_embedding = (
                self._get_embedding(self.post)
                if self.post
                else None
            )

            # Acceptance defines infection in your stated model.
            self.preference_state = 1
            if not was_adopted:
                self.adoption_time = period
            self.new_adopter = int(not was_adopted)
            self.embedding = self.profile_embedding
            self.parent_message_id = source_agent.message_id
            self.original_message_id = source_agent.original_message_id or source_agent.message_id
            self.message_id = self.generate_message_id()

        # One diffusion record per valid acceptance/rejection attempt.
        self.save_diffusion_event(
            diffusion_events,
            period,
            source_agent,
            target_stance_before,
            target_stance_after,
            influence_prob,
            decision_text,
        )

        self.save_message_event(
            message_events,
            period,
            source_agent,
            influence_prob,
            0.0,
            1.0 - self._cosine_similarity(source_agent.post_embedding, self.post_embedding),
            1.0 - self._cosine_similarity(old_profile_embedding, self.profile_embedding),
            old_profile,
            old_profile_embedding,
            self.profile,
            self.profile_embedding,
            source_post,
            source_agent.post_embedding,
            self.post,
            self.post_embedding,
            target_stance_before,
            target_stance_after,
            decision_text,
        )

        for event in (diffusion_events[-1], message_events[-1]):
            event.update(decision_record)
            event['new_adoption'] = int(accepted and not was_adopted)
        print(f"period={period}, source={source_agent.id}, target={self.id}, "
              f"decision={decision_text}, mode={mode}, reason={result['reason']}")

        return accepted

    def calculate_influence_prob(
        self,
        source_agent
    ):

        # ----------------------------------------------------------
        # 1. Profile similarity
        # ----------------------------------------------------------

        profile_similarity = 0.0

        if (
            self.embedding is not None
            and source_agent.embedding is not None
        ):

            profile_similarity = (
                self._cosine_similarity(
                    self.embedding,
                    source_agent.embedding
                )
            )

            profile_similarity = (
                profile_similarity + 1.0
            ) / 2.0

        # ----------------------------------------------------------
        # 2. Content-profile similarity
        # ----------------------------------------------------------

        content_profile_similarity = 0.0

        if (
            source_agent.post_embedding is not None
            and self.embedding is not None
        ):

            content_profile_similarity = (
                self._cosine_similarity(
                    source_agent.post_embedding,
                    self.embedding
                )
            )

            content_profile_similarity = (
                content_profile_similarity + 1.0
            ) / 2.0

        # ----------------------------------------------------------
        # 3. Stance similarity
        # ----------------------------------------------------------

        stance_similarity = (
            1.0
            - abs(
                self.stance
                - source_agent.stance
            ) / 2.0
        )

        stance_similarity = np.clip(
            stance_similarity,
            0.0,
            1.0
        )

        # ----------------------------------------------------------
        # 4. Degree similarity
        # ----------------------------------------------------------

        max_degree = max(
            1,
            self.scenario.agent_num - 1
        )

        target_degree_normalized = (
            self.degree / max_degree
        )

        source_degree_normalized = (
            source_agent.degree / max_degree
        )

        degree_similarity = (
            1.0
            - abs(
                target_degree_normalized
                - source_degree_normalized
            )
        )

        degree_similarity = np.clip(
            degree_similarity,
            0.0,
            1.0
        )

        # ----------------------------------------------------------
        # 5. Weighted influence
        # ----------------------------------------------------------

        influence_prob = (

            self.scenario.alpha_profile
            * profile_similarity

            +

            self.scenario.alpha_content
            * content_profile_similarity

            +

            self.scenario.alpha_stance
            * stance_similarity

            +

            self.scenario.alpha_degree
            * degree_similarity
        )

        influence_prob = np.clip(
            influence_prob,
            0.0,
            1.0
        )

        return float(
            influence_prob
        )

    # ==============================================================
    # Save diffusion event
    # ==============================================================

    def save_diffusion_event(
        self,
        diffusion_events,
        period,
        source_agent,
        target_stance_before,
        target_stance_after,
        influence_prob,
        behavior
    ):

        diffusion_events.append({

            "period":
                period,

            "source_agent":
                source_agent.id,

            "target_agent":
                self.id,

            # ------------------------------------------------------
            # Target latent state
            # ------------------------------------------------------

            "target_stance_before":
                target_stance_before,

            "target_stance_after":
                target_stance_after,

            "target_delta_stance":
                (
                    target_stance_after
                    - target_stance_before
                ),

            # ------------------------------------------------------
            # Source network position
            # ------------------------------------------------------

            "source_degree_centrality":
                source_agent.degree_centrality,

            "source_betweenness_centrality":
                source_agent.betweenness_centrality,

            # ------------------------------------------------------
            # Target network position
            # ------------------------------------------------------

            "target_degree_centrality":
                self.degree_centrality,

            "target_betweenness_centrality":
                self.betweenness_centrality,

            # ------------------------------------------------------
            # Diffusion
            # ------------------------------------------------------

            "influence_probability":
                influence_prob,

            "interaction":
                self.exposure_count,

            "behavior": behavior,
            "adopted": int(str(behavior).upper() == "ACCEPT"),
        })

    # ==============================================================
    # Save message event
    # ==============================================================

    def save_message_event(
        self,
        message_events,
        period,
        source_agent,
        influence_prob,
        estimated_stance_shift,
        estimated_stance_shift_post,
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
        decision
    ):

        message_events.append({
    "period": period,
    # ------------------------------------------------------
    # Message lineage
    # ------------------------------------------------------
    "message_id": self.message_id,
    "parent_message_id": self.parent_message_id,
    "original_message_id": self.original_message_id,

    # ------------------------------------------------------
    # Agents
    # ------------------------------------------------------
    "source_agent": source_agent.id,
    "target_agent": self.id,

    # ------------------------------------------------------
    # Latent state
    # ------------------------------------------------------
    "source_stance": source_agent.stance,

    # Combined inference
    #"Inferr_stance": Inferr_stance,
    "estimated_stance_shift": estimated_stance_shift,

    # Post-based inference
    #"Inferr_stance_post": Inferr_stance_post,
    "estimated_stance_shift_post": estimated_stance_shift_post,

    # Profile-based inference
    #"Inferr_stance_profile": Inferr_stance_profile,
    "estimated_stance_shift_profile": estimated_stance_shift_profile,

    # Ground-truth simulator state
    "target_stance_before": target_stance_before,
    "target_stance_after": target_stance_after,

    # ------------------------------------------------------
    # Source network position
    # ------------------------------------------------------
    "source_degree_centrality": source_agent.degree_centrality,
    "source_betweenness_centrality": source_agent.betweenness_centrality,

    # ------------------------------------------------------
    # Target network position
    # ------------------------------------------------------
    "target_degree_centrality": self.degree_centrality,
    "target_betweenness_centrality": self.betweenness_centrality,

    # ------------------------------------------------------
    # Influence
    # ------------------------------------------------------
    "influence_probability": influence_prob,

    # ------------------------------------------------------
    # Received message
    # ------------------------------------------------------
    "received_post": received_post,
    "embedding_received_post": received_post_embedding,

    # ------------------------------------------------------
    # Generated message
    # ------------------------------------------------------
    "generated_post": generated_post,
    "embedding_generated_post": generated_post_embedding,
    # ------------------------------------------------------
    # Experienced message
    # ------------------------------------------------------
    "experienced_post": self.internalized,
    "experienced_embedding_post": self.internalized_embedding,
    # ------------------------------------------------------
    # Profile
    # ------------------------------------------------------
    "old_profile": old_profile,
    "old_profile_embedding": old_profile_embedding,

    "new_profile": new_profile,
    "new_profile_embedding": new_profile_embedding,

    # ------------------------------------------------------
    # Seed
    # ------------------------------------------------------
    "is_seed": int(self.isSeed),
    "decision": decision
})
