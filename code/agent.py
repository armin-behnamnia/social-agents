import random
import uuid
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Melodie import NetworkAgent

from source.tools import text2embedding
from source.LLM_module import generate_information, goal_promotion, accept_message
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
        target_susceptibility_rate
    ):

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

    def infection_stance_aware(self, agents: "AgentList[LLMSocialAgent]", 
        period: int, diffusion_events: list, message_events: list, source_agent):

        if self.id == source_agent.id: print("same_target_id, target_id: ", self.id) ;return

        if (source_agent.post is None or not str(source_agent.post).strip()):
           # print(f"WARNING: source Agent {source_agent.id} has an empty post. Skipping influence; target_id: ", self.id)
            return

        # ----------------------------------------------------------
        # 2. Prepare source message embedding
        # ----------------------------------------------------------

        if source_agent.post_embedding is None: 
            source_agent.post_embedding = self._get_embedding(source_agent.post)
            #print(f"WARNING: source Agent {source_agent.id} has no valid post embedding. Skipping influence. target_id:", self.id)
            return

        # ----------------------------------------------------------
        # 3. Prepare target profile embedding
        # ----------------------------------------------------------

        if self.embedding is None and self.profile: 
            self.embedding = self._get_embedding(self.profile)
            self.profile_embedding = self.embedding

        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # 4. Calculate influence probability
        # ----------------------------------------------------------

        influence_prob = self.calculate_influence_prob(source_agent); influence_prob = float(np.clip(influence_prob, 0.0, 1.0))

        # ----------------------------------------------------------
        # 5. Update latent internal state
        # ----------------------------------------------------------

        target_stance_before = float(self.stance); target_stance_after = float(self.stance)

        if self.scenario.stance_influence_rate > 0:

            (target_stance_before, target_stance_after) = self.update_agent_stance(source_agent=source_agent,
                source_influence_strength=influence_prob,
                target_susceptibility_rate=(self.scenario.stance_influence_rate))

        stance_delta = (target_stance_after - target_stance_before)

        # ----------------------------------------------------------
        # 6. Store accepted message
        # ----------------------------------------------------------
        #acceptance_result = accept_message (source_agent.post, self.profile, self.exposure_count, self.stance )
        #if acceptance_result is contain accept:

        Internalized_message = generate_response_message( source_agent.post, self.profile, abs(stance_delta) ) #Experienced_message
        self.internalized = Internalized_message 
        self.internalized_embedding = self._get_embedding(self.internalized)

        # ----------------------------------------------------------
        # 7. Check whether the target crosses the adoption threshold
        # ----------------------------------------------------------

        # reached_adoption_threshold = (target_stance_after <= propagation_threshold)


        new_profile = None; generated_post = None;
        new_profile = update_profile(received=self.internalized , profile=self.profile)
        new_profile_embedding = self._get_embedding(new_profile)
        generated_post = generate_information(received= self.internalized , profile=self.profile, old_post= self.post)
        generated_post_embedding = self._get_embedding(generated_post)
     
        # ----------------------------------------------------------
        # 8. Protect against invalid LLM output
        # ----------------------------------------------------------

        if (generated_post is not None):
            Inferr_stance_post = 0 #Inference_stance_post( source_agent.post, generated_post)
            estimated_stance_shift_post = 1- self._cosine_similarity(generated_post_embedding, source_agent.post_embedding)

        old_profile = self.profile
        old_profile_embedding = self._get_embedding(old_profile)
        old_post = self.post
        self.post = self.internalized; self.post_embedding = self.internalized_embedding
        if new_profile is not None:
            estimated_stance_shift_profile = 1- self._cosine_similarity(old_profile_embedding, new_profile_embedding)
            
        if new_profile is not None and generated_post is not None:
            estimated_stance_shift = estimated_stance_shift_post * estimated_stance_shift_profile

        # ----------------------------------------------------------
        # 9. Message lineage
        # ----------------------------------------------------------

        parent_message_id = (source_agent.message_id)
        original_message_id = (source_agent.original_message_id if source_agent.original_message_id is not None else source_agent.message_id)
        self.message_id = (self.generate_message_id())
        self.parent_message_id = (parent_message_id)
        self.original_message_id = (original_message_id)

        # ----------------------------------------------------------
        # 10. Save message event
        # ----------------------------------------------------------

        self.post = str(generated_post).strip()
        self.post_embedding = generated_post_embedding
        self.profile = new_profile; self.profile_embedding = new_profile_embedding
        post_changed = (generated_post is not None and str(generated_post).strip() != str(old_post or "").strip())
        profile_changed = (new_profile is not None and new_profile != old_profile)
        emerged = post_changed or profile_changed
        #if emerged: self.preference_state = 1
        self.save_message_event(message_events, period, source_agent, influence_prob, estimated_stance_shift,
        estimated_stance_shift_post, estimated_stance_shift_profile, old_profile, self.profile_embedding,
        new_profile, new_profile_embedding, source_agent.post, source_agent.post_embedding, generated_post,
        generated_post_embedding, target_stance_before, target_stance_after, emerged)
        self.save_diffusion_event( diffusion_events, period, source_agent, target_stance_before, target_stance_after, influence_prob, emerged)
        return emerged

    def infection(self, agents, period,diffusion_events, message_events, source_agent):
        if self.id == source_agent.id:
            print("initial return, same id,  target_id: ", self.id); 
            return False

        source_post = source_agent.post
        if not source_post or not str(source_post).strip():
            print("initial return, !post, target_id: ", self.id);
            return False

        # Ensure source embedding exists
        if source_agent.post_embedding is None:
            source_agent.post_embedding = self._get_embedding(source_post)

        if source_agent.post_embedding is None:
            print("initial return,!post_embedding, target_id: ", self.id);
            return False

        # Save the target's previous state
        old_post = self.post
        old_profile = self.profile.copy() if isinstance(self.profile, dict) else self.profile

        target_stance_before = float(self.stance)
        target_stance_after = target_stance_before
        influence_prob = float(np.clip(
            self.calculate_influence_prob(source_agent), 0.0, 1.0
        ))

        # Decide whether to accept the message
        self.received_message_history.append(source_post)
        decision = accept_message(source_post, self.profile, self.received_message_history)
        decision = str(decision).strip().upper()
        if "REJECT" in decision:
            self.save_message_event(message_events, period, source_agent, influence_prob, 0,
            0, 0, old_profile, self.profile_embedding,
            "", "", source_agent.post, source_agent.post_embedding,"","", target_stance_before,
            target_stance_after, decision)
            self.save_diffusion_event( diffusion_events, period, source_agent, target_stance_before, target_stance_after, influence_prob, decision)
        print("Decision: ", decision, "target_id: ", self.id);return False
        #elif "Accept" in decision:
        #   print("Message Accepted: ",decision)
        # Internalize and respond to the received message
        new_profile = update_profile(
            received=source_post,
            profile=self.profile
        )
        new_profile_embedding = self._get_embedding(new_profile)
        generated_post = generate_information(
            received=source_post,
            profile=self.profile,
            old_post=self.post
        )
        generated_post_embedding = self._get_embedding(generated_post)
        # Validate outputs and retain the old values if needed
        if not generated_post or not str(generated_post).strip():
            generated_post = old_post

        if new_profile is None:
            new_profile = old_profile

        self.save_message_event(message_events, period, source_agent, influence_prob, 0,
            0,0 ,old_profile, self.profile_embedding,
            new_profile, new_profile_embedding, source_agent.post, source_agent.post_embedding, generated_post,
            generated_post_embedding, target_stance_before, target_stance_after, decision)
        # Detect emergence from observable changes
        post_changed = (
            generated_post is not None
            and generated_post != old_post
        )
        profile_changed = new_profile != old_profile
        emerged = post_changed or profile_changed

        if not emerged:
            print("dont emerged, return, target_id: ", self.id);
            return False

        # Update the target's observable state
        self.post = str(generated_post).strip()
        self.post_embedding = self._get_embedding(self.post)
        self.profile = new_profile
        self.profile_embedding = self._get_embedding(str(new_profile))

        # Mark the target as emerged/infected
        self.preference_state = 1
        print("Final emerged, saved, return, target_id: ", self.id);
        #return False 
        self.save_diffusion_event( diffusion_events, period, source_agent, target_stance_before, target_stance_after, influence_prob, emerged)
        # Record events here using your existing event methods.
        # Ensure estimated shifts and behavior are defined first.

        return True

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
            "adopted": int(behavior == "accept"),
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
