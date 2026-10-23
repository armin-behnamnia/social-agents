import os
import numpy as np
import pandas as pd
from Melodie import Environment
from Melodie import AgentList
from source.agent import LLMSocialAgent
from source.scenario import LLMScenario
from source.tools import text2embedding
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

class LLMEnvironment(Environment):
    scenario: "LLMScenario"

    def __init__(self):
        super().__init__()
        self.observer_states = {}

    def setup(self):
        print(">>> ENVIRONMENT SETUP")

        self.s0 = 0
        self.s1 = 0
        self.alteration_degree = 0
        self.Resistant = 0
        self.Neutral = 0
        self.Susceptive = 0 
        self.Infective = 0

        if not hasattr(self, "diffusion_events"):
            self.diffusion_events = []
            self.message_events = []
    def initialize_observer(self, agents):
        """
        Initialize the latent-state observer for all agents.
        The observer does NOT access the true stance of agents.
        It only initializes a prior distribution over the continuous
        latent stance state s_i in [-1, 1].
        """
        self.observer_states = {}
        num_particles = self.scenario.num_particles
        state_min = self.scenario.inference_state_min
        state_max = self.scenario.inference_state_max
        for agent in agents:
            # --------------------------------------------------
            # 1. Initialize prior particles
            # --------------------------------------------------
            particles = np.random.uniform(state_min,state_max,size=num_particles)
            # --------------------------------------------------
            # 2. Initialize particle weights
            # --------------------------------------------------
            weights = np.ones(num_particles,dtype=float) / num_particles
            # --------------------------------------------------
            # 3. Initial state estimate
            # --------------------------------------------------
            estimated_stance = np.average(particles,weights=weights)
            state_variance = np.average((particles - estimated_stance) ** 2,weights=weights)
            # --------------------------------------------------
            # 4. Store observer state
            # --------------------------------------------------
            self.observer_states[agent.id] = {
                "particles": particles,
                "weights": weights,
                "estimated_stance": float(estimated_stance), 
                "state_variance": float(state_variance), "history": []} 

    def build_observation(self, agent):

        observation_window = self.scenario.observation_window
        recent_received_messages = agent.received_message_history[-observation_window:]
        recent_generated_messages = agent.generated_message_history[-observation_window:]
        recent_behaviors = agent.behavior_history[-observation_window:]
        recent_interactions = agent.interaction_history[-observation_window:]

        return {
            "profile": agent.profile,
            "received_messages": recent_received_messages,
            "generated_messages": recent_generated_messages,
            "behaviors": recent_behaviors,
            "interactions": recent_interactions,
            "degree_centrality": agent.degree_centrality,
            "betweenness_centrality": agent.betweenness_centrality
        }
    def update_observer(self, agent, period):
        observer_state = self.observer_states[agent.id]
        observation = self.build_observation(agent)
        particles = observer_state["particles"]
        weights = observer_state["weights"]
        # --------------------------------------------------
        # 1. Prediction
        # --------------------------------------------------
        process_noise = self.scenario.process_noise
        particles = (particles+ np.random.normal(0,process_noise,size=len(particles)))
        particles = np.clip(particles,self.scenario.inference_state_min,self.scenario.inference_state_max)
        # --------------------------------------------------
        # 2. Observation likelihood
        # --------------------------------------------------
        likelihoods = self.calculate_observation_likelihood(particles,observation) 
        # --------------------------------------------------
        # 3. Bayesian weight update
        # --------------------------------------------------
        weights = weights * likelihoods
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            weights = (np.ones(len(particles))/ len(particles))
        else:
            weights = weights / weight_sum
        effective_sample_size = (1.0 / np.sum(weights ** 2))
        threshold = (self.scenario.resampling_threshold* len(particles))

        if effective_sample_size < threshold:
            particles, weights = (self.resample_particles(particles,weights))
        # --------------------------------------------------
        # 4. Estimate posterior
        # --------------------------------------------------
        estimated_stance = np.average(particles,weights=weights)
        variance = np.average((particles - estimated_stance) ** 2,weights=weights)
        # --------------------------------------------------
        # 5. Save observer state
        # --------------------------------------------------
        observer_state["particles"] = particles
        observer_state["weights"] = weights
        observer_state["estimated_stance"] = float(estimated_stance)
        observer_state["state_variance"] = float(variance)
        observer_state["history"].append({"period": period,"estimated_stance": float(estimated_stance),"state_variance": float(variance)})
    def resample_particles(self, particles, weights):
        n = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0
        positions = (np.arange(n) + np.random.random()) / n
        indexes = np.searchsorted(cumulative_sum,positions)
        particles = particles[indexes]
        weights = np.ones(n) / n
        return particles, weights

    def calculate_observation_likelihood(self,particles,observation):
        behavior_likelihood = (self.behavior_likelihood(particles,observation))
        message_likelihood = (self.message_likelihood(particles,observation))
        history_likelihood = (self.history_likelihood(particles,observation))
        likelihood = (behavior_likelihood* message_likelihood* history_likelihood)
        return np.maximum(likelihood, 1e-12)

    def history_likelihood(self, particles, observation):
        interactions = observation.get("interactions", [])
        if not interactions:
            return np.ones(len(particles), dtype=float)
        return np.ones(len(particles), dtype=float)
    def message_likelihood(self, particles, observation):
        messages = observation.get("messages", [])
        if not messages:
            return np.ones(len(particles), dtype=float)
        return np.ones(len(particles), dtype=float)

    def behavior_likelihood(self, particles, observation):
        """
        P(observation | latent stance)
        Each particle represents a continuous latent stance in [-1, 1].
        """
        behaviors = observation.get("behaviors", [])
        if not behaviors:
            return np.ones(len(particles), dtype=float)
        observed_behavior = behaviors[-1]
        likelihoods = np.zeros(len(particles), dtype=float)
        for i, stance in enumerate(particles):
            # Map continuous stance [-1, 1] to acceptance probability [0, 1]
            acceptance_probability = 1.0 / (1.0 + np.exp(-self.scenario.observation_beta * stance))
            if observed_behavior == "accept":
                likelihoods[i] = acceptance_probability
            elif observed_behavior == "reject":
                likelihoods[i] = 1.0 - acceptance_probability
            else:
                likelihoods[i] = 1e-3
        return likelihoods

    def selective_post(self, messages, messages_embeddings, target_agent):
        similarities = []
        for embedding in messages_embeddings:
            similarity = cosine_similarity(embedding, target_agent.profile_embedding)
        similarities.append(similarity)
        best_idx = similarities.index(max(similarities))
        return messages[best_idx], messages_embeddings[best_idx]

    def select_post(self, source_agent, target_agent): # Default scenario type is Generic
        if self.scenario.scenario_type == "Selective":
            source_agent.post, source_agent.post_embedding  = self.selective_post(self.scenario.sub_influence_msgs, self.scenario.sub_influence_msgs_embeddings, target_agent )
        elif self.scenario.scenario_type == "Generative":
            source_agent.post = goal_promotion(target_agent.profile)
            source_agent.post_embedding = self.text2embedding(source_agent.post)
        elif self.scenario.scenario_type == "RL_Generative":
            print("How the post is selected or spread, and how feedback is received, should be clearly defined.")
    def agents_infection_stance_aware(self, agents: "AgentList[LLMSocialAgent]", period: int, k=1): #تا زمانی که همه یک دور آلوده نشده‌اند ادامه بده.مناسب موقعی که براساس پیام باقیمانده در مغز الوده گی منتشر میشود
        print(f"\n===== PERIOD {period} =====\n")
        # Initialize the queue with initially infected seeds
        queue = deque()

        for agent in agents:
            if agent.preference_state == 1 and agent.stance <= 0:
                queue.append(agent)

        processed = set()

        while queue:
            source_agent = queue.popleft()
            source_id = id(source_agent)

            # Prevent processing the same node more than once
            if source_id in processed:
                continue

            processed.add(source_id)

            neighbors = source_agent.network.get_neighbors(source_agent)

            for neighbor_category, neighbor_id in neighbors:
                target_agent = agents.get_agent(neighbor_id)
                target_id = id(target_agent)
    
                # Skip already processed nodes
                if target_id in processed:
                    continue

                target_agent.exposure_count += 1
                if target_agent.exposure_count > k: continue
                # if source_agent.stance < 0:
                #     self.select_post(source_agent, target_agent)
                target_agent.infection_stance_aware(agents,period,self.diffusion_events,self.message_events,source_agent)
                self.update_observer(target_agent, period)
                # Newly infected targets can spread information too
                queue.append(target_agent)

        print(f"Processed nodes: {len(processed)}")
        print(f"Total diffusion events: {len(self.diffusion_events)}")
        # At the end preference_stance target nodes should change to 0 and for the next run should k be 2.

    def agents_infection(self, agents, period: int, k=1):
        print(f"\n===== PERIOD {period} =====\n")

        queue = deque(
            a for a in agents
            if a.preference_state == 1 and a.stance <= 0
        )
        queued = {id(a) for a in queue}
        processed = set()

        while queue:
            source = queue.popleft()
            sid = id(source)
            queued.discard(sid)

            if sid in processed:
                continue
            processed.add(sid)

            for _, neighbor_id in source.network.get_neighbors(source):
                target = agents.get_agent(neighbor_id)
                tid = id(target)

                if tid in processed or target.preference_state == 1: continue

                target.exposure_count += 1

                #if target.exposure_count < k: continue
                #if source.stance<0:
                #    self.select_post(source, target)
                emerged = target.infection(agents, period, self.diffusion_events, self.message_events, source)

                if emerged:
                    self.update_observer(target, period)
 
                    if tid not in queued:
                        queue.append(target)
                        queued.add(tid)

        print(f"Processed nodes: {len(processed)}")
        print(f"Total diffusion events: {len(self.diffusion_events)}")

    def calc_population_infection_state(self, agents: "AgentList[LLMSocialAgent]"):
        self.setup()
        for agent in agents:
            # print(f"Agent {agent.id}: influencer = {agent.influencer}")  # دیباگ
            if agent.preference_state == 0:
                self.s0 += 1
            elif agent.preference_state == 1:
                self.s1 += 1
            #if agent.latent_state == "Resistant": self.Resistant+=1;
            #elif agent.latent_state == "Neutral": self.Neutral+=1;
            #elif agent.latent_state == "Susceptive": self.Susceptive+=1;
            #elif agent.latent_state == "Infective": self.Infective+=1;
        self.information_analysis(agents)
        #print("calc_population_infection")

    def save_information_analysis(self, agents: "AgentList[LLMSocialAgent]", method, k=1):
        output_dir = f"data/output_{method}_{k}"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(
            output_dir,
            f"user_posts_{str(self.scenario.id)}.txt"
        )
        with open(filename, 'w', encoding="utf-8") as file:
            for agent in agents:
                file.write("User ID" + str(agent.id + 1) + ": " + str(agent.post or "") + "\n")
        filename = os.path.join(
            output_dir,
            f"user_accepts_{str(self.scenario.id)}.txt"
        )
        with open(filename, 'w', encoding="utf-8") as file:
            for agent in agents:
                if agent.influencer is not None:
                    prefix = "User ID " + str(agent.influencer.id+1) + " -> "
                else:
                    prefix = ""
                file.write(prefix + "User ID " + str(agent.id + 1) + ": " + agent.internalized + "\n")

    def information_analysis(self, agents: "AgentList[LLMSocialAgent]"):
        active_users = []
        for agent in agents:
            if agent.influencer is not None:
                active_users.append(agent)
                # print("active users increased") #here does not increase active users..

        alteration_degrees = []
        for au in active_users:
            m_k_altered = au.post_embedding
            m_k = au.accept_embedding
            if m_k is None or m_k_altered is None:
                alteration_degree = np.nan
            else:
                alteration_degree = float(1 - cosine_similarity(m_k.reshape(1, -1),m_k_altered.reshape(1, -1))[0][0])

            #alteration_degree = 1 - cosine_similarity(m_k.reshape(1, -1), m_k_altered.reshape(1, -1))
            alteration_degrees.append(alteration_degree)
            # print("alteration_degree: ", alteration_degree )
        # alteration_degrees = np.squeeze(np.array(alteration_degrees))
        # print("alteration_degree after seueeze: ", alteration_degrees)
        self.alteration_degree = np.float64(np.mean(alteration_degrees))
        #print("alteration_degree after seueeze: ", self.alteration_degree )
        # print(self.alteration_degree)

    def save_diffusion_events(self, method, k):
        output_dir = f"data/output_{method}_{k}"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(
            output_dir,
            f"diffusion_events_{self.scenario.id}.csv"
        )
        #filename = (f"data/output/"f"diffusion_events_"f"{self.scenario.id}.csv")

        print("\n===== SAVE DIFFUSION EVENTS =====")
        print("Number of events:", len(self.diffusion_events))
        print("Output file:", filename)

        df = pd.DataFrame(self.diffusion_events)

        print("DataFrame shape:", df.shape)
        print("Columns:", df.columns.tolist())

        if not df.empty:
            df["scenario"] = self.scenario.id
            print(df.head())

        df.to_csv(filename,index=False,encoding="utf-8-sig")

        print("Diffusion events saved successfully.")
        print("=================================\n")

        filename = os.path.join(
            output_dir,
            f"Message_events_{self.scenario.id}.csv"
        )

        print("\n===== SAVE Message EVENTS =====")
        print("Number of events:", len(self.message_events))
        print("Output file:", filename)

        df = pd.DataFrame(self.message_events)

        print("DataFrame shape:", df.shape)
        #print("Columns:", df.columns())

        if not df.empty:
            df["scenario"] = self.scenario.id
            print(df.head())

        df.to_csv(filename,index=False,encoding="utf-8-sig")
