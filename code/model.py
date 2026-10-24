from typing import TYPE_CHECKING

import numpy as np
import random
from source.stance import generate_stances
from source.stance import assign_latent_state
import Melodie
from Melodie import Model
from Melodie import Edge,Network
from source import data_info
from source.agent import LLMSocialAgent
from source.data_collector import LLMSocialNetworkDataCollector
from source.environment import LLMEnvironment
from source.scenario import LLMScenario
from source.tools import text2embedding, load_pickle, create_user_embedding
import networkx as nx
if TYPE_CHECKING:
    from Melodie import AgentList

class LLMModel(Model):
    scenario: "LLMScenario"

    def create(self):
        self.agents: "AgentList[LLMSocialAgent]" = self.create_agent_list(LLMSocialAgent)
        self.environment: LLMEnvironment = self.create_environment(LLMEnvironment)
        self.data_collector = self.create_data_collector(LLMSocialNetworkDataCollector)
        self.network = self.create_network()


    def assign_agent_stances(self):
        params = self.scenario.stance_params
        stances = generate_stances(
            num_agents=self.scenario.agent_num,
            support_ratio=params["support_ratio"],
            support_alpha=params["support_distribution"]["alpha"],
            support_beta=params["support_distribution"]["beta"],
            oppose_alpha=params["oppose_distribution"]["alpha"],
            oppose_beta=params["oppose_distribution"]["beta"],
        )
        for agent, stance in zip(self.agents, stances):
            # Ground-truth latent state
            agent.stance = float(stance)
            #agent.update_stance_label()	

    def calculate_degree_centrality(self):
        num_agents = len(self.agents)
        for agent in self.agents:
            neighbors = self.network.get_neighbors(agent)
            degree = len(neighbors)
            if num_agents > 1:
                agent.degree_centrality = degree / (num_agents - 1)
            else:
                agent.degree_centrality = 0.0

    def calculate_betweenness_centrality(self):
        import networkx as nx
        G = nx.DiGraph()
        for agent in self.agents:
            G.add_node(agent.id)
        for agent in self.agents:
            neighbors = self.network.get_neighbors(agent)
            for _, neighbor_id in neighbors:
                G.add_edge(agent.id, neighbor_id)
        centrality = nx.betweenness_centrality(
            G,
            normalized=True
        )
        for agent in self.agents:
            agent.betweenness_centrality = float(
                centrality.get(agent.id, 0.0)
            )
    def setup(self):
        # ---------------------------------------------------------
        # 1. Initialize agents
        # ---------------------------------------------------------
        self.agents.setup_agents(
            agents_num=self.scenario.agent_num,
            params_df=self.scenario.get_dataframe(
                data_info.agent_params
            )
        )
        # ---------------------------------------------------------
        # 2. Create social network
        # ---------------------------------------------------------
        if self.scenario.get_network_params() != {}:
            self.network.setup_agent_connections(
                agent_lists=[self.agents],
                network_type=self.scenario.network_type,
                network_params=self.scenario.get_network_params(),
            )
        else:
            network = Network(model=LLMSocialAgent)
            self.network = network
            for agent in self.agents:
                self.network.add_agent(agent)
            filename = (
                f"data/input/"
                f"{self.scenario.network_type}.txt"
            )
            with open(filename, "r") as file:
                for line in file:
                    if not line.strip():
                        continue
                    u, i = map(
                        int,
                        line.strip().split()[:2]
                    )
                    edge = Edge(
                        category_1=0,
                        agent_1_id=u,
                        category_2=0,
                        agent_2_id=i,
                        edge_properties={}
                    )
                    self.network.add_edge(
                        (0, u),
                        (0, i),
                        edge
                    )
        # ---------------------------------------------------------
        # 3. Initialize latent stance
        # ---------------------------------------------------------
        self.assign_agent_stances() 
        # ---------------------------------------------------------
        # 4. Observable network context
        # ---------------------------------------------------------
        self.calculate_degree_centrality()
        self.calculate_betweenness_centrality()

    def run_stance_aware(self, method="", k=1):
        for period in self.iterator(self.scenario.period_num):
            print(f"period: {period}",flush=True)
           # if period > 0 :
            if method =="stance_aware":
                print("stance_aware infection is opend"); #breakpoint()
                self.environment.agents_infection_stance_aware(self.agents, period, k)
                print("stance_aware infection is closed")
            else:
                print("Exposure-aware infection is opend")
                self.environment.agents_infection(self.agents, period)
                print("Exposure-aware infection is closed")
            self.environment.calc_population_infection_state(self.agents)
            self.data_collector.collect(period)
        self.data_collector.save()
        self.environment.save_information_analysis(self.agents, method, k)
        self.environment.save_diffusion_events(method, k)
        print("Simulation finished.", flush=True)

    def run_exposure(self, method="", k=1):
        print("self.scenario.period_num: ", self.scenario.period_num)
        for period in self.iterator(self.scenario.period_num):
            print(f"period: {period}",flush=True)
            #if period > 0 :
            if method =="stance_aware":
                print("stance_aware infection is opend"); #breakpoint()
                self.environment.agents_infection_stance_aware(self.agents, period,k)
                print("stance_aware infection is closed")
            else:
                print("Exposure-aware infection is opend")
                self.environment.agents_infection(self.agents, period)
                print("Exposure-aware infection is closed")
            self.environment.calc_population_infection_state(self.agents)
            self.data_collector.collect(period)
        self.data_collector.save()
        self.environment.save_information_analysis(self.agents, method, k)
        self.environment.save_diffusion_events(method,k)
        print("Simulation finished.", flush=True)

    def run(self, method=""):
        self.initialisation(); m=2
        self.run_stance_aware( method="stance_aware", k=m)
        self.run_exposure( method="", k=m)
          
    def load_agent_profiles(self):
        user_profiles = []
        with open("data/input/user_profile.txt", "r") as file:
            for line in file.readlines():
                if line not in ['\n', '\r\n']:
                    user_profile = line.split(": ",1)[1]
                    user_profiles.append(user_profile)

        try:
            user_embeddings = load_pickle("data/input/user_embedding.pickle")

        except:
            create_user_embedding()
            user_embeddings = load_pickle("data/input/user_embedding.pickle")

        print("len(user_profiles): ",len(user_profiles))

        for index in range (len(self.agents)):
            agent = self.agents[index]
            agent.profile = (user_profiles[index])
            agent.embedding = (user_embeddings[index])

    def select_seed_agents(self):

        negative_stance_agents = [agent for agent in self.agents if agent.stance < 0]

        seed_size = (
            self.scenario.seed_size
        )

        if len(
            negative_stance_agents
        ) < seed_size:

            raise ValueError(
                "Not enough negative-stance "
                "agents to initialize the "
                "requested seed set."
            )

        self.seed_set = random.sample(
            negative_stance_agents,
            seed_size
        )

    def initialize_seed_messages(self):
        for seed in self.seed_set:
            # ---------------------------------------
            # Initialize diffusion state
            # ---------------------------------------
            seed.preference_state = 1
            seed.post = (
                self.scenario.general_influence_msg
            )
            seed.accept = (
                self.scenario.general_influence_msg
            )
            seed.post_embedding = (
                text2embedding(seed.post)
            )
            seed.isSeed = True
            seed.adoption_time = 0
            seed.new_adopter = 0
            # ---------------------------------------
            # Message lineage
            # ---------------------------------------
            seed.message_id = (
                seed.generate_message_id()
            )
            seed.parent_message_id = None
            seed.original_message_id = (
                seed.message_id
            )

            # ---------------------------------------
            # Observable seed trace
            # ---------------------------------------
            seed.received_message_history.append({
                "period": 0,
                "source": None,
                "text": seed.post
            })
            seed.behavior_history.append({
                "period": 0,
                "source": None,
                "behavior": "seed"
            })

            seed.interaction_history.append({

                "period": 0,
                "source": None,
                "message": seed.post,
                "behavior": "seed"
            })

            # ---------------------------------------
            # Save seed message event
            # ---------------------------------------

            self.environment.message_events.append({
                "period": 0,
                "message_id": seed.message_id,
                "agent_id": seed.id,
                "parent_message_id": None,
                "original_message_id": seed.original_message_id,
                "source_agent": None,
                "accepted_message": None,
                "generated_message": seed.post,
                # Hidden state is NOT exposed
                "source_stance": None,
                "source_stance_strength": None,
                "target_stance": None,
                "target_stance_strength": None,
                "is_seed": 1 })

    def initialisation(self):

        self.load_agent_profiles()

        self.environment.initialize_observer(self.agents)

        self.select_seed_agents()

        self.initialize_seed_messages()

