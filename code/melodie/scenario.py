from Melodie import Scenario
from source.agent import LLMSocialAgent
from Melodie import AgentList
from source.tools import text2embedding

class LLMScenario(Scenario):
    def setup(self):
        self.debug = False
        self.period_num: int = 10
        self.run_num: int = 1
        self.agent_num: int = 50
        # Inference
        self.num_particles = 10
        self.inference_state_min = -1.0
        self.inference_state_max = 1.0
        self.process_noise = 0.08
        self.adoption_threshold = 0.0
        self.network_type: str = "erdos_renyi_graph"
        self.network_param_p: float = 0.05
        self.social_influence_param: float = 0.33
        self.content_influence_param: float = 0.34
        self.network_influence_param: float = 0.33
        self.network_param_directed: str = "True"
        self.alpha_profile = 0.2  
        self.alpha_content = 0.2
        self.alpha_stance = 0.2
        self.alpha_degree = 0.2
        self.alpha_degree_stance = 0.2
        self.initial_infected_pecentage: float = 0.1 # for 100 should change to 0.03
        self.stance_threshold = 0.7
        self.scenario_type = "Generic" #"Selective" , "Generative", "RL_Generative"
        self.general_influence_msg: str = "Discover LocalBiz, a service designed to help local businesses grow and succeed. Explore practical solutions for your business and discover new opportunities for growth. #LocalBiz"
        self.sub_influence_msgs =["LocalBiz helps small businesses improve their online presence and attract more customers.","LocalBiz offers affordable digital tools to help businesses manage costs and improve efficiency.","LocalBiz connects local businesses with new customers and supports local economic growth."]
        self.sub_influence_msgs_embeddings = [text2embedding(x) for x in self.sub_influence_msgs]
        self.influence_param: float = 0.2
        self.stance_influence_rate: float = 0.2
        self.id_scenario: int = 0
        self.seed_size = 5
        #self.agents = AgentList(LLMSocialAgent)
        Scenario.stance_params = {
        "support_ratio": 0.60,
        "support_distribution": {
        "alpha": 6,
        "beta": 2},
        "oppose_distribution": {
        "alpha": 8,
        "beta": 2}}
        if self.debug:
            self.period_num: int = 2
            self.run_num: int = 1
            self.agent_num: int = 10
            self.initial_infected_pecentage: float = 0.3
            self.seed_size =3
            Scenario.stance_params = {
            "support_ratio": 0.60,
            "support_distribution": {
            "alpha": 6,
            "beta": 2},
            "oppose_distribution": {
            "alpha": 2,
            "beta": 7}}
        # self.stance_strength_method = "degree"
        self.alpha_degree = 0.2
        self.observation_window = 5
        self.observation_beta = 3.0
        self.resampling_threshold = 0.5
        #self.initialize_agents(self.agents)

    def get_network_params(self):
        if self.network_type == "erdos_renyi_graph":
            network_params = {"p": self.network_param_p, "directed": self.network_param_directed}
        elif self.network_type == "facebook":
            network_params = {}
        else:
            raise NotImplementedError
        return network_params

    def initialize_agents(self, agents: "AgentList[LLMSocialAgent]"):
        """
        مقداردهی اولیه agents با استفاده از seed_size
        این تابع رو بعد از ساخت agents صدا بزن
        """
        print(f"🔄 Initializing {len(agents)} agents...")
        
        # ۱. اول setup هر agent رو اجرا کن
        for agent in agents:
            agent.setup()
        
        # ۲. حالا seedها رو مشخص کن
        num_seeds = int(len(agents) * self.seed_size)
        print(f"📊 Setting {num_seeds} seeds ({self.seed_size*100}%)...")
        negative_agents = [
            agent for agent in agents
            if agent.stance < 0
            ]
        seed_agents = random.sample(negative_agents, num_seeds)
        for agent in seed_agents :
            agent.isSeed = True
            agent.preference_state = 1  # seedها از قبل آلوده
            agent.post = self.influence_msg #f"Seed Post {i+1}"
            agent.accept = agent.post
            print(f"   ✅ Agent {agent.id}: SEED")
        
        print(f"✅ Initialization complete!")
