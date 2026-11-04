import random
from pathlib import Path

from Melodie import Scenario
from source.agent import LLMSocialAgent
from Melodie import AgentList
from source.tools import text2embedding

# Change only this title to select a row from the Topics worksheet.
# Available: VoluntaryVaccination, VerifyBeforeSharing,
# HealthMisinformationCorrection, ClimateFriendlyTravel
DATA_TITLE = "SharedSkillSpaces" #"localbiz", "SharedSkillSpaces" , "LocalElections" 
# Keep topics.xlsx beside this scenario.py, independent of working directory.
DATA_FILE = Path(__file__).resolve().parent / "topics.xlsx"


def load_topic(data_file, title):
    """Read a selected topic from the Topics worksheet of an Excel workbook."""
    from openpyxl import load_workbook

    path = Path(data_file)
    if not path.is_file():
        raise FileNotFoundError(f"Topic dataset not found: {path}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if "Topics" not in workbook.sheetnames:
            raise ValueError("Topic workbook must contain a 'Topics' worksheet.")
        rows = workbook["Topics"].iter_rows(values_only=True)
        headers = next(rows, ())
        headers = [str(value).strip() if value is not None else "" for value in headers]
        nonempty_headers = [name for name in headers if name]
        if len(nonempty_headers) != len(set(nonempty_headers)):
            raise ValueError("Duplicate column names in Topics worksheet.")
        required = ("title", "stance_topic", "general_influence_msg")
        if any(field not in headers for field in required):
            raise ValueError(f"Topics worksheet requires columns: {', '.join(required)}")
        message_columns = sorted(
            (name for name in headers
             if name.startswith("sub_influence_msg_")
             and name.removeprefix("sub_influence_msg_").isdigit()),
            key=lambda name: int(name.removeprefix("sub_influence_msg_")),
        )
        if not message_columns:
            raise ValueError("Provide sub_influence_msg_1, sub_influence_msg_2, etc.")
        records = []
        for row_number, values in enumerate(rows, start=2):
            if all(value is None or value == "" for value in values):
                continue
            record = dict(zip(headers, values))
            messages = [record.get(column) for column in message_columns
                        if record.get(column) is not None and record.get(column) != ""]
            record["sub_influence_msgs"] = messages
            record["research_context"] = record.get("research_context") or ""
            records.append(record)
    finally:
        workbook.close()
    if not records:
        raise ValueError("Topics worksheet contains no scenario rows.")

    topics = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each scenario must be an object.")
        for field in ("title", "stance_topic", "general_influence_msg"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Each scenario needs a non-empty string: {field}")
        record_title = record["title"]
        if record_title in topics:
            raise ValueError(f"Duplicate topic title: {record_title}")
        messages = record.get("sub_influence_msgs")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, str) or not message.strip() for message in messages
        ):
            raise ValueError(f"Invalid sub_influence_msgs for {record_title}")
        topics[record_title] = record
    if title not in topics:
        raise ValueError(
            f"Unknown DATA_TITLE {title!r}. Available titles: {', '.join(topics)}"
        )
    return topics[title]


class LLMScenario(Scenario):
    @property
    def scenario_name(self):
        """Compatibility name; an explicitly assigned name takes precedence."""
        explicit = getattr(self, '_scenario_name', None)
        if explicit:
            return explicit
        topic = getattr(self, 'topic', None) or 'localbiz'
        message_type = (getattr(self, 'initial_message_type', None)
                        or getattr(self, 'scenario_type', 'Generic'))
        return f'{topic}_{message_type}'

    @scenario_name.setter
    def scenario_name(self, value):
        self._scenario_name = value

    def setup(self):
        self.debug = False
        self.evaluation_mode = 'reference'  # 'legacy' retains the original two-condition study
        self.forecast_horizons = [1, 2]
        self.top_k = 10
        
        #self.stance_topic = 'Using LocalBiz is beneficial for helping local businesses grow.'
        #self.stance_topic = 'Taking precautions such as ventilation, well-fitting masks in crowded indoor spaces, and staying home when sick is beneficial for reducing COVID-19 transmission.'
        self.text_stance_observer = False  # enable explicit topic-anchored LLM estimates
        self.random_seed = 10 # Change between independent repetitions.
        self.period_num: int = 10
        self.run_num: int = 1
        self.agent_num: int = 100
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
        self.initial_infected_pecentage: float = 0.05 # for 100 should change to 0.03
        self.stance_threshold = 0.7

        self.scenario_type = "Generic"  # Generic, Selective, or Generative
        self.data_title = "localbiz" # VoluntaryVaccination , VerifyBeforeSharing , HealthMisinformationCorrection , ClimateFriendlyTravel
        selected = load_topic(DATA_FILE, self.data_title)
        self.topic = selected["title"]
        self.scenario_name = f"{self.topic}_{self.scenario_type}"
        self.stance_topic = selected["stance_topic"]
        self.general_influence_msg = selected["general_influence_msg"]
        self.sub_influence_msgs = list(selected["sub_influence_msgs"])
        self.topic_research_context = selected.get("research_context", "")

        self.sub_influence_msgs_embeddings = [text2embedding(x) for x in self.sub_influence_msgs]
        self.influence_param: float = 0.2
        self.stance_influence_rate: float = 0.2
        self.id_scenario: int = 0
        self.seed_size = 10
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
