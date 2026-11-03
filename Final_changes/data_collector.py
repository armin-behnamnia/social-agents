from Melodie import DataCollector


class LLMSocialNetworkDataCollector(DataCollector):

    def setup(self):

        # -----------------------------
        # Agent state
        # -----------------------------
        self.add_agent_property(
            "agents",
            "preference_state"
        )

        self.add_agent_property(
            "agents",
            "post"
        )

        # -----------------------------
        # Stance
        # -----------------------------
        self.add_agent_property(
            "agents",
            "stance"
        )

        self.add_agent_property(
            "agents",
            "support_probability"
        )

        # -----------------------------
        # Network / diffusion
        # -----------------------------
        self.add_agent_property(
            "agents",
            "degree"
        )

        self.add_agent_property(
            "agents",
            "num_infected_neighbors"
        )

        self.add_agent_property(
            "agents",
            "exposure_count"
        )

        self.add_agent_property(
            "agents",
            "new_adopter"
        )

        self.add_agent_property(
            "agents",
            "adoption_time"
        )

        # -----------------------------
        # Environment
        # -----------------------------
        self.add_environment_property("s0")

        self.add_environment_property("s1")

        self.add_environment_property(
            "alteration_degree"
        )

        #self.add_environment_property(
        #    "Resistant"
        #)

        #self.add_environment_property(
        #    "Neutral"
        #)

        #self.add_environment_property(
        #    "Susceptive"
        #)

        #self.add_environment_property(
        #    "Infective"
        #)