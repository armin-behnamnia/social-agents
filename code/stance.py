import numpy as np


def generate_stances(
    num_agents,
    support_ratio,
    support_alpha,
    support_beta,
    oppose_alpha,
    oppose_beta
):
    """
    Generate continuous stance values in [-1, 1].

    Positive values  -> Support
    Negative values  -> Oppose
    """

    n_support = int(round(num_agents * support_ratio))
    n_oppose = num_agents - n_support

    # Support intensity: [0, 1]
    support_strength = np.random.beta(
        support_alpha,
        support_beta,
        n_support
    )

    # Oppose intensity: [0, 1]
    oppose_strength = np.random.beta(
        oppose_alpha,
        oppose_beta,
        n_oppose
    )

    # Convert opposition strength to negative stance
    support_stances = support_strength
    oppose_stances = -oppose_strength

    # Combine
    stances = np.concatenate([
        support_stances,
        oppose_stances
    ])

    # Randomly distribute across agents
    np.random.shuffle(stances)
    return stances

def update_stance(
    target_stance,
    source_stance,
    influence_strength
):
    """
    Update target stance toward source stance.
    """

    new_stance = (
        target_stance
        + influence_strength
        * (source_stance - target_stance)
    )
    print("Delta_Stance: ", float(source_stance - target_stance))
    return float(np.clip(new_stance, -1.0, 1.0))

def assign_latent_state(self):
        if self.stance > 0.5:
            self.latent_state = "Resistant"
        elif self.stance > 0:
            self.latent_state = "Neutral"
        elif self.stance > -0.5:
            self.latent_state = "Susceptive"
        else:
            self.latent_state = "Infective"

        return self.latent_state