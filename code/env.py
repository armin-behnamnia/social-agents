"""Gymnasium environment wrapper around the diffusion simulation.

This is the RL entry point (Option B: persuasion agents). The
environment exposes the *persuader-side* control problem:

- action: a seed message (index into a message bank, or free text via
  `action_space=Discrete` over a configurable bank), delivered by the
  seed agent at the start of an episode;
- reward: realized latent stance shift of the target population
  (measured from simulator ground truth), optionally discounted across
  periods to credit multi-hop propagation;
- episode: one full diffusion run over `n_periods`.

A finer-grained per-message policy interface lives in
`persuader_policy.py` (message generation as the action space).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .scenario import LLMScenario
from .simulation import Simulation

DEFAULT_MESSAGE_BANK = [
    "I believe this policy is harmful for our community. The evidence is "
    "clear and people deserve to know the truth.",
    "Have you actually read the studies on this? The data show the opposite "
    "of what officials claim. Ask yourself who benefits.",
    "My neighbor worked in this field for 20 years and says the same thing: "
    "this policy quietly hurts ordinary families like ours.",
    "Everyone I talk to is starting to doubt this. It is not about left or "
    "right, it is about basic accountability and real numbers.",
    "Sharing this again because it matters: independent auditors found "
    "serious flaws. Ignoring it will not make the problems go away.",
]


class LatentStanceDiffusionEnv(gym.Env):
    """Seeding/persuasion control environment.

    action: int in [0, len(message_bank)) selecting the seed message,
        OR a string when `free_text_action=True` (the message itself).
    observation: [n_agents] vector of latent stances visible ONLY when
        `observe_latent=True` (default False: RL agent sees the same
        observable-side features as the external observer).
    reward: negative mean stance of the population (persuasion toward
        the source's negative stance), summed over periods, optionally
        discounted by `reward_gamma`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: LLMScenario | None = None,
        n_periods: int = 10,
        message_bank: list[str] | None = None,
        reward_gamma: float = 1.0,
        observe_latent: bool = False,
        topology: str = "watts_strogatz",
        initial_stances: np.ndarray | None = None,
        free_text_action: bool = False,
    ):
        super().__init__()
        self.scenario = scenario or LLMScenario()
        self.n_periods = n_periods
        self.message_bank = list(message_bank or DEFAULT_MESSAGE_BANK)
        self.reward_gamma = reward_gamma
        self.observe_latent = observe_latent
        self.topology = topology
        self.initial_stances = initial_stances
        self.free_text_action = free_text_action

        if free_text_action:
            self.action_space = spaces.Text(max_length=500)
        else:
            self.action_space = spaces.Discrete(len(self.message_bank))

        obs_dim = self.scenario.agent_num
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._sim: Simulation | None = None
        self._step_count = 0
        self._episode_seed: int | None = None

    # ------------------------------------------------------------------

    def _make_sim(self) -> Simulation:
        episode_scenario = LLMScenario(**vars(self.scenario))
        episode_scenario.rng_seed = self._episode_seed
        return Simulation(
            scenario=episode_scenario,
            topology=self.topology,
            initial_stances=self.initial_stances,
        )

    def _observation(self) -> np.ndarray:
        assert self._sim is not None
        if self.observe_latent:
            return self._sim.stances().astype(np.float32)
        # Observer-side features: stance labels are hidden; expose
        # observable proxies (adoption state mapped to [-1, 1]).
        obs = np.array(
            [
                -1.0 if a.preference_state == 1 else 0.0
                for a in self._sim.agents
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._episode_seed = seed
        if seed is None:
            self._episode_seed = int(np.random.default_rng().integers(0, 2**31))
        self._sim = self._make_sim()
        self._step_count = 0
        return self._observation(), {}

    def step(self, action):
        assert self._sim is not None, "call reset() first"

        if self._step_count == 0:
            # The chosen action seeds the initial message.
            if self.free_text_action:
                message = str(action)
            else:
                message = self.message_bank[int(action)]
            seed_agent = self._sim.agents[0]
            seed_agent.post = message
            seed_agent.post_embedding = seed_agent._get_embedding(message)

        self._sim.step()
        self._step_count += 1

        obs = self._observation()
        mean_stance = float(self._sim.stances().mean())
        reward = -mean_stance  # persuasion reward: drive population negative
        terminated = self._step_count >= self.n_periods
        truncated = False

        info: dict[str, Any] = {
            "period": self._sim.period,
            "mean_stance": mean_stance,
            "n_adopted": self._sim.summary()["n_adopted"],
        }
        return obs, float(reward), terminated, truncated, info
