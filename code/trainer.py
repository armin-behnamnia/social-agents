"""CPU RL trainer for the elite persuader (approach 2, stage 1).

REINFORCE with a learned value baseline (advantage actor-critic
style updates). Deliberately dependency-free (pure numpy policy) so it
runs on CPU without torch/tensorflow. The action space is small
(Discrete(180)) and the observation is 13-d, so a linear/MLP softmax
policy trained with REINFORCE is sufficient and fast.

Swap-in point for approach 3 (direct language policy): replace
`RLPolicy` with a TRL PPO/GRPO trainer whose action is the message
tokens; the env interface (reset/step, reward from ground truth)
stays identical.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rl_env import ElitePersuaderEnv, OBS_DIM
from .strategy import N_STRATEGIES

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Policy network (pure numpy MLP with softmax head)
# ----------------------------------------------------------------------

class RLPolicy:
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_STRATEGIES,
                 hidden: int = 64, lr: float = 3e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        self.rng = rng
        # two-layer MLP
        self.W1 = rng.normal(0, np.sqrt(2 / obs_dim), (obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, n_actions))
        self.b2 = np.zeros(n_actions)
        # value head
        self.Wv = rng.normal(0, np.sqrt(2 / hidden), (hidden, 1))
        self.bv = np.zeros(1)

    # forward -----------------------------------------------------------

    def _hidden(self, obs: np.ndarray) -> np.ndarray:
        h = np.tanh(obs @ self.W1 + self.b1)
        return h

    def logits(self, obs: np.ndarray) -> np.ndarray:
        return self._hidden(obs) @ self.W2 + self.b2

    def probs(self, obs: np.ndarray) -> np.ndarray:
        z = self.logits(obs) - self.logits(obs).max()
        p = np.exp(z)
        return p / p.sum()

    def value(self, obs: np.ndarray) -> float:
        return float((self._hidden(obs) @ self.Wv + self.bv)[0])

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        p = self.probs(obs)
        if greedy:
            return int(np.argmax(p))
        return int(self.rng.choice(self.n_actions, p=p))

    # backward ----------------------------------------------------------

    def update(self, obs, action, advantage, entropy_coef: float = 0.01):
        obs = np.asarray(obs, dtype=np.float64)
        h = self._hidden(obs)
        logits = h @ self.W2 + self.b2
        z = logits - logits.max()
        p = np.exp(z)
        p = p / p.sum()

        dlogits = -p.copy()
        dlogits[action] += 1.0  # grad of log pi(a|s)

        # policy gradient (ascent on advantage)
        dW2 = np.outer(h, dlogits) * advantage
        db2 = dlogits * advantage

        dh = (self.W2 @ dlogits) * advantage * (1 - h * h)

        # entropy bonus gradient: dH/dlogits = -p (log p + 1)
        logp = np.log(p + 1e-12)
        dent = -p * (logp + 1.0)
        dW2 += np.outer(h, dent) * entropy_coef
        db2 += dent * entropy_coef
        dh += (self.W2 @ dent) * entropy_coef * (1 - h * h)

        dW1 = np.outer(obs, dh)
        db1 = dh

        # value regression (toward return) via simple gradient step
        v = float((h @ self.Wv + self.bv)[0])
        dv = (v - advantage)  # value target approximated by the return

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.Wv -= self.lr * 2.0 * dv * h.reshape(-1, 1)
        self.bv -= self.lr * 2.0 * dv

    def entropy(self, obs) -> float:
        p = self.probs(obs)
        return float(-(p * np.log(p + 1e-12)).sum())


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------

@dataclass
class TrainConfig:
    episodes: int = 300
    gamma: float = 0.99
    lr: float = 3e-3
    entropy_coef: float = 0.01
    hidden: int = 64
    seed: int = 0
    eval_every: int = 25
    eval_episodes: int = 10
    n_periods: int = 8
    agent_num: int = 15
    log_every: int = 10
    output_dir: str = "output/rl"
    save_path: str = "output/rl/policy.npz"


@dataclass
class EpisodeRecord:
    episode: int
    reward: float
    steps: int
    mean_stance: float
    n_adopted: int


def train(
    env: ElitePersuaderEnv,
    config: TrainConfig | None = None,
) -> tuple[RLPolicy, list[EpisodeRecord]]:
    config = config or TrainConfig()
    policy = RLPolicy(
        obs_dim=OBS_DIM,
        n_actions=N_STRATEGIES,
        hidden=config.hidden,
        lr=config.lr,
        seed=config.seed,
    )

    records: list[EpisodeRecord] = []
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for episode in range(1, config.episodes + 1):
        obs, _ = env.reset(seed=config.seed * 100003 + episode)
        trajectory = []  # (obs, action, reward)
        done = False
        while not done:
            action = policy.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append((obs, action, reward))
            obs = next_obs
            done = terminated or truncated

        # discounted returns
        returns = []
        G = 0.0
        for (_, _, r) in reversed(trajectory):
            G = r + config.gamma * G
            returns.append(G)
        returns.reverse()

        # advantage via value baseline
        for (o, a, _), G in zip(trajectory, returns):
            v = policy.value(o)
            advantage = G - v
            policy.update(o, a, advantage, entropy_coef=config.entropy_coef)

        records.append(
            EpisodeRecord(
                episode=episode,
                reward=sum(r for _, _, r in trajectory),
                steps=len(trajectory),
                mean_stance=info["mean_stance"],
                n_adopted=info["n_adopted"],
            )
        )

        if episode % config.log_every == 0:
            avg = np.mean([r.reward for r in records[-config.log_every:]])
            logger.info(
                "episode %d/%d avg_reward=%.4f mean_stance=%.3f (%.1fs elapsed)",
                episode, config.episodes, avg, records[-1].mean_stance,
                time.time() - t0,
            )

        if episode % config.eval_every == 0:
            eval_score = evaluate(env, policy, episodes=config.eval_episodes)
            logger.info("eval @ %d: greedy mean reward %.4f", episode, eval_score)

    policy.save(config.save_path)
    (out_dir / "training_log.json").write_text(
        json.dumps([vars(r) for r in records], indent=2)
    )
    return policy, records


def evaluate(env: ElitePersuaderEnv, policy: RLPolicy,
             episodes: int = 10, greedy: bool = True) -> float:
    """Mean episode reward under the greedy policy."""
    total = 0.0
    for k in range(episodes):
        obs, _ = env.reset(seed=900_000 + k)
        done = False
        while not done:
            action = policy.act(obs, greedy=greedy)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    return total / episodes


# ----------------------------------------------------------------------
# Message-bank baseline comparison
# ----------------------------------------------------------------------

def evaluate_message_bank(env: ElitePersuaderEnv, seed: int = 0) -> float:
    """Random-policy baseline over the strategy space (uniform).

    Comparable to the paper's fixed message bank: no learning, fixed
    or uniformly random strategy selection.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    episodes = 10
    for k in range(episodes):
        obs, _ = env.reset(seed=700_000 + k)
        done = False
        while not done:
            action = int(rng.integers(N_STRATEGIES))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    return total / episodes


# persistence ----------------------------------------------------------

def _policy_save(self: RLPolicy, path: str | Path):
    np.savez(
        path,
        W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
        Wv=self.Wv, bv=self.bv,
    )


def _policy_load(self: RLPolicy, path: str | Path) -> RLPolicy:
    data = np.load(path)
    self.W1 = data["W1"]; self.b1 = data["b1"]
    self.W2 = data["W2"]; self.b2 = data["b2"]
    self.Wv = data["Wv"]; self.bv = data["bv"]
    return self


RLPolicy.save = _policy_save
RLPolicy.load = _policy_load
