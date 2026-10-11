"""GPU PPO trainer for the elite persuader (approach 2, stage 1).

Torch-based replacement/upgrade of the numpy REINFORCE trainer:

- MLP actor-critic (obs -> hidden -> policy head + value head),
  trained with clipped-surrogate PPO on a GPU (H200-class) device.
- Supports the compound action space of `ElitePersuaderEnv` in learned
  mode: Discrete(n_agents * N_STRATEGIES) with per-step action masks
  (`info["action_mask"]`), so the policy learns BOTH which target node
  to address and which persuasion strategy to use.
- Vectorized rollout collection: multiple independent episode seeds
  per PPO iteration (synchronous; the simulator itself is CPU-side,
  the network updates are GPU-side).
- Checkpoints, training log JSON, periodic greedy evaluation, and a
  random-strategy baseline (message bank) for comparison.

The paper's final goal — post content to target nodes to maximize
network-wide influence — is operationalized as: reward = ground-truth
stance movement of targeted agents toward the persuader (+ optional
cascade term for whole-network movement).
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch is required for PPO training (pip install torch)"
    ) from exc

from .rl_env import ElitePersuaderEnv
from .strategy import N_STRATEGIES

logger = logging.getLogger(__name__)


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------
# Actor-critic network
# ----------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        # obs may be a flat vector or a (n_agents, d) matrix; flatten
        # both to a single feature vector.
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)
        self.n_actions = n_actions

    def _flatten(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() > 2:
            return obs.flatten(1)
        if obs.dim() == 1 and self.body[0].in_features != obs.shape[0]:
            # matrix obs passed unbatched: flatten
            return obs.reshape(-1)
        return obs

    def forward(self, obs: torch.Tensor):
        h = self.body(self._flatten(obs))
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor | None = None,
        greedy: bool = False,
    ) -> int:
        """Sample (or argmax) an action; mask is bool [n_actions]."""
        with torch.no_grad():
            logits, _ = self.forward(obs.unsqueeze(0))
            logits = logits.squeeze(0)
            if mask is not None:
                neg = torch.finfo(logits.dtype).min
                logits = torch.where(mask, logits, torch.full_like(logits, neg))
            if greedy:
                return int(logits.argmax().item())
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().item())

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor
    ):
        logits, values = self.forward(obs)
        neg = torch.finfo(logits.dtype).min
        logits = torch.where(masks.bool(), logits, torch.full_like(logits, neg))
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class PPOConfig:
    # environment
    n_periods: int = 10
    agent_num: int = 30
    n_persuaders: int = 1  # RL-controlled seed agents (shared policy)
    topology: str = "watts_strogatz"
    topic: str = "the policy"
    reward_cascade_discount: float = 0.5  # credit network-wide movement
    message_cost: float = 0.0
    shaping_accept_bonus: float = 0.05
    persuade_all_neighbors: bool = False

    # training
    iterations: int = 200
    episodes_per_iteration: int = 8  # rollout episodes per update
    rollout_workers: int = 1  # >1: thread-per-episode parallel rollouts
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    ppo_epochs: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    seed: int = 0
    target_kl: float | None = 0.05

    # evaluation / logging
    eval_every: int = 20
    eval_episodes: int = 10
    log_every: int = 5
    output_dir: str = "output/ppo"
    device: str | None = None  # None -> auto (cuda if available)

    def n_actions(self) -> int:
        # learned compound action space
        return self.agent_num * N_STRATEGIES


# ----------------------------------------------------------------------
# Rollout collection
# ----------------------------------------------------------------------

def collect_episode(
    env: ElitePersuaderEnv,
    policy: ActorCritic,
    device: torch.device,
    seed: int,
    greedy: bool = False,
):
    """Run one episode; returns (obs, actions, rewards, masks, infos).

    Multi-persuader: env.n_persuaders > 1 means each period expands
    into one policy decision per persuader (shared policy; action mask
    is identical across persuaders, observations differ only in the
    is_acting column). Each decision is stored as its own transition
    with its own reward component, so an episode of P periods with N
    persuaders yields P*N transitions.
    """
    n_persuaders = int(getattr(env, "n_persuaders", 1))
    obs_list, act_list, rew_list, mask_list = [], [], [], []
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        mask = info.get("action_mask")
        if mask is None:
            mask = np.ones(policy.n_actions, dtype=bool)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)

        if n_persuaders == 1:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action = policy.act(obs_t, mask=mask_t, greedy=greedy)
            actions = action
            step_obs = [(obs, action, mask)]
        else:
            actions, step_obs = [], []
            for i in range(n_persuaders):
                obs_i = env.observation(actor=i)
                obs_t = torch.as_tensor(
                    obs_i, dtype=torch.float32, device=device
                )
                action = policy.act(obs_t, mask=mask_t, greedy=greedy)
                actions.append(action)
                step_obs.append((obs_i, action, mask))

        obs, reward, terminated, truncated, info = env.step(actions)
        for (step_obs_i, act, m), rew in zip(
            step_obs, np.atleast_1d(reward)
        ):
            obs_list.append(step_obs_i)
            act_list.append(act)
            rew_list.append(rew)
            mask_list.append(m)
        done = terminated or truncated
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(act_list, dtype=np.int64),
        np.asarray(rew_list, dtype=np.float32),
        np.asarray(mask_list, dtype=bool),
        info,
    )


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, last_value: float,
    gamma: float, lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard GAE: returns advantages and returns-to-go."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t < T - 1 else last_value
        delta = rewards[t] + gamma * next_value - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------

class PPOTrainer:
    def __init__(
        self,
        env: ElitePersuaderEnv,
        config: PPOConfig | None = None,
        env_factory=None,
    ):
        """env_factory: () -> ElitePersuaderEnv, used to build one env
        per rollout worker (parallel mode). Falls back to env.clone()."""
        self.env = env
        self.config = config or PPOConfig()
        self.env_factory = env_factory
        self.device = resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        obs_shape = env.observation_space.shape
        obs_dim = int(np.prod(obs_shape))  # matrix obs: n_agents * d
        n_actions = int(env.action_space.n)
        self.policy = ActorCritic(
            obs_dim, n_actions, hidden=self.config.hidden
        ).to(self.device)

        # per-worker env + a CPU-side copy of the policy for acting
        # (thread-safe reads; weights refreshed before each rollout)
        self._worker_envs: list[ElitePersuaderEnv] | None = None
        self._worker_policies: list[ActorCritic] | None = None
        if self.config.rollout_workers > 1:
            self._worker_envs = (
                [env_factory() for _ in range(self.config.rollout_workers)]
                if env_factory
                else [env.clone() for _ in range(self.config.rollout_workers)]
            )
            self._worker_policies = []
            for _ in range(self.config.rollout_workers):
                p = ActorCritic(obs_dim, n_actions, hidden=self.config.hidden)
                p.load_state_dict(self.policy.state_dict())
                p.to("cpu")
                self._worker_policies.append(p)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.body.parameters()},
                {
                    "params": self.policy.policy_head.parameters(),
                    "lr": self.config.actor_lr,
                },
                {
                    "params": self.policy.value_head.parameters(),
                    "lr": self.config.critic_lr,
                },
            ]
        )

    # ------------------------------------------------------------------

    def _rollout_batch(self):
        """Collect `episodes_per_iteration` episodes.

        rollout_workers > 1: episodes run in a thread pool (one env per
        thread; LLM renders + simulator steps release the GIL on I/O
        and numpy work), with a CPU copy of the current policy.
        """
        if self.config.rollout_workers > 1:
            return self._rollout_batch_parallel()

        all_obs, all_act, all_rew, all_mask = [], [], [], []
        all_adv, all_ret = [], []
        episode_stats = []
        for k in range(self.config.episodes_per_iteration):
            seed = (
                self.config.seed * 100_003
                + self._iteration * 1_000
                + k
            )
            obs, act, rew, mask, info = collect_episode(
                self.env, self.policy, self.device, seed
            )
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device)
                _, values = self.policy.forward(obs_t)
            values_np = values.cpu().numpy()
            adv, ret = compute_gae(
                rew, values_np, 0.0, self.config.gamma, self.config.gae_lambda
            )
            all_obs.append(obs)
            all_act.append(act)
            all_rew.append(rew)
            all_mask.append(mask)
            all_adv.append(adv)
            all_ret.append(ret)
            episode_stats.append(
                {
                    "episode_reward": float(rew.sum()),
                    "mean_stance": info.get("mean_stance"),
                    "n_adopted": info.get("n_adopted"),
                    "steps": len(rew),
                }
            )

        batch = {
            "obs": np.concatenate(all_obs),
            "act": np.concatenate(all_act),
            "mask": np.concatenate(all_mask),
            "adv": np.concatenate(all_adv),
            "ret": np.concatenate(all_ret),
        }
        return batch, episode_stats

    def _rollout_batch_parallel(self):
        n_workers = self.config.rollout_workers

        # refresh worker policy weights (CPU copies)
        state = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        for p in self._worker_policies:
            p.load_state_dict(state)

        def run_one(k: int):
            seed = (
                self.config.seed * 100_003
                + self._iteration * 1_000
                + k
            )
            env = self._worker_envs[k % n_workers]
            policy = self._worker_policies[k % n_workers]
            return collect_episode(env, policy, torch.device("cpu"), seed)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(
                pool.map(run_one, range(self.config.episodes_per_iteration))
            )

        all_obs, all_act, all_rew, all_mask = [], [], [], []
        all_adv, all_ret = [], []
        episode_stats = []
        for obs, act, rew, mask, info in results:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self.device)
                _, values = self.policy.forward(obs_t)
            values_np = values.cpu().numpy()
            adv, ret = compute_gae(
                rew, values_np, 0.0, self.config.gamma, self.config.gae_lambda
            )
            all_obs.append(obs)
            all_act.append(act)
            all_rew.append(rew)
            all_mask.append(mask)
            all_adv.append(adv)
            all_ret.append(ret)
            episode_stats.append(
                {
                    "episode_reward": float(rew.sum()),
                    "mean_stance": info.get("mean_stance"),
                    "n_adopted": info.get("n_adopted"),
                    "steps": len(rew),
                }
            )

        batch = {
            "obs": np.concatenate(all_obs),
            "act": np.concatenate(all_act),
            "mask": np.concatenate(all_mask),
            "adv": np.concatenate(all_adv),
            "ret": np.concatenate(all_ret),
        }
        return batch, episode_stats

    # ------------------------------------------------------------------

    def _update(self, batch) -> dict:
        obs = torch.as_tensor(batch["obs"], device=self.device)
        act = torch.as_tensor(batch["act"], device=self.device)
        mask = torch.as_tensor(batch["mask"], device=self.device)
        adv = torch.as_tensor(batch["adv"], device=self.device)
        ret = torch.as_tensor(batch["ret"], device=self.device)

        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        with torch.no_grad():
            old_log_probs, _, _ = self.policy.evaluate_actions(obs, act, mask)

        N = obs.shape[0]
        idx = np.arange(N)
        metrics = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n_updates = 0
        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(idx)
            approx_kl = 0.0
            for start in range(0, N, self.config.minibatch_size):
                mb = idx[start : start + self.config.minibatch_size]
                log_probs, entropy, values = self.policy.evaluate_actions(
                    obs[mb], act[mb], mask[mb]
                )
                ratio = torch.exp(log_probs - old_log_probs[mb])
                surr1 = ratio * adv[mb]
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.clip_ratio,
                        1 + self.config.clip_ratio,
                    )
                    * adv[mb]
                )
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((values - ret[mb]) ** 2).mean()
                loss = (
                    pi_loss
                    + self.config.value_coef * v_loss
                    - self.config.entropy_coef * entropy.mean()
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (
                        (old_log_probs[mb] - log_probs).mean().item()
                    )
                metrics["pi_loss"] += pi_loss.item()
                metrics["v_loss"] += v_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["kl"] = max(metrics["kl"], abs(approx_kl))
                n_updates += 1
            if (
                self.config.target_kl is not None
                and abs(approx_kl) > self.config.target_kl
            ):
                break

        for k in metrics:
            metrics[k] /= max(1, n_updates)
        return metrics

    # ------------------------------------------------------------------

    def train(self) -> tuple[ActorCritic, list[dict]]:
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict] = []
        self._iteration = 0
        t0 = time.time()
        for iteration in range(1, cfg.iterations + 1):
            self._iteration = iteration
            batch, episode_stats = self._rollout_batch()
            metrics = self._update(batch)

            mean_reward = float(
                np.mean([s["episode_reward"] for s in episode_stats])
            )
            mean_stance = float(
                np.mean(
                    [s["mean_stance"] for s in episode_stats if s["mean_stance"] is not None]
                )
            )
            rec = {
                "iteration": iteration,
                "mean_episode_reward": mean_reward,
                "mean_final_stance": mean_stance,
                "mean_adopted": float(
                    np.mean(
                        [s["n_adopted"] for s in episode_stats if s["n_adopted"] is not None]
                    )
                ),
                **{f"train_{k}": v for k, v in metrics.items()},
                "elapsed_s": time.time() - t0,
            }
            history.append(rec)

            if iteration % cfg.log_every == 0:
                logger.info(
                    "iter %d/%d reward=%.4f final_stance=%.3f adopted=%.1f "
                    "pi_loss=%.4f v_loss=%.4f (%.1fs)",
                    iteration, cfg.iterations, mean_reward, mean_stance,
                    rec["mean_adopted"], metrics["pi_loss"], metrics["v_loss"],
                    rec["elapsed_s"],
                )

            if iteration % cfg.eval_every == 0:
                score = self.evaluate(episodes=cfg.eval_episodes)
                history[-1]["eval_greedy_reward"] = score
                logger.info(
                    "eval @ %d: greedy mean reward %.4f", iteration, score
                )

            # checkpoint
            if iteration % max(1, cfg.iterations // 10) == 0 or iteration == cfg.iterations:
                self.save(out_dir / "policy.pt")

        self.save(out_dir / "policy.pt")
        (out_dir / "training_log.json").write_text(
            json.dumps(
                {"config": asdict(cfg), "history": history}, indent=2
            )
        )
        return self.policy, history

    # ------------------------------------------------------------------

    def evaluate(self, episodes: int = 10, greedy: bool = True) -> float:
        """Mean episode reward; parallel across worker envs when configured."""
        if self.config.rollout_workers > 1 and self._worker_envs:
            state = {k: v.cpu() for k, v in self.policy.state_dict().items()}
            for p in self._worker_policies:
                p.load_state_dict(state)

            def run_one(k: int):
                seed = 900_000 + k
                env = self._worker_envs[k % len(self._worker_envs)]
                policy = self._worker_policies[k % len(self._worker_policies)]
                _, _, rew, _, _ = collect_episode(
                    env, policy, torch.device("cpu"), seed, greedy=greedy
                )
                return float(rew.sum())

            with ThreadPoolExecutor(
                max_workers=len(self._worker_envs)
            ) as pool:
                totals = list(pool.map(run_one, range(episodes)))
            return float(np.mean(totals))

        total = 0.0
        for k in range(episodes):
            seed = 900_000 + k
            _, _, rew, _, _ = collect_episode(
                self.env, self.policy, self.device, seed, greedy=greedy
            )
            total += float(rew.sum())
        return total / episodes

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model_state": self.policy.state_dict(),
                "n_actions": self.policy.n_actions,
                "obs_dim": self.policy.body[0].in_features,
                "hidden": self.config.hidden,
            },
            path,
        )

    @classmethod
    def load_policy(
        cls, path: str | Path, device: str | None = None
    ) -> ActorCritic:
        ckpt = torch.load(path, map_location=resolve_device(device))
        policy = ActorCritic(
            ckpt["obs_dim"], ckpt["n_actions"], hidden=ckpt["hidden"]
        )
        policy.load_state_dict(ckpt["model_state"])
        return policy.to(resolve_device(device))


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------

def evaluate_random_baseline(
    env: ElitePersuaderEnv,
    episodes: int = 10,
    seed: int = 0,
    workers: int = 1,
) -> float:
    """Uniform-random compound actions (no learning control).

    Parallel-safe: each thread gets a cloned env and its own RNG.
    """
    rng = np.random.default_rng(seed)

    def run_one(k: int) -> float:
        local_env = env.clone() if workers > 1 else env
        local_rng = np.random.default_rng(seed * 1000 + k)
        n_persuaders = int(getattr(local_env, "n_persuaders", 1))
        obs, info = local_env.reset(seed=700_000 + k)
        total, done = 0.0, False
        while not done:
            mask = info.get("action_mask")
            n = local_env.action_space.n
            actions = []
            for _ in range(n_persuaders):
                if mask is not None and not mask.all():
                    valid = np.flatnonzero(mask)
                    actions.append(int(local_rng.choice(valid)))
                else:
                    actions.append(int(local_rng.integers(n)))
            action = actions[0] if n_persuaders == 1 else actions
            obs, reward, terminated, truncated, info = local_env.step(action)
            total += float(np.sum(np.atleast_1d(reward)))
            done = terminated or truncated
        return total

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            totals = list(pool.map(run_one, range(episodes)))
    else:
        totals = [run_one(k) for k in range(episodes)]
    return float(np.mean(totals))


def evaluate_heuristic_baseline(
    env: ElitePersuaderEnv, episodes: int = 10, workers: int = 1
) -> float:
    """Always target the first (lowest-id reachable) candidate with a
    fixed mid-range strategy — the 'fixed message bank' analogue."""

    def run_one(k: int) -> float:
        local_env = env.clone() if workers > 1 else env
        n_persuaders = int(getattr(local_env, "n_persuaders", 1))
        obs, info = local_env.reset(seed=600_000 + k)
        done, total = False, 0.0
        from .strategy import N_STRATEGIES as n_strats

        single_action = n_strats // 2  # first target, mid strategy
        action = (
            single_action
            if n_persuaders == 1
            else [single_action] * n_persuaders
        )
        while not done:
            obs, reward, terminated, truncated, info = local_env.step(action)
            total += float(np.sum(np.atleast_1d(reward)))
            done = terminated or truncated
        return total

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            totals = list(pool.map(run_one, range(episodes)))
    else:
        totals = [run_one(k) for k in range(episodes)]
    return float(np.mean(totals))
