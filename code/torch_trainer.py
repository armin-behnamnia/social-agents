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
from .strategy import N_STRATEGIES, STRATEGY_DIMS

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
    """MLP actor-critic with an optional *factorized* policy head.

    The action space factorizes as ``target (n_targets) x strategy
    (N_STRATEGIES)`` (compound ``agent_id * N_STRATEGIES + strategy``,
    see `rl_env._decode_action`), and the strategy itself factorizes as
    the Cartesian product of four categorical dimensions
    (`strategy.STRATEGY_DIMS`). A flat 180-way (or agent_num*180-way)
    softmax has to explore and score every combination independently;
    four small heads (appeal x tone x emphasis x intensity) plus a
    target head share statistical strength across combinations and
    massively shrink the effective parameter count. The joint logits are
    reconstructed exactly (sum of the per-factor logits at each combo),
    so the interface is unchanged: `forward` returns
    ``(logits [B, n_actions], value [B])``.

    ``factorized`` is auto-detected when None: factorize iff the action
    space divides evenly into ``target x N_STRATEGIES`` (this includes
    the strategy-only ``random_neighbor`` case, where n_targets == 1).
    Non-divisible action spaces fall back to a flat head, so the class
    stays backwards-compatible with arbitrary test spaces.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 128,
        factorized: bool | None = None,
    ):
        super().__init__()
        # obs may be a flat vector or a (n_agents, d) matrix; flatten
        # both to a single feature vector.
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.n_actions = n_actions

        n_targets = (
            n_actions // N_STRATEGIES if n_actions % N_STRATEGIES == 0 else 0
        )
        if factorized is None:
            factorized = n_targets >= 1
        self.factorized = bool(factorized and n_targets >= 1)
        self.n_targets = int(n_targets) if self.factorized else 0

        if self.factorized:
            self.target_head = (
                nn.Linear(hidden, self.n_targets)
                if self.n_targets > 1
                else None
            )
            self.strategy_heads = nn.ModuleList(
                [nn.Linear(hidden, d) for d in STRATEGY_DIMS]
            )
        else:
            self.policy_head = nn.Linear(hidden, n_actions)

    def _flatten(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() > 2:
            return obs.flatten(1)
        if obs.dim() == 1 and self.body[0].in_features != obs.shape[0]:
            # matrix obs passed unbatched: flatten
            return obs.reshape(-1)
        return obs

    def _strategy_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Combine the four categorical heads into (..., N_STRATEGIES).

        The reduction order matches `strategy.np_ravel`/`np_unravel`
        (row-major over STRATEGY_DIMS), so logit column `k` is exactly
        strategy index `k`.
        """
        logits = self.strategy_heads[0](h)  # (..., d0)
        for head in self.strategy_heads[1:]:
            logits = logits.unsqueeze(-1) + head(h).unsqueeze(-2)
            logits = logits.flatten(-2)  # (..., prod dims so far)
        return logits

    def _policy_logits(self, h: torch.Tensor) -> torch.Tensor:
        if not self.factorized:
            return self.policy_head(h)
        strat = self._strategy_logits(h)  # (..., N_STRATEGIES)
        if self.target_head is None:
            return strat
        tgt = self.target_head(h)  # (..., n_targets)
        # target * N_STRATEGIES + strategy, matching the env encoding
        return (tgt.unsqueeze(-1) + strat.unsqueeze(-2)).flatten(-2)

    def forward(self, obs: torch.Tensor):
        h = self.body(self._flatten(obs))
        return self._policy_logits(h), self.value_head(h).squeeze(-1)

    def policy_parameters(self):
        """Actor-side parameters (for the optimizer's actor group)."""
        if self.factorized:
            params = list(self.strategy_heads.parameters())
            if self.target_head is not None:
                params += list(self.target_head.parameters())
            return params
        return list(self.policy_head.parameters())


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
    # Entropy schedule — a fixed tiny entropy bonus under-explored the
    # action space and let the policy collapse onto one strategy family.
    # When both start and end are set the coefficient is linearly
    # annealed over `iterations` (high early, low late) and
    # `entropy_coef` is ignored. Set either to None to use the constant.
    entropy_coef_start: float | None = 0.05
    entropy_coef_end: float | None = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    factorized_policy: bool = True  # target x strategy factorized heads
    seed: int = 0
    target_kl: float | None = 0.05

    # evaluation / logging
    eval_every: int = 20
    eval_episodes: int = 10
    log_every: int = 5
    # Tier-5 measurement: paired greedy/random/heuristic evaluation on
    # shared episode seeds every `paired_eval_every` iterations (0=off).
    # Paired seeds remove between-seed variance, so this is a far more
    # reliable model-selection signal than unpaired per-run evals.
    paired_eval_every: int = 0
    paired_eval_episodes: int = 20
    output_dir: str = "output/ppo"
    device: str | None = None  # None -> auto (cuda if available)

    def n_actions(self) -> int:
        # learned compound action space
        return self.agent_num * N_STRATEGIES


def _ckpt_is_factorized(ckpt: dict) -> bool:
    """Infer whether a checkpoint used a factorized policy head."""
    if "factorized" in ckpt and ckpt["factorized"] is not None:
        return bool(ckpt["factorized"])
    return "policy_head.weight" not in ckpt.get("model_state", {})


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


# ----------------------------------------------------------------------
# Rollout collection
# ----------------------------------------------------------------------

def collect_episode(
    env: ElitePersuaderEnv,
    policy: ActorCritic,
    device: torch.device,
    seed: int,
    greedy: bool = False,
    stats: list | None = None,
):
    """Run one episode; returns (obs, actions, rewards, masks, infos).

    Multi-persuader: env.n_persuaders > 1 means each period expands
    into one policy decision per persuader (shared policy; action mask
    is identical across persuaders, observations differ only in the
    is_acting column). Each decision is stored as its own transition
    with its own reward component, so an episode of P periods with N
    persuaders yields P*N transitions.

    If `stats` is a list, one small record is appended per policy
    decision: ``{strategy_index, reward, persuasiveness}``. This feeds
    the Tier-5 instrumentation (per-strategy value table, judge-score
    distribution) without changing the return signature.
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
        if stats is not None:
            rew_arr = np.atleast_1d(reward)
            for pidx, rec in enumerate(info.get("persuader_actions", [])):
                if pidx >= len(rew_arr):
                    break
                score = rec.get("persuasiveness")
                stats.append(
                    {
                        "strategy_index": int(rec["strategy_index"]),
                        "reward": float(rew_arr[pidx]),
                        "persuasiveness": (
                            None if score is None else float(score)
                        ),
                        "judge_src": rec.get("judge_src"),
                        "goal_align": rec.get("goal_align"),
                    }
                )
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
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.policy = self._build_policy().to(self.device)

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
            self._worker_policies = [
                self._build_policy().to("cpu")
                for _ in range(self.config.rollout_workers)
            ]
            for p in self._worker_policies:
                p.load_state_dict(self.policy.state_dict())

        self.optimizer = self._build_optimizer()

        # Tier-5 instrumentation: running per-strategy reward / count
        # (a "strategy value table": mean reward per action) and the
        # judge-score history, so collapse / saturation is visible.
        self._strategy_value_sum = np.zeros(N_STRATEGIES, dtype=np.float64)
        self._strategy_value_count = np.zeros(N_STRATEGIES, dtype=np.int64)
        self._judge_scores: list[float] = []
        # identity-vs-content lookup counters for the BT reward model:
        # any "text" means the id index missed and we silently fell back.
        self._judge_src_counts: dict[str, int] = {"id": 0, "text": 0, "none": 0}
        # running mean of the message goal-alignment a(m) (Track B)
        self._goal_align_sum = 0.0
        self._goal_align_count = 0

    # ------------------------------------------------------------------

    def _build_policy(self) -> ActorCritic:
        return ActorCritic(
            self.obs_dim,
            self.n_actions,
            hidden=self.config.hidden,
            factorized=self.config.factorized_policy,
        )

    def _build_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            [
                {"params": self.policy.body.parameters()},
                {
                    "params": self.policy.policy_parameters(),
                    "lr": self.config.actor_lr,
                },
                {
                    "params": self.policy.value_head.parameters(),
                    "lr": self.config.critic_lr,
                },
            ]
        )

    # ------------------------------------------------------------------
    # Tier-5 instrumentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize(values) -> dict:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.size == 0:
            return {
                "n": 0, "mean": None, "std": None,
                "min": None, "max": None, "saturated_frac": None,
            }
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            # fraction of judge scores pinned near the ceiling — a cheap
            # guard against the policy learning to exploit the judge
            "saturated_frac": float((arr >= 0.95).mean()),
        }

    def _accumulate_stats(self, stats: list) -> None:
        for s in stats:
            si = s["strategy_index"]
            if 0 <= si < N_STRATEGIES:
                self._strategy_value_sum[si] += s["reward"]
                self._strategy_value_count[si] += 1
            score = s.get("persuasiveness")
            if score is not None:
                self._judge_scores.append(float(score))
            src = s.get("judge_src")
            if src in self._judge_src_counts:
                self._judge_src_counts[src] += 1
            ga = s.get("goal_align")
            if ga is not None:
                self._goal_align_sum += float(ga)
                self._goal_align_count += 1

    def _strategy_entropy(self) -> float:
        """Entropy (nats) of the empirical strategy distribution."""
        c = self._strategy_value_count.astype(np.float64)
        total = c.sum()
        if total <= 0:
            return float("nan")
        p = c / total
        nz = p[p > 0]
        return float(-(nz * np.log(nz)).sum())

    def _write_strategy_value(self, out_dir: Path) -> None:
        from .strategy import describe, index_to_strategy

        cnt = self._strategy_value_count
        mean = np.divide(
            self._strategy_value_sum, cnt,
            out=np.zeros_like(self._strategy_value_sum),
            where=cnt > 0,
        )
        rows = [
            {
                "strategy_index": int(si),
                "strategy": describe(index_to_strategy(si)),
                "count": int(cnt[si]),
                "mean_reward": float(mean[si]),
                "total_reward": float(self._strategy_value_sum[si]),
            }
            for si in range(N_STRATEGIES)
            if cnt[si] > 0
        ]
        rows.sort(key=lambda r: r["mean_reward"], reverse=True)
        (out_dir / "strategy_value.json").write_text(
            json.dumps(
                {
                    "strategy_entropy": self._strategy_entropy(),
                    "n_decisions": int(cnt.sum()),
                    "rows": rows,
                },
                indent=2,
            )
        )

    def _entropy_coef_for(self, iteration: int) -> float:
        """Linear entropy-coefficient anneal (high early -> low late)."""
        cfg = self.config
        if cfg.entropy_coef_start is None or cfg.entropy_coef_end is None:
            return cfg.entropy_coef
        if cfg.iterations <= 1:
            return cfg.entropy_coef_start
        frac = (iteration - 1) / (cfg.iterations - 1)
        frac = min(max(frac, 0.0), 1.0)
        return cfg.entropy_coef_start + frac * (
            cfg.entropy_coef_end - cfg.entropy_coef_start
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
            stats: list = []
            obs, act, rew, mask, info = collect_episode(
                self.env, self.policy, self.device, seed, stats=stats
            )
            self._accumulate_stats(stats)
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
            stats: list = []
            out = collect_episode(
                env, policy, torch.device("cpu"), seed, stats=stats
            )
            return (*out, stats)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(
                pool.map(run_one, range(self.config.episodes_per_iteration))
            )

        all_obs, all_act, all_rew, all_mask = [], [], [], []
        all_adv, all_ret = [], []
        episode_stats = []
        for obs, act, rew, mask, info, stats in results:
            self._accumulate_stats(stats)
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

    def _update(self, batch, entropy_coef: float | None = None) -> dict:
        obs = torch.as_tensor(batch["obs"], device=self.device)
        act = torch.as_tensor(batch["act"], device=self.device)
        mask = torch.as_tensor(batch["mask"], device=self.device)
        adv = torch.as_tensor(batch["adv"], device=self.device)
        ret = torch.as_tensor(batch["ret"], device=self.device)

        if entropy_coef is None:
            entropy_coef = self.config.entropy_coef

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
                    - entropy_coef * entropy.mean()
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

    def train(
        self,
        resume_from: str | Path | None = None,
        prior_history: list[dict] | None = None,
    ) -> tuple[ActorCritic, list[dict]]:
        """Run PPO training.

        `resume_from`: path to a checkpoint written by `save()` (or the
        default `policy.pt`). The policy weights and, when present, the
        optimizer state are restored, and training continues from the
        stored iteration. `prior_history` re-attaches the per-iteration
        records collected before the interruption so the final
        `training_log.json` is complete.
        """
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict] = list(prior_history or [])
        start_iter = 1
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=self.device)
            want_factorized = _ckpt_is_factorized(ckpt)
            if want_factorized != self.policy.factorized:
                logger.info(
                    "checkpoint policy head is %s; rebuilding policy to "
                    "match before loading weights",
                    "factorized" if want_factorized else "flat",
                )
                self.config.factorized_policy = want_factorized
                self.policy = self._build_policy().to(self.device)
                if self._worker_policies is not None:
                    self._worker_policies = [
                        self._build_policy().to("cpu")
                        for _ in range(len(self._worker_policies))
                    ]
                self.optimizer = self._build_optimizer()
            self.policy.load_state_dict(ckpt["model_state"])
            if ckpt.get("optimizer_state") is not None:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state"])
                except (ValueError, KeyError) as exc:  # noqa: BLE001
                    logger.warning(
                        "could not restore optimizer state (%s); "
                        "continuing with a fresh optimizer", exc
                    )
            else:
                logger.warning(
                    "checkpoint has no optimizer state; continuing with "
                    "a fresh optimizer (weights restored)"
                )
            start_iter = int(ckpt.get("iteration", 0)) + 1
            if ckpt.get("iteration") is None:
                # Older checkpoints (pre-resume support) did not store the
                # iteration. Fall back to the last iteration in the
                # reconstructed history, so we do not silently restart at 1.
                last_known = max(
                    (int(r.get("iteration", 0)) for r in history),
                    default=0,
                )
                start_iter = last_known + 1
                logger.warning(
                    "checkpoint has no stored iteration; resuming from "
                    "iteration %d (from prior history)", last_known,
                )
            logger.info(
                "resumed from %s at iteration %d/%d",
                resume_from, start_iter - 1, cfg.iterations,
            )

        self._iteration = start_iter - 1
        # keep elapsed_s monotonic across a resume
        prev_elapsed = float(history[-1].get("elapsed_s", 0.0)) if history else 0.0
        # the interrupted run may have advanced past the last checkpoint;
        # drop prior records at or after the resume point to avoid
        # duplicate iterations in the curve
        history = [
            r for r in history if int(r.get("iteration", 0)) < start_iter
        ]
        t0 = time.time()
        for iteration in range(start_iter, cfg.iterations + 1):
            self._iteration = iteration
            entropy_coef = self._entropy_coef_for(iteration)
            n_judge_before = len(self._judge_scores)
            batch, episode_stats = self._rollout_batch()
            metrics = self._update(batch, entropy_coef=entropy_coef)

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
                "train_entropy_coef": entropy_coef,
                "train_strategy_entropy": self._strategy_entropy(),
                "train_judge": self._summarize(
                    self._judge_scores[n_judge_before:]
                ),
                "train_judge_src": dict(self._judge_src_counts),
                "train_goal_align": (
                    self._goal_align_sum / self._goal_align_count
                    if self._goal_align_count else None
                ),
                "elapsed_s": prev_elapsed + (time.time() - t0),
            }
            history.append(rec)

            if iteration % cfg.log_every == 0:
                logger.info(
                    "iter %d/%d reward=%.4f final_stance=%.3f adopted=%.1f "
                    "pi_loss=%.4f v_loss=%.4f ent=%.3f (entc=%.4f) "
                    "judge[id=%d text=%d none=%d] (%.1fs)",
                    iteration, cfg.iterations, mean_reward, mean_stance,
                    rec["mean_adopted"], metrics["pi_loss"], metrics["v_loss"],
                    metrics["entropy"], entropy_coef,
                    self._judge_src_counts["id"],
                    self._judge_src_counts["text"],
                    self._judge_src_counts["none"],
                    rec["elapsed_s"],
                )
                if self._judge_src_counts["text"]:
                    logger.warning(
                        "judge identity-index MISS: %d lookups fell back to "
                        "the content path by iter %d — the BT table is "
                        "incomplete for this (strategy, profile) space",
                        self._judge_src_counts["text"], iteration,
                    )

            if iteration % cfg.eval_every == 0:
                score = self.evaluate(episodes=cfg.eval_episodes)
                st = getattr(self, "_last_eval_stats", None) or {}
                judge = st.get("judge") or {}
                history[-1]["eval_greedy_reward"] = score
                history[-1]["eval_strategy_entropy"] = st.get("strategy_entropy")
                history[-1]["eval_action_entropy"] = st.get("action_entropy")
                history[-1]["eval_judge"] = st.get("judge")
                history[-1]["eval_top_strategies"] = st.get("top_strategies")
                logger.info(
                    "eval @ %d: greedy mean reward %.4f | strat_H=%s "
                    "action_H=%s | judge~%s sat=%s | top=%s",
                    iteration, score,
                    _fmt(st.get("strategy_entropy")),
                    _fmt(st.get("action_entropy")),
                    _fmt(judge.get("mean")),
                    _fmt(judge.get("saturated_frac")),
                    st.get("top_strategies"),
                )
                self._write_strategy_value(out_dir)

            if (
                cfg.paired_eval_every > 0
                and iteration % cfg.paired_eval_every == 0
            ):
                paired = self.evaluate_paired()
                history[-1]["paired_eval"] = paired
                logger.info(
                    "paired eval @ %d: greedy %.3f | random %.3f "
                    "(d%+.3f t=%.2f) | heuristic %.3f (d%+.3f t=%.2f)",
                    iteration, paired["greedy"], paired["random"],
                    paired["vs_random"]["mean"], paired["vs_random"]["t"],
                    paired["heuristic"], paired["vs_heuristic"]["mean"],
                    paired["vs_heuristic"]["t"],
                )

            # checkpoint (every ~10 iters, plus the final one): frequent
            # enough that an interruption costs little work
            if iteration % max(1, cfg.iterations // 30) == 0 or iteration == cfg.iterations:
                self.save(out_dir / "policy.pt", iteration=iteration)

            # incremental log so an interruption never loses the curve
            (out_dir / "training_log.json").write_text(
                json.dumps(
                    {"config": asdict(cfg), "history": history}, indent=2
                )
            )

        self.save(out_dir / "policy.pt", iteration=cfg.iterations)
        (out_dir / "training_log.json").write_text(
            json.dumps(
                {"config": asdict(cfg), "history": history}, indent=2
            )
        )
        self._write_strategy_value(out_dir)
        return self.policy, history

    # ------------------------------------------------------------------

    def evaluate(self, episodes: int = 10, greedy: bool = True) -> float:
        """Mean episode reward; parallel across worker envs when configured.

        Side-effect: detailed Tier-5 stats for the last call are stored on
        ``self._last_eval_stats`` (see `evaluate_stats`).
        """
        stats = self.evaluate_stats(episodes=episodes, greedy=greedy)
        return stats["mean_reward"]

    def _run_eval_episode(
        self, env, policy, device, seed: int, greedy: bool
    ) -> dict:
        """One eval episode with per-step instrumentation.

        Collects: episode reward; per-strategy choice counts; the judge
        (persuasiveness) scores; and the masked policy entropy at each
        decision (a direct measure of population-level exploration).
        """
        n_persuaders = int(getattr(env, "n_persuaders", 1))
        obs, info = env.reset(seed=seed)
        done, total = False, 0.0
        strat_counts = np.zeros(N_STRATEGIES, dtype=np.int64)
        judge_scores: list[float] = []
        entropies: list[float] = []
        while not done:
            mask = info.get("action_mask")
            mask_t = (
                torch.as_tensor(mask, dtype=torch.bool, device=device)
                if mask is not None else None
            )
            actions = []
            for i in range(n_persuaders):
                obs_i = env.observation(actor=i) if n_persuaders > 1 else obs
                obs_t = torch.as_tensor(
                    obs_i, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    logits, _ = policy.forward(obs_t.unsqueeze(0))
                    logits = logits.squeeze(0)
                    if mask_t is not None:
                        neg = torch.finfo(logits.dtype).min
                        logits = torch.where(
                            mask_t, logits, torch.full_like(logits, neg)
                        )
                    ent = torch.distributions.Categorical(
                        logits=logits
                    ).entropy()
                    entropies.append(float(ent.item()))
                actions.append(
                    policy.act(obs_t, mask=mask_t, greedy=greedy)
                )
            action = actions[0] if n_persuaders == 1 else actions
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(np.sum(np.atleast_1d(reward)))
            for rec in info.get("persuader_actions", []):
                si = int(rec["strategy_index"])
                if 0 <= si < N_STRATEGIES:
                    strat_counts[si] += 1
                js = rec.get("persuasiveness")
                if js is not None:
                    judge_scores.append(float(js))
            done = terminated or truncated
        return {
            "reward": total,
            "strategy_counts": strat_counts,
            "judge_scores": judge_scores,
            "entropies": entropies,
        }

    def evaluate_stats(self, episodes: int = 10, greedy: bool = True) -> dict:
        """Run eval episodes and aggregate Tier-5 diagnostics."""
        if self.config.rollout_workers > 1 and self._worker_envs:
            state = {k: v.cpu() for k, v in self.policy.state_dict().items()}
            for p in self._worker_policies:
                p.load_state_dict(state)

            def run_one(k: int):
                seed = 900_000 + k
                env = self._worker_envs[k % len(self._worker_envs)]
                policy = self._worker_policies[k % len(self._worker_policies)]
                return self._run_eval_episode(
                    env, policy, torch.device("cpu"), seed, greedy
                )

            with ThreadPoolExecutor(
                max_workers=len(self._worker_envs)
            ) as pool:
                results = list(pool.map(run_one, range(episodes)))
        else:
            results = [
                self._run_eval_episode(
                    self.env, self.policy, self.device, 900_000 + k, greedy
                )
                for k in range(episodes)
            ]

        rewards = np.array([r["reward"] for r in results], dtype=np.float64)
        counts = np.sum(
            [r["strategy_counts"] for r in results], axis=0
        )
        all_judge = [s for r in results for s in r["judge_scores"]]
        all_ent = [e for r in results for e in r["entropies"]]
        total_counts = counts.sum()
        if total_counts > 0:
            p = counts / total_counts
            nz = p[p > 0]
            strat_entropy = float(-(nz * np.log(nz)).sum())
        else:
            strat_entropy = None

        from .strategy import describe, index_to_strategy

        top_idx = np.argsort(counts)[::-1][:3]
        top = [
            {
                "strategy_index": int(si),
                "strategy": describe(index_to_strategy(int(si))),
                "count": int(counts[si]),
            }
            for si in top_idx
            if counts[si] > 0
        ]

        stats = {
            "episodes": episodes,
            "mean_reward": float(rewards.mean()),
            "reward_se": (
                float(rewards.std(ddof=1) / np.sqrt(len(rewards)))
                if len(rewards) > 1 else None
            ),
            "strategy_counts": counts.tolist(),
            "strategy_entropy": strat_entropy,
            "top_strategies": top,
            "judge": self._summarize(all_judge),
            "action_entropy": float(np.mean(all_ent)) if all_ent else None,
        }
        self._last_eval_stats = stats
        return stats

    # ------------------------------------------------------------------
    # Paired evaluation (Tier-5: model selection with lower variance)
    # ------------------------------------------------------------------

    def _run_baseline_episode(self, kind, seed, env, policy, rng) -> float:
        obs, info = env.reset(seed=seed)
        done, total = False, 0.0
        n_persuaders = int(getattr(env, "n_persuaders", 1))
        heuristic_idx = N_STRATEGIES // 2
        while not done:
            mask = info.get("action_mask")
            if kind == "greedy":
                mask_t = (
                    torch.as_tensor(mask, dtype=torch.bool)
                    if mask is not None else None
                )
                acts = [
                    policy.act(
                        torch.as_tensor(
                            env.observation(actor=i)
                            if n_persuaders > 1 else obs,
                            dtype=torch.float32,
                        ),
                        mask=mask_t, greedy=True,
                    )
                    for i in range(n_persuaders)
                ]
            elif kind == "heuristic":
                acts = [heuristic_idx] * n_persuaders
            else:  # random
                # an all-False mask means "no influenceable targets"
                # (cascade mode exhausts the crowd): sampling from it
                # would raise, so fall back to a uniform draw
                valid = (
                    None
                    if mask is None or np.all(mask) or not np.any(mask)
                    else np.flatnonzero(mask)
                )
                acts = [
                    int(rng.choice(valid)) if valid is not None
                    else int(rng.integers(env.action_space.n))
                    for _ in range(n_persuaders)
                ]
            action = acts[0] if n_persuaders == 1 else acts
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(np.sum(np.atleast_1d(reward)))
            done = terminated or truncated
        return total

    def evaluate_paired(self, episodes: int | None = None) -> dict:
        """Greedy/random/heuristic on shared seeds (paired deltas + t).

        Mirrors `code/eval_final.py` but runs mid-training, so the RL
        curve can be selected on a low-variance paired comparison instead
        of the noisy unpaired `eval_episodes` number.
        """
        episodes = int(episodes or self.config.paired_eval_episodes)
        n_workers = max(1, self.config.rollout_workers)
        base_envs = (
            self._worker_envs
            if self._worker_envs
            else [self.env.clone()]
        )
        state = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        eval_policy = self._build_policy().to("cpu")
        eval_policy.load_state_dict(state)
        eval_policy.eval()

        def run_one(k: int):
            seed = 500_000 + k
            env = base_envs[k % len(base_envs)]
            g = self._run_baseline_episode(
                "greedy", seed, env.clone(), eval_policy, None
            )
            r = self._run_baseline_episode(
                "random", seed + 7_919, env.clone(), eval_policy,
                np.random.default_rng(seed + 7_919),
            )
            h = self._run_baseline_episode(
                "heuristic", seed, env.clone(), eval_policy, None
            )
            return g, r, h

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                rows = list(pool.map(run_one, range(episodes)))
        else:
            rows = [run_one(k) for k in range(episodes)]
        rows = np.asarray(rows, dtype=np.float64)
        g, r, h = rows[:, 0], rows[:, 1], rows[:, 2]

        def paired(a: np.ndarray, b: np.ndarray) -> dict:
            d = a - b
            se = (
                float(d.std(ddof=1) / np.sqrt(len(d)))
                if len(d) > 1 else float("nan")
            )
            t = float(d.mean() / se) if se and se > 0 else float("inf")
            return {"mean": float(d.mean()), "se": se, "t": t}

        return {
            "episodes": episodes,
            "greedy": float(g.mean()),
            "random": float(r.mean()),
            "heuristic": float(h.mean()),
            "vs_random": paired(g, r),
            "vs_heuristic": paired(g, h),
        }

    def save(self, path: str | Path, iteration: int | None = None) -> None:
        torch.save(
            {
                "model_state": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "n_actions": self.policy.n_actions,
                "obs_dim": self.policy.body[0].in_features,
                "hidden": self.config.hidden,
                "factorized": bool(self.policy.factorized),
                "iteration": int(iteration) if iteration is not None else None,
            },
            path,
        )

    @classmethod
    def load_policy(
        cls, path: str | Path, device: str | None = None
    ) -> ActorCritic:
        ckpt = torch.load(path, map_location=resolve_device(device))
        policy = ActorCritic(
            ckpt["obs_dim"], ckpt["n_actions"], hidden=ckpt["hidden"],
            factorized=_ckpt_is_factorized(ckpt),
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
                if mask is not None and np.any(mask):
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
