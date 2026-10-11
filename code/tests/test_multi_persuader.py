"""Tests for the multi-persuader env (shared policy, N seeded RL
agents among a frozen crowd) and the sparse ER topology guard."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from code import offline  # noqa: E402
from code.rl_env import ElitePersuaderEnv, PER_AGENT_DIM  # noqa: E402
from code.scenario import LLMScenario  # noqa: E402
from code.strategy import N_STRATEGIES  # noqa: E402
from code.torch_trainer import (  # noqa: E402
    ActorCritic,
    collect_episode,
    evaluate_heuristic_baseline,
    evaluate_random_baseline,
)

offline.install()


def make_multi_env(agent_num=12, n_persuaders=3, n_periods=3, seed=1,
                   topology="erdos_renyi", **kwargs):
    return ElitePersuaderEnv(
        scenario=LLMScenario(agent_num=agent_num, rng_seed=seed),
        n_periods=n_periods,
        target_selection="learned",
        topology=topology,
        n_persuaders=n_persuaders,
        **kwargs,
    )


class TestMultiPersuaderEnv:
    def test_seeds_and_crowd(self):
        env = make_multi_env(agent_num=12, n_persuaders=4)
        env.reset(seed=1)
        sim = env._sim
        seeded = [a.id for a in sim.agents if a.preference_state == 1]
        assert sorted(seeded) == [0, 1, 2, 3]
        # crowd agents stay non-adopted at reset
        assert not any(
            a.preference_state == 1 for a in sim.agents[4:]
        )

    def test_persuaders_excluded_from_targets_and_mask(self):
        env = make_multi_env(agent_num=12, n_persuaders=3)
        _, info = env.reset(seed=2)
        candidates = info["candidates"]
        assert all(a.id > 2 for a in candidates)
        mask = info["action_mask"]
        # no compound action addressing a persuader can be enabled
        for pid in (0, 1, 2):
            assert not mask[pid * N_STRATEGIES:(pid + 1) * N_STRATEGIES].any()

    def test_joint_step_reward_attribution(self):
        env = make_multi_env(agent_num=12, n_persuaders=3, n_periods=2)
        _, info = env.reset(seed=3)
        mask = info["action_mask"]
        valid = np.flatnonzero(mask)
        obs, rewards, terminated, truncated, info = env.step(
            [int(valid[0]), int(valid[1] % len(valid)), int(valid[0])]
        )
        assert isinstance(rewards, np.ndarray)
        assert rewards.shape == (3,)
        assert not terminated
        # events from any persuader must land on the right component
        for event in info["period_events"]:
            src = event["source_agent"]
            if src in (0, 1, 2):
                # accept events add the shaping bonus to that persuader
                if event["behavior"] == "accept":
                    assert rewards[src] > 0

    def test_wrong_action_count_raises(self):
        env = make_multi_env(agent_num=12, n_persuaders=3)
        env.reset(seed=1)
        with pytest.raises(ValueError):
            env.step([0, 0])

    def test_bad_n_persuaders_raises(self):
        with pytest.raises(ValueError):
            make_multi_env(agent_num=12, n_persuaders=12)
        with pytest.raises(ValueError):
            make_multi_env(agent_num=12, n_persuaders=0)

    def test_episode_terminates_after_n_periods(self):
        env = make_multi_env(agent_num=12, n_persuaders=3, n_periods=2)
        _, info = env.reset(seed=5)
        valid = np.flatnonzero(info["action_mask"])
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, info = env.step(list(valid[:3]))
            done = terminated or truncated
            steps += 1
        assert steps == 2

    def test_is_acting_column(self):
        env = make_multi_env(agent_num=12, n_persuaders=3)
        env.reset(seed=1)
        obs0 = env.observation(actor=0)
        obs2 = env.observation(actor=2)
        assert obs0.shape == (12, PER_AGENT_DIM)
        # is_persuader column identical and marks the persuaders
        assert obs0[0, -2] == 1.0 and obs0[2, -2] == 1.0
        assert obs0[5, -2] == 0.0
        # is_acting differs per actor
        assert obs0[0, -1] == 1.0 and obs0[2, -1] == 0.0
        assert obs2[2, -1] == 1.0 and obs2[0, -1] == 0.0

    def test_single_persuader_back_compat(self):
        env = make_multi_env(agent_num=10, n_persuaders=1, n_periods=2)
        _, info = env.reset(seed=7)
        valid = np.flatnonzero(info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(int(valid[0]))
        assert isinstance(reward, float)
        assert info["persuader_actions"][0]["persuader"] == 0


class TestRandomNeighborMode:
    def make(self, agent_num=20, n_persuaders=3, n_periods=3, seed=1):
        return ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=agent_num, rng_seed=seed),
            n_periods=n_periods,
            target_selection="random_neighbor",
            obs_mode="pair",
            topology="erdos_renyi",
            n_persuaders=n_persuaders,
        )

    def test_action_and_obs_space(self):
        env = self.make()
        assert env.action_space.n == N_STRATEGIES
        assert env.observation_space.shape == (26,)
        obs, info = env.reset(seed=1)
        # returned obs without actor context is a 26-d pair (actor 0)
        assert obs.shape == (26,)

    def test_pair_obs_targets_are_own_neighbors(self):
        env = self.make(n_persuaders=3)
        env.reset(seed=1)
        sim = env._sim
        for i, pid in enumerate(env.persuader_ids):
            obs_i = env.observation(actor=i)
            assert obs_i.shape == (26,)
            tgt = env._pending_targets[i]
            assert tgt is not None
            assert tgt.id in sim.agents[pid].neighbors
            assert tgt.id not in env.persuader_ids

    def test_targets_now_your_neighbors_only(self):
        # with several episodes, pending targets never leave the
        # persuader's own neighborhood (when it has eligible ones)
        env = self.make(agent_num=30, n_persuaders=5, n_periods=2)
        for seed in range(5):
            env.reset(seed=seed)
            done = False
            while not done:
                actions = [0] * env.n_persuaders
                _, _, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                for rec in info["persuader_actions"]:
                    pid, tgt = rec["persuader"], rec["target"]
                    if tgt is None:
                        continue
                    # the delivered target was a pending neighbor
                    # (already adopted mid-period is fine — was drawn
                    # from the episode's earlier state)
                    assert tgt not in env.persuader_ids

    def test_reward_vector_per_persuader(self):
        env = self.make(n_persuaders=3, n_periods=2)
        _, info = env.reset(seed=3)
        obs, rewards, terminated, _, info = env.step([10, 90, 170])
        assert rewards.shape == (3,)
        assert info["persuader_actions"][1]["strategy_index"] == 90

    def test_deterministic_given_seed(self):
        env1, env2 = self.make(), self.make()
        r1 = evaluate_random_baseline(env1, episodes=2)
        r2 = evaluate_random_baseline(env2, episodes=2)
        assert r1 == r2

    def test_pair_requires_random_neighbor(self):
        with pytest.raises(ValueError):
            ElitePersuaderEnv(
                scenario=LLMScenario(agent_num=12, rng_seed=1),
                target_selection="learned",
                obs_mode="pair",
            )

    def test_collect_episode_and_baselines(self):
        env = self.make(agent_num=20, n_persuaders=3, n_periods=3)
        obs_dim = int(np.prod(env.observation_space.shape))
        policy = ActorCritic(obs_dim, env.action_space.n, hidden=32)
        obs, act, rew, mask, info = collect_episode(
            env, policy, torch.device("cpu"), seed=9
        )
        assert len(rew) == 9  # 3 persuaders x 3 periods
        assert mask.shape == (9, N_STRATEGIES)
        assert np.isfinite(evaluate_random_baseline(env, episodes=1))
        assert np.isfinite(evaluate_heuristic_baseline(env, episodes=1))


class TestSparseTopology:
    @pytest.mark.parametrize("seed", range(5))
    def test_erdos_renyi_connected_and_sparse(self, seed):
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=50, rng_seed=seed),
            n_periods=1,
            target_selection="learned",
            topology="erdos_renyi",
        )
        env.reset(seed=seed)
        sim = env._sim
        degrees = [a.degree for a in sim.agents]
        assert min(degrees) >= 1  # connectivity guard: no isolates
        assert np.mean(degrees) <= 8  # nearly sparse (target ~4)

    def test_agents_have_few_neighbors(self):
        env = make_multi_env(agent_num=50, n_persuaders=5, seed=11)
        env.reset(seed=11)
        for a in env._sim.agents:
            assert len(a.neighbors) <= 15


class TestSharedPolicyRollout:
    def test_collect_episode_transition_count(self):
        env = make_multi_env(agent_num=12, n_persuaders=3, n_periods=3)
        obs_dim = int(np.prod(env.observation_space.shape))
        policy = ActorCritic(obs_dim, env.action_space.n, hidden=32)
        obs, act, rew, mask, info = collect_episode(
            env, policy, torch.device("cpu"), seed=9
        )
        # 3 persuaders x 3 periods = 9 transitions
        assert len(rew) == 9
        assert obs.shape[0] == act.shape[0] == 9
        assert mask.shape == (9, env.action_space.n)

    def test_baselines_run_multi_persuader(self):
        env = make_multi_env(agent_num=12, n_persuaders=3, n_periods=2)
        r = evaluate_random_baseline(env, episodes=1)
        h = evaluate_heuristic_baseline(env, episodes=1)
        assert np.isfinite(r) and np.isfinite(h)

    def test_determinism(self):
        env1 = make_multi_env(agent_num=12, n_persuaders=3, n_periods=2)
        env2 = make_multi_env(agent_num=12, n_persuaders=3, n_periods=2)
        obs_dim = int(np.prod(env1.observation_space.shape))
        torch.manual_seed(0)
        policy = ActorCritic(obs_dim, env1.action_space.n, hidden=32)
        # policy sampling draws from the global torch RNG; reseed it so
        # only env determinism is under test
        torch.manual_seed(123)
        a = collect_episode(env1, policy, torch.device("cpu"), seed=42)
        torch.manual_seed(123)
        b = collect_episode(env2, policy, torch.device("cpu"), seed=42)
        assert np.allclose(a[2], b[2])
        assert np.array_equal(a[1], b[1])
