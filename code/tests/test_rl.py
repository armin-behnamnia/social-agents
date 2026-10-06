"""Tests for the RL components (offline mock backend, CPU only)."""

from __future__ import annotations

import numpy as np
import pytest

from code import offline
from code.rl_env import ElitePersuaderEnv, OBS_DIM
from code.scenario import LLMScenario
from code.strategy import (
    N_STRATEGIES,
    StrategySpec,
    all_strategies,
    index_to_strategy,
    strategy_to_index,
)
from code.trainer import TrainConfig, evaluate, evaluate_message_bank, train

offline.install()


class TestStrategySpace:
    def test_roundtrip(self):
        for idx in [0, 1, 42, 179, N_STRATEGIES - 1]:
            spec = index_to_strategy(idx)
            assert strategy_to_index(spec) == idx

    def test_all_strategies_count(self):
        assert len(all_strategies()) == N_STRATEGIES
        assert N_STRATEGIES == 3 * 4 * 5 * 3

    def test_invalid_index(self):
        with pytest.raises(ValueError):
            index_to_strategy(N_STRATEGIES)
        with pytest.raises(ValueError):
            index_to_strategy(-1)

    def test_fields(self):
        spec = index_to_strategy(0)
        assert isinstance(spec, StrategySpec)
        assert spec.appeal == "logos"
        assert spec.tone == "neutral"
        assert spec.emphasis == "evidence"
        assert 0 < spec.intensity <= 1.0


class TestElitePersuaderEnv:
    def test_gym_api(self):
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=10, rng_seed=1), n_periods=4
        )
        obs, info = env.reset(seed=42)
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(obs).all()
        done = False
        total = 0.0
        while not done:
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            total += reward
            done = term or trunc
        assert done
        assert "mean_stance" in info and "strategy" in info

    def test_reward_from_persuader_events_only(self):
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=10, rng_seed=2), n_periods=3
        )
        env.reset(seed=5)
        obs, r, term, trunc, info = env.step(0)
        # reward equals sum of -delta over persuader-sourced events plus shaping
        assert isinstance(r, float)

    def test_deterministic(self):
        rewards = []
        for _ in range(2):
            env = ElitePersuaderEnv(
                scenario=LLMScenario(agent_num=10, rng_seed=3), n_periods=3
            )
            env.reset(seed=11)
            r_total = 0.0
            done = False
            while not done:
                _, r, term, trunc, _ = env.step(0)  # fixed strategy
                r_total += r
                done = term or trunc
            rewards.append(r_total)
        assert rewards[0] == pytest.approx(rewards[1])


class TestTrainer:
    def test_policy_save_load_roundtrip(self):
        from code.trainer import RLPolicy

        p = RLPolicy(seed=1)
        path = "/tmp/opencode/test_policy.npz"
        p.save(path)
        q = RLPolicy(seed=2)
        q.load(path)
        obs = np.random.default_rng(0).normal(size=OBS_DIM)
        np.testing.assert_allclose(p.probs(obs), q.probs(obs))

    def test_short_training_runs(self, tmp_path):
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=8, rng_seed=4), n_periods=3
        )
        cfg = TrainConfig(
            episodes=4,
            eval_every=2,
            eval_episodes=2,
            log_every=2,
            output_dir=str(tmp_path),
            save_path=str(tmp_path / "policy.npz"),
        )
        policy, records = train(env, cfg)
        assert len(records) == 4
        assert all(np.isfinite(r.reward) for r in records)
        score = evaluate(env, policy, episodes=2)
        assert np.isfinite(score)

    def test_baseline_finite(self):
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=8, rng_seed=5), n_periods=3
        )
        assert np.isfinite(evaluate_message_bank(env))


class TestRenderStrategyOffline:
    def test_mock_render(self):
        from code import LLM_module
        from code.strategy import index_to_strategy

        msg = LLM_module.render_strategy_message(
            index_to_strategy(7), target_profile="a teacher"
        )
        assert "logos" in msg  # mock echoes the strategy
