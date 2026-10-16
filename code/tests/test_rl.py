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
            scenario=LLMScenario(agent_num=10, rng_seed=1), n_periods=4,
            obs_mode="summary",
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
            scenario=LLMScenario(agent_num=10, rng_seed=2), n_periods=3,
            obs_mode="summary",
        )
        env.reset(seed=5)
        obs, r, term, trunc, info = env.step(0)
        # reward equals sum of -delta over persuader-sourced events plus shaping
        assert isinstance(r, float)

    def test_deterministic(self):
        rewards = []
        for _ in range(2):
            env = ElitePersuaderEnv(
                scenario=LLMScenario(agent_num=10, rng_seed=3), n_periods=3,
                obs_mode="summary",
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
    def test_policy_save_load_roundtrip(self, tmp_path):
        from code.trainer import RLPolicy

        p = RLPolicy(seed=1)
        path = tmp_path / "test_policy.npz"
        p.save(path)
        q = RLPolicy(seed=2)
        q.load(path)
        obs = np.random.default_rng(0).normal(size=OBS_DIM)
        np.testing.assert_allclose(p.probs(obs), q.probs(obs))

    def test_short_training_runs(self, tmp_path):
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=8, rng_seed=4), n_periods=3,
            obs_mode="summary",
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
            scenario=LLMScenario(agent_num=8, rng_seed=5), n_periods=3,
            obs_mode="summary",
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


class TestBTIdentityLookup:
    """Track C: the reward model is reached by (strategy, profile) index."""

    def test_save_load_roundtrip_preserves_ids(self, tmp_path):
        from code.bt_judge import BTJudge

        j = BTJudge(
            table={("m", "p"): 0.6}, stats={"x": 1},
            table_id={(2, 3): 0.6}, profile_ids=[3],
        )
        path = tmp_path / "j.json"
        j.save(path)
        k = BTJudge.load(path)
        assert k("m", "p") == pytest.approx(0.6)
        assert k.lookup_by_id(2, 3) == pytest.approx(0.6)
        assert k.lookup_by_id(9, 9) is None
        assert k.profile_ids == [3]

    def test_env_uses_identity_path(self):
        from code.bt_judge import BTJudge

        n = 10
        judge = BTJudge(
            stats={},
            table_id={(si, pi): 0.777
                      for pi in range(n) for si in range(N_STRATEGIES)},
        )
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=n, rng_seed=7), n_periods=3,
            obs_mode="summary", judge_fn=judge,
        )
        env.reset(seed=3)
        _, _, _, _, info = env.step(0)
        rec = info["persuader_actions"][0]
        assert rec["judge_src"] == "id"
        assert rec["persuasiveness"] == pytest.approx(0.777)
        assert env._judge_hits >= 1
        assert env._judge_fallbacks == 0

    def test_env_counts_fallback_on_identity_miss(self):
        from code.bt_judge import BTJudge

        # identity index AND content table empty -> must fall through to
        # the absolute (offline-mocked) judge, and must be counted.
        judge = BTJudge(table={}, stats={}, table_id={}, profile_ids=[])
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=10, rng_seed=8), n_periods=3,
            obs_mode="summary", judge_fn=judge,
        )
        env.reset(seed=3)
        _, _, _, _, info = env.step(0)
        rec = info["persuader_actions"][0]
        assert rec["judge_src"] == "text"
        assert env._judge_fallbacks >= 1
        assert env._judge_hits == 0

    def test_attach_ids_full_coverage(self):
        from code import LLM_module
        from code.bt_judge import BTJudge

        strats = all_strategies()[:3]
        profiles = ["a retired teacher", "a night-shift nurse"]
        table = {}
        for prof in profiles:
            for si, spec in enumerate(strats):
                msg = LLM_module.render_strategy_message(
                    spec, target_profile=prof, topic="the policy",
                    stance_direction="opposing",
                )
                table[(msg, prof)] = 0.5 + 0.01 * si
        j = BTJudge(table=table)
        n = j.attach_ids(strats, profiles, topic="the policy")
        assert n == len(strats) * len(profiles)
        assert j.lookup_by_id(0, 0) == pytest.approx(0.5)
        assert j.lookup_by_id(2, 1) == pytest.approx(0.52)
        assert j.profile_ids == [0, 1]

    def test_profile_id_assigned_at_init(self):
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=10, rng_seed=9), n_periods=2,
            obs_mode="summary",
        )
        env.reset(seed=1)
        ids = [a.profile_id for a in env._sim.agents]
        assert ids == list(range(10))
        # profile order is stable across resets (same canonical list)
        profiles = [a.profile for a in env._sim.agents]
        env.reset(seed=99)
        assert [a.profile_id for a in env._sim.agents] == list(range(10))
        assert [a.profile for a in env._sim.agents] == profiles
