"""Tests for the GPU PPO trainer and learned-target env (mock backend)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from code import offline  # noqa: E402
from code.rl_env import ElitePersuaderEnv, OBS_DIM  # noqa: E402
from code.scenario import LLMScenario  # noqa: E402
from code.strategy import N_STRATEGIES  # noqa: E402
from code.torch_trainer import (  # noqa: E402
    ActorCritic,
    PPOConfig,
    PPOTrainer,
    compute_gae,
    evaluate_heuristic_baseline,
    evaluate_random_baseline,
)
from code.profiles import generate_synthetic_profiles  # noqa: E402

offline.install()


def make_env(agent_num=10, n_periods=4, seed=1, **kwargs):
    return ElitePersuaderEnv(
        scenario=LLMScenario(agent_num=agent_num, rng_seed=seed),
        n_periods=n_periods,
        target_selection="learned",
        **kwargs,
    )


class TestProfiles:
    def test_synthetic_profiles_deterministic(self):
        a = generate_synthetic_profiles(20, seed=7)
        b = generate_synthetic_profiles(20, seed=7)
        assert a == b
        assert len(a) == 20
        assert all(isinstance(p, str) and len(p) > 20 for p in a)

    def test_balanced_gender(self):
        # template alternates genders by index parity
        profiles = generate_synthetic_profiles(10, seed=3)
        men = sum(" man " in " " + p + " " for p in profiles)
        assert men == 5


class TestEmbeddingBackends:
    def test_fallback_when_no_credentials(self, monkeypatch):
        import code.tools as tools
        from code.config import EmbeddingConfig

        monkeypatch.setattr(
            tools, "GLOBAL_EMBEDDING_CONFIG", EmbeddingConfig(
                base_url="", api_key="", model="x"
            )
        )
        tools._model = None
        tools._backend = None
        tools._cache.clear()
        v = tools.text2embedding("hello persuasion")
        assert v.shape == (256,)
        assert tools.embedding_backend() == "hash-fallback"

    def test_api_config_detection(self):
        from code.config import EmbeddingConfig

        placeholder = EmbeddingConfig(
            base_url="https://api.example.com/v1",
            api_key="sk-REPLACE_ME",
        )
        assert not placeholder.has_credentials()
        real = EmbeddingConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-real",
        )
        assert real.has_credentials()


class TestLearnedTargetEnv:
    def test_compound_action_space(self):
        env = make_env(agent_num=10)
        assert env.action_space.n == 10 * N_STRATEGIES

    def test_matrix_observation(self):
        from code.rl_env import PER_AGENT_DIM

        env = make_env(agent_num=10)
        obs, info = env.reset(seed=42)
        assert obs.shape == (10, PER_AGENT_DIM)
        mask = info["action_mask"]
        assert mask.dtype == bool
        assert mask.shape == (env.action_space.n,)
        # mask marks agent-id slots: agent ids in candidates
        cand_ids = {a.id for a in info["candidates"]}
        for aid in range(10):
            slot = aid * N_STRATEGIES
            if aid in cand_ids:
                assert mask[slot : slot + N_STRATEGIES].all()
            else:
                assert not mask[slot : slot + N_STRATEGIES].any()

    def test_chosen_target_is_reached(self):
        env = make_env(agent_num=10, n_periods=3)
        obs, info = env.reset(seed=42)
        candidates = info["candidates"]
        assert len(candidates) > 0
        # action by agent id: pick candidate with highest id, strategy 5
        aid = max(c.id for c in candidates)
        action = aid * N_STRATEGIES + 5
        obs, r, term, trunc, info = env.step(action)
        assert info["chosen_target"] == aid
        assert info["strategy"].appeal is not None
        assert isinstance(r, float)

    def test_masked_out_action_falls_back(self):
        env = make_env(agent_num=10, n_periods=2)
        obs, info = env.reset(seed=42)
        candidates = info["candidates"]
        cand_ids = {c.id for c in candidates}
        # pick an agent id NOT in candidates
        invalid_id = next(i for i in range(10) if i not in cand_ids)
        action = invalid_id * N_STRATEGIES
        obs, r, term, trunc, info = env.step(action)
        assert info["chosen_target"] == candidates[0].id

    def test_deterministic_episode(self):
        totals = []
        for _ in range(2):
            env = make_env(agent_num=10, n_periods=3)
            env.reset(seed=11)
            total, done = 0.0, False
            while not done:
                _, r, term, trunc, _ = env.step(0)
                total += r
                done = term or trunc
            totals.append(total)
        assert totals[0] == pytest.approx(totals[1])

    def test_legacy_obs_mode(self):
        from code.rl_env import OBS_DIM

        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=10, rng_seed=2),
            n_periods=3,
            target_selection="learned",
            obs_mode="summary",
        )
        obs, info = env.reset(seed=5)
        assert obs.shape == (OBS_DIM,)

    def test_legacy_mode_still_works(self):
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=10, rng_seed=2),
            n_periods=3,
            target_selection="highest_degree",
        )
        assert env.action_space.n == N_STRATEGIES
        obs, info = env.reset(seed=5)
        assert "action_mask" in info
        _, r, term, trunc, info = env.step(0)
        assert np.isfinite(r)


class TestGAE:
    def test_gae_simple(self):
        rewards = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        values = np.array([0.5, 0.25, 0.0], dtype=np.float32)
        adv, ret = compute_gae(rewards, values, 0.0, gamma=1.0, lam=1.0)
        # gamma=lam=1: delta_t = r_t + V_{t+1} - V_t, adv_t = sum of future deltas
        # delta = [0.75, -0.25, 0] -> adv = [0.5, -0.25, 0]
        np.testing.assert_allclose(adv, [0.5, -0.25, 0.0], rtol=1e-5)
        np.testing.assert_allclose(ret, adv + values, rtol=1e-5)


class TestActorCritic:
    def test_masked_act(self):
        policy = ActorCritic(OBS_DIM, 100, hidden=16)
        obs = torch.zeros(OBS_DIM)
        mask = torch.zeros(100, dtype=torch.bool)
        mask[7] = True
        for _ in range(5):
            assert policy.act(obs, mask=mask) == 7
        assert policy.act(obs, mask=mask, greedy=True) == 7

    def test_forward_shapes(self):
        policy = ActorCritic(OBS_DIM, 50, hidden=16)
        obs = torch.randn(4, OBS_DIM)
        logits, values = policy(obs)
        assert logits.shape == (4, 50)
        assert values.shape == (4,)

    def test_matrix_obs_forward(self):
        from code.rl_env import PER_AGENT_DIM

        policy = ActorCritic(10 * PER_AGENT_DIM, 50, hidden=16)
        obs = torch.randn(10, PER_AGENT_DIM)  # unbatched matrix
        a = policy.act(obs)
        assert 0 <= a < 50
        obs_b = torch.randn(4, 10, PER_AGENT_DIM)  # batched
        logits, values = policy(obs_b)
        assert logits.shape == (4, 50)
        assert values.shape == (4,)


class TestPPOTrainer:
    def test_short_training_runs(self, tmp_path):
        env = make_env(agent_num=8, n_periods=3)
        cfg = PPOConfig(
            agent_num=8,
            n_periods=3,
            iterations=3,
            episodes_per_iteration=2,
            eval_every=2,
            eval_episodes=2,
            output_dir=str(tmp_path),
            device="cpu",
        )
        trainer = PPOTrainer(env, cfg)
        policy, history = trainer.train()
        assert len(history) == 3
        assert all(np.isfinite(h["mean_episode_reward"]) for h in history)
        assert (tmp_path / "policy.pt").exists()
        assert (tmp_path / "training_log.json").exists()

    def test_save_load_roundtrip(self, tmp_path):
        env = make_env(agent_num=8, n_periods=2)
        cfg = PPOConfig(
            agent_num=8, iterations=1, episodes_per_iteration=1,
            output_dir=str(tmp_path), device="cpu",
        )
        trainer = PPOTrainer(env, cfg)
        trainer.train()
        loaded = PPOTrainer.load_policy(tmp_path / "policy.pt", device="cpu")
        # matrix-mode obs matching the saved policy input dim
        obs = torch.randn(
            env.observation_space.shape[0],
            env.observation_space.shape[1],
        )
        mask = torch.ones(loaded.n_actions, dtype=torch.bool)
        a1 = trainer.policy.act(obs, mask=mask, greedy=True)
        a2 = loaded.act(obs, mask=mask, greedy=True)
        assert a1 == a2


class TestStrategySpaceGuard:
    """A checkpoint only loads under the strategy space it was trained
    with (``PERSUASION_STRATEGY_SPACE``: v1 = 180 arms / 4 factorized
    heads, v2 = 72 arms / 3). Loading one under the other space used to
    fail with a state_dict error naming ``policy_head.weight``, which
    points nowhere near the cause; the guard must name both.
    """

    def _trained(self, tmp_path):
        env = make_env(agent_num=8, n_periods=2)
        cfg = PPOConfig(
            agent_num=8, iterations=1, episodes_per_iteration=1,
            output_dir=str(tmp_path), device="cpu",
        )
        PPOTrainer(env, cfg).train()
        return tmp_path / "policy.pt"

    def test_checkpoint_records_its_space(self, tmp_path):
        ckpt = torch.load(self._trained(tmp_path), map_location="cpu")
        assert ckpt["n_strategies"] == N_STRATEGIES
        assert ckpt["strategy_space"] in ("v1", "v2")

    def test_mismatch_names_the_cause_and_the_fix(self, tmp_path, monkeypatch):
        path = self._trained(tmp_path)
        from code import torch_trainer as tt

        monkeypatch.setattr(tt, "N_STRATEGIES", N_STRATEGIES * 5)
        with pytest.raises(RuntimeError) as excinfo:
            PPOTrainer.load_policy(path, device="cpu")
        msg = str(excinfo.value)
        # it must say what is wrong and what to do about it
        assert "PERSUASION_STRATEGY_SPACE" in msg
        assert str(N_STRATEGIES) in msg
        assert "policy_head" not in msg, "leak-free: no state_dict internals"

    def test_matching_space_still_loads(self, tmp_path):
        path = self._trained(tmp_path)
        loaded = PPOTrainer.load_policy(path, device="cpu")
        # 8 agents x N_STRATEGIES strategies: the guard is a no-op here
        assert loaded.n_actions == 8 * N_STRATEGIES

    def test_older_checkpoints_without_the_field_are_tolerated(self, tmp_path):
        """Back-compat: checkpoints written before the field still load."""
        path = self._trained(tmp_path)
        ckpt = torch.load(path, map_location="cpu")
        ckpt.pop("n_strategies")
        ckpt.pop("strategy_space")
        torch.save(ckpt, path)
        PPOTrainer.load_policy(path, device="cpu")  # must not raise


class TestBaselines:
    def test_baselines_finite(self):
        env = make_env(agent_num=8, n_periods=3)
        assert np.isfinite(evaluate_random_baseline(env, episodes=2))
        assert np.isfinite(evaluate_heuristic_baseline(env, episodes=2))

    def test_parallel_rollouts_match_config(self, tmp_path):
        env = make_env(agent_num=8, n_periods=3)
        cfg = PPOConfig(
            agent_num=8,
            n_periods=3,
            iterations=2,
            episodes_per_iteration=4,
            rollout_workers=2,
            eval_every=2,
            eval_episodes=2,
            output_dir=str(tmp_path),
            device="cpu",
        )
        trainer = PPOTrainer(env, cfg)
        assert len(trainer._worker_envs) == 2
        policy, history = trainer.train()
        assert len(history) == 2
        # worker policies must be in sync with the main policy
        main_state = {
            k: v.cpu() for k, v in trainer.policy.state_dict().items()
        }
        for p in trainer._worker_policies:
            for k in main_state:
                np.testing.assert_allclose(
                    p.state_dict()[k].numpy(), main_state[k].numpy()
                )

    def test_parallel_baselines_finite(self):
        env = make_env(agent_num=8, n_periods=3)
        assert np.isfinite(
            evaluate_random_baseline(env, episodes=4, workers=2)
        )
        assert np.isfinite(
            evaluate_heuristic_baseline(env, episodes=4, workers=2)
        )

    def test_cuda_available_or_ci_cpu(self):
        # documents the intended GPU path; CI uses cpu fallback
        from code.torch_trainer import resolve_device

        dev = resolve_device("cpu")
        assert str(dev) == "cpu"
