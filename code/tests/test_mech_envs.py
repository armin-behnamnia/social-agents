"""Tests for the imported colleague mechanisms (code/mech_sim.py) and
their RL-env wrapping. Offline mock backend, CPU only.

The mechanisms' semantics are pinned to code/melodie/ + root agent.py:
- cascade: within-period BFS, stance pull, emergence=observable change;
- exposure: persistent cross-period queue, LLM accept gate, no stance
  dynamics, next-period sources must have stance <= 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from code import LLM_module, offline
from code.mech_sim import MechanismSim, beta_stance_init
from code.rl_env import ElitePersuaderEnv
from code.scenario import LLMScenario
from code.simulation import Simulation

offline.install()


def _profiles(n: int) -> list[str]:
    return [f"Mock user {i} profile about topic {i % 5}." for i in range(n)]


class TestBetaInit:
    def test_range_and_shape(self):
        s = beta_stance_init(200, seed=0)
        assert s.shape == (200,)
        assert np.all(s >= -1.0) and np.all(s <= 1.0)

    def test_ratio_and_determinism(self):
        s1 = beta_stance_init(200, seed=7, support_ratio=0.5)
        s2 = beta_stance_init(200, seed=7, support_ratio=0.5)
        assert np.array_equal(s1, s2)
        # 50/50 mix
        assert abs((s1 > 0).mean() - 0.5) < 0.01
        # support ~ Beta(6,2) is skewed high; oppose ~ -Beta(8,2) skewed low
        assert s1[s1 > 0].mean() > 0.55
        assert s1[s1 < 0].mean() < -0.6


class TestCascade:
    def _sim(self, n=12, seed=5, stance=-0.05, **kw):
        scenario = LLMScenario(agent_num=n, rng_seed=seed)
        return MechanismSim(
            scenario=scenario,
            topology="watts_strogatz",
            profiles=_profiles(n),
            initial_stances=np.full(n, stance),
            seed_ids=(0,),
            mechanism="cascade",
            **kw,
        )

    def test_within_period_bfs_sweeps_component(self):
        """Offline stubs make every exposure emerge; slightly negative
        crowd members become sources immediately -> the whole connected
        component adopts in ONE period (paper mode needs many)."""
        sim = self._sim()
        assert sum(a.preference_state == 1 for a in sim.agents) == 1
        sim.step()
        adopted = {a.id for a in sim.agents if a.preference_state == 1}
        assert adopted == set(range(sim.scenario.agent_num))
        # 2-hop+ nodes adopted within a single period
        assert sim.period == 1
        assert all(a.adoption_time == 0 for a in sim.agents)

    def test_stance_pulled_toward_source(self):
        sim = self._sim(stance=0.4)
        sim.step()
        adopted = [a for a in sim.agents if a.id != 0 and a.preference_state == 1]
        assert adopted, "expected some adoptions"
        assert all(a.stance < 0.4 for a in adopted)

    def test_event_schema_matches_paper_mode(self):
        sim = self._sim()
        sim.step()
        assert sim.diffusion_events
        e = sim.diffusion_events[-1]
        for key in (
            "period", "source_agent", "target_agent",
            "target_stance_before", "target_stance_after",
            "target_delta_stance", "influence_probability",
            "behavior", "adopted",
        ):
            assert key in e
        assert e["behavior"] in ("emerge", "no_emerge")
        assert e["mechanism"] == "cascade"

    def test_determinism(self):
        a = self._sim()
        b = self._sim()
        a.step()
        b.step()
        assert a.summary()["n_adopted"] == b.summary()["n_adopted"]
        assert a.stances().tolist() == b.stances().tolist()


class TestExposure:
    def _sim(self, n=12, seed=5, stances=None, **kw):
        scenario = LLMScenario(agent_num=n, rng_seed=seed)
        if stances is None:
            stances = np.full(n, 0.65)
        return MechanismSim(
            scenario=scenario,
            topology="watts_strogatz",
            profiles=_profiles(n),
            initial_stances=stances,
            seed_ids=(0,),
            mechanism="exposure",
            **kw,
        )

    def test_accept_gate_adopts_and_stalls_with_positive_crowd(
        self, monkeypatch
    ):
        """All-positive crowd: adopters stay positive, never qualify as
        sources (stance<=0) -> diffusion stalls after the seed's direct
        neighbors. This is the faithful consequence of their queue rule
        and the reason exposure mode needs the Beta stance init."""
        monkeypatch.setattr(
            LLM_module, "accept_message", lambda *a, **k: "ACCEPT"
        )
        sim = self._sim()
        sim.step()
        n1 = sim.summary()["n_adopted"]
        assert n1 > 1  # seed + its neighbors
        for _ in range(3):
            sim.step()
        assert sim.summary()["n_adopted"] == n1  # stalled

    def test_beta_init_lets_adopters_propagate(self, monkeypatch):
        monkeypatch.setattr(
            LLM_module, "accept_message", lambda *a, **k: "ACCEPT"
        )
        n = 24
        stances = beta_stance_init(n, seed=3)
        sim = self._sim(n=n, seed=3, stances=stances)
        sim.step()
        n1 = sim.summary()["n_adopted"]
        sim.step()
        n2 = sim.summary()["n_adopted"]
        assert n2 > n1  # negative-stance adopters became sources

    def test_reject_gate_blocks_everything(self, monkeypatch):
        monkeypatch.setattr(
            LLM_module, "accept_message", lambda *a, **k: "REJECT"
        )
        sim = self._sim()
        sim.step()
        assert sim.summary()["n_adopted"] == 1  # the seed alone
        assert all(
            e["behavior"] == "reject" for e in sim.diffusion_events
        )

    def test_no_stance_dynamics(self, monkeypatch):
        """Exposure mode never touches the latent stance, even on
        accepted/emerged exposures."""
        monkeypatch.setattr(
            LLM_module, "accept_message", lambda *a, **k: "ACCEPT"
        )
        sim = self._sim()
        before = sim.stances().copy()
        sim.step()
        sim.step()
        after = sim.stances()
        assert np.array_equal(before, after)
        assert all(
            e["target_delta_stance"] == 0.0 for e in sim.diffusion_events
        )

    def test_determinism(self, monkeypatch):
        monkeypatch.setattr(
            LLM_module, "accept_message", lambda *a, **k: "ACCEPT"
        )
        a = self._sim()
        b = self._sim()
        for _ in range(3):
            a.step()
            b.step()
        assert a.summary() == b.summary()


class TestMechEnvWrap:
    def test_default_env_stays_paper(self):
        env = ElitePersuaderEnv(scenario=LLMScenario(agent_num=10))
        assert env.mechanism == "paper"
        sim = env.reset(seed=1) and env._sim
        assert type(sim) is Simulation

    def test_invalid_mechanism_rejected(self):
        with pytest.raises(ValueError):
            ElitePersuaderEnv(
                scenario=LLMScenario(agent_num=10), mechanism="nope"
            )

    @pytest.mark.parametrize("mechanism", ["cascade", "exposure"])
    def test_env_runs_on_both_mechanisms(self, mechanism, monkeypatch):
        if mechanism == "exposure":
            monkeypatch.setattr(
                LLM_module, "accept_message", lambda *a, **k: "ACCEPT"
            )
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=10, rng_seed=1),
            n_periods=3,
            topology="watts_strogatz",
            target_selection="random_neighbor",
            obs_mode="pair",
            n_persuaders=2,
            mechanism=mechanism,
            mech_beta_init=(mechanism == "exposure"),
            reward_adoption_bonus=0.5,
        )
        obs, info = env.reset(seed=42)
        assert np.isfinite(obs).all()
        done = False
        steps = 0
        while not done:
            action = [
                env.action_space.sample()
                for _ in range(env.n_persuaders)
            ]
            obs, reward, done, _, info = env.step(action)
            assert np.isfinite(np.asarray(reward)).all()
            assert info["mechanism"] == mechanism
            steps += 1
        assert steps == 3
        assert 0 < info["n_adopted"] <= 10

    def test_exposure_adoption_bonus_flows(self, monkeypatch):
        monkeypatch.setattr(
            LLM_module, "accept_message", lambda *a, **k: "ACCEPT"
        )
        env = ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=12, rng_seed=2),
            n_periods=2,
            topology="watts_strogatz",
            target_selection="random_neighbor",
            obs_mode="pair",
            n_persuaders=1,
            persuade_all_neighbors=True,  # reach every neighbor
            mechanism="exposure",
            reward_adoption_bonus=1.0,
            persuader_persuasiveness=False,
        )
        env.reset(seed=0)
        total = 0.0
        done = False
        while not done:
            _, r, done, _, info = env.step(0)
            total += float(r)
        # reward comes ONLY from the adoption bonus here (no stance
        # deltas in exposure mode) and must therefore be positive.
        assert total > 0.0
        assert info["n_adopted"] > 1


class TestExhaustedCrowd:
    """Cascade mode adopts on every exposure, so the crowd is routinely
    exhausted before the period budget ends. The env must then return a
    well-defined observation/mask and idle, and the trainer's masked
    samplers must not raise (regression: all-False mask -> empty
    np.flatnonzero -> ValueError, and every logit -inf -> NaN entropy).
    """

    def _exhausted_env(self, monkeypatch):
        monkeypatch.setattr(
            LLM_module, "generate_response_message",
            lambda *a, **k: "I have changed my mind completely.",
        )
        monkeypatch.setattr(
            LLM_module, "update_profile",
            lambda received=None, profile=None, **k: "converted reader",
        )
        monkeypatch.setattr(
            LLM_module, "generate_information",
            lambda *a, **k: "I now fully support this policy.",
        )
        return ElitePersuaderEnv(
            scenario=LLMScenario(agent_num=10, rng_seed=3),
            n_periods=5,
            topology="complete",  # everyone reachable in one period
            n_persuaders=1,
            persuade_all_neighbors=True,
            mechanism="cascade",
            mech_beta_init=True,
            persuader_persuasiveness=False,
        )

    def test_periods_continue_after_crowd_is_adopted(self, monkeypatch):
        env = self._exhausted_env(monkeypatch)
        env.reset(seed=0)
        last = None
        reward = 0.0
        for _ in range(5):
            obs, reward, terminated, truncated, last = env.step(0)
            assert obs.shape == obs.shape  # well-defined every period
        assert last["n_adopted"] == env.scenario.agent_num
        assert last["action_mask"].any(), (
            "an all-False mask is undefined for categorical sampling"
        )
        assert np.isfinite(np.atleast_1d(reward)).all()

    def test_random_baseline_survives_exhausted_crowd(self, monkeypatch):
        from code.torch_trainer import evaluate_random_baseline

        env = self._exhausted_env(monkeypatch)
        total = evaluate_random_baseline(env, episodes=2, seed=0)
        assert np.isfinite(total)
