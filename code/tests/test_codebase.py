"""Unit tests for the fixed codebase. Run: pytest -q"""

from __future__ import annotations

import numpy as np
import pytest

import code.LLM_module as llm
from code.agent import LLMSocialAgent
from code.env import LatentStanceDiffusionEnv
from code.scenario import LLMScenario
from code.simulation import Simulation


def make_agents(n=6, seed=0):
    scenario = LLMScenario(agent_num=n, rng_seed=seed)
    rng = np.random.default_rng(seed)
    agents = []
    for i in range(n):
        a = LLMSocialAgent(i, scenario=scenario, rng=rng)
        a.profile = f"Agent {i} is a test user who enjoys topic {i % 3}."
        a.embedding = a._get_embedding(a.profile)
        a.profile_embedding = a.embedding
        a.degree = 2
        agents.append(a)
    return scenario, agents


class TestParseStanceShift:
    def test_plain_number(self):
        assert llm.parse_stance_shift("0.3") == pytest.approx(0.3)

    def test_embedded_number(self):
        assert llm.parse_stance_shift("The shift is 0.25.") == pytest.approx(0.25)

    def test_clamping(self):
        assert llm.parse_stance_shift("1.7") == pytest.approx(1.0)
        assert llm.parse_stance_shift("-0.2") == pytest.approx(0.0)

    def test_no_number(self):
        assert llm.parse_stance_shift("I cannot estimate this.") == 0.0

    def test_none(self):
        assert llm.parse_stance_shift(None) == 0.0


class TestAgentStance:
    def test_partial_transition(self):
        scenario, agents = make_agents()
        target, source = agents[1], agents[0]
        source.stance = -1.0
        target.stance = 0.6
        before, after = target.update_agent_stance(
            source_agent=source,
            source_influence_strength=0.5,
            target_susceptibility_rate=scenario.stance_influence_rate,
        )
        assert before == pytest.approx(0.6)
        assert after == pytest.approx(0.6 + 0.5 * 0.4 * (-1.6))
        assert -1.0 <= after <= 1.0

    def test_labels(self):
        scenario, agents = make_agents()
        a = agents[0]
        a.stance = 0.9
        a.update_agent_stance(a, 0.0, 0.0)
        assert a.stance_label == "Supportive"
        a.stance = -0.9
        a.update_agent_stance(a, 0.0, 0.0)
        assert a.stance_label == "Resistant"


class TestProbabilisticAcceptance:
    """BUGFIX 1: acceptance must follow the stochastic gate."""

    def test_rejection_possible(self, monkeypatch):
        scenario, agents = make_agents()
        target, source = agents[1], agents[0]
        source.stance = -1.0
        source.preference_state = 1
        source.post = "Test source message with a viewpoint."
        source.post_embedding = source._get_embedding(source.post)
        source.message_id = source.generate_message_id()
        source.original_message_id = source.message_id
        target.stance = 0.6

        # Force rejection: rng always returns 1.0 (>= influence_prob).
        target._rng = np.random.default_rng(12345)
        monkeypatch.setattr(
            type(target), "calculate_influence_prob", lambda self, s: 0.0
        )
        diffusion_events, message_events = [], []
        target.infection_impact(1, diffusion_events, message_events, source)
        assert target.stance == pytest.approx(0.6)  # unchanged
        assert len(diffusion_events) == 1
        assert diffusion_events[0]["behavior"] == "reject"
        assert len(message_events) == 0

    def test_acceptance_possible(self, monkeypatch):
        scenario, agents = make_agents()
        target, source = agents[1], agents[0]
        source.stance = -1.0
        source.preference_state = 1
        source.post = "Test source message with a viewpoint."
        source.post_embedding = source._get_embedding(source.post)
        source.message_id = source.generate_message_id()
        source.original_message_id = source.message_id
        target.stance = 0.9

        monkeypatch.setattr(
            type(target), "calculate_influence_prob", lambda self, s: 1.0
        )
        monkeypatch.setattr(
            llm, "generate_response_message", lambda post, profile, delta_stance: "internalized text"
        )
        diffusion_events, message_events = [], []
        target.infection_impact(1, diffusion_events, message_events, source)
        assert target.stance < 0.9
        assert diffusion_events[0]["behavior"] == "accept"


class TestEmbeddingGuard:
    """BUGFIX 2: first-time sources must not be spuriously skipped."""

    def test_first_time_source_proceeds(self, monkeypatch):
        scenario, agents = make_agents()
        target, source = agents[1], agents[0]
        source.stance = -1.0
        source.preference_state = 1
        source.post = "A fresh source post that has never been embedded."
        # post_embedding intentionally left None (first-time source)
        source.message_id = source.generate_message_id()
        source.original_message_id = source.message_id
        target.stance = 0.6
        monkeypatch.setattr(
            type(target), "calculate_influence_prob", lambda self, s: 1.0
        )
        monkeypatch.setattr(
            llm, "generate_response_message", lambda post, profile, delta_stance: "internalized text"
        )
        monkeypatch.setattr(llm, "update_profile", lambda received, profile: "new profile")
        monkeypatch.setattr(llm, "generate_information", lambda received, profile: "new post")
        monkeypatch.setattr(llm, "inference_stance_post", lambda m, p: 0.3)
        monkeypatch.setattr(llm, "inference_stance_profile", lambda m, o, n: 0.4)
        monkeypatch.setattr(llm, "inference_stance", lambda m, p, o, n: 0.35)

        diffusion_events, message_events = [], []
        target.infection_impact(1, diffusion_events, message_events, source)
        assert source.post_embedding is not None  # computed, not skipped
        assert len(diffusion_events) == 1  # BUGFIX 3: saved exactly once


class TestNoDuplicateEvents:
    """BUGFIX 3: exactly one diffusion event per interaction."""

    def test_single_diffusion_event(self, monkeypatch):
        scenario, agents = make_agents()
        target, source = agents[1], agents[0]
        source.stance = -1.0
        source.preference_state = 1
        source.post = "Source message for the duplicate-event test."
        source.post_embedding = source._get_embedding(source.post)
        source.message_id = source.generate_message_id()
        source.original_message_id = source.message_id
        target.stance = 0.9  # stays positive after one interaction: no adoption
        monkeypatch.setattr(
            type(target), "calculate_influence_prob", lambda self, s: 1.0
        )
        monkeypatch.setattr(
            llm, "generate_response_message", lambda post, profile, delta_stance: "internalized text"
        )
        monkeypatch.setattr(llm, "update_profile", lambda received, profile: "new profile")
        monkeypatch.setattr(llm, "generate_information", lambda received, profile: "new post")
        monkeypatch.setattr(llm, "inference_stance_post", lambda m, p: 0.3)
        monkeypatch.setattr(llm, "inference_stance_profile", lambda m, o, n: 0.4)
        monkeypatch.setattr(llm, "inference_stance", lambda m, p, o, n: 0.35)

        diffusion_events, message_events = [], []
        target.infection_impact(1, diffusion_events, message_events, source)
        # Non-adoption path (stance stays above threshold): one event.
        assert len(diffusion_events) == 1
        # Non-adopted agent must not produce message events (BUGFIX 6).
        assert len(message_events) == 0


class TestBehaviorSemantics:
    """BUGFIX 4: `behavior` must not be redefined by stance sign."""

    def test_accept_with_positive_stance(self, monkeypatch):
        scenario, agents = make_agents()
        target, source = agents[1], agents[0]
        source.stance = -1.0
        source.preference_state = 1
        source.post = "Weakly persuasive source message."
        source.post_embedding = source._get_embedding(source.post)
        source.message_id = source.generate_message_id()
        source.original_message_id = source.message_id
        target.stance = 0.9
        # deterministic acceptance: influence_prob 1.0 forces the gate open
        monkeypatch.setattr(
            type(target), "calculate_influence_prob", lambda self, s: 1.0
        )
        monkeypatch.setattr(
            llm, "generate_response_message", lambda post, profile, delta_stance: "internalized text"
        )
        diffusion_events, message_events = [], []
        target.infection_impact(1, diffusion_events, message_events, source)
        # accepted (gate passed), stance still positive
        assert diffusion_events[0]["behavior"] == "accept"
        assert target.stance > 0
        assert diffusion_events[0]["adopted"] == 0


class TestNoneSafety:
    """BUGFIX 5: None LLM outputs must not crash event saving."""

    def test_none_profile_handled(self, monkeypatch):
        scenario, agents = make_agents()
        target, source = agents[1], agents[0]
        source.stance = -1.0
        source.preference_state = 1
        source.post = "Source message for the None-safety test."
        source.post_embedding = source._get_embedding(source.post)
        source.message_id = source.generate_message_id()
        source.original_message_id = source.message_id
        target.stance = 0.2  # small: adoption will trigger
        monkeypatch.setattr(
            type(target), "calculate_influence_prob", lambda self, s: 1.0
        )
        monkeypatch.setattr(
            llm, "generate_response_message", lambda post, profile, delta_stance: "internalized text"
        )
        monkeypatch.setattr(llm, "update_profile", lambda received, profile: "new profile")
        monkeypatch.setattr(llm, "generate_information", lambda received, profile: None)  # None!
        monkeypatch.setattr(llm, "inference_stance_profile", lambda m, o, n: 0.4)

        diffusion_events, message_events = [], []
        target.infection_impact(1, diffusion_events, message_events, source)
        assert len(message_events) == 1
        event = message_events[0]
        assert event["generated_post"] is None
        assert event["Inferr_stance_post"] is None  # defined, not NameError
        assert event["Inferr_stance"] is None


class TestSimulation:
    def _mock_llm(self, monkeypatch):
        monkeypatch.setattr(
            llm, "generate_response_message", lambda post, profile, delta_stance: "internalized text"
        )
        monkeypatch.setattr(llm, "update_profile", lambda received, profile: "updated profile text")
        monkeypatch.setattr(
            llm, "generate_information", lambda received, profile: "generated post text"
        )
        monkeypatch.setattr(llm, "inference_stance_post", lambda m, p: 0.3)
        monkeypatch.setattr(llm, "inference_stance_profile", lambda m, o, n: 0.4)
        monkeypatch.setattr(llm, "inference_stance", lambda m, p, o, n: 0.35)

    def test_deterministic_given_seed(self, monkeypatch):
        self._mock_llm(monkeypatch)
        sim1 = Simulation(scenario=LLMScenario(agent_num=10, rng_seed=7))
        sim1.run(5)
        sim2 = Simulation(scenario=LLMScenario(agent_num=10, rng_seed=7))
        sim2.run(5)
        np.testing.assert_allclose(sim1.stances(), sim2.stances())

    def test_diffusion_progresses(self, monkeypatch):
        self._mock_llm(monkeypatch)
        sim = Simulation(scenario=LLMScenario(agent_num=12, rng_seed=3))
        summaries = sim.run(8)
        assert summaries[-1]["mean_stance"] < 0.65  # persuasion took effect
        assert summaries[-1]["n_events"] > 0

    def test_no_duplicate_events_engine(self, monkeypatch):
        self._mock_llm(monkeypatch)
        sim = Simulation(scenario=LLMScenario(agent_num=12, rng_seed=3))
        sim.run(5)
        # each (period, source, target) triple must be unique
        keys = [
            (e["period"], e["source_agent"], e["target_agent"])
            for e in sim.diffusion_events
        ]
        assert len(keys) == len(set(keys))


class TestEnv:
    @pytest.fixture(autouse=True)
    def _mock_llm(self, monkeypatch):
        monkeypatch.setattr(
            llm, "generate_response_message", lambda post, profile, delta_stance: "internalized text"
        )
        monkeypatch.setattr(llm, "update_profile", lambda received, profile: "updated profile")
        monkeypatch.setattr(llm, "generate_information", lambda received, profile: "new post")
        monkeypatch.setattr(llm, "inference_stance_post", lambda m, p: 0.3)
        monkeypatch.setattr(llm, "inference_stance_profile", lambda m, o, n: 0.4)
        monkeypatch.setattr(llm, "inference_stance", lambda m, p, o, n: 0.35)

    def test_gym_api(self):
        env = LatentStanceDiffusionEnv(
            scenario=LLMScenario(agent_num=10, rng_seed=1), n_periods=4
        )
        obs, info = env.reset(seed=42)
        assert obs.shape == (10,)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
        assert terminated
        assert "mean_stance" in info

    def test_env_deterministic(self):
        rewards = []
        for _ in range(2):
            env = LatentStanceDiffusionEnv(
                scenario=LLMScenario(agent_num=10, rng_seed=5), n_periods=3
            )
            env.reset(seed=11)
            r_total = 0.0
            done = False
            while not done:
                _, r, terminated, truncated, _ = env.step(0)
                r_total += r
                done = terminated or truncated
            rewards.append(r_total)
        assert rewards[0] == pytest.approx(rewards[1])
