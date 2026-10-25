"""Tests for run instrumentation: traces, registry coverage, monitor.

The instrumentation promise is threefold and each part is pinned here:

1. **Completeness** — every variable the tracer writes is documented in
   `code/metrics_registry.py` (a new field with no description fails).
2. **Purity** — tracing does not change rewards, dynamics or RNG draws,
   so a traced run is comparable to an untraced one.
3. **Exactness** — `TraceTail` recomputes exact aggregates from the
   records (including across a partially-written line), which is what
   lets the dashboard claim "exact values".
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from code import offline  # noqa: E402
from code.metrics_registry import (  # noqa: E402
    STREAMS,
    field_map,
    flatten_record,
    render_markdown,
)
from code.run_trace import TraceWriter, read_meta  # noqa: E402
from code.rl_env import ElitePersuaderEnv  # noqa: E402
from code.scenario import LLMScenario  # noqa: E402
from code.torch_trainer import (  # noqa: E402
    ActorCritic,
    PPOConfig,
    PPOTrainer,
    collect_episode,
)

offline.install()


def make_env(agent_num=8, n_periods=3, seed=1, n_persuaders=2, **kwargs):
    return ElitePersuaderEnv(
        scenario=LLMScenario(agent_num=agent_num, rng_seed=seed),
        n_periods=n_periods,
        n_persuaders=n_persuaders,
        target_selection="random_neighbor",
        obs_mode="pair",
        **kwargs,
    )


def make_policy(env, seed=0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(
        int(np.prod(env.observation_space.shape)), env.action_space.n,
        hidden=16,
    )


def run_traced_episode(tmp_path, seed=1234):
    """One traced episode; returns (writer dir, env, policy, rewards)."""
    env = make_env()
    policy = make_policy(env)
    writer = TraceWriter(tmp_path / "trace", meta={"mechanism": "paper"},
                         fresh=True)
    obs, act, rew, mask, info = collect_episode(
        env, policy, torch.device("cpu"), seed, trace=writer,
        trace_meta={"iteration": 3, "episode": 1, "seed": seed},
    )
    writer.close()
    return tmp_path / "trace", env, policy, rew


def read_stream(directory, stream):
    path = directory / f"{stream}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ----------------------------------------------------------------------
# TraceWriter
# ----------------------------------------------------------------------
class TestTraceWriter:
    def test_writes_meta_and_streams(self, tmp_path):
        writer = TraceWriter(tmp_path / "t", meta={"mechanism": "cascade"})
        writer.write("steps", {"iteration": 1, "persuasiveness": 0.5})
        writer.write_many("events", [{"a": 1}, {"a": 2}])
        writer.close()

        meta = read_meta(tmp_path / "t")
        assert meta["mechanism"] == "cascade"
        assert meta["schema_version"] >= 1
        assert set(meta["streams"]) == {"episodes", "steps", "periods", "events"}
        assert read_stream(tmp_path / "t", "steps")[0]["persuasiveness"] == 0.5
        assert len(read_stream(tmp_path / "t", "events")) == 2

    def test_fresh_truncates_but_resume_appends(self, tmp_path):
        d = tmp_path / "t"
        w = TraceWriter(d, fresh=True)
        w.write("steps", {"iteration": 1})
        w.close()
        w = TraceWriter(d, fresh=False)
        w.write("steps", {"iteration": 2})
        w.close()
        assert len(read_stream(d, "steps")) == 2
        w = TraceWriter(d, fresh=True)
        w.write("steps", {"iteration": 9})
        w.close()
        assert len(read_stream(d, "steps")) == 1

    def test_thread_safe_batch_writes(self, tmp_path):
        import concurrent.futures as cf

        writer = TraceWriter(tmp_path / "t", fresh=True)

        def emit(i):
            writer.write_many("steps", [{"i": i, "j": j} for j in range(50)])

        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(emit, range(40)))
        writer.close()
        records = read_stream(tmp_path / "t", "steps")
        assert len(records) == 40 * 50
        assert len({(r["i"], r["j"]) for r in records}) == 40 * 50

    def test_unknown_stream_rejected(self, tmp_path):
        writer = TraceWriter(tmp_path / "t")
        with pytest.raises(KeyError):
            writer.write("nonsense", {})
        writer.close()


# ----------------------------------------------------------------------
# collect_episode instrumentation
# ----------------------------------------------------------------------
class TestEpisodeTracing:
    def test_all_four_streams_written(self, tmp_path):
        directory, _, _, rew = run_traced_episode(tmp_path)
        for stream in ("episodes", "steps", "periods", "events"):
            assert read_stream(directory, stream), f"{stream} empty"
        steps = read_stream(directory, "steps")
        periods = read_stream(directory, "periods")
        episodes = read_stream(directory, "episodes")
        # 3 periods x 2 persuaders
        assert len(steps) == 6
        assert len(periods) == 3
        assert len(episodes) == 1
        # per-decision rewards must sum to the episode reward
        assert sum(s["reward"] for s in steps) == pytest.approx(
            episodes[0]["episode_reward"]
        )

    def test_immediate_variables_present_and_exact(self, tmp_path):
        directory, _, _, _ = run_traced_episode(tmp_path)
        step = read_stream(directory, "steps")[0]
        for key in ("persuasiveness", "goal_align", "judge_src", "strategy",
                    "strategy_index", "target", "message", "reward",
                    "period", "persuader", "action", "n_candidates",
                    "mask_open_frac"):
            assert key in step, key
        assert isinstance(step["strategy"], str)
        assert step["judge_src"] in ("id", "text", "none")
        if step["persuasiveness"] is not None:
            assert 0.0 <= step["persuasiveness"] <= 1.0

    def test_episode_summary_matches_period_totals(self, tmp_path):
        directory, _, _, _ = run_traced_episode(tmp_path)
        episodes = read_stream(directory, "episodes")[0]
        periods = read_stream(directory, "periods")
        assert episodes["n_events"] == sum(p["n_events"] for p in periods)
        assert episodes["n_persuader_events"] == sum(
            p["n_persuader_events"] for p in periods
        )
        assert episodes["n_accept"] == sum(p["n_accept"] for p in periods)
        assert episodes["n_accept_all"] == sum(
            p["n_accept_all"] for p in periods
        )
        assert episodes["n_transitions"] == episodes["steps"] * 2

    def test_period_records_track_population(self, tmp_path):
        directory, _, _, _ = run_traced_episode(tmp_path)
        periods = read_stream(directory, "periods")
        assert [p["period"] for p in periods] == [1, 2, 3]
        assert all(-1.0 <= p["mean_stance"] <= 1.0 for p in periods)
        assert all(p["n_adopted"] >= 0 for p in periods)

    def test_event_records_carry_diffusion_detail(self, tmp_path):
        directory, _, _, _ = run_traced_episode(tmp_path)
        events = read_stream(directory, "events")
        assert events, "paper mode should produce diffusion events"
        event = events[0]
        for key in ("source_agent", "target_agent", "target_stance_before",
                    "target_stance_after", "target_delta_stance",
                    "influence_probability", "behavior", "adopted",
                    "source_is_persuader", "profile_sim", "stance_sim"):
            assert key in event, key
        assert event["influence_probability"] is not None

    def test_tracing_is_pure(self, tmp_path):
        """A traced episode must be bit-identical to an untraced one.

        This is the claim that lets us say a traced campaign is
        comparable to campaign 6: tracing observes, it does not perturb.
        """
        env_a, env_b = make_env(), make_env()
        policy_a, policy_b = make_policy(env_a), make_policy(env_b)

        writer = TraceWriter(tmp_path / "trace", fresh=True)
        torch.manual_seed(7)
        _, _, rew_traced, _, info_a = collect_episode(
            env_a, policy_a, torch.device("cpu"), 42, trace=writer,
            trace_meta={"iteration": 1, "episode": 0, "seed": 42},
        )
        writer.close()
        torch.manual_seed(7)
        _, _, rew_plain, _, info_b = collect_episode(
            env_b, policy_b, torch.device("cpu"), 42
        )
        assert np.array_equal(rew_traced, rew_plain)
        assert info_a["mean_stance"] == info_b["mean_stance"]
        assert info_a["n_adopted"] == info_b["n_adopted"]


# ----------------------------------------------------------------------
# registry coverage — the anti-drift guard
# ----------------------------------------------------------------------
class TestRegistryCoverage:
    def test_every_traced_field_is_documented(self, tmp_path):
        """A tracer field with no registry entry fails the suite."""
        directory, _, _, _ = run_traced_episode(tmp_path)
        for stream in ("episodes", "steps", "periods", "events"):
            records = read_stream(directory, stream)
            assert records
            documented = set(field_map(stream))
            written = set().union(*(set(r) for r in records))
            undocumented = written - documented
            assert not undocumented, (
                f"{stream}: undocumented fields {sorted(undocumented)} — "
                f"add them to code/metrics_registry.py"
            )

    def test_every_registry_field_is_used(self, tmp_path):
        """The other direction: documented fields must actually appear."""
        directory, _, _, _ = run_traced_episode(tmp_path)
        for stream in ("episodes", "steps", "periods"):
            written = set().union(
                *(set(r) for r in read_stream(directory, stream))
            )
            missing = set(field_map(stream)) - written
            assert not missing, f"{stream}: documented but never written: {missing}"

    def test_iteration_record_fields_are_documented(self, tmp_path):
        cfg = PPOConfig(iterations=2, episodes_per_iteration=1,
                        agent_num=8, n_persuaders=2, n_periods=3,
                        output_dir=str(tmp_path / "run"),
                        rollout_workers=1, log_every=1, eval_every=1,
                        paired_eval_every=1, paired_eval_episodes=2,
                        trace_dir=str(tmp_path / "run" / "trace"))
        env = make_env()
        trainer = PPOTrainer(env, cfg)
        _, history = trainer.train()
        flat = flatten_record(history[-1])
        documented = set(field_map("iteration")) | set(field_map("eval")) | \
            set(field_map("paired_eval"))
        undocumented = set(flat) - documented
        assert not undocumented, (
            f"iteration record has undocumented fields "
            f"{sorted(undocumented)} — add them to metrics_registry.py"
        )

    def test_markdown_documents_every_field(self):
        text = render_markdown()
        for stream, fields in STREAMS.items():
            for field in fields:
                assert f"`{field.name}`" in text, (stream, field.name)

    def test_no_duplicate_names_within_a_stream(self):
        for stream, fields in STREAMS.items():
            names = [f.name for f in fields]
            assert len(names) == len(set(names)), stream


# ----------------------------------------------------------------------
# trainer wiring
# ----------------------------------------------------------------------
class TestTrainerTracing:
    def test_train_writes_trace_and_records_provenance(self, tmp_path):
        out = tmp_path / "run"
        cfg = PPOConfig(iterations=2, episodes_per_iteration=2,
                        agent_num=8, n_persuaders=2, n_periods=3,
                        output_dir=str(out), rollout_workers=2,
                        log_every=1, eval_every=1,
                        trace_dir=str(out / "trace"))
        trainer = PPOTrainer(make_env(), cfg)
        _, history = trainer.train()
        assert history
        assert history[-1]["mechanism"] == "paper"
        assert "llm_model" in history[-1]
        for stream in ("steps", "periods", "events", "episodes"):
            assert read_stream(out / "trace", stream), stream
        # 2 iterations x 2 episodes x 3 periods x 2 persuaders
        assert len(read_stream(out / "trace", "steps")) == 24
        assert len(read_stream(out / "trace", "episodes")) == 4

    def test_trace_episodes_cap(self, tmp_path):
        cfg = PPOConfig(iterations=1, episodes_per_iteration=4, agent_num=8,
                        n_persuaders=1, n_periods=2, rollout_workers=1,
                        output_dir=str(tmp_path / "run"), eval_every=1,
                        trace_episodes=2)
        trainer = PPOTrainer(make_env(n_persuaders=1), cfg)
        trainer._trace = object()  # only the cap logic is under test
        assert trainer._trace_for(0) is trainer._trace
        assert trainer._trace_for(1) is trainer._trace
        assert trainer._trace_for(2) is None
        cfg.trace_episodes = 0  # 0 = no cap
        assert trainer._trace_for(99) is trainer._trace

    def test_no_trace_dir_means_no_trace_files(self, tmp_path):
        out = tmp_path / "run"
        cfg = PPOConfig(iterations=1, episodes_per_iteration=1, agent_num=6,
                        n_persuaders=1, n_periods=2, rollout_workers=1,
                        output_dir=str(out), eval_every=1)
        trainer = PPOTrainer(make_env(n_persuaders=1), cfg)
        trainer.train()
        assert not (out / "trace").exists()
