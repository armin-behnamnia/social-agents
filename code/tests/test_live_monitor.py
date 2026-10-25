"""Tests for the live monitor: incremental tailing, exact aggregates,
and the rendered dashboard (values + descriptions + charts)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from code import live_monitor as LM
from code.metrics_registry import STREAMS, render_markdown


def write_lines(path: Path, records: list[dict], partial: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r) + "\n" for r in records)
    if partial:
        payload = payload.rstrip("\n")
    with path.open("a") as fh:
        fh.write(payload)


class TestTraceTail:
    def test_incremental_tailing_counts_only_complete_lines(self, tmp_path):
        path = tmp_path / "steps.jsonl"
        tail = LM.TraceTail(path, "steps")
        assert tail.poll() == 0  # file does not exist yet

        write_lines(path, [{"iteration": 1, "reward": 0.5},
                           {"iteration": 1, "reward": 1.5}])
        assert tail.poll() == 2
        assert tail.records == 2
        assert tail.poll() == 0  # nothing new

        # a partially-written line must not be counted, then completes
        write_lines(path, [{"iteration": 2, "reward": 2.5}], partial=True)
        assert tail.poll() == 0
        with path.open("a") as fh:
            fh.write("\n")
        assert tail.poll() == 1
        assert tail.records == 3

    def test_numeric_aggregates_are_exact(self, tmp_path):
        path = tmp_path / "steps.jsonl"
        write_lines(path, [
            {"iteration": 1, "reward": 0.0, "persuasiveness": 0.25},
            {"iteration": 1, "reward": 1.0, "persuasiveness": 0.75},
            {"iteration": 2, "reward": 2.0, "persuasiveness": None},
        ])
        tail = LM.TraceTail(path, "steps")
        tail.poll()
        stats = tail.stats("reward")
        assert stats == {"n": 3, "mean": 1.0, "min": 0.0, "max": 2.0, "last": 2.0}
        # None must be skipped, not coerced to 0
        assert tail.stats("persuasiveness")["n"] == 2
        assert tail.stats("persuasiveness")["mean"] == pytest.approx(0.5)

    def test_per_iteration_means(self, tmp_path):
        path = tmp_path / "steps.jsonl"
        write_lines(path, [
            {"iteration": 1, "reward": 1.0},
            {"iteration": 1, "reward": 3.0},
            {"iteration": 2, "reward": 5.0},
        ])
        tail = LM.TraceTail(path, "steps")
        tail.poll()
        assert tail.iter_series("reward") == [(1.0, 2.0), (2.0, 5.0)]

    def test_histogram_bins_sum_to_n_with_edge_clipping(self, tmp_path):
        path = tmp_path / "steps.jsonl"
        write_lines(path, [
            {"persuasiveness": 0.0},   # first edge bin
            {"persuasiveness": 0.5},   # middle
            {"persuasiveness": 1.0},   # last edge bin
            {"persuasiveness": -3.0},  # below range -> first bin
            {"persuasiveness": 9.0},   # above range -> last bin
        ])
        tail = LM.TraceTail(path, "steps")
        tail.poll()
        counts = tail.hists["persuasiveness"]
        assert sum(counts) == 5
        assert counts[0] == 2 and counts[-1] == 2

    def test_categorical_and_strategy_tables(self, tmp_path):
        path = tmp_path / "steps.jsonl"
        write_lines(path, [
            {"judge_src": "id", "strategy": "angle=safety", "reward": 1.0},
            {"judge_src": "id", "strategy": "angle=safety", "reward": 3.0},
            {"judge_src": "text", "strategy": "angle=risk", "reward": 0.0},
        ])
        tail = LM.TraceTail(path, "steps")
        tail.poll()
        assert dict(tail.cats["judge_src"]) == {"id": 2, "text": 1}
        entry = tail.by_strategy["angle=safety"]
        assert entry["count"] == 2 and entry["reward_sum"] == 4.0

    def test_per_period_means(self, tmp_path):
        path = tmp_path / "periods.jsonl"
        write_lines(path, [
            {"period": 1, "mean_stance": 0.4, "n_adopted": 2},
            {"period": 1, "mean_stance": 0.6, "n_adopted": 4},
            {"period": 2, "mean_stance": 0.2, "n_adopted": 6},
        ])
        tail = LM.TraceTail(path, "periods")
        tail.poll()
        assert tail.period_series("mean_stance") == [(1.0, 0.5), (2.0, 0.2)]
        assert tail.period_series("n_adopted") == [(1.0, 3.0), (2.0, 6.0)]

    def test_malformed_lines_are_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "steps.jsonl"
        path.write_text('{"reward": 1.0}\nnot json\n{"reward": 3.0}\n')
        tail = LM.TraceTail(path, "steps")
        assert tail.poll() == 2
        assert tail.bad_lines == 1
        assert tail.stats("reward")["mean"] == 2.0


class TestRunState:
    @staticmethod
    def make_run(tmp_path, mechanism="cascade") -> Path:
        run = tmp_path / "run"
        trace = run / "trace"
        trace.mkdir(parents=True)
        (trace / "meta.json").write_text(json.dumps({
            "schema_version": 1, "mechanism": mechanism,
            "streams": ["episodes", "steps", "periods", "events"],
        }))
        history = []
        for it in (1, 2):
            history.append({
                "iteration": it, "mechanism": mechanism,
                "llm_model": "test/model",
                "mean_episode_reward": 1.0 + it, "mean_final_stance": -0.1 * it,
                "mean_adopted": 40.0 + it, "train_pi_loss": -0.01,
                "train_v_loss": 0.02, "train_entropy": 4.0, "train_kl": 0.001,
                "train_entropy_coef": 0.05, "train_strategy_entropy": 4.5,
                "train_judge": {"n": 10, "mean": 0.5, "std": 0.2,
                                "min": 0.1, "max": 0.9,
                                "saturated_frac": 0.0},
                "train_judge_src": {"id": 10, "text": 0, "none": 0},
                "train_goal_align": 0.8, "elapsed_s": 100.0 * it,
            })
        (run / "training_log.json").write_text(json.dumps({
            "config": {"iterations": 300, "agent_num": 50, "n_persuaders": 5,
                       "topology": "erdos_renyi", "seed": 0, "n_periods": 10,
                       "episodes_per_iteration": 32, "rollout_workers": 160,
                       "judge": "bt", "trace_episodes": 0,
                       "paired_eval_every": 20},
            "history": history,
        }))
        (run / "strategy_value.json").write_text(json.dumps({
            "strategy_entropy": 4.4, "n_decisions": 3,
            "rows": [{"strategy_index": 0, "strategy": "angle=safety",
                      "count": 2, "mean_reward": 0.5, "total_reward": 1.0}],
        }))
        write_lines(trace / "steps.jsonl", [
            {"iteration": 1, "episode": 0, "step": 0, "period": 1,
             "persuader": 0, "target": 7, "strategy_index": 0,
             "strategy": "angle=safety", "message": "hello world",
             "persuasiveness": 0.42, "judge_src": "id", "goal_align": 0.7,
             "reward": 0.3, "n_candidates": 4, "mask_open_frac": 1.0,
             "mechanism": mechanism, "ts": time.time()},
            {"iteration": 2, "episode": 0, "step": 0, "period": 1,
             "persuader": 2, "target": None, "strategy_index": 0,
             "strategy": "angle=safety", "message": "hi",
             "persuasiveness": None, "judge_src": "none", "goal_align": None,
             "reward": 0.0, "n_candidates": 0, "mask_open_frac": 1.0,
             "mechanism": mechanism, "ts": time.time()},
        ])
        write_lines(trace / "periods.jsonl", [
            {"iteration": 1, "episode": 0, "step": 0, "period": 1,
             "mean_stance": 0.3, "n_adopted": 41, "n_accept": 1,
             "n_accept_all": 3, "n_persuader_events": 1, "n_events": 4,
             "n_candidates": 4, "mask_open_frac": 1.0,
             "period_reward_sum": 0.3, "period_reward_mean": 0.3,
             "mechanism": mechanism, "ts": time.time()},
        ])
        write_lines(trace / "events.jsonl", [
            {"iteration": 1, "episode": 0, "step": 0, "period": 0,
             "source_agent": 0, "source_is_persuader": True, "target_agent": 7,
             "target_stance_before": 0.6, "target_stance_after": 0.3,
             "target_delta_stance": -0.3, "influence_probability": 0.55,
             "behavior": "accept", "adopted": 1, "interaction": "exposure",
             "mechanism": mechanism, "source_degree_centrality": 0.5,
             "target_degree_centrality": 0.5, "profile_sim": 0.5,
             "content_sim": 0.5, "stance_sim": 0.5, "degree_sim": 0.5,
             "persuasiveness": 0.42, "ts": time.time()},
        ])
        write_lines(trace / "episodes.jsonl", [
            {"iteration": 1, "episode": 0, "seed": 5, "mechanism": mechanism,
             "steps": 1, "episode_reward": 0.3, "mean_stance": 0.3,
             "n_adopted": 41, "n_events": 4, "n_persuader_events": 1,
             "n_accept": 1, "n_accept_all": 3, "n_idle_actions": 0,
             "n_transitions": 1, "ts": time.time()},
        ])
        return run

    def test_status_and_series(self, tmp_path):
        rs = LM.RunState(self.make_run(tmp_path))
        rs.poll()
        st = rs.status()
        assert st["mechanism"] == "cascade"
        assert st["iteration"] == 2
        assert st["iterations_total"] == 300
        assert st["model"] == "test/model"
        assert set(st["trace_streams"]) == {"steps", "periods", "events",
                                            "episodes"} - {"events"} | {"events"}
        assert rs.series("mean_episode_reward") == [(1.0, 2.0), (2.0, 3.0)]
        assert rs.history_stats("mean_adopted")["mean"] == pytest.approx(41.5)

    def test_variable_rows_cover_every_registry_field(self, tmp_path):
        rs = LM.RunState(self.make_run(tmp_path))
        rs.poll()
        rows = LM.variable_rows(rs)
        expected = {(stream, f.name) for stream, fields in STREAMS.items()
                    for f in fields}
        assert {(r["stream"], r["name"]) for r in rows} == expected

    def test_trace_values_reach_the_variable_table(self, tmp_path):
        rs = LM.RunState(self.make_run(tmp_path))
        rs.poll()
        rows = {(r["stream"], r["name"]): r for r in LM.variable_rows(rs)}
        assert rows[("steps", "persuasiveness")]["last"] == 0.42
        assert rows[("steps", "persuasiveness")]["n"] == 1  # None skipped
        assert rows[("events", "target_delta_stance")]["mean"] == -0.3

    def test_state_payload_is_json_serialisable(self, tmp_path):
        rs = LM.RunState(self.make_run(tmp_path))
        rs.poll()
        payload = LM.state_payload(rs)
        text = json.dumps(payload, default=str)
        assert "mean_episode_reward" in text
        assert payload["traces"]["steps"]["records"] == 2
        assert payload["traces"]["steps"]["per_iteration_mean"]["reward"]

    def test_untraced_run_is_reported_honestly(self, tmp_path):
        run = self.make_run(tmp_path)
        for name in ("steps", "periods", "events", "episodes"):
            (run / "trace" / f"{name}.jsonl").unlink()
        rs = LM.RunState(run)
        rs.poll()
        assert rs.status()["trace_streams"] == []
        body = LM.render_body(rs)
        assert "without <code>--trace</code>" in body
        # iteration metrics still render
        assert "Learning curve" in body


class TestRendering:
    def test_body_has_values_descriptions_and_charts(self, tmp_path):
        rs = LM.RunState(TestRunState.make_run(tmp_path))
        rs.poll()
        body = LM.render_body(rs)
        assert "mean_episode_reward" in body           # value + registry row
        assert "PPO update index" in body              # description
        assert body.count("<svg") >= 8                 # charts
        assert "Immediate variables" in body
        assert "trace/steps.jsonl" in body
        assert ">None<" not in body
        assert ">nan<" not in body

    def test_page_is_self_contained_with_refresh(self, tmp_path):
        rs = LM.RunState(TestRunState.make_run(tmp_path))
        rs.poll()
        page = LM.render_page(rs, refresh_s=7)
        assert page.startswith("<!doctype html>")
        assert "REFRESH_MS = 7000" in page
        assert "fetch('partial'" in page
        # no external resources: the dashboard must work offline
        assert "http://" not in page and "https://" not in page

    def test_text_report_lists_kpis_and_variables(self, tmp_path):
        rs = LM.RunState(TestRunState.make_run(tmp_path))
        rs.poll()
        text = LM.render_text(rs)
        assert "LIVE VALUES" in text and "Episode reward" in text
        assert "Message persuasiveness" in text
        assert "VARIABLES" in text and "mean_episode_reward" in text
        assert "PERIOD MEANS BY PERIOD" in text

    def test_chart_primitives_survive_degenerate_input(self):
        assert "no data" in LM.svg_line([])
        assert "no data" in LM.svg_hist([0, 0], 0.0, 1.0, title="t")
        assert "<svg" in LM.svg_bars([], [], title="t")
        assert "<svg" in LM.svg_spark([1.0])
        # single point, constant series
        one = LM.svg_line([{"label": "x", "pts": [(1.0, 5.0)]}])
        assert "<svg" in one and "circle" in one
        flat = LM.svg_line([{"label": "x", "pts": [(1.0, 2.0), (2.0, 2.0)]}])
        assert "polyline" in flat

    def test_find_newest_run(self, tmp_path):
        old = tmp_path / "output" / "old"
        new = tmp_path / "output" / "new"
        for d in (old, new):
            d.mkdir(parents=True)
            (d / "training_log.json").write_text("{}")
        import os
        os.utime(old / "training_log.json", (1, 1))
        os.utime(new / "training_log.json", (time.time(), time.time()))
        assert LM.find_newest_run(tmp_path / "output") == new


class TestMetricsDoc:
    def test_checked_in_doc_matches_the_registry(self):
        """METRICS.md must be regenerated when the registry changes."""
        doc = Path("METRICS.md")
        if not doc.is_file():
            pytest.skip("METRICS.md not generated yet")
        assert doc.read_text() == render_markdown(), (
            "METRICS.md is stale — run "
            "`PYTHONPATH=. .venv/bin/python code/metrics_registry.py --markdown`"
        )
