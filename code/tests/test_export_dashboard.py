"""Tests for the LLMxIO dashboard exporter (code/export_dashboard.py).

The interesting half of the export is the per-agent *variable sidecar*
(``agent_variables_50_repub_<n>.json``): the dashboard's Twitter schema
has no opinion variables, so this is the only place the node inspector
can learn an agent's current stance. The tests pin the accounting a
reader will rely on:

- every series has one entry per timeline iteration (index i = the
  population snapshot after period i, which is iteration i on the
  dashboard's timeline);
- cumulative counters are true running totals (carried forward, then
  incremented by this period's events);
- the summed exposure counter equals the number of influence events, so
  no event is dropped or double counted;
- the adopted series is sticky, because the dashboard shows it as state;
- every variable ships a non-empty description, and the series/variable
  sets agree in both directions.

Everything is built from a synthetic trajectory, so the tests do not
depend on output/ artefacts.
"""

from __future__ import annotations

import csv
import json

import pytest

from code.export_dashboard import export

N_AGENTS = 6
PERSUADERS = [0, 1]

# (node, [neighbours]) — a small connected graph, every agent reachable
ADJACENCY = [
    [0, [2, 3]],
    [1, [4]],
    [2, [5]],
    [3, [2]],
    [4, [3]],
    [5, [0]],
]

# events[period_index] — period 0 always has none (it is the t0 snapshot)
EVENTS = [
    [],
    [
        {"source": 0, "target": 2, "delta_stance": -0.5, "behavior": "accept",
         "adopted": True, "from_persuader": True},
        {"source": 0, "target": 3, "delta_stance": -0.2, "behavior": "accept",
         "adopted": False, "from_persuader": True},
    ],
    [
        {"source": 2, "target": 5, "delta_stance": -0.3, "behavior": "accept",
         "adopted": True, "from_persuader": False},
        {"source": 0, "target": 3, "delta_stance": -0.4, "behavior": "accept",
         "adopted": True, "from_persuader": True},
    ],
]

STANCES = [
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [-1.0, -1.0, -0.1, 0.3, 0.5, 0.5],
    [-1.0, -1.0, -0.1, -0.1, 0.5, -0.1],
]
ADOPTED = [[], [2], [2, 3, 5]]


def _trajectory() -> dict:
    periods = []
    for i, stances in enumerate(STANCES):
        periods.append({
            "actions": [],
            "events": EVENTS[i],
            "after": {
                "tag": f"t{i}",
                "period": i,
                "stances": stances,
                "adopted": ADOPTED[i],
                "posts": {str(a): f"post {a}" for a in ADOPTED[i]},
            },
        })
    return {
        "seed": 0,
        "agent_num": N_AGENTS,
        "n_periods": len(periods) - 1,
        "n_persuaders": len(PERSUADERS),
        "persuader_ids": PERSUADERS,
        "adjacency": ADJACENCY,
        "periods": periods,
        "rewards": [],
        "total_reward": 0.0,
        "final_mean_stance": -0.3,
        "final_adopted": len(ADOPTED[-1]),
        "initial_mean_stance": 0.5,
    }


@pytest.fixture()
def exported(tmp_path):
    """Export the synthetic trajectory and return (info, sidecar)."""
    traj_path = tmp_path / "trajectory.json"
    traj_path.write_text(json.dumps(_trajectory()))
    info = export(traj_path, tmp_path / "data", run_file_num=42,
                  label="synthetic", profile_seed=1234,
                  campaign_hashtag="#PolicyAlert", write_personas=True)
    sidecar = json.loads(
        (tmp_path / "data" / "v1" / "42"
         / "agent_variables_50_repub_42.json").read_text())
    return info, sidecar


# --- a frozen-stance episode: the imported `exposure` mechanism ----------
# Stances are identical at every snapshot and every increment is 0.0, yet
# agents still adopt (the flip is behavioural). This is what run 11 looks
# like, and the panel must not promise stance movement it cannot show.
FROZEN_STANCE_ROW = [0.5, -0.6, -0.4, 0.2, 0.5, -0.1]
FROZEN_STANCES = [list(FROZEN_STANCE_ROW) for _ in STANCES]
FROZEN_EVENTS = [
    [],
    [{"source": 0, "target": 2, "delta_stance": 0.0, "behavior": "accept",
      "adopted": True, "from_persuader": True}],
    [{"source": 2, "target": 5, "delta_stance": 0.0, "behavior": "accept",
      "adopted": True, "from_persuader": False}],
]
FROZEN_ADOPTED = [[], [2], [2, 5]]


def _frozen_trajectory() -> dict:
    traj = _trajectory()
    for i, period in enumerate(traj["periods"]):
        period["events"] = FROZEN_EVENTS[i]
        period["after"]["stances"] = FROZEN_STANCES[i]
        period["after"]["adopted"] = FROZEN_ADOPTED[i]
    traj["final_adopted"] = len(FROZEN_ADOPTED[-1])
    return traj


@pytest.fixture()
def frozen_exported(tmp_path):
    """Export a frozen-stance trajectory and return (info, sidecar)."""
    traj_path = tmp_path / "trajectory.json"
    traj_path.write_text(json.dumps(_frozen_trajectory()))
    info = export(traj_path, tmp_path / "data", run_file_num=43,
                  label="frozen", profile_seed=1234,
                  campaign_hashtag="#PolicyAlert", write_personas=True)
    sidecar = json.loads(
        (tmp_path / "data" / "v1" / "43"
         / "agent_variables_50_repub_43.json").read_text())
    return info, sidecar


class TestSidecarShape:
    def test_all_three_files_written(self, exported, tmp_path):
        info, _ = exported
        for key in ("follow_log", "memory", "agent_variables"):
            assert key in info
            assert (tmp_path / "data").exists()
            assert info[key]

    def test_one_iteration_per_snapshot(self, exported):
        _, side = exported
        assert side["iterations"] == [0, 1, 2]
        assert side["iteration_labels"] == ["t0", "t1", "t2"]
        assert side["n_agents"] == N_AGENTS

    def test_every_agent_present_and_named(self, exported):
        _, side = exported
        assert set(side["agents"]) == {
            "io00", "io01", "u02", "u03", "u04", "u05"}
        assert side["agents"]["io00"]["role"] == "persuader"
        assert side["agents"]["u02"]["role"] == "organic"

    def test_series_lengths_are_consistent(self, exported):
        _, side = exported
        n_iter = len(side["iterations"])
        for name, agent in side["agents"].items():
            for key, vals in agent["series"].items():
                assert len(vals) == n_iter, f"{name}.{key}"

    def test_stance_series_matches_snapshots(self, exported):
        _, side = exported
        for aid in range(N_AGENTS):
            expected = [STANCES[i][aid] for i in range(len(STANCES))]
            name = f"io{aid:02d}" if aid in PERSUADERS else f"u{aid:02d}"
            assert side["agents"][name]["series"]["stance"] == expected

    def test_static_degree_matches_topology(self, exported):
        _, side = exported
        for node, neighbours in ADJACENCY:
            name = f"io{node:02d}" if node in PERSUADERS \
                else f"u{node:02d}"
            assert side["agents"][name]["degree"] == len(neighbours)


class TestCumulativeAccounting:
    def test_exposures_are_a_true_running_total(self, exported):
        _, side = exported
        # agent 3 is hit in period 1 and again in period 2
        assert side["agents"]["u03"]["series"]["exposure_cum"] == [0, 1, 2]
        # agent 5 only in period 2 — the period-1 zero must carry forward
        assert side["agents"]["u05"]["series"]["exposure_cum"] == [0, 0, 1]

    def test_summed_exposures_equal_event_count(self, exported):
        _, side = exported
        total = sum(a["series"]["exposure_cum"][-1]
                    for a in side["agents"].values())
        assert total == sum(len(e) for e in EVENTS)

    def test_influence_is_the_signed_running_sum(self, exported):
        _, side = exported
        assert side["agents"]["u02"]["series"]["influence_cum"] == \
            [0, -0.5, -0.5]
        assert side["agents"]["u03"]["series"]["influence_cum"] == \
            [0, -0.2, -0.6]

    def test_from_persuader_counts_only_persuader_sources(self, exported):
        _, side = exported
        # u03 is hit by persuader 0 twice
        assert side["agents"]["u03"]["series"]["from_persuader_cum"] == \
            [0, 1, 2]
        # u05 is hit by u02 (a converted crowd agent), never by a persuader
        assert side["agents"]["u05"]["series"]["from_persuader_cum"] == \
            [0, 0, 0]

    def test_adopt_events_count_flagged_events(self, exported):
        _, side = exported
        assert side["agents"]["u03"]["series"]["adopt_events_cum"] == \
            [0, 0, 1]

    def test_adopted_series_is_sticky(self, exported):
        _, side = exported
        for name, agent in side["agents"].items():
            adopted = agent["series"]["adopted"]
            assert adopted == sorted(adopted), f"{name} un-adopted"
            assert set(adopted) <= {0, 1}
        assert side["agents"]["u02"]["series"]["adopted"] == [0, 1, 1]
        assert side["agents"]["u05"]["series"]["adopted"] == [0, 0, 1]

    def test_persuaders_are_untouched(self, exported):
        _, side = exported
        for name in ("io00", "io01"):
            series = side["agents"][name]["series"]
            assert series["exposure_cum"] == [0, 0, 0]
            assert series["influence_cum"] == [0, 0, 0]


class TestReceivedLog:
    def test_each_event_appears_once_under_its_target(self, exported):
        _, side = exported
        seen = [e for a in side["agents"].values() for e in a["received"]]
        assert len(seen) == sum(len(e) for e in EVENTS)
        assert sum(1 for e in seen if e["from_persuader"]) == 3

    def test_log_carries_the_event_fields(self, exported):
        _, side = exported
        log = side["agents"]["u03"]["received"]
        assert [e["iteration"] for e in log] == [1, 2]
        assert log[0]["source"] == "io00"
        assert log[0]["delta_stance"] == -0.2
        assert log[0]["behavior"] == "accept"
        assert log[0]["adopted"] is False
        assert log[1]["adopted"] is True

    def test_log_is_chronological(self, exported):
        _, side = exported
        for agent in side["agents"].values():
            its = [e["iteration"] for e in agent["received"]]
            assert its == sorted(its)


class TestDescriptions:
    def test_every_variable_is_documented(self, exported):
        _, side = exported
        for var in side["variables"] + side["static_variables"]:
            assert var["key"] and var["label"]
            assert var["description"].strip()
            assert len(var["description"]) > 40, var["key"]

    def test_variable_keys_are_unique_and_cover_every_series(self, exported):
        _, side = exported
        keys = [v["key"] for v in side["variables"]]
        assert len(keys) == len(set(keys))
        series_keys = set()
        for agent in side["agents"].values():
            series_keys |= set(agent["series"])
        assert series_keys == set(keys)

    def test_stance_is_charted_and_binary_is_not(self, exported):
        _, side = exported
        by_key = {v["key"]: v for v in side["variables"]}
        assert by_key["stance"]["chart"] is True
        assert by_key["adopted"]["chart"] is False
        assert by_key["stance"]["kind"] == "continuous"


class TestFrozenStanceWording:
    """The panel's wording has to match the series it describes.

    In the imported `exposure` mechanism the stance update is skipped
    outright, so a description that says the stance "moves by the judged
    increment" would promise movement the file cannot show.
    """

    def test_frozen_episode_says_the_stance_is_fixed(self, frozen_exported):
        _, side = frozen_exported
        by_key = {v["key"]: v for v in side["variables"]}
        stance = by_key["stance"]["description"]
        assert "FIXED" in stance
        assert "no stance dynamics" in stance
        assert "Mutable" not in stance

    def test_frozen_episode_says_nothing_was_pushed(self, frozen_exported):
        _, side = frozen_exported
        by_key = {v["key"]: v for v in side["variables"]}
        assert "Always 0" in by_key["influence_cum"]["description"]

    def test_frozen_episode_really_has_flat_stances(self, frozen_exported):
        _, side = frozen_exported
        for agent in side["agents"].values():
            assert len(set(agent["series"]["stance"])) == 1
            assert agent["series"]["influence_cum"] == [0, 0, 0]
        # ...and still adopts, so the panel is not just describing a no-op
        assert any(a["series"]["adopted"][-1] for a in side["agents"].values())

    def test_every_variable_is_documented_when_frozen(self, frozen_exported):
        _, side = frozen_exported
        for var in side["variables"] + side["static_variables"]:
            assert var["key"] and var["label"]
            assert len(var["description"].strip()) > 40, var["key"]

    def test_moving_episode_keeps_the_mutable_wording(self, exported):
        _, side = exported
        by_key = {v["key"]: v for v in side["variables"]}
        assert "Mutable" in by_key["stance"]["description"]
        assert "FIXED" not in by_key["stance"]["description"]
        assert "Always 0" not in by_key["influence_cum"]["description"]


class TestCsvHygiene:
    def test_csvs_have_no_trailing_newline(self, exported, tmp_path):
        """papaparse emits a spurious empty record on a trailing newline."""
        for key in ("follow_log", "memory"):
            _, data = exported
            raw = (tmp_path / "data" / "v1" / "42").glob(
                f"v1_{'follow_log' if key == 'follow_log' else 'memory'}"
                "_50_repub_42.csv")
            path = next(raw)
            assert not path.read_bytes().endswith(b"\n")

    def test_memory_csv_header_is_the_dashboard_schema(self, exported,
                                                       tmp_path):
        path = tmp_path / "data" / "v1" / "42" \
            / "v1_memory_50_repub_42.csv"
        with path.open() as fh:
            header = next(csv.reader(fh))
        assert header == [
            "TweetID", "Agent Name", "Iteration", "Content", "Type",
            "Retweets", "Likes", "Replies", "Retweeted From",
            "Retweet Reason", "Strategy", "Commented On",
        ]

    def test_adoption_becomes_a_retweet_of_the_message_that_reached_it(
            self, exported, tmp_path):
        path = tmp_path / "data" / "v1" / "42" \
            / "v1_memory_50_repub_42.csv"
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        retweets = [r for r in rows if r["Type"] == "retweet"]
        # 3 adoptions in the synthetic trace (u02, u05, u03->again)
        assert len(retweets) == 3
        for rt in retweets:
            assert rt["Retweeted From"], "retweet must point at its parent"
