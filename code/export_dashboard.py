#!/usr/bin/env python
"""Export a recorded persuasion episode to the LLMxIO Dashboard format.

The dashboard (github.com/SIGNALS-Lab/LLMxIO_Dashboard) renders a 3D
social-media network from three CSVs per run plus three persona CSVs.
Our model is an opinion-diffusion / persuasion simulation, so we map it
onto that schema:

- each agent            -> a persona
    * seed persuaders   -> IO agents (``io<id>``)
    * everyone else     -> organic users (``u<id>``)
- the interaction graph -> ``follow_log`` (who follows whom, iteration 0)
- each period's persuader message        -> a tweet (type ``influence``)
- each *adoption*                        -> the adopter "retweets" the
  message that reached them (type ``retweet``, ``Retweeted From`` = that
  tweet's id) -> a propagation edge
- each non-adopting *accept*             -> the target "comments" on it
  (type ``comment``, ``Commented On`` = that tweet's id) -> a reply edge

So the dashboard's 3D network + timeline + cascade/hashtag analytics
render our diffusion process: seeds emit messages, adopters re-share
them (cascades), and the whole topology is visible from t0.

Usage::

    PYTHONPATH=. .venv/bin/python -m code.export_dashboard \
        --trajectory output/.../viz/trajectory.json \
        --data-dir /path/to/LLMxIO_Dashboard/data \
        --run-file-num 9 --label "RL persuasion (judge)"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------- personas

_IO_SUMMARY = (
    "You are an online user sponsored by a government-sponsored "
    "information campaign, engaged in online discussions about the "
    "policy. Your objective is to persuade others that the policy is "
    "harmful and should be rejected."
)


def _agent_name(agent_id: int, persuader_ids: set[int]) -> str:
    return (f"io{agent_id:02d}" if agent_id in persuader_ids
            else f"u{agent_id:02d}")


def _finish(path: Path) -> None:
    """Drop the trailing newline the csv module appends.

    papaparse (used by the dashboard) emits a spurious empty record when a
    file ends in a newline, and the dashboard does not guard against it
    (e.g. ``row.name.toLowerCase()`` throws). The shipped data files end
    without a trailing newline; match that.
    """
    data = path.read_bytes()
    path.write_bytes(data.rstrip(b"\r\n"))


def _write_personas(personas_dir: Path, names, persuader_ids,
                    profiles) -> None:
    """Rewrite the three persona CSVs the dashboard loads.

    Names must be single tokens (the dashboard rejects names with spaces).
    """
    personas_dir.mkdir(parents=True, exist_ok=True)
    organic, io_repub = [], []
    for aid, name in names.items():
        prof = profiles[aid] if aid < len(profiles) else "A social user."
        if aid in persuader_ids:
            io_repub.append({"Name": name, "Summary": _IO_SUMMARY})
        else:
            organic.append({"Name": name, "Summary": prof,
                            "Leaning": "Democrat"})

    def _dump(path: Path, rows, fields) -> None:
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        _finish(path)

    _dump(personas_dir / "uselection_user_personas_40.csv", organic,
          ["Name", "Summary", "Leaning"])
    # no Democratic IO agents in our scenario
    _dump(personas_dir / "uselection_influence_personas_dem_10.csv", [],
          ["Name", "Summary"])
    _dump(personas_dir / "uselection_influence_personas_repub_10.csv",
          io_repub, ["Name", "Summary"])


# ------------------------------------------------------- agent variables

# The dashboard's Twitter schema has no opinion variables, so a node
# description cannot say what our simulation is actually about. This
# sidecar carries them: one record per agent holding a time series per
# variable (index ``i`` = the population snapshot after period ``i``,
# i.e. iteration ``i`` on the dashboard's timeline) plus the log of
# influence events that agent received. ``AgentInspector.tsx`` (see
# code/dashboard/llmxio_integration.patch) fetches it and renders it, so
# opening a node shows that agent's variables **at the iteration the
# timeline is currently showing** — most importantly its current stance.
#
# Every variable carries its own ``label``/``unit``/``description`` so
# the dashboard renders a self-documenting panel: the same
# "descriptions travel with the numbers" rule as METRICS.md.

_VARIABLES = [
    {
        "key": "stance",
        "label": "Current stance",
        "unit": "[-1, +1]",
        "kind": "continuous",
        "chart": True,
        "description": (
            "The agent's opinion position on the policy after this "
            "period. -1 = fully aligned with the persuaders' view (the "
            "policy should be rejected), +1 = fully supports the policy, "
            "0 = neutral. Mutable: it moves by the judged persuasion "
            "increment of every influence the agent receives, so it can "
            "change in every period."
        ),
    },
    {
        "key": "adopted",
        "label": "Adopted (view flipped)",
        "unit": "0 / 1",
        "kind": "binary",
        "chart": False,
        "description": (
            "1 once the agent's preference state has flipped to the "
            "persuaders' side (it crossed the adoption threshold). "
            "Sticky: once 1 it stays 1 for the rest of the episode, so "
            "this is 'has ever adopted', not 'adopted this period'."
        ),
    },
    {
        "key": "exposure_cum",
        "label": "Influences received",
        "unit": "events (cumulative)",
        "kind": "count",
        "chart": True,
        "description": (
            "How many influence events have reached this agent so far "
            "(from persuaders or from other agents it is connected to). "
            "Each one is a judged persuasion attempt whose increment can "
            "move the stance."
        ),
    },
    {
        "key": "influence_cum",
        "label": "Net stance push received",
        "unit": "sum of Δstance (cumulative)",
        "kind": "continuous",
        "chart": True,
        "description": (
            "Sum of the signed persuasion increments this agent has "
            "received. Negative = pushed toward the persuaders' view. "
            "Differs from the stance change itself, which is clipped to "
            "[-1, +1] and gated by the acceptance rule."
        ),
    },
    {
        "key": "from_persuader_cum",
        "label": "Influences from a persuader",
        "unit": "events (cumulative)",
        "kind": "count",
        "chart": False,
        "description": (
            "The subset of influences that came directly from one of the "
            "5 RL persuaders rather than from a converted crowd agent."
        ),
    },
    {
        "key": "adopt_events_cum",
        "label": "Adoption events",
        "unit": "events (cumulative)",
        "kind": "count",
        "chart": False,
        "description": (
            "How many individual influence events this agent adopted on "
            "(normally 0 or 1: the event that flipped it)."
        ),
    },
]

_STATIC_VARIABLES = [
    {
        "key": "role",
        "label": "Role",
        "description": (
            "persuader = one of the RL-controlled campaign accounts that "
            "the policy drives; organic = a crowd agent that can only be "
            "influenced."
        ),
    },
    {
        "key": "degree",
        "label": "Network degree",
        "description": (
            "Number of neighbours in the interaction graph, i.e. how many "
            "agents can directly influence it (and how many it can pass a "
            "message on to)."
        ),
    },
    {
        "key": "initial_stance",
        "label": "Initial stance",
        "description":
            "The agent's stance before any influence (period t0).",
    },
    {
        "key": "leaning",
        "label": "Persona leaning",
        "description":
            "The synthetic persona's political leaning, used by the "
            "persuader policy as part of the target observation.",
    },
    {
        "key": "profile",
        "label": "Persona profile",
        "description": (
            "The synthetic profile used to condition message generation "
            "and judge scoring for this agent."
        ),
    },
]

# The imported `exposure` mechanism has NO stance dynamics at all -- its
# stance update is literally skipped ("FROZEN: no stance update" in
# `code/mech_sim.py::_expose_exposure`). There, an adoption is a purely
# behavioural flip (an LLM ACCEPT), and a stance never moves. A panel that
# described `stance` as "mutable, moves by the judged increment" would then
# promise movement the series cannot show, so the wording follows the data:
# if no influence in the episode carries a non-zero increment and every
# stance series is flat, the agent's stance was fixed for the whole run and
# the panel says so.
_FROZEN_DESCRIPTIONS = {
    "stance": (
        "The agent's opinion position on the policy, held FIXED for the "
        "whole episode: the imported exposure mechanism has no stance "
        "dynamics, so this is the agent's initial position and it does not "
        "move. What changes in this arm is behaviour, not opinion -- see "
        "'Adopted' and 'Influences received'."
    ),
    "adopted": (
        "1 once the agent has accepted an influence and flipped its "
        "preference state to the persuaders' side. In the exposure "
        "mechanism the trigger is an LLM ACCEPT decision (there is no "
        "stance threshold to cross). Sticky: once 1 it stays 1 for the "
        "rest of the episode, so this is 'has ever adopted', not 'adopted "
        "this period'."
    ),
    "exposure_cum": (
        "How many influence events have reached this agent so far (from "
        "persuaders or from other agents it is connected to). Each one is "
        "a judged persuasion attempt, but in this mechanism it cannot move "
        "the stance (frozen); it can only be accepted or not, and "
        "acceptance is what flips 'Adopted'."
    ),
    "influence_cum": (
        "Sum of the signed persuasion increments this agent has received. "
        "Always 0 in this mechanism: the exposure mechanism has no stance "
        "dynamics, so every increment is zero and no stance is ever "
        "pushed."
    ),
}


def _variables_for(stance_frozen: bool) -> list:
    """Variable metadata, worded for whether stances actually move here."""
    out = []
    for var in _VARIABLES:
        var = dict(var)
        if stance_frozen and var["key"] in _FROZEN_DESCRIPTIONS:
            var["description"] = _FROZEN_DESCRIPTIONS[var["key"]]
        out.append(var)
    return out


def _write_agent_variables(out_dir: Path, run_file_num: int, label: str,
                           traj_path: Path, traj: dict, names, persuader_ids,
                           profiles) -> Path:
    """Write ``agent_variables_50_repub_<n>.json`` next to the two CSVs."""
    periods = traj["periods"]
    n = int(traj["agent_num"])
    iters = list(range(len(periods)))
    stances = [[float(x) for x in p["after"]["stances"]] for p in periods]
    adopted = [set(int(a) for a in p["after"]["adopted"]) for p in periods]
    degree = {i: 0 for i in range(n)}
    for node, neighbors in traj["adjacency"]:
        degree[int(node)] = len(neighbors)

    series_keys = ("exposure_cum", "influence_cum", "from_persuader_cum",
                   "adopt_events_cum")
    series = {aid: {k: [0] * len(periods) for k in series_keys}
              for aid in range(n)}
    received: dict[int, list] = {aid: [] for aid in range(n)}

    for i, per in enumerate(periods):
        if i:  # carry the running totals forward, then add this period's
            for aid in range(n):
                for k in series_keys:
                    series[aid][k][i] = series[aid][k][i - 1]
        for e in per["events"]:
            s, t = int(e["source"]), int(e["target"])
            if not (0 <= t < n):
                continue
            d = float(e.get("delta_stance") or 0.0)
            adopted_event = bool(e.get("adopted")) or bool(
                e.get("reached_adoption_threshold"))
            series[t]["exposure_cum"][i] += 1
            series[t]["influence_cum"][i] += d
            if s in persuader_ids:
                series[t]["from_persuader_cum"][i] += 1
            if adopted_event:
                series[t]["adopt_events_cum"][i] += 1
            received[t].append({
                "iteration": i,
                "source": names.get(s, ""),
                "delta_stance": round(d, 4),
                "behavior": e.get("behavior"),
                "adopted": adopted_event,
                "from_persuader": s in persuader_ids,
            })

    agents = {}
    for aid in range(n):
        prof = profiles[aid] if aid < len(profiles) else "A social user."
        agents[names[aid]] = {
            "agent_id": aid,
            "role": "persuader" if aid in persuader_ids else "organic",
            "leaning": "Republican (IO)" if aid in persuader_ids
                       else "Democrat",
            "profile": prof,
            "degree": degree[aid],
            "series": {
                "stance": [round(stances[i][aid], 4) for i in iters],
                "adopted": [1 if aid in adopted[i] else 0 for i in iters],
                "influence_cum": [round(x, 4)
                                  for x in series[aid]["influence_cum"]],
                **{k: series[aid][k] for k in series_keys
                   if k != "influence_cum"},
            },
            "received": received[aid],
        }

    # Did stances move at all in this episode? (see _FROZEN_DESCRIPTIONS)
    any_push = any(
        abs(float(e.get("delta_stance") or 0.0)) > 1e-9
        for per in periods for e in per["events"]
    )
    stance_frozen = not any_push and all(
        len({round(stances[i][aid], 6) for i in iters}) == 1
        for aid in range(n)
    )

    payload = {
        "run": run_file_num,
        "label": label,
        "source": str(traj_path),
        "n_agents": n,
        "n_periods": traj.get("n_periods"),
        "iterations": iters,
        "iteration_labels": [p["after"].get("tag", f"t{i}")
                             for i, p in enumerate(periods)],
        "variables": _variables_for(stance_frozen),
        "static_variables": _STATIC_VARIABLES,
        "agents": agents,
    }
    path = out_dir / f"agent_variables_50_repub_{run_file_num}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return path


# ---------------------------------------------------------------- export

def export(traj_path: Path, data_dir: Path, run_file_num: int,
           label: str, profile_seed: int, campaign_hashtag: str,
           write_personas: bool) -> dict:
    traj = json.loads(Path(traj_path).read_text())
    n = int(traj["agent_num"])
    persuader_ids = set(int(x) for x in traj["persuader_ids"])
    periods = traj["periods"]
    names = {i: _agent_name(i, persuader_ids) for i in range(n)}

    # deterministic personas (same seed the env uses)
    try:
        from .profiles import generate_synthetic_profiles
        profiles = generate_synthetic_profiles(n, seed=profile_seed)
    except Exception:  # pragma: no cover - exporter convenience
        profiles = ["A social user." for _ in range(n)]

    out_dir = data_dir / "v1" / str(run_file_num)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"v1_{{kind}}_50_repub_{run_file_num}.csv"
    follow_path = out_dir / stem.format(kind="follow_log")
    memory_path = out_dir / stem.format(kind="memory")

    # ---- follow_log: the topology, all at iteration 0 -----------------
    n_follows = 0
    with follow_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Iteration", "Agent Name", "Followed User", "Reason"])
        for node, neighbors in traj["adjacency"]:
            for j in neighbors:
                w.writerow([0, names[int(node)], names[int(j)],
                            "network topology link"])
                n_follows += 1
    _finish(follow_path)

    # ---- memory: tweets + retweet/reply edges -------------------------
    rows = []
    tid = 0
    # (agent_id) -> id of that agent's most recent *reachable* tweet
    reachable: dict[int, int] = {}
    tweet_text: dict[int, str] = {}  # tid -> plain content (no hashtag)

    def _tweet(author_id, iteration, content, ttype, **extra):
        nonlocal tid
        tid += 1
        row = {
            "TweetID": tid,
            "Agent Name": names[author_id],
            "Iteration": iteration,
            "Content": content,
            "Type": ttype,
            "Retweets": 0, "Likes": 0, "Replies": 0,
            "Retweeted From": "", "Retweet Reason": "",
            "Strategy": "", "Commented On": "",
        }
        plain = extra.pop("_plain", None)
        row.update(extra)
        rows.append(row)
        tweet_text[tid] = str(plain or content)
        return tid

    # iteration 0: seeds post
    seed_posts = periods[0]["after"]["posts"]
    for pid in sorted(persuader_ids):
        post = seed_posts.get(str(pid)) or seed_posts.get(pid) or ""
        reachable[pid] = _tweet(
            pid, 0, f"{post} {campaign_hashtag}".strip(),
            "influence", Strategy="seed", _plain=post)

    # periods 1..K carry the actions taken and the diffusion they caused
    for k in range(1, len(periods)):
        per = periods[k]
        adopted_now = set(int(a) for a in per["after"]["adopted"])
        posts_now = per["after"]["posts"]

        # persuader messages this period
        for a in per["actions"]:
            pid = int(a["persuader"])
            reachable[pid] = _tweet(
                pid, k, f"{a['message']} {campaign_hashtag}".strip(),
                "influence", Strategy=a.get("strategy_label") or "",
                _plain=a["message"])

        # diffusion: adoption -> retweet, accept -> comment
        for e in per["events"]:
            s, t = int(e["source"]), int(e["target"])
            parent = reachable.get(s)
            if parent is None:
                continue
            behavior = e.get("behavior")
            adopted = bool(e.get("adopted")) or (t in adopted_now)
            if adopted:
                content = posts_now.get(str(t)) or posts_now.get(t) \
                    or "re-sharing"
                reachable[t] = _tweet(
                    t, k, f"{content} {campaign_hashtag}".strip(),
                    "retweet", **{"Retweeted From": parent,
                                  "Retweet Reason": behavior or "accept",
                                  "_plain": content})
            elif behavior == "accept":
                # A reply: quote the message it responds to (no campaign
                # hashtag, so hashtag-adoption analytics count only the
                # influence posts and their re-shares).
                quoted = tweet_text.get(parent, "")[:160]
                _tweet(t, k, f"Re: {quoted}".strip(), "comment",
                       **{"Commented On": parent})

    with memory_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "TweetID", "Agent Name", "Iteration", "Content", "Type",
            "Retweets", "Likes", "Replies", "Retweeted From",
            "Retweet Reason", "Strategy", "Commented On"])
        w.writeheader()
        w.writerows(rows)
    _finish(memory_path)

    # per-agent opinion variables (stance, exposures, influence log) that
    # the dashboard's Twitter schema has no room for
    vars_path = _write_agent_variables(
        out_dir, run_file_num, label, Path(traj_path), traj, names,
        persuader_ids, profiles)

    if write_personas:
        _write_personas(data_dir / "v1" / "personas", names,
                        persuader_ids, profiles)

    info = {
        "label": label,
        "run_file_num": run_file_num,
        "follow_log": str(follow_path),
        "memory": str(memory_path),
        "agent_variables": str(vars_path),
        "n_agents": n, "n_persuaders": len(persuader_ids),
        "n_follows": n_follows, "n_tweets": len(rows),
        "n_retweets": sum(1 for r in rows if r["Type"] == "retweet"),
        "n_comments": sum(1 for r in rows if r["Type"] == "comment"),
        "final_adopted": traj.get("final_adopted"),
    }
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--data-dir", required=True,
                    help="path to the dashboard's data/ directory")
    ap.add_argument("--run-file-num", type=int, default=9)
    ap.add_argument("--label", default="RL persuasion")
    ap.add_argument("--profile-seed", type=int, default=1234)
    ap.add_argument("--campaign-hashtag", default="#PolicyAlert")
    ap.add_argument("--no-personas", action="store_true")
    args = ap.parse_args()

    info = export(Path(args.trajectory), Path(args.data_dir),
                  args.run_file_num, args.label, args.profile_seed,
                  args.campaign_hashtag, not args.no_personas)
    print("[+] exported run:", json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
