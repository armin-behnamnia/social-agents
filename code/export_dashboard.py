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

    if write_personas:
        _write_personas(data_dir / "v1" / "personas", names,
                        persuader_ids, profiles)

    info = {
        "label": label,
        "run_file_num": run_file_num,
        "follow_log": str(follow_path),
        "memory": str(memory_path),
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
