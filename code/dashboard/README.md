# LLMxIO Dashboard integration

Visualize `social-agents` persuasion simulations with the
[SIGNALS-Lab/LLMxIO_Dashboard](https://github.com/SIGNALS-Lab/LLMxIO_Dashboard)
(3D social-network + timeline + cascade / centrality / IO-coordination
analytics).

The dashboard expects a Twitter-style schema. This directory holds the
bridge:

| file | purpose |
|------|---------|
| `../../code/export_dashboard.py` | converts a recorded episode (`trajectory.json`) into the dashboard's CSV schema **and** the per-agent variable sidecar |
| `llmxio_integration.patch` | the dashboard edits (run mapping, labels, **node inspector variables panel**, offline font) |
| `setup_dashboard.sh` | end-to-end: clone → patch → export → serve |

## What the mapping does

Our model is an opinion-diffusion / persuasion simulation, not a Twitter
sim. `export_dashboard.py` maps it onto the dashboard's schema:

| social-agents | dashboard |
|---------------|-----------|
| agent `i` | persona `u<i>` (organic user) / `io<i>` (seed persuader → IO agent) |
| network topology (`adjacency`) | `follow_log` edges at iteration 0 (whole network visible from t0) |
| seed posts (persuaders, t0) | tweets of type `influence` |
| persuader message each period | tweets of type `influence`, `Strategy` = `angle/evidence/address` |
| an **adoption** (target flips) | the adopter **retweets** the message that reached it → a propagation/cascade edge |
| a non-adopting **accept** | the target **comments** on that message → a reply edge |

Agents that merely observe (reject) produce no tweet, so the graph stays
readable. The campaign hashtag (`#PolicyAlert`, configurable) is attached
only to influence posts and their re-shares, so the dashboard's
hashtag-adoption analytics count *actual* adoption, not chatter.

## Node descriptions: the variables sidecar

The dashboard's own schema is Twitter-shaped — it has no opinion
variables, so a node's description could only ever show tweet traffic
(followers, tweets, replies). The exporter therefore also writes
`public/data/v1/<n>/agent_variables_50_repub_<n>.json`, and the patched
`AgentInspector` renders it **read at the iteration the timeline is
currently showing**:

- the agent's **current stance** (`-1` … `+1`), colour-coded, with a
  sparkline of its whole trajectory and the change since `t0`;
- every other time-series variable (adoptions, cumulative exposures,
  signed influence received, persuader-sourced exposures, adoption
  events) with label, unit and value;
- the static context (role, degree, initial stance, leaning, profile);
- the log of influences that actually reached the agent — who, when, the
  stance shift, the behaviour, and whether it came from a seed persuader.

```jsonc
{
  "iterations": [0, 1, 2, /* ... */],          // dashboard timeline
  "iteration_labels": ["t0", "t1", "t2"],       // episode periods
  "variables": [                                // one entry per series
    { "key": "stance", "label": "Stance", "unit": "[-1, +1]",
      "kind": "continuous", "chart": true,
      "description": "Current stance toward the policy: ..." }
  ],
  "static_variables": [ /* role, degree, initial_stance, leaning, profile */ ],
  "agents": {
    "u09": {
      "role": "organic", "degree": 3, "leaning": "persuadable",
      "series": { "stance": [0.42, 0.11, -0.37], "adopted": [0, 0, 1] },
      "received": [
        { "iteration": 1, "source": "io00", "delta_stance": -0.31,
          "behavior": "emerge", "adopted": false, "from_persuader": true }
      ]
    }
  }
}
```

Conventions worth knowing when reading it:

- **`series[i]` is the population snapshot after period `i`**, i.e. the
  dashboard's iteration `i`, so `series[currentIteration]` is what the
  panel shows. Every series has exactly `n_periods + 1` entries.
- Cumulative counters are true running totals (carried forward, then
  incremented by that period's events), so
  `Σ_agents exposure_cum[-1] == total influence events` — nothing is
  dropped or double-counted.
- `adopted` is a 0/1 **state** series (sticky), not an event count;
  `adopt_events_cum` is the event count.
- Persuaders are included (role `persuader`, constant stance `-1`) so
  the graph and the panel agree.
- Every variable carries a non-empty `description`; the panel exposes it
  as a hover title, and the registry it comes from is also what
  `code/export_dashboard.py` documents inline.
- The file is **optional**: for an upstream run (or ours before this
  change) the fetch 404s and the panel simply does not render.

## Prerequisites

- Node.js ≥ 18 (tested on v24), npm.
- The project venv (`.venv`) with `PYTHONPATH=.`.

## Quick start

```bash
cd /workspace/armin/social-agents

# 1. record an episode from each already-run campaign-6 policy
for spec in \
  "output/ppo_v2_bt/live_run2:baseline (no goal-delta)" \
  "output/ppo_v2_goal/live_goal_judge:trained (judge + goal-delta)"; do
  d="${spec%%:*}"; lbl="${spec#*:}"
  PERSUASION_STRATEGY_SPACE=v2 PYTHONPATH=. .venv/bin/python -m code.visualize \
    --mode live --policy "$d/policy.pt" --output-dir "$d/viz" \
    --agent-num 50 --n-persuaders 5 --n-periods 10 \
    --topology erdos_renyi --target-selection random_neighbor --obs-mode pair \
    --seed 900000 --label "$lbl"
done

# 2. clone + patch the dashboard, export the runs, and serve it (port 3009)
code/dashboard/setup_dashboard.sh /tmp/LLMxIO_Dashboard \
  output/ppo_v2_bt/live_run2/viz/trajectory.json:8 \
  output/ppo_v2_goal/live_goal_judge/viz/trajectory.json:9
```

Both runs are episodes of **experiments that were already run** (campaign 6:
the BT baseline and the judge + goal-δ arm) — no new training is required.

Mechanism episodes (runs 10 = C7 cascade, 11 = C8 exposure) are recorded
the same way, but the episode must be generated by the **same environment
the arm trained under** — `code/visualize` exposes the trainer's flags so
the two cannot drift:

```bash
# C7 cascade, from an existing checkpoint (needs the same strategy space
# the arm trained with: v2 = 72 arms)
PERSUASION_STRATEGY_SPACE=v2 PYTHONPATH=. .venv/bin/python -m code.visualize \
  --mode live --policy output/ppo_mech_cascade/live/policy.pt \
  --output-dir output/ppo_mech_cascade/live/viz_cascade \
  --agent-num 50 --n-persuaders 5 --n-periods 10 \
  --topology erdos_renyi --target-selection random_neighbor --obs-mode pair \
  --mechanism cascade --mech-beta-init --judge bt \
  --bt-table output/ppo_v2_bt/bt_judge.json --align judge --goal-delta \
  --seed 900000 --label "C7 cascade"

PYTHONPATH=. .venv/bin/python -m code.export_dashboard \
  --trajectory output/ppo_mech_cascade/live/viz_cascade/trajectory.json \
  --data-dir /tmp/LLMxIO_Dashboard/dashboard/public/data --run-file-num 10 \
  --label "RL Persuasion — C7 cascade (stance-aware)" --no-personas
```

For run 11 (exposure) use `--mechanism exposure --mech-beta-init
--reward-adoption-bonus 1.0 --align judge` and `PERSUASION_GOAL_LAMBDA=0.2`.
Loading a checkpoint under the *wrong* strategy space now fails with an
actionable error naming the env var (it used to fail with a state_dict
message about `policy_head.weight`).

Then open <http://localhost:3009> and pick a run in the *Experiment Run*
dropdown: 8 = campaign-6 baseline, 9 = campaign-6 judge + goal-δ,
10 = C7 cascade (stance-aware), 11 = C8 exposure (stance-less). Click any
node for its description — including its current stance at whatever
iteration the timeline is showing. (The port defaults to `3009`; override
with `PORT=3000 code/dashboard/setup_dashboard.sh ...`.)

`setup_dashboard.sh <dashboard_dir> <traj:runnum> [<traj:runnum> ...]`
- clones the dashboard at the pinned commit if `<dashboard_dir>` is absent,
- applies `llmxio_integration.patch`,
- runs `py -m code.export_dashboard` for each `traj:runnum` into
  `<dashboard_dir>/dashboard/public/data`,
- `npm ci` + `npm run dev`.

## Dashboard edits (`llmxio_integration.patch`)

Pinned to dashboard commit `9f913ee`:

1. `utils/dataParser.ts` — runs 8–11 point at our data via the legacy
   path `data/v1/<n>/v1_{follow_log,memory}_50_repub_<n>.csv`.
2. `components/SettingsPanel.tsx` — label runs 8–11 in the dropdown.
3. `store/simulationStore.ts` — default run = 10 (C7 cascade).
4. `components/AgentInspector.tsx` — fetch
   `agent_variables_50_repub_<run>.json` and render the
   [variables panel](#node-descriptions-the-variables-sidecar)
   (current stance + sparkline at the iteration on screen, the other
   variables with labels/units/descriptions, agent context, and the
   influences received). Self-contained: one loader with a per-run cache,
   and it renders nothing when the fetch 404s.
5. `utils/precomputedMetrics.ts` — runs with no narrative/hashtag
   sidecar (ours) 404 quietly instead of logging an error on every load.
6. `app/layout.tsx` — drop `next/font/google` (no route to
   fonts.googleapis.com here) so `next build` succeeds; falls back to the
   system sans stack.
7. `__tests__/components/AgentInspectorVariables.test.tsx` — 5 render
   tests for the panel (stance at the shown iteration, labelled
   variables, descriptions as hover titles, the received-influence log,
   and the 404 path).

## Verified

- **Patch applies cleanly** to a pristine clone at the pinned commit and
  is **type-clean** (`npx tsc --noEmit`) outside `__tests__/`; `next build`
  previously type-checked the same patch.
- **No test regressions**: the upstream suite is already largely stale
  (pristine: 46 failed / 107 passed). With the patch: 46 failed /
  112 passed — i.e. exactly the 5 new passing tests, no new failures.
- The dashboard's *real* TypeScript pipeline (`loadSimulationData` →
  `processSimulationData` → `precomputeAllMetrics`) was bundled with
  esbuild and run in Node against the exported data:

| run | agents | edges | maxIter |
|-----|--------|-------|---------|
| 8 (baseline) | 50 | 180 follows + 70 replies + 12 retweets = 262 | 10 |
| 9 (trained) | 50 | 180 follows + 80 replies + 19 retweets = 279 | 10 |

Baseline run: 16/50 adopted (mean stance 0.485 → 0.181, reward 8.61);
trained run: 19/50 adopted (0.485 → 0.142, 9.02). Each has 5 IO
(Republican) + 45 organic (Democratic) agents; all 8 metric blocks
precompute without error.

- The **variables sidecar** is checked on a real episode (run 10, C7
  cascade): 50 agents × 11 iterations, `Σ exposure_cum[-1]` equals the
  trajectory's event count exactly, `influence_cum` equals the signed sum
  of the received deltas, the `stance` series reproduces the trajectory
  snapshots for all 50 agents, no agent is logged as exposing itself, and
  every persuader stays at stance `-1` with no received events.
  `code/tests/test_export_dashboard.py` pins all of this on a synthetic
  trajectory (22 tests).

## Caveats

- The dashboard's optional per-run sidecars (`narrative_embeddings.json`,
  `hashtag_adoption_metrics.json`) are absent for our runs; the loaders
  return `null` and the corresponding panels stay empty. Our own sidecar
  (`agent_variables_*.json`) is different: it is written by the exporter
  and, unlike those two, *is* consumed — see the variables panel above.
- The dashboard's current iteration is the exported `Iteration` column,
  which is the episode period; the variables sidecar uses the same
  numbering (index `i` = snapshot after period `i`), which is why
  `series[currentIteration]` is the right readout at any point on the
  timeline.
- The personas CSVs are rewritten (our agents replace the uselection
  personas) because the dashboard loads personas globally.
- CSV files must **not** end with a trailing newline: papaparse emits a
  spurious empty record that the dashboard does not guard against. The
  exporter strips it.
