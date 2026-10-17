# LLMxIO Dashboard integration

Visualize `social-agents` persuasion simulations with the
[SIGNALS-Lab/LLMxIO_Dashboard](https://github.com/SIGNALS-Lab/LLMxIO_Dashboard)
(3D social-network + timeline + cascade / centrality / IO-coordination
analytics).

The dashboard expects a Twitter-style schema. This directory holds the
bridge:

| file | purpose |
|------|---------|
| `../../code/export_dashboard.py` | converts a recorded episode (`trajectory.json`) into the dashboard's CSV schema |
| `llmxio_integration.patch` | the small dashboard edits (run mapping, labels, offline font) |
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

## Prerequisites

- Node.js ≥ 18 (tested on v24), npm.
- The project venv (`.venv`) with `PYTHONPATH=.`.

## Quick start

```bash
cd /workspace/armin/social-agents

# 1. record an episode from a trained policy (or omit --policy for random)
PERSUASION_STRATEGY_SPACE=v2 PYTHONPATH=. .venv/bin/python -m code.visualize \
  --mode live --policy output/ppo_v2_goal/live_goal_judge/policy.pt \
  --output-dir output/ppo_v2_goal/live_goal_judge/viz \
  --agent-num 50 --n-persuaders 5 --n-periods 10 \
  --topology erdos_renyi --target-selection random_neighbor --obs-mode pair \
  --seed 900000

# 2. clone + patch the dashboard, export the runs, and serve it
code/dashboard/setup_dashboard.sh /tmp/LLMxIO_Dashboard \
  output/ppo_v2_goal/live_goal_judge/viz/trajectory.json:9 \
  output/ppo_v2_goal/random_viz/trajectory.json:8
```

Then open <http://localhost:3000> and pick **"RL Persuasion — social-agents
(judge)"** (run 9) or **"RL Persuasion — random policy"** (run 8) in the
*Experiment Run* dropdown.

`setup_dashboard.sh <dashboard_dir> <traj:runnum> [<traj:runnum> ...]`
- clones the dashboard at the pinned commit if `<dashboard_dir>` is absent,
- applies `llmxio_integration.patch`,
- runs `py -m code.export_dashboard` for each `traj:runnum` into
  `<dashboard_dir>/dashboard/public/data`,
- `npm ci` + `npm run dev`.

## Dashboard edits (`llmxio_integration.patch`)

Pinned to dashboard commit `9f913ee`:

1. `utils/dataParser.ts` — runs 8/9 point at our data via the legacy
   path `data/v1/<n>/v1_{follow_log,memory}_50_repub_<n>.csv`.
2. `components/SettingsPanel.tsx` — relabel the run-8/9 dropdown options.
3. `store/simulationStore.ts` — default run = 9 (our trained policy).
4. `app/layout.tsx` — drop `next/font/google` (no route to
   fonts.googleapis.com here) so `next build` succeeds; falls back to the
   system sans stack.

## Verified

The dashboard's *real* TypeScript pipeline (`loadSimulationData` →
`processSimulationData` → `precomputeAllMetrics`) was bundled with esbuild
and run in Node against the exported data:

| run | agents | edges | maxIter |
|-----|--------|-------|---------|
| 8 (random) | 50 | 180 follows + 56 replies + 10 retweets = 246 | 10 |
| 9 (trained) | 50 | 180 follows + 80 replies + 19 retweets = 279 | 10 |

Trained run: 5 IO (Republican) + 45 organic (Democratic); all 8 metric
blocks precompute without error. `next build` type-checks the patch.

## Caveats

- The dashboard's optional per-run sidecars (`narrative_embeddings.json`,
  `hashtag_adoption_metrics.json`) are absent for our runs; the loaders
  return `null` and the corresponding panels stay empty.
- The personas CSVs are rewritten (our agents replace the uselection
  personas) because the dashboard loads personas globally.
- CSV files must **not** end with a trailing newline: papaparse emits a
  spurious empty record that the dashboard does not guard against. The
  exporter strips it.
