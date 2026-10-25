# Metrics and variables

**Generated** by `python code/metrics_registry.py` from `code/metrics_registry.py` — do not edit by hand.

Every variable a run records, with its meaning. The live dashboard (`code/live_monitor.py`) renders exactly these fields and their descriptions, so the two can never disagree.

## Files

| file | stream | contents |
|---|---|---|
| `<run>/training_log.json` | iteration / eval / paired_eval | per-update aggregates |
| `<run>/strategy_value.json` | strategy | per-arm value table |
| `<run>/trace/steps.jsonl` | steps | immediate per-decision variables |
| `<run>/trace/periods.jsonl` | periods | per-period population state |
| `<run>/trace/events.jsonl` | events | per-hop diffusion events |
| `<run>/trace/episodes.jsonl` | episodes | per-episode summaries |
| `<run>/trace/meta.json` | — | trace provenance (schema, config) |

Trace streams are written only when training runs with `--trace`.

## How to view a run

```bash
# live dashboard (auto-detects the newest run under output/)
PYTHONPATH=. .venv/bin/python code/live_monitor.py --port 3010

# same data in the terminal
PYTHONPATH=. .venv/bin/python code/live_monitor.py --text

# machine-readable exact values
curl -s localhost:3010/api/state | jq .
```

## Iteration metrics (`training_log.json` → `history[]`)

| variable | kind | unit / range | description |
|---|---|---|---|
| `iteration` | id | count | PPO update index (1-based). One update consumes episodes_per_iteration rollout episodes. |
| `mechanism` | cat | category | Contagion mechanism in force (paper / cascade / exposure), recorded per iteration for provenance. |
| `llm_model` | str | model id | LLM model id the run used (provenance; the gateway serves more than one model and .env is not versioned). |
| `mean_episode_reward` | num | reward units | Mean UNDISCOUNTED episode return over this iteration's rollout episodes: the summed per-period team reward. This is the learning curve's y-axis; see the reward decomposition at the top of this module for what it contains. |
| `mean_final_stance` | num | stance [-1,1] | Population mean latent stance at the END of the episode (-1 oppose … +1 support). The persuader argues the opposing side, so LOWER is better (campaigns start near +0.65). |
| `mean_adopted` | num | agents | Mean number of agents with preference_state == 1 at episode end (includes the persuader seeds; 50 of 50 means the whole population adopted). |
| `train_pi_loss` | num | loss | PPO clipped policy (actor) loss, averaged over minibatches and epochs of this update. |
| `train_v_loss` | num | loss | Critic (value) loss, averaged over minibatches and epochs. |
| `train_entropy` | num | nats | Mean policy entropy over the batch, in nats. Falls as the policy commits to a strategy; near 0 means near-deterministic choices (which is also when it can lock onto a judge exploit). |
| `train_kl` | num | nats | Approximate KL(old‖new) between the policy before and after the update. Large values mean the update moved the distribution a lot; the update is skipped past --target-kl. |
| `train_entropy_coef` | num | coefficient | Entropy bonus coefficient actually used this iteration (linearly annealed from entropy_coef_start to entropy_coef_end). |
| `train_strategy_entropy` | num | nats | Entropy (nats) of the CUMULATIVE empirical strategy distribution over the whole run so far — not this iteration. Monotone-ish; use it to see exploration breadth, not per-update change. |
| `train_judge.n` | num | messages | How many delivered messages were scored this iteration (idle periods, where the crowd is exhausted, are excluded rather than counted as a fabricated 1.0). |
| `train_judge.mean` | num | score [0,1] | Mean persuasiveness of the messages delivered this iteration: the reward model's score (BT expected win rate in bt mode, the absolute 0-1 LLM score otherwise). This is the IMMEDIATE signal the policy is optimising against. |
| `train_judge.std` | num | score | Spread of this iteration's persuasiveness scores; a collapsing std means the policy has narrowed onto one message family. |
| `train_judge.min` | num | score [0,1] | Worst message score of the iteration. |
| `train_judge.max` | num | score [0,1] | Best message score of the iteration. |
| `train_judge.saturated_frac` | num | fraction [0,1] | Fraction of this iteration's scores >= 0.95. A rising value is the signature of judge exploitation (the policy found the reward model's ceiling), so watch it alongside reward. |
| `train_judge_src.id` | num | lookups | Messages scored through the BT identity index (strategy, profile) — the intended, model-free path. |
| `train_judge_src.text` | num | lookups | Messages that missed the identity index and fell back to content-based scoring. Any growth here means the reward model is not what we think it is; the trainer warns. |
| `train_judge_src.none` | num | lookups | Persuader-decisions with no judge lookup (idle period: no eligible target). |
| `train_goal_align` | num | alignment [0,1] | Mean Track-B goal alignment a(m) of this iteration's delivered messages: how well each message serves the persuader's goal, independent of whether the target accepts it. |
| `elapsed_s` | num | seconds | Wall-clock seconds since the run started (monotone across a --resume). |

## Evaluation metrics (added every `--eval-every` iterations)

| variable | kind | unit / range | description |
|---|---|---|---|
| `eval_greedy_reward` | num | reward units | Greedy-policy mean episode reward over --eval-episodes episodes with FRESH seeds. Noisy (unpaired) — prefer the paired eval for model selection. |
| `eval_strategy_entropy` | num | nats | Entropy of the eval episodes' strategy-usage distribution. |
| `eval_action_entropy` | num | nats | Mean per-decision policy entropy during eval. |
| `eval_judge.mean` | num | score [0,1] | Mean persuasiveness of messages delivered in eval episodes. |
| `eval_judge.n` | num | messages | Number of judged messages in the eval batch. |
| `eval_judge.std` | num | score | Spread of eval persuasiveness scores. |
| `eval_judge.min` | num | score [0,1] | Worst eval message score. |
| `eval_judge.max` | num | score [0,1] | Best eval message score. |
| `eval_judge.saturated_frac` | num | fraction [0,1] | Fraction of eval scores >= 0.95. |
| `eval_top_strategies` | list | table | Three most-used strategies in eval, with counts. |

## Paired-evaluation metrics (every `--paired-eval-every` iterations)

| variable | kind | unit / range | description |
|---|---|---|---|
| `paired_eval.episodes` | num | count | Episodes per arm in the paired evaluation. |
| `paired_eval.greedy` | num | reward units | Mean reward of the current policy (greedy) on the shared eval seeds. |
| `paired_eval.random` | num | reward units | Mean reward of a random-action policy on the same seeds. |
| `paired_eval.heuristic` | num | reward units | Mean reward of the scripted heuristic policy on the same seeds. |
| `paired_eval.vs_random.mean` | num | reward units | Mean PER-EPISODE delta (greedy − random) on shared seeds. Pairing removes between-seed variance, so this is the reliable 'is the policy actually better?' number. |
| `paired_eval.vs_random.se` | num | reward units | Standard error of that delta. |
| `paired_eval.vs_random.t` | num | t | t-statistic of the delta (|t| > ~2 is conventionally the significance line). |
| `paired_eval.vs_heuristic.mean` | num | reward units | Mean per-episode delta (greedy − heuristic) on shared seeds. |
| `paired_eval.vs_heuristic.se` | num | reward units | Standard error of that delta. |
| `paired_eval.vs_heuristic.t` | num | t | t-statistic vs the scripted heuristic. |

## Immediate variables — per persuasion decision (`trace/steps.jsonl`)

| variable | kind | unit / range | description |
|---|---|---|---|
| `iteration` | id | count | PPO update index. |
| `episode` | id | count | Rollout episode index within the iteration. |
| `step` | id | count | Policy-decision index within the episode (equals period - 1). |
| `seed` | id | seed | Episode seed — joins the record to the exact deterministic episode. |
| `period` | id | count | 1-based COMPLETED diffusion period (the simulator's internal counter + 1). |
| `persuader` | id | agent id | Agent id of the RL-controlled seed that acted. |
| `action` | id | action index | Sampled compound action index (strategy arm under random_neighbor target selection). |
| `strategy_index` | id | action index | Index of the chosen strategy in the strategy space. |
| `strategy` | cat | spec | The chosen strategy spec, e.g. 'angle=safety, evidence=statistic, address=worker'. |
| `target` | id | agent id | Target agent id this message was aimed at; null when the crowd was exhausted and the persuader idled. |
| `message` | str | text | The rendered message text delivered. |
| `persuasiveness` | num | score [0,1] | IMMEDIATE: the reward model's score for this exact message → target persona. Multiplies the influence probability, so it is the policy's lever on the dynamics. null when idle. (histogram 0…1, 20 bins) |
| `judge_src` | cat | category | How the score was resolved: 'id' = BT identity index (intended), 'text' = content fallback, 'none' = not judged (idle). |
| `goal_align` | num | alignment [0,1] | IMMEDIATE: Track-B alignment a(m) ∈ [0,1] of this message with the persuader's goal (null when not computed / idle). (histogram 0…1, 20 bins) |
| `reward` | num | reward units | IMMEDIATE: the full reward this persuader received for this period (all terms of the decomposition in the module docstring). (histogram -1…3, 32 bins) |
| `n_candidates` | num | agents | Number of eligible targets remaining after this step (the action mask's universe). |
| `mask_open_frac` | num | fraction [0,1] | Fraction of the action space the mask left available. 0 means the crowd was exhausted (the persuader idled rather than acting on a fabricated argmax). (histogram 0…1, 10 bins) |
| `mechanism` | cat | category | Contagion mechanism in force (paper / cascade / exposure). |
| `ts` | ts | epoch seconds | Wall-clock time the step was recorded. |

## Per-period population state (`trace/periods.jsonl`)

| variable | kind | unit / range | description |
|---|---|---|---|
| `iteration` | id | count | PPO update index. |
| `episode` | id | count | Rollout episode index. |
| `step` | id | count | Policy-decision index within the episode. |
| `seed` | id | seed | Episode seed — joins the record to the exact deterministic episode. |
| `period` | id | count | 1-based COMPLETED diffusion period (the simulator's internal counter + 1). |
| `mechanism` | cat | category | Mechanism in force. |
| `mean_stance` | num | stance [-1,1] | Population mean latent stance at the END of this period. (histogram -1…1, 20 bins) |
| `n_adopted` | num | agents | Agents with preference_state == 1 at the end of this period. |
| `n_accept` | num | events | PERSUADER-sourced accept/emerge events this period — exactly the counts that earn the accept-shaping reward term. |
| `n_accept_all` | num | events | All accept/emerge events this period, any source (persuader + organic crowd growth). |
| `n_persuader_events` | num | events | Diffusion events this period whose source was one of the RL persuaders — the part of the contagion the policy can claim. |
| `n_events` | num | events | All diffusion events this period (persuader + crowd). |
| `n_candidates` | num | agents | Eligible targets after the period. |
| `mask_open_frac` | num | fraction [0,1] | Fraction of the action space still available after the period. |
| `period_reward_sum` | num | reward units | Sum of per-persuader rewards this period (the team's total). |
| `period_reward_mean` | num | reward units | Mean per-persuader reward this period. |
| `ts` | ts | epoch seconds | Wall-clock time of the period. |

## Diffusion events (`trace/events.jsonl`)

| variable | kind | unit / range | description |
|---|---|---|---|
| `iteration` | id | count | PPO update index. |
| `episode` | id | count | Rollout episode index. |
| `step` | id | count | Policy-decision index within the episode. |
| `seed` | id | seed | Episode seed — joins the record to the exact deterministic episode. |
| `period` | id | count | The simulator's internal period counter at the hop (0-based). The 1-based label of the same period is `periods.period` for the same (iteration, episode, step) — join on `step`. |
| `source_agent` | id | agent id | Agent whose post caused the exposure. |
| `source_is_persuader` | flag | bool | True when the source is one of the RL-controlled persuaders — attributes each hop to the policy or to organic growth. |
| `target_agent` | id | agent id | Agent that was exposed. |
| `target_stance_before` | num | stance [-1,1] | Target's latent stance before the hop. (histogram -1…1, 20 bins) |
| `target_stance_after` | num | stance [-1,1] | Target's latent stance after the hop. |
| `target_delta_stance` | num | stance delta | IMMEDIATE: stance movement caused by this single hop (negative = toward the persuader's side). Zero by construction in exposure mode, where adoption is a gate decision, not a stance pull. (histogram -1…1, 20 bins) |
| `influence_probability` | num | probability [0,1] | IMMEDIATE: probability the hop fired, from the weighted similarity terms below times message persuasiveness. (histogram 0…1, 20 bins) |
| `behavior` | cat | category | Outcome: 'accept' (paper/exposure gate), 'emerge' (cascade internalisation), or the no-change case. |
| `adopted` | flag | bool | 1 when the target ended this hop with preference_state == 1. |
| `interaction` | cat | category | 'exposure' for every mechanism event (one event per exposure). |
| `mechanism` | cat | category | Mechanism in force. |
| `source_degree_centrality` | num | centrality | Source's degree centrality in the contact network. (histogram 0…1, 10 bins) |
| `source_betweenness_centrality` | num | centrality | Source's betweenness centrality. |
| `target_degree_centrality` | num | centrality | Target's degree centrality. (histogram 0…1, 10 bins) |
| `target_betweenness_centrality` | num | centrality | Target's betweenness centrality. |
| `profile_sim` | num | [0,1] | Profile similarity term of the influence probability. (histogram 0…1, 10 bins) |
| `content_sim` | num | [0,1] | Post-content similarity term. (histogram 0…1, 10 bins) |
| `stance_sim` | num | [0,1] | Stance similarity term. (histogram 0…1, 10 bins) |
| `degree_sim` | num | [0,1] | Degree similarity term. (histogram 0…1, 10 bins) |
| `persuasiveness` | num | [0,1] | The source message's judged persuasiveness that scaled this hop (1.0 for crowd agents in the paper's original dynamics). (histogram 0…1, 20 bins) |
| `goal_align` | num | alignment [0,1] | Track-B alignment of the source message, when computed. |
| `ts` | ts | epoch seconds | Wall-clock time of the event. |

## Episode summaries (`trace/episodes.jsonl`)

| variable | kind | unit / range | description |
|---|---|---|---|
| `iteration` | id | count | PPO update index. |
| `episode` | id | count | Rollout episode index. |
| `seed` | id | seed | Episode seed (deterministic from run seed, iteration, episode). |
| `mechanism` | cat | category | Mechanism in force. |
| `steps` | num | count | Policy decisions in the episode (= periods × persuaders). |
| `episode_reward` | num | reward units | Summed team reward for the episode. (histogram 0…20, 40 bins) |
| `mean_stance` | num | stance [-1,1] | Population mean stance at episode end. |
| `n_adopted` | num | agents | Adopters (preference_state == 1) at episode end. |
| `n_events` | num | events | Total diffusion events in the episode. |
| `n_persuader_events` | num | events | Episode's diffusion events sourced by a persuader. |
| `n_accept` | num | events | PERSUADER-sourced accept/emerge events in the episode. |
| `n_accept_all` | num | events | All accept/emerge events in the episode, any source. |
| `n_idle_actions` | num | decisions | Decisions where the persuader had no eligible target and idled. |
| `n_transitions` | num | transitions | PPO transitions the episode contributed. |
| `ts` | ts | epoch seconds | Wall-clock time of the episode. |

## Strategy value table (`strategy_value.json`)

| variable | kind | unit / range | description |
|---|---|---|---|
| `strategy_index` | id | action index | Index of the strategy arm. |
| `strategy` | cat | spec | The strategy spec string. |
| `count` | num | decisions | Times this arm was chosen so far (whole run). |
| `mean_reward` | num | reward units | Mean reward received when this arm was chosen — the empirical value table the policy is (indirectly) learning. |
| `total_reward` | num | reward units | Summed reward of this arm. |
| `strategy_entropy` | num | nats | Entropy of the cumulative strategy distribution (same quantity as train_strategy_entropy, kept next to the table). |
| `n_decisions` | num | decisions | Total policy decisions recorded. |

## Run status (derived by the monitor)

| variable | kind | unit / range | description |
|---|---|---|---|
| `run_dir` | str | path | Directory being monitored. |
| `mechanism` | cat | category | Mechanism from the run config. |
| `iteration` | num | count | Latest recorded iteration. |
| `iterations_total` | num | count | Configured total iterations (the denominator). |
| `elapsed_s` | num | seconds | Latest recorded elapsed time. |
| `iters_per_min` | num | iters/min | Observed iteration rate over the run so far. |
| `eta_s` | num | seconds | Extrapolated seconds to the configured final iteration at the observed rate. |
| `updated_age_s` | num | seconds | Seconds since training_log.json was last written — the liveness signal (a stale age on a running campaign means it died). |
| `trace_streams` | list | list | Which trace streams exist (empty = this run was started without --trace, so immediate variables were not recorded). |
| `trace_episodes` | num | count | Rollout episodes traced per iteration (0 = all). |
| `model` | str | model id | LLM the run was configured with. |
| `workers` | num | threads | Rollout worker threads. |
| `episodes_per_iteration` | num | episodes | Rollout episodes per update. |
| `agent_num` | num | agents | Crowd size. |
| `n_persuaders` | num | persuaders | RL-controlled seed agents. |
| `topology` | str | network | Contact network topology. |
| `seed` | num | seed | Base random seed. |
| `n_periods` | num | periods | Diffusion periods per episode. |
| `obs_mode` | str | mode | Persuader observation mode. |
| `target_selection` | str | mode | How targets are chosen (learned / random_neighbor). |
| `judge` | str | mode | Reward model in use (absolute / bt). |
| `paired_eval_every` | num | iterations | Iterations between paired evaluations (0 = off). |

## Reward decomposition

Each persuader receives one reward per period, the sum of:

- **stance movement** — `‑Δs` over events the persuader sourced (negative delta = target moved toward the persuader);
- **accept shaping** — `shaping_accept_bonus` per `accept`/`emerge` event it caused;
- **adoption bonus** — `reward_adoption_bonus` per newly adopted target (used by the exposure arm, where `Δs ≡ 0`);
- **goal alignment** — `PERSUASION_GOAL_LAMBDA · a(m)` per delivered message;
- **message cost** — `message_cost` per delivered message (default 0);
- **cascade discount** — `reward_cascade_discount · Δ(population mean stance)`, shared by the team (counterfactual credit for network-wide movement).

`mean_episode_reward` is the summed team-mean of these terms per episode, so its scale is not comparable across arms that enable different terms — compare an arm against its own baselines.
