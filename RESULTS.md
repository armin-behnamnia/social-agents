# RL Persuader — Run Wrap-Up & Results

Date: 2026-10-10
Scope: all PPO runs in `output/ppo/` (logs in `logs/`), covering the
three-stage evolution of the persuader environment:
learned target selection → matrix observation → LLM-judged
persuasiveness gate.

## Setup (common to all runs)

- Env: `ElitePersuaderEnv` — 30 agents, 10 periods per episode,
  Watts–Strogatz network, initial stances ≈ +0.65 (opposing), topic
  "the policy". Reward = target stance shift + acceptance shaping +
  0.5 × population cascade term.
- Action space: 5400 = 30 candidate targets × 180 persuasion
  strategies; messages rendered by the frozen LLM (cache-warmed),
  delivered to the chosen target only.
- PPO (GPU, `torch_trainer.py`): 200 iterations × 8 episodes/iter
  (mock runs shorter), 8 rollout workers, hidden 128, γ=0.99,
  GAE λ=0.95, clip 0.2, actor/critic lr 3e-4/1e-3, entropy coef 0.01.
- Eval: every 20 iters, greedy policy, 10 episodes; final comparison
  vs random and highest-degree heuristics (10 episodes each).

## Runs

| Run | Date | Duration | Env version | Best greedy eval | Final greedy | Random | Heuristic | Δ vs random | Δ vs heuristic |
|---|---|---|---|---|---|---|---|---|---|
| mock_validation | Oct 8 | minutes | mock LLM | — | 3.435 | 3.697 | 3.481 | −0.262 | −0.047 |
| mock_run (120 it) | Oct 8 | ~30 min | mock LLM | — | 3.451 | 3.697 | 3.481 | −0.246 | −0.031 |
| **live_run1** | Oct 9 | 101 min | learned target, 13-d summary obs | 2.57 @60 | **2.597** | 2.789 | 2.639 | **−0.191 (−6.9%)** | −0.041 |
| **live_run2_matrix** | Oct 10 | 106 min | + (30×14) per-agent matrix obs | 3.11 @180 | **2.683** | 2.789 | 2.639 | −0.105 (−3.8%) | **+0.045 (+1.7%)** |
| **live_run3_persuasive** | Oct 10 | 66 min | + LLM-judged persuasiveness gate | 2.29 @160 | **2.245** | 1.981 | 1.981 | **+0.264 (+13.3%)** | **+0.264 (+13.3%)** |

All artifacts: `output/ppo/<run>/{policy.pt, training_log.json, results.json}`;
summary figure: `output/ppo/runs_summary.png`.

## What changed between runs (why)

1. **live_run1 — learned compound action.** The policy picks
   (target node, strategy) from an action mask, but observes only a
   13-d mean-pooled summary of candidate targets. That summary
   cannot distinguish *which* node is selected, so the target half of
   the action space is unlearnable. Confirmed post-hoc: policy
   entropy collapsed from 6.88 → 3.04 (premature convergence) and the
   final greedy policy lost to both baselines.
2. **live_run2 — matrix observation.** Observation becomes a
   (30 agents × 14 features) matrix (profile-embedding pool,
   centralities, exposure, stance proxy, is-candidate, is-adopted), so
   both target and strategy choices are conditioned on per-node
   features. Entropy stayed healthy (6.78 → 5.85); the policy edged
   out the heuristic (+1.7%) but still lost to random (−3.8%).
3. **live_run3 — persuasiveness gate.** `judge_persuasiveness()`
   (LLM, temp 0) scores how well the rendered message fits the target
   persona and multiplies it into the stance-influence probability —
   *only* for the learned persuader's messages; crowd agents keep the
   paper's original dynamics. This finally ties message content (hence
   strategy choice) to reward: **first run where the learned policy
   beats both baselines, +13.3% over random and heuristic.**

Note the baselines dropped from 2.79 to 1.98 in run3 because the gate
damps diffusion for *any* policy (adoptions fell from ~4.5 to ~2.3 per
episode, mean final stance 0.45 → 0.52). Absolute rewards are not
comparable across runs — judge each run against its own baselines.

## Conclusions

- **Headline: 200 PPO iterations produce no measurable training-reward
  improvement.** The best configuration (run3) shows a small, noisy
  (+13%) separation from baselines at *greedy eval* time, not a rising
  learning curve. Treat this campaign as infrastructure validation
  plus a weak positive signal, not as a demonstrated RL win.
- **RL persuasion beats hand-crafted baselines only when the reward
  actually sees strategy quality.** Learned targeting + a full
  per-node observation + an LLM persuasiveness gate were all
  necessary; the incremental fixes in runs 1–2 were not sufficient on
  their own.
- **Learned end-to-end pipeline works:** 5400-action space, action
  masking, cached LLM rendering (5400 message cache warmed in ~3 s),
  8-worker parallel rollouts, GPU PPO — 66–106 min per 200-iteration
  run with no failed iterations.
- **Mock mode validates plumbing, not learning.** Both mock runs
  plateau below baselines, as expected: the deterministic mock LLM
  carries no real persuasion signal.
- **Caveats:**
  - 10-episode evals are noisy (per-episode σ ≈ 0.5 → SE ≈ 0.17);
    run3's +0.264 gap (~1.5 SE) is promising but not yet conclusive.
    Greedy eval swings (e.g. run2 @180 = 3.11 vs @200 = 2.39) confirm
    the noise level.
  - Single 30-agent-scale setting, one topic, one network topology.
  - **No within-run training-reward improvement in any run** (linear
    slopes over 200 iters: run1 −0.0004, run2 +0.0003, run3 +0.0006
    reward/iter — i.e., flat). The run3 gain lives in the *greedy
    eval*: +0.0024/iter slope, last-3 evals ≈ 2.21 vs 1.98 baseline,
    but with per-eval SE ≈ 0.17 this is ~1–1.5 SE — a consistent
    trend, not a statistically firm win. Root causes: only 80
    transitions per PPO update against a 5400-action space, and reward
    dominated by stochastic crowd dynamics the persuader matters
    little to.

## Known operational issues (non-blocking)

- Intermittent `.llm_cache/<hash>.tmp` rename warnings from concurrent
  cache writers retrying the same entry; retried successfully every
  time. Worth a lock or best-effort rename when running >4 workers.
- live runs 2/3 reused the warm LLM cache (3 s warmup vs 368 s on the
  first live run) — cache reuse across runs is working as designed.

## Next steps (not done here)

1. Re-run live_run3 config with more eval episodes (≥30) or multiple
   training seeds to firm up the +13% gap.
2. Longer training (500+ iters) now that the signal exists; check
   whether target-selection learning emerges before entropy collapse.
3. Ablate the persuasiveness gate (on/off) at fixed seed to quantify
   its causal contribution.
4. Scale to larger networks / other topologies / other topics for
   robustness claims.

Verification of this report's numbers: `results.json` and
`training_log.json` under each run dir; code state passes the full
suite (74 tests, `pytest code/tests -q`).

---

# Campaign 2 — 50 agents, 5 learning persuaders (shared policy)

Date: 2026-10-11. Setup: 5 RL persuaders (seed agents 0–4, one shared
PPO policy, per-persuader reward attribution, joint per-period
actions) among 45 frozen crowd agents; nearly-sparse Erdős–Rényi
network (avg degree 4, connectivity-guarded); 10 periods/episode;
GPU PPO, parallel rollouts.

## Run 5p/live_run1 — global compound targeting (INTERIM, interrupted)

250 iters × 16 eps planned, infrastructure kill at iter 25/250
(no code error; checkpoint `policy.pt` @ iter 25). Formal wrap-up
evaluation (20 episodes each, `code/eval_wrapup.py`):

| Policy | Reward | vs random |
|---|---|---|
| **Random baseline** | **10.004** | — |
| Heuristic | 6.761 | −3.24 |
| Learned greedy @ iter 25 | 7.562 | **−2.44** |

**Finding (qualitative, visualized):** the greedy policy mode-collapsed
— all five persuaders targeted the *same node with the same strategy*
every period. Mechanism: global candidate space + shared policy +
near-identical per-actor observations (single is-acting bit) ⇒
argmax picks the same action for all actors; and the reward even
pays such stacking short-term (per-source credited events). At real
evaluation scale it loses to uniform-random targeting by −24%.
Replays: `output/ppo5p/live_run1/viz/trained_at25/` (trained) and
`.../viz/random/` (random baseline): trained achieved 21/50 adopted
vs 12/50 on the same network seed (concentrated multi-hit does
adopt single targets) yet still loses on the reward objective.

**Design response (spec-mandated):** each persuader now targets a
*randomly drawn eligible own-neighbor* per period; the policy chooses
only the strategy (`Discrete(180)`), observing a 26-d vector
(persuader features ‖ target features, `obs_mode="pair"`).
Fixes the collapse structurally (disjoint neighborhoods), shrinks the
action space 50×, and matches the intended interaction protocol.

## Run ppo5p_rn/live_run1 — random-neighbor target + learned strategy

IN PROGRESS at report time (230/300 iterations; 8 eps/iter; 8 workers;
render+judge caches warm). Greedy-eval trajectory (8 episodes each):

| iter | 20 | 60 | 100 | 140 | 180 | 220 |
|---|---|---|---|---|---|---|
| eval | 8.63 | 10.42 | 10.49 | 10.14 | **11.06** | 10.45 |

First-3 evals mean 8.94 → last-3 mean 10.57 (+18%): **the first rising
evaluation curve across both campaigns** (all campaign-1 runs were
flat). Training reward ≈ 9.9–10.4/episode; adoptions ≈ 14–16 of 50;
mean final stance ≈ 0.19 (from +0.65). Value loss settled 3.8 → ~0.3;
no divergence, healthy entropy.

Artifacts: `output/ppo5p_rn/live_run1/training_progress.png`
(live-parsed 4-panel chart), `viz/trained_at210/` and `viz/random/`
(GIF + interactive HTML action/event log + trajectory.json).
Formal baselines (results.json) are written automatically at iter 300;
the final section will be updated on completion.

Episode replays (same network + persona seed, live LLM, 10 periods):

| Replay | Total reward | Adopted | Mean stance 0.485 -> |
|---|---|---|---|
| Trained @ iter ~210 | **9.79** | **21/50** | **+0.101** |
| Random strategy | 8.91 | 17/50 | +0.181 |

The replay visualizes the redesigned interaction protocol working as
specified: each persuader hits a *distinct own-neighbor* every period
(no more stacked targeting). One caveat visible in the frames: the
learned policy has converged onto a single strategy family
(ethos/confident) for (almost) all pairs — strategy-level convergence
without target stacking. Single-episode numbers are illustrative; the
formal comparison is the 8+ episode eval/baselines at completion.

Artifacts (rewatch): `output/ppo5p_rn/live_run1/viz/trained_at210/`
and `.../viz/random/` — `episode.html` is self-contained.
