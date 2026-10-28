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

## Run ppo5p_rn/live_run1 — random-neighbor target + learned strategy (COMPLETE)

Finished 2026-10-12 (300/300 iterations; 8 eps/iter; 8 workers;
render+judge caches warm; resumed from a checkpoint at iter 260).
Greedy-eval trajectory (10 episodes each, full run):

| iter | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 | 220 | 240 | 260 | 280 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval | 8.63 | 8.55 | 10.42 | 9.49 | 10.49 | 10.66 | 10.14 | 9.81 | **11.06** | 10.20 | 10.45 | 10.55 | 10.43 | **11.71** | 10.19 |

Early rise is real (first-3 evals mean 9.20 → last-3 mean 10.78, +17%),
and it **does survive rigorous evaluation**. The auto-written
`results.json` uses *unpaired* seeds (learned 900k, random 700k,
heuristic 600k) and only 10 episodes; per-episode σ is ≈1.5, so those
baselines are badly noise-limited and are **superseded** by the paired
evaluation below.

### Paired evaluation (the verdict) — `results_final.json`

`code/eval_final.py`, 50 seeds shared by all three policies
(`--episodes 50 --workers 8`, observer inference off to match training):

| Policy | Reward (50 eps) | SE |
|---|---|---|
| Random | 10.038 | 0.204 |
| **Learned greedy** | **10.625** | 0.221 |
| Heuristic | 10.520 | 0.217 |

| Paired comparison | Δ | SE | t | wins |
|---|---|---|---|---|
| learned − random | **+0.587 (+5.8%)** | 0.122 | **4.81** | 39/50 |
| learned − heuristic | +0.105 (+1.0%) | 0.152 | 0.69 | 30/50 |

**The learned policy significantly beats uniform-random strategy
(+5.8%, t = 4.8, winning 39/50 paired seeds) and is statistically tied
with the fixed-mid-strategy heuristic (+1.0%, t = 0.69).** This is the
first statistically firm positive result across both campaigns — but a
modest one: RL adds real signal over chance and does **not** clearly
exceed a well-chosen constant strategy.

(The unpaired `results.json` reported learned *losing* to the heuristic
by −5.1% and barely beating random. That was an artifact of σ ≈ 1.5 with
only 10 independent-seed episodes; the paired design removes the
between-seed variance and is the correct comparison.)

**Interpretation.** In `random_neighbor` mode the only learned degree of
freedom is the strategy (on a randomly drawn neighbor), and the policy
converged onto a single strategy family — so the achievable ceiling is
"a good constant," exactly what the heuristic is. Beating random by a
significant margin shows the strategy signal is real and learnable;
failing to beat the heuristic shows PPO found *a* good constant, not the
best one. Consistent with campaign 1: RL's edge tracks how strongly the
reward couples to a genuinely policy-controlled decision.

**Revised campaign-2 headline:** the random-neighbor protocol fixed the
mode-collapse **and** produced the first significant learning signal
(+5.8% over random, paired), but because target selection was removed the
learned policy only matches — does not exceed — the best fixed strategy.
Next lever: restore a policy-controlled decision the reward tracks
strongly, with disjointness constraints to avoid the earlier collapse.

Artifacts: `output/ppo5p_rn/live_run1/{policy.pt,training_log.json,results.json,results_final.json}`
(the two result files written at completion / by the paired re-eval),
`training_progress.png`, `viz/trained_at210/` and `viz/random/`
(GIF + interactive HTML action/event log + trajectory.json).

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
without target stacking. This replay happens to favor the trained
policy, but single-episode numbers are illustrative only: the paired
50-episode evaluation (above) is the formal comparison — the learned
policy significantly beats random and statistically ties the heuristic.

Artifacts (rewatch): `output/ppo5p_rn/live_run1/viz/trained_at210/`
and `.../viz/random/` — `episode.html` is self-contained.

---

# Campaign 3 — repaired readout (Bradley–Terry) + redesigned strategy space

Date: 2026-10-14. The campaign-2 ceiling analysis pinned the binding
defect on the **reward channel** (#2) — the frozen LLM judge collapsed
180 distinct messages onto 4–6 scores, so changing the action barely
changed the reward. Campaign 3 attacks that directly and redefines the
action.

## 1. New action space (v2) — `code/strategy_v2.py`

Action = **angle** (6: cost / safety / fairness / freedom / community /
effectiveness) × **evidence** (4: statistic / expert / anecdote / peers)
× **address** (3: taxpayer / parent / worker) = **72 arms**, each with a
checkable render constraint (digit ⇒ statistic, named source ⇒ expert,
first-person detail ⇒ anecdote). Selected with
`PERSUASION_STRATEGY_SPACE=v2` (v1's 180-arm grid stays the default).
Pilot realisability (nearest-neighbour purity, chance in parens):
angle **0.85** (0.17), address **0.77** (0.33), evidence 0.27 (0.25) —
the concrete slots are adopted by the generator; the old `emphasis` had
sat at chance.

## 2. Bradley–Terry pairwise judge — `code/bt_judge.py`

3,600 (post, target) arms played off head-to-head (31,084 comparisons),
Bradley–Terry fit, scores shifted to the absolute judge's mean. The
readout repair is large and reaches all the way to the network reward:

| Readout | Score SD | Distinct scores | Modal bucket share | Landscape spread | Spread ÷ SE |
|---|---|---|---|---|---|
| absolute 0–10 judge | 0.072 | 4–6 | 54% @ 0.90 | ~0.90 | ≈ 1× |
| **Bradley–Terry** | **0.233** | **691** | **32% @ 0.98** | **p10–p90 7.59** | **≈ 18×** |

Variance decomposition of the BT score: **49.5%** action main effect,
**0.0%** target main effect (comparisons are within-target by
construction), **50.5%** target × action interaction.

Network-reward landscape (all five persuaders fixed to one arm, 8 paired
seeds, `report/strategy_value_bt.json`): mean **7.62**, SD **3.04**,
range **10.58**, per-episode SE **0.43**. Best single constant
**12.80** = `safety / anecdote / worker`; worst **2.22** =
`freedom / expert / parent`. Structure is clean and interpretable —
**every top-5 arm addresses the worker; every bottom-5 is the freedom
angle**.

**Caveat (the open question).** The *per-target* oracle only beats the
best constant by **+0.006** in score, because the best constant already
sits near the 1.0 ceiling. So BT now ranks **actions** sharply but still
barely rewards **matching the target** — target × action conditioning of
the *readout* remains weak. Campaign 3 tests whether the reward-scale
signal (gravity ≈ 18× SE) is enough for PPO to climb.

## 3. Training run — `output/ppo_v2_bt/live_run2` (COMPLETE)

v2 space + BT judge, 300 iters × 32 eps = **9,600 episodes**, 160
rollout workers, paired eval every 20.

**Compute note (why not the planned 300 × 320).** Live episodes are
dominated by crowd-agent generation: `generate_response_message` embeds
`delta_stance` at 4 decimals, so it is uncacheable across episode seeds
and every episode fires ~40 fresh LLM calls. Measured throughput on this
box: **29.7 s/episode serial, 2.27 s/ep @ 48 workers, 1.59 s/ep @ 192**
(the remote server saturates at ~26 req/s). The first launch at
96,000 episodes projected to **~60–70 h** and was killed; the shipped
config (9,600) fits in ~4–5 h at >300 episodes/iteration-equivalent.
Cache-warming via `delta` quantisation is the obvious next speedup if a
much larger budget is wanted.

### Final verdict

Run completed **300/300** iterations (launched 2026-10-14, resumed 2026-10-15
from iter 20, finished 2026-10-16 ≈01:09; ~17 h wall at ~199 s/iter).
`results.json`:

| | reward |
|---|---|
| learned policy (greedy) | **13.014** |
| random baseline | 8.604 |
| "heuristic" baseline | 3.180 |
| improvement over random | **+4.410** |
| improvement over "heuristic" | +9.833 |

Paired eval @ 300 (50 shared seeds): greedy **13.377** vs random **8.929**
(**+4.448, t=14.35**) vs "heuristic" 2.930 (+10.447).

Across all 15 paired evals (every 20 iters) the learned policy stayed above
random from iteration 20 onward, flat and stable late: 13.31 → 13.34 →
13.34 → 13.46 → 13.43 → **13.47** (peak, iter 260) → 13.42 → 13.38.

- **vs random:** +4.45, t ≈ 14–15 → firmly beats random.
- **vs best constant (12.80):** greedy 13.38 → **+0.58**, so it does beat the
  best single arm — but the margin is modest and eval-seed dependent.
- The "heuristic" baseline (2.93) is **not a tuned constant** — it is v2 arm
  #36 (`freedom/statistic/taxpayer`, a bottom arm). The meaningful bars are
  random and best-constant.

Learned behaviour: converges on `evidence=anecdote` + `address=worker`,
cycling the angle between `effectiveness` (#68) and `fairness` (#32) — the same
region the independent landscape probe ranked best. Final strategy entropy
**0.39** (a couple of arms, not one).

**Caveat.** Judge saturation at the end is `~0.995` (90–98% in the top
bucket): the policy rides the readout ceiling. The +4.45 is real against
*this* readout, but the readout scores persona-fit/persuasiveness, not the
message's direction toward the goal. That is the next lever — readout
hardening first, then goal-direction injection (see `IMPLEMENTATION_PLAN.md`,
Tracks C and B).

The report mirror of this section lives in `report/index.html` → *campaign 3
verdict* (under the Bradley–Terry appendix).

---

# Campaign 4 — constant calibration (Track A)

Date: 2026-10-16. Track A makes the dynamics constants defensible rather
than inherited: the four α's (`0.25` each) and the stance step β (`0.4`)
were placeholders. The paper defines `p_ij ← InfluenceProbability(...)` and
`Δs_j ← StateTransition(j, m)` but **never expands either** (verified by
extracting `Agentic_Information_diffusion.pdf`), so both must be
*justified*, not merely *chosen*.

Two scripts, each reproducible (data hash + script version recorded):

## A1. `code/fit_alphas.py` — the α's are not identifiable

Runs frozen crowd simulations (real `KubeAI/jina-v5-text` embeddings, 200
episodes × 12 periods → **16,995 exposures**, accept rate 0.744) and logs
the four components per exposure (the instrumentation now carries
`profile_sim / content_sim / stance_sim / degree_sim` on every
`diffusion_event`).

| view | result |
|---|---|
| linear recovery (regress logged `p` on components, α≥0) | **[0.252, 0.244, 0.250, 0.252]**, mean\|resid\| **8.2e-5** |
| logistic fit `P(accept)=σ(Σα·sim+b)`, α≥0 | [0.38, 0.15, 1.16, 0.00], b −0.12, logloss 0.561 |
| design-matrix effective rank | **2.28 / 4**, cond 15.1 |
| per-feature std | profile 0.044, content 0.025, stance 0.221, degree **0.015** |

The linear recovery is exact — it just confirms the instrumentation (the
accept gate *is* the weighted sum). The decisive number is the **effective
rank 2.28/4**: profile- and degree-similarity are near-constant within an
episode, so the four α's are **not separately identifiable from behavior**.
The logistic coefficients differ wildly from 0.25 while the design is
collinear — a fitted α would be noise.

**Decision: keep the equal-weight prior (0.25).** It is now documented as an
*unidentifiable prior*, not an arbitrary guess; and the influence *scale*
(not the shape) is what the dynamics can constrain — calibrated jointly with
β below.

## A2. `code/calibrate_beta.py` — β alone cannot match the paper

Sweeps the effective strength **ρ = β·κ** (κ scales the four α's jointly;
β and κ enter the dynamics only through `Δs = p·β·(s_src−s)` with
`p = κ·Σ0.25·sim`, so the dynamics pin down ρ, not β alone). Targets,
confirmed verbatim from the paper:

- early mean ΔS (t ∈ 1..3) ≈ **−0.168**
- late mean ΔS (t ≥ 16) ≈ **−0.107**
- plateau S ≈ **−0.55** (from +0.65 over t ∈ [0,19])

Fine sweep (20 episodes × 24 periods, real embeddings):

| β | κ | ρ | early ΔS | late ΔS | plateau S | score |
|---|---|---|---|---|---|---|
| 0.90 | 2 | 1.80 | −0.161 | **0.000** | −0.457 | 1.030 |
| **0.90** | **3** | **2.70** | **−0.163** | **0.000** | **−0.466** | **1.024** |
| 0.95 | 2 | 1.90 | −0.192 | 0.000 | −0.709 | 1.104 |
| 1.00 | 2 | 2.00 | −0.227 | 0.000 | −1.000 | 1.791 |

**Findings.**

1. **Early ΔS and the plateau are jointly matchable** at ρ ≈ 2
   (β≈0.9, κ≈2–3): early −0.16 vs −0.168, plateau −0.46 to −0.55.
2. **Late ΔS is structurally wrong**: every setting gives late ΔS ≈ **0**,
   against the paper's −0.107. Our transition reaches a *static* fixed
   point (once the network adopts, `s_src − s → 0` and updates stop), where
   the paper maintains *sustained, decaying* late updates ("cognitive
   saturation"). Pushing ρ higher doesn't fix the shape — it triggers a
   sharp all-or-nothing cascade to S = −1 (ρ ≥ 2.0), with no stable
   intermediate consensus.
3. Therefore **β is not the binding knob**: the residual is the
   *functional form* of the transition — exactly the text-independent
   `pull-to-source` that **Track B** replaces with a goal-direction term.

**Decision: keep β = 0.4** (v1/v2 comparability), now annotated in
`code/config.py` with the ρ evidence and the structural caveat. The
dynamics mismatch is handed to Track B; re-tuning β would be fitting a
shape problem with a scale knob.

## A3. Status

- α and β are now *justified* (as an unidentifiable prior and a
  non-binding default) with reproducible evidence
  (`output/calibration/alphas.json`, `beta.json`, `beta_fine.json`,
  `beta_sweep.png`, `beta_fine.png`).
- Defaults unchanged, so campaign-1/2/3 runs stay comparable.
- 4 new tests (`influence_terms`, event components, `nnls`,
  `fit_logistic_nn`); **83/83 pass**.

---

# Campaign 5 — goal-direction injection (Track B)

Date: 2026-10-16. Campaign 3's caveat and Track A's negative result both
point here: the reward and the stance delta were **text-direction-blind**.
The readout scores persona-fit ("persuasive *for this reader*"), and the
transition `Δs = p·β·(s_src − s)` pulls a target toward the source's stance
regardless of what the post argues. So a persuasive-but-off-goal post was
rewarded identically to an on-goal one — and Track A showed the resulting
dynamics cannot match the paper's smooth late-phase updates.

## B1. The goal is now a proposition

`code/config.py` / `code/scenario.py` define:

- `goal_proposition` = *"This policy is harmful and should be rejected."*
- `goal_anti_proposition` = *"This policy is beneficial and should be
  supported."*

derived from the seed post (`code/simulation.py`), replacing the bare
placeholder `topic="the policy"`.

## B2. `code/goal_align.py` — measuring `a(m) ∈ [0,1]`

Three interchangeable backends:

| backend | method | directional sanity (anti / neutral / pro) |
|---|---|---|
| **semdiff** (default) | ½·(cos(m,G_goal) − cos(m,G_anti) + 1), reuses embeddings | 0.665 / 0.513 / 0.340 |
| judge | cached LLM goal-axis prompt, −1..+1 | 1.00 / 0.50 / −1.00 (on short posts) |
| nli | local MNLI entailment | falls back to semdiff if `transformers` absent |

**Caveats (measured).** `semdiff` is deterministic and free but *weakly*
separates (anti 0.67 vs pro 0.34 — monotone but compressed). The `judge`
backend separates sharply on unambiguous posts but is **flaky on mixed
sentences** (a longer pro post with a hedging clause scored 0.95 instead of
~0.0). Default is `semdiff`.

## B3. Injection into the stance delta

`code/agent.py::update_agent_stance` now accepts `goal_align` and, when
Track B is on, replaces the source stance with a goal-projected one:

```
s_eff = s_src·(2a(m) − 1)          # a=1 → s_src, a=0.5 → 0, a=0 → −s_src
s'    = clip(s + p·β·(s_eff − s))  # an off-goal post moves the target AWAY
```

Gated by `PERSUASION_GOAL_DELTA=1` / `--goal-delta` (default off, so
v1/v2/v3 stay comparable); `--align {none,semdiff,nli,judge}` selects the
backend. The rendered persuader post is scored in `rl_env` (`message_goal_align`),
logged per decision (`goal_align` on the action record and the diffusion
event), and surfaced as `train_goal_align` in the trainer.

## B4. Validation — `code/validate_goal.py`

40 episodes × 6 periods, `semdiff`, injection on (626 scored decisions):

| check | result |
|---|---|
| directional sanity | **anti 0.665 > neutral 0.513 > pro 0.340** ✓ |
| corr( a(m), realized −Δs ) | **+0.177** |
| mean −Δs, on-goal (a≥0.5) | **+0.0373** |
| mean −Δs, off-goal (a<0.5) | +0.0253 |

On-goal posts are rewarded ≈1.5× more than off-goal ones, and the
off-goal advantage that existed under the old text-blind transition is
attenuated. Directional sanity holds; the correlation is modest because
crowd messages cluster near a≈0.5.

## B5. Status

- [x] Goal is a named proposition in config.
- [x] `a(m)` backends implemented, cached, offline-safe; logged per decision.
- [x] Injection behind `--goal-delta` / `PERSUASION_GOAL_DELTA`.
- [x] Offline validation (directional sanity + `a`↔`−Δs`).
- [ ] **A/B** (current vs goal-delta), 50-seed paired, 300×32 — deferred to
  the next full campaign (item 16), which now also benefits from the
  Track A finding that the transition *form* (not β) was the dynamics
  bottleneck.

6 new tests (`TestGoalDirection`: transition direction, semdiff formula,
none-off, env injection+logging); **87/87 pass**.

---

# Campaign 6 — goal-direction A/B and the `a(m)` backend (Track B5)

Date: 2026-10-16. This closes Track B's open A/B item. Matched arms run
against the no-goal baseline (`output/ppo_v2_bt/live_run2`) under an
**identical** protocol — v2 space, BT readout, 50 agents / 5 persuaders /
`erdos_renyi` / 10 periods, `--obs-mode pair`, `--target-selection
random_neighbor`, seed 0, paired eval on 50 shared seeds every 20 iters.
The only difference is the goal-alignment backend:

| arm | flags | dir |
|---|---|---|
| baseline | (none) | `output/ppo_v2_bt/live_run2` (300 iters) |
| goal (semdiff) | `--goal-delta --align semdiff` | `output/ppo_v2_goal/live_goal` (60 iters) |
| goal (judge) | `--goal-delta --align judge` | `output/ppo_v2_goal/live_goal_judge` (60 iters) |

## C6.1 The finding: the backend, not the injection, is the crux

Campaign 5 validated `a(m)` on **hand-built** posts — and on those, every
backend passes directional sanity. But the messages the policy actually
renders are *strategy-templated*, and the embedding backends are
**direction-blind** on them. Scoring the 12 real v2 strategy messages
(`code/goal_backend_check.py`) `/report/goal_backends.json`):

| backend | directional sanity (short anti / pro) | mean `a(m)` on 12 **real** messages |
|---|---|---|
| semdiff (proposition) | 0.613 / 0.312 | **0.508** (neutral) |
| semdiff (seed-anchored) | 0.679 / 0.321 | **0.519** (neutral) |
| **judge** (LLM) | **1.00 / 0.00** | **0.975** (on-goal) |

The rendered messages are unambiguously anti-policy ("this policy throws
[safety] into chaos", "won't hit its own target—only 12%…"), yet both
embedding variants read them at ≈0.51 — the cosine does not capture the
stance/negation of the template. The judge reads the same corpus at 0.975.

## C6.2 Consequence: the semdiff arm collapses

With `s_eff = s_src·(2a−1)`, `a≈0.5` zeroes the pull. So the semdiff arm's
diffusion **stalls** (tail means, iters 30+):

| arm | reward | final_stance | adopted | `train_goal_align` |
|---|---|---|---|---|
| baseline | 12.75 | 0.072 | 22.77 | — |
| goal (semdiff) | **5.15** | **0.414** | **5.02** | 0.515 (flat, all 60 iters) |
| goal (judge) | **11.30** | **0.134** | **18.63** | **0.966** |

Paired eval (greedy, 50 shared seeds) — the semdiff arm still *learns*
(`d>0`) but from a neutralised environment; the judge arm tracks the
baseline to within 0.02 at iter 60:

| iter | baseline | goal (semdiff) | goal (judge) |
|---|---|---|---|
| 20 | 13.432 | 5.473 (d+1.59) | 12.237 (d+3.75) |
| 40 | 13.367 | 5.505 (d+1.62) | 12.319 (d+3.84) |
| 60 | 12.487 | 5.552 (d+1.67) | **12.467** (d+3.98) |

The policy earns high BT persuasion (`judge~0.99 sat=0.97`) yet the network
barely moves: the reward (persona-fit) pays for posts the direction-blind
transition then neuters. `train_goal_align` stays pinned at 0.514 for all
60 iterations — the policy has no incentive to raise `a(m)`, because the
reward never sees it.

## C6.3 Two fixes that came out of this

1. **Persuader-only injection.** Campaign 5 also scored every *crowd* post
   (lazily, on top of the persuader's), which scored thousands of extra
   LLM/embedding calls and turned the crowd's own physics direction-blind —
   an environment-wide confound. The injection now applies **only to the RL
   persuader's messages** (`message_goal_align`, set in `rl_env`);
   the crowd keeps the paper's original transition. `agent.py`,
   `rl_env.py`.
2. **Backend choice is load-bearing.** The judge backend is required for
   `a(m)` to mean anything on templated messages; `semdiff` may stay the
   free default for smoke/offline work but must not be trusted for a real
   goal-direction campaign.

## C6.4 Judge arm — the corrected result

The corrected arm (`--align judge`, persuader-only, 60 iters) confirms the
diagnosis. It holds `train_goal_align = 0.966` (vs semdiff's flat 0.514) and
tracks the no-goal baseline almost exactly — tail reward **11.30 vs 12.75**,
final stance **0.134 vs 0.072**, adopted **18.6 vs 22.8**, and paired-eval
greedy **12.467 vs 12.487** at iter 60 (Δ 0.02). So with a backend that
actually resolves message direction, the injection is **not** harmful: the
policy still learns (`d+3.98`) and the diffusion is preserved; the only
messages attenuated are genuinely off-goal ones.

(The arm was interrupted twice by external infrastructure — an unexplained
process kill at iter 41 and a 503 from the shared embedding endpoint at
iter 55 — and resumed each time from its checkpoint; the 60-iteration curve
in `report/ab_goal.json` is stitched from the per-segment records.)

**Bottom line.** Goal-direction is measurable and safe *iff the `a(m)`
backend is direction-aware*. `semdiff` is fit for smoke tests only; a real
campaign must use `judge` (or a calibrated equivalent). The next full
300×32 run (item 16) should therefore run `--goal-delta --align judge`.

---

# D. Simulation visualization — LLMxIO 3D dashboard

We mapped a recorded episode onto the
[`SIGNALS-Lab/LLMxIO_Dashboard`](https://github.com/SIGNALS-Lab/LLMxIO_Dashboard)
schema so the persuasion dynamics can be explored as an interactive 3D
network (`code/export_dashboard.py`, `code/dashboard/`).

**Mapping.** Agent → persona; the interaction graph → `follow_log` (all
edges at t0, so the whole network is visible immediately); seed/persuader
messages → `influence` tweets; each **adoption** → the adopter *retweets*
the message that reached it (a propagation/cascade edge); each
non-adopting **accept** → the target *comments* on it (a reply edge).
Persuaders are classified as **IO agents** (`io<id>`), the rest as organic
users (`u<id>`). The campaign hashtag rides only on influence posts and
their re-shares, so hashtag-adoption analytics track real adoption.

**Exported runs** — both are episodes of **already-run** campaign-6
experiments (matched protocol: v2 space, 50 agents, 5 persuaders,
erdos_renyi, 10 periods, `obs_mode=pair`, `random_neighbor`, seed 900000):

| run | experiment | adopted | mean stance | reward | edges (F/Rt/Rp) |
|-----|------------|---------|-------------|--------|-----------------|
| 8 | BT baseline (`ppo_v2_bt/live_run2`) | 16/50 | 0.485 → 0.181 | 8.61 | 180 / 12 / 70 |
| 9 | judge + goal-δ (`ppo_v2_goal/live_goal_judge`) | 19/50 | 0.485 → **0.142** | 9.02 | 180 / 19 / 80 |

The trained arm flips ~19% more agents than the baseline and drives the
mean stance lower — visible directly as denser re-share cascades in the
dashboard's timeline. Served on **http://localhost:3009** (port set in
`code/dashboard/setup_dashboard.sh`); the *Simulation visualization*
section of `report/index_2.html` summarises the same runs.

**Validation.** Rather than eyeball it, we bundled the dashboard's *real*
TypeScript pipeline (`loadSimulationData` → `processSimulationData` →
`precomputeAllMetrics`) with esbuild and ran it in Node against the
exported CSVs: both runs load 50 agents, build the expected edges, and
compute all 8 metric blocks without error; `next build` type-checks the
integration patch. Raw data is committed under `report/dashboard/`.

## D.1 Node descriptions now show the simulation, not just the tweets

Clicking a node used to describe tweet traffic only (followers, tweets,
replies): the dashboard's schema has no opinion variables, so nothing on
screen said what the simulation is *about* — above all an agent's current
stance. Two changes fix that (both additive; runs without the new file are
unaffected):

1. **`code/export_dashboard.py`** also writes
   `public/data/v1/<n>/agent_variables_50_repub_<n>.json`: one record per
   agent holding a *time series per variable* (index `i` = the population
   snapshot after period `i`, i.e. dashboard iteration `i`, so
   `series[currentIteration]` is what the timeline is showing) plus the log
   of influences that reached it. Six series — `stance`, `adopted`,
   `exposure_cum`, `influence_cum` (signed), `from_persuader_cum`,
   `adopt_events_cum` — and five statics (`role`, `degree`,
   `initial_stance`, `leaning`, `profile`). **Every variable carries a
   non-empty `description`**, which is what the panel's hover titles and
   the panel's own footer refer to.
2. **`code/dashboard/llmxio_integration.patch`** (pinned to dashboard
   `9f913ee`) patches `AgentInspector` to fetch and render it: the
   **current stance** large and colour-coded (red = supports the policy,
   blue = aligned with the persuaders — same convention as
   `code/visualize.py`), with a sparkline of the whole trajectory, a
   marker at the iteration on screen, and `Δ` since `t0`; then the other
   variables at that iteration with label/unit; the agent's static
   context; and the received-influence log (who, when, the stance shift,
   the behaviour, whether it came from a persuader). A missing file
   **404s and the panel does not render**, so the patch is inert on runs
   it does not apply to.

**Verified rather than asserted.** `code/tests/test_export_dashboard.py`
(27 tests) pins the accounting on a synthetic trajectory: series lengths,
cumulative counters summing to the event count (nothing dropped or
double-counted), sticky adoption, `influence_cum` as the signed sum of
deltas, no self-exposure, `from_persuader_cum` counting only persuader
sources, and a description on every variable. On the real run 10 episode
(C7 cascade, 50 agents × 11 iterations) we re-checked the same
invariants against the trajectory itself. The patch adds 5 render tests
(`__tests__/components/AgentInspectorVariables.test.tsx`), applies cleanly
to a pristine clone at the pinned commit, is type-clean outside the
dashboard's already-stale `__tests__`, and adds **no** failures to its
suite (pristine: 46 failed / 107 passed; patched: 46 / 112).

Run 10 (C7 cascade) and run 11 (C8 exposure) are selectable in the
dashboard's run dropdown, **both now re-recorded from the final
(iteration-300) policies**; see §F for what they show. Each is **verified
against its own trajectory**: the cumulative counters equal the event log
(nothing dropped or double-counted), no agent is exposed to itself, the
`stance` series equals the recorded snapshot at every iteration (max
deviation 5e-5 — the 4-decimal rounding), and
`stance[p] = stance[p-1] + Σ deltas@p` holds for 50/50 agents in run 11
(run 10's 45 shifting agents agree to the same rounding). Run 10 shows
**50/50 adopted** via 45 single-period stance shifts; run 11 shows
**13/50 adopted** over 37 influences with stances frozen throughout, as
the mechanism requires.

**The wording follows the data.** `exposure` freezes stances outright
(§F.2), so the generic "mutable, moves by the judged increment" text would
promise movement the series cannot show.
`code/export_dashboard.py` therefore infers whether an episode moved any
stance at all — no non-zero increment *and* every stance flat — and words
`stance` / `adopted` / `exposure_cum` / `influence_cum` to match: run 11
reads "held FIXED for the whole episode … What changes in this arm is
behaviour, not opinion", run 10 is unchanged. Five tests pin both branches
(`code/tests/test_export_dashboard.py`, 27 tests).

## E. Colleague mechanisms wrapped as RL environments (prep, no campaign yet)

The Melodie reimplementation (PR #3) ships **two propagation
mechanisms**, now ported onto our core as `MechanismSim`
(`code/mech_sim.py`) and selectable via
`train_persuader --mechanism {paper,cascade,exposure}`:

| | `cascade` (stance-ful) | `exposure` (stance-less) |
|---|---|---|
| source rule | adopted AND stance ≤ 0 | adopted AND stance ≤ 0 |
| per-period spread | BFS — emerged nodes re-queued **same period** | persistent queue — new adopters spread **next period** only |
| stance dynamics | pull × influence × susceptibility (0.2 in their config; ours parameterizes) | **none** (frozen forever) |
| acceptance | none — every exposure runs the LLM internalization chain | LLM `accept_message` ACCEPT/REJECT gate |
| adoption | emergence = observable post/profile change | emergence, then preference_state=1 |
| RL reward channel | stance deltas + emerge shaping + judge λ·a(m) (goal-δ hook live) | **Δs ≡ 0** → adoption bonus + accept shaping + judge only |

Both modes retrain end-to-end through the PPO stack (mock-mode smoke,
2×6 episodes, passed); 16 new pinning tests; defaults unchanged.

**Consequences for training before we launch a campaign** — the
open design questions:

1. **Population stance init.** Faithful = their Beta(6,2)/-Beta(8,2)
   60/40 mix (`--mech-beta-init`). **Both** mechanisms need it, for a
   reason specific to their code: their source rule is "adopted AND
   stance ≤ 0" (our paper mode just needs `preference_state == 1` — no
   stance gate). With our all-positive init, an exposure pulls an
   adopter only to ≈ +0.32, adopted agents are never re-exposed, and
   the cascade dies after the persuaders' direct neighbours (exposure
   mode: the queue is admitted only from stance ≤ 0, same stall). Both
   stalls are pinned by tests (test_mech_envs.py). So mechanism
   campaigns run with beta init; constants (α, susceptibility), action
   space, reward plumbing and protocol stay matched to campaign 6.
   A 1–2 iteration constant-init control arm can demonstrate the live
   stall cheaply if wanted.
2. **Reward for the stance-less regime.** No stance deltas exist, so
   Track-B goal-δ is a no-op there; `--reward-adoption-bonus` +
   accept shaping + judge λ·a(m) must carry the signal. Value to set
   empirically (proposed start: bonus 1.0, shaping 0.05).
3. **Observation source.** Their observer is a particle filter over
   *observable* histories; ours reads true latent stance. v1 keeps
   true-state obs (ablation lever); a faithful run should report both.
4. **LLM-call cost.** Cascade internalization = 3 calls per exposure;
   within-period BFS can make this deep. Cached + deterministic
   estimator temperature mitigates; `--mode mock` smoke passes. A full
   300×32 campaign will need the cache-warmup step and probably
   `--n-periods` unchanged.
5. **Action semantics.** The persuader controls only its seed message
   content; organic re-sharing is endogenous LLM internalization — the
   message mutates hop-to-hop. That's the realization of their
   `RL_Generative` TODO, and makes multi-hop credit assignment
   (cascade discount) materially more important than in paper mode.
6. **Their α's differ**: 0.20 each (plus a fifth unused
   `alpha_degree_stance`) vs our 0.25. Keep ours for comparability;
   note in any paper comparison.

**Decisions (2026-10-24)**: campaign protocol *matched to campaign 6*
(α/susceptibility = ours, v2 space, BT judge, 50/5/erdos_renyi/10,
pair, random_neighbor, seed 0, 300×32, paired eval 20/50); exposure
reward = adoption bonus 1.0 + accept shaping 0.05 + judge λ·a(m)
(`PERSUASION_GOAL_LAMBDA=0.2`, λ new — campaign 6 never used reward
shaping); persuader observes true state (particle-filter obs deferred);
full 300×32 for both mechanisms. One unavoidable deviation: exposure
runs use `--mech-beta-init` (mechanism is non-functional otherwise).
Runner: `code/run_mech_campaigns.sh` (preflights the chat endpoint,
runs C7 cascade → C8 exposure serially into `output/ppo_mech_*/live`).


### §E.1 Model choice for the mechanism campaigns (2026-10-25)

The gateway (`http://192.168.0.34:8453/v1`) serves exactly two models:
`deepseek-dynamo/deepseek-v4.1-flash` (reasoning model) and
`dynamo qwen38 27b/qwen38-27b-fp8`. Measured with the probe now kept as
`code/probe_models.py` (unique inputs, so nothing is served from cache;
run while C7 was loading the endpoint, i.e. under contention):

| model | c=1 mean | c=16 | c=64 | judge mean/sd | accept rate |
|---|---|---|---|---|---|
| deepseek-v4.1-flash | 0.80 s | 16.1 calls/s | 37.3 calls/s | 0.39 / 0.23 | 3/10 |
| qwen38-27b-fp8 | 0.70 s | 14.4 calls/s | 42.2 calls/s | 0.37 / 0.30 | 3/10 |

Both are **compliant and calibrated for our call shapes**: the accept
gate returns exactly `ACCEPT`/`REJECT`, BT compare a bare `A/B/EQUAL`,
the judge a parseable number with a non-degenerate spread (5 distinct
values, neither model pins to 0 or 10) and the same 30% acceptance rate.
Per-call the 27B FP8 is ~1.1-1.4x faster; aggregate throughput at our
operating tier is a wash, because the gateway — not the model — is the
bottleneck.

**Consumption we impose** (same probe, cache-file census): ~2,500-3,000
new (uncacheable) calls/minute, i.e. **~42-50 calls/s sustained**, or
~**1.8M unique LLM calls per 300-iteration arm**. Compute is negligible
by comparison: one process, ~2.4 GB RSS, ~0.4 core, 234 I/O-blocked
threads, **814 MiB of one H200** (the other 7 GPUs at ~99% on this box
are other users' jobs).

**Decision.** Keep `deepseek-v4.1-flash` for the running C7/C8 pair: the
throughput difference is ≤1.1x at c=64, C7 was already underway, and
mixing models across the two arms would confound the mechanism
comparison. Use `qwen38-27b-fp8` for high-volume, low-stakes work where
no reward depends on the text (smoke/CI, dashboard copy, mock-ish
ablations such as a repeating-cascade C7b) — it is far cheaper to serve
per call at equivalent compliance. Any future change of model for a
*scored* campaign should use one model per campaign.

**The real speedup is call volume, not the model.** Cascade pays ~3.8
LLM calls per diffusion event, dominated by the crowd's internalisation
chain (measured composition in §E.2).

> **Correction (see §E.2).** An earlier guess in this section — that
> quantising the `delta_stance` embedded in `generate_response_message`
> was "the highest-leverage change", because it makes the prompt
> uncacheable across episodes — is **wrong for cascade**, and is
> superseded by the measurement: that path (`render_strategy_message`)
> is only **~16%** of cascade calls. The dominant ~79% is the crowd
> internalisation chain, whose prompts embed the *freshly generated*
> source post and are therefore uncacheable by construction, no matter
> how the deltas are formatted. Measured ceiling for that idea: ~16%.

### §E.2 What actually bounds wall-clock — "can more resources speed this up?" (2026-10-25)

Question asked directly, so it was measured rather than reasoned about.
**Answer: more of *our* resources do nothing; the run is a function of
LLM call volume × per-call latency ÷ concurrency, on a shared gateway we
already use heavily.**

**1. Our own box is idle.** Trainer at iteration 28: **39.9% of ONE
core** (of 176), **2.4 GB of 2,011 GB** RAM, **234/234 threads blocked on
I/O** (zero runnable), **0 MiB GPU** in use. Local CPU, GPU, RAM and disk
off the critical path.

**2. The gateway scales with concurrency, sublinearly** (probe, unique
inputs, gateway otherwise idle, `deepseek-dynamo/deepseek-v4.1-flash`):

| concurrency | 16 | 32 | 64 | 128 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|
| throughput | 27.2/s | 38.4/s | 51.9/s | 63.4/s | 70.9/s | 74.9/s | 90.4/s |
| mean latency | 0.57 s | 0.77 s | 1.09 s | 1.84 s | 3.22 s | 4.59 s | 5.02 s |

It is *not* hard-saturated at our operating point (16× concurrency buys
~3.3× rate), and throughput ≈ concurrency / latency throughout.

**3. Our calls are ~2× heavier than the probe's, and the campaign sits
at the resulting rate.** Live C7 fresh-call census: **2,282 new cache
entries / 60 s = 38/s** (29.8/s over a 200 s window), i.e. ~4–5 s mean
latency at 160 workers — where the probe does 63–66/s at 1.8–2.0 s. The
difference is prompt size. Intercepting `LLM_module.call_llm` (single
choke point) over one cascade episode (12 agents, 2 persuaders, 3
periods, live endpoint) gives the per-diffusion-event composition:

| caller | calls/episode | share | median prompt | ≈tokens |
|---|---|---|---|---|
| `generate_response_message` | 10 | 26% | 1076 ch | 269 |
| `update_profile` | 10 | 26% | 715 ch | 179 |
| `generate_information` | 10 | 26% | 692 ch | 173 |
| `render_strategy_message` | 6 | 16% | 652 ch | 163 |
| `judge_persuasiveness` | 2 | 5% | 884 ch | 221 |
| **total** | **38** | | | |

**3.80 calls per diffusion event**, of which the crowd internalisation
chain (`response → profile → information`) is **79%**. Outputs are *short*
(median 188 ch ≈ 47 tokens, p95 253), so the cost is **prefill, not
generation**.

**4. The accounting closes exactly.** 32 episodes × ~35 diffusion events
× 3.80 calls ≈ **4,250 calls/iteration**; at the observed 139 s/iteration
that is **30.6 calls/s** — the measured rate. 100% of wall-clock is LLM
calls; there is no local compute to reclaim.

**5. Ranked levers** (all measured or derived from the curve above):

| lever | expected gain | cost |
|---|---|---|
| fuse/eliminate one of the 3 internalisation calls | up to ~26% each | deviates from the imported mechanism |
| trim the internalisation prompts (prefill-bound) | ~10–30% | context/fidelity loss |
| raise concurrency 160 → 320+ | ~1.2–1.6× | latency inflation + queue pressure on a shared server |
| 32 → 16 episodes/iteration | 2× (linear) | halves statistical power per iteration |
| cache the persuader render (quantise `delta_stance`) | ≤16% on cascade | — |
| switch model | ≤1.1× (§E.1) | confounds the mechanism comparison |
| **more of our own CPU / GPU / RAM** | **0** | — |
| capacity for the gateway itself | direct — it *is* the bottleneck | not ours to add; 7 of 8 H200s are at 97–99% with other jobs |

**Operational consequence.** Because the server still scales with
concurrency while our client sits idle, the *campaign* (C7+C8) can be
compressed by running both arms **concurrently** rather than serially:
aggregate demand ~60/s is inside the envelope measured above, so both
arms should finish in roughly the time one takes alone (local cost only:
2 processes, ~5 GB, ~320 threads). For a *single* arm nothing helps
except fewer/smaller calls. 64–96 workers remains the efficient
operating point; 160 buys the same rate at higher latency.

### §E.3 The parallel launch died on a 503 burst — and what now prevents it (2026-10-25)

Running the two arms together at 160 workers each (320 threads) killed the
campaign ~2 minutes in: the gateway answered **503 Service Unavailable**
in a burst longer than the retry window, `call_llm` exhausted its 3
attempts (~10 s of backoff) and raised `LLMError`, and the runner's
`wait` returned. The failure is recorded in
`output/mech_campaigns_parallel.log`; nothing was lost that a checkpoint
could not recover, because both arms checkpoint every 10 iterations and
`--resume`/`--resume-log` re-attach the curve.

Three changes now stand between a 503 burst and a lost campaign
(commits `8112b6f`, `4e2c5cd`):

| change | why |
|---|---|
| env-tunable retries (`LLM_MAX_RETRIES=6`), backoff base 2 s, **capped** (30 s) and **jittered** in [0.5, 1.5) | the burst outlasted 3 attempts; lockstep retries from hundreds of workers are what *sustain* a 503 burst |
| per-arm workers 160 → **96** (2 × 96 = 192 threads) | 320 was the load that provoked it; §E.2 says 96 costs almost nothing in throughput |
| a bounded retry wrapper around each arm in `run_mech_campaigns.sh` (restart from the arm's own checkpoint, up to `MECH_MAX_ATTEMPTS`), with failure propagation | a crash now costs the work since the last checkpoint, not the campaign |

Relaunch (`resume-cascade-parallel`): **C7 resumed** from its iteration-20
checkpoint — its log was rewritten to `1…21` as designed (records ≥ the
checkpoint are dropped and recomputed) — and **C8 started fresh**, since it
had no checkpoint yet; its trace was truncated so iteration numbers cannot
double-count. After the relaunch: ~40–50 calls/s aggregate, **0 failed
attempts** in the first quarter hour.

One cosmetic artifact, recorded for honesty: the *trace* is append-only on
resume, so the crashed attempt's first ~10 decisions of iteration 21
remain in `output/ppo_mech_cascade/live/trace/` ahead of the re-run's
iteration 21. The training log itself is clean (the resume drops
duplicates); the monitor groups by iteration, so it only makes that one
iteration denser.

# F. Mechanism campaign — final results (2026-10-26)

Both arms ran the full **300 iterations × 32 episodes** protocol of campaign 6
(α=0.25, susceptibility 0.4, v2 strategy space, BT judge, 50 agents / 5
persuaders / erdos_renyi / 10 periods, `--obs-mode pair`,
`--target-selection random_neighbor`, seed 0, paired eval on 50 shared seeds
every 20 iterations). C7 completed in 13.4 h, C8 in 9.0 h.

## F.1 C7 cascade: the imported emergence rule collapses the cascade into one period

The mechanism adopts a target when something *observable* changed
(`code/mech_sim.py:244`, their rule):
`emerged = post_changed or profile_changed`. But `_expose_cascade`
regenerates both through the LLM internalisation chain
(`generate_response_message` → `update_profile` → `generate_information`),
and a generative backend almost always returns *something* different — so
`emerged` is effectively **always true**, and a target flips on its first
exposure. Because emerged targets also join the queue *within the same
period* (`mech_sim.py:179`, their within-period BFS), adoption reaches the
whole reachable component in **period 0**:

| evidence | value |
|---|---|
| `mean_adopted` over all **300** iterations | **50.0 / 50 every single iteration** |
| `mean_final_stance`, same iterations | −0.058 … +0.004 (essentially unmoved) |
| live trace, exposures by period (0-based), 300-iteration total 403,245 | **400,554 in period 0, 2,678 in period 1, 13 in period 2, 0 in periods 3–9** |
| dashboard run 10 (same protocol, seed 900000, final policy) | 45 stance shifts in one period; **50/50 adopted** |

**Consequence.** From period 1 on, every agent has `preference_state == 1`,
so `_eligible_target` rejects everyone: **periods 1–9 deliver nothing and
earn exactly zero reward** (the trace confirms `n_candidates = 0` and
`target = None` in every such row — 97.6% of the episode's messages). The
multi-period diffusion problem is degenerate. What the policy can still
affect is its *first-period* strategy choice — 5 persuaders acting into a
fresh network — and §G shows what that reward actually consists of: **77%
stance pull, 16% accept shaping and 7.5% population-wide cascade credit, with
no judge term at all** (`λ = 0`
unless `PERSUASION_GOAL_LAMBDA` is set, which C7 does not set). The rest of
the action space is inert.

*(Correction, 2026-10-27: an earlier revision of this section said periods
1–9 "issue zero LLM calls" and that the reward "keeps a judge term" with a
rising `train_judge`. Both are wrong. The policy still emits an action and a
message render every period — it is cached, so it is cheap, but it is not
zero work — and the BT judge never enters C7's reward; see §G.1.)*

This is *not* a port defect — it is their rule (emergence = any observable
change) meeting a backend that always emits text; the faithful port is
what makes the collapse visible. Two ways forward, and they mean different
things:

- **(a) keep C7 as a single-round persuasion-quality benchmark.** It is
  honest (we report only period-0 behaviour) and cheap: only period 0
  spends gateway calls, so C7 now costs a fraction of C8.
- **(b) make emergence proportional to the stance shift** so the cascade
  takes several periods. That is a *new* mechanism, not the imported one,
  and must be labelled as a departure.

**Decision (2026-10-25): keep (a).** C7 runs to 300 iterations unchanged
and is reported as a single-round persuasion-quality benchmark; C8 carries
the multi-period diffusion question. Rationale: (a) is faithful to the
import (the whole point of the exercise), and it is now *cheap* — only
period 0 spends gateway calls, so the cascade arm costs a fraction of the
exposure arm and does not compete with it for throughput. A
shift-proportional variant would be a new mechanism and is therefore a
separate experiment, not a patch to this one.

**Final (2026-10-26): C7 completed 300/300 in 13.4 h.** Under option (a) it is
a clean, stable single-round result: the paired eval improves monotonically
from 1.413 @20 to **1.684 @300** and beats *both* baselines in **all 15**
evals (vs random t = +4.3 … +7.7, mean **+7.16**; vs heuristic up to
t = **+20.2**). Message quality rises with it — eval judge 0.931 → 0.979 —
and, importantly, the eval strategy entropy does **not** collapse
(1.70 → 1.82), so the policy is not narrowing onto one angle. `results.json`'s
own greedy read is **1.750** vs random 1.167 / heuristic 0.743. The arm is
what it was advertised as: a first-round persuasion-quality benchmark, and
the policy clearly learns it.

## F.2 C8 exposure: the multi-period structure is alive (final)

Eleven iterations in, `mean_adopted` moved between **13.5 and 18.6 of 50**;
across the full 300 iterations it oscillates in the same band
(**13.5 … 21.6**, ending at 17.6), on a reward scale of ≈ 14 (this arm's
scale: +1.0 per persuader-sourced adoption, +0.05 accept shaping, λ=0.2
judge). The final trace confirms the structure is real: **528,917
exposures spread across all ten periods**
(48,000 / 59,343 / 61,511 / 58,457 / 54,414 / 51,693 / 49,825 / 48,931 /
48,468 / 48,275), versus C7's 400,554-of-403,245 in period 0 (§F.1). By
construction the arm has **no stance dynamics at all** — adoption moves
through a **persistent cross-period queue** gated by an LLM accept decision
— so unlike C7 it uses all ten periods. This is the arm where the
multi-period RL problem actually exists.

**Do not read `mean_final_stance` as progress in this arm.** Stances are
frozen, so the only way that sentence can move is if the episode's stance
*draw* moves: the trainer averages 32 episodes per iteration, each
initialised from `beta_stance_init` under its own RNG state, so the
`+0.011 … +0.030` seen over eleven iterations is the noise of a redraw, not
something the policy did. The recorded run-11 episode (re-recorded from the
final policy) makes the point exactly: across its 37 influences **not one
carries a non-zero increment**, all 50 agents hold their `t0` stance through
all eleven snapshots (population mean `-0.024 → -0.024`), and adoption still
spreads over periods 1–10, ending at 13/50. So the learning signal here is
`mean_adopted` and the reward; the stance series is a constant — and as of
§D.1 the node panel says so instead of implying motion.

**Reading the two arms together.** They are not directly comparable:
different adoption rules, different reward scales, and C7's reward is
nearly constant while C8's reflects genuine partial adoption. C8's curve
is the one that speaks to *diffusion over time*; C7's speaks only to
*first-round message quality*. Both are traced and live-plotted
(§E.2 / `METRICS.md`); monitors run on ports 3010 (cascade) and 3011
(exposure), and the run dashboard is on 3009 (§D.1).

**F.2a Final evaluation signal — C8 lands at parity with random, not
below it (corrects the earlier preliminary read).** The paired eval (50
episodes, identical seeds) every 20 iterations gives 15 points per arm:

| eval | greedy | random | heuristic |
|------|--------|--------|-----------|
| C7 cascade @20 | **1.413** | 1.079 (t=+4.26) | 0.755 (t=+13.73) |
| C7 cascade @300 | **1.684** | 1.079 (t=+7.70) | 0.755 (t=+20.22) |
| C8 exposure @20 | 13.962 | 14.143 (t=−1.48) | 13.702 (t=+1.93) |
| C8 exposure @300 | **14.359** | 14.143 (t=+2.90) | 13.702 (t=+5.03) |

C7 beats **both** baselines in **all 15** evals: vs random t = +4.3 … +7.7
(mean **+7.16**), vs heuristic t = +13.7 … +20.3 (mean +19.31). That is the
message-quality lesson §F.1 predicted, and it is stable.

C8's full vs-random series is
`−1.5 −9.6 −1.7 +4.9 −3.7 +6.0 −3.3 +0.4 +0.3 +4.1 +1.3 +3.5 −0.6 +5.7 +2.9`
→ mean **+0.58**: **parity with the random baseline**, with large
eval-to-eval variance; against the heuristic it is positive in 14 of 15
evals (mean +4.04).

**Correction.** An earlier revision of this section flagged C8 as "flat to
worse", on the strength of the first two evals (@20 t=−1.48, @40 t=−9.57)
plus two supporting observations. Both supporting observations **reverse by
iteration 300**, so the flag did not hold and is withdrawn:

- **strategy entropy** was 1.31 at @20 and 0.91 at @40 (which looked like
  narrowing onto one angle), but is **2.13 at @300** — it widened; it did
  not collapse.
- **reward** was written as 14.2 → 14.0; the true curve is essentially flat
  **14.20 → 14.30** with `mean_adopted` oscillating 13.5–21.6.

What survives is the substance: C8's judge score rises a lot (0.523 →
**0.808**) and its greedy reward stays above the heuristic, yet reward
stays flat and its edge over *random* sits at zero ± noise. That is
consistent with the reward being dominated by the adoption bonus, which the
LLM accept gate decides rather than the strategy — the policy improves the
one component it can move (`a(m)`) without moving adoption. So the honest
final C8 statement is: **message quality up, adoption and reward flat at
parity with random, and the mechanism's multi-period diffusion alive
(§F.2)**. Whether a reward re-shaping — shifting weight off the adoption
bonus toward `a(m)` — closes the gap is a *new experiment*, not a fix to
this arm.

# G. Analytics — where the reward actually comes from (2026-10-27)

Everything in this section is measured from what the trainer itself recorded
(`periods.jsonl`, `steps.jsonl`, `events.jsonl`), not reconstructed from
assumption: `code/analyze_mech_campaign.py` writes `report/mech_analysis.json`.
The audit path is anchored by the fact that the trace's own
`period_reward_sum` reproduces the logged `mean_episode_reward` to **1e-5**
(max |residual| 0.05 over 280–300 iterations), so the trace is a faithful
record of the reward.

## G.1 The reward algebra, recovered from the data

`code/rl_env.py::step` assembles
`reward = −Δstance + 0.05·n_accept + B·n_adopt + λ·Σa(m) − cost`. C8's stance
term is identically zero (the mechanism freezes stances), so regressing its
reward on the recorded counts and message scores must recover the weights
exactly — and it does:

| arm | recovered from the data | R² | max\|resid\| |
|---|---|---|---|
| C8 exposure | `reward = 1.0500·n_accept + 0.2000·Σa(m) + 0.0000` | **1.000000** | 0.0000 |
| C7 cascade | `reward = 0.3145·n_accept + 0.0002·Σa(m) − 0.0005` | 0.941 | 1.275 |

C8's coefficients are exactly the documented ones — 1.05 = accept shaping
(0.05) + adoption bonus (1.00), both landing on the same count because in the
exposure arm an accepted exposure *is* an adoption — and 0.20 is
`PERSUASION_GOAL_LAMBDA`. C7's regression is contaminated precisely because
the reward has **λ = 0**: the unobserved stance-pull term covaries with
`n_accept`, so the fit puts 0.31 on it and leaves a 1.27 residual. That is
the empirical proof that **the BT judge is not in C7's reward**.

**Two different judges, and "quality" means both.** The reward's alignment
term is `a(m)` = the **goal-axis judge** (a separate LLM call returning
−1…+1, normalised to 0…1), *not* the BT `persuasiveness` score that the
observations and the eval report. They are near-independent: r = **+0.14**
(C7) and **−0.01** (C8). So "message quality improved" is two different
claims depending on which judge you mean, and the report has to name which.

## G.2 Per-episode decomposition

| arm | reward / episode | stance pull | accept shaping | adoption bonus | alignment λ·Σa(m) | cascade credit |
|---|---|---|---|---|---|---|
| C7 cascade | **1.593** | 1.221 (**76.6%**) | 0.254 (15.9%) | — | — (λ = 0) | 0.119 (7.5%) |
| C8 exposure | **14.159** | 0 (frozen) | 0.246 (1.7%) | 4.922 (34.8%) | 8.991 (**63.5%**) | 0 |

Split by *variance* — i.e. by which term can actually carry a gradient:

| arm | reward var | stance pull | accept shaping | adoption bonus | alignment |
|---|---|---|---|---|---|
| C7 cascade | 0.1726 | **104.2%** | 0.8% | — | — (cascade 3.9%) |
| C8 exposure | 0.1082 | — | 0.2% | **68.9%** | 25.1% |

### G.2a The two stance terms are different things (and a correction)

C7 has **two** terms that both measure stance movement, and they must not be
confused:

- **stance pull** — `Σ −Δs` over events **sourced by a persuader**. Per event,
  attributed to *that* persuader, magnitude ∝ how far the target moved. It is
  the direct, local, dense signal: 1.221/episode (76.6%), 5.08 events/episode.
- **cascade credit** — `κ·(mean_stance_before − mean_stance_after)`, with
  `κ = 0.5`. Per *period*, a **population aggregate**, and added to **every**
  persuader (it is a cooperative/team term). Because it telescopes, its episode
  total is `κ·Σ_all(−Δs)/50` — it counts movement caused by **anyone**,
  including the ~40/episode organic BFS events the persuader never touched
  directly. 0.119/episode (7.5%).

The distinction matters because C7's propagation is a *cascade*: the persuader
directly influences ~5 agents while ~40 more move through organic spread, so a
purely local reward would give no credit for the indirect majority. The cascade
term closes that gap — at the price of a shared, unfocused signal (a persuader
is credited for movement it did not cause, and debited for movement against).

*(Correction, 2026-10-28: an earlier revision of this section listed the C7
cascade credit as 0.001 (0.04%). That was wrong twice over — the telescoping
used only the recorded period means 1…10, missing the period-0 movement where
almost all of it happens, and it used κ = 0.5 rather than the effective
`5·κ = 2.5` in the team-summed reward, since the term is broadcast to all five
persuaders. With both fixed the decomposition closes to −0.00000.)*

## G.3 Finding 1 — C8's reward is about two-thirds a constant

The alignment term is not merely the largest part of C8's reward; it is nearly
**invariant**: per-episode `Σa(m) = 44.955 ± 0.824` (coefficient of variation
**1.8%**), so `0.2·Σa(m) = 8.99 ± 0.16` of a 14.16 reward. Two consequences
follow directly from the numbers:

- Only **~35%** of C8's reward can move with behaviour at all (the adoption
  bonus); the other 64% is a near-fixed offset every episode collects.
- The *only* term with a training trend is the alignment term
  (**+0.00058/iter, r = 0.62**); the adoption term is flat
  (**+0.000054/iter, r = 0.09**), and total reward trends +0.00064/iter
  (r = 0.59).

So C8's small reward gain over 300 iterations was bought almost entirely on
the saturated, low-variance term — **not** on adoptions. That is the
mechanical account of §F.2a's parity: the policy cannot buy conversions, and
the component it *can* push is the one that barely touches the reward.

## G.4 Finding 2 — C7's reward is the mirror image

**77%** of C7's reward is the stance pull and it carries essentially all the
variance (104% of it — more than 100% because it is *negatively* correlated
with the cascade credit, r = −0.40). Its trend (+0.00097/iter, r = 0.69) *is*
the learning, and it is the reason the eval improves. Note the causal
chain this implies: the BT judge is never rewarded, yet it rose
(0.931 → 0.979). It rose **because** it is instrumental — a better-scoring
message produces a larger stance pull (§G.5), and the pull is what pays.
That is a cleaner result than "the judge improved": the policy learned to
persuade, and the judge improved as a by-product.

## G.5 Quality → outcome: the mechanism works, and it saturates

Direct persuader→target exposures (45,491 for C7, 59,698 for C8), binned by
message persuasiveness:

| persuasiveness decile | C7 mean \|Δstance\| | C8 adopt rate |
|---|---|---|
| lowest (p ≈ 0.36–0.62) | 0.159 | 0.476 |
| 2nd | 0.209 | 0.648 |
| 3rd | 0.244 | 0.699 |
| 4th | — | 0.761 |
| 5th | — | 0.830 |
| 6th | — | 0.883 |
| 7th | — | 0.866 |
| top (p = 1.0) | **0.263** | **0.930** |

Both links are monotone. C7's message quality buys **+65%** stance movement
from bottom to top decile. C8's accept gate **does read quality** — adoption
rises from 0.48 to 0.93 — but it saturates: the top three deciles move only
0.81 → 0.93, and **32% of all messages already sit at p = 1.0**. The policy
lives in the flat part of the curve, which is why improving the judge further
(0.52 → 0.81 over training) bought no adoptions.

Judge saturation, for completeness: **67.8%** of C7's judged messages score
exactly 1.0 (mean 0.938), against 31.9% for C8 (mean 0.777). A benchmark whose
judge is pinned at the ceiling for two-thirds of its decisions has little
headroom left — relevant to how much §F.1's result can be pushed.

## G.6 Finding 3 — C7's action budget is almost entirely idle

**97.6%** of C7's per-episode messages (48.8 of 50.0) fall in periods where
`n_candidates = 0`: no target, no judge call, reward exactly 0. The message is
still rendered each period (`code/rl_env.py` renders before checking for a
candidate), but from the disk cache, so it is cheap rather than free. This
corrects §F.1's "zero LLM calls" claim and explains why C7 finished in 13.4 h
against C8's 9.0 h *despite* C8 making 10× the judge calls: C7's extra time is
rollout overhead, not inference on useful work.

## G.7 What this changes about the conclusions

1. **C7 stands, with a stated ceiling.** It is a real single-round learning
   result (stance pull, +0.32 reward over the run, all 15 evals beating both
   baselines), but its judge is saturated at 1.0 for 68% of decisions and 98%
   of its action budget is idle.
2. **C8's null result is a reward-design finding, not a mechanism finding.**
   The imported exposure mechanism diffuses properly (§F.2) and its accept
   gate demonstrably responds to message quality (§G.5). What fails is the
   *reward*: 63.5% of it is a near-constant alignment offset whose weight
   (λ = 0.2) sits on a saturated, low-variance measure, leaving the policy a
   ~35% variable slice that an LLM gate — not the strategy — decides.
3. **The concrete next lever** (a new experiment, not a patch): either drop or
   variance-normalise the constant alignment offset so the gradient is not
   diluted, and/or redefine the alignment reward on the *converting* subset
   (adoptions per exposure, or `a(m)` only on accepted exposures) so the term
   the policy can move is the term that pays. §G.1's recovered algebra gives
   the exact arithmetic to size that change.


