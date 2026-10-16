# Implementation Plan — Dynamics & Readout Integrity

Status: **proposed, not started.** These are the next-run changes; the live
`output/ppo_v2_bt/live_run2` run is unaffected.

**Priority: C > A > B.** Track C is essential and high priority — it is a
correctness/robustness fix that removes a silent failure mode from the reward
readout, and it must land before any further training that we intend to trust.
A and B are improvements that build on a readout we can rely on.

Scope: the three open issues raised after the v2/BT campaign.

- **A.** The dynamics constants (`α = 0.25` on the four `p` terms, `β = 0.4`
  stance step) are unjustified placeholders — make them defensible.
- **B.** The message's *direction toward the goal* is never measured; inject
  it into the stance delta so the reward can't pay for directional noise.
- **C.** The BT readout is keyed on live-rendered message text; harden the
  keying so it can't silently degrade to the saturated absolute judge.

---

## 0. Corrected premise for C (measured, then fixed)

The claim "the live message and the pre-built BT messages have near-zero
chance of intersection" is **false as the system stands today**, and I
verified it directly:

```
table entries: 3600
profile exact-in-table: 50 / 50
cell hits: 360   miss: 0   / 360     # 5 profiles × all 72 strategies
```

and the live run confirms it — iteration 1 of `live_run2` logs
`train_judge = {mean: 0.734, std: 0.237}`, which **is** the BT table's
`score_mean 0.742 / score_sd 0.233`, not the absolute judge's.

**Why they intersect** (the mechanism, so we know what protects it):

1. Rendering is **deterministic** — `render_strategy_message` calls the LLM at
   `temperature=0.0` (`code/LLM_module.py:379`).
2. `call_llm` is **content-addressed**: the cache key is
   `sha256({model, prompt, system, temperature, max_tokens})`
   (`code/LLM_module.py`, `call_llm`). Same prompt + same model ⇒ same cached
   bytes, so the training render returns the exact table string.
3. Both the BT build (`code/train_persuader.py:458`) and the env
   (`code/rl_env.py:510`) render with the same `topic="the policy"` and
   `stance_direction="opposing"`, and both use
   `generate_synthetic_profiles(50, seed=1234)`.

So the intersection is currently 100% — **but only by coincidence of three
invariants that nothing enforces.** The real problem the comment exposes is
that the join is a **raw string equality on a whole paragraph**, with a
**silent fallback** to the absolute judge on miss. Any of these breaks it
quietly, reintroducing the saturation the BT repair was built to remove:

- the LLM model/version changes between build and training (cache key drift);
- the cache is evicted and the server is not bit-deterministic at temp 0;
- `topic`, prompt template, or profile seed changes;
- a strategy arm is added/removed, or a target's profile is updated.

Fix: **C below** — key on `(strategy_index, profile_id)`, assert coverage at
startup, and make any fallback loud.

---

## Track A — Calibrate the dynamics constants

### A0. Problem
- `α_profile/α_content/α_stance/α_degree = 0.25` (`code/config.py:101-104`) are
  equal *by assumption*; the paper defines `p_ij ← InfluenceProbability(...)`
  but never expands it.
- `β = stance_influence_rate = 0.4` (`code/config.py:88`) is inherited, not
  derived. It sets the reward scale (`reward ≈ β·p·gap`), convergence speed,
  and time-to-adopt.

### A1. Fit the α's from logged events (data-driven)
- Data already exists: every interaction logs `influence_probability`, source/
  target ids, and `behavior ∈ {accept,reject}` in `sim.diffusion_events`
  (`code/agent.py:543`). All four similarity components are recomputable from
  the logged source/target state.
- Build `code/fit_alphas.py`:
  1. Replay logged episodes (or run a short frozen-policy batch) and emit one
     row per interaction: `[profile_sim, content_sim, stance_sim, degree_sim, accepted]`.
  2. Fit a logistic model `P(accept) = σ(Σ α_k · sim_k + b)` (constrained to
     `α_k ≥ 0`).
  3. Export the fitted α's and a calibration plot (predicted vs empirical
     accept rate by decile).
- Fallback if the design matrix is collinear (profile/degree are near-constant
  within an episode): report effective rank and fall back to a **sweep** rather
  than a fit.

### A2. Calibrate β to the paper's dynamics
- The paper reports targets we can match (extract from
  `Agentic_Information_diffusion.pdf`): early mean `ΔS ≈ −0.168` (`t∈[1,3]`),
  late `ΔS ≈ −0.107` (`t≥16`), convergence to `S ≈ −0.55`.
- Add a `code/calibrate_beta.py` sweep of `β ∈ {0.1 … 1.0}` on the frozen
  policy; score each by (i) plateau `S`, (ii) early/late mean `|ΔS|`,
  (iii) periods-to-adopt. Pick the β (and, jointly, the p scale) minimizing
  distance to the paper's three numbers.
- Report the chosen β with the sweep curve, not as a bare constant.

### A3. Acceptance criteria
- α's and β are produced by a script, written to config, with the source
  data hash + script version recorded (reproducible).
- A one-page justification (fit quality / sweep) lands in `RESULTS.md`.
- Defaults unchanged until the calibration passes; v1/v2 runs remain comparable.

**Status: DONE (2026-10-16) — with a negative result, honestly reported.**
- **α:** not separately identifiable (effective rank **2.28/4**; profile &
  degree near-constant). Keep the equal-weight prior, now documented as an
  *unidentifiable prior*; the linear recovery is exact (0.25, resid 8e-5) and
  serves as an instrumentation check. `code/fit_alphas.py` →
  `output/calibration/alphas.json`.
- **β:** the dynamics identify only **ρ = β·κ** (κ = α scale). ρ ≈ 2
  (β≈0.9, κ≈2–3) jointly matches early ΔS (−0.16 vs −0.168) and the plateau
  (−0.46/−0.55), but **late ΔS is structurally 0** vs the paper's −0.107 (our
  transition reaches a static fixed point; no stable intermediate consensus).
  β is therefore **not the binding knob** — the residual is the transition
  *form*, which Track B addresses. Keep β = 0.4. `code/calibrate_beta.py` →
  `output/calibration/beta*.json`.
- Config annotated; `RESULTS.md` Campaign 4; 4 new tests (83/83 pass).

---

## Track B — Inject goal direction into the stance delta

### B0. Problem
Direction toward the goal is never measured. The text enters `p` only through
`content_profile_similarity` (post↔**target persona** fit) and the judge
(persuasive-**for-the-reader**) — **both are fit, neither is direction**. And
the transition `Δs = p·β·(s_src − s)` (`code/agent.py:135`) is
**text-independent**: the target is pulled toward `s_src = −1` regardless of
what the post argues. So a persuasive-but-off-goal post still raises `p` and
still moves the target, and can be rewarded.

### B1. Define the goal as an anchor (prerequisite)
- Add `G_goal` (a proposition string, e.g. *"This policy is harmful and should
  be rejected"*) and its negation `G_anti` to config; default derived from the
  seed post in `code/simulation.py:100` so the goal is finally explicit.
- This replaces the current placeholder `topic="the policy"` with a defined
  proposition (and a `topic` string for rendering).

### B2. Measure alignment `a(m)` of a rendered message
- Implement `code/goal_align.py` with three interchangeable backends, cheapest
  first:
  1. **Embedding semantic differential:** `a(m) = ½·(cos(m, G_goal) − cos(m,
     G_anti) + 1) ∈ [0,1]` — reuses the existing embedding backend, ~free.
  2. **NLI entailment (local):** `entail(m, G_goal) − entail(m, G_anti)`,
     frozen MNLI model, offline.
  3. **LLM goal-axis judge:** a single cached prompt returning `−1…+1`
     ("does this argue for or against G_goal?"), reusing `call_llm` + cache.
- Add an `--align {none,semdiff,nli,judge}` flag. Cache by message text.

### B3. Inject into the stance delta
Two candidate forms; **recommend (ii)**:
- (i) **Gate in `p`:** `p ← p · a(m)` — simplest, but starves learning early.
- (ii) **Direction in the delta (recommended):** replace the effective source
  stance by a goal-projected one, so the *sign of the movement* follows the
  message:
  ```
  s_eff(m) = s_src · (2·a(m) − 1)          # a=1 -> s_src, a=0 -> -s_src (wrong way), a=0.5 -> 0
  s' = clip(s + p·β·(s_eff(m) − s), −1, 1)
  ```
  An off-goal or off-topic post then moves the target *toward* its prior (or
  not at all), so it cannot earn `−Δs` reward. Optionally add a small explicit
  term `reward += λ·a(m)` for a denser early signal.
- Implement in `code/agent.py:update_agent_stance` behind
  `PERSUASION_GOAL_DELTA=1` (default off, so v2 stays comparable). The
  signature already threads the source; pass the computed `a(m)` (stored on the
  source as `message_goal_align`, alongside `message_persuasiveness`).

### B4. Validation
- `a(m)` must correlate with realized `−Δs` and rank the seed message highly.
- Directional sanity: a hand-built *pro*-policy post must yield `a≈0` and
  non-positive reward; an *anti* post from the same arm yields `a≈1`.
- A/B: current vs B3(ii), 50-seed paired eval, same protocol as the campaign
  reports. Report goal metrics (`mean_final_stance`, `adopted`) alongside reward.

### B5. Acceptance criteria
- Goal is a named proposition in config; `a(m)` is logged per decision.
- Off-goal posts are measurably penalized (negative/zero reward) without
  collapsing message variety (watch the variety verdict in
  `report/message_variety.py`).
- `update_agent_stance` change is unit-tested in `code/tests/test_rl.py` and
  covered by the offline mock (`code/offline.py`).

**Status: DONE (2026-10-16) — implementation + offline validation + A/B (campaign 6).**
- **B1:** goal is a named proposition (`goal_proposition` /
  `goal_anti_proposition`) in `code/config.py` + `code/scenario.py`, derived
  from the seed post — no more bare `topic` placeholder.
- **B2:** `code/goal_align.py` — `semdiff` (default, reuses embeddings),
  `nli` (offline fallback), `judge` (cached LLM), all cached. Measured
  directional sanity: anti 0.665 > neutral 0.513 > pro 0.340 (semdiff).
  Caveats recorded: semdiff is weak-but-monotone; the LLM judge is sharp on
  clean posts but flaky on mixed/hedged ones.
- **B3:** injected as `s_eff = s_src·(2a−1)` in `update_agent_stance`, behind
  `--goal-delta` / `PERSUASION_GOAL_DELTA`; scored on the persuader post,
  logged per decision (`goal_align`), surfaced as `train_goal_align`.
- **B4:** `code/validate_goal.py` passes — directional sanity ✓,
  corr(a, −Δs)=+0.177, on-goal −Δs (+0.0373) ≈ 1.5× off-goal (+0.0253).
- **B5 A/B:** DONE — campaign 6 (`report/ab_goal.json`, `code/ab_goal.py`).
  Matched goal-delta arms vs the no-goal baseline at shared paired-eval
  iterations. **The backend is the crux:** on the real strategy messages the
  embedding backends read `a(m)≈0.51` (neutral) and the semdiff arm collapses
  (tail reward 5.15 vs 12.75); the LLM judge reads `a(m)≈0.97` and its arm
  tracks the baseline (`@60` greedy 12.467 vs 12.487) while still learning
  (`d+3.98`). Injection scoped to the persuader's messages (crowd keeps the
  paper's transition). Item 16's full campaign should use `--align judge`.
- 6 new tests; **87/87 pass**.

---

## Track C — Harden the BT readout keying  ⭐ PRIORITY 1

### C0. Why this is first
The readout is the *binding constraint* on everything the policy can learn.
Today it works only by coincidence of three unenforced invariants (temp-0
determinism, content-addressed cache, matching seed/topic). Any drift sends
training silently back to the saturated absolute judge — the exact failure the
BT repair removed. Fixing the join is cheap and protects every later result.

### C1. Key by identity, not text
- [x] Assign a stable `profile_id` per agent at sim init (index into
  `generate_synthetic_profiles(50, seed=1234)`); store it on the agent.
- [x] In `build_table` (`code/bt_judge.py:155`), additionally index the result
  by `(strategy_index, profile_id)` — keep the `(message, profile)` table for
  audit/report.
- [x] Extend `BTJudge` with an identity lookup
  `lookup_by_id(strategy_idx, profile_id) -> float | None`.
- [x] Change the call site (`code/rl_env.py:518`) to pass `strategy_idx` and the
  target's `profile_id`; make `_judge_score` (`code/rl_env.py:448`) do an O(1)
  identity hit and not depend on re-rendered text.

### C2. Assert coverage at startup
- [x] On load (`code/train_persuader.py:452`), assert
  `n_cells == N_STRATEGIES × n_profiles` for the configured space/seed.
- [x] **Fail loudly** (raise, not warn) on any missing cell or a table built
  for a different space/seed.
- [x] Migration: `BTJudge.attach_ids(strategies, profiles, topic)` derives the
  identity index for tables written before it existed (cached renders) and
  persists it — the existing `bt_judge.json` needed no full rebuild.

### C3. Make fallback loud and measured
- [x] Count table hits vs fallbacks per iteration; added `train_judge_src`
  (`{id, text, none}`) to every history record and `judge[id= text= none=]`
  to the periodic log line in `code/torch_trainer.py`.
- [x] Emit a warning the first time a fallback occurs (`text > 0`); counts live
  in `training_log.json`.

### C4. Tests
- [x] `code/tests/test_rl.py`: normal path is a pure identity lookup
  (`test_env_uses_identity_path` — asserts `judge_src == "id"`, fallback 0).
- [x] A drifting/near-miss key falls back **visibly**
  (`test_env_counts_fallback_on_identity_miss` — asserts `judge_src == "text"`
  and the counter fires), never silently; persistence and migration covered.

### C5. Acceptance criteria
- [x] Changing the model, topic, prompt template, or profile seed **cannot**
  silently route training to the absolute judge — a missing identity cell
  raises at startup, and is otherwise counted + logged at runtime.
- [x] Fallback rate is 0 on the happy path and is logged/visible otherwise.

**Status: DONE (2026-10-16).** Migration indexed 3600/3600 cells; an
end-to-end smoke run reported `id=15, text=0, none=0` (zero fallbacks), and
`pytest code/tests/` passes 79/79.

---

## To-do list (ordered)

**Now — Track C (priority 1, before any further trusted training)**
1. [x] `profile_id` per agent at sim init (`code/simulation.py`, `code/agent.py`).
2. [x] Identity index + `lookup_by_id` in `code/bt_judge.py` (`build_table`,
   `BTJudge`).
3. [x] Rewire `_judge_score` / call site to identity lookup
   (`code/rl_env.py`).
4. [x] Startup coverage assertion, fail-loud (`code/train_persuader.py`).
5. [x] Hit/fallback counters in `train_judge` (`code/torch_trainer.py`).
6. [x] Tests in `code/tests/test_rl.py` (normal + drift).
7. [x] Index the table with the identity index; verified a live smoke run
   hits 100% by identity (`id=15, text=0, none=0`).

**Next — Track A (calibration, no RL-path changes)**
8. [x] `code/fit_alphas.py`: fit/audit the four α's from logged events —
   result: **not identifiable** (effective rank 2.28/4); keep equal weights.
9. [x] `code/calibrate_beta.py`: joint (β, κ) sweep vs the paper's dynamics —
   result: early ΔS + plateau match at ρ≈2, **late ΔS structurally 0**; β is
   not the binding knob.
10. [x] Chosen α/β + evidence written into `code/config.py` comments and
    `RESULTS.md` (Campaign 4); defaults unchanged for comparability.

**Then — Track B (goal direction, behind a flag)**
11. [x] Define `G_goal`/`G_anti` proposition in config (replace placeholder topic).
12. [x] `code/goal_align.py`: `a(m)` backends (semdiff / NLI / goal-axis judge).
13. [x] Inject into the delta in `update_agent_stance` behind
    `PERSUASION_GOAL_DELTA=1` (`code/agent.py:135`).
14. [x] Validation: `a(m)` vs realized `−Δs`; directional sanity (pro vs anti).
15. [x] 50-seed paired A/B (goal-delta vs baseline) — campaign 6, both
    backends; folded into report + `RESULTS.md`.
    — deferred to item 16 (needs the full paired run).

**Finally**
16. [ ] Re-run the 300×32 campaign on a fresh output dir (v2 baselines stay
    comparable); regenerate `report/index.html`.

## Sequencing
1. **C** first (cheap, removes a silent failure mode; no behavior change on the
   happy path). **Do not start a fresh trusted training run before C lands.**
2. **A** (calibration scripts; produces constants + evidence, doesn't touch RL
   code paths).
3. **B** (largest; behind a flag, needs the A/B).
4. Re-run the 300×32 campaign once A+C are in and B is validated offline; fold
   results into `report/index.html` and `RESULTS.md`.

## Test/ops notes
- All new LLM-touching code must work under `code/offline.py`'s mock and run
  without the live endpoint.
- New CLI flags: `--align`, `--goal-delta`, `--bt-key {text,identity}`.
- No change to `output/ppo_v2_bt/live_run2`; new runs go to a fresh output dir
  so v2 baselines stay comparable.
