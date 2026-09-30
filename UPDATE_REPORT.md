# Codebase Update Report

Date: 2026-09-27
Scope: full remediation of `code/` based on the code review, plus
infrastructure required for RL training (Option B).

## 1. Critical bug fixes

| # | Bug (original) | Fix | Where |
|---|---|---|---|
| 1 | `accepted = True` hardcoded — the stochastic acceptance gate of Algorithm 1 was commented out; `calculate_influence_prob` was dead code; every exposure was accepted | Restored `accepted = rand_value < influence_prob` with a seeded `np.random.Generator` | `agent.py` step 8 |
| 2 | Embedding guard printed "no valid post embedding" and returned *even when the embedding was just computed successfully* — every first-time source was skipped with a spurious warning | Guard now only skips when the embedding is genuinely `None` after computation | `agent.py` step 4 |
| 3 | Non-adopted agents were saved to `diffusion_events` **twice** (before generation and after), inflating the 161-event dataset | Single `save_diffusion_event` call per interaction; regression test asserts unique (period, source, target) triples | `agent.py` steps 12-14, `tests/test_codebase.py::TestNoDuplicateEvents` |
| 4 | `behavior` redefined after stance update as `"accept" if stance < 0` — silently corrupted the downstream `adopted` flag | `behavior` fixed by the stochastic gate; adoption tracked separately via `reached_adoption_threshold` | `agent.py` step 10 |
| 5 | `Inferr_stance*` / `estimated_stance_shift*` assigned only inside `if` guards but passed unconditionally → `NameError` on any `None` LLM output | All variables pre-initialized to `None`; each computation individually guarded; event schema stores `None` explicitly | `agent.py` step 17, regression test `TestNoneSafety` |
| 6 | Commented-out `return` let non-adopted agents generate content and propagate in the same period (contradicting Algorithm 1 and the paper's "propagation starts next period") | Early return restored for the non-adoption path | `agent.py` step 14 |
| 7 | **Adoption threshold deadlock**: default `adoption_threshold = -0.5` combined with `stance_influence_rate = 0.4` makes adoption unreachable (max one-step shift ≈ 0.44, and targets with stance ≤ 0 are ineligible for further influence) | Default threshold corrected to `0.0` (consistent with the original code's `stance_after < 0` adoption semantics), with an explanatory comment | `config.py`, `scenario.py` |
| 8 | Observer functions returned raw LLM strings; "Return only one numerical value" was a prompt plea, not a guarantee | `parse_stance_shift()` extracts and validates a float in [0, 1] (last-number heuristic, clamping, 0.0 fallback) | `LLM_module.py` |

## 2. Design fixes

- **Temperature split**: observer inference now runs at temperature 0.0
  (was 0.8 — injecting noise directly into the measured quantity);
  generation stays at 0.8. (`config.py`)
- **Retry + backoff + caching**: single `call_llm()` entry point with
  exponential-backoff retries, empty-output detection, and an on-disk
  response cache (`.llm_cache/`). Five copy-pasted parameter blocks
  (~90% duplication) collapsed into one helper. (`LLM_module.py`)
- **Configuration externalized**: model name, base URL, API key,
  temperatures, max_tokens, scenario parameters moved from hardcoded
  literals to `config.py` (env-var overridable). (`config.py`)
- **Missing dependencies removed/replaced**: the codebase imported
  `Melodie`, `source.tools`, and `source.scenario`, none of which were
  provided. Added `base_agent.py` (minimal `NetworkAgent` stand-in),
  `tools.py` (sentence-transformers embeddings with a deterministic
  hash fallback), and `scenario.py`. (`base_agent.py`, `tools.py`,
  `scenario.py`)
- **Logging instead of prints**: `print`-based debugging (including a
  `[ADOPTION]` banner on every accepted event) replaced with the
  `logging` module. (throughout)
- **Event schema**: embeddings serialized as plain lists (JSON-safe);
  `influencer` stores agent id rather than an object reference
  (previously created unserializable cycles).
- **Embedding efficiency**: old profile embedding reuses the cached
  `profile_embedding` instead of recomputing per event.
- **Cosine similarity hardened**: shape-mismatch and zero-norm guards;
  no sklearn dependency at call time.

## 3. New infrastructure

- **`simulation.py`** — the missing engine: population initialization,
  network construction (Watts-Strogatz default; BA / ER / complete
  available), seed-agent setup, and the period-based diffusion loop.
  Deterministic given `rng_seed`.
- **`network.py`** — topology builders with degree / degree-centrality
  / betweenness computation (previously referenced but absent).
- **`env.py`** — `LatentStanceDiffusionEnv`, a Gymnasium wrapper
  exposing the persuasion control problem (action = seed message,
  reward = realized population stance shift from ground truth,
  observer-side observations by default). RL entry point.
- **`offline.py`** — deterministic mock LLM backend for CI, demos, and
  fast RL rollouts without a live server.
- **`tests/test_codebase.py`** — 18 tests covering every bugfix above
  plus determinism, Gym API compliance, and no-duplicate-event
  invariants. All pass.

## 4. Project hygiene

- `requirements.txt`, `README.md` (setup, run, structure).
- Virtual environment at `.venv/` in the project root.
- Bilingual Farsi/English comments unified to English.

## 5. Verification

```
$ .venv/bin/python -m pytest code/tests -q
18 passed in 0.23s
```

Offline end-to-end demo (mock LLM): diffusion progresses, events
accumulate without duplicates, mean stance declines over periods.

## 6. Known limitations / next steps

- The paper's reported numbers were produced by the buggy code
  (always-accept, duplicate events); **all results must be re-run and
  re-reported** before submission.
- Hash-fallback embeddings are deterministic but low-quality; install
  `sentence-transformers` for real experiments (backends are not
  numerically compatible — pick one per experiment).
- RL (Option B) should start from `env.py`; a fine-grained
  per-message policy interface is the next milestone.
