# Melodie reimplementation (colleague's branch)

These are the versions of `LLM_module.py`, `agent.py`, `config.py`,
`scenario.py`, and `tools.py` from `ShaghayeghNajari-patch-3` (merged via
PR #3 + follow-up uploads), preserved here because the same paths under
`code/` carry the RL-persuader stack that `rl_env`/`train_persuader`
import. The two stacks are parallel implementations of the paper:

- `code/*.py` (ours) — lightweight non-Melodie simulation, RL persuader,
  BT judge, v2 strategy space; relative imports, unit-tested.
- this directory + `code/{analyzer,environment,model,run_simulator,
  simulator,stance}.py` + root `agent.py` (theirs) — Melodie-framework
  reimplementation (`from Melodie import ...`, `source.*` imports,
  particle-filter inference, stance Beta distributions, LocalBiz
  campaign messages, `Sensitivity_evaluation.ipynb`).

Theirs expect a Melodie project layout (package `source/`, root
`config.py`); see `code/run_simulator.py` for the intended entrypoint.

## Upstream tracking

`agent.py` here is refreshed to upstream `main` @ `4083fad` (2026-10-25,
"Add files via upload"). That commit:

- replaced the two duplicate event-log calls per infection (they
  referenced undefined locals — the defect our port records as
  `MELFIX 3`) with zero-placeholders. **Mechanism semantics unchanged**:
  no change to the `emerged` guard, the BFS, or adoption.
- added `LLMEnvironment.reset(agents)` (clears `emerged` + both event
  lists) and calls it between the stance-aware and the exposure phase in
  `LLMModel.run()`, with `m` 2 -> 3 — i.e. cascade -> reset -> exposure is
  now one pipeline. The reset exists because `emerged` is sticky, so an
  agent can be influenced only once per phase; clearing it lets the
  exposure phase reach agents the cascade already converted.
- two defects, left as-is because this directory is a verbatim archive:
  `def reset(agents)` is missing `self`, so `environment.reset(...)`
  raises `NameError`; and `m`/`k` is inert (`if target.exposure_count <
  k` is commented out), so `m=3` has no effect.

`code/environment.py` and `code/model.py` carry those upstream edits
directly (our RL stack does not import them). `code/agent.py` keeps ours,
as above.
