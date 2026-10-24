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
