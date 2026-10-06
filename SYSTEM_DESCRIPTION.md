# System Description — Agentic Information Diffusion with Latent Stance Dynamics & RL Persuaders

This document describes the complete implemented system: the fixed and
extended simulation codebase, the observer-side inference pipeline,
and the reinforcement-learning layer for training persuasion agents
(Stage 1, approach 2), including the planned path to a direct language
policy (approach 3) and multi-agent scaling.

---

## 1. Purpose

The system models information diffusion in a social network of
LLM-based agents in which every agent carries a **latent stance**
`s ∈ [-1, 1]` that evolves as messages are received and internalized.
It supports three research activities:

1. **Diffusion simulation** — message propagation coupled to latent
   stance transitions (paper's contribution 1).
2. **Observer-side inference** — an external LLM observer reconstructs
   per-interaction stance shifts and full stance trajectories from
   observable evidence: generated posts, profile changes, or both
   (paper's contributions 2 and 3).
3. **RL persuasion training** — a learned persuader policy selects
   messaging strategies to maximize real stance shifts in the
   population, with reward taken from simulator ground truth (new,
   Stage-1 single-agent RL; approach 2 of the RL plan).

## 2. Architecture Overview

```
                        ┌──────────────────────────────────────────────┐
                        │                 config.py                    │
                        │  LLM backend, scenario parameters (env-var   │
                        │  overridable, no hardcoded values)           │
                        └───────────────┬──────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
┌───────▼────────┐             ┌────────▼─────────┐            ┌────────▼────────┐
│  LLM_module.py │             │   tools.py       │            │   scenario.py   │
│  chat layer:   │             │  embeddings      │            │  experiment     │
│  retry/cache/  │             │  (MiniLM or      │            │  parameters     │
│  parsing +     │             │  hash fallback)  │            │                 │
│  all prompts   │             └────────┬─────────┘            └────────┬────────┘
│  incl. strategy│                      │                               │
│  renderer      │                      │                               │
└───────┬────────┘                      │                               │
        │                               │                               │
┌───────▼───────────────────────────────▼───────────────────────────────▼────────┐
│                            simulation.py / network.py                          │
│   population init, topology (WS/BA/ER/complete), centralities, seed agent,     │
│   period-based diffusion loop, event logging (diffusion + message events)      │
└───────┬────────────────────────────────────────────────────────────┬───────────┘
        │                                                            │
┌───────▼─────────────────┐                              ┌────────────▼───────────┐
│      agent.py           │                              │       env.py          │
│  LLMSocialAgent:        │                              │  Gymnasium wrapper:   │
│  stochastic acceptance  │                              │  seed-message control │
│  gate, linear stance    │                              │  (baseline env)       │
│  transition, adoption,  │                              └────────────────────────┘
│  message/profile gen.,  │
│  observer inference,    │                              ┌───────────────────────┐
│  event schemas          │                              │      rl_env.py       │
└─────────────────────────┘                              │  ElitePersuaderEnv:   │
                                                         │  strategy-vector     │
┌─────────────────────────┐                              │  actions, reward =   │
│      offline.py         │                              │  ground-truth stance │
│  deterministic mock LLM │                              │  shift + shaping     │
│  (tests, CI, RL debug)  │                              └───────────┬───────────┘
└─────────────────────────┘                                          │
                                                        ┌─────────────▼──────────┐
                                                        │  strategy.py +         │
                                                        │  trainer.py            │
                                                        │  180-strategy space;   │
                                                        │  numpy A2C/REINFORCE   │
                                                        │  policy, evaluation,   │
                                                        │  message-bank baseline │
                                                        └────────────────────────┘
```

## 3. Simulation Core

### 3.1 Agent lifecycle per interaction (`agent.py: infection_impact`)

1. **Eligibility checks** — only sources with stance below the
   adoption threshold can influence; only positive, non-adopted
   targets can be influenced; self-influence blocked.
2. **Stochastic acceptance gate** — `accepted ~ U(0,1) < p_influence`
   (Algorithm 1 of the paper; was hardcoded `True` in the original
   code — bugfix).
3. **Latent stance transition** — linear DeGroot-style update:
   `s' = s + p_influence · susceptibility · (s_source − s)`,
   clipped to [-1, 1].
4. **Internalization rendering** — LLM (Prompt 4) writes the
   *internalized* portion of the message, conditioned on `|Δs|`.
5. **Adoption** — crossing the adoption threshold (default 0.0) makes
   the agent a propagator from the next period.
6. **Observable content generation** (adopted agents) — new profile
   (Prompt 3) and new post (Prompt 2) via LLM.
7. **Observer-side inference** — three evidence modalities
   (post-only, profile-only, combined), each producing (a) an LLM
   observer estimate in [0, 1] (parsed and validated) and (b) an
   embedding-distance estimate (1 − cosine similarity).
8. **Event logging** — one diffusion event and (for adopters) one
   message event per interaction, including ground-truth stances,
   lineage ids, profiles, posts, embeddings, and all estimates.

### 3.2 Engine (`simulation.py`)

- Population of `agent_num` agents, Watts-Strogatz topology by
  default (Barabasi-Albert, Erdos-Renyi, complete available), degree /
  betweenness centralities computed at init.
- Seed agent (id 0) starts at stance −1.0 with the campaign message;
  all other agents start at +0.65 (paper setup).
- One `step()` = one diffusion period: every propagating source
  addresses its eligible neighbors; `run(n)` iterates.
- Deterministic given `rng_seed` (single shared `np.random.Generator`
  drives all acceptance gates).

### 3.3 LLM layer (`LLM_module.py`)

- Single `call_llm()` entry point: exponential-backoff retries,
  empty-output detection, on-disk response cache (`.llm_cache/`).
- **Temperature split**: generation 0.8, observer inference 0.0
  (estimators must be near-deterministic).
- `parse_stance_shift()` extracts and validates the observer's numeric
  answer (last-number heuristic, clamp to [0, 1], 0.0 fallback).
- Prompts 1–5 from the paper, plus `render_strategy_message()` — the
  frozen-LLM decoder from strategy specs to persuasion messages (RL
  approach 2).

### 3.4 Embeddings (`tools.py`)

- `sentence-transformers/all-MiniLM-L6-v2` when installed; otherwise a
  deterministic hash-based bag-of-words fallback (256-d) so tests and
  CI run without model downloads. Backends are not numerically
  compatible — pick one per experiment.

### 3.5 Offline mock (`offline.py`)

- Deterministic canned responses for every LLM-backed function,
  including the strategy renderer. Used for tests, CI, demos, and RL
  pipeline debugging. Stance dynamics (which don't depend on LLM
  output) remain fully functional under the mock.

## 4. Observer-Side Inference (evaluation pipeline)

For each accepted interaction the message event stores:

| Field | Meaning |
|---|---|
| `target_stance_before/after` | simulator ground truth |
| `Inferr_stance_post` | LLM observer estimate from post evidence |
| `Inferr_stance_profile` | LLM observer estimate from profile evidence |
| `Inferr_stance` | LLM observer estimate, combined evidence |
| `estimated_stance_shift_*` | embedding-distance estimates (baseline) |

Trajectory reconstruction: accumulate per-agent estimates across
events and compare to ground-truth trajectories (paper §5.2/5.3
analysis). **Important**: the paper's reported numbers were produced
by the pre-fix code (always-accept, duplicated events) and must be
re-generated with this codebase before submission.

## 5. RL Layer — Stage 1, Approach 2 (implemented)

### 5.1 Design decision: single-agent RL with a frozen population

One learned **elite persuader** operates among frozen LLM agents.
This keeps the environment stationary (RL theory applies), credit
assignment clean, and models the research question "can an agent learn
to persuade?" The crowd acting as-is is the experimental control.
Multi-agent scaling (opposed persuaders → full population with
parameter sharing) is the planned Stage 2/3 path (§7).

### 5.2 Strategy space (`strategy.py`)

The policy does not emit text. It selects a `StrategySpec`:

| Dimension | Values |
|---|---|
| appeal | logos / pathos / ethos |
| tone | neutral / urgent / empathetic / confrontational |
| emphasis | evidence / personal_story / social_proof / risk / authority |
| intensity | 0.33 / 0.66 / 1.00 |

Cartesian product → **Discrete(180)** actions. `render_strategy_message()`
(frozen LLM, temperature 0) decodes a spec into a ≤50-word message
conditioned on the target profile.

### 5.3 Environment (`rl_env.py: ElitePersuaderEnv`)

- **Observation (13-d)**: mean-pooled target profile embedding (8-d) +
  degree centrality, betweenness, exposure count, observable stance
  proxy, network mean-stance signal. Compact by design for CPU
  training.
- **Action**: Discrete(180) strategy index.
- **Step**: render message → deliver from persuader (agent 0) → run
  one diffusion period.
- **Reward**: sum over persuader-sourced events of `−Δs_target`
  (ground-truth shift toward the persuader) + acceptance shaping
  bonus − per-message cost; optional discounted cascade term
  (population mean-stance movement).
- **Episode**: `n_periods` diffusion periods. Duck-typed Gymnasium API
  (`reset`/`step`), seeds propagate for reproducibility.

### 5.4 Trainer (`trainer.py`)

- `RLPolicy`: pure-numpy MLP (13 → 64 → 180) with softmax policy head
  and value head; REINFORCE with value baseline (advantage
  actor-critic updates), entropy regularization. **CPU-only by
  design** — no torch/tensorflow dependency.
- `train()`: episodic loop, discounted returns, advantage updates,
  periodic greedy evaluation, training log JSON + policy checkpoint
  (`npz`).
- `evaluate()`: greedy mean episode reward on held-out seeds.
- `evaluate_message_bank()`: uniform-random-strategy baseline (the
  no-learning control, comparable to the paper's fixed message bank).

### 5.5 Verified behavior

- 29 pytest tests pass (bugfix regressions, determinism, Gym API,
  strategy round-trips, save/load, short training runs).
- Offline training demo: 120 episodes in <1s on CPU with the mock
  backend — the full loop (render → diffuse → reward → update →
  eval → baseline comparison → checkpoint) executes correctly.
  Meaningful policy improvement requires the **live LLM renderer**,
  where strategy choice actually changes message content and hence
  influence probabilities.

## 6. How to Run

```bash
# environment
python3 -m venv .venv && .venv/bin/pip install -r code/requirements.txt

# tests (offline, no LLM needed)
.venv/bin/python -m pytest code/tests -q

# baseline simulation (needs live LLM, e.g. local Ollama)
.venv/bin/python -c "
from code.simulation import Simulation, generate_profiles
from code.scenario import LLMScenario
sim = Simulation(scenario=LLMScenario(agent_num=50),
                 profiles=generate_profiles(50))
for s in sim.run(20): print(s)
sim.save_events('output')"

# RL training (needs live LLM for real signal; mock for pipeline debug)
.venv/bin/python -c "
from code import offline; offline.install()   # omit for live LLM
from code.rl_env import ElitePersuaderEnv
from code.scenario import LLMScenario
from code.trainer import TrainConfig, train, evaluate, evaluate_message_bank
env = ElitePersuaderEnv(scenario=LLMScenario(agent_num=15), n_periods=8)
policy, _ = train(env, TrainConfig(episodes=300))
print('baseline:', evaluate_message_bank(env))
print('learned: ', evaluate(env, policy))"

# RL smoke test with the mock backend
.venv/bin/python -m pytest code/tests/test_rl.py -q
```

LLM backend: local Ollama by default; override with `LLM_BASE_URL`,
`LLM_MODEL`, `LLM_API_KEY` environment variables.

## 7. Roadmap (not yet implemented)

### Stage 2 — Adversarial persuaders
A second seeded agent with an opposing stance and its own reward sign;
training via iterated best response against frozen snapshots or
simultaneous self-play. Extension points: `simulation.py` (multiple
seeds), `rl_env.py` (per-persuader reward decomposition — the event
schema already records `source_agent`).

### Stage 3 — Full population (true MARL)
One **shared policy conditioned on agent identity and own stance**
(parameter sharing) plays all roles at single-agent training cost;
per-agent rewards from `diffusion_events` ground-truth deltas. CTDE
methods (MAPPO/QMIX) only if coordination effects require them.

### Approach 3 — Direct language policy
Replace the strategy-vector action with token-level generation:
fine-tune the message-rendering LLM itself with PPO/GRPO (TRL/verl),
reward = same ground-truth stance shift. The env interface
(`reset`/`step`, reward plumbing) stays identical — swap
`RLPolicy` + `render_strategy_message` for the TRL trainer. Guard
against reward hacking (prompt-injection-style exploits of target
LLMs) with style constraints and out-of-distribution penalties; report
honestly if found.

### Research TODOs
- Re-run the paper's experiments with the fixed code and regenerate
  all tables/figures (previous numbers are invalid).
- Observer calibration: address the combined-evidence collapse with
  isotonic regression / temperature scaling on observer outputs.
- Human-data validation of reconstructed stance trajectories.
