# Agentic Information Diffusion — Latent Stance Dynamics (refactored)

Simulation of information diffusion in a social network of LLM-based
agents, where each agent carries a latent, evolving stance in [-1, 1].
An external observer LLM tries to reconstruct per-interaction stance
shifts from observable evidence (generated posts, profile updates).

## Layout

```text
code/
├── config.py         # LLM + embedding + scenario config (.env secrets)
├── LLM_module.py     # LLM layer: retries, cache, numeric parsing
├── tools.py          # embeddings via external API (hash fallback)
├── base_agent.py     # minimal NetworkAgent stand-in (no Melodie dep)
├── agent.py          # LLMSocialAgent: stance transitions, events
├── network.py        # topology builders + centralities
├── profiles.py       # synthetic or LLM persona generation
├── scenario.py       # scenario dataclass
├── simulation.py     # engine: init, seeding, diffusion loop
├── strategy.py       # 180-strategy persuasion space
├── rl_env.py         # ElitePersuaderEnv (learned target + strategy)
├── env.py            # Gymnasium wrapper (message-bank baseline env)
├── trainer.py        # numpy REINFORCE trainer (CPU baseline)
├── torch_trainer.py  # GPU PPO trainer (action masking, GAE)
├── train_persuader.py# training CLI (mock / live modes)
├── offline.py        # deterministic mock LLM backend
├── tests/            # pytest suite (46 tests)
└── requirements.txt
```

## Setup

```bash
# conda env with torch (GPU) is expected at /opt/conda
python -m pip install -r code/requirements.txt
```

LLM + embeddings are **external APIs** (no local models). Configure via
`.env` (never committed):

```bash
cp .env.example .env
# then edit .env:
#   LLM_BASE_URL=...      # OpenAI-compatible endpoint
#   LLM_API_KEY=...       # your key
#   LLM_MODEL=...         # chat model
#   EMBEDDING_MODEL=...   # optional, defaults to text-embedding-3-small
```

## Run

```bash
# tests (no API needed — mock backend)
python -m pytest code/tests -q

# GPU PPO training — pipeline validation (mock LLM, no API)
python -m code.train_persuader --mode mock --iterations 120

# GPU PPO training — live LLM (requires .env credentials)
python -m code.train_persuader --mode live --iterations 200 --agent-num 50

# full simulation (requires live LLM backend)
python -c "
from code.simulation import Simulation, generate_profiles
from code.scenario import LLMScenario
profiles = generate_profiles(50)
sim = Simulation(scenario=LLMScenario(agent_num=50), profiles=profiles)
for s in sim.run(20): print(s)
sim.save_events('output')
"
```

Outputs land in `output/ppo/<run>/`: `policy.pt` (checkpoint),
`training_log.json` (per-iteration metrics), `results.json` (final
learned vs random vs heuristic baseline comparison).

## Key behavioral notes

- Acceptance of a message is stochastic (Algorithm 1 of the paper);
  controlled by `LLMScenario.rng_seed` for reproducibility.
- Observer inference runs at temperature 0; generation at 0.8.
- LLM replies are cached in `.llm_cache/` — delete to force re-query.
- Embeddings use sentence-transformers if installed, otherwise a
  deterministic hash fallback (not numerically compatible with each
  other; pick one per experiment).
