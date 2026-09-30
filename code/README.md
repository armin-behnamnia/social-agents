# Agentic Information Diffusion — Latent Stance Dynamics (refactored)

Simulation of information diffusion in a social network of LLM-based
agents, where each agent carries a latent, evolving stance in [-1, 1].
An external observer LLM tries to reconstruct per-interaction stance
shifts from observable evidence (generated posts, profile updates).

## Layout

```
code/
├── config.py        # LLM + scenario configuration (was hardcoded)
├── LLM_module.py    # LLM layer: retries, cache, numeric parsing
├── tools.py         # embeddings (sentence-transformers or fallback)
├── base_agent.py    # minimal NetworkAgent stand-in (no Melodie dep)
├── agent.py         # LLMSocialAgent: stance transitions, events
├── network.py       # topology builders + centralities
├── scenario.py      # scenario dataclass
├── simulation.py    # engine: init, seeding, diffusion loop
├── env.py           # Gymnasium wrapper (RL entry point)
├── tests/           # pytest suite (bugfix regression tests)
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r code/requirements.txt
```

LLM backend defaults to a local Ollama server (`llama3.1:latest`);
override with env vars `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`.

## Run

```bash
# tests (no LLM required — mocks)
.venv/bin/python -m pytest code/tests -q

# full simulation (requires a live LLM backend)
.venv/bin/python -c "
from code.simulation import Simulation, generate_profiles
from code.scenario import LLMScenario
profiles = generate_profiles(50)
sim = Simulation(scenario=LLMScenario(agent_num=50), profiles=profiles)
for s in sim.run(20): print(s)
sim.save_events('output')
"

# RL environment smoke test
.venv/bin/python -c "
from code.env import LatentStanceDiffusionEnv
from code.scenario import LLMScenario
env = LatentStanceDiffusionEnv(scenario=LLMScenario(agent_num=10), n_periods=5)
obs, _ = env.reset(seed=1)
done = False
while not done:
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    done = term or trunc
print(info)
"
```

## Key behavioral notes

- Acceptance of a message is stochastic (Algorithm 1 of the paper);
  controlled by `LLMScenario.rng_seed` for reproducibility.
- Observer inference runs at temperature 0; generation at 0.8.
- LLM replies are cached in `.llm_cache/` — delete to force re-query.
- Embeddings use sentence-transformers if installed, otherwise a
  deterministic hash fallback (not numerically compatible with each
  other; pick one per experiment).
