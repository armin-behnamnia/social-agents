# Codebase Graphify — Architecture Diagrams

Visual maps of the agentic information-diffusion system and its RL
training pipeline. All diagrams are [Mermaid](https://mermaid.js.org):
they render on GitHub/GitLab and in VS Code (with a mermaid plugin).

---

## 1. Module dependency graph

Who imports whom (arrows point from importer to imported module).

```mermaid
graph TD
    subgraph CONFIG["⚙ config layer"]
        CONFIG_PY["config.py<br/>(LLM + embedding + scenario config,<br/>.env secret loading)"]
    end

    subgraph LLM_LAYER["🗣 LLM layer"]
        LLM_MOD["LLM_module.py<br/>call_llm: retry · cache · parse<br/>Prompts 1–5 · strategy renderer"]
        OFFLINE["offline.py<br/>deterministic mock LLM<br/>(patches LLM_module)"]
    end

    subgraph EMB["🧮 embeddings"]
        TOOLS["tools.py<br/>embedding API (OpenAI-compatible)<br/>or hash fallback"]
    end

    subgraph POPULATION["👥 population"]
        BASE["base_agent.py<br/>NetworkAgent (no Melodie)"]
        AGENT["agent.py<br/>LLMSocialAgent<br/>acceptance gate · stance transition<br/>adoption · events"]
        NETWORKX["network.py<br/>WS / BA / ER / complete<br/>degree · betweenness"]
        PROFILES["profiles.py<br/>synthetic or LLM personas"]
        SCENARIO["scenario.py<br/>LLMScenario dataclass"]
    end

    subgraph ENGINE["🔬 engine"]
        SIM["simulation.py<br/>period-based diffusion loop<br/>event logging"]
    end

    subgraph RL["🤖 RL layer"]
        STRATEGY["strategy.py<br/>180 strategies<br/>(appeal × tone × emphasis × intensity)"]
        RL_ENV["rl_env.py<br/>ElitePersuaderEnv<br/>learned target-node selection<br/>+ strategy action"]
        ENV_LEGACY["env.py<br/>LatentStanceDiffusionEnv<br/>(message-bank baseline env)"]
        TRAINER_CPU["trainer.py<br/>numpy REINFORCE (CPU)"]
        TRAINER_GPU["torch_trainer.py<br/>PPO actor-critic (GPU)<br/>action masking · GAE"]
        CLI["train_persuader.py<br/>CLI: mock / live modes"]
    end

    subgraph TESTS["✅ tests"]
        TESTS_CODE["tests/test_codebase.py"]
        TESTS_RL["tests/test_rl.py"]
        TESTS_GPU["tests/test_gpu_trainer.py"]
    end

    CONFIG_PY --> LLM_MOD
    OFFLINE -. patches .-> LLM_MOD

    LLM_MOD --> SIM
    AGENT --> LLM_MOD
    AGENT --> TOOLS
    AGENT --> BASE
    AGENT --> SCENARIO
    SIM --> AGENT
    SIM --> NETWORKX
    SIM --> SCENARIO
    SIM --> PROFILES
    PROFILES --> LLM_MOD
    RL_ENV --> SIM
    RL_ENV --> STRATEGY
    RL_ENV --> LLM_MOD
    RL_ENV --> PROFILES
    RL_ENV --> SCENARIO
    ENV_LEGACY --> SIM
    TRAINER_CPU --> RL_ENV
    TRAINER_CPU --> STRATEGY
    TRAINER_GPU --> RL_ENV
    TRAINER_GPU --> STRATEGY
    CLI --> TRAINER_GPU
    CLI --> RL_ENV
    CLI --> CONFIG_PY

    TESTS_CODE --> AGENT
    TESTS_CODE --> SIM
    TESTS_CODE --> ENV_LEGACY
    TESTS_RL --> RL_ENV
    TESTS_RL --> TRAINER_CPU
    TESTS_GPU --> TRAINER_GPU
    TESTS_GPU --> RL_ENV
```

---

## 2. One diffusion period (paper Algorithm 1, as implemented)

```mermaid
flowchart TD
    START(("period t")) --> SOURCES["snapshot active sources<br/>preference_state=1 and post≠∅"]
    SOURCES --> TARGETS["for each source i:<br/>targets = neighbors j with<br/>preference_state=0 ∧ stance_j > 0"]
    TARGETS --> GATE1{"source stance ≤<br/>adoption threshold?"}
    GATE1 -- no --> SKIP["skip interaction"]
    GATE1 -- yes --> GATE2{"rand() &lt; p_influence?<br/>(stochastic gate)"}
    GATE2 -- reject --> EVT1["log diffusion event<br/>(behavior=reject)"]
    GATE2 -- accept --> TRANS["latent stance transition<br/>s_j ← s_j + p·β·(s_i − s_j)"]
    TRANS --> INTERNAL["LLM Prompt 4:<br/>internalized message"]
    INTERNAL --> ADOPT{"s_j ≤ adoption<br/>threshold?"}
    ADOPT -- no --> EVT2["log diffusion event<br/>(behavior=accept, Δs)"]
    ADOPT -- yes --> GEN["LLM Prompts 2+3:<br/>new post + new profile<br/>observer inference (Prompt 5)<br/>log message event"]
    EVT2 --> NEXT["next target"]
    GEN --> NEXT
    EVT1 --> NEXT
    SKIP --> NEXT
    NEXT --> DONE{"all interactions<br/>processed?"}
    DONE -- no --> TARGETS
    DONE -- yes --> END(("period t+1"))
```

---

## 3. RL training loop (GPU PPO + LLM renderer + simulator)

```mermaid
flowchart LR
    subgraph EPISODE["one episode (seeded)"]
        RESET["env.reset(seed)<br/>new network + personas<br/>+ initial stances +0.65"] --> OBS
        OBS["observation (13-d)<br/>profile-emb pool · centralities<br/>exposure · stance proxy · mean stance"]
        OBS --> POLICY
        POLICY["policy π (GPU)<br/>MLP actor-critic"]
        MASK["action mask<br/>(reachable targets × 180 strategies)"] --> POLICY
        POLICY --> ACTION["a = (target node,<br/>strategy spec)"]
        ACTION --> RENDER["frozen LLM renders<br/>persuasion message<br/>(≤50 words, temp 0)"]
        RENDER --> DELIVER["persuader (agent 0) posts<br/>message to chosen target"]
        DELIVER --> PERIOD["simulate 1 diffusion period<br/>(Algorithm 1, CPU)"]
        PERIOD --> REWARD["reward = Σ −Δs_target<br/>+ acceptance bonus<br/>+ cascade λ·(mean stance drop)"]
        REWARD --> OBS
    end

    REWARD --> BUFFER["rollout buffer<br/>(obs, action, reward, mask)"]
    BUFFER --> GAE["GAE(γ, λ)<br/>advantages + returns"]
    GAE --> PPO["PPO update (GPU)<br/>clipped surrogate · entropy<br/>value regression"]
    PPO --> POLICY
```

---

## 4. Secrets & external services

No local LLMs or embedding models are run — everything goes through
external OpenAI-compatible APIs configured in the gitignored `.env`.

```mermaid
flowchart TD
    ENVFILE[".env (GITIGNORED)<br/>LLM_BASE_URL / LLM_API_KEY / LLM_MODEL<br/>EMBEDDING_* (optional, defaults to LLM_*)"] --> LOAD["config.py:_load_dotenv()<br/>(env vars win over file)"]
    LOAD --> CFG["GLOBAL_LLM_CONFIG"]
    LOAD --> ECFG["GLOBAL_EMBEDDING_CONFIG"]
    CFG --> CALL["LLM_module.call_llm()"]
    CALL -->|chat/completions| REMOTE["remote LLM API<br/>(OpenAI / Together / vLLM / ...)"]
    CALL --> CACHE[(".llm_cache/<br/>on-disk response cache")]
    ECFG --> TOOLS["tools.py text2embedding()"]
    TOOLS -->|embeddings API| EREMOTE["remote embedding API"]
    TOOLS -. "no credentials:<br/>hash fallback (offline)" .-> FALLBACK["deterministic 256-d<br/>hash embedding"]
    EXAMPLE[".env.example<br/>(committed placeholder)"] -. "cp to .env,<br/>fill in keys" .-> ENVFILE
```

---

## 5. Module responsibility table

| Module | Responsibility | Key classes/functions |
|---|---|---|
| `config.py` | LLM + embedding + scenario config, `.env` secret loading | `LLMConfig`, `EmbeddingConfig` |
| `LLM_module.py` | LLM calls, retries, cache, prompt bank | `call_llm`, `render_strategy_message`, Prompts 1–5 |
| `tools.py` | embeddings via external API or hash fallback | `text2embedding` |
| `base_agent.py` | minimal agent base (Melodie-free) | `NetworkAgent` |
| `agent.py` | stance dynamics, acceptance gate, adoption, events | `LLMSocialAgent.infection_impact` |
| `network.py` | topology + centralities | `build_network` |
| `profiles.py` | persona generation (synthetic or LLM) | `generate_synthetic_profiles` |
| `scenario.py` | experiment parameters | `LLMScenario` |
| `simulation.py` | diffusion engine | `Simulation`, `generate_profiles` |
| `strategy.py` | 180-dim persuasion strategy space | `StrategySpec`, `index_to_strategy` |
| `rl_env.py` | persuader env: learned target + strategy actions | `ElitePersuaderEnv` |
| `env.py` | legacy message-bank env | `LatentStanceDiffusionEnv` |
| `trainer.py` | CPU REINFORCE baseline | `RLPolicy`, `train` |
| `torch_trainer.py` | GPU PPO (action masking, GAE) | `PPOTrainer`, `ActorCritic` |
| `train_persuader.py` | training CLI (mock/live) | `main` |
| `offline.py` | mock LLM backend | `install` |
