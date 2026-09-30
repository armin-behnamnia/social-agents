"""Central configuration for the LLM-backed simulation.

All previously hardcoded values (model name, base URL, temperatures,
max_tokens, paths) live here so experiments are reproducible and the
LLM backend can be swapped without touching call sites.
"""

import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    base_url: str = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key: str = os.environ.get("LLM_API_KEY", "ollama")
    model: str = os.environ.get("LLM_MODEL", "llama3.1:latest")

    # Agent generation is sampled; the observer is an estimator and must
    # run near-deterministically.
    generation_temperature: float = 0.8
    inference_temperature: float = 0.0

    generation_max_tokens: int = 500
    inference_max_tokens: int = 64
    profile_max_tokens: int = 4000

    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    timeout_seconds: float = 120.0


@dataclass
class ScenarioConfig:
    """Replaces the Melodie scenario parameters referenced by agent.py."""

    agent_num: int = 50

    # Latent stance bounds
    inference_state_min: float = -1.0
    inference_state_max: float = 1.0

    # Stance transition: s' = s + influence * susceptibility * (s_src - s)
    stance_influence_rate: float = 0.4

    # |stance| threshold for qualitative labels
    stance_threshold: float = 0.5

    # Adoption threshold: an agent propagates once stance <= this value.
    # Must be 0.0 (or slightly above): with the linear transition rule a
    # positive-stance target can never cross a strongly negative
    # threshold, and it becomes ineligible for further influence once
    # its stance drops to <= 0, so adoption would never trigger.
    adoption_threshold: float = 0.0

    # Influence-probability weights
    alpha_profile: float = 0.25
    alpha_content: float = 0.25
    alpha_stance: float = 0.25
    alpha_degree: float = 0.25

    # Global RNG seed (None => entropy)
    seed: int | None = 42

    # Observer evidence sources enabled
    use_post_evidence: bool = True
    use_profile_evidence: bool = True

    # Event log output
    output_dir: str = "output"


GLOBAL_LLM_CONFIG = LLMConfig()
GLOBAL_SCENARIO = ScenarioConfig()
