"""Central configuration for the LLM-backed simulation.

All previously hardcoded values (model name, base URL, temperatures,
max_tokens, paths) live here so experiments are reproducible and the
LLM backend can be swapped without touching call sites.

Secrets (LLM_API_KEY, LLM_BASE_URL) are loaded from environment
variables, optionally populated from a gitignored `.env` file in the
project root (see `.env.example`). Secrets must never be committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader: KEY=VALUE lines; existing env vars win.

    Avoids a python-dotenv dependency; comments (#) and blank lines are
    skipped. Only used for local convenience — production secrets should
    come from real environment variables.
    """
    env_path = path or _PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


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

    # Max concurrent workers for batched LLM calls (live mode).
    max_workers: int = int(os.environ.get("LLM_MAX_WORKERS", "8"))

    def has_credentials(self) -> bool:
        """True when a non-placeholder API key is configured."""
        key = (self.api_key or "").strip()
        return bool(key) and "REPLACE_ME" not in key and key != "ollama"


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


@dataclass
class EmbeddingConfig:
    """Embedding API (OpenAI-compatible /v1/embeddings endpoint).

    When unset/placeholder, `tools.text2embedding` falls back to a
    deterministic hash embedding so mock-mode training and tests run
    offline.
    """

    base_url: str = os.environ.get(
        "EMBEDDING_BASE_URL", os.environ.get("LLM_BASE_URL", "")
    )
    api_key: str = os.environ.get(
        "EMBEDDING_API_KEY", os.environ.get("LLM_API_KEY", "")
    )
    model: str = os.environ.get(
        "EMBEDDING_MODEL", "text-embedding-3-small"
    )
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    # In-memory cache: (model, text) -> vector (RL resets re-embed the
    # same personas every episode; caching removes the API cost).
    enable_cache: bool = True

    def has_credentials(self) -> bool:
        key = (self.api_key or "").strip()
        url = (self.base_url or "").strip()
        return (
            bool(key)
            and "REPLACE_ME" not in key
            and bool(url)
            and "example.com" not in url
        )


GLOBAL_LLM_CONFIG = LLMConfig()
GLOBAL_SCENARIO = ScenarioConfig()
GLOBAL_EMBEDDING_CONFIG = EmbeddingConfig()
