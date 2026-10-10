"""Text embedding utilities with a deterministic local fallback.

Embeddings are served by an external OpenAI-compatible
`/v1/embeddings` API (see `EMBEDDING_*` in `.env`) — no local models
are run. When the API is not configured (or fails), a deterministic
hash-based bag-of-words fallback keeps tests, CI, and mock-mode RL
runnable offline. The two backends are NOT numerically compatible;
pick one per experiment.

An in-memory cache deduplicates repeated texts (RL episode resets
re-embed the same personas), and batching is applied when supported.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from typing import Sequence

import numpy as np

from .config import GLOBAL_EMBEDDING_CONFIG

logger = logging.getLogger(__name__)

_FALLBACK_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_model = None
_backend: str | None = None
_cache: dict[tuple[str, str], np.ndarray] = {}
_cache_lock = threading.Lock()


def _fallback_embedding(text: str) -> np.ndarray:
    vec = np.zeros(_FALLBACK_DIM, dtype=np.float32)
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.md5(token.encode()).digest()
        idx = int.from_bytes(digest[:4], "little") % _FALLBACK_DIM
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


class _ApiEmbedder:
    """OpenAI-compatible /v1/embeddings client with retry + cache."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._dim: int | None = None

    def _client(self):
        from openai import OpenAI

        return OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.timeout_seconds,
        )

    @property
    def dim(self) -> int:
        if self._dim is None:
            vec = self.embed_batch(["dimension probe"])[0]
            self._dim = int(vec.shape[0])
        return self._dim

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        cfg = self.cfg
        results: list[np.ndarray | None] = [None] * len(texts)
        uncached: list[int] = []

        if cfg.enable_cache:
            with _cache_lock:
                for i, t in enumerate(texts):
                    hit = _cache.get((cfg.model, t))
                    if hit is not None:
                        results[i] = hit
                    else:
                        uncached.append(i)
        else:
            uncached = list(range(len(texts)))

        if uncached:
            batch_texts = [texts[i] for i in uncached]
            vectors = self._request_embeddings(batch_texts)
            with _cache_lock:
                for i, vec in zip(uncached, vectors):
                    results[i] = vec
                    _cache[(cfg.model, texts[i])] = vec

        return [r if r is not None else _fallback_embedding(texts[i])
                for i, r in enumerate(results)]

    def _request_embeddings(self, batch: list[str]) -> list[np.ndarray]:
        cfg = self.cfg
        last_error: Exception | None = None
        for attempt in range(1, cfg.max_retries + 1):
            try:
                response = self._client().embeddings.create(
                    model=cfg.model,
                    input=[b if b.strip() else " " for b in batch],
                )
                vectors = []
                for item in response.data:
                    vec = np.asarray(item.embedding, dtype=np.float32)
                    norm = float(np.linalg.norm(vec))
                    if norm > 0:
                        vec = vec / norm
                    vectors.append(vec)
                if len(vectors) != len(batch):
                    raise ValueError(
                        f"embedding API returned {len(vectors)} of "
                        f"{len(batch)} vectors"
                    )
                return vectors
            except Exception as exc:  # noqa: BLE001 - transient API errors
                last_error = exc
                wait = cfg.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "embedding API attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt, cfg.max_retries, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"embedding API failed after {cfg.max_retries} attempts"
        ) from last_error


def _get_model():
    """Return the embedder, or False when the API is not configured."""
    global _model, _backend
    if _model is not None:
        return _model
    cfg = GLOBAL_EMBEDDING_CONFIG
    if cfg.has_credentials():
        embedder = _ApiEmbedder(cfg)
        try:
            embedder.dim  # probe: fails fast on bad URL/key/model
            _model = embedder
            _backend = f"api:{cfg.model}"
            return _model
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "embedding API probe failed (%s); "
                "falling back to hash embeddings", exc,
            )
    _backend = "hash-fallback"
    _model = False  # sentinel: fallback mode
    return _model


def text2embedding(text: str | Sequence[str]):
    """Embed a string (or list of strings) into a unit-norm vector.

    Returns a 1-D np.ndarray for a single string, or a 2-D array
    (n, d) for a sequence. Empty/None input yields a zero vector.
    """
    if isinstance(text, str):
        if not text.strip():
            return np.zeros(_embedding_dim(), dtype=np.float32)
        return _embed_batch([text])[0]
    texts = [t if isinstance(t, str) and t.strip() else "" for t in text]
    if not texts:
        return np.zeros((0, _embedding_dim()), dtype=np.float32)
    return _embed_batch(texts)


def _embedding_dim() -> int:
    model = _get_model()
    if model is False:
        return _FALLBACK_DIM
    return model.dim


def _embed_batch(texts: list[str]) -> np.ndarray:
    model = _get_model()
    if model is False:
        return np.stack([_fallback_embedding(t) for t in texts])
    vecs = model.embed_batch(texts)
    return np.asarray(vecs, dtype=np.float32)


def embedding_backend() -> str:
    _get_model()
    return _backend or "unknown"
