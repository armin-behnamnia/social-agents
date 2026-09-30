"""Text embedding utilities with a deterministic local fallback.

The original code imported `source.tools.text2embedding`, which was never
provided. This module offers:
- `text2embedding`: sentence-transformers if available, otherwise a
  deterministic hash-based bag-of-words fallback so the simulation,
  tests, and analysis run without heavyweight dependencies or network
  access. The two backends are NOT numerically compatible; pick one per
  experiment via config.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_FALLBACK_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_model = None
_backend: str | None = None


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


def _get_model():
    global _model, _backend
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _backend = "sentence-transformers/all-MiniLM-L6-v2"
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "sentence-transformers unavailable (%s); "
            "using deterministic hash-based fallback embeddings "
            "(NOT compatible with the MiniLM backend)",
            exc,
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
    return int(model.get_sentence_embedding_dimension())


def _embed_batch(texts: list[str]) -> np.ndarray:
    model = _get_model()
    if model is False:
        return np.stack([_fallback_embedding(t) for t in texts])
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def embedding_backend() -> str:
    _get_model()
    return _backend or "unknown"
