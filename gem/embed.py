"""Similarity layer for dedup + neighbor candidate selection.

Per the MVP: plain numpy cosine, no FAISS. Pluggable backend:
  - LexicalEmbedder  — zero-dependency hashed bag-of-words; runs immediately.
  - STEmbedder       — sentence-transformers all-MiniLM-L6-v2 (lazy import); the upgrade
                       the plan calls for, swap in when torch is installed.

Both expose embed(text) -> np.ndarray. `search` ranks store nodes by cosine to a query.
For the small MVP graphs, neighbor selection can also just take all nodes; embeddings
matter mainly for dedup of near-identical restatements.
"""

from __future__ import annotations

import re

import numpy as np

from .store import MemoryStore, Node

_TOKEN = re.compile(r"[a-z0-9]+")


class LexicalEmbedder:
    """Hashed bag-of-words -> fixed-dim vector. Cheap, dependency-free, good enough to
    catch near-duplicate restatements. Not semantic — that's what STEmbedder is for."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            vec[hash(tok) % self.dim] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n else vec


class STEmbedder:
    """sentence-transformers backend. Lazy import so the package works without torch."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        v = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)


_DEFAULT_EMB = None


def default_embedder():
    """Prefer real semantic embeddings (STEmbedder) — they matter for conflict-detection
    retrieval (lexical can't connect e.g. 'got a raise' to 'salary', so the ingest conflict
    check misses it). Fall back to the zero-dependency LexicalEmbedder only if
    sentence-transformers isn't installed, so the package still runs out of the box.
    Cached as a singleton: the MiniLM model loads once and is reused across GEM instances
    (embedders are stateless), so a per-scenario `GEM()` doesn't reload it each time."""
    global _DEFAULT_EMB
    if _DEFAULT_EMB is None:
        try:
            _DEFAULT_EMB = STEmbedder()
        except Exception:
            import sys
            print(
                "[GEM] sentence-transformers is not installed; falling back to LexicalEmbedder.\n"
                "      WARNING: this affects correctness, not only retrieval quality. Conflict "
                "detection relies on semantic\n      similarity at ingest. The lexical fallback "
                "matches only on shared surface tokens, so a new fact that\n      supersedes a "
                "stored one without overlapping vocabulary may not be detected as a conflict; the "
                "cascade\n      will not fire and the superseded fact will be retained as valid, "
                "with no error raised. Install\n      sentence-transformers, or pass an explicit "
                "embedder= to GEM(), for reliable conflict detection. The\n      lexical fallback "
                "is intended for dependency-free smoke tests only.",
                file=sys.stderr,
            )
            _DEFAULT_EMB = LexicalEmbedder()
    return _DEFAULT_EMB


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def search(store: MemoryStore, query_emb: np.ndarray, k: int = 5) -> list[tuple[Node, float]]:
    """Top-k active nodes by cosine similarity to query_emb."""
    scored = [
        (n, cosine(query_emb, n.embedding))
        for n in store.active_nodes()
        if n.embedding is not None
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]
