"""Shared test fixtures: deterministic mock LLM + embedder so the cascade LOGIC is testable
without any network call. The LLM-judged paths get scripted responses; everything else is
exact."""

from __future__ import annotations

import hashlib
import numpy as np
import pytest


class FakeLLM:
    """Scriptable stand-in for OllamaClient. `json_fn(system, user) -> dict` drives chat_json
    (classify/derive_links); `text_fn` drives chat. Counts calls so cost tests can assert."""

    def __init__(self, json_fn=None, text_fn=None):
        self._json = json_fn or (lambda s, u: {"label": "UNRELATED", "revised_content": None})
        self._text = text_fn or (lambda s, u: "")
        self.json_calls = 0
        self.chat_calls = 0

    def is_up(self):
        return True

    def chat_json(self, system, user, *, retries=1, temperature=None):
        self.json_calls += 1
        return self._json(system, user)

    def chat(self, system, user, *, json_mode=False, temperature=None):
        self.chat_calls += 1
        return self._text(system, user)


class FakeEmbedder:
    """Deterministic embedder (md5-based, no hash-seed randomness). Pass `vectors` to pin exact
    vectors for specific texts when a test needs controlled cosine similarities."""

    def __init__(self, vectors=None):
        self.vectors = vectors or {}

    def embed(self, text: str) -> np.ndarray:
        for key, vec in self.vectors.items():
            if key in text:
                return np.asarray(vec, dtype=np.float32)
        h = hashlib.md5(text.encode("utf-8")).digest()
        return np.frombuffer(h, dtype=np.uint8).astype(np.float32)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def classify_json(label, revised=None):
    return {"label": label, "revised_content": revised, "reasoning": ""}
