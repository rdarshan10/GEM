"""Backend-agnostic LLM client.

Talks to Ollama's REST API over HTTP (default http://localhost:11434), so the only
dependency is `requests` — no `ollama` Python package required. The gate is about
whether a *local* model can do the job, so the default backend is Ollama, but the
same interface could point at any OpenAI-compatible /chat endpoint later.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import requests


def _normalize_host(host: str) -> str:
    """Ollama's OLLAMA_HOST env var is often scheme-less (e.g. '127.0.0.1:11435').
    requests needs a scheme, so add one if missing."""
    host = host.strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


@dataclass
class LLMConfig:
    model: str = os.environ.get("GEM_MODEL", "gpt-oss:120b-cloud")
    host: str = _normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    temperature: float = 0.0          # deterministic: this is measurement, not generation
    num_ctx: int = 8192               # must hold a 4k-token chunk + prompt comfortably
    timeout: int = 600                # absorbs first-call model load (6.6GB into memory)
    think: bool = False               # reasoning models: extraction wants fast JSON, not a think trace
    keep_alive: str = "30m"           # keep the model resident so only the first call pays load cost

    def __post_init__(self):
        self.host = _normalize_host(self.host)


class OllamaClient:
    """Thin wrapper over Ollama's /api/chat. Deterministic by default."""

    def __init__(self, config: LLMConfig | None = None):
        self.cfg = config or LLMConfig()

    def is_up(self) -> bool:
        try:
            r = requests.get(f"{self.cfg.host}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def available_models(self) -> list[str]:
        r = requests.get(f"{self.cfg.host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def chat(self, system: str, user: str, *, json_mode: bool = False,
             temperature: float | None = None) -> str:
        """Single-turn chat. Returns the raw assistant string."""
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self.cfg.think,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"
        r = requests.post(
            f"{self.cfg.host}/api/chat", json=payload, timeout=self.cfg.timeout
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def chat_json(self, system: str, user: str, *, retries: int = 1,
                  temperature: float | None = None) -> dict:
        """Chat constrained to JSON output, parsed. Falls back to brace-extraction
        if the model wraps the object in prose. On a hard parse failure, retries at a
        higher temperature (determinism would just reproduce the same bad output).
        An explicit `temperature` (used for self-consistency voting) is honored on the
        first attempt instead of the deterministic default."""
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            temp = temperature if attempt == 0 else 0.4
            raw = self.chat(system, user, json_mode=True, temperature=temp)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                try:
                    return _salvage_json(raw)
                except ValueError as e:
                    last_err = e
        raise last_err  # exhausted retries


def _salvage_json(raw: str) -> dict:
    """Best-effort recovery when a local model emits stray text around the JSON."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from model output:\n{raw[:500]}")


import time as _time


@dataclass
class GroqConfig:
    model: str = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    api_key: str = os.environ.get("GROQ_API_KEY", "")
    base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.0
    timeout: int = 120
    min_interval: float = 8.0   # seconds between calls — respects 6K tok/min @ ~800 tok/call


class GroqClient:
    """OpenAI-compatible client for Groq. Same interface as OllamaClient (chat/chat_json), so
    it's a drop-in boundary model. Paced to stay under the free-tier rate limits and honors
    429 Retry-After. Used to test the accessible 8B tier (llama-3.1-8b-instant)."""

    def __init__(self, config: GroqConfig | None = None):
        self.cfg = config or GroqConfig()
        self._last = 0.0

    def is_up(self) -> bool:
        return bool(self.cfg.api_key)

    def available_models(self) -> list[str]:
        return [self.cfg.model]

    def _pace(self):
        dt = _time.time() - self._last
        if dt < self.cfg.min_interval:
            _time.sleep(self.cfg.min_interval - dt)
        self._last = _time.time()

    def chat(self, system: str, user: str, *, json_mode: bool = False,
             temperature: float | None = None) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.cfg.temperature if temperature is None else temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        for attempt in range(4):
            self._pace()
            r = requests.post(f"{self.cfg.base_url}/chat/completions", json=payload,
                              headers=headers, timeout=self.cfg.timeout)
            if r.status_code == 429:                      # rate limited -> honor Retry-After
                wait = float(r.headers.get("retry-after", 5))
                _time.sleep(wait + 0.5)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError("Groq: exhausted retries on 429")

    def chat_json(self, system: str, user: str, *, retries: int = 1,
                  temperature: float | None = None) -> dict:
        last_err = None
        for attempt in range(retries + 1):
            temp = temperature if attempt == 0 else 0.4
            raw = self.chat(system, user, json_mode=True, temperature=temp)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                try:
                    return _salvage_json(raw)
                except ValueError as e:
                    last_err = e
        raise last_err


def make_llm():
    """Factory: GEM_LLM=groq -> GroqClient (llama-3.1-8b-instant); else OllamaClient."""
    if os.environ.get("GEM_LLM", "").lower() == "groq":
        return GroqClient()
    return OllamaClient(LLMConfig())
