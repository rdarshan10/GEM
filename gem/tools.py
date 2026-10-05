"""GEM as agent tools — the same shape as Supermemory's (add_memory with save/forget, search_memory,
list_memories, scoped by container_tag), plus the part only GEM has: every write reports which
derived facts it invalidated, and get_stale / why expose them.

Use it three ways:
  - MCP server:            python -m gem.mcp_server            (see gem/mcp_server.py)
  - your own agent loop:   tools = GemTools(); client.messages.create(..., tools=TOOLS)
                           then result = tools.call(block.name, block.input)
  - OpenAI-style loop:     openai_tools() gives the same definitions in function-calling format

Storage (env GEM_STORE): "json" (default) — one file per container under GEM_HOME (~/.gem);
"falkor" — FalkorDB graph gem_<tag> at GEM_FALKOR_HOST:GEM_FALKOR_PORT; "memory" — not persisted.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict

from .memory import Memory

DEFAULT_TAG = "default"
FORGET_MIN_SCORE = 0.75     # forget-by-content needs a close match; otherwise it asks for an id

_TAG = {"type": "string", "description": "Memory space to use (letters, digits, _ and -). "
                                         "Defaults to 'default'."}

TOOLS = [
    {
        "name": "add_memory",
        "description": (
            "Save a fact to long-term memory, or forget one. Saving a fact that changes something "
            "already known updates it AND re-checks every fact derived from it: the result lists "
            "facts that were corrected ('revised') and facts that are no longer reliable "
            "('invalidated'). Tell the user about invalidated facts that matter. "
            "Use derived_from to record which stored facts (by id) this one was worked out from, "
            "so future changes cascade to it; if omitted, dependencies are inferred."),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact, as one self-contained sentence."},
                "action": {"type": "string", "enum": ["save", "forget"], "default": "save",
                           "description": "'forget' retracts a stored fact (give fact_id, or the "
                                          "fact's text in content) and invalidates what depended on it."},
                "fact_id": {"type": "string", "description": "Id of the fact to forget (action=forget)."},
                "derived_from": {"type": "array", "items": {"type": "string"},
                                 "description": "Ids of stored facts this fact depends on."},
                "container_tag": _TAG,
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_memory",
        "description": ("Find stored facts relevant to a query. 'results' holds facts that are still "
                        "valid; 'stale' holds matching facts that went out of date because something "
                        "they depended on changed — don't use their values, ask the user instead."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                "include_stale": {"type": "boolean", "default": False},
                "container_tag": _TAG,
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_memories",
        "description": "List all stored facts in a memory space, with their status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_stale": {"type": "boolean", "default": False},
                "container_tag": _TAG,
            },
        },
    },
    {
        "name": "get_stale",
        "description": ("List facts that are no longer reliable because something they depended on "
                        "changed. Reconfirm these with the user before acting on them."),
        "input_schema": {"type": "object", "properties": {"container_tag": _TAG}},
    },
    {
        "name": "why",
        "description": "Show the facts a stored fact was derived from (why it may have gone stale).",
        "input_schema": {
            "type": "object",
            "properties": {"fact_id": {"type": "string"}, "container_tag": _TAG},
            "required": ["fact_id"],
        },
    },
]


def openai_tools() -> list[dict]:
    """TOOLS in OpenAI / Groq function-calling format."""
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in TOOLS]


def _clean_tag(tag: str | None) -> str:
    tag = (tag or DEFAULT_TAG).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tag):
        raise ValueError("container_tag must be 1-64 letters, digits, '_' or '-'")
    return tag


def make_store(tag: str, kind: str | None = None):
    kind = (kind or os.environ.get("GEM_STORE", "json")).lower()
    if kind == "json":
        from .json_store import JsonStore
        home = os.environ.get("GEM_HOME", os.path.join("~", ".gem"))
        return JsonStore(os.path.join(home, f"{tag}.json"))
    if kind == "falkor":
        from .falkor_store import FalkorStore
        return FalkorStore(host=os.environ.get("GEM_FALKOR_HOST", "localhost"),
                           port=int(os.environ.get("GEM_FALKOR_PORT", "6379")), graph=f"gem_{tag}")
    if kind == "memory":
        return None                                   # Memory's default in-memory store
    raise ValueError(f"unknown GEM_STORE {kind!r} (use json, falkor or memory)")


def _fact(f) -> dict:
    d = asdict(f)
    if d.get("score") is None:
        d.pop("score", None)
    else:
        d["score"] = round(d["score"], 3)
    return d


class GemTools:
    """Dispatcher over one Memory per container tag. The LLM and embedder are shared across
    containers and created on first use (model loading is slow)."""

    def __init__(self, llm=None, embedder=None, store_kind: str | None = None):
        self._llm, self._embedder, self._kind = llm, embedder, store_kind
        self._mems: dict[str, Memory] = {}

    def memory(self, tag: str | None = None) -> Memory:
        tag = _clean_tag(tag)
        if tag not in self._mems:
            if self._llm is None:
                from .llm import make_llm
                self._llm = make_llm()
            if self._embedder is None:
                from .embed import default_embedder
                self._embedder = default_embedder()
            self._mems[tag] = Memory(llm=self._llm, embedder=self._embedder,
                                     store=make_store(tag, self._kind))
        return self._mems[tag]

    def call(self, name: str, args: dict | None = None) -> dict:
        """Run one tool. Returns a JSON-serialisable dict; errors come back as {"error": ...}
        so a model can read and recover from them instead of crashing the loop."""
        args = dict(args or {})
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(**args)
        except (KeyError, ValueError, TypeError) as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # --- tools -------------------------------------------------------------- #
    def _t_add_memory(self, content, action="save", fact_id=None, derived_from=None,
                      container_tag=None):
        m = self.memory(container_tag)
        if action == "forget":
            if not fact_id:
                hits = m.search(content, k=3)
                if not hits or hits[0].score < FORGET_MIN_SCORE:
                    return {"error": "no stored fact matches closely enough to forget; "
                                     "pass fact_id", "candidates": [_fact(h) for h in hits]}
                fact_id = hits[0].id
            forgotten = m.get(fact_id)
            if forgotten is None:
                raise KeyError(f"no fact {fact_id!r}")
            r = m.forget(fact_id)
            return {"action": "forget", "forgotten": _fact(m.get(fact_id)),
                    "invalidated": [_fact(f) for f in r.invalidated],
                    "revised": [_fact(f) for f in r.revised]}
        if action != "save":
            raise ValueError("action must be 'save' or 'forget'")
        r = m.add(content, derived_from=derived_from)
        return {"action": "save", "id": r.id,
                "revised": [_fact(f) for f in r.revised],
                "invalidated": [_fact(f) for f in r.invalidated]}

    def _t_search_memory(self, query, limit=5, include_stale=False, container_tag=None):
        k = max(1, min(int(limit), 50))
        hits = self.memory(container_tag).search(query, k=2 * k, include_stale=True)
        if include_stale:
            return {"results": [_fact(f) for f in hits[:k]]}
        valid = [f for f in hits if f.status == "ACTIVE" and not f.needs_review][:k]
        stale = [f for f in hits if f.status != "ACTIVE" or f.needs_review][:k]
        # stale matches are reported separately: the agent learns the fact exists but is out of date
        return {"results": [_fact(f) for f in valid], "stale": [_fact(f) for f in stale]}

    def _t_list_memories(self, include_stale=False, container_tag=None):
        return {"memories": [_fact(f) for f in self.memory(container_tag).facts(include_stale)]}

    def _t_get_stale(self, container_tag=None):
        return {"stale": [_fact(f) for f in self.memory(container_tag).stale]}

    def _t_why(self, fact_id, container_tag=None):
        m = self.memory(container_tag)
        if m.get(fact_id) is None:
            raise KeyError(f"no fact {fact_id!r}")
        parents = m._g.store.derived_from_targets(fact_id)
        return {"fact": _fact(m.get(fact_id)),
                "derived_from": [_fact(m.get(p.id)) for p in parents]}
