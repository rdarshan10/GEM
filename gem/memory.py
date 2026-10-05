"""Public library API — the 6-line experience.

GEM's engine is powerful but low-level (two-pass ingest, typed edges, cascade internals).
This module is the thin, stable facade a user actually imports. The whole pitch fits in one
screen:

    from gem import Memory

    m = Memory()
    auth = m.add("Auth uses JWT tokens")
    m.add("Tests mock the JWT verifier", derived_from=[auth])
    m.add("CI requires the JWT_SECRET env var", derived_from=[auth])

    result = m.add("We migrated auth from JWT to session cookies")
    print(result.invalidated)     # -> the two derived facts, now flagged stale

    m.search("how do tests handle auth?")   # returns only ACTIVE facts (stale ones excluded)

The one thing this buys over a flat vector memory: when `auth` changes, the facts DERIVED
from it go stale automatically — even though "Tests mock the JWT verifier" is not textually
similar to "we migrated to session cookies", so a similarity-only memory would never re-examine
it. That is the entire product, exposed in two methods (`add`, `search`).
"""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass, field

from . import prompts
from .engine import GEM, GEMConfig
from .embed import cosine
from .store import Status


@dataclass
class Fact:
    """A memory fact as seen by library users (no engine internals leak)."""
    id: str
    content: str
    status: str                       # "ACTIVE" | "STALE" | "SUPERSEDED"
    needs_review: bool = False
    confidence: float = 1.0
    score: float | None = None        # similarity to the query (search results only)


@dataclass
class AddResult:
    """What an `add()` did:
      id          — the new fact's id.
      invalidated — existing facts now STALE/SUPERSEDED (or flagged needs_review) because
                    they depended, transitively, on something this change altered. Do not
                    trust these for their value until reconfirmed.
      revised     — existing facts corrected IN PLACE (still ACTIVE, new content) because
                    the change directly updated them.
    `invalidated` is the one the cascade earns: dependents that flat memory would miss."""
    id: str
    invalidated: list[Fact] = field(default_factory=list)
    revised: list[Fact] = field(default_factory=list)

    def __bool__(self) -> bool:        # truthy if the write changed anything downstream
        return bool(self.invalidated or self.revised)


def _to_fact(n) -> Fact:
    return Fact(id=n.id, content=n.content, status=n.status.value,
                needs_review=bool(n.meta.get("needs_review")), confidence=n.confidence)


class Memory:
    """Dependency-aware memory. A drop-in layer for agents that ACT on derived facts.

    Two methods carry the API:
      add(content, derived_from=None) -> AddResult
          Store a fact. If it conflicts with existing memory, resolve it AND cascade the
          consequences down DERIVED_FROM edges, returning what went stale. `derived_from`
          pins dependencies explicitly; if omitted, they are inferred.
      search(query, k=5, include_stale=False) -> list[Fact]
          Retrieve relevant facts. Stale/superseded facts are excluded by default — that
          exclusion is the staleness fix made visible at read time.

    `cascade=False` reduces this to a flat memory (resolve direct conflicts, never
    propagate) — useful as the honest A/B baseline. `conservative=True` downgrades
    destructive invalidations to recoverable STALE+needs_review (recommended when a weaker
    model drives the cascade)."""

    def __init__(self, llm=None, embedder=None, store=None, *,
                 cascade: bool = True, conservative: bool = False):
        cfg = GEMConfig(cascade_enabled=cascade, conservative_invalidation=conservative)
        self._g = GEM(llm=llm, embedder=embedder, store=store, config=cfg)
        self._profile_cache: tuple[str, dict] | None = None   # (facts signature, profile)

    # --- write -------------------------------------------------------------- #
    def add(self, content: str, derived_from: list[str] | None = None) -> AddResult:
        """Store a fact; if it conflicts with memory, resolve it and cascade the consequences.

        `derived_from` — ids this fact depends on. PIN these for correctness-critical facts:
        explicit edges are exact. If omitted, GEM INFERS the dependencies (an extra LLM pass).
        Inference is convenient but best-effort — measured ~85–88% recall / ~75–84% precision
        across domains (small N; see runs/diag_derive_*.txt), so a minority of cascades may be
        missed or spurious. Rule of thumb: pin what you can't afford to get wrong, infer the rest.
        """
        with self._write():
            before = self._snapshot()
            node = self._g.ingest(content, parents=derived_from)
            return self._changes(node.id, before, skip=node.id)

    def forget(self, fact_id: str) -> AddResult:
        """Retract a fact and cascade: everything derived from it is re-checked and goes stale
        if it depended on it. Returns the forgotten fact's id and what the cascade changed."""
        with self._write():
            node = self._g.store.get(fact_id)
            if node is None:
                raise KeyError(fact_id)
            before = self._snapshot()
            self._g.retract(node)
            return self._changes(fact_id, before, skip=fact_id)

    def _snapshot(self) -> dict:
        return {n.id: (n.status, n.content) for n in self._g.store.all_nodes()}

    def _changes(self, fact_id: str, before: dict, *, skip: str) -> AddResult:
        invalidated, revised = [], []
        for n in self._g.store.all_nodes():
            if n.id == skip:
                continue
            prev = before.get(n.id)
            if not prev or (prev[0] == n.status and prev[1] == n.content):
                continue                      # untouched
            f = _to_fact(n)
            if n.status != Status.ACTIVE or n.meta.get("needs_review"):
                invalidated.append(f)         # no longer trustworthy
            else:
                revised.append(f)             # corrected in place, still usable
        return AddResult(id=fact_id, invalidated=invalidated, revised=revised)

    # persistent stores that can be shared between processes (JsonStore) expose transaction() for
    # writes (lock + catch up + save) and refresh() for reads; other stores need neither
    def _write(self):
        tx = getattr(self._g.store, "transaction", None)
        return tx() if tx else nullcontext()

    def _read(self) -> None:
        refresh = getattr(self._g.store, "refresh", None)
        if refresh:
            refresh()

    def load(self, content: str, derived_from: list[str] | None = None) -> str:
        """Bulk-load a fact you already trust (skips the conflict scan). For seeding a known,
        non-conflicting initial memory fast — not for new observations."""
        with self._write():
            return self._g.ingest(content, parents=(derived_from or []), check_conflicts=False).id

    # --- read --------------------------------------------------------------- #
    def search(self, query: str, k: int = 5, include_stale: bool = False) -> list[Fact]:
        self._read()
        q = self._g.embedder.embed(query)
        scored = [(n, cosine(q, n.embedding)) for n in self._g.store.all_nodes()
                  if n.embedding is not None
                  and (include_stale or n.status == Status.ACTIVE)]
        scored.sort(key=lambda t: t[1], reverse=True)
        out = []
        for n, sim in scored[:k]:
            f = _to_fact(n)
            f.score = float(sim)
            out.append(f)
        return out

    def get(self, fact_id: str) -> Fact | None:
        self._read()
        n = self._g.store.get(fact_id)
        return _to_fact(n) if n else None

    @property
    def stale(self) -> list[Fact]:
        self._read()
        return [_to_fact(n) for n in self._g.store.all_nodes()
                if n.status != Status.ACTIVE or n.meta.get("needs_review")]

    def facts(self, include_stale: bool = True) -> list[Fact]:
        self._read()
        return [_to_fact(n) for n in self._g.store.all_nodes()
                if include_stale or n.status == Status.ACTIVE]

    def why(self, fact_id: str) -> list[str]:
        """The DERIVED_FROM parents of a fact — 'this is stale because it depended on …'."""
        self._read()
        return [p.content for p in self._g.store.derived_from_targets(fact_id)]

    def profile(self, max_facts: int = 300) -> dict:
        """A short summary of what memory currently holds (Supermemory's get_profile), built by the
        LLM from ACTIVE facts only — stale facts are listed separately as things to reconfirm,
        never folded into the summary. Cached until the set of active facts changes."""
        self._read()
        nodes = self._g.store.all_nodes()
        active = [n for n in nodes if n.status == Status.ACTIVE and not n.meta.get("needs_review")]
        active = active[-max_facts:]                       # the most recent ones if memory is large
        reconfirm = [_to_fact(n) for n in nodes
                     if n.status == Status.STALE or n.meta.get("needs_review")]
        sig = hashlib.sha1("\n".join(f"{n.id}\t{n.content}" for n in active).encode()).hexdigest()
        if self._profile_cache and self._profile_cache[0] == sig:
            summary = self._profile_cache[1]
        else:
            summary = ""
            if active:
                summary = self._g.llm.chat(
                    prompts.PROFILE_SYSTEM,
                    prompts.PROFILE_USER.format(facts="\n".join(f"- {n.content}" for n in active)),
                ).strip()
            self._profile_cache = (sig, summary)
        return {"summary": summary, "fact_count": len(active), "reconfirm": reconfirm}
