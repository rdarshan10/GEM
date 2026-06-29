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

from dataclasses import dataclass, field

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

    # --- write -------------------------------------------------------------- #
    def add(self, content: str, derived_from: list[str] | None = None) -> AddResult:
        """Store a fact; if it conflicts with memory, resolve it and cascade the consequences.

        `derived_from` — ids this fact depends on. PIN these for correctness-critical facts:
        explicit edges are exact. If omitted, GEM INFERS the dependencies (an extra LLM pass).
        Inference is convenient but best-effort — measured ~85–88% recall / ~75–84% precision
        across domains (small N; see runs/diag_derive_*.txt), so a minority of cascades may be
        missed or spurious. Rule of thumb: pin what you can't afford to get wrong, infer the rest.
        """
        before = {n.id: (n.status, n.content) for n in self._g.store.all_nodes()}
        node = self._g.ingest(content, parents=derived_from)
        invalidated, revised = [], []
        for n in self._g.store.all_nodes():
            if n.id == node.id:
                continue
            prev = before.get(n.id)
            if not prev or (prev[0] == n.status and prev[1] == n.content):
                continue                      # untouched
            f = _to_fact(n)
            if n.status != Status.ACTIVE or n.meta.get("needs_review"):
                invalidated.append(f)         # no longer trustworthy
            else:
                revised.append(f)             # corrected in place, still usable
        return AddResult(id=node.id, invalidated=invalidated, revised=revised)

    def load(self, content: str, derived_from: list[str] | None = None) -> str:
        """Bulk-load a fact you already trust (skips the conflict scan). For seeding a known,
        non-conflicting initial memory fast — not for new observations."""
        return self._g.ingest(content, parents=(derived_from or []),
                              check_conflicts=False).id

    # --- read --------------------------------------------------------------- #
    def search(self, query: str, k: int = 5, include_stale: bool = False) -> list[Fact]:
        q = self._g.embedder.embed(query)
        scored = [(n, cosine(q, n.embedding)) for n in self._g.store.all_nodes()
                  if n.embedding is not None
                  and (include_stale or n.status == Status.ACTIVE)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [_to_fact(n) for n, _ in scored[:k]]

    def get(self, fact_id: str) -> Fact | None:
        n = self._g.store.get(fact_id)
        return _to_fact(n) if n else None

    @property
    def stale(self) -> list[Fact]:
        return [_to_fact(n) for n in self._g.store.all_nodes()
                if n.status != Status.ACTIVE or n.meta.get("needs_review")]

    def facts(self, include_stale: bool = True) -> list[Fact]:
        return [_to_fact(n) for n in self._g.store.all_nodes()
                if include_stale or n.status == Status.ACTIVE]

    def why(self, fact_id: str) -> list[str]:
        """The DERIVED_FROM parents of a fact — 'this is stale because it depended on …'."""
        return [p.content for p in self._g.store.derived_from_targets(fact_id)]
