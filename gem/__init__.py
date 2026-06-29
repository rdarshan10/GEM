"""GEM — Governed Evolving Memory: dependency-aware memory invalidation.

The one thing GEM does that flat/vector memory (Mem0, ChatGPT memory, plain RAG) does not:
when a fact changes, the facts DERIVED from it go stale automatically — down a typed
DERIVED_FROM chain, multi-hop — while unrelated facts survive. A similarity-only memory
never re-examines a dependent that isn't textually similar to the change, so it serves the
stale fact confidently. GEM re-examines it via the dependency edge.

Public API (see `gem.memory`):
    from gem import Memory
    m = Memory()
    a = m.add("Auth uses JWT tokens")
    m.add("Tests mock the JWT verifier", derived_from=[a])
    r = m.add("We migrated auth to session cookies")
    r.invalidated          # -> the derived fact, now flagged stale
    m.search("auth in tests")   # -> ACTIVE facts only

Honest scope: this earns its cost when memory drives ACTIONS off chains of derived facts
and reads outnumber writes. For flat atomic facts (preferences, profile fields) it is
overhead — see PRODUCTIONIZATION.md for the measured win/lose regimes.
"""

from .memory import Memory, Fact, AddResult

__all__ = ["Memory", "Fact", "AddResult"]
__version__ = "0.1.0"
