"""Cascade logic — the heart — tested deterministically with a mock LLM and embedder.
Covers: multi-hop propagation, semantic stop, cycle guard, every _apply label, conservative
fail-safe, decision caching, and the similarity-gate cost lever."""

import numpy as np
import pytest

from gem.engine import GEM, GEMConfig
from gem.store import EdgeType, Status
from conftest import FakeLLM, FakeEmbedder, classify_json


def _existing(user: str) -> str:
    for line in user.splitlines():
        if line.startswith("EXISTING memory:"):
            return line.split(":", 1)[1].strip()
    return ""


def _is_change_desc(user: str) -> bool:
    return any(k in user for k in ("has changed", "no longer reliable",
                                   "partially invalidated", "superseded"))


def mumbai_responder(system, user):
    """Conflict-check: only the location conflicts with the move. Boundary: commute->unknown,
    wake->partial, timezone->survives."""
    existing = _existing(user)
    if not _is_change_desc(user):
        return classify_json("UPDATES", "I live in Mumbai") if "Bangalore" in existing \
            else classify_json("UNRELATED")
    if "commute" in existing:
        return classify_json("UPDATES", None)            # unknown new value -> STALE
    if "wake" in existing:
        return classify_json("PARTIALLY_UPDATES", "I wake (time to reconfirm)")
    if "timezone" in existing:
        return classify_json("UNRELATED")                # semantic stop
    return classify_json("UNRELATED")


def _mumbai_gem(**cfg_kw):
    g = GEM(llm=FakeLLM(json_fn=mumbai_responder), embedder=FakeEmbedder(),
            config=GEMConfig(**cfg_kw))
    loc = g.ingest("I live in Bangalore", parents=[], check_conflicts=False)
    com = g.ingest("My commute is 45 minutes", parents=[loc.id], check_conflicts=False)
    wake = g.ingest("I wake at 7am to beat traffic", parents=[com.id], check_conflicts=False)
    tz = g.ingest("My timezone is IST", parents=[loc.id], check_conflicts=False)
    return g, dict(loc=loc, com=com, wake=wake, tz=tz)


def test_multihop_cascade_and_semantic_stop():
    g, n = _mumbai_gem()
    g.ingest("I now live in Mumbai", parents=[])
    assert g.store.get(n["loc"].id).content == "I live in Mumbai"        # updated
    assert g.store.get(n["com"].id).status == Status.STALE               # unknown -> stale
    assert g.store.get(n["wake"].id).confidence < 1.0                    # partial -> penalised
    assert g.store.get(n["tz"].id).status == Status.ACTIVE               # semantic stop survives


def test_contradicts_supersedes():
    g = GEM(llm=FakeLLM(json_fn=lambda s, u: classify_json("CONTRADICTS")),
            embedder=FakeEmbedder())
    n = g.ingest("the meeting is on", parents=[], check_conflicts=False)
    g.ingest("the meeting is cancelled", parents=[])
    assert g.store.get(n.id).status == Status.SUPERSEDED


def test_conservative_downgrades_supersede():
    g = GEM(llm=FakeLLM(json_fn=lambda s, u: classify_json("CONTRADICTS")),
            embedder=FakeEmbedder(), config=GEMConfig(conservative_invalidation=True))
    n = g.ingest("the meeting is on", parents=[], check_conflicts=False)
    g.ingest("the meeting is cancelled", parents=[])
    cur = g.store.get(n.id)
    assert cur.status == Status.STALE                  # recoverable, not destructive
    assert cur.meta.get("needs_review") is True


def test_cycle_guard_terminates():
    # A depends on B and B depends on A; updating must not infinite-loop
    g = GEM(llm=FakeLLM(json_fn=lambda s, u: classify_json("UPDATES", "x")),
            embedder=FakeEmbedder())
    a = g.ingest("assumption A ships Q3", parents=[], check_conflicts=False)
    b = g.ingest("plan B paced to Q3", parents=[], check_conflicts=False)
    g.store.add_edge(a.id, b.id, EdgeType.DERIVED_FROM)
    g.store.add_edge(b.id, a.id, EdgeType.DERIVED_FROM)
    g.ingest("the ship date slipped to Q4", parents=[])   # must return, not hang
    assert True


def test_decision_cache_avoids_repeat_calls():
    llm = FakeLLM(json_fn=lambda s, u: classify_json("UNRELATED"))
    g = GEM(llm=llm, embedder=FakeEmbedder(), config=GEMConfig(cache_decisions=True))
    g._classify("salary is 80k", "got a raise")
    g._classify("salary is 80k", "got a raise")           # identical -> cache
    assert g.stats["cache_hits"] == 1
    assert llm.json_calls == 1                            # only one real call


def test_similarity_gate_skips_dissimilar_neighbors():
    # trigger (Mumbai) is a CONFLICT with Bangalore (cos ~0.71: above the 0.5 gate, below the
    # 0.97 dedup) and clearly dissimilar to the teal distractor (cos 0, below the gate).
    vecs = {"Bangalore": [1, 0, 0], "Mumbai": [0.7, 0.7, 0], "teal": [0, 0, 1]}

    def responder(s, u):
        return classify_json("UPDATES", "I live in Mumbai") if "Bangalore" in _existing(u) \
            else classify_json("UNRELATED")

    g = GEM(llm=FakeLLM(json_fn=responder), embedder=FakeEmbedder(vecs),
            config=GEMConfig(candidate_k=10, conflict_sim_threshold=0.5))
    g.ingest("I live in Bangalore", parents=[], check_conflicts=False)
    g.ingest("My favorite color is teal", parents=[], check_conflicts=False)   # distractor
    g.ingest("I now live in Mumbai", parents=[])
    assert g.stats["sim_skipped"] >= 1                   # the teal distractor was gated out
    # the gate must NOT skip the real conflict: location still got updated
    assert any(nd.content == "I live in Mumbai" for nd in g.store.all_nodes())


def test_multi_direct_conflict_on_chain_revises_each_node_once():
    """Regression: when ONE trigger directly conflicts with SEVERAL nodes in the same
    DERIVED_FROM chain, the root's cascade already revises the descendants. Each conflict
    must NOT fire an independent cascade (that re-revises the subchain with a conflicting
    result — the deep-chain interference bug). Here the trigger directly conflicts with BOTH
    launch (root) and campaign (middle); embargo (leaf) only conflicts transitively."""
    # distinct, non-colliding keys so the FakeEmbedder gives controlled (non-dedup) similarities
    vecs = {"launch": [1, 0, 0, 0], "campaign": [0, 1, 0, 0],
            "embargo": [0, 0, 1, 0], "release": [0.5, 0.5, 0.5, 0.5]}

    def responder(s, u):
        existing = _existing(u)
        if not _is_change_desc(u):                       # PASS A: direct conflict scan
            if "launch" in existing:
                return classify_json("UPDATES", "The launch is December 1")
            if "campaign" in existing:                   # trigger ALSO hits the middle node
                return classify_json("UPDATES", "The campaign starts November 17")
            return classify_json("UNRELATED")            # embargo: no direct conflict
        return classify_json("UPDATES", "revised by cascade")   # cascade: invalidate dependents

    g = GEM(llm=FakeLLM(json_fn=responder), embedder=FakeEmbedder(vecs))
    launch = g.ingest("The product launch date is October 1", parents=[], check_conflicts=False)
    campaign = g.ingest("The marketing campaign starts in September", parents=[launch.id],
                        check_conflicts=False)
    embargo = g.ingest("The press embargo lifts later", parents=[campaign.id],
                       check_conflicts=False)

    g.ingest("The release date moved to December", parents=[])

    revised_campaign = [t for t in g.trace if t.lstrip().startswith(f"revise {campaign.id}:")]
    revised_embargo = [t for t in g.trace if t.lstrip().startswith(f"revise {embargo.id}:")]
    assert len(revised_campaign) == 1                    # middle node revised ONCE, not twice
    assert len(revised_embargo) == 1                     # leaf not re-cascaded by a 2nd pass
    assert any("skip re-revision" in t for t in g.trace)  # the redundant action was skipped
    # and the chain still fully invalidated (the fix preserves correctness, doesn't suppress it)
    assert g.store.get(campaign.id).content == "revised by cascade"
    assert g.store.get(embargo.id).content == "revised by cascade"


def test_escalation_only_confirms_destructive():
    cheap = FakeLLM(json_fn=lambda s, u: classify_json(
        "UPDATES" if "conflict" in _existing(u) else "UNRELATED"))
    capable = FakeLLM(json_fn=lambda s, u: classify_json("UPDATES", "confirmed"))
    g = GEM(llm=capable, cheap_llm=cheap, embedder=FakeEmbedder(),
            config=GEMConfig(escalate=True))
    g._classify("a benign fact", "something")             # cheap UNRELATED -> no escalation
    g._classify("a conflict fact", "something")           # cheap UPDATES -> escalate to capable
    assert g.stats["cheap_calls"] == 2
    assert g.stats["capable_calls"] == 1                  # only the destructive one escalated
