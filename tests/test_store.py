"""MemoryStore CRUD + typed-edge queries — fully deterministic, no LLM."""

import numpy as np
import pytest

from gem.store import MemoryStore, Node, EdgeType, Status, Provenance


@pytest.fixture
def store():
    return MemoryStore()


def _node(s, content, emb=(1.0, 0.0)):
    return s.add_node(Node(id=s.new_id(), content=content, embedding=np.array(emb, "float32")))


def test_new_id_unique(store):
    ids = {store.new_id() for _ in range(100)}
    assert len(ids) == 100


def test_add_get_update(store):
    n = _node(store, "lives in Bangalore")
    assert store.get(n.id).content == "lives in Bangalore"
    n.content = "lives in Mumbai"
    store.update_node(n)
    assert store.get(n.id).content == "lives in Mumbai"
    assert store.get("nonexistent") is None


def test_active_vs_all(store):
    a = _node(store, "a")
    b = _node(store, "b")
    b.status = Status.STALE
    store.update_node(b)
    assert len(store.all_nodes()) == 2
    assert [n.id for n in store.active_nodes()] == [a.id]


def test_edge_dedup(store):
    a, b = _node(store, "a"), _node(store, "b")
    store.add_edge(a.id, b.id, EdgeType.DERIVED_FROM)
    store.add_edge(a.id, b.id, EdgeType.DERIVED_FROM)   # duplicate
    assert len([e for e in store.edges() if e.type == EdgeType.DERIVED_FROM]) == 1


def test_dependents_and_targets(store):
    loc, com, tz = _node(store, "location"), _node(store, "commute"), _node(store, "timezone")
    store.add_edge(com.id, loc.id, EdgeType.DERIVED_FROM)   # commute depends on location
    store.add_edge(tz.id, loc.id, EdgeType.DERIVED_FROM)    # timezone depends on location
    deps = {n.id for n in store.dependents(loc.id)}
    assert deps == {com.id, tz.id}
    targets = {n.id for n in store.derived_from_targets(com.id)}
    assert targets == {loc.id}


def test_associated_not_a_dependent(store):
    a, b = _node(store, "a"), _node(store, "b")
    store.add_edge(b.id, a.id, EdgeType.ASSOCIATED)
    assert store.dependents(a.id) == []   # ASSOCIATED is not a DERIVED_FROM dependent


def test_node_defaults(store):
    n = _node(store, "x")
    assert n.status == Status.ACTIVE
    assert n.confidence == 1.0
    assert n.provenance_type == Provenance.FACT
