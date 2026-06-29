"""Embedding + retrieval layer: cosine, FAISS-vs-numpy parity, graph-proximity, and the
correctness-critical fallback warning."""

import numpy as np
import pytest

from gem import embed as E
from gem.store import MemoryStore, Node, EdgeType
from gem.retrieval import FaissIndex, semantic_search, graph_proximity_search


def test_cosine():
    a = np.array([1.0, 0.0]); b = np.array([1.0, 0.0]); c = np.array([0.0, 1.0])
    assert abs(E.cosine(a, b) - 1.0) < 1e-6
    assert abs(E.cosine(a, c)) < 1e-6
    assert E.cosine(None, a) == 0.0


def test_lexical_embedder_deterministic_within_run():
    e = E.LexicalEmbedder(dim=64)
    v1, v2 = e.embed("hello world"), e.embed("hello world")
    assert np.allclose(v1, v2)
    assert E.cosine(e.embed("the cat sat"), e.embed("the cat sat")) > 0.99


def _store_with(vectors):
    s = MemoryStore()
    nodes = []
    for content, vec in vectors:
        nodes.append(s.add_node(Node(id=s.new_id(), content=content,
                                     embedding=np.array(vec, "float32"))))
    return s, nodes


def test_faiss_matches_numpy():
    s, nodes = _store_with([("a", [1, 0, 0]), ("b", [0, 1, 0]), ("c", [0.9, 0.1, 0])])
    idx = FaissIndex.from_store(s, dim=3)
    q = np.array([1, 0, 0], "float32")
    faiss_ids = [n.id for n, _ in semantic_search(s, idx, q, k=3)]
    numpy_ids = [n.id for n, _ in E.search(s, q, k=3)]
    assert faiss_ids == numpy_ids


def test_faiss_remove():
    s, nodes = _store_with([("a", [1, 0]), ("b", [0, 1])])
    idx = FaissIndex.from_store(s, dim=2)
    idx.remove(nodes[0].id)
    hits = idx.search(np.array([1, 0], "float32"), k=5)
    assert nodes[0].id not in {nid for nid, _ in hits}


def test_graph_proximity_surfaces_dependents():
    # root strongly matches the query; dependents are dissimilar but graph-connected
    s, nodes = _store_with([("home", [1, 0, 0]), ("commute", [0, 1, 0]), ("wake", [0, 0, 1])])
    root, com, wake = nodes
    s.add_edge(com.id, root.id, EdgeType.DERIVED_FROM)
    s.add_edge(wake.id, com.id, EdgeType.DERIVED_FROM)
    idx = FaissIndex.from_store(s, dim=3)
    q = np.array([1, 0, 0], "float32")            # matches only 'home' semantically
    sem = {n.id for n, _ in semantic_search(s, idx, q, k=1)}
    gph = {n.id for n, _ in graph_proximity_search(s, idx, q, k=3)}
    assert sem == {root.id}
    assert {com.id, wake.id} <= gph               # graph pulled in the dependents


def test_default_embedder_is_cached_singleton():
    E._DEFAULT_EMB = None
    a = E.default_embedder()
    b = E.default_embedder()
    assert a is b


def test_fallback_warns_loudly(monkeypatch, capsys):
    class Boom:
        def __init__(self, *a, **k):
            raise ImportError("no sentence-transformers")
    monkeypatch.setattr(E, "STEmbedder", Boom)
    E._DEFAULT_EMB = None
    emb = E.default_embedder()
    err = capsys.readouterr().err
    assert isinstance(emb, E.LexicalEmbedder)
    assert "WARNING" in err and "correctness" in err.lower()
    E._DEFAULT_EMB = None     # reset for other tests
