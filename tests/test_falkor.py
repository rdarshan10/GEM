"""FalkorDB store parity — runs only if a FalkorDB server is reachable on localhost:6379,
otherwise skipped (so the suite stays green without Docker)."""

import numpy as np
import pytest

from gem.store import Node, EdgeType, Status


def _falkor_or_skip():
    try:
        from gem.falkor_store import FalkorStore
        s = FalkorStore(clear_on_start=True)
        return s
    except Exception as e:
        pytest.skip(f"FalkorDB not reachable: {e}")


def test_falkor_crud_and_dependents():
    s = _falkor_or_skip()
    a = Node(id=s.new_id(), content="lives in Bangalore",
             embedding=np.array([0.1, 0.2], "float32"))
    b = Node(id=s.new_id(), content="commute 45 min",
             embedding=np.array([0.3, 0.4], "float32"))
    s.add_node(a); s.add_node(b)
    s.add_edge(b.id, a.id, EdgeType.DERIVED_FROM)
    s.add_edge(b.id, a.id, EdgeType.DERIVED_FROM)        # dedup via MERGE

    got = s.get(a.id)
    assert got.content == "lives in Bangalore"
    assert np.allclose(got.embedding, [0.1, 0.2], atol=1e-3)   # embedding rehydrated
    assert [n.id for n in s.dependents(a.id)] == [b.id]
    assert len([e for e in s.edges() if e.type == EdgeType.DERIVED_FROM]) == 1

    b.status = Status.STALE; b.confidence = 0.5
    s.update_node(b)
    assert s.get(b.id).status == Status.STALE
    assert [n.id for n in s.active_nodes()] == [a.id]
    s.clear()


def test_falkor_new_id_seeds_from_max(tmp_path=None):
    """ids must survive a 'restart' by seeding from the max existing id, not count(n)."""
    from gem.falkor_store import FalkorStore
    try:
        s = FalkorStore(clear_on_start=True)
    except Exception as e:
        pytest.skip(f"FalkorDB not reachable: {e}")
    n = Node(id="n5", content="x", embedding=np.array([0.0, 0.0], "float32"))
    s.add_node(n)
    s2 = FalkorStore(clear_on_start=False)              # fresh handle, existing data
    assert int(s2.new_id().lstrip("n")) > 5            # seeded past the max id, not from count
    s.clear()
