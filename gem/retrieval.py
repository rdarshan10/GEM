"""Unit 2 — vector + graph retrieval.

Three layers, each earning its place (per the build plan):
  FaissIndex          — replaces the O(N) numpy cosine scan with a real ANN index, so a
                        scale claim is even arguable. Inner-product over L2-normalized
                        vectors == cosine.
  semantic_search     — FAISS top-k, filtered to ACTIVE nodes.
  graph_proximity_search — the NOVEL part: spreading activation over the typed graph.
                        A query that matches one memory in a dependency cluster also
                        surfaces the REST of that cluster (commute, schedule, ...) by
                        propagating a discounted share of each semantic seed's score along
                        DERIVED_FROM / ASSOCIATED edges. A flat vector store cannot do this;
                        it's retrieval that exploits the typed graph nobody else has.

The graph-proximity claim is measured, not asserted — see gem/retrieval_ablation.py.
"""

from __future__ import annotations

import numpy as np

from .store import MemoryStore, Node, Status, EdgeType


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


class FaissIndex:
    """Thin wrapper over faiss.IndexIDMap(IndexFlatIP). Maps string node ids <-> int ids."""

    def __init__(self, dim: int):
        import faiss
        self._faiss = faiss
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._node_of: dict[int, str] = {}
        self._int_of: dict[str, int] = {}
        self._counter = 0

    def add(self, node_id: str, emb: np.ndarray) -> None:
        if node_id in self._int_of:          # update -> remove old vector first
            self.remove(node_id)
        iid = self._counter
        self._counter += 1
        self._node_of[iid] = node_id
        self._int_of[node_id] = iid
        vec = _normalize(np.asarray(emb, dtype=np.float32)).reshape(1, -1)
        self._index.add_with_ids(vec, np.array([iid], dtype=np.int64))

    def remove(self, node_id: str) -> None:
        iid = self._int_of.pop(node_id, None)
        if iid is not None:
            self._index.remove_ids(np.array([iid], dtype=np.int64))
            self._node_of.pop(iid, None)

    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []
        q = _normalize(np.asarray(query_emb, dtype=np.float32)).reshape(1, -1)
        scores, ids = self._index.search(q, min(k, self._index.ntotal))
        out = []
        for sc, iid in zip(scores[0], ids[0]):
            if iid != -1 and iid in self._node_of:
                out.append((self._node_of[iid], float(sc)))
        return out

    @classmethod
    def from_store(cls, store: MemoryStore, dim: int) -> "FaissIndex":
        idx = cls(dim)
        for n in store.all_nodes():
            if n.embedding is not None:
                idx.add(n.id, n.embedding)
        return idx


def semantic_search(store: MemoryStore, index: FaissIndex, query_emb: np.ndarray,
                    k: int = 5, big_k: int = 20) -> list[tuple[Node, float]]:
    """FAISS top-k, over-fetched then filtered to ACTIVE nodes."""
    hits = index.search(query_emb, big_k)
    out = []
    for nid, score in hits:
        n = store.get(nid)
        if n is not None and n.status == Status.ACTIVE:
            out.append((n, score))
        if len(out) >= k:
            break
    return out


def graph_proximity_search(store: MemoryStore, index: FaissIndex, query_emb: np.ndarray,
                           k: int = 5, *, seeds: int = 5, alpha: float = 0.5,
                           hops: int = 2, big_k: int = 20) -> list[tuple[Node, float]]:
    """Semantic seeds + spreading activation over typed edges.

    score(n) = semantic_sim(q, n) + sum over seeds s of alpha^d * sim(q, s),
    where d is the DERIVED_FROM/ASSOCIATED graph distance from s to n (<= hops). A node that
    is graph-adjacent to strong semantic matches gets surfaced even if its own text is a
    weak match — which is exactly the dependency-neighborhood a flat store misses.
    """
    base = {nid: sc for nid, sc in index.search(query_emb, big_k)}
    activation: dict[str, float] = dict(base)

    # spreading activation from the top semantic seeds along typed edges (both directions)
    seed_items = sorted(base.items(), key=lambda t: t[1], reverse=True)[:seeds]
    for sid, sscore in seed_items:
        frontier = {sid}
        visited = {sid}
        for d in range(1, hops + 1):
            nxt = set()
            for nid in frontier:
                for nb in _typed_neighbors(store, nid):
                    if nb in visited:
                        continue
                    activation[nb] = activation.get(nb, 0.0) + (alpha ** d) * sscore
                    nxt.add(nb)
                    visited.add(nb)
            frontier = nxt

    scored = []
    for nid, score in activation.items():
        n = store.get(nid)
        if n is not None and n.status == Status.ACTIVE:
            scored.append((n, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]


def _typed_neighbors(store: MemoryStore, node_id: str) -> set[str]:
    """Both directions over DERIVED_FROM and ASSOCIATED (not CONTRADICTS)."""
    out = set()
    for e in store.edges():
        if e.type in (EdgeType.DERIVED_FROM, EdgeType.ASSOCIATED):
            if e.src_id == node_id:
                out.add(e.dst_id)
            elif e.dst_id == node_id:
                out.add(e.src_id)
    return out
