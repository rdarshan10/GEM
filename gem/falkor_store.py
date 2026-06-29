"""FalkorDB-backed store — the optional persistence backend.

Implements the same interface as the in-memory MemoryStore (duck-typed), so the cascade
engine is unchanged: `GEM(store=FalkorStore())`. FalkorDB is a Redis-module graph database
queried with Cypher; it runs as a server (Docker `falkordb/falkordb` or FalkorDB Cloud),
so this is an OPTIONAL backend — the in-memory store remains the zero-dependency default.

`pip install falkordb`, then e.g. `docker run -p 6379:6379 falkordb/falkordb`.

Nodes are :Memory nodes; edges are typed relationships (DERIVED_FROM / ASSOCIATED /
CONTRADICTS). Embeddings are stored as a JSON string property and rehydrated to numpy.
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from .store import Node, Edge, EdgeType, Status, Provenance

_VALID_EDGE = {e.value for e in EdgeType}


class FalkorStore:
    def __init__(self, host: str = "localhost", port: int = 6379,
                 graph: str = "gem", clear_on_start: bool = False):
        from falkordb import FalkorDB  # lazy import
        self._db = FalkorDB(host=host, port=port)
        self._g = self._db.select_graph(graph)
        if clear_on_start:
            self.clear()
        # Seed the id counter from the MAX existing id, not count(n). Counting breaks after
        # deletions: count can fall below live ids, so a fresh counter would reissue an
        # existing id and silently overwrite a persisted node on restart. Parse the numeric
        # suffix of every id and start past the largest.
        start = 1
        try:
            res = self._g.query("MATCH (n:Memory) RETURN n.id")
            mx = 0
            for row in res.result_set:
                digits = "".join(ch for ch in str(row[0]) if ch.isdigit())
                if digits:
                    mx = max(mx, int(digits))
            start = mx + 1
        except Exception:
            pass
        self._id_counter = itertools.count(start)

    # --- helpers -------------------------------------------------------------- #
    def clear(self) -> None:
        self._g.query("MATCH (n) DETACH DELETE n")

    @staticmethod
    def _emb_to_str(emb) -> str | None:
        if emb is None:
            return None
        arr = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
        return json.dumps(arr)

    @staticmethod
    def _node_from_props(p: dict) -> Node:
        emb = p.get("embedding")
        embedding = (np.array(json.loads(emb), dtype=np.float32)
                     if emb not in (None, "") else None)
        meta = p.get("meta")
        return Node(
            id=p["id"],
            content=p["content"],
            embedding=embedding,
            provenance_type=Provenance(p.get("provenance_type", "FACT")),
            salience=float(p.get("salience", 1.0)),
            confidence=float(p.get("confidence", 1.0)),
            status=Status(p.get("status", "ACTIVE")),
            ttl=p.get("ttl"),
            meta=json.loads(meta) if meta else {},
        )

    def _props(self, node: Node) -> dict:
        return {
            "id": node.id,
            "content": node.content,
            "embedding": self._emb_to_str(node.embedding),
            "provenance_type": node.provenance_type.value,
            "salience": float(node.salience),
            "confidence": float(node.confidence),
            "status": node.status.value,
            "ttl": node.ttl,
            "meta": json.dumps(node.meta) if node.meta else None,
        }

    # --- nodes ---------------------------------------------------------------- #
    def new_id(self, prefix: str = "n") -> str:
        return f"{prefix}{next(self._id_counter)}"

    def add_node(self, node: Node) -> Node:
        self._g.query(
            "CREATE (n:Memory {id:$id, content:$content, embedding:$embedding, "
            "provenance_type:$provenance_type, salience:$salience, confidence:$confidence, "
            "status:$status, ttl:$ttl, meta:$meta})",
            params=self._props(node),
        )
        return node

    def get(self, node_id: str) -> Node | None:
        res = self._g.query(
            "MATCH (n:Memory {id:$id}) RETURN n.id, n.content, n.embedding, "
            "n.provenance_type, n.salience, n.confidence, n.status, n.ttl, n.meta",
            params={"id": node_id},
        )
        if not res.result_set:
            return None
        return self._row_to_node(res.result_set[0])

    @staticmethod
    def _row_to_node(row) -> Node:
        keys = ["id", "content", "embedding", "provenance_type", "salience",
                "confidence", "status", "ttl", "meta"]
        return FalkorStore._node_from_props(dict(zip(keys, row)))

    def update_node(self, node: Node) -> None:
        self._g.query(
            "MATCH (n:Memory {id:$id}) SET n.content=$content, n.embedding=$embedding, "
            "n.provenance_type=$provenance_type, n.salience=$salience, "
            "n.confidence=$confidence, n.status=$status, n.ttl=$ttl, n.meta=$meta",
            params=self._props(node),
        )

    def _all(self) -> list[Node]:
        res = self._g.query(
            "MATCH (n:Memory) RETURN n.id, n.content, n.embedding, n.provenance_type, "
            "n.salience, n.confidence, n.status, n.ttl, n.meta"
        )
        return [self._row_to_node(r) for r in res.result_set]

    def all_nodes(self) -> list[Node]:
        return self._all()

    def active_nodes(self) -> list[Node]:
        return [n for n in self._all() if n.status == Status.ACTIVE]

    # --- edges ---------------------------------------------------------------- #
    def add_edge(self, src_id: str, dst_id: str, etype: EdgeType) -> Edge:
        t = etype.value
        if t not in _VALID_EDGE:                 # guard: relationship type is interpolated
            raise ValueError(f"bad edge type {t!r}")
        self._g.query(
            f"MATCH (a:Memory {{id:$src}}), (b:Memory {{id:$dst}}) "
            f"MERGE (a)-[:{t}]->(b)",
            params={"src": src_id, "dst": dst_id},
        )
        return Edge(src_id, dst_id, etype)

    def edges(self) -> list[Edge]:
        out = []
        for t in _VALID_EDGE:
            res = self._g.query(
                f"MATCH (a:Memory)-[:{t}]->(b:Memory) RETURN a.id, b.id"
            )
            out.extend(Edge(r[0], r[1], EdgeType(t)) for r in res.result_set)
        return out

    def derived_from_targets(self, node_id: str) -> list[Node]:
        res = self._g.query(
            "MATCH (n:Memory {id:$id})-[:DERIVED_FROM]->(p:Memory) "
            "RETURN p.id, p.content, p.embedding, p.provenance_type, p.salience, "
            "p.confidence, p.status, p.ttl, p.meta",
            params={"id": node_id},
        )
        return [self._row_to_node(r) for r in res.result_set]

    def dependents(self, node_id: str) -> list[Node]:
        """Nodes DERIVED_FROM node_id — the one graph hop the cascade walks."""
        res = self._g.query(
            "MATCH (dep:Memory)-[:DERIVED_FROM]->(n:Memory {id:$id}) "
            "RETURN dep.id, dep.content, dep.embedding, dep.provenance_type, dep.salience, "
            "dep.confidence, dep.status, dep.ttl, dep.meta",
            params={"id": node_id},
        )
        return [self._row_to_node(r) for r in res.result_set]

    # --- debug ---------------------------------------------------------------- #
    def describe(self) -> str:
        lines = ["NODES:"]
        for n in self._all():
            tag = "" if n.status == Status.ACTIVE else f"  [{n.status.value}]"
            conf = "" if n.confidence >= 0.999 else f"  conf={n.confidence:.2f}"
            lines.append(f"  {n.id}: {n.content}{tag}{conf}")
        lines.append("EDGES:")
        for e in self.edges():
            lines.append(f"  {e.src_id} --{e.type.value.lower()}--> {e.dst_id}")
        return "\n".join(lines)
