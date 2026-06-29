"""In-memory graph store for the MVP.

Nodes are memories; edges are typed. The cascade walks DERIVED_FROM edges only —
ASSOCIATED edges are never followed for invalidation (that's the typed-edge distinction
doing its job). This is deliberately a thin dict/adjacency implementation behind a small
interface so KuzuDB can replace it later (Unit 1) without the cascade code changing.

Node.status drives validity: ACTIVE / STALE / SUPERSEDED.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class EdgeType(str, Enum):
    ASSOCIATED = "ASSOCIATED"        # semantic relatedness, NO validity dependency
    DERIVED_FROM = "DERIVED_FROM"    # target's validity depends on source; cascade walks this
    CONTRADICTS = "CONTRADICTS"      # recorded conflict between two nodes


class Status(str, Enum):
    ACTIVE = "ACTIVE"
    STALE = "STALE"                  # validity in doubt (e.g. unknown new value)
    SUPERSEDED = "SUPERSEDED"        # replaced/negated


class Provenance(str, Enum):
    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    DERIVED_FACT = "DERIVED_FACT"
    BELIEF = "BELIEF"
    PROCEDURE = "PROCEDURE"


@dataclass
class Node:
    id: str
    content: str
    embedding: object = None          # numpy vector or None (set by embedder)
    provenance_type: Provenance = Provenance.FACT
    salience: float = 1.0             # MVP: stubbed constant-ish; decay is post-MVP
    confidence: float = 1.0           # lowered when a conflict can't be cleanly resolved
    status: Status = Status.ACTIVE
    ttl: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Edge:
    src_id: str                       # for DERIVED_FROM: the dependent (derived) node
    dst_id: str                       # for DERIVED_FROM: the node it depends on
    type: EdgeType


class MemoryStore:
    """Thin in-memory graph. The interface (add/get/update node, add edge, dependents)
    is what later KuzuDB-backed stores must implement."""

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._id_counter = itertools.count(1)

    # --- nodes ---------------------------------------------------------------- #
    def new_id(self, prefix: str = "n") -> str:
        return f"{prefix}{next(self._id_counter)}"

    def add_node(self, node: Node) -> Node:
        self._nodes[node.id] = node
        return node

    def get(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def update_node(self, node: Node) -> None:
        self._nodes[node.id] = node

    def all_nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def active_nodes(self) -> list[Node]:
        return [n for n in self._nodes.values() if n.status == Status.ACTIVE]

    # --- edges ---------------------------------------------------------------- #
    def add_edge(self, src_id: str, dst_id: str, etype: EdgeType) -> Edge:
        # avoid duplicate identical edges
        for e in self._edges:
            if e.src_id == src_id and e.dst_id == dst_id and e.type == etype:
                return e
        edge = Edge(src_id, dst_id, etype)
        self._edges.append(edge)
        return edge

    def edges(self) -> list[Edge]:
        return list(self._edges)

    def derived_from_targets(self, node_id: str) -> list[Node]:
        """Nodes that `node_id` is DERIVED_FROM (its parents / what it depends on)."""
        out = []
        for e in self._edges:
            if e.type == EdgeType.DERIVED_FROM and e.src_id == node_id:
                n = self._nodes.get(e.dst_id)
                if n:
                    out.append(n)
        return out

    def dependents(self, node_id: str) -> list[Node]:
        """Nodes DERIVED_FROM `node_id` — i.e. those whose validity depends on it.
        This is the one graph hop the cascade walks."""
        out = []
        for e in self._edges:
            if e.type == EdgeType.DERIVED_FROM and e.dst_id == node_id:
                n = self._nodes.get(e.src_id)
                if n:
                    out.append(n)
        return out

    # --- debug ---------------------------------------------------------------- #
    def describe(self) -> str:
        lines = ["NODES:"]
        for n in self._nodes.values():
            tag = "" if n.status == Status.ACTIVE else f"  [{n.status.value}]"
            conf = "" if n.confidence >= 0.999 else f"  conf={n.confidence:.2f}"
            lines.append(f"  {n.id}: {n.content}{tag}{conf}")
        lines.append("EDGES:")
        for e in self._edges:
            arrow = {"DERIVED_FROM": "--derived-from-->",
                     "ASSOCIATED": "--associated-->",
                     "CONTRADICTS": "--contradicts-->"}[e.type.value]
            lines.append(f"  {e.src_id} {arrow} {e.dst_id}")
        return "\n".join(lines)
