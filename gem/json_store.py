"""File-backed store: the in-memory graph, persisted to one JSON file.

Zero-dependency persistence for a single process (an agent's MCP server, a CLI). The whole graph is
rewritten atomically on save(); Memory calls save() once per write operation, not once per node
change, so a long cascade costs one write. For multi-process or large graphs use FalkorStore.
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile

import numpy as np

from .store import Edge, EdgeType, MemoryStore, Node, Provenance, Status

_VERSION = 1


class JsonStore(MemoryStore):
    def __init__(self, path: str):
        super().__init__()
        self.path = os.path.abspath(os.path.expanduser(path))
        if os.path.exists(self.path):
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("nodes", []):
            emb = d.get("embedding")
            self._nodes[d["id"]] = Node(
                id=d["id"], content=d["content"],
                embedding=None if emb is None else np.asarray(emb, dtype=np.float32),
                provenance_type=Provenance(d.get("provenance_type", Provenance.FACT.value)),
                salience=d.get("salience", 1.0), confidence=d.get("confidence", 1.0),
                status=Status(d.get("status", Status.ACTIVE.value)), ttl=d.get("ttl"),
                meta=d.get("meta", {}))
        self._edges = [Edge(e["src"], e["dst"], EdgeType(e["type"])) for e in data.get("edges", [])]
        # continue numbering past the largest existing id so a restart never reissues one
        nums = [int(i[1:]) for i in self._nodes if i[1:].isdigit()]
        self._id_counter = itertools.count(max(nums, default=0) + 1)

    def save(self) -> None:
        data = {
            "version": _VERSION,
            "nodes": [{
                "id": n.id, "content": n.content,
                "embedding": None if n.embedding is None else np.asarray(n.embedding).tolist(),
                "provenance_type": n.provenance_type.value, "salience": n.salience,
                "confidence": n.confidence, "status": n.status.value, "ttl": n.ttl,
                "meta": n.meta,
            } for n in self._nodes.values()],
            "edges": [{"src": e.src_id, "dst": e.dst_id, "type": e.type.value} for e in self._edges],
        }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)            # atomic: a crash never leaves a half-written file
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
