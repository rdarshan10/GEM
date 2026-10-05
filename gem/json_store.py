"""File-backed store: the in-memory graph, persisted to one JSON file.

Zero-dependency persistence, safe for several processes sharing one file (e.g. two agents running
the MCP server). Memory wraps every write in transaction(): an OS file lock (released by the OS if
the process dies), a reload if another process changed the file, the write, then an atomic save.
Reads call refresh() so they see other processes' writes. Writers are serialised, so a slow
cascade holds the lock for its duration; for many concurrent writers use FalkorStore.
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile
import time
from contextlib import contextmanager

import numpy as np

from .store import Edge, EdgeType, MemoryStore, Node, Provenance, Status

_VERSION = 1


def _retry(fn, attempts: int = 100, delay: float = 0.02):
    """Windows can't replace (or briefly open) a file another process is reading at that instant;
    lock-free readers hold it for milliseconds, so retry a few times before giving up."""
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


class _FileLock:
    """Exclusive lock on `path` via the OS (msvcrt on Windows, flock elsewhere)."""

    def __init__(self, path: str, timeout: float):
        self.path, self.timeout = path, timeout
        self._f = None

    def __enter__(self):
        self._f = open(self.path, "a+")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._lock()
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    self._f.close()
                    raise TimeoutError(f"memory store is locked by another process: {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc):
        try:
            self._unlock()
        finally:
            self._f.close()

    if os.name == "nt":
        def _lock(self):
            import msvcrt
            self._f.seek(0)
            msvcrt.locking(self._f.fileno(), msvcrt.LK_NBLCK, 1)

        def _unlock(self):
            import msvcrt
            self._f.seek(0)
            msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        def _lock(self):
            import fcntl
            fcntl.flock(self._f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def _unlock(self):
            import fcntl
            fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)


class JsonStore(MemoryStore):
    def __init__(self, path: str, *, lock_timeout: float = 120.0):
        super().__init__()
        self.path = os.path.abspath(os.path.expanduser(path))
        self.lock_timeout = lock_timeout
        self._sig = None          # (mtime_ns, size) of the file as last loaded/saved
        self._depth = 0           # transaction nesting
        self.refresh()

    def _file_sig(self):
        try:
            st = os.stat(self.path)
            return (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            return None

    def refresh(self) -> None:
        """Reload if the file changed since this process last read or wrote it."""
        sig = self._file_sig()
        if sig is not None and sig != self._sig:
            self._load()
            self._sig = sig

    @contextmanager
    def transaction(self):
        """Lock, catch up with other processes, run the write, save. On error the in-memory
        graph is reloaded from disk so a half-applied cascade is never kept or saved."""
        if self._depth:
            yield
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with _FileLock(self.path + ".lock", self.lock_timeout):
            self._depth += 1
            try:
                self.refresh()
                yield
                self.save()
            except BaseException:
                self._sig = None
                self._reset()
                self.refresh()
                raise
            finally:
                self._depth -= 1

    def _reset(self) -> None:
        self._nodes, self._edges = {}, []
        self._id_counter = itertools.count(1)

    def _read_file(self) -> dict:
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _load(self) -> None:
        data = _retry(self._read_file)
        self._reset()
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
            # atomic: a crash never leaves a half-written file
            _retry(lambda: os.replace(tmp, self.path))
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        self._sig = self._file_sig()
