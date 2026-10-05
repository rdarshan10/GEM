"""Cheap typed decisions (Jev) in front of the LLM — the LLM is called only where it's needed.

The engine makes three kinds of judgment: the ingest conflict scan, derive_links, and the per-dependent
check in the cascade. A Decider answers them in batches with probabilities instead of free text, and
TieredJudge turns those probabilities into an action:

  conflict scan  — confidently no conflict          -> skip (no LLM)
                   new fact fully covers the memory  -> rewrite it to the new fact (no LLM)
                   anything else                     -> today's LLM classify
  derive_links   — confidently depends               -> DERIVED_FROM edge (no LLM)
                   in between                        -> LLM derive_links on those candidates only
                   confidently independent           -> no edge
  cascade        — confidently unaffected            -> semantic stop (no LLM)
                   affected, new value unknown       -> STALE + needs_review (no LLM)
                   affected, new value follows       -> LLM classify (it writes the rewrite)
                   unsure                            -> LLM classify

Safety: stopping needs high confidence (jev_stop), so a mistake leaves a fact flagged for review rather
than missed; Jev never supersedes a dependent on its own; and if Jev fails the engine falls back to the
LLM path and counts it in DEGRADED["jev"] — the cascade never silently stops.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Sequence

from .classify import Label, INVALIDATING

# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #


@dataclass
class ConflictVerdict:
    probs: dict[Label, float]     # P(label) over the 6 classify labels
    p_covered: float              # P(the new fact alone fully replaces the memory)


@dataclass
class ImpactVerdict:
    probs: dict[int, float]       # 0 unaffected, 1 partly, 2 value changed/unknown, 3 invalid
    p_known: float                # P(the new value follows from the change)


class Decider(Protocol):
    def conflicts(self, new_fact: str, existing: Sequence[str]) -> list[ConflictVerdict | None]: ...
    def depends(self, new_fact: str, candidates: Sequence[str]) -> list[float | None]: ...
    def impact(self, change: str, dependents: Sequence[str]) -> list[ImpactVerdict | None]: ...


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #


class TieredJudge:
    """Pure probability -> action rules. Thresholds come from GEMConfig."""

    def __init__(self, cfg):
        self.cfg = cfg

    def conflict(self, v: ConflictVerdict | None) -> str:
        """'skip' | 'covered' | 'llm'"""
        if v is None:
            return "llm"
        p_inv = sum(p for lbl, p in v.probs.items() if lbl in INVALIDATING)
        if 1.0 - p_inv >= self.cfg.jev_stop:
            return "skip"
        top = max(v.probs, key=v.probs.get)
        if (top in (Label.UPDATES, Label.REPLACES) and v.probs[top] >= self.cfg.jev_stop
                and v.p_covered >= self.cfg.jev_stop):
            return "covered"
        return "llm"

    def link(self, p: float | None) -> str:
        """'edge' | 'llm' | 'none'"""
        if p is None:
            return "llm"
        if p >= self.cfg.jev_edge:
            return "edge"
        return "llm" if p > self.cfg.jev_edge_low else "none"

    def impact(self, v: ImpactVerdict | None) -> str:
        """'stop' | 'stale' | 'rewrite' | 'llm'"""
        if v is None:
            return "llm"
        p0 = v.probs.get(0, 0.0)
        if p0 >= self.cfg.jev_stop:
            return "stop"
        if p0 > self.cfg.jev_uncertain_low:
            return "llm"
        return "rewrite" if v.p_known >= self.cfg.jev_known else "stale"


# --------------------------------------------------------------------------- #
# Jev (TypeSafe) decider
# --------------------------------------------------------------------------- #

_LABELS = {
    Label.UPDATES.value: "Same attribute of the same entity, and a new value is given.",
    Label.CONTRADICTS.value: "Negates `existing[i]` but gives no replacement value.",
    Label.PARTIALLY_UPDATES.value: "Part of `existing[i]` is affected and needs a rewrite, not deletion.",
    Label.EXTENDS.value: "Adds detail; `existing[i]` is still fully valid.",
    Label.REPLACES.value: "A procedure or policy in `existing[i]` is superseded by a new one.",
    Label.UNRELATED.value: ("A different attribute, or no validity interaction. Sharing a place, name, "
                            "number or topic is not a conflict."),
}

_IMPACT = [
    "Unaffected: the memory is still fully true after the change.",
    "Partly affected: the memory is mostly true but one detail should be reconfirmed.",
    "Value changed: the memory's value is now different or unknown.",
    "Invalid: the memory is no longer true at all.",
]


def _chunks(items: Sequence[str], n: int):
    for start in range(0, len(items), n):
        yield start, items[start:start + n]


class JevDecider:
    """Batched typed questions to Jev via typesafe-sdk (`pip install gem-memory[jev]`).

    One request per chunk of at most `batch` items — Jev gets worse with long context. Raises on
    transport/API errors; the engine catches them and falls back to the LLM path."""

    def __init__(self, client=None, *, model: str | None = None, batch: int = 8):
        if client is None:
            from typesafe_sdk import TypeSafeClient     # optional dependency
            client = TypeSafeClient(model=model or os.environ.get("TYPESAFE_DEFAULT_MODEL") or None)
        self.client = client
        self.batch = batch
        self.calls = 0

    def _ask(self, state, questions):
        self.calls += 1
        return self.client.system_one(state=state, questions=questions)

    def conflicts(self, new_fact, existing):
        from typesafe_sdk import Choice, Noul
        out: list[ConflictVerdict | None] = []
        for _, chunk in _chunks(list(existing), self.batch):
            qs = {}
            for i in range(len(chunk)):
                qs[f"c{i}_label"] = Choice(
                    instructions=(f"How does `new_fact` DIRECTLY affect the validity of `existing[{i}]`? "
                                  "A conflict needs the same attribute of the same entity."),
                    criteria=_LABELS)
                qs[f"c{i}_covered"] = Noul(
                    instructions=(f"Does `new_fact` alone state everything `existing[{i}]` said, with the "
                                  f"new value, so `existing[{i}]` can be replaced by it without losing "
                                  "any information?"),
                    criteria={"true": "Nothing in the memory would be lost.",
                              "false": "The memory has other details, or is not about the same attribute."})
            resp = self._ask({"new_fact": new_fact, "existing": list(chunk)}, qs)
            for i in range(len(chunk)):
                lab, cov = resp.choices.get(f"c{i}_label"), resp.nouls.get(f"c{i}_covered")
                if lab is None or cov is None:
                    out.append(None)
                    continue
                probs = {Label(k): float(p) for k, p in lab.probabilities.items() if k in _LABELS}
                out.append(ConflictVerdict(probs=probs, p_covered=float(cov.noul)))
        return out

    def depends(self, new_fact, candidates):
        from typesafe_sdk import Noul
        out: list[float | None] = []
        for _, chunk in _chunks(list(candidates), self.batch):
            qs = {f"d{i}_depends": Noul(
                instructions=(f"Was `new_fact` worked out from `candidates[{i}]`, so that if "
                              f"`candidates[{i}]` changed, `new_fact` might no longer hold? "
                              "Sharing a topic is not enough."),
                criteria={"true": "new_fact depends on this candidate.",
                          "false": "new_fact does not depend on this candidate."})
                for i in range(len(chunk))}
            resp = self._ask({"new_fact": new_fact, "candidates": list(chunk)}, qs)
            for i in range(len(chunk)):
                a = resp.nouls.get(f"d{i}_depends")
                out.append(None if a is None else float(a.noul))
        return out

    def impact(self, change, dependents):
        from typesafe_sdk import Noul, Score
        out: list[ImpactVerdict | None] = []
        for _, chunk in _chunks(list(dependents), self.batch):
            qs = {}
            for i in range(len(chunk)):
                qs[f"d{i}_affected"] = Score(
                    instructions=(f"Which specific property of the changed fact does `dependents[{i}]` "
                                  "depend on (often a category such as country, time zone or brand, not "
                                  "the exact value), and did `change` alter it?"),
                    criteria=_IMPACT)
                qs[f"d{i}_value_known"] = Noul(
                    instructions=(f"If `dependents[{i}]` is affected, does `change` give enough "
                                  "information to state its new value?"),
                    criteria={"true": "The new value follows from the change.",
                              "false": "The new value is unknown, or the memory is unaffected."})
            state = {"change": change,
                     "note": "Each memory in `dependents` was derived from the changed fact.",
                     "dependents": list(chunk)}
            resp = self._ask(state, qs)
            for i in range(len(chunk)):
                sc, kn = resp.scores.get(f"d{i}_affected"), resp.nouls.get(f"d{i}_value_known")
                if sc is None or kn is None:
                    out.append(None)
                    continue
                out.append(ImpactVerdict(probs={int(k): float(p) for k, p in sc.probabilities.items()},
                                         p_known=float(kn.noul)))
        return out


class FakeDecider:
    """Scriptable stand-in for tests. Each fn maps one item to its verdict (or raises)."""

    def __init__(self, conflict_fn=None, depends_fn=None, impact_fn=None):
        self._c, self._d, self._i = conflict_fn, depends_fn, impact_fn
        self.calls = 0

    def conflicts(self, new_fact, existing):
        self.calls += 1
        return [self._c(e, new_fact) if self._c else None for e in existing]

    def depends(self, new_fact, candidates):
        self.calls += 1
        return [self._d(c, new_fact) if self._d else None for c in candidates]

    def impact(self, change, dependents):
        self.calls += 1
        return [self._i(d, change) if self._i else None for d in dependents]
