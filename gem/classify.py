"""The classify primitive and the derive_links pass — the two LLM calls the cascade rests on.

classify(existing, new) -> (Label, revised_content|None)
derive_links(new, candidates) -> [candidate_id, ...]

Both reuse the Unit 0 Ollama client (default model gpt-oss:120b-cloud), which Unit 0
validated for exactly this kind of structured extraction/classification.
"""

from __future__ import annotations

import time
from enum import Enum

from .llm import OllamaClient
from . import prompts
from .store import Node


# Integrity counter: how many LLM calls fell back to their safe default (exhausted retries),
# tagged BY CALLSITE. The two failures corrupt different layers and must be told apart:
#   classify     -> a degraded UNRELATED looks exactly like a legitimate semantic stop,
#                   silently corrupting a cascade DECISION.
#   derive_links -> a degraded [] means a DERIVED_FROM edge silently never forms, corrupting
#                   GRAPH CONSTRUCTION at ingest; the scenario then fails looking like a logic
#                   bug, not a rate-limit artifact.
# Any nonzero total means the run is invalid and must be discarded — not excused.
DEGRADED = {"classify": 0, "derive_links": 0}


def reset_degraded() -> None:
    DEGRADED["classify"] = 0
    DEGRADED["derive_links"] = 0


def degraded_total() -> int:
    return DEGRADED["classify"] + DEGRADED["derive_links"]


def _robust_json(llm: OllamaClient, system: str, user: str,
                 *, retries: int = 2, backoff: float = 1.5) -> dict | None:
    """Wrap chat_json with retry + graceful degradation. Local/cloud models throw
    malformed JSON and transient HTTP errors in normal use; the cascade must survive a
    single bad call rather than crash a long run. Returns None when all attempts fail —
    callers degrade to a safe default (UNRELATED / no links) so propagation stops cleanly
    instead of exploding."""
    last = None
    for attempt in range(retries + 1):
        try:
            return llm.chat_json(system, user)
        except Exception as e:                       # JSON parse OR request/HTTP error
            last = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    return None


class Label(str, Enum):
    CONTRADICTS = "CONTRADICTS"
    UPDATES = "UPDATES"
    PARTIALLY_UPDATES = "PARTIALLY_UPDATES"
    EXTENDS = "EXTENDS"
    REPLACES = "REPLACES"
    UNRELATED = "UNRELATED"


# labels that mean "this memory's validity is affected and the cascade should continue"
INVALIDATING = {Label.CONTRADICTS, Label.UPDATES, Label.PARTIALLY_UPDATES, Label.REPLACES}


def _parse_classify(out: dict) -> tuple[Label, str | None]:
    raw_label = str(out.get("label", "UNRELATED")).upper().strip()
    try:
        label = Label(raw_label)
    except ValueError:
        label = Label.UNRELATED
    revised = out.get("revised_content")
    if isinstance(revised, str):
        revised = revised.strip() or None
        if revised and revised.lower() in ("null", "none", "unknown"):
            revised = None
    else:
        revised = None
    return label, revised


def classify(llm: OllamaClient, existing_content: str, new_fact: str) -> tuple[Label, str | None]:
    """Single LLM call. Returns (label, revised_content). revised_content is the proposed
    corrected text for UPDATES/PARTIALLY_UPDATES, or None when the new value is unknown
    or the label doesn't rewrite the memory."""
    out = _robust_json(
        llm,
        prompts.CLASSIFY_SYSTEM,
        prompts.CLASSIFY_USER.format(existing=existing_content, new=new_fact),
    )
    if out is None:                       # degrade safely: no confident conflict -> stop branch
        DEGRADED["classify"] += 1
        return Label.UNRELATED, None
    return _parse_classify(out)


def classify_consistent(llm: OllamaClient, existing_content: str, new_fact: str,
                        *, samples: int = 3, temperature: float = 0.5,
                        certainty: float = 0.6,
                        seed: tuple[Label, str | None] | None = None
                        ) -> tuple[Label, str | None, bool]:
    """Self-consistency voting for small/noisy models. Majority-votes the label over
    `samples` samples (a prior `seed` decision counts as the first vote, so escalation
    callers don't waste their first pass). Returns (label, revised, certain); `certain` is
    False when the model can't agree with itself (top label below `certainty`) — the caller
    routes uncertain decisions to the fail-safe (STALE + needs_review).

    Two jobs, with different strength: it STABILISES noisy calls (only when the model is
    right >50% per sample — an unconfirmed precondition on any given model), and it DETECTS
    uncertainty via disagreement (nearly unconditional — only needs the model to be
    inconsistent when unsure). The detector is the part that stands on its own."""
    from collections import Counter
    sys_p = prompts.CLASSIFY_SYSTEM
    usr_p = prompts.CLASSIFY_USER.format(existing=existing_content, new=new_fact)
    labels: list[Label] = []
    revised_for: dict[Label, str | None] = {}
    if seed is not None:
        labels.append(seed[0])
        revised_for.setdefault(seed[0], seed[1])
    for i in range(max(0, samples - len(labels))):
        try:
            out = llm.chat_json(sys_p, usr_p, retries=0, temperature=temperature)
        except Exception:
            continue
        lbl, rev = _parse_classify(out)
        labels.append(lbl)
        revised_for.setdefault(lbl, rev)
    if not labels:
        DEGRADED["classify"] += 1
        return Label.UNRELATED, None, False
    counts = Counter(labels)
    top, n = counts.most_common(1)[0]
    certain = (n / len(labels)) >= certainty
    return top, revised_for.get(top), certain


def derive_links(llm: OllamaClient, new_fact: str, candidates: list[Node],
                 confirm: bool = False) -> list[str]:
    """Which candidate node ids the new fact is causally DERIVED FROM. Distinct from
    classify: dependence, not conflict.

    `confirm=True` adds a precision pass: each proposed edge is re-checked in isolation and
    dropped unless the strict change-test confirms it. Costs +1 call per proposed edge; trades a
    little recall for precision (fewer spurious edges -> less over-cascade)."""
    if not candidates:
        return []
    block = "\n".join(f"  {c.id}: {c.content}" for c in candidates)
    out = _robust_json(
        llm,
        prompts.DERIVE_SYSTEM,
        prompts.DERIVE_USER.format(new=new_fact, candidates=block),
    )
    if out is None:                       # degrade safely: no inferred dependencies
        DEGRADED["derive_links"] += 1
        return []
    ids = out.get("derived_from", []) if isinstance(out, dict) else []
    valid = {c.id for c in candidates}
    proposed = [str(i) for i in ids if str(i) in valid]
    if not confirm or not proposed:
        return proposed
    id2c = {c.id: c for c in candidates}
    kept = []
    for cid in proposed:                  # precision pass: confirm each edge in isolation
        v = _robust_json(
            llm,
            prompts.DERIVE_CONFIRM_SYSTEM,
            prompts.DERIVE_CONFIRM_USER.format(new=new_fact, cand=id2c[cid].content),
        )
        if v is None:                     # degrade safely: keep the edge rather than wrongly drop
            DEGRADED["derive_links"] += 1
            kept.append(cid)
        elif v.get("depends"):
            kept.append(cid)
    return kept
