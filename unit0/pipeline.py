"""The Unit 0 end-to-end chain, operating on an already-chunked record.

    chunk stream  ->  extract facts  ->  resolve conflicts (keep latest correct)
                  ->  answer each query from resolved state

This is deliberately the *thinnest* thing that exercises the hard part: pulling the
right fact out of messy chunk text and keeping the current value as later chunks
silently override earlier ones. No graph, no embeddings, no propagation — Unit 0 is
single-hop only, on purpose.

State is a flat dict keyed by (entity, attribute) -> current value. That is the
single-hop analogue of the cascade: a real keep-latest store with conflict
classification deciding what "latest correct" means.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import requests

from gem.llm import OllamaClient
from . import prompts
from . import scorer


def _log(msg: str) -> None:
    """Flushed progress line — so a slow CPU run never looks frozen."""
    print(msg, flush=True)
    sys.stdout.flush()


@dataclass
class Record:
    """One benchmark row, pre-chunked. `data.py` builds these from the raw dataset."""
    chunks: list[str]
    questions: list[str]
    answers: list[str]
    record_id: str = ""


@dataclass
class PipelineConfig:
    verbose: bool = False
    # cap facts kept per (entity, attribute) history; we only need the current value
    log_extractions: bool = False


@dataclass
class RunResult:
    record_id: str
    predictions: list[str]
    golds: list[str]
    score: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)


def _key(entity: str, attribute: str) -> tuple[str, str]:
    return (entity.strip().lower(), attribute.strip().lower())


class Unit0Pipeline:
    def __init__(self, llm: OllamaClient, config: PipelineConfig | None = None):
        self.llm = llm
        self.cfg = config or PipelineConfig()

    # --- step 1: extract atomic facts from a chunk --------------------------- #
    def extract(self, chunk: str) -> list[dict]:
        try:
            out = self.llm.chat_json(
                prompts.EXTRACT_SYSTEM,
                prompts.EXTRACT_USER_TEMPLATE.format(chunk=chunk),
            )
        except (ValueError, requests.exceptions.RequestException) as e:
            # a slow/failed call skips this chunk instead of killing the whole run
            _log(f"    [warn] extract failed ({type(e).__name__}); skipping chunk")
            return []
        facts = out.get("facts", []) if isinstance(out, dict) else []
        # keep only well-formed triples
        clean = []
        for f in facts:
            if isinstance(f, dict) and f.get("entity") and f.get("attribute"):
                clean.append(
                    {
                        "entity": str(f["entity"]),
                        "attribute": str(f["attribute"]),
                        "value": str(f.get("value", "")),
                    }
                )
        return clean

    # --- step 2: resolve a new fact against current state -------------------- #
    def resolve(self, state: dict, fact: dict) -> None:
        k = _key(fact["entity"], fact["attribute"])
        new_value = fact["value"]
        if k not in state:
            state[k] = {
                "entity": fact["entity"],
                "attribute": fact["attribute"],
                "value": new_value,
            }
            return
        current = state[k]["value"]
        if _norm(current) == _norm(new_value):
            return  # identical restatement, nothing to do
        try:
            out = self.llm.chat_json(
                prompts.RESOLVE_SYSTEM,
                prompts.RESOLVE_USER_TEMPLATE.format(
                    entity=fact["entity"],
                    attribute=fact["attribute"],
                    current_value=current,
                    new_value=new_value,
                ),
            )
        except (ValueError, requests.exceptions.RequestException):
            # parse/timeout failure: default to keep-latest, the dominant case here
            state[k]["value"] = new_value
            return
        label = str(out.get("label", "UPDATES")).upper()
        resolved = str(out.get("current_value", new_value)).strip()
        if label in ("UPDATES", "CONTRADICTS"):
            state[k]["value"] = resolved or new_value
        # EXTENDS / UNRELATED -> leave current value untouched

    # --- step 3: answer a query from resolved state -------------------------- #
    def answer(self, state: dict, question: str) -> str:
        facts_block = _render_state(state)
        try:
            return self.llm.chat(
                prompts.ANSWER_SYSTEM,
                prompts.ANSWER_USER_TEMPLATE.format(
                    facts_block=facts_block, question=question
                ),
            )
        except requests.exceptions.RequestException:
            return ""

    # --- full record run ----------------------------------------------------- #
    def run_record(self, rec: Record) -> RunResult:
        state: dict = {}
        n_chunks = len(rec.chunks)
        for ci, chunk in enumerate(rec.chunks):
            t0 = time.time()
            facts = self.extract(chunk)
            for fact in facts:
                self.resolve(state, fact)
            _log(f"  ingest chunk {ci + 1}/{n_chunks}: "
                 f"+{len(facts)} facts -> {len(state)} keys ({time.time() - t0:.0f}s)")

        _log(f"  resolved state: {len(state)} (entity,attribute) keys; "
             f"answering {len(rec.questions)} questions ...")

        preds = []
        for qi, q in enumerate(rec.questions):
            t0 = time.time()
            pred = self.answer(state, q)
            preds.append(pred)
            if self.cfg.verbose:
                _log(f"  Q{qi + 1}/{len(rec.questions)} ({time.time() - t0:.0f}s): "
                     f"{pred[:60]!r}")
        result = RunResult(
            record_id=rec.record_id,
            predictions=preds,
            golds=list(rec.answers),
            score=scorer.score(preds, list(rec.answers)),
            state=state,
        )
        return result


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _render_state(state: dict) -> str:
    lines = []
    for v in state.values():
        lines.append(f"- {v['entity']} | {v['attribute']}: {v['value']}")
    return "\n".join(lines) if lines else "(no facts)"
