"""Loader for the FactConsolidation single-hop data (MemoryAgentBench
Conflict_Resolution split, rows whose qa_pair_ids carry the `_sh_` family).

Verified structure (see unit0/inspect_data*.py for the derivation):

  split  Conflict_Resolution, 8 rows
    rows 0-3 : factconsolidation_mh_{6k,32k,64k,262k}   (multi-hop)
    rows 4-7 : factconsolidation_sh_{6k,32k,64k,262k}   (SINGLE-HOP)

  row fields:
    context   : "Here is a list of facts:\n0. <fact>\n1. <fact>\n..."  (~456 facts)
    questions : list[str]   (100)
    answers   : list[list[str]]   (100; each a list of acceptable gold strings)

Conflicts are buried as later-overrides-earlier: the same (subject, relation) is
stated more than once with different values, and the LATEST statement is the gold.
The benchmark streams the context in, so we chunk it on fact boundaries and feed the
chunks sequentially to the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

from .pipeline import Record

REPO = "ai-hyz/MemoryAgentBench"
SPLIT = "Conflict_Resolution"

# Map a friendly size name to its single-hop row index.
SH_ROWS = {"6k": 4, "32k": 5, "64k": 6, "262k": 7}
MH_ROWS = {"6k": 0, "32k": 1, "64k": 2, "262k": 3}

# ~4 chars per token; keep a chunk well under the model's num_ctx so the extraction
# prompt + chunk fit comfortably. 12k chars ≈ 3k tokens.
DEFAULT_CHUNK_CHARS = 12_000


@dataclass
class LoadConfig:
    size: str = "6k"            # which single-hop context size
    multi_hop: bool = False     # Unit 0 stays single-hop; flag is for later units
    chunk_chars: int = DEFAULT_CHUNK_CHARS
    max_questions: int | None = None   # cap Q/A for fast iteration


def _chunk_facts(context: str, chunk_chars: int) -> list[str]:
    """Split the fact list into chunks on line boundaries, never mid-fact."""
    lines = context.split("\n")
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for ln in lines:
        if size + len(ln) + 1 > chunk_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(ln)
        size += len(ln) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _normalize_answers(raw) -> list[list[str]]:
    """Each answer should be a list[str] of acceptable golds; coerce stragglers."""
    out = []
    for a in raw:
        if isinstance(a, list):
            out.append([str(x) for x in a])
        else:
            out.append([str(a)])
    return out


def load_record(cfg: LoadConfig | None = None) -> Record:
    cfg = cfg or LoadConfig()
    rows = MH_ROWS if cfg.multi_hop else SH_ROWS
    if cfg.size not in rows:
        raise ValueError(f"size must be one of {list(rows)}; got {cfg.size!r}")
    idx = rows[cfg.size]

    ds = load_dataset(REPO, split=SPLIT)
    row = ds[idx]

    chunks = _chunk_facts(row["context"], cfg.chunk_chars)
    questions = list(row["questions"])
    answers = _normalize_answers(row["answers"])

    if cfg.max_questions is not None:
        questions = questions[: cfg.max_questions]
        answers = answers[: cfg.max_questions]

    family = "mh" if cfg.multi_hop else "sh"
    return Record(
        chunks=chunks,
        questions=questions,
        answers=answers,
        record_id=f"factconsolidation_{family}_{cfg.size}",
    )
