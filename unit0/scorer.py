"""SubEM scorer — the metric MemoryAgentBench uses for Conflict_Resolution.

SubEM ("substring exact match"): the prediction is correct if the normalized gold
answer appears as a substring of the normalized prediction. This is more forgiving
than strict EM (the model may wrap the entity in a sentence) but still requires the
exact gold entity to be present — which is the point of a conflict benchmark: did
you land on the *current correct value*, not an outdated one.

Normalization mirrors the SQuAD-style cleanup: lowercase, strip punctuation, drop
articles, collapse whitespace.
"""

from __future__ import annotations

import re
import string


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sub_em(prediction: str, gold: str) -> bool:
    """True if normalized gold is a substring of normalized prediction."""
    pred_n = normalize(prediction)
    gold_n = normalize(gold)
    if not gold_n:
        return False
    return gold_n in pred_n


def sub_em_any(prediction: str, golds: list[str]) -> bool:
    """SubEM against a list of acceptable gold answers (any match counts)."""
    return any(sub_em(prediction, g) for g in golds)


def score(predictions: list[str], golds) -> dict:
    """Aggregate SubEM over aligned prediction/gold lists. Returns accuracy + counts.

    Each gold may be a single string or a list of acceptable strings (any match
    counts) — FactConsolidation answers are lists like ["Belgium"].
    """
    assert len(predictions) == len(golds), "predictions/golds length mismatch"
    hits = []
    for p, g in zip(predictions, golds):
        golds_i = g if isinstance(g, list) else [g]
        hits.append(sub_em_any(p, golds_i))
    n = len(hits)
    correct = sum(hits)
    return {
        "subem": correct / n if n else 0.0,
        "correct": correct,
        "total": n,
        "hits": hits,
    }
