"""Offline self-test: validates everything that does NOT need Ollama.

Run: python -m unit0.selftest
Covers: SubEM scoring, data loading + chunking, and the keep-latest resolve logic
(using a stub LLM so no model is required).
"""

from __future__ import annotations

from . import scorer
from .data import LoadConfig, load_record
from .pipeline import Unit0Pipeline, PipelineConfig


def test_scorer():
    assert scorer.sub_em("The answer is Berlin.", "Berlin")
    assert scorer.sub_em("berlin", "Berlin")              # case-insensitive
    assert not scorer.sub_em("Paris", "Berlin")
    assert scorer.sub_em_any("It is football.", ["soccer", "football"])
    r = scorer.score(["Berlin", "Paris"], [["Berlin"], ["London"]])
    assert r["correct"] == 1 and r["total"] == 2, r
    print("scorer: OK")


def test_data_loading():
    rec = load_record(LoadConfig(size="6k", max_questions=5))
    assert rec.chunks, "no chunks produced"
    assert len(rec.questions) == 5
    assert len(rec.answers) == 5
    assert all(isinstance(a, list) for a in rec.answers), "answers must be lists"
    # context reassembled from chunks should preserve the conflicting facts
    joined = "\n".join(rec.chunks)
    assert "Hines Ward" in joined
    print(f"data: OK ({len(rec.chunks)} chunks for sh_6k, "
          f"{sum(len(c) for c in rec.chunks)} chars total)")
    return rec


class _StubLLM:
    """Stands in for OllamaClient: extraction returns nothing (we drive resolve
    directly), resolve always says UPDATES (keep latest)."""
    def chat_json(self, system, user):
        return {"label": "UPDATES", "current_value": ""}
    def chat(self, system, user, json_mode=False):
        return ""


def test_resolve_keeps_latest():
    pipe = Unit0Pipeline(_StubLLM(), PipelineConfig())
    state = {}
    pipe.resolve(state, {"entity": "Hines Ward", "attribute": "position", "value": "wide receiver"})
    pipe.resolve(state, {"entity": "Hines Ward", "attribute": "position", "value": "cornerback"})
    k = ("hines ward", "position")
    # stub returns current_value="" -> falls back to new_value on UPDATES
    assert state[k]["value"] == "cornerback", state
    # unrelated attribute coexists
    pipe.resolve(state, {"entity": "Germany", "attribute": "continent", "value": "Europe"})
    assert ("germany", "continent") in state
    print("resolve keep-latest: OK")


if __name__ == "__main__":
    test_scorer()
    test_resolve_keeps_latest()
    test_data_loading()
    print("\nAll offline self-tests passed.")
