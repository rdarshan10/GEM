"""Unit 0 pipeline (the gate): extract -> resolve (keep-latest) -> answer, with a mock LLM."""

import pytest

from unit0.pipeline import Unit0Pipeline, Record
from conftest import FakeLLM


def test_extract_keeps_well_formed_triples():
    facts = {"facts": [
        {"entity": "Germany", "attribute": "capital", "value": "Berlin"},
        {"entity": "", "attribute": "x", "value": "y"},          # bad: no entity -> dropped
        {"attribute": "no entity"},                              # malformed -> dropped
    ]}
    p = Unit0Pipeline(FakeLLM(json_fn=lambda s, u: facts))
    out = p.extract("some chunk")
    assert out == [{"entity": "Germany", "attribute": "capital", "value": "Berlin"}]


def test_extract_survives_bad_json():
    class Boom(FakeLLM):
        def chat_json(self, system, user, **kw):
            raise ValueError("bad")
    assert Unit0Pipeline(Boom()).extract("chunk") == []          # degrades to empty, no crash


def test_resolve_keep_latest():
    # classify says UPDATES with the new value -> the later value wins
    p = Unit0Pipeline(FakeLLM(json_fn=lambda s, u: {"label": "UPDATES", "current_value": "Africa"}))
    state = {}
    p.resolve(state, {"entity": "Germany", "attribute": "continent", "value": "Europe"})
    assert state[("germany", "continent")]["value"] == "Europe"  # first write
    p.resolve(state, {"entity": "Germany", "attribute": "continent", "value": "Africa"})
    assert state[("germany", "continent")]["value"] == "Africa"  # updated to latest


def test_resolve_identical_value_is_noop():
    calls = {"n": 0}

    def jf(s, u):
        calls["n"] += 1
        return {"label": "UPDATES", "current_value": "x"}
    p = Unit0Pipeline(FakeLLM(json_fn=jf))
    state = {}
    p.resolve(state, {"entity": "A", "attribute": "b", "value": "same"})
    p.resolve(state, {"entity": "A", "attribute": "b", "value": "same"})   # identical -> no LLM
    assert calls["n"] == 0


def test_run_record_scores_subem():
    # extraction yields the fact; answer echoes the stored value
    facts = {"facts": [{"entity": "Germany", "attribute": "continent", "value": "Africa"}]}

    def text_fn(system, user):
        return "Africa"                                          # answer step
    p = Unit0Pipeline(FakeLLM(json_fn=lambda s, u: facts, text_fn=text_fn))
    rec = Record(chunks=["chunk"], questions=["Which continent is Germany in?"],
                 answers=[["Africa"]], record_id="t")
    result = p.run_record(rec)
    assert result.score["subem"] == 1.0
