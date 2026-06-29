"""classify primitive, derive_links, voting, and the degraded-call integrity counter —
driven by the mock LLM."""

import pytest

from gem import classify as C
from gem.store import Node
from conftest import FakeLLM, classify_json


def test_parse_classify_labels():
    assert C._parse_classify({"label": "UPDATES", "revised_content": "x"}) == (C.Label.UPDATES, "x")
    assert C._parse_classify({"label": "updates", "revised_content": "x"})[0] == C.Label.UPDATES
    assert C._parse_classify({"label": "GARBAGE"})[0] == C.Label.UNRELATED      # unknown -> UNRELATED
    assert C._parse_classify({"label": "UPDATES", "revised_content": "null"})[1] is None
    assert C._parse_classify({"label": "UPDATES", "revised_content": "  "})[1] is None


def test_classify_returns_parsed():
    llm = FakeLLM(json_fn=lambda s, u: classify_json("CONTRADICTS"))
    label, revised = C.classify(llm, "old", "new")
    assert label == C.Label.CONTRADICTS and revised is None
    assert llm.json_calls == 1


def test_classify_degrades_on_failure():
    C.reset_degraded()

    class Boom(FakeLLM):
        def chat_json(self, system, user, **kw):
            raise ValueError("bad json")

    label, revised = C.classify(Boom(), "old", "new")
    assert label == C.Label.UNRELATED                 # safe default
    assert C.DEGRADED["classify"] == 1                # counted


def test_derive_links_filters_to_valid_ids():
    cands = [Node(id="n1", content="a"), Node(id="n2", content="b")]
    llm = FakeLLM(json_fn=lambda s, u: {"derived_from": ["n1", "bogus"], "reasoning": ""})
    assert C.derive_links(llm, "new fact", cands) == ["n1"]   # bogus dropped
    assert C.derive_links(llm, "new fact", []) == []          # no candidates -> no call


def test_classify_consistent_majority_and_certainty():
    seq = ["UPDATES", "UPDATES", "UNRELATED"]            # 2/3 majority -> UPDATES, not unanimous
    state = {"i": 0}

    def responder(s, u):
        lbl = seq[state["i"] % len(seq)]
        state["i"] += 1
        return classify_json(lbl)

    label, revised, certain = C.classify_consistent(FakeLLM(json_fn=responder), "old", "new",
                                                    samples=3, certainty=0.6)
    assert label == C.Label.UPDATES
    assert certain is True                              # 2/3 >= 0.6


def test_classify_consistent_split_is_uncertain():
    seq = ["UPDATES", "UNRELATED", "CONTRADICTS"]        # 1/3 each -> not certain
    state = {"i": 0}

    def responder(s, u):
        lbl = seq[state["i"] % len(seq)]; state["i"] += 1
        return classify_json(lbl)

    _, _, certain = C.classify_consistent(FakeLLM(json_fn=responder), "old", "new",
                                          samples=3, certainty=0.6)
    assert certain is False
