"""The ask-the-user loop: a fact goes stale, the agent asks, and the user either gives the new value
(resolve) or says the old one still holds (confirm). Mock LLM + embedder, no network."""

import pytest

from gem.json_store import JsonStore
from gem.memory import Memory
from gem.store import Status
from gem.tools import GemTools
from conftest import FakeLLM, FakeEmbedder, classify_json
from test_engine import _existing, _is_change_desc


def responder(system, user):
    """Location conflicts with a move; the commute, alarm and rent depend on it and their new values
    are unknown; an answer about the commute updates the stale commute and nothing else."""
    if "derived_from" in system:
        return {"derived_from": []}
    existing = _existing(user)
    new = user.split("NEW statement:", 1)[-1]
    if _is_change_desc(user):
        if any(w in existing for w in ("commute", "alarm", "rent")):
            return classify_json("UPDATES", None)          # value unknown -> stale
        return classify_json("UNRELATED")
    if "live in" in existing and ("moved" in new or "live in" in new):
        return classify_json("UPDATES", "I live in " + new.strip().split()[-1])
    if "commute" in existing and "commute" in new:
        return classify_json("UPDATES", new.strip())
    return classify_json("UNRELATED")


def _mem(tmp_path=None):
    store = JsonStore(str(tmp_path / "m.json")) if tmp_path else None
    m = Memory(llm=FakeLLM(json_fn=responder), embedder=FakeEmbedder(), store=store)
    ids = {}
    ids["loc"] = m.load("I live in Bangalore")
    ids["com"] = m.load("My commute is 45 minutes", derived_from=[ids["loc"]])
    ids["alarm"] = m.load("My alarm is set for 6:30 am", derived_from=[ids["com"]])
    ids["rent"] = m.load("My rent is 30,000 rupees", derived_from=[ids["loc"]])
    m.add("I moved to Mumbai", derived_from=[])
    return m, ids


def test_move_leaves_facts_waiting_for_an_answer():
    m, ids = _mem()
    waiting = {f.id for f in m.stale}
    assert waiting == {ids["com"], ids["alarm"], ids["rent"]}
    assert all(f.needs_review for f in m.stale)               # every stale fact is flagged


def test_answer_resolves_the_stale_fact_and_takes_its_place():
    m, ids = _mem()
    r = m.add("My commute is 30 minutes", resolves=[ids["com"]])
    assert [f.id for f in r.resolved] == [ids["com"]]
    assert m.get(ids["com"]).status == "SUPERSEDED"
    assert ids["com"] not in {f.id for f in m.stale}         # no longer asked about
    s = m._g.store
    assert [p.id for p in s.derived_from_targets(r.id)] == [ids["loc"]]       # inherits parent
    assert [p.id for p in s.derived_from_targets(ids["alarm"])] == [r.id]     # dependents re-linked
    assert [f.id for f in r.reconfirm] == [ids["alarm"]]     # still needs its own answer
    assert ids["rent"] in {f.id for f in m.stale}             # unrelated stale fact untouched


def test_answer_is_matched_to_the_stale_fact_automatically():
    m, ids = _mem()
    r = m.add("My commute is 30 minutes")                    # no resolves given
    assert [f.id for f in r.resolved] == [ids["com"]]
    assert {f.id for f in m.stale} == {ids["alarm"], ids["rent"]}


def test_a_later_change_reaches_the_answer():
    m, ids = _mem()
    new = m.add("My commute is 30 minutes", resolves=[ids["com"]]).id
    m.add("I moved to Pune", derived_from=[])
    assert m.get(new).status == "STALE"                      # the cascade still finds it


def test_confirm_restores_the_fact_and_what_went_stale_because_of_it():
    m, ids = _mem()
    r = m.confirm(ids["com"])
    assert m.get(ids["com"]).status == "ACTIVE" and not m.get(ids["com"]).needs_review
    assert [f.id for f in r.restored] == [ids["alarm"]]      # stale only because the commute was
    assert m.get(ids["alarm"]).status == "ACTIVE"
    assert m.get(ids["rent"]).status == "STALE"              # stale because of the move: unchanged


def test_resolution_survives_a_restart(tmp_path):
    m, ids = _mem(tmp_path)
    new = m.add("My commute is 30 minutes", resolves=[ids["com"]]).id
    m2 = Memory(llm=FakeLLM(json_fn=responder), embedder=FakeEmbedder(),
                store=JsonStore(str(tmp_path / "m.json")))
    assert m2.get(ids["com"]).status == "SUPERSEDED"
    assert [p.id for p in m2._g.store.derived_from_targets(ids["alarm"])] == [new]
    assert {f.id for f in m2.stale} == {ids["alarm"], ids["rent"]}


def test_unknown_resolves_id_is_an_error():
    m, _ = _mem()
    with pytest.raises(KeyError):
        m.add("My commute is 30 minutes", resolves=["n999"])


def test_tool_flow_ask_answer_confirm(tmp_path, monkeypatch):
    monkeypatch.setenv("GEM_HOME", str(tmp_path))
    t = GemTools(llm=FakeLLM(json_fn=responder), embedder=FakeEmbedder(), store_kind="json")
    loc = t.call("add_memory", {"content": "I live in Bangalore", "derived_from": []})["id"]
    com = t.call("add_memory", {"content": "My commute is 45 minutes", "derived_from": [loc]})["id"]
    alarm = t.call("add_memory", {"content": "My alarm is set for 6:30 am", "derived_from": [com]})["id"]
    t.call("add_memory", {"content": "I moved to Mumbai", "derived_from": []})

    found = t.call("search_memory", {"query": "commute", "limit": 10})
    assert com in [f["id"] for f in found["stale"]]           # the agent sees what to ask

    r = t.call("add_memory", {"content": "My commute is 30 minutes", "resolves": [com]})
    assert [f["id"] for f in r["resolved"]] == [com]
    assert [f["id"] for f in r["reconfirm"]] == [alarm]
    assert com not in [f["id"] for f in t.call("get_stale")["stale"]]
    assert "My commute is 30 minutes" in [f["content"] for f in
                                          t.call("search_memory", {"query": "commute", "limit": 10})["results"]]

    c = t.call("add_memory", {"action": "confirm", "fact_id": alarm})
    assert c["confirmed"]["status"] == "ACTIVE"
    assert alarm not in [f["id"] for f in t.call("get_stale")["stale"]]
    assert "error" in t.call("add_memory", {"action": "confirm"})              # needs fact_id
    assert "error" in t.call("add_memory", {"content": "x", "resolves": ["n999"]})
    assert "error" in t.call("add_memory", {"content": ""})                     # nothing to save
