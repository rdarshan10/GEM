"""GEM as a tool: JSON persistence, forget-with-cascade, the tool dispatcher, and the MCP server's
tool list. Mock LLM + embedder, no network."""

import json

import numpy as np
import pytest

from gem.json_store import JsonStore
from gem.memory import Memory
from gem.store import EdgeType, Status
from gem.tools import TOOLS, GemTools, openai_tools
from conftest import FakeLLM, FakeEmbedder, classify_json
from test_engine import mumbai_responder, _existing, _is_change_desc


def move_responder(system, user):
    """Like mumbai_responder, but only the move itself conflicts with the location (so saving
    other facts through the conflict scan leaves it alone)."""
    if "derived_from" in system:
        return {"derived_from": []}
    if not _is_change_desc(user):
        new = user.split("NEW statement:", 1)[-1]
        return classify_json("UPDATES", "I live in Mumbai")             if "Mumbai" in new and "Bangalore" in _existing(user) else classify_json("UNRELATED")
    return mumbai_responder(system, user)


def _unit(i, dim=16):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def _mem(tmp_path, name="m.json", llm=None):
    return Memory(llm=llm or FakeLLM(json_fn=mumbai_responder), embedder=FakeEmbedder(),
                  store=JsonStore(str(tmp_path / name)))


def test_json_store_round_trip(tmp_path):
    m = _mem(tmp_path)
    loc = m.load("I live in Bangalore")
    com = m.load("My commute is 45 minutes", derived_from=[loc])
    m.add("I now live in Mumbai", derived_from=[])

    m2 = _mem(tmp_path)                                   # fresh process, same file
    s = m2._g.store
    assert s.get(loc).content == "I live in Mumbai"
    assert s.get(com).status == Status.STALE
    assert [p.id for p in s.derived_from_targets(com)] == [loc]
    assert isinstance(s.get(loc).embedding, np.ndarray)
    new = m2.load("I like tea")
    assert new not in (loc, com) and int(new[1:]) > 3     # ids continue past the saved ones


def test_forget_cascades_to_dependents(tmp_path):
    def llm_fn(system, user):                             # every dependent depends on the location
        return classify_json("UPDATES", None)
    m = _mem(tmp_path, llm=FakeLLM(json_fn=llm_fn))
    loc = m.load("I live in Bangalore")
    com = m.load("My commute is 45 minutes", derived_from=[loc])
    wake = m.load("I wake at 7am to beat traffic", derived_from=[com])
    r = m.forget(loc)
    assert m.get(loc).status == "SUPERSEDED"
    assert {f.id for f in r.invalidated} == {com, wake}
    with pytest.raises(KeyError):
        m.forget("n999")


def _tools(tmp_path, monkeypatch, llm=None, embedder=None):
    monkeypatch.setenv("GEM_HOME", str(tmp_path))
    return GemTools(llm=llm or FakeLLM(json_fn=move_responder),
                    embedder=embedder or FakeEmbedder(), store_kind="json")


def test_tool_flow_save_search_stale_why(tmp_path, monkeypatch):
    t = _tools(tmp_path, monkeypatch)
    loc = t.call("add_memory", {"content": "I live in Bangalore", "derived_from": []})["id"]
    com = t.call("add_memory", {"content": "My commute is 45 minutes",
                                "derived_from": [loc]})["id"]
    r = t.call("add_memory", {"content": "I now live in Mumbai", "derived_from": []})
    assert [f["id"] for f in r["invalidated"]] == [com]
    assert [f["content"] for f in r["revised"]] == ["I live in Mumbai"]

    found = t.call("search_memory", {"query": "commute", "limit": 10})
    assert com not in [f["id"] for f in found["results"]]            # stale is withheld...
    assert [f["id"] for f in found["stale"]] == [com]                # ...but reported as stale
    assert all("score" in f for f in found["results"])
    assert [f["id"] for f in t.call("get_stale")["stale"]] == [com]
    assert t.call("why", {"fact_id": com})["derived_from"][0]["id"] == loc
    assert (tmp_path / "default.json").exists()


def test_forget_by_content_and_by_id(tmp_path, monkeypatch):
    emb = FakeEmbedder(vectors={"Bangalore": _unit(0), "tea": _unit(1), "drinks": _unit(2)})
    t = _tools(tmp_path, monkeypatch, embedder=emb)
    a = t.call("add_memory", {"content": "I live in Bangalore", "derived_from": []})["id"]
    r = t.call("add_memory", {"content": "I live in Bangalore", "action": "forget"})
    assert r["forgotten"]["id"] == a and r["forgotten"]["status"] == "SUPERSEDED"
    b = t.call("add_memory", {"content": "I like tea", "derived_from": []})["id"]
    vague = t.call("add_memory", {"content": "something about drinks", "action": "forget"})
    assert "error" in vague and "candidates" in vague                 # no close match -> ask for id
    assert t.call("add_memory", {"content": "", "action": "forget",
                                 "fact_id": b})["forgotten"]["id"] == b


def test_containers_are_isolated(tmp_path, monkeypatch):
    t = _tools(tmp_path, monkeypatch)
    t.call("add_memory", {"content": "Work fact", "derived_from": [], "container_tag": "work"})
    assert t.call("list_memories", {"container_tag": "home"})["memories"] == []
    assert len(t.call("list_memories", {"container_tag": "work"})["memories"]) == 1
    assert (tmp_path / "work.json").exists()


def test_bad_input_returns_errors_not_exceptions(tmp_path, monkeypatch):
    t = _tools(tmp_path, monkeypatch)
    assert "error" in t.call("nope")
    assert "error" in t.call("why", {"fact_id": "n42"})
    assert "error" in t.call("list_memories", {"container_tag": "../etc"})
    assert "error" in t.call("add_memory", {"content": "x", "action": "delete"})
    assert "error" in t.call("search_memory", {})                     # missing required arg


def test_tool_definitions_are_consistent():
    names = [t["name"] for t in TOOLS]
    assert names == ["add_memory", "search_memory", "list_memories", "get_profile", "get_stale",
                     "why"]
    assert all(hasattr(GemTools, f"_t_{n}") for n in names)
    assert [f["function"]["name"] for f in openai_tools()] == names
    json.dumps(TOOLS)


def test_mcp_server_lists_the_tools():
    pytest.importorskip("mcp.server.mcpserver")
    import asyncio
    from gem import mcp_server
    listed = asyncio.run(mcp_server.server.list_tools())
    assert sorted(t.name for t in listed) == sorted(t["name"] for t in TOOLS)


# --------------------------------------------------------------------------- #
# get_profile
# --------------------------------------------------------------------------- #

def test_profile_uses_valid_facts_lists_stale_and_caches(tmp_path, monkeypatch):
    prompts_seen = []

    def text_fn(system, user):
        prompts_seen.append(user)
        return "Lives in Mumbai."
    llm = FakeLLM(json_fn=move_responder, text_fn=text_fn)
    t = _tools(tmp_path, monkeypatch, llm=llm)
    assert t.call("get_profile") == {"summary": "", "fact_count": 0, "reconfirm": []}
    loc = t.call("add_memory", {"content": "I live in Bangalore", "derived_from": []})["id"]
    com = t.call("add_memory", {"content": "My commute is 45 minutes", "derived_from": [loc]})["id"]
    t.call("add_memory", {"content": "I now live in Mumbai", "derived_from": []})

    p = t.call("get_profile")
    assert p["summary"] == "Lives in Mumbai." and [f["id"] for f in p["reconfirm"]] == [com]
    assert "commute" not in prompts_seen[-1]                          # stale facts never summarised
    t.call("get_profile")
    assert llm.chat_calls == 1                                        # cached until facts change
    t.call("add_memory", {"content": "I like tea", "derived_from": []})
    t.call("get_profile")
    assert llm.chat_calls == 2


# --------------------------------------------------------------------------- #
# several processes sharing one JSON store
# --------------------------------------------------------------------------- #

def test_two_instances_on_one_file_see_each_other(tmp_path):
    a, b = _mem(tmp_path), _mem(tmp_path)
    x = a.load("I live in Bangalore")
    assert b.get(x).content == "I live in Bangalore"                  # b refreshes on read
    y = b.load("I like tea")
    assert y != x                                                     # b caught up before writing
    assert {f.id for f in a.facts()} == {x, y}


def test_failed_write_is_rolled_back(tmp_path):
    m = _mem(tmp_path)
    x = m.load("I live in Bangalore")

    def boom(*a, **k):
        raise RuntimeError("llm down mid-cascade")
    m._g.ingest = boom
    with pytest.raises(RuntimeError):
        m.add("I now live in Mumbai")
    assert [f.id for f in m.facts()] == [x]
    assert [f.id for f in _mem(tmp_path).facts()] == [x]


def test_lock_blocks_a_second_writer(tmp_path):
    from gem.json_store import _FileLock
    path = str(tmp_path / "x.lock")
    with _FileLock(path, timeout=1):
        with pytest.raises(TimeoutError):
            with _FileLock(path, timeout=0.2):
                pass
    with _FileLock(path, timeout=1):                                  # released afterwards
        pass


_WRITER = r'''
import sys, hashlib, numpy as np
sys.path.insert(0, sys.argv[3])
from gem.memory import Memory
from gem.json_store import JsonStore

class LLM:
    def chat_json(self, system, user, **kw):
        return {"derived_from": []} if "derived_from" in system else {"label": "UNRELATED"}
    def chat(self, *a, **k):
        return ""

class Emb:
    def embed(self, text):
        return np.frombuffer(hashlib.md5(text.encode()).digest(), dtype=np.uint8).astype(np.float32)

m = Memory(llm=LLM(), embedder=Emb(), store=JsonStore(sys.argv[1]))
for i in range(8):
    m.add(f"writer {sys.argv[2]} fact {i}")
'''


def test_concurrent_processes_lose_no_writes(tmp_path):
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = str(tmp_path / "shared.json")
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, path, str(w), root])
             for w in range(4)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    facts = Memory(llm=FakeLLM(), embedder=FakeEmbedder(), store=JsonStore(path)).facts()
    assert len(facts) == 32 and len({f.id for f in facts}) == 32
    assert {f.content for f in facts} == {f"writer {w} fact {i}" for w in range(4) for i in range(8)}
