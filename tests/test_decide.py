"""Jev decider layer: TieredJudge routing for every row of the plan's table, the two safety rules
(stopping needs high confidence; Jev never supersedes a dependent), LLM fallback when Jev fails, and
JevDecider's request/response handling against a fake typesafe client."""

from types import SimpleNamespace

import pytest

from gem import classify as C
from gem.classify import Label
from gem.decide import (ConflictVerdict, FakeDecider, ImpactVerdict, JevDecider, TieredJudge)
from gem.engine import GEM, GEMConfig
from gem.store import Status
from conftest import FakeLLM, FakeEmbedder, classify_json

judge = TieredJudge(GEMConfig())


def cv(label, p=0.97, covered=0.0):
    rest = (1 - p) / 5
    return ConflictVerdict(probs={l: (p if l == label else rest) for l in Label}, p_covered=covered)


def iv(p0, p_known=0.1, top=2):
    probs = {0: p0, 1: 0.0, 2: 0.0, 3: 0.0}
    probs[top] += 1 - p0
    return ImpactVerdict(probs=probs, p_known=p_known)


# --------------------------------------------------------------------------- #
# routing table
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("verdict,expected", [
    (cv(Label.UNRELATED), "skip"),
    (cv(Label.EXTENDS), "skip"),                          # non-invalidating counts as no conflict
    (cv(Label.UPDATES, covered=0.95), "covered"),
    (cv(Label.UPDATES, covered=0.3), "llm"),              # compound fact: LLM writes the rewrite
    (cv(Label.CONTRADICTS), "llm"),                       # destructive: LLM confirms
    (cv(Label.UNRELATED, p=0.7), "llm"),                  # unsure
    (None, "llm"),
])
def test_conflict_routing(verdict, expected):
    assert judge.conflict(verdict) == expected


@pytest.mark.parametrize("p,expected", [(0.9, "edge"), (0.6, "edge"), (0.4, "llm"),
                                        (0.1, "none"), (None, "llm")])
def test_link_routing(p, expected):
    assert judge.link(p) == expected


@pytest.mark.parametrize("verdict,expected", [
    (iv(0.97), "stop"),
    (iv(0.05, p_known=0.1), "stale"),
    (iv(0.05, p_known=0.9), "rewrite"),
    (iv(0.79), "llm"),                                    # leaning unaffected, not sure enough to stop
    (iv(0.89), "llm"),
    (None, "llm"),
])
def test_impact_routing(verdict, expected):
    assert judge.impact(verdict) == expected


# --------------------------------------------------------------------------- #
# engine integration
# --------------------------------------------------------------------------- #

def _graph(decider, llm=None):
    g = GEM(llm=llm or FakeLLM(), embedder=FakeEmbedder(), decider=decider)
    loc = g.ingest("I live in Bangalore", parents=[], check_conflicts=False)
    com = g.ingest("My commute is 45 minutes", parents=[loc.id], check_conflicts=False)
    wake = g.ingest("I wake at 7am to beat traffic", parents=[com.id], check_conflicts=False)
    tz = g.ingest("My timezone is IST", parents=[loc.id], check_conflicts=False)
    return g, dict(loc=loc, com=com, wake=wake, tz=tz)


def _mumbai_decider(**over):
    def conflict(existing, new):
        return cv(Label.UPDATES, covered=0.95) if "Bangalore" in existing else cv(Label.UNRELATED)

    def impact(dep, change):
        if "timezone" in dep:
            return iv(0.97)
        return over.get("impact", iv(0.03))
    return FakeDecider(conflict_fn=conflict, impact_fn=impact)


def test_jev_runs_the_whole_cascade_without_the_llm():
    llm = FakeLLM()
    g, n = _graph(_mumbai_decider(), llm)
    g.ingest("I now live in Mumbai", parents=[])
    assert g.store.get(n["loc"].id).content == "I now live in Mumbai"   # covered -> rewritten
    assert g.store.get(n["loc"].id).status == Status.ACTIVE
    assert g.store.get(n["com"].id).status == Status.STALE
    assert g.store.get(n["wake"].id).status == Status.STALE              # 2nd hop, still no LLM
    assert g.store.get(n["tz"].id).status == Status.ACTIVE               # semantic stop
    assert llm.json_calls == 0
    assert g.stats["jev_calls"] > 0 and g.stats["jev_decided"] > 0


def test_jev_never_supersedes_a_dependent():
    # even "Invalid" with certainty only marks the dependent STALE + needs_review
    g, n = _graph(_mumbai_decider(impact=iv(0.0, top=3)))
    g.ingest("I now live in Mumbai", parents=[])
    com = g.store.get(n["com"].id)
    assert com.status == Status.STALE and com.meta.get("needs_review")


def test_unsure_dependent_goes_to_the_llm():
    def llm_fn(system, user):
        return classify_json("PARTIALLY_UPDATES", "My commute (to reconfirm)") \
            if "commute" in user.split("NEW")[0] else classify_json("UNRELATED")
    llm = FakeLLM(json_fn=llm_fn)
    g, n = _graph(_mumbai_decider(impact=iv(0.6)), llm)
    g.ingest("I now live in Mumbai", parents=[])
    assert llm.json_calls >= 1
    com = g.store.get(n["com"].id)
    assert com.content == "My commute (to reconfirm)" and com.status == Status.ACTIVE


def test_known_new_value_goes_to_the_llm_for_the_rewrite():
    llm = FakeLLM(json_fn=lambda s, u: classify_json("UPDATES", "My commute is 20 minutes")
                  if "commute" in u.split("NEW")[0] else classify_json("UNRELATED"))
    g, n = _graph(_mumbai_decider(impact=iv(0.02, p_known=0.9)), llm)
    g.ingest("I now live in Mumbai", parents=[])
    assert g.store.get(n["com"].id).content == "My commute is 20 minutes"


def test_jev_failure_falls_back_to_the_llm():
    from test_engine import mumbai_responder

    def boom(*a):
        raise ConnectionError("jev down")
    C.reset_degraded()
    llm = FakeLLM(json_fn=mumbai_responder)
    g, n = _graph(FakeDecider(conflict_fn=boom, impact_fn=boom), llm)
    g.ingest("I now live in Mumbai", parents=[])
    assert C.DEGRADED["jev"] > 0 and C.degraded_total() == 0
    assert g.store.get(n["loc"].id).content == "I live in Mumbai"      # LLM path's answer
    assert g.store.get(n["com"].id).status == Status.STALE
    assert g.store.get(n["tz"].id).status == Status.ACTIVE


def test_derive_links_sends_only_unsure_candidates_to_the_llm():
    seen = []

    def llm_fn(system, user):
        seen.append(user)
        return {"derived_from": []}
    g = GEM(llm=FakeLLM(json_fn=llm_fn), embedder=FakeEmbedder(),
            decider=FakeDecider(depends_fn=lambda c, new: {"A": 0.95, "B": 0.4}.get(c[0], 0.05)))
    a = g.ingest("A: I live in Bangalore", parents=[], check_conflicts=False)
    g.ingest("B: I like dosa", parents=[], check_conflicts=False)
    g.ingest("C: My favourite colour is blue", parents=[], check_conflicts=False)
    new = g.ingest("My commute is 45 minutes", check_conflicts=False)
    assert [p.id for p in g.store.derived_from_targets(new.id)] == [a.id]
    assert len(seen) == 1 and "B: I like dosa" in seen[0] and "Bangalore" not in seen[0]


def test_default_config_uses_no_decider():
    assert GEMConfig().decider == "llm"
    assert GEM(llm=FakeLLM(), embedder=FakeEmbedder()).decider is None


# --------------------------------------------------------------------------- #
# JevDecider request/response handling (no network)
# --------------------------------------------------------------------------- #

class FakeClient:
    """Answers every question; records requests so batching can be checked."""

    def __init__(self):
        self.requests = []

    def system_one(self, state, questions):
        self.requests.append((state, questions))
        choices, nouls, scores = {}, {}, {}
        for name, q in questions.items():
            if q.type == "choice":
                choices[name] = SimpleNamespace(probabilities={"UPDATES": 0.8, "UNRELATED": 0.2})
            elif q.type == "noul":
                nouls[name] = SimpleNamespace(noul=0.7)
            else:
                scores[name] = SimpleNamespace(probabilities={0: 0.1, 1: 0.1, 2: 0.7, 3: 0.1})
        return SimpleNamespace(choices=choices, nouls=nouls, scores=scores)


def test_jev_decider_batches_and_parses():
    pytest.importorskip("typesafe_sdk")
    client = FakeClient()
    d = JevDecider(client, batch=8)
    out = d.impact("change", [f"dep {i}" for i in range(10)])
    assert len(client.requests) == 2 and d.calls == 2                 # 8 + 2
    assert len(client.requests[1][0]["dependents"]) == 2
    assert out[9].probs[2] == 0.7 and out[9].p_known == 0.7
    c = d.conflicts("new", ["a", "b"])
    assert c[0].probs[Label.UPDATES] == 0.8 and c[0].p_covered == 0.7
    assert d.depends("new", ["a"]) == [0.7]
