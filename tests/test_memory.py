"""Public facade (gem.Memory) — tested deterministically with the mock LLM + embedder.
Covers the 6-line user experience: add pins dependencies, a conflicting add cascades and
reports what went stale, search excludes stale, the flat mode doesn't propagate."""

from gem import Memory, Fact, AddResult
from gem.engine import GEM
from conftest import FakeLLM, FakeEmbedder, classify_json


def _existing(user: str) -> str:
    for line in user.splitlines():
        if line.startswith("EXISTING memory:"):
            return line.split(":", 1)[1].strip()
    return ""


def _is_change_desc(user: str) -> bool:
    return any(k in user for k in ("has changed", "no longer reliable",
                                   "partially invalidated", "superseded"))


def _jwt_responder(system, user):
    """The migration DIRECTLY conflicts only with the auth fact; the derived test/CI facts
    are not direct contradictions — they go stale only via the cascade (change-desc path)."""
    existing = _existing(user).lower()
    if not _is_change_desc(user):
        return classify_json("CONTRADICTS") if existing.startswith("auth uses") \
            else classify_json("UNRELATED")    # tests/CI: no DIRECT conflict
    return classify_json("CONTRADICTS")        # cascade: dependents of the changed auth fact


def _mem(**kw) -> Memory:
    return Memory(llm=FakeLLM(json_fn=_jwt_responder), embedder=FakeEmbedder(), **kw)


def test_add_returns_id_and_cascade_invalidates_dependents():
    m = _mem()
    auth = m.load("Auth uses JWT bearer tokens")
    m.load("Tests mock the JWT verifier", derived_from=[auth])
    m.load("CI requires a JWT_SECRET variable", derived_from=[auth])

    r = m.add("We migrated auth from JWT to session cookies")
    assert isinstance(r, AddResult) and r.id
    contents = {f.content for f in r.invalidated}
    assert "Tests mock the JWT verifier" in contents
    assert "CI requires a JWT_SECRET variable" in contents
    assert bool(r) is True                      # truthy: a cascade happened


def test_search_excludes_stale_by_default():
    m = _mem()
    auth = m.load("Auth uses JWT bearer tokens")
    m.load("Tests mock the JWT verifier", derived_from=[auth])
    m.add("We migrated auth from JWT to session cookies")

    active = {f.content for f in m.search("how do tests authenticate", k=10)}
    assert "Tests mock the JWT verifier" not in active           # stale -> hidden
    withstale = {f.content for f in m.search("how do tests authenticate",
                                             k=10, include_stale=True)}
    assert "Tests mock the JWT verifier" in withstale            # opt-in to see it


def test_flat_mode_does_not_propagate():
    m = _mem(cascade=False)
    auth = m.load("Auth uses JWT bearer tokens")
    m.load("Tests mock the JWT verifier", derived_from=[auth])
    r = m.add("We migrated auth from JWT to session cookies")
    # flat resolves the direct conflict on the auth fact but never reaches the derived test fact
    assert "Tests mock the JWT verifier" not in {f.content for f in r.invalidated}


def test_stale_and_why():
    m = _mem()
    auth = m.load("Auth uses JWT bearer tokens")
    dep = m.load("Tests mock the JWT verifier", derived_from=[auth])
    m.add("We migrated auth from JWT to session cookies")
    assert any(f.content == "Tests mock the JWT verifier" for f in m.stale)
    assert "Auth uses JWT bearer tokens" in m.why(dep)           # provenance explains staleness
