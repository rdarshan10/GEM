"""Quickstart — the 90-second "why would a coding agent want this?" demo.

A dev-assistant agent keeps project memory. It learns a few facts about how auth works,
and crucially some facts DERIVED from that (how tests fake a login, what CI needs). Then the
codebase migrates JWT -> session cookies. We show two memories side by side:

  flat  — resolves the direct conflict on the auth fact, but the derived test/CI facts stay
          ACTIVE and the agent will happily write test setup mocking a verifier that no longer
          exists. This is the silent-staleness bug.
  GEM   — the same change cascades down the DERIVED_FROM edges: the test and CI facts go STALE,
          so a later "how do tests authenticate?" retrieval excludes them and the agent knows
          to reconfirm.

Run:  python -m gem.quickstart        (needs an LLM; defaults to gpt-oss:120b via OLLAMA_HOST)
"""

from __future__ import annotations

from . import Memory


# NOTE on demo design (honest): the dependents must NOT share surface tokens with the change,
# or a flat memory catches them by direct similarity and the cascade buys nothing (that is a
# real effect — see PRODUCTIONIZATION/gap_experiment). Here the change reassigns OWNERSHIP; the
# dependents are config/process premised on the old owner and don't textually conflict with the
# new owner, so flat's direct-conflict check passes them as "unrelated" and serves them stale.
PROJECT_FACTS = [
    # (fact, depends-on-index)   index -1 = root (no parent)
    ("The billing service is owned by the Payments team", -1),
    ("Production billing alerts route to the Payments team's on-call rotation", 0),
    ("The deploy approval for billing requires a Payments team reviewer", 0),
    ("The repository follows trunk-based development", -1),   # unrelated control
]

CHANGE = "Ownership of the billing service was transferred to the new Revenue team"
QUERY = "Who approves a billing deploy and who gets paged, and is that still current?"


def _seed(m: Memory) -> None:
    ids = []
    for fact, dep in PROJECT_FACTS:
        parent = [ids[dep]] if dep >= 0 else []
        ids.append(m.load(fact, derived_from=parent))


def main() -> None:
    flat, gem = Memory(cascade=False), Memory(cascade=True)
    _seed(flat); _seed(gem)

    print("=" * 78)
    print("CODEBASE CHANGE:", CHANGE)
    print("=" * 78)

    fr = flat.add(CHANGE)
    gr = gem.add(CHANGE)

    def _summarize(label, r):
        touched = r.revised + r.invalidated
        print(f"\n{label} touched {len(touched)} derived fact(s) "
              f"({len(r.revised)} corrected, {len(r.invalidated)} flagged stale):")
        for f in r.revised:
            print(f"     [corrected] {f.content}")
        for f in r.invalidated:
            print(f"     [STALE]     {f.content}")
        if not touched:
            print("     (none — derived facts left as-is)")

    _summarize("flat memory", fr)
    _summarize("GEM  memory", gr)

    print("\n" + "-" * 78)
    print(f"AGENT RETRIEVES for: {QUERY!r}")
    print("-" * 78)
    print("flat memory returns (agent will act on this):")
    for f in flat.search(QUERY, k=3):
        print(f"     [{f.status}] {f.content}")
    print("GEM memory returns (stale test setup correctly withheld):")
    for f in gem.search(QUERY, k=3):
        print(f"     [{f.status}] {f.content}")

    print("\n" + "=" * 78)
    stale_marker = "Payments"          # any surviving 'Payments' process fact is now wrong
    flat_trap = any(stale_marker in f.content for f in flat.search(QUERY, k=4))
    gem_safe = not any(stale_marker in f.content for f in gem.search(QUERY, k=4))
    print(f"flat agent still serves stale 'Payments team' process facts : {flat_trap}")
    print(f"GEM agent correctly withholds them                          : {gem_safe}")
    if flat_trap and gem_safe:
        print("That gap is the silent action-on-stale-state bug GEM closes for derived facts.")
    else:
        print("NOTE: flat and GEM agreed here — for this change the dependents were catchable")
        print("by direct similarity, so the cascade added nothing (an honest non-gap case).")


if __name__ == "__main__":
    main()
