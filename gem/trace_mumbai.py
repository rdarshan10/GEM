"""The Mumbai 4-hop cascade trace — the demo the whole MVP exists to show.

One upstream fact changes, four hops invalidate down a DERIVED_FROM chain, and one
branch is correctly pruned:

    (lives in Bangalore)  --< (timezone IST)          # UNRELATED -> survives (Mumbai is IST too)
            ^
            |  DERIVED_FROM
    (commute 45 min) -> (wake 7am to beat traffic) -> (daily briefing 6:45am)

    ingest "I now live in Mumbai":
      lives in Bangalore     UPDATES        -> "lives in Mumbai"
        commute 45 min       UPDATES(unknown) -> STALE
          wake 7am           PARTIALLY_UPDATES -> rewritten, lower confidence
            briefing 6:45am  UPDATES(unknown) -> STALE
        timezone IST         UNRELATED      -> semantic stop, untouched

Run:  python -m gem.trace_mumbai
Needs Ollama reachable (OLLAMA_HOST) with the default model (gpt-oss:120b-cloud).
"""

from __future__ import annotations

from .engine import GEM
from .store import Status


def build_graph(make_store=None) -> GEM:
    g = GEM(store=make_store() if make_store else None)
    # DERIVED_FROM links pinned explicitly so the graph STRUCTURE is deterministic;
    # the cascade's invalidation decisions still come from the LLM at each hop.
    loc = g.ingest("I live in Bangalore", parents=[])
    commute = g.ingest("My commute to work is 45 minutes", parents=[loc.id])
    wake = g.ingest("I wake at 7am to beat the traffic", parents=[commute.id])
    g.ingest("My daily briefing is scheduled for 6:45am", parents=[wake.id])
    g.ingest("My timezone is IST", parents=[loc.id])   # hard-negative branch
    return g


def main():
    import sys
    make_store = None
    if "--falkor" in sys.argv:
        from .falkor_store import FalkorStore
        make_store = lambda: FalkorStore(clear_on_start=True)
        print("backend: FalkorDB (localhost:6379)\n")
    g = build_graph(make_store)

    print("=" * 70)
    print("BEFORE — graph state")
    print("=" * 70)
    print(g.store.describe())

    print("\n" + "=" * 70)
    print('TRIGGER — ingest "I now live in Mumbai"')
    print("=" * 70)
    g.trace.clear()
    g.ingest("I now live in Mumbai", parents=[])

    print("\ncascade trace:")
    for line in g.trace:
        print("  " + line)

    print("\n" + "=" * 70)
    print("AFTER — graph state")
    print("=" * 70)
    print(g.store.describe())

    # quick automated check of the expected outcome
    print("\n" + "=" * 70)
    print("EXPECTED-OUTCOME CHECK")
    print("=" * 70)
    checks = []
    tz = next((n for n in g.store.all_nodes() if "timezone" in n.content.lower()), None)
    if tz:
        checks.append(("timezone IST survives (ACTIVE)", tz.status == Status.ACTIVE))
    stale_or_changed = [
        n for n in g.store.all_nodes()
        if ("commute" in n.content.lower() or "briefing" in n.content.lower())
    ]
    for n in stale_or_changed:
        checks.append((f"'{n.content[:40]}...' invalidated",
                       n.status != Status.ACTIVE or n.confidence < 1.0))
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")


if __name__ == "__main__":
    main()
