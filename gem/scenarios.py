"""Cascade scenario suite — the MVP slice of the propagation eval (full set is Unit 5.5).

Each scenario pins a small DERIVED_FROM graph, fires one trigger observation, and
declares ground truth: which memories SHOULD be invalidated and which SHOULD survive.
The scorer measures propagation correctness against that — rewarding correct pruning of
hard negatives, not just aggressive invalidation.

Setup memories are loaded with check_conflicts=False (no LLM, deterministic structure);
only the trigger drives the cascade. Categories span 2/3/4-hop chains, hard negatives,
divergent parents, unknown-value updates, a cycle, and EXTENDS non-propagation, across
personal / home-intelligence / work domains (per the anti-rigging guidance).

Run:  python -m gem.scenarios            # all
      python -m gem.scenarios --only 2   # one scenario (0-indexed)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from .engine import GEM
from .store import Status


@dataclass
class Scenario:
    name: str
    category: str
    facts: list[str]               # setup memories, in order
    parents: list[list[int]]       # parents[i] = indices of facts[i]'s DERIVED_FROM parents
    trigger: str                   # the new observation that should drive a cascade
    expect_invalid: list[bool]     # per setup fact: should the trigger invalidate it?
    note: str = ""


SCENARIOS: list[Scenario] = [
    Scenario(
        name="rent-follows-city (2-hop)",
        category="personal / positive",
        facts=["I live in Berlin", "My monthly rent is 1200 euros"],
        parents=[[], [0]],
        trigger="I have moved to Munich",
        expect_invalid=[True, True],
        note="rent was derived from the city; moving invalidates it (value now unknown)",
    ),
    Scenario(
        name="mumbai-commute (4-hop)",
        category="personal / positive",
        facts=[
            "I live in Bangalore",
            "My commute to work is 45 minutes",
            "I wake at 7am to beat the traffic",
            "My daily briefing is scheduled for 6:45am",
        ],
        parents=[[], [0], [1], [2]],
        trigger="I now live in Mumbai",
        expect_invalid=[True, True, True, True],
        note="the headline 4-hop chain: city -> commute -> wake -> briefing",
    ),
    Scenario(
        name="timezone-survives-same-zone (hard negative)",
        category="personal / hard-negative",
        facts=["I live in Bangalore", "My timezone is IST"],
        parents=[[], [0]],
        trigger="I now live in Mumbai",
        expect_invalid=[True, False],
        note="Mumbai is also IST -> timezone must survive (correct pruning)",
    ),
    Scenario(
        name="tax-country-survives-city-move (hard negative)",
        category="personal / hard-negative",
        facts=["I live in Lyon, France", "I file my income taxes in France"],
        parents=[[], [0]],
        trigger="I have moved to Paris",
        expect_invalid=[True, False],
        note="still in France -> tax country unaffected",
    ),
    Scenario(
        name="commute-plan-divergent-parents (positive branch)",
        category="personal / divergent-parents",
        facts=[
            "I live in Bangalore",
            "I work 9 to 5",
            "I plan a 7am commute to beat the Bangalore traffic",
        ],
        parents=[[], [], [0, 1]],
        trigger="I now live in Mumbai",
        expect_invalid=[True, False, True],
        note="commute plan depends on the city (traffic); work hours untouched",
    ),
    Scenario(
        name="charger-survives-move (divergent hard negative)",
        category="home / divergent-parents",
        facts=[
            "I live in Seattle",
            "I own a Tesla",
            "My garage charger is a Tesla Wall Connector",
        ],
        parents=[[], [], [0, 1]],
        trigger="I have moved to Portland",
        expect_invalid=[True, False, False],
        note="charger depends on the car, not the city -> survives the move",
    ),
    Scenario(
        name="raise-unknown-amount (unknown-value)",
        category="personal / unknown-value",
        facts=["My salary is 80k dollars", "I budget 2000 dollars a month for rent"],
        parents=[[], [0]],
        trigger="I just got a raise",
        expect_invalid=[True, True],
        note="raise gives no amount -> salary STALE, budget derived -> invalid",
    ),
    Scenario(
        name="thermostat-baseline (home, 2-hop positive)",
        category="home / positive",
        facts=[
            "The living room baseline temperature is 21C",
            "The evening thermostat is set to 23C, which is the baseline plus 2 degrees",
        ],
        parents=[[], [0]],
        trigger="I changed the living room baseline temperature to 18C",
        expect_invalid=[True, True],
        note=("derived CONCRETE value (23C) must go stale when baseline changes. "
              "Insight: a memory phrased as a pure FORMULA ('target is baseline + 2') "
              "legitimately survives a baseline change — the rule still holds — whereas a "
              "concrete computed value does not. The cascade respects that distinction."),
    ),
    Scenario(
        name="api-region-sla (work, 3-hop positive)",
        category="work / positive",
        facts=[
            "Our API is hosted on AWS us-east-1",
            "Our latency budget assumes us-east-1 at about 20ms",
            "Our SLA promises p99 under 50ms based on that latency budget",
        ],
        parents=[[], [0], [1]],
        trigger="We migrated the API to AWS ap-south-1",
        expect_invalid=[True, True, True],
        note="region -> latency budget -> SLA",
    ),
    Scenario(
        name="pet-detail-extends (EXTENDS non-propagation)",
        category="personal / extends",
        facts=["I have a dog named Rex", "Rex is a golden retriever"],
        parents=[[], [0]],
        trigger="Rex loves to swim in the lake",
        expect_invalid=[False, False],
        note="adding a detail must NOT invalidate anything",
    ),
    Scenario(
        name="belief-cycle (cycle guard)",
        category="robustness / cycle",
        facts=[
            "Assumption A: the project ships in Q3",
            "Plan B: hiring is paced to a Q3 ship (depends on A)",
        ],
        parents=[[1], [0]],   # A derived from B and B derived from A -> cycle
        trigger="The project ship date has slipped to Q4",
        expect_invalid=[True, True],
        note="cyclic dependency must terminate (visited-set guard), not hang",
    ),
]


def run_scenario(s: Scenario, *, verbose: bool = False, make_store=None, cfg=None) -> dict:
    g = GEM(store=make_store() if make_store else None, config=cfg)
    # two-phase setup: create ALL nodes first, then wire DERIVED_FROM edges. This handles
    # forward references and cycles (a node depending on one created later).
    nodes = [g.ingest(f, parents=[], check_conflicts=False) for f in s.facts]
    originals = [n.content for n in nodes]
    from .store import EdgeType
    for i, pidx in enumerate(s.parents):
        for j in pidx:
            g.store.add_edge(nodes[i].id, nodes[j].id, EdgeType.DERIVED_FROM)

    g.trace.clear()
    g.ingest(s.trigger, parents=[])   # the trigger drives the cascade

    per_node = []
    for i, node in enumerate(nodes):
        cur = g.store.get(node.id)
        # "affected" = anything changed: status, confidence, OR content (an in-place
        # UPDATE corrects the node and keeps it ACTIVE — that still counts as affected).
        affected = (cur.status != Status.ACTIVE or cur.confidence < 1.0
                    or cur.content != originals[i])
        per_node.append({
            "fact": s.facts[i],
            "expected_invalid": s.expect_invalid[i],
            "got_invalid": affected,
            "correct": affected == s.expect_invalid[i],
            "status": cur.status.value,
            "content": cur.content,
        })
    correct = sum(p["correct"] for p in per_node)
    result = {
        "name": s.name,
        "category": s.category,
        "passed": correct == len(per_node),
        "node_correct": correct,
        "node_total": len(per_node),
        "per_node": per_node,
        "trace": list(g.trace),
    }
    if verbose:
        _print_scenario(s, result)
    return result


def _print_scenario(s: Scenario, r: dict) -> None:
    flag = "PASS" if r["passed"] else "FAIL"
    print(f"\n[{flag}] {s.name}  ({s.category})  {r['node_correct']}/{r['node_total']}")
    print(f"       trigger: {s.trigger}")
    for p in r["per_node"]:
        mark = "ok" if p["correct"] else "XX"
        exp = "invalid" if p["expected_invalid"] else "survive"
        got = "invalid" if p["got_invalid"] else "survive"
        print(f"   [{mark}] expect {exp:7} got {got:7} | {p['status']:10} {p['content']}")
    if not r["passed"]:
        print("       cascade trace:")
        for line in r["trace"]:
            print("         " + line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GEM cascade scenario suite")
    ap.add_argument("--only", type=int, default=None, help="run a single scenario by index")
    ap.add_argument("--falkor", action="store_true",
                    help="run against a FalkorDB backend (localhost:6379) instead of in-memory")
    ap.add_argument("--trace", action="store_true",
                    help="print the full cascade trace for every scenario (incl. passes), to "
                         "confirm negative-case ties are restraint-WITH-edges (traversed to the "
                         "boundary and decided UNRELATED) not restraint-by-empty-graph")
    args = ap.parse_args(argv)

    make_store = None
    if args.falkor:
        from .falkor_store import FalkorStore
        make_store = lambda: FalkorStore(clear_on_start=True)
        print("backend: FalkorDB (localhost:6379)\n")

    from . import classify as C
    C.reset_degraded()
    scenarios = [SCENARIOS[args.only]] if args.only is not None else SCENARIOS
    results = []
    for s in scenarios:
        r = run_scenario(s, verbose=True, make_store=make_store)
        if args.trace:
            print("       cascade trace (traversal + decisions):")
            for line in r["trace"]:
                print("         " + line)
        results.append(r)

    passed = sum(r["passed"] for r in results)
    node_correct = sum(r["node_correct"] for r in results)
    node_total = sum(r["node_total"] for r in results)
    deg = C.degraded_total()
    print("\n" + "=" * 60)
    print(f"scenarios passed:  {passed}/{len(results)}")
    print(f"node-level correct: {node_correct}/{node_total} "
          f"({node_correct / node_total:.0%})")
    print(f"integrity: {deg} degraded calls"
          f"{'  <- DISCARD (rate-limited)' if deg else ' (clean)'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
