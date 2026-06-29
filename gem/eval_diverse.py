"""Reality check — deep, radically-diverse scenarios across domains the relocation-heavy eval
never touched, with complex structures (multi-parent DAGs, deep chains, subtle partials, and
cross-domain hard negatives). Many require real DOMAIN KNOWLEDGE and subtle reasoning.

WHY THIS EXISTS: a narrow, location-only eval reported 100% and could not have surfaced a
real ENGINE bug. This eval did. The 6-hop "project launch slip" chain originally failed
(5/6): when one trigger directly conflicts with SEVERAL nodes in the same DERIVED_FROM chain,
the ingest fired an independent cascade per conflict, and a descendant-conflict re-revised a
subchain the root's cascade had already corrected — leaving an intermediate node (the press
embargo) wrongly ACTIVE. That is a mechanism defect, not a model/domain gap, and only a deep
chain where every node is relative to the same root exposes it. The fix (engine.py ingest
step 5: share one visited frontier across conflict-actions, process ancestors first) is
guarded by tests/test_engine.py::test_multi_direct_conflict_on_chain_revises_each_node_once.

Result after the fix (gpt-oss:120b-cloud): 12/12 scenarios, 37/37 nodes (100%) — medical,
legal, finance, multi-parent DAG, belief revision, deep chains, and cross-domain hard
negatives. The takeaway is NOT "it's always perfect"; it's that diverse/deep structure is
what finds the real failures, and this one is now closed.

Run:  python -m gem.eval_diverse
"""

from __future__ import annotations

from .scenarios import Scenario, run_scenario

DIVERSE = [
    Scenario(  # MEDICAL — needs to know DOACs don't need INR monitoring
        name="medical: warfarin -> DOAC",
        category="medical",
        facts=["The patient is anticoagulated with warfarin",
               "The patient's INR target is 2.0 to 3.0",
               "The patient has weekly INR blood draws scheduled"],
        parents=[[], [0], [1]],
        trigger="The patient was switched from warfarin to apixaban",
        expect_invalid=[True, True, True],
        note="apixaban is a DOAC — INR target and INR draws become irrelevant",
    ),
    Scenario(  # LEGAL — chain
        name="legal: governing law change",
        category="legal",
        facts=["The contract is governed by New York law",
               "Contract disputes are resolved in New York courts",
               "We retain New York litigation counsel for this contract"],
        parents=[[], [0], [1]],
        trigger="The contract was amended to be governed by Delaware law",
        expect_invalid=[True, True, True],
    ),
    Scenario(  # LEGAL — cross-domain hard negative
        name="legal: sales office is unrelated",
        category="legal / hard-negative",
        facts=["The contract is governed by New York law",
               "We retain New York litigation counsel for this contract"],
        parents=[[], [0]],
        trigger="We opened a satellite sales office in Texas",
        expect_invalid=[False, False],
    ),
    Scenario(  # INFRA — chain
        name="infra: datastore migration",
        category="software",
        facts=["Our primary datastore is PostgreSQL",
               "We back up the database nightly with pg_dump",
               "The disaster-recovery runbook restores from the latest pg_dump"],
        parents=[[], [0], [1]],
        trigger="We migrated the primary datastore to DynamoDB",
        expect_invalid=[True, True, True],
    ),
    Scenario(  # FINANCE — chain
        name="finance: sold the position",
        category="finance",
        facts=["I hold 100 shares of AAPL",
               "My technology allocation is 40 percent",
               "I am overweight technology relative to my target allocation"],
        parents=[[], [0], [1]],
        trigger="I sold all of my AAPL shares",
        expect_invalid=[True, True, True],
    ),
    Scenario(  # ORG — star fan-out
        name="org: manager reorg",
        category="org",
        facts=["Alice is my manager",
               "I send Alice a weekly status report",
               "Alice approves my time-off requests",
               "Alice writes my annual performance review"],
        parents=[[], [0], [0], [0]],
        trigger="After the reorg, Bob is now my manager",
        expect_invalid=[True, True, True, True],
    ),
    Scenario(  # DIET — subtle, needs to see fish is now allowed
        name="diet: vegetarian -> pescatarian",
        category="lifestyle",
        facts=["I am vegetarian",
               "I instructed the office caterer to never serve me meat or fish"],
        parents=[[], [0]],
        trigger="I now eat fish, so I am pescatarian",
        expect_invalid=[True, True],
        note="the 'no fish' instruction is now wrong",
    ),
    Scenario(  # CLOUD — multi-parent DAG (cost depends on region+type, NOT the AMI)
        name="cloud: instance-type change (DAG)",
        category="cloud / divergent-parents",
        facts=["My EC2 instances run in us-east-1",
               "My EC2 instances use the m5.large instance type",
               "My EC2 instances boot from AMI ami-0abc123",
               "My monthly cost estimate of 400 dollars assumes us-east-1 m5.large"],
        parents=[[], [], [], [0, 1]],
        trigger="I changed my instances to the m5.xlarge instance type",
        expect_invalid=[False, True, False, True],
        note="cost depends on region+type; type changed -> cost stale; region/AMI survive",
    ),
    Scenario(  # PROJECT — deep 6-hop chain
        name="project: launch slip (6-hop)",
        category="project / deep-chain",
        facts=["The product launch date is October 1",
               "The marketing campaign starts September 15, two weeks before launch",
               "The press embargo lifts on September 15 to match the campaign",
               "Analyst briefings are scheduled for September 14, the day before embargo lift",
               "The demo video must be finalized by September 10 for the briefings",
               "Video production kicks off August 20 to finish by September 10"],
        parents=[[], [0], [1], [2], [3], [4]],
        trigger="The launch date slipped to December 1",
        expect_invalid=[True, True, True, True, True, True],
    ),
    Scenario(  # BELIEF revision — chain
        name="belief: benchmark overturns choice",
        category="reasoning",
        facts=["We believe approach A is the fastest option",
               "We chose approach A for the project",
               "The Q3 timeline assumes approach A's performance"],
        parents=[[], [0], [1]],
        trigger="New benchmarks show approach B is three times faster than approach A",
        expect_invalid=[True, True, True],
    ),
    Scenario(  # SECURITY — cross-domain hard negative
        name="security: unrelated email switch",
        category="security / hard-negative",
        facts=["My password manager is 1Password",
               "I store my work 2FA codes in 1Password"],
        parents=[[], [0]],
        trigger="I switched my personal email provider to Fastmail",
        expect_invalid=[False, False],
    ),
    Scenario(  # API — conditional/opportunity
        name="api: tier upgrade frees a throttle",
        category="software",
        facts=["Our API plan is the free tier, limited to 100 requests per minute",
               "Our batch sync is throttled to 100 requests per minute to fit the free tier"],
        parents=[[], [0]],
        trigger="We upgraded to the enterprise tier with a 10000 requests-per-minute limit",
        expect_invalid=[True, True],
        note="the throttle is now based on a stale limit",
    ),
]


def main() -> int:
    print(f"DIVERSE reality check — {len(DIVERSE)} deep cross-domain scenarios\n")
    results = [run_scenario(s, verbose=True) for s in DIVERSE]
    passed = sum(r["passed"] for r in results)
    nc = sum(r["node_correct"] for r in results)
    nt = sum(r["node_total"] for r in results)
    print("\n" + "=" * 64)
    print(f"scenarios passed: {passed}/{len(results)}   node accuracy {nc}/{nt} ({nc/nt:.0%})")
    print("=" * 64)
    fails = [r["name"] for r in results if not r["passed"]]
    if fails:
        print("FAILED (where reality breaks the cascade — honest output):")
        for f in fails:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
