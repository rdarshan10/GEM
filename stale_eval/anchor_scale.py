"""Scaled test of property-stamped anchoring vs eager vs anchor-trust.

12 deep-dependency scenarios across domains:
  6 POSITIVE — the root change DOES invalidate the distant leaf (must INVALIDATE).
  6 NEGATIVE — the root changes but the leaf's depended-on PROPERTY is preserved (must SURVIVE).

The negatives are the discriminators: anchor_trust (root moved -> stale) should OVER-invalidate
them; anchor_prop (check only if the depended-on property changed) and eager should respect them.

Run:  python stale_eval/anchor_scale.py
"""

from __future__ import annotations

import sys
import time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gem.llm import make_llm
from anchor_test import (build, verdict_prop, verdict_eager,
                         build_prop2, verdict_prop2)

SCENARIOS = [
    # ---- POSITIVE: leaf must be invalidated ----
    dict(name="launch-slip", expect_invalid=True,
         chain=["The product launch is scheduled for December 1",
                "The marketing campaign starts November 17, two weeks before launch",
                "The press embargo lifts on November 17 to match the campaign",
                "Analyst briefings are booked for November 16, before the embargo"],
         trigger="The launch slipped from December 1 to March 1"),
    dict(name="warfarin-doac", expect_invalid=True,
         chain=["The patient is anticoagulated with warfarin",
                "The INR target is 2.0 to 3.0",
                "Weekly INR blood draws are scheduled",
                "A Friday lab appointment is booked for the INR draw"],
         trigger="The patient was switched from warfarin to apixaban"),
    dict(name="office-tz-move", expect_invalid=True,
         chain=["I work from the New York office",
                "My team standup is at 10am Eastern time",
                "I do my prep call at 9am Eastern, before standup"],
         trigger="I transferred to the London office"),
    dict(name="ev-switch", expect_invalid=True,
         chain=["I drive a gasoline car",
                "I budget 200 dollars a month for fuel",
                "I fill up at the Shell on Main Street every week"],
         trigger="I replaced my car with a fully electric one"),
    dict(name="pay-cut", expect_invalid=True,
         chain=["My salary is 120,000 dollars",
                "My monthly budget assumes a 120k salary",
                "I auto-transfer 2,000 dollars to savings each month"],
         trigger="I took a 40 percent pay cut"),
    dict(name="manager-reorg", expect_invalid=True,
         chain=["Alice is my manager",
                "I send Alice a weekly status report",
                "The report is due to Alice every Friday at 5pm"],
         trigger="After the reorg, Bob replaced Alice as my manager"),
    # ---- NEGATIVE: leaf must survive (root changed, property preserved) ----
    dict(name="samecity-tz", expect_invalid=False,
         chain=["I live in Bangalore",
                "My working hours are set to India IST",
                "My daily standup is at 9am IST"],
         trigger="I moved from Bangalore to Chennai"),
    dict(name="samestate-tax", expect_invalid=False,
         chain=["I live in San Francisco",
                "I file California state income taxes",
                "I use California tax form 540"],
         trigger="I moved from San Francisco to Los Angeles"),
    dict(name="samefuel-budget", expect_invalid=False,
         chain=["I drive a Honda Civic",
                "I budget 200 dollars a month for gasoline"],
         trigger="I replaced it with a Toyota Camry"),
    dict(name="sameos-backup", expect_invalid=False,
         chain=["I use an iPhone 13",
                "My photos back up automatically to iCloud",
                "I rely on iCloud to recover my photos"],
         trigger="I upgraded to an iPhone 15"),
    dict(name="samecountry-plug", expect_invalid=False,
         chain=["I live in Berlin, Germany",
                "My devices use the EU two-pin plug standard",
                "I bought a set of EU-plug chargers"],
         trigger="I moved from Berlin to Munich"),
    dict(name="samediet-box", expect_invalid=False,
         chain=["I am vegetarian",
                "I order the meat-free lunch box",
                "The meat-free box is delivered every Tuesday"],
         trigger="I became fully vegan"),
]


def run():
    llm = make_llm()
    agg = {m: {"pos": 0, "neg": 0} for m in ("eager", "prop", "prop2")}
    npos = sum(s["expect_invalid"] for s in SCENARIOS)
    nneg = len(SCENARIOS) - npos
    print(f"SCALED ANCHOR TEST — {npos} positive + {nneg} negative deep scenarios\n")
    print(f"{'scenario':16} {'expect':8} {'eager':9} {'prop':9} {'prop2':9}")
    for s in SCENARIOS:
        exp = s["expect_invalid"]
        for attempt in range(5):                      # retry on transient 429 rate-limits
            try:
                eager = verdict_eager(s, llm)
                mp, sp, lp = build(s, llm, with_property=True)
                prop = verdict_prop(mp, sp, lp, llm)
                m2, s2, l2 = build_prop2(s, llm)
                prop2 = verdict_prop2(m2, s2, l2, llm)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(8 * (attempt + 1))
        bucket = "pos" if exp else "neg"
        for name, v in (("eager", eager), ("prop", prop), ("prop2", prop2)):
            agg[name][bucket] += int(bool(v) == exp)

        def cell(v):
            return ("INVAL" if v else "survv") + ("✓" if bool(v) == exp else "✗")
        print(f"{s['name']:16} {'inval' if exp else 'survv':8} {cell(eager):9} {cell(prop):9} {cell(prop2):9}",
              flush=True)

    print("\n" + "=" * 56)
    print(f"{'method':8} {'POS acc':12} {'NEG acc':12} {'TOTAL':10}")
    for m in ("eager", "prop", "prop2"):
        p, n = agg[m]["pos"], agg[m]["neg"]
        print(f"{m:8} {f'{p}/{npos}':12} {f'{n}/{nneg}':12} {f'{p+n}/{len(SCENARIOS)} ({(p+n)/len(SCENARIOS):.0%})':10}")
    print("=" * 56)
    print("Read: does the value-grounded 2-step 'prop2' close prop's gap to eager — esp. the")
    print("category-level negatives (samediet, samefuel) + the manager-role positive?")


if __name__ == "__main__":
    run()
