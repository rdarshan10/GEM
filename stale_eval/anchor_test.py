"""Does property-stamped anchoring get BOTH deep positives AND restraint?

Two scenarios, each a chain where the change hits the ROOT and we judge the distant LEAF:
  POS (launch 6-hop): root slips Dec1->Mar1  => leaf (production date) MUST be invalidated.
  NEG (relocation):   root Bangalore->Chennai (SAME timezone) => leaf (IST standup) MUST survive.

Three deciders on the leaf:
  eager        — GEM eager cascade (reference; hop-by-hop, handles both).
  anchor_trust — root content moved -> stale. O(1), no confirm. Catches depth, but OVER-invalidates.
  anchor_prop  — stamp the PROPERTY the leaf depends on (1 call at write); on change, check only if
                 THAT property changed. ~O(1) + 1 property-check. Should get depth AND restraint.

Run:  python stale_eval/anchor_test.py
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gem import Memory
from gem.store import Status
from gem.llm import make_llm
from adapter import _contradictor

PROPERTY_EXTRACT_SYS = (
    "A memory fact's validity depends on a SPECIFIC PROPERTY of an upstream root fact (not the whole "
    "root). Name that single property the dependent hinges on — e.g. the root's DATE, its COUNTRY/"
    "time-zone rather than the exact city, the PERSON's role rather than the person. "
    "Respond ONLY JSON: {\"property\": \"...\"}")

PROP_CHECK_SYS = (
    "A dependent memory hinges on a specific PROPERTY of its root fact, and the root has changed. "
    "Decide whether THAT property actually changed. If the property is preserved, the dependent "
    "SURVIVES. Respond ONLY JSON: {\"property_changed\": true/false, \"reason\": \"...\"}")

POS = dict(
    chain=["The product launch is scheduled for December 1",
           "The marketing campaign kicks off November 17, two weeks before launch",
           "The press embargo lifts on November 17 to match the campaign",
           "Analyst briefings are booked for November 16, before the embargo",
           "The demo video must be finalized by November 14 for the briefings",
           "Video production starts October 20 to finish the demo by November 14"],
    trigger="The launch has slipped from December 1 to March 1",
    expect_invalid=True)

NEG = dict(
    chain=["I live in Bangalore",
           "My working hours are set to the India IST timezone",
           "My daily standup is scheduled in IST",
           "I join the team standup at 9am IST"],
    trigger="I moved from Bangalore to Chennai",
    expect_invalid=False)   # Chennai is also IST -> the standup time survives


def _root_of(g, node):
    root, seen = node, set()
    while True:
        parents = g.store.derived_from_targets(root.id)
        if not parents or root.id in seen:
            return root
        seen.add(root.id)
        root = parents[0]


def build(scn, llm, with_property):
    mem = Memory(cascade=False)
    ids = []
    for i, f in enumerate(scn["chain"]):
        ids.append(mem.load(f, derived_from=([ids[i - 1]] if i else [])))
    g = mem._g
    leaf_id = ids[-1]
    stamps = {}
    for n in g.store.all_nodes():
        root = _root_of(g, n)
        if root.id == n.id:
            continue
        prop = None
        if with_property:
            out = llm.chat_json(PROPERTY_EXTRACT_SYS, f"DEPENDENT: {n.content}\nROOT: {root.content}")
            prop = out.get("property") if isinstance(out, dict) else None
        stamps[n.id] = (root.id, root.content, prop)
    mem.add(scn["trigger"])           # eager detection updates/supersedes the root
    return mem, stamps, leaf_id


def _root_new_value(g, root):
    contra = _contradictor(g, root)
    return contra.content if contra else root.content   # new value lives in the contradictor


def verdict_trust(mem, stamps, leaf_id):
    g = mem._g
    root_id, old, _ = stamps[leaf_id]
    root_now = g.store.get(root_id)
    moved = root_now.content != old or root_now.status != Status.ACTIVE or _contradictor(g, root_now)
    return bool(moved)


def verdict_prop(mem, stamps, leaf_id, llm):
    g = mem._g
    root_id, old, prop = stamps[leaf_id]
    root_now = g.store.get(root_id)
    moved = root_now.content != old or root_now.status != Status.ACTIVE or _contradictor(g, root_now)
    if not moved:
        return False
    new = _root_new_value(g, root_now)
    v = llm.chat_json(PROP_CHECK_SYS, f"DEPENDENT hinges on: {prop}\nROOT was: {old}\nROOT now: {new}")
    return bool(isinstance(v, dict) and v.get("property_changed"))


# --- prop2: value-grounded 2-step. (1) extract the CATEGORY-level property + its value at write;
# (2) on change, decide if that property's VALUE differs under the new root. Category-pushing fixes
# prop's misses (Alice->"manager role", vegetarian->"excludes meat", Honda->"fuel type").
PROPERTY_EXTRACT2_SYS = (
    "A dependent memory's validity hinges on ONE property of an upstream root fact — and it is "
    "usually a CATEGORY, not the exact surface value: the COUNTRY or TIME-ZONE rather than the city; "
    "the ROLE rather than the person; the underlying CONSTRAINT (excludes meat) rather than the label "
    "(vegetarian); the fuel TYPE rather than the car model; the platform/OS rather than the model "
    "number. Name that category-level property and its VALUE for the current root. "
    "Respond ONLY JSON: {\"property\": \"<category-level property>\", \"value\": \"<its value now>\"}")

PROP_RECHECK2_SYS = (
    "A dependent memory hinges on a category-level PROPERTY whose value WAS as given. The root has "
    "changed. Given the NEW root, decide whether that property's VALUE is now DIFFERENT. If the value "
    "is preserved (even if surface wording changed — e.g. a different city in the same time-zone, a "
    "stricter diet that still excludes meat), it is NOT changed and the dependent SURVIVES. "
    "Respond ONLY JSON: {\"value_now\": \"...\", \"changed\": true/false}")


def build_prop2(scn, llm):
    mem = Memory(cascade=False)
    ids = []
    for i, f in enumerate(scn["chain"]):
        ids.append(mem.load(f, derived_from=([ids[i - 1]] if i else [])))
    g = mem._g
    leaf_id = ids[-1]
    stamps = {}
    for n in g.store.all_nodes():
        root = _root_of(g, n)
        if root.id == n.id:
            continue
        out = llm.chat_json(PROPERTY_EXTRACT2_SYS, f"DEPENDENT: {n.content}\nROOT: {root.content}")
        prop = out.get("property") if isinstance(out, dict) else None
        val = out.get("value") if isinstance(out, dict) else None
        stamps[n.id] = (root.id, root.content, prop, val)
    mem.add(scn["trigger"])
    return mem, stamps, leaf_id


def verdict_prop2(mem, stamps, leaf_id, llm):
    g = mem._g
    root_id, old, prop, val = stamps[leaf_id]
    root_now = g.store.get(root_id)
    moved = root_now.content != old or root_now.status != Status.ACTIVE or _contradictor(g, root_now)
    if not moved:
        return False
    new = _root_new_value(g, root_now)
    v = llm.chat_json(PROP_RECHECK2_SYS, f"PROPERTY: {prop}\nITS VALUE WAS: {val}\nNEW ROOT: {new}")
    return bool(isinstance(v, dict) and v.get("changed"))


def verdict_eager(scn, llm):
    mem = Memory(cascade=True)
    ids = []
    for i, f in enumerate(scn["chain"]):
        ids.append(mem.load(f, derived_from=([ids[i - 1]] if i else [])))
    mem.add(scn["trigger"])
    leaf = mem._g.store.get(ids[-1])
    return leaf.status != Status.ACTIVE or leaf.meta.get("needs_review") or leaf.content != scn["chain"][-1]


def run():
    llm = make_llm()
    print("ANCHOR PROPERTY-STAMP TEST — deep positive (must invalidate) + hard negative (must survive)\n")
    print(f"{'scenario':10} {'expect':9} {'eager':12} {'anchor_trust':14} {'anchor_prop':12}")
    for name, scn in (("POS", POS), ("NEG", NEG)):
        exp = scn["expect_invalid"]
        eager = verdict_eager(scn, llm)
        mt, st, lt = build(scn, llm, with_property=False)
        trust = verdict_trust(mt, st, lt)
        mp, sp, lp = build(scn, llm, with_property=True)
        prop = verdict_prop(mp, sp, lp, llm)

        def mark(v):
            return ("INVALID" if v else "survive") + (" ✓" if v == exp else " ✗")
        print(f"{name:10} {'invalid' if exp else 'survive':9} {mark(eager):12} {mark(trust):14} {mark(prop):12}")
    print("\nWin = anchor_prop matches expectation on BOTH (depth positive AND hard-negative restraint),")
    print("where anchor_trust over-invalidates the negative.")


if __name__ == "__main__":
    run()
