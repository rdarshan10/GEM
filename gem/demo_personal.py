"""Personal-intelligence demo — a life change rippling through your assistant's memory.

A personal AI has stored facts about your daily life. Some are DERIVED from one life-anchor
(you commute to an office): your alarm, your transit pass, your morning coffee stop. Then the
anchor changes — you go fully remote. GEM cascades the change so those derived facts go stale,
while the genuinely-unrelated facts (you're vegetarian, a dentist appt) correctly survive.

HONEST NOTE this demo prints at the end: the separation here is MARGINAL. Personal-life facts
cluster by domain — "train", "station", "commute" all sit semantically near "stopped commuting"
— so a flat memory's direct-conflict check catches MOST of these dependents on its own. GEM's
edge narrows to the dependents premised on the anchor but textually DISTANT from the change
(e.g. a transit pass, which "remote job" does not directly contradict). So personal intelligence
is GEM's most resonant *story* but a smaller measured *lift* than structurally-separated domains
(coding ownership transfer, org reorg, belief revision). The demo computes the real gap live.

Run:  python -m gem.demo_personal     (needs an LLM; defaults to gpt-oss:120b via OLLAMA_HOST)
"""

from __future__ import annotations

import sys

from . import Memory

# Windows consoles default to cp1252; models can emit Unicode (e.g. narrow no-break space).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LIFE = [
    # (fact, depends-on-index)   index -1 = a life-anchor (no parent)
    ("I commute to the downtown office by train every weekday", -1),
    ("My alarm is set for 6:30am to catch the 7:15 train", 0),
    ("I keep a monthly transit pass", 0),
    ("I grab a coffee at the station kiosk at 7:00am on my way", 1),
    ("I'm vegetarian", -1),                       # unrelated life-fact (must survive)
    ("My dentist appointment is on the 14th", -1),  # unrelated life-fact (must survive)
]

CHANGE = "I switched to a fully remote job and stopped going into the office"
QUERY = "What does my weekday morning look like and when should I wake up?"


def _seed(m: Memory) -> None:
    ids = []
    for fact, dep in LIFE:
        ids.append(m.load(fact, derived_from=([ids[dep]] if dep >= 0 else [])))


def _summarize(label: str, r) -> None:
    touched = r.revised + r.invalidated
    print(f"\n{label} touched {len(touched)} derived fact(s) "
          f"({len(r.revised)} corrected, {len(r.invalidated)} flagged stale):")
    for f in r.revised:
        print(f"     [corrected] {f.content}")
    for f in r.invalidated:
        print(f"     [STALE]     {f.content}")
    if not touched:
        print("     (none — derived facts left as-is)")


def main() -> None:
    flat, gem = Memory(cascade=False), Memory(cascade=True)
    _seed(flat); _seed(gem)

    print("=" * 78)
    print("LIFE CHANGE:", CHANGE)
    print("=" * 78)
    _summarize("flat memory", flat.add(CHANGE))
    _summarize("GEM  memory", gem.add(CHANGE))

    print("\n" + "-" * 78)
    print(f"YOUR ASSISTANT IS ASKED: {QUERY!r}")
    print("-" * 78)
    print("flat-memory assistant sees (and will plan from):")
    for f in flat.search(QUERY, k=4):
        print(f"     [{f.status}] {f.content}")
    print("GEM-memory assistant sees (stale commute routine withheld):")
    for f in gem.search(QUERY, k=4):
        print(f"     [{f.status}] {f.content}")

    print("\n" + "=" * 78)
    # Compute the REAL gap dynamically (no hardcoded claim): facts this life change made
    # stale that the flat assistant STILL treats as current. These are the silent-staleness
    # bugs the cascade closes — whichever ones they happen to be this run.
    flat_active = {f.content for f in flat.facts(include_stale=False)}
    gem_stale = {f.content for f in gem.stale}
    missed_by_flat = sorted(c for c in gem_stale if c in flat_active)
    veg_kept = any("vegetarian" in f.content for f in gem.facts(include_stale=False))

    if missed_by_flat:
        print("Facts your life change made stale that the FLAT assistant still treats as current")
        print("(GEM flagged them; flat would plan from them):")
        for c in missed_by_flat:
            print(f"     - {c}")
        print(f"\nGEM also kept the genuinely-unrelated facts (e.g. vegetarian): {veg_kept}")
        print("That is the personal-AI win — but note it's MARGINAL here: flat caught the")
        print("commute-adjacent facts by direct similarity; the cascade's edge is the few")
        print("dependents (like the transit pass) that aren't textually close to the change.")
    else:
        print("Flat and GEM agreed on every fact this run — the dependents were all catchable")
        print("by direct similarity, so the cascade added nothing (an honest non-gap case).")
        print("Personal-life facts often cluster near their anchor, so this happens here more")
        print("than in structurally-separated domains.")


if __name__ == "__main__":
    main()
