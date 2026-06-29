"""Unit 5.5 — the propagation eval (the headline's proof).

No public benchmark tests dependency-aware invalidation, so this is a purpose-built,
TRANSPARENTLY GENERATED eval. The whole defense of a self-built eval is that a skeptical
reader can see it wasn't hand-tuned to flatter the system — so scenarios come from
documented TEMPLATES across multiple domains, ground truth is derived from each template's
structure (not eyeballed per-instance), and the same set runs against both GEM and a flat
baseline.

Generation method (fully here, by design):
  Each template is a dependency-chain family with slots (cities, values, ...). Filling the
  slots yields a concrete scenario whose ground truth (which memories MUST invalidate vs
  survive) is fixed by the template's structure, independent of the values chosen. We sweep
  the slot values to produce many instances per template. Categories mirror the hand-built
  slice: N-hop positive chains, hard negatives (a derived fact that must SURVIVE the
  trigger), divergent parents (survive when the changed parent doesn't matter), unknown-value
  updates, and EXTENDS non-propagation.

Baseline: GEM with cascade_enabled=False — it resolves the DIRECT conflict (keep-latest) but
never walks DERIVED_FROM. That isolates exactly what the cascade adds: a flat memory leaves
every downstream dependent stale-but-unmarked.

Scoring: per-node propagation correctness (did each memory end up invalidated/surviving as
ground truth requires), aggregated to node accuracy + scenario pass rate, reported for GEM
vs baseline side by side.

Run:  python -m gem.eval --limit 30          # sample
      python -m gem.eval                      # full generated set
      python -m gem.eval --falkor             # on the FalkorDB backend
"""

from __future__ import annotations

import argparse

from .engine import GEMConfig
from .scenarios import Scenario, run_scenario
from . import classify as C


# --------------------------------------------------------------------------- #
# Template generators — each returns concrete Scenarios with structural ground truth
# --------------------------------------------------------------------------- #

CITIES_SAME_ZONE = [   # (from_city, to_city) pairs in the SAME timezone/country
    ("Bangalore", "Mumbai"), ("Chennai", "Delhi"), ("Pune", "Hyderabad"),
    ("Lyon", "Paris"), ("Munich", "Berlin"), ("Osaka", "Tokyo"),
    ("Kolkata", "Jaipur"), ("Nice", "Bordeaux"), ("Hamburg", "Cologne"),
    ("Nagoya", "Kyoto"), ("Ahmedabad", "Surat"), ("Marseille", "Toulouse"),
    ("Frankfurt", "Stuttgart"), ("Fukuoka", "Sapporo"),
]
RELOCATE = [   # (from, to) generic relocations
    ("Berlin", "Munich"), ("Seattle", "Portland"), ("Austin", "Denver"),
    ("Boston", "Chicago"), ("Madrid", "Valencia"), ("Toronto", "Calgary"),
    ("London", "Bristol"), ("Sydney", "Melbourne"), ("Dublin", "Cork"),
    ("Lisbon", "Porto"), ("Warsaw", "Krakow"), ("Oslo", "Bergen"),
    ("Denver", "Phoenix"), ("Miami", "Atlanta"), ("Nashville", "Memphis"),
    ("Vancouver", "Ottawa"),
]


def t_relocation_chain() -> list[Scenario]:
    """N-hop positive: city -> commute -> wake -> briefing. All must invalidate."""
    out = []
    for frm, to in RELOCATE:
        out.append(Scenario(
            name=f"relocation-4hop[{frm}->{to}]",
            category="generated / nhop-positive",
            facts=[
                f"I live in {frm}",
                f"My commute to work is 45 minutes in {frm}",
                f"I wake at 7am to beat the {frm} traffic",
                "My daily briefing is scheduled for 6:45am",
            ],
            parents=[[], [0], [1], [2]],
            trigger=f"I now live in {to}",
            expect_invalid=[True, True, True, True],
        ))
    return out


def t_timezone_hard_negative() -> list[Scenario]:
    """Hard negative: timezone derived from city must SURVIVE a same-zone move."""
    out = []
    for frm, to in CITIES_SAME_ZONE:
        out.append(Scenario(
            name=f"timezone-survives[{frm}->{to}]",
            category="generated / hard-negative",
            facts=[f"I live in {frm}", "My timezone is unchanged by city within the zone"],
            parents=[[], [0]],
            trigger=f"I now live in {to}",
            expect_invalid=[True, False],
            note="same-zone move must not invalidate timezone",
        ))
    return out


def t_divergent_parents() -> list[Scenario]:
    """Derived from [home, device]; a home move must NOT invalidate a device-bound fact."""
    devices = [
        ("a Tesla", "My garage charger is a Tesla Wall Connector"),
        ("a PlayStation 5", "My TV is set up for PlayStation 5 gaming"),
        ("an espresso machine", "My kitchen counter holds an espresso machine"),
        ("a road bike", "My road bike is tuned for racing"),
        ("a film camera", "My film camera takes 35mm rolls"),
        ("a Stratocaster", "My Stratocaster is strung with light-gauge strings"),
        ("a smart fridge", "My smart fridge tracks groceries automatically"),
        ("a treadmill", "My treadmill is set to a 5k training program"),
        ("a vinyl turntable", "My turntable spins at 33 and 45 rpm"),
        ("a drone", "My drone is registered for aerial photography"),
        ("a sewing machine", "My sewing machine is threaded for denim work"),
        ("a telescope", "My telescope is calibrated for planetary viewing"),
    ]
    out = []
    for (frm, to), (dev, derived) in zip(RELOCATE, devices):
        out.append(Scenario(
            name=f"divergent-survives[{frm}->{to}|{dev}]",
            category="generated / divergent-parents",
            facts=[f"I live in {frm}", f"I own {dev}", derived],
            parents=[[], [], [0, 1]],
            trigger=f"I have moved to {to}",
            expect_invalid=[True, False, False],
            note="derived fact depends on the device, not the city -> survives the move",
        ))
    return out


def t_sensor_baseline() -> list[Scenario]:
    """Home sensor: a CONCRETE derived value must go stale when its baseline changes."""
    out = []
    for base, new in [(21, 18), (20, 24), (19, 22), (23, 17), (22, 19), (18, 25),
                      (24, 20), (17, 21), (25, 18), (20, 23), (19, 26), (21, 16)]:
        target = base + 2
        out.append(Scenario(
            name=f"sensor-baseline[{base}->{new}C]",
            category="generated / nhop-positive",
            facts=[
                f"The living room baseline temperature is {base}C",
                f"The evening thermostat is set to {target}C, the baseline plus 2 degrees",
            ],
            parents=[[], [0]],
            trigger=f"I changed the living room baseline temperature to {new}C",
            expect_invalid=[True, True],
        ))
    return out


def t_region_sla() -> list[Scenario]:
    """Work 3-hop: region -> latency budget -> SLA. All invalidate on migration."""
    out = []
    regions = [("us-east-1", "ap-south-1"), ("eu-west-1", "us-west-2"),
               ("ap-northeast-1", "sa-east-1"), ("us-west-1", "eu-central-1"),
               ("ca-central-1", "ap-southeast-2"), ("eu-north-1", "us-east-2"),
               ("ap-south-1", "eu-west-3"), ("sa-east-1", "ap-northeast-2"),
               ("us-east-2", "af-south-1"), ("eu-central-1", "me-south-1")]
    for frm, to in regions:
        out.append(Scenario(
            name=f"region-sla[{frm}->{to}]",
            category="generated / nhop-positive",
            facts=[
                f"Our API is hosted on AWS {frm}",
                f"Our latency budget assumes {frm} at about 20ms",
                "Our SLA promises p99 under 50ms based on that latency budget",
            ],
            parents=[[], [0], [1]],
            trigger=f"We migrated the API to AWS {to}",
            expect_invalid=[True, True, True],
        ))
    return out


def t_unknown_value() -> list[Scenario]:
    """Unknown-value update: raise with no amount -> salary + derived budget go stale."""
    out = []
    for sal, rent in [("80k", "2000"), ("120k", "3200"), ("95k", "2500"),
                      ("70k", "1800"), ("140k", "3800"), ("88k", "2200"),
                      ("110k", "2900"), ("60k", "1500"), ("130k", "3500"),
                      ("75k", "1900")]:
        out.append(Scenario(
            name=f"raise-unknown[{sal}]",
            category="generated / unknown-value",
            facts=[f"My salary is {sal} dollars",
                   f"I budget {rent} dollars a month for rent based on my salary"],
            parents=[[], [0]],
            trigger="I just got a raise",
            expect_invalid=[True, True],
        ))
    return out


def t_extends_negative() -> list[Scenario]:
    """EXTENDS: adding a detail must invalidate nothing."""
    out = []
    pets = [("Rex", "a golden retriever", "Rex loves to swim in the lake"),
            ("Milo", "a tabby cat", "Milo enjoys sleeping on the windowsill"),
            ("Bruno", "a beagle", "Bruno is great with kids"),
            ("Luna", "a husky", "Luna howls at sirens"),
            ("Coco", "a parrot", "Coco can mimic the doorbell"),
            ("Max", "a labrador", "Max fetches the morning paper"),
            ("Bella", "a corgi", "Bella loves car rides"),
            ("Shadow", "a black cat", "Shadow hides during thunderstorms"),
            ("Ziggy", "a terrier", "Ziggy digs in the backyard"),
            ("Nala", "a maine coon", "Nala drinks from the tap")]
    for name, breed, detail in pets:
        out.append(Scenario(
            name=f"extends[{name}]",
            category="generated / extends-negative",
            facts=[f"I have a pet named {name}", f"{name} is {breed}"],
            parents=[[], [0]],
            trigger=detail,
            expect_invalid=[False, False],
        ))
    return out


def t_org_reporting() -> list[Scenario]:
    """Work star fan-out: two facts derived from 'who I report to' both invalidate."""
    mgrs = [("Alice", "Bob"), ("Priya", "Sam"), ("Chen", "Dana"),
            ("Omar", "Lena"), ("Yuki", "Marco"), ("Rosa", "Ivan")]
    out = []
    for frm, to in mgrs:
        out.append(Scenario(
            name=f"org-reporting[{frm}->{to}]",
            category="generated / nhop-positive",
            facts=[f"I report to {frm}",
                   f"My weekly 1:1 is on {frm}'s calendar",
                   f"My OKRs are reviewed by {frm}"],
            parents=[[], [0], [0]],
            trigger=f"I now report to {to}",
            expect_invalid=[True, True, True],
        ))
    return out


def t_language_hard_negative() -> list[Scenario]:
    """National language derived from country must SURVIVE a within-country move."""
    rows = [("Munich", "Hamburg", "German"), ("Lyon", "Nantes", "French"),
            ("Osaka", "Sendai", "Japanese"), ("Valencia", "Bilbao", "Spanish"),
            ("Turin", "Bologna", "Italian"), ("Porto", "Braga", "Portuguese")]
    out = []
    for frm, to, lang in rows:
        out.append(Scenario(
            name=f"language-survives[{frm}->{to}]",
            category="generated / hard-negative",
            facts=[f"I live in {frm}", f"I speak {lang} day to day"],
            parents=[[], [0]],
            trigger=f"I moved to {to}, still in the same country",
            expect_invalid=[True, False],
        ))
    return out


def t_subscription() -> list[Scenario]:
    """Billing 2-hop: concrete monthly bill derived from plan tier goes stale on change."""
    rows = [("Netflix", "Premium", "Standard", "19.99"),
            ("Spotify", "Family", "Individual", "16.99"),
            ("Notion", "Business", "Plus", "15.00"),
            ("Adobe", "All Apps", "Photography", "59.99"),
            ("AWS", "Enterprise Support", "Business Support", "15000"),
            ("GitHub", "Enterprise", "Team", "21.00"),
            ("Dropbox", "Advanced", "Plus", "20.00"),
            ("Figma", "Organization", "Professional", "12.00")]
    out = []
    for svc, hi, lo, price in rows:
        out.append(Scenario(
            name=f"subscription[{svc}:{hi}->{lo}]",
            category="generated / nhop-positive",
            facts=[f"My {svc} plan is the {hi} tier",
                   f"My monthly {svc} bill is {price} dollars on the {hi} tier"],
            parents=[[], [0]],
            trigger=f"I downgraded my {svc} plan to the {lo} tier",
            expect_invalid=[True, True],
        ))
    return out


# --------------------------------------------------------------------------- #
# EXTREMES — real memory is messy: deep chains, wide fan-out, subtle near-misses,
# multi-parent facts. These stress the cascade where it's most likely to break; some
# are genuinely hard and a perfect score is NOT expected (that's the point).
# --------------------------------------------------------------------------- #

def t_deep_chain() -> list[Scenario]:
    """6-hop chain — far past the typical demo depth. Every link must invalidate."""
    out = []
    for frm, to in RELOCATE[:6]:
        out.append(Scenario(
            name=f"deep-6hop[{frm}->{to}]",
            category="extreme / deep-chain",
            facts=[
                f"I live in {frm}",
                f"My office is a 30 minute drive from my {frm} home",
                "I leave home at 8am for that drive",
                "I eat breakfast at 7:15am before leaving",
                "I set my alarm for 6:45am to make breakfast",
                "My smart light is scheduled to turn on at 6:40am",
            ],
            parents=[[], [0], [1], [2], [3], [4]],
            trigger=f"I now live in {to}, a 10 minute walk from a new office",
            expect_invalid=[True, True, True, True, True, True],
            note="tests deep propagation; relies on each hop's value depending on the prior",
        ))
    return out


def t_wide_fanout() -> list[Scenario]:
    """One city with 5 direct dependents — some invalidate, some SURVIVE. Tests
    selective fan-out (the whole point of typed edges + semantic stop)."""
    out = []
    for frm, to in CITIES_SAME_ZONE[:5]:
        out.append(Scenario(
            name=f"wide-fanout[{frm}->{to}]",
            category="extreme / wide-fanout",
            facts=[
                f"I live in {frm}",                              # 0 trigger target
                f"My commute from {frm} is 40 minutes",          # 1 invalidate
                "My timezone is unaffected by the city",          # 2 survive (same zone)
                f"My rent in {frm} is 1500 dollars",             # 3 invalidate
                "My native language is unaffected by the city",   # 4 survive
                f"My gym is a 5 minute walk from my {frm} place", # 5 invalidate
            ],
            parents=[[], [0], [0], [0], [0], [0]],
            trigger=f"I now live in {to}",
            expect_invalid=[True, True, False, True, False, True],
            note="5-way fan-out: 3 invalidate, 2 must be pruned",
        ))
    return out


def t_tax_boundary() -> list[Scenario]:
    """The subtle near-miss: a within-state move keeps state tax; a cross-state move to a
    no-income-tax state invalidates it. Same surface shape, opposite ground truth."""
    out = []
    # within-state (tax SURVIVES)
    for frm, to, state in [("San Francisco", "San Diego", "California"),
                           ("Buffalo", "Albany", "New York"),
                           ("Dallas", "Houston", "Texas")]:
        out.append(Scenario(
            name=f"tax-within-state[{frm}->{to}]",
            category="extreme / near-miss-negative",
            facts=[f"I live in {frm}, {state}",
                   f"I pay {state} state income tax"],
            parents=[[], [0]],
            trigger=f"I moved to {to}",
            expect_invalid=[True, False],
            note="same state -> state tax obligation unchanged",
        ))
    # cross-state into a no-income-tax state (tax INVALIDATES)
    for frm, frm_state, to in [("San Francisco", "California", "Austin, Texas"),
                               ("Portland", "Oregon", "Seattle, Washington"),
                               ("Chicago", "Illinois", "Miami, Florida")]:
        out.append(Scenario(
            name=f"tax-cross-state[{frm}->{to}]",
            category="extreme / near-miss-positive",
            facts=[f"I live in {frm}, {frm_state}",
                   f"I pay {frm_state} state income tax"],
            parents=[[], [0]],
            trigger=f"I moved to {to}",
            expect_invalid=[True, True],
            note="moved to a no-state-income-tax state -> obligation invalidated",
        ))
    return out


def t_multi_parent() -> list[Scenario]:
    """A fact derived from THREE parents. Change one parent that matters (recipe depends
    on the oven) vs one that doesn't (the wall color) — separate scenarios, opposite truth."""
    out = []
    out.append(Scenario(
        name="multi-parent-matters",
        category="extreme / divergent-parents",
        facts=[
            "My oven's max temperature is 250C",
            "My kitchen walls are painted sage green",
            "I own a cast-iron skillet",
            "My pizza recipe bakes at 250C for 8 minutes in the oven",
        ],
        parents=[[], [], [], [0, 1, 2]],
        trigger="I replaced my oven with one that maxes out at 200C",
        expect_invalid=[True, False, False, True],
        note="recipe depends on the oven temp (changed) -> invalid; walls/skillet untouched",
    ))
    out.append(Scenario(
        name="multi-parent-irrelevant",
        category="extreme / divergent-parents",
        facts=[
            "My oven's max temperature is 250C",
            "My kitchen walls are painted sage green",
            "I own a cast-iron skillet",
            "My pizza recipe bakes at 250C for 8 minutes in the oven",
        ],
        parents=[[], [], [], [0, 1, 2]],
        trigger="I repainted my kitchen walls navy blue",
        expect_invalid=[False, True, False, False],
        note="wall color changed, but the recipe does NOT depend on it -> recipe SURVIVES",
    ))
    return out


# --------------------------------------------------------------------------- #
# CHAOS / adversarial reality — input that does NOT conform to the system's clean
# assumptions: no-op observations, messy colloquial phrasing, irrelevant noise mixed in,
# oblique triggers that imply a change without stating it, off-domain chains, and triggers
# that change two roots at once. The system is EXPECTED to lose points here; reporting that
# honestly (per category) is what makes the headline number credible.
# --------------------------------------------------------------------------- #

def t_noop_observations() -> list[Scenario]:
    """The most common real case: an observation that changes nothing. Everything must
    SURVIVE. Tests the false-positive rate — a cascade that over-fires fails here."""
    out = []
    setups = [
        (["I live in Denver", "My commute is 25 minutes", "I wake at 7am"],
         [[], [0], [1]], "I had a really productive day at work today"),
        (["My salary is 90k dollars", "I save 1000 dollars a month"],
         [[], [0]], "I watched a great documentary last night"),
        (["My API runs on AWS us-east-1", "My SLA is p99 under 50ms"],
         [[], [0]], "The team had a nice lunch to celebrate the launch"),
        (["I own a Tesla", "My charger is a Wall Connector"],
         [[], [0]], "It rained heavily this afternoon"),
        (["My oven maxes at 250C", "My pizza bakes at 250C for 8 minutes"],
         [[], [0]], "I reorganized my bookshelf by color"),
    ]
    for i, (facts, parents, trig) in enumerate(setups):
        out.append(Scenario(
            name=f"noop[{i}]",
            category="chaos / no-op",
            facts=facts, parents=parents, trigger=trig,
            expect_invalid=[False] * len(facts),
            note="irrelevant observation -> nothing should change",
        ))
    return out


def t_messy_phrasing() -> list[Scenario]:
    """Same logical relocation cascade, but stated the way people actually talk —
    colloquial, run-on, emotional. Ground truth identical to the clean version."""
    rows = [
        ("Berlin", "ok so we FINALLY got the keys, officially Munich people now, "
                   "boxes everywhere lol"),
        ("Seattle", "welp, big news — packed up the whole apartment and we're down in "
                    "Portland as of this weekend"),
        ("Austin", "can't believe it but the move happened, Denver is home now, "
                   "still finding my way around"),
    ]
    out = []
    for frm, trig in rows:
        out.append(Scenario(
            name=f"messy-phrasing[{frm}]",
            category="chaos / messy-phrasing",
            facts=[f"I live in {frm}",
                   f"My commute to work is 45 minutes in {frm}",
                   f"I wake at 7am to beat the {frm} traffic"],
            parents=[[], [0], [1]],
            trigger=trig,
            expect_invalid=[True, True, True],
            note="oblique/colloquial trigger must still cascade",
        ))
    return out


def t_noise_amid_signal() -> list[Scenario]:
    """A real cascade buried among unrelated facts that must stay untouched. Tests
    selective invalidation when most of memory is irrelevant to the change."""
    out = []
    rows = [("Lyon", "Paris"), ("Munich", "Berlin"), ("Osaka", "Tokyo")]
    for frm, to in rows:
        out.append(Scenario(
            name=f"noise-amid-signal[{frm}->{to}]",
            category="chaos / noise",
            facts=[
                f"I live in {frm}",                       # 0 trigger target
                f"My commute from {frm} takes 35 minutes", # 1 invalidate
                "I am allergic to peanuts",                # 2 noise -> survive
                "My favorite color is teal",               # 3 noise -> survive
                "I have a younger sister named Maya",       # 4 noise -> survive
            ],
            parents=[[], [0], [], [], []],
            trigger=f"I now live in {to}",
            expect_invalid=[True, True, False, False, False],
            note="only the location chain moves; unrelated facts survive",
        ))
    return out


def t_offbeat_domains() -> list[Scenario]:
    """Domains outside personal/home/work, with clear structural ground truth."""
    return [
        Scenario(
            name="medical-dose",
            category="chaos / off-domain",
            facts=["My blood pressure medication dose is 10mg daily",
                   "My pharmacy refill is set for 30 tablets of 10mg"],
            parents=[[], [0]],
            trigger="My doctor changed my prescription to a different dose",
            expect_invalid=[True, True],
            note="dose changed (amount unknown) -> refill spec stale",
        ),
        Scenario(
            name="flight-itinerary",
            category="chaos / off-domain",
            facts=["My flight departs at 6pm Friday",
                   "My airport taxi is booked for 3pm Friday",
                   "I set an out-of-office starting 2pm Friday"],
            parents=[[], [0], [1]],
            trigger="The airline rebooked my flight to 9am Friday",
            expect_invalid=[True, True, True],
        ),
        Scenario(
            name="fitness-goal",
            category="chaos / off-domain",
            facts=["My goal weight is 75kg",
                   "My daily calorie target is 2000 to reach 75kg"],
            parents=[[], [0]],
            trigger="I changed my goal to gaining muscle at 82kg",
            expect_invalid=[True, True],
        ),
        Scenario(
            name="legal-jurisdiction-noop",
            category="chaos / off-domain",
            facts=["My company is incorporated in Delaware",
                   "Our contracts are governed by Delaware law"],
            parents=[[], [0]],
            trigger="We opened a new sales office in Chicago",
            expect_invalid=[False, False],
            note="a sales office does not change incorporation -> no-op",
        ),
    ]


def t_multiple_changes() -> list[Scenario]:
    """One trigger that changes TWO independent roots at once — both their chains must move,
    unrelated facts must not."""
    return [
        Scenario(
            name="dual-change-job-and-city",
            category="chaos / multi-change",
            facts=[
                "I live in Boston",                       # 0 -> changes
                "My commute in Boston is 30 minutes",      # 1 derived from 0
                "I work at Acme Corp",                     # 2 -> changes
                "My Acme badge opens the 4th floor",       # 3 derived from 2
                "I have a cat named Pepper",               # 4 noise -> survive
            ],
            parents=[[], [0], [], [2], []],
            trigger="I moved to Denver and started a new job at Globex",
            expect_invalid=[True, True, True, True, False],
            note="two independent chains invalidate from one trigger; the cat survives",
        ),
    ]


GENERATORS = [
    t_relocation_chain, t_timezone_hard_negative, t_divergent_parents,
    t_sensor_baseline, t_region_sla, t_unknown_value, t_extends_negative,
    t_org_reporting, t_language_hard_negative, t_subscription,
    # extremes
    t_deep_chain, t_wide_fanout, t_tax_boundary, t_multi_parent,
    # chaos / adversarial reality
    t_noop_observations, t_messy_phrasing, t_noise_amid_signal,
    t_offbeat_domains, t_multiple_changes,
]


def generate() -> list[Scenario]:
    out = []
    for g in GENERATORS:
        out.extend(g())
    return out


def generate_stratified(per_template: int) -> list[Scenario]:
    """Take the first `per_template` scenarios from EACH template — a representative slice
    spanning all 19 templates/categories that completes inside the cloud rate-limit window,
    so a determinism run isn't silently degraded by quota throttling on a multi-hour job."""
    out = []
    for g in GENERATORS:
        out.extend(g()[:per_template])
    return out


# --------------------------------------------------------------------------- #
# Run + score GEM vs flat baseline
# --------------------------------------------------------------------------- #

def _aggregate(results: list[dict]) -> dict:
    passed = sum(r["passed"] for r in results)
    nc = sum(r["node_correct"] for r in results)
    nt = sum(r["node_total"] for r in results)
    return {"scenarios": len(results), "passed": passed,
            "node_correct": nc, "node_total": nt,
            "node_acc": nc / nt if nt else 0.0}


def _by_category(results: list[dict]) -> dict:
    cats: dict = {}
    for r in results:
        a = cats.setdefault(r["category"], {"scen": 0, "passed": 0, "nc": 0, "nt": 0})
        a["scen"] += 1
        a["passed"] += int(r["passed"])
        a["nc"] += r["node_correct"]
        a["nt"] += r["node_total"]
    return cats


def _print_categories(gem_results: list[dict], flat_results: list[dict] | None = None) -> None:
    """Per-category GEM vs baseline. Read the TIES as carefully as the wins: a category where
    GEM gives no lift is either (a) no real dependency depth to exploit (fine) or (b)
    derive_links failed to build the edges so the cascade had nothing to walk (silent bug)."""
    gem_cats = _by_category(gem_results)
    flat_cats = _by_category(flat_results) if flat_results else {}
    print("\nby category (GEM vs baseline):")
    for c in sorted(gem_cats):
        a = gem_cats[c]
        gacc = a["nc"] / a["nt"] if a["nt"] else 0.0
        if c in flat_cats and flat_cats[c]["nt"]:
            bacc = flat_cats[c]["nc"] / flat_cats[c]["nt"]
            lift = (gacc - bacc) * 100
            tie = "  <- TIE: no lift (check: no depth, or derive_links missed edges?)" \
                if abs(lift) < 1e-9 else ""
            print(f"  {c:30} GEM {gacc:>4.0%}  base {bacc:>4.0%}  ({lift:+.0f}pt){tie}")
        else:
            print(f"  {c:30} GEM {gacc:>4.0%}  ({a['passed']}/{a['scen']} scen)")


def _print_integrity() -> None:
    cl, dl = C.DEGRADED["classify"], C.DEGRADED["derive_links"]
    tot = cl + dl
    if tot == 0:
        print("integrity:       clean (0 degraded calls)")
        return
    print("!" * 70)
    print(f"RUN INVALID — {tot} degraded LLM calls (classify={cl}, derive_links={dl}).")
    print("DISCARD this run and re-run; do NOT report the number. A degraded classify")
    print("corrupts a cascade decision; a degraded derive_links drops a DERIVED_FROM edge.")
    print("!" * 70)


def _safe_run(s, cfg, make_store, *, retries=2, quiet=False):
    """One transient cloud error shouldn't kill a long run; retry then skip."""
    for attempt in range(retries + 1):
        try:
            return run_scenario(s, make_store=make_store, cfg=cfg)
        except Exception as e:
            if attempt == retries:
                if not quiet:
                    print(f"      [skip] {s.name}: {type(e).__name__}: {e}", flush=True)
                return None
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GEM propagation eval (Unit 5.5)")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N scenarios")
    ap.add_argument("--falkor", action="store_true", help="use the FalkorDB backend")
    ap.add_argument("--no-baseline", action="store_true", help="skip the flat baseline run")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the GEM eval N times and report run-to-run determinism")
    ap.add_argument("--per-template", type=int, default=None,
                    help="stratified slice: take the first K scenarios from each template "
                         "(spans all categories, completes inside the rate-limit window)")
    args = ap.parse_args(argv)

    make_store = None
    if args.falkor:
        from .falkor_store import FalkorStore
        make_store = lambda: FalkorStore(clear_on_start=True)

    if args.per_template:
        scenarios = generate_stratified(args.per_template)
    else:
        scenarios = generate()
    if args.limit:
        scenarios = scenarios[: args.limit]
    C.reset_degraded()
    print(f"{len(scenarios)} scenarios from {len(GENERATORS)} templates"
          f"{f' (stratified {args.per_template}/template)' if args.per_template else ''}"
          f"{' (FalkorDB)' if args.falkor else ''}\n")

    gem_cfg = GEMConfig(cascade_enabled=True)
    flat_cfg = GEMConfig(cascade_enabled=False)

    # ---- determinism mode: run GEM `repeat` times, report consistency ---- #
    # Integrity is gated PER RUN: the counter is reset before each pass and a pass with any
    # degraded call is DISCARDED, not averaged in. Rate-limiting in one pass must not corrupt
    # the determinism number computed over the clean passes (the exact failure that produced a
    # bogus "mean 72%" before this gating existed).
    if args.repeat > 1:
        cat_of = {s.name: s.category for s in scenarios}
        run_accs, pass_vectors, discarded = [], [], 0
        for run in range(args.repeat):
            C.reset_degraded()
            results = [r for s in scenarios
                       if (r := _safe_run(s, gem_cfg, make_store, quiet=True)) is not None]
            agg = _aggregate(results)
            deg = C.degraded_total()
            if deg > 0:
                discarded += 1
                print(f"run {run + 1}/{args.repeat}: DISCARDED "
                      f"({deg} degraded calls; raw {agg['node_acc']:.1%} is rate-limit "
                      f"corrupted, not real)", flush=True)
                continue
            run_accs.append(agg["node_acc"])
            pass_vectors.append({r["name"]: r["passed"] for r in results})
            print(f"run {run + 1}/{args.repeat}: {agg['passed']}/{agg['scenarios']} scen, "
                  f"{agg['node_correct']}/{agg['node_total']} nodes "
                  f"({agg['node_acc']:.1%})  clean", flush=True)

        print("\n" + "=" * 64)
        if not run_accs:
            print("NO CLEAN RUNS — every pass was rate-limited. Re-run when quota resets "
                  "(smaller --per-template / --repeat).")
            print("=" * 64)
            return 1
        names = set().union(*[set(v) for v in pass_vectors])
        flippers = [n for n in names if len({v.get(n) for v in pass_vectors}) > 1]
        print(f"determinism over {len(run_accs)} CLEAN runs "
              f"({discarded} discarded as rate-limited):")
        print(f"  node accuracy: min {min(run_accs):.1%}  mean "
              f"{sum(run_accs) / len(run_accs):.1%}  max {max(run_accs):.1%}")
        print(f"  scenarios that flipped pass/fail at least once: "
              f"{len(flippers)}/{len(names)}")
        for n in sorted(flippers):
            cat = cat_of.get(n, "?")
            kind = "NEGATIVE (boundary-classify instability)" \
                if any(w in cat for w in ("negative", "no-op", "extends")) \
                else "POSITIVE (cascade-decision instability)"
            print(f"    ~ {n:38} [{kind}]")
        print("=" * 64)
        return 0

    # ---- single pass: GEM vs flat baseline, with per-category breakdown ---- #
    gem_results, flat_results, errors = [], [], 0
    for i, s in enumerate(scenarios):
        gr = _safe_run(s, gem_cfg, make_store)
        if gr is None:
            errors += 1
            continue
        gem_results.append(gr)
        line = f"[{i + 1:>3}/{len(scenarios)}] {'PASS' if gr['passed'] else 'FAIL'} " \
               f"{gr['node_correct']}/{gr['node_total']}  {s.name}"
        if not args.no_baseline:
            fr = _safe_run(s, flat_cfg, make_store)
            if fr is not None:
                flat_results.append(fr)
                line += f"   | baseline {fr['node_correct']}/{fr['node_total']}"
        print(line, flush=True)

    g = _aggregate(gem_results)
    print("\n" + "=" * 64)
    if errors:
        print(f"(skipped {errors} scenario(s) after transient errors)")
    print(f"GEM (cascade):   {g['passed']}/{g['scenarios']} scenarios, "
          f"{g['node_correct']}/{g['node_total']} nodes ({g['node_acc']:.0%})")
    if not args.no_baseline:
        b = _aggregate(flat_results)
        print(f"flat baseline:   {b['passed']}/{b['scenarios']} scenarios, "
              f"{b['node_correct']}/{b['node_total']} nodes ({b['node_acc']:.0%})")
        print(f"cascade lift:    +{(g['node_acc'] - b['node_acc']) * 100:.0f} "
              f"node-accuracy points")
    _print_integrity()
    print("=" * 64)
    _print_categories(gem_results, flat_results if not args.no_baseline else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
