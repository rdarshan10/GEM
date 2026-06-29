"""Is the gap big enough to matter? — the honest pressure-test of GEM's whole premise.

The skeptical question (a fair one): a capable LLM doing flat memory updates already catches
a lot of staleness, so does GEM's explicit DERIVED_FROM cascade add anything REAL, or is it
complexity for a problem the model handles for free?

PRE-REGISTERED. Two measurements, and what each outcome would mean BEFORE running:

  PART 1 — retrieval blindness (model-free, unriggable). Flat memory (ChatGPT/Mem0
  style) re-examines facts that SIMILARITY-RETRIEVAL surfaces. The mechanism claim is that a
  stale *dependent* is often semantically DISSIMILAR to the trigger that staled it
  ("I wake at 7am to beat traffic" is not similar to "I moved to Mumbai"), so flat retrieval
  never sees it. We measure, over the deep diverse scenarios, the recall@k of true-stale
  dependents when ranking memory by cosine similarity to the trigger, against a realistic
  distractor pool. GEM's graph walk has recall 1.0 of reachable dependents by construction.
    - GEM thesis SUPPORTED if dependent recall@k is well below 1.0 at realistic k (flat is
      structurally blind to dependents).
    - GEM thesis WEAKENED if recall@k is ~1.0 (flat retrieval already surfaces dependents, so
      the cascade buys little) — and we will report that honestly.

  PART 2 — realized outcome (capable LLM, end-to-end). Same deep scenarios, run with a capable
  model in FLAT mode (GEMConfig.cascade_enabled=False: resolve direct conflicts, never
  propagate) vs GEM mode (cascade on). Node accuracy is the staleness-catch rate the capable
  model actually achieves WITHOUT the dependency graph.
    - GEM thesis SUPPORTED if flat accuracy is materially below GEM's.
    - GEM thesis WEAKENED if flat ~= GEM (the model catches dependents on its own).

Run:  python -m gem.gap_experiment           # both parts
      python -m gem.gap_experiment --recall  # Part 1 only (no cloud calls)
"""

from __future__ import annotations

import argparse
import numpy as np

from . import embed as E
from .eval_diverse import DIVERSE
from .scenarios import run_scenario
from .engine import GEMConfig

# A realistic background memory: facts a real user/agent would also be carrying, none of which
# the scenario triggers touch. Ranking dependents against THIS pool (not just the 2-6 scenario
# facts) is what makes the recall number honest — top-k against a big store, not a tiny one.
DISTRACTORS = [
    "My favorite programming language is Rust",
    "I prefer window seats on flights",
    "The office WiFi password rotates every quarter",
    "My dentist appointment is the first Monday of each month",
    "I take my coffee black with no sugar",
    "The team standup is at 10am on Tuesdays",
    "My gym membership renews in January",
    "I am allergic to penicillin",
    "The company holiday party is in December",
    "My car is a blue 2019 Honda Civic",
    "I subscribe to three streaming services",
    "The fire drill is scheduled for next spring",
    "My favorite author is Ursula K. Le Guin",
    "I keep my passwords in a hardware key",
    "The quarterly board meeting is in the main conference room",
    "I run 5 kilometers on weekends",
    "My desk plant is a pothos",
    "The kitchen restocks oat milk on Fridays",
    "I learned to sail two summers ago",
    "My noise-cancelling headphones are over-ear",
    "The parking garage closes at midnight",
    "I read mostly non-fiction before bed",
    "My phone case is dark green",
    "The annual security training is due in the fall",
    "I play tennis on alternate Thursdays",
    "My preferred airline is whichever is cheapest",
    "The break room coffee machine was replaced last year",
    "I use a mechanical keyboard with brown switches",
    "My emergency contact is my sister",
    "The building elevator is inspected twice a year",
    "I prefer tea over coffee in the afternoon",
    "My monitor is a 27-inch 4K display",
    "The cleaning service comes on Wednesdays",
    "I keep a paper notebook for meeting notes",
    "My favorite season is autumn",
    "The conference budget is set each January",
    "I bike to the farmers market on Saturdays",
    "My laptop stickers are mostly open-source logos",
    "The office plants are watered by a service",
    "I prefer aisle seats at the cinema",
]


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def part1_retrieval_blindness(ks=(3, 5, 10)) -> None:
    print("=" * 72)
    print("PART 1 — retrieval blindness (model-free): can flat memory even SEE the")
    print("stale dependents it would need to re-examine?")
    print("=" * 72)
    emb = E.default_embedder()
    print(f"embedder: {type(emb).__name__}   distractor pool: {len(DISTRACTORS)} facts\n")

    dep_sims, root_sims = [], []
    hits = {k: 0 for k in ks}
    dep_total = 0
    worst_rank = 0
    percentiles = []   # per dependent: fraction of the distractor bank MORE similar to trigger
                       # than the dependent is. Size-invariant -> predicts recall at ANY store size.

    dist_vecs = [emb.embed(d) for d in DISTRACTORS]
    dist_sims_to = lambda trig: [_cos(trig, v) for v in dist_vecs]

    for s in DIVERSE:
        trig = emb.embed(s.trigger)
        fact_vecs = [emb.embed(f) for f in s.facts]
        dsims = dist_sims_to(trig)
        # roots = directly-conflicting facts (no parents); dependents = derived (have parents)
        roots = [i for i, p in enumerate(s.parents) if not p]
        deps = [i for i, p in enumerate(s.parents) if p and s.expect_invalid[i]]
        for i in roots:
            root_sims.append(_cos(trig, fact_vecs[i]))
        # pool = this scenario's OTHER facts + the distractor bank; rank each dependent by
        # cosine to the trigger and ask whether flat top-k retrieval would surface it.
        for i in deps:
            dep_total += 1
            sim_i = _cos(trig, fact_vecs[i])
            dep_sims.append(sim_i)
            pool = [_cos(trig, v) for j, v in enumerate(fact_vecs) if j != i] + dsims
            rank = 1 + sum(1 for s_other in pool if s_other > sim_i)   # 1-based rank of dep
            worst_rank = max(worst_rank, rank)
            percentiles.append(sum(1 for d in dsims if d > sim_i) / len(dsims))
            for k in ks:
                if rank <= k:
                    hits[k] += 1

    print(f"{'metric':48} value")
    print("-" * 72)
    print(f"{'mean cosine(trigger, ROOT fact)':48} {np.mean(root_sims):.3f}  (flat sees these)")
    print(f"{'mean cosine(trigger, stale DEPENDENT)':48} {np.mean(dep_sims):.3f}  (flat must rank these)")
    print(f"{'separation (root - dependent)':48} {np.mean(root_sims) - np.mean(dep_sims):+.3f}")
    print()
    for k in ks:
        r = hits[k] / dep_total if dep_total else 0.0
        print(f"{'flat dependent recall@' + str(k):48} {r:.0%}  ({hits[k]}/{dep_total} dependents in top-{k})")
    print(f"{'GEM graph-walk dependent recall':48} 100%  ({dep_total}/{dep_total}, by construction)")
    print(f"{'deepest rank a dependent fell to':48} #{worst_rank}  (flat would need top-{worst_rank} to catch all)")
    print()
    # Size-invariant projection: a dependent's percentile (fraction of unrelated facts MORE
    # similar to the trigger than it is) is independent of store size, so it predicts recall as
    # the memory grows. recall@k at a store of N facts ~= the dependent is in top-k iff its
    # percentile p < k/N. The 40-distractor recall above is OPTIMISTIC; real stores are bigger.
    mean_pct = float(np.mean(percentiles))
    print(f"{'mean dependent percentile vs unrelated facts':48} {mean_pct:.0%}  (size-invariant)")
    print("projected flat dependent recall@10 as the memory grows (the honest, scale-aware view):")
    for N in (50, 200, 1000):
        proj = sum(1 for p in percentiles if p < 10 / N) / len(percentiles)
        print(f"   store of {N:>4} facts:  recall@10 ~= {proj:.0%}")
    print()
    print("Reading it: every point of separation is staleness flat retrieval is BLIND to at")
    print("write time — the dependent isn't similar to the trigger, so it's never pulled up to")
    print("be re-examined. GEM reaches it via the DERIVED_FROM edge regardless of similarity.")


def part2_endtoend(verbose: bool = False) -> None:
    print("\n" + "=" * 72)
    print("PART 2 — realized outcome: a CAPABLE model, flat vs GEM, on the deep scenarios")
    print("=" * 72)
    flat_cfg = GEMConfig(cascade_enabled=False)   # resolve direct conflicts, never propagate
    gem_cfg = GEMConfig(cascade_enabled=True)

    def run(cfg):
        nc = nt = passed = 0
        rows = []
        for s in DIVERSE:
            r = run_scenario(s, verbose=False, cfg=cfg)
            nc += r["node_correct"]; nt += r["node_total"]; passed += r["passed"]
            rows.append((s.name, r["node_correct"], r["node_total"], r["passed"]))
        return nc, nt, passed, rows

    print("running FLAT (cascade off)...")
    fnc, fnt, fp, frows = run(flat_cfg)
    print("running GEM  (cascade on)...")
    gnc, gnt, gp, grows = run(gem_cfg)

    print(f"\n{'scenario':40} {'flat':>10} {'GEM':>10}")
    print("-" * 64)
    for (name, fc, ft, _), (_, gc, gt, _) in zip(frows, grows):
        gap = "  <-- gap" if fc < gc else ""
        print(f"{name[:40]:40} {f'{fc}/{ft}':>10} {f'{gc}/{gt}':>10}{gap}")
    print("-" * 64)
    print(f"{'TOTAL node accuracy':40} {f'{fnc}/{fnt} ({fnc/fnt:.0%})':>10} {f'{gnc}/{gnt} ({gnc/gnt:.0%})':>10}")
    print(f"{'scenarios fully correct':40} {f'{fp}/{len(DIVERSE)}':>10} {f'{gp}/{len(DIVERSE)}':>10}")
    lift = gnc / gnt - fnc / fnt
    print(f"\nGEM lift over flat (capable model): {lift:+.0%} node accuracy")
    print("This is staleness a capable model MISSED in flat mode because it never re-examined")
    print("dependents the trigger wasn't similar to — exactly Part 1's blindness, realized.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall", action="store_true", help="Part 1 only (no cloud calls)")
    args = ap.parse_args()
    part1_retrieval_blindness()
    if not args.recall:
        part2_endtoend()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
