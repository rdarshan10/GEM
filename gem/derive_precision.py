"""Lever #1 — derive_links PRECISION via a confirmation pass.

derive_links infers edges at ~88% recall / ~84% precision (runs/diag_derive_diverse.txt): ~1 in 6
inferred edges is spurious -> over-cascade (staling facts that were fine). The confirmation pass
(classify.derive_links(confirm=True)) re-checks each proposed edge in isolation and drops the ones
that fail the strict change-test. This SHOULD raise precision; the question is how much recall it
costs. Same diverse set + same all-other-candidates protocol as the baseline, so numbers compare.

Run (needs a healthy capable model):  python -m gem.derive_precision
"""

from __future__ import annotations

from .store import Node
from .eval_diverse import DIVERSE
from . import classify as C
from .llm import make_llm


def measure(llm, confirm: bool):
    tot_truth = tot_inf = tot_hit = 0
    for s in DIVERSE:
        truth = {(i, j) for i, pl in enumerate(s.parents) for j in pl}
        if not truth:
            continue
        nodes = [Node(id=f"n{i + 1}", content=c) for i, c in enumerate(s.facts)]
        id2idx = {n.id: i for i, n in enumerate(nodes)}
        inferred = set()
        for i, n in enumerate(nodes):
            cands = [m for k, m in enumerate(nodes) if k != i]
            for cid in C.derive_links(llm, n.content, cands, confirm=confirm):
                inferred.add((i, id2idx[cid]))
        tot_truth += len(truth)
        tot_inf += len(inferred)
        tot_hit += len(truth & inferred)
    rec = tot_hit / tot_truth if tot_truth else 0.0
    prec = tot_hit / tot_inf if tot_inf else 0.0
    return tot_hit, tot_truth, tot_inf, rec, prec


def main() -> int:
    llm = make_llm()
    C.reset_degraded()
    print("derive_links PRECISION lever — confirmation pass OFF vs ON (diverse scenarios)\n")
    print(f"{'mode':18} {'recall':16} {'precision':18} {'spurious edges':14}")
    rows = {}
    for label, confirm in (("baseline (off)", False), ("confirm pass (on)", True)):
        hit, truth, inf, rec, prec = measure(llm, confirm)
        rows[label] = (rec, prec, inf - hit)
        print(f"{label:18} {f'{hit}/{truth} ({rec:.0%})':16} {f'{hit}/{inf} ({prec:.0%})':18} {inf - hit:<14}")
    deg = C.DEGRADED["classify"] + C.DEGRADED["derive_links"]
    print(f"\nintegrity: {deg} degraded calls{' <- DISCARD' if deg else ' (clean)'}")
    (r0, p0, s0), (r1, p1, s1) = rows["baseline (off)"], rows["confirm pass (on)"]
    print(f"\nconfirm pass: precision {p0:.0%} -> {p1:.0%} ({p1 - p0:+.0%}), "
          f"recall {r0:.0%} -> {r1:.0%} ({r1 - r0:+.0%}), spurious edges {s0} -> {s1}")
    print("Verdict: worth it iff precision gain (less over-cascade) outweighs the recall cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
