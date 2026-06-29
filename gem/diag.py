"""Diagnostics for two questions the propagation eval can't answer on its own.

1. edges(...)        — ZERO-LLM. Build a scenario's setup graph (no trigger) and print its
                       DERIVED_FROM edges. Proves the edges the cascade walks actually exist,
                       so a negative-case TIE is restraint-with-edges, not restraint-by-empty
                       -graph. Setup ingests use check_conflicts=False + parents=[] -> no LLM.

2. derive_recall(...) — THE RELOCATED TEST. The eval pins edges and bypasses derive_links, so
                       derive_links' recall is otherwise unmeasured. Here we ingest each
                       scenario's facts in order with parents=None (derive_links INFERS the
                       edges), then compare the inferred DERIVED_FROM set against the harness's
                       ground-truth parents. Reports recall/precision — i.e. does provenance
                       inference rebuild the intended dependency structure? (Uses the LLM.)

Run:  python -m gem.diag edges            # zero-LLM edge-presence on the tie categories
      python -m gem.diag derive [N]       # derive_links recall over first N scenarios (LLM)
"""

from __future__ import annotations

import sys

from .engine import GEM
from .store import EdgeType
from .eval import generate
from . import classify as C


def _setup_only(s) -> tuple[GEM, list]:
    """Two-phase setup with NO trigger and NO LLM: create nodes, pin edges."""
    g = GEM()
    nodes = [g.ingest(f, parents=[], check_conflicts=False) for f in s.facts]
    for i, pidx in enumerate(s.parents):
        for j in pidx:
            g.store.add_edge(nodes[i].id, nodes[j].id, EdgeType.DERIVED_FROM)
    return g, nodes


def edges(name_substrs: list[str]) -> None:
    """Print pinned DERIVED_FROM edges for scenarios whose name matches any substring."""
    scen = [s for s in generate() if any(sub in s.name for sub in name_substrs)]
    for s in scen:
        g, nodes = _setup_only(s)
        df = [(e.src_id, e.dst_id) for e in g.store.edges()
              if e.type == EdgeType.DERIVED_FROM]
        id2content = {n.id: n.content for n in nodes}
        print(f"\n{s.name}  [{s.category}]")
        if not df:
            print("  (NO derived_from edges — flat scenario, nothing to cascade)")
        for src, dst in df:
            print(f"  {id2content[src][:42]!r:46} --derived_from--> {id2content[dst][:38]!r}")


def derive_recall(limit: int | None = None) -> None:
    """Measure derive_links recall: re-infer edges with parents=None and compare to ground truth."""
    scen = generate()
    if limit:
        scen = scen[:limit]
    C.reset_degraded()
    tot_truth = tot_inferred = tot_hit = 0
    per_scenario = []
    for s in scen:
        # ground-truth child->parent pairs (by index)
        truth = {(i, j) for i, pl in enumerate(s.parents) for j in pl}
        if not truth:
            continue  # flat scenarios have no provenance to infer
        g = GEM()
        nodes = []
        for fact in s.facts:
            # ingest in order, derive_links INFERS parents from already-present candidates
            n = g.ingest(fact, parents=None, check_conflicts=False)
            nodes.append(n)
        id2idx = {n.id: i for i, n in enumerate(nodes)}
        inferred = {(id2idx[e.src_id], id2idx[e.dst_id]) for e in g.store.edges()
                    if e.type == EdgeType.DERIVED_FROM
                    and e.src_id in id2idx and e.dst_id in id2idx}
        hit = truth & inferred
        tot_truth += len(truth)
        tot_inferred += len(inferred)
        tot_hit += len(hit)
        miss = truth - inferred
        per_scenario.append((s.name, len(hit), len(truth), len(inferred), miss, s))
        flag = "" if not miss else "  <- MISSED edges"
        print(f"  {s.name:40} recall {len(hit)}/{len(truth)}  "
              f"(inferred {len(inferred)}){flag}", flush=True)

    rec = tot_hit / tot_truth if tot_truth else 0.0
    prec = tot_hit / tot_inferred if tot_inferred else 0.0
    print("\n" + "=" * 64)
    print(f"derive_links recall:    {tot_hit}/{tot_truth} ground-truth edges ({rec:.0%})")
    print(f"derive_links precision: {tot_hit}/{tot_inferred} inferred edges ({prec:.0%})")
    deg = C.DEGRADED["classify"] + C.DEGRADED["derive_links"]
    print(f"integrity: {deg} degraded calls"
          f"{' <- DISCARD' if deg else ' (clean)'}")
    print("=" * 64)
    print("\nA recall gap here would surface on harder positive cases in real ingest, where")
    print("edges are inferred, not pinned. This is the provenance risk the cascade eval can't see.")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "edges"
    if cmd == "edges":
        # the seven tied categories — prove their edges exist
        edges(["timezone-survives", "language-survives", "tax-within-state",
               "charger-survives", "raise-unknown", "multi-parent", "noop"])
    elif cmd == "derive":
        limit = int(argv[1]) if len(argv) > 1 else None
        derive_recall(limit)
    else:
        print(f"unknown command {cmd!r}; use 'edges' or 'derive [N]'")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
