"""Unit 2 step-3 ablation: does graph-proximity retrieval actually earn its place?

A feature you assert is worth nothing; this measures it. Build dependency-clustered memories,
issue a query that semantically matches the ROOT of a cluster, and measure how much of the
cluster's DEPENDENT neighborhood each method surfaces in top-k:
  - semantic-only (FAISS cosine)
  - semantic + graph-proximity (spreading activation over typed edges)

Metric: recall of the dependent members @k.

HONEST FRAMING (the caveat the design must carry): "relevant" here = "facts that depend on the
queried fact" — the context-expansion use case (when you ask about X, surface what's downstream
of X). That is exactly what a flat vector store cannot do and what the typed graph enables. It
is NOT a claim about precise single-fact lookup, where semantic search is already fine. So this
shows a capability flat stores lack, measured — not a universal "better retrieval" claim.

Run:  python -m gem.retrieval_ablation
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import GEM
from .embed import STEmbedder
from .retrieval import FaissIndex, semantic_search, graph_proximity_search


@dataclass
class Cluster:
    root: str
    dependents: list[str]   # facts that depend on root (semantically distinct from the query)
    query: str              # matches the root, NOT the dependents


CLUSTERS = [
    Cluster("I live in Bangalore",
            ["My commute to work is 45 minutes",
             "I wake at 7am to beat the morning traffic",
             "My daily briefing is scheduled for 6:45am"],
            "Where is my home located?"),
    Cluster("I work at Acme Corporation",
            ["My building badge opens the fourth floor",
             "My manager for performance reviews is Alice",
             "My assigned parking spot is number 12"],
            "Who is my employer?"),
    Cluster("I drive a Tesla Model 3",
            ["My garage charger is a Tesla Wall Connector",
             "My auto insurance policy is with Geico",
             "My license plate is 7XYZ123"],
            "What car do I own?"),
    Cluster("My annual salary is 90000 dollars",
            ["I budget 2000 dollars a month for rent",
             "I contribute 1000 dollars monthly to savings",
             "My federal tax bracket is 24 percent"],
            "What is my income?"),
    Cluster("I am studying for a PhD in molecular biology",
            ["My thesis defense is scheduled for next spring",
             "My advisor signs off on my lab budget",
             "My fellowship stipend renews each September"],
            "What degree am I pursuing?"),
]


def _recall(retrieved: list[str], targets: set[str]) -> float:
    return len(set(retrieved) & targets) / len(targets) if targets else 0.0


def main() -> int:
    emb = STEmbedder()
    sem_recalls, graph_recalls = [], []
    K = 4   # tight budget vs a memory of ~20 nodes, so retrieval is actually selective

    # ONE shared memory: every cluster's nodes are distractors for every other query. This is
    # the realistic setting — the dependents must compete against the whole store for a top-k
    # slot, and being semantically dissimilar to the query they lose that competition unless
    # the graph surfaces them.
    g = GEM(embedder=emb)
    cluster_targets = []
    for c in CLUSTERS:
        root = g.ingest(c.root, parents=[], check_conflicts=False)
        deps = [g.ingest(d, parents=[root.id], check_conflicts=False) for d in c.dependents]
        cluster_targets.append((c, {d.content for d in deps}))
    idx = FaissIndex.from_store(g.store, dim=384)
    print(f"graph-proximity ablation — {len(g.store.all_nodes())}-node shared memory, "
          f"recall of dependent neighborhood @{K}\n")

    for c, targets in cluster_targets:
        q = emb.embed(c.query)
        sem = [n.content for n, _ in semantic_search(g.store, idx, q, k=K)]
        gph = [n.content for n, _ in graph_proximity_search(g.store, idx, q, k=K)]
        sr, gr = _recall(sem, targets), _recall(gph, targets)
        sem_recalls.append(sr)
        graph_recalls.append(gr)
        print(f"  {c.query!r}")
        print(f"     semantic-only recall: {sr:.0%}   graph-proximity recall: {gr:.0%}")

    n = len(CLUSTERS)
    sm, gm = sum(sem_recalls) / n, sum(graph_recalls) / n
    print("\n" + "=" * 60)
    print(f"mean dependent-recall @{K}:  semantic-only {sm:.0%}   graph-proximity {gm:.0%}")
    print(f"graph-proximity lift: +{(gm - sm) * 100:.0f} points")
    print("=" * 60)
    print("\nCaveat (honest): 'relevant' = facts that DEPEND on the queried fact (context")
    print("expansion). This is the capability a flat vector store lacks; it is not a claim")
    print("about precise single-fact lookup, where semantic search is already sufficient.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
