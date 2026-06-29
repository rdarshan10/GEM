# Related Work: GEM vs Mem0 vs Zep

GEM is **not** a faster or more scalable Mem0/Zep — those are production systems and GEM is a
research prototype. GEM targets one mechanism that both leave open: **dependency-aware
invalidation** — when an upstream fact changes, propagating staleness to the facts *derived*
from it. This document states what each system does, where they're stronger, and the precise
gap GEM fills, so the contribution isn't overclaimed.

> This document covers the **production** systems (Mem0, Zep). For the **academic** incumbent that
> targets the same problem — the **STALE** benchmark and its **CUPMem** solution — and an honest
> head-to-head read, see [stale_eval/FINDINGS.md](stale_eval/FINDINGS.md).

## What each does

**Mem0** ([mem0.ai/research](https://mem0.ai/research)) — single-pass, **ADD-only** extraction;
agent-confirmed facts stored with equal weight. Strength is **retrieval**: multi-signal scoring
(semantic + keyword + entity) at high token efficiency (<7K tokens/call), with broad benchmarks
(LoCoMo 92.5, LongMemEval 94.4, BEAM 64.1@1M / 48.6@10M). It does **not** model dependencies,
invalidate superseded facts, or propagate contradictions.

**Zep / Graphiti** ([github.com/getzep/zep](https://github.com/getzep/zep)) — a **temporal**
knowledge graph; every fact carries `valid_at` / `invalid_at`. When a new fact contradicts an
old one, the old one is marked invalid at that time — genuine contradiction handling, and more
than Mem0 does. Production-grade, sub-200ms retrieval. But its invalidation is **direct only**:
it retires the fact that was superseded; it does **not** propagate to facts whose validity
*depended on* the retired one, and its edges are semantic relationships, not typed by
validity-dependency.

## The gap both leave open

> When an upstream fact changes, what happens to the facts **derived** from it?

- **Mem0:** nothing — ADD-only, so the stale derived fact simply persists.
- **Zep:** a derived fact like *"wake 7am to beat traffic"* is **not directly contradicted** by
  *"moved to Mumbai"* — there is no contradicting fact for the temporal mechanism to fire on.
  So it stays `valid` and silently wrong.

Neither system has a notion of *"this memory's validity depends on that one."*

## What GEM adds

1. **Typed edges** — `DERIVED_FROM` (validity depends on source) vs `ASSOCIATED` (merely
   related). The cascade walks `DERIVED_FROM` only; `ASSOCIATED` is never followed for
   invalidation. This typed distinction is what makes *selective* propagation possible and is
   absent from Zep's semantic graph.
2. **Multi-hop cascade** — change one fact, walk `DERIVED_FROM`, mark the chain stale
   (Mumbai: city → commute → wake → briefing, 4 hops).
3. **Semantic pruning** — stop where the change genuinely doesn't matter (timezone survives a
   same-zone move). Neither Mem0 nor Zep has this, because neither cascades.
4. **Two distinct passes** — conflict-check ("does this contradict?") kept separate from
   provenance ("does this depend on?"). Zep folds contradiction into one temporal mechanism;
   GEM separates detection from dependency.

## Honest scorecard

(GEM numbers throughout use **`gpt-oss:120b-cloud`** as the boundary model.)

| capability | Mem0 | Zep | GEM (gpt-oss:120b) |
|---|---|---|---|
| retrieval quality | strong | yes | ST embeddings + FAISS + **graph-proximity** |
| latency | fast | <200ms | seconds (LLM per hop) |
| scale (1M–10M memories) | benchmarked | production | prototype (~hundreds) |
| broad memory benchmarks (LoCoMo etc.) | yes | yes | no (different problem) |
| temporal / direct invalidation | no | yes | yes |
| **dependency-aware cascade** | **no** | **no** | **yes — only one** |

## Measured: Mem0 on the propagation eval (argued → measured)

The architectural argument above is now a number. Mem0 was configured per its docs with the SAME
model GEM uses (gpt-oss:120b) and a local embedder, then run through the propagation scenarios
(`gem/baseline_mem0.py`, protocol + fairness rules in `BASELINE_PROTOCOL.md`):

| scoring | GEM | Mem0 |
|---|---|---|
| **S1 native** (model-free state read — the clean architectural number) | **100%** | **44%** |
| **S2 selective retrieval** (fair MiniLM embedder both sides) | **100%** | **56%** |
| **S3 full retrieval** (re-derivation confound active) | **100%** | **56%** |

**GEM beats Mem0 at every column** — on a comparison that is fair (verified no-staleness-field,
named bulk-load paths, untuned distractors, model-free S1). But the win's *reason* is more
interesting than "retrieval-independence," and the honest details:

- **S1 is the unimpeachable number** (GEM 100% vs Mem0 44%): pure state read, no judge, no
  retrieval at scoring time — zero re-derivation confound in either direction. Mem0 has no
  staleness field, so it cannot natively mark a *derived* fact stale.
- **GEM is NOT retrieval-independent** (S2 caught this). With the *lexical* fallback, GEM dropped
  to 88% — its ingest-time conflict-check missed the conflict among distractors. A controlled A/B
  (swap to the MiniLM embedder, only that variable) recovered it to 100%, proving the cause was
  ingest-retrieval, not classification.
- **Co-retrieval confound, surfaced against us:** in the fair-embedder S2, GEM co-retrieves the
  trigger 94% vs Mem0 62% (different native representations of the same memory), so GEM's S2 100%
  is *partly* aided by re-derivation, not purely the eager flag. The clean isolation is S1.

**The refined thesis (the real contribution):** eager invalidation does not make the cascade
retrieval-*independent*; it **converts a per-query dependency into a per-write one.** Mem0 must
re-derive staleness correctly on *every read*; GEM must detect the conflict *once*, at ingest —
then the STALE flag is permanent and every subsequent read is correct regardless of retrieval.
"Did we catch it once" vs "do we catch it every time" — a better failure profile, not its
elimination. (S2 was worth running precisely because it broke the naive prediction and produced
this sharper, defensible claim.)

**Zep (architectural claim, not measured here).** Unlike Mem0, Zep's `invalid_at` *is* a native
staleness signal, so Zep would earn full credit on the **directly-contradicted** root (the cases
Mem0 also can't mark) — but it still **cannot propagate to derived dependents**, because its edges
are temporal relationships, not typed by validity-dependency. So the architectural story is the
same as Mem0 with Zep winning the direct-contradiction cases: neither cascades. This is stated
from architecture (a *weaker/safer* claim — we credit Zep with more than Mem0, no strawman); it
would be measured if a reviewer asks, with the Neo4j+Graphiti setup.

## Why a head-to-head benchmark is apples-to-oranges

Mem0/Zep report **retrieval-recall** benchmarks (LoCoMo, LongMemEval) — can you fetch the right
memory from a long history. GEM's propagation eval measures something they don't: **did a
downstream derived fact go stale when its source changed.** Their benchmarks wouldn't register
GEM's mechanism, and GEM doesn't compete on their scale/latency. The point isn't "GEM beats
Mem0" — it's that GEM implements the missing operation both leave undefined.

A related signal: on FactConsolidation (counterfactual edits), GEM scores 77% open-book but the
same model scores **0% closed-book** (`runs/closedbook_mh6k.txt`) — i.e. the benchmark tests
memory *use* (overriding the model's priors), a stricter axis than retrieval recall. That axis,
and the propagation eval, are where dependency-aware memory is the right tool.

## Retrieval that exploits the typed graph (Unit 2 step 3)

The one retrieval axis where GEM does something the others structurally cannot: **graph-proximity
search** (`gem/retrieval.py`). On top of FAISS semantic search, it spreads activation along the
`DERIVED_FROM`/`ASSOCIATED` edges, so a query that matches one memory in a dependency cluster also
surfaces the cluster's *dependent neighborhood* — facts that are downstream of the query but
semantically dissimilar to it. A flat vector store (Mem0) or an untyped/temporal graph (Zep)
cannot do this: there is no "depends-on" edge to follow. Measured (`gem/retrieval_ablation.py`,
20-node shared memory): mean dependent-recall @4 rises from **40% (semantic-only) to 67%
(+graph-proximity)** — a +27-point lift, though *not* uniform (it ties where semantic already
wins). Honest scope: "relevant" here means context-expansion (surface what depends on X), not
precise single-fact lookup. It is a capability flat stores lack, shown rather than asserted.
