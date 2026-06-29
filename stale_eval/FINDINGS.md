# STALE method exploration — findings

Evaluating GEM against the **STALE** benchmark (arxiv 2605.06527, "Can LLM Agents Know When
Their Memories Are No Longer Valid?") and its solution **CUPMem**, plus prototyping alternative
invalidation mechanisms. All runs on `gpt-oss:120b-cloud` unless noted. Scored with STALE's own
judge rubric (Dim1 state-resolution, Dim2 premise-resistance, Dim3 policy-adaptation).

## Context
- The problem GEM targets (propagated / Type-II invalidation: a change cascades to dependent
  facts) is **externally validated** — STALE benchmarks it; Memora finds 64% of agent errors are
  "outdated memory not forgotten." It is **not** uncontested: STALE + CUPMem already address it.
- **CUPMem ≠ GEM** (read from their code): CUPMem is *ontology-driven* (fixed buckets→tracks) with
  a 4-stage query pipeline (readout→premise_verifier→basis_recovery→action_grounding). GEM is
  *graph-driven* (explicit inferred `DERIVED_FROM` edges) with a simple query side. GEM's
  differentiation: schema-free, auditable (`why()`), far simpler. CUPMem hits 68% on STALE.

## Harness
- `adapter.py` — runs methods on STALE-format data, scores with STALE's judge. Drop-in for their
  `run_target_model.py` (same I/O); their judge is reused unchanged.
- `path1.json` — 5 hand-authored personal implicit-conflict scenarios (cascade-favorable).
- `methods.py` — `TriggerMemory` (inverted invalidator index + dedup + lifecycle cleanup);
  bounded lazy retrieval helper.
- `depth_test.py` — 6-hop chain; change hits root, query asks about the distant leaf.

## What is SOLID (high confidence)
1. **The graph is load-bearing — shallow AND deep.** Bounded pure-retrieval `lazy` (no graph)
   dropped to 67–73% because it misses conflicts not embedding-similar to the query
   ("vegan" ≁ "chicken caesar"). The graph connects them regardless of similarity.
2. **Dead ends (measured, dropped):**
   - `gemv` (CUPMem-style query verification): did **not** help, slightly hurt (over-hedged → Dim3
     "too vague" fails). Simplicity beat it.
   - `trigger` (#2, pre-enumerated invalidators + **embedding** match): detection is **broken** —
     "moved to Mumbai" ↔ "relocates to a different city" cosine = **0.21**. Concrete→abstract
     matching needs reasoning, not embedding distance. #2's "cheap no-LLM detection" premise fails.
   - bounded pure-`lazy`: fails without the graph (see #1).
3. **`TriggerMemory` lifecycle (answers "won't the index bloat?"):** inverted index dedups shared
   invalidators (3 entries for 5 fact-links) and `invalidate`/`delete` prune dead facts from every
   trigger + drop empty triggers → index stays ~proportional to *active* facts. (Structure is
   sound; its embedding *detector* is what fails, per #2.)
4. **Eager cascade carries the change hop-by-hop.** Naive graph-guided `lazy` (find one stale
   ancestor → one check) **failed 6-hop even with pinned edges (0/3)**: the root was UPDATED
   in-place (stayed ACTIVE), so a status-only ancestor check found nothing, and the "what changed"
   signal was lost. Fix: reconstruct the whole chain leaf→root with change markers
   (status + CONTRADICTS-edge contradictor) and reason over it in one call.

## What is UNRESOLVED (needs a clean, larger run)
- **Does graph-guided lazy match eager on DEPTH?** The chain-reconstruction fix improved glazy
  (0→1–2/3) but the confirming runs were **corrupted by transient cloud degradation** (eager
  swung 3/3→1/3 on the identical scenario; ~13–21s/call). Inconclusive.
- **Does GEM approach CUPMem's 68% on real STALE data?** Not measured. Gated by `derive_links`
  recall (~85–88%) on messy dialogue and by STALE's 150K-token haystacks (GEM's cost regime).
- **Shallow ranking** (glazy 87 vs gem 80) is within 5-scenario noise (gem also ran 100% once).

## Architecture direction (per evidence, not yet fully proven)
> Write: extract facts + build cheap explicit `DERIVED_FROM` edges + **eager direct-conflict
> detection** — but **no eager propagation**. Query: retrieve → **walk the graph** (reconstruct
> the chain leaf→root) → one bounded reasoning call → answer.

Rationale, each from a finding: keep the graph (load-bearing for depth + dissimilar conflicts);
eager *detection* (bounded-lazy failed without it); lazy *propagation* (the cost-wall fix); reason
don't embedding-match (0.21 cosine); simple query side (`gemv` lost). The **single open question**
is whether lazy propagation truly recovers eager on deep chains — to settle on a clean run.

## Next (when cloud is healthy / on Groq)
1. Clean depth test ×N: glazy(chain-reconstruction) vs eager vs pure-lazy on the 6-hop.
2. Path-2 subset (~40–50 real STALE scenarios, Groq-8B for rate limits): GEM vs full-context vs —
   ideally — CUPMem, judged by STALE's judge. Report the gap to 68% honestly.
