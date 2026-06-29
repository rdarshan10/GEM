# Non-Goals & Productionization Roadmap (Unit 11)

GEM is a **research prototype** built to prove one thesis: dependency-aware invalidation
along typed `DERIVED_FROM` edges works, measurably, against a fair baseline. It is *not* a
production service, and several things a production memory system would require are
**deliberate non-goals** here. This document names each, explains why it is correctly out
of scope for the research bar, and sketches what closing it would take — so the boundary is
a documented decision, not an omission.

The one production concern that *was* worth building even at the prototype stage: a
**retry + JSON-guard wrapper** on every LLM call (`gem/classify.py:_robust_json`). Local and
cloud models emit malformed JSON and transient HTTP errors in normal use; without this the
cascade crashes on the first bad response. It's cheap, it's load-bearing for any real use,
and it stabilizes long eval runs — so it's in. Its companion is an **integrity counter**
(`classify.DEGRADED`): every time retries are exhausted and a call falls back to its safe
default, it's counted, so a degraded run (e.g. cloud rate-limiting) is reported rather than
silently trusted. Everything below is intentionally not.

---

## The six non-goals

### 1. Automated test suite + CI — BUILT
- **Status:** done. A `pytest` suite (`tests/`, **44 tests**) covers the deterministic layers
  with a mock LLM + mock embedder so the cascade *logic* is exercised with no network: store
  CRUD/edges, SubEM scorer, embedding + FAISS-vs-numpy parity, graph-proximity, classify parsing
  + voting + degraded counter, the **cascade graph-walk** (multi-hop, semantic stop, cycle guard,
  every `_apply` label, conservative fail-safe, decision caching, similarity gate, escalation),
  the eval generator's ground-truth structure, the Unit 0 pipeline, and FalkorDB parity
  (auto-skips without a server). Coverage on the core logic modules is 82–100% (`engine.py` 88%);
  the 0%-covered modules are runnable harness scripts, validated by execution. A GitHub Actions
  workflow (`.github/workflows/test.yml`) runs it on push — no cloud creds needed (all mocked).
- **Remaining:** recorded-fixture tests for the real LLM-judged paths (currently validated by
  reproducible `runs/` artifacts), and wiring the cascade-determinism check into CI as a nightly.

### 2. Engine fault-tolerance beyond the LLM wrapper
- **Status:** built — every `classify`/`derive_links` call retries, and **on exhausted retries
  returns a safe default** (`classify`→`UNRELATED`, `derive_links`→`[]`) so the cascade branch
  stops cleanly instead of throwing; the fallback is counted (see integrity counter above).
  Broader fault-tolerance — transactional ingest, idempotent re-ingest, dead-letter on repeated
  failure — is absent.
- **Why out of scope:** single-process, single-user eval runs don't exercise these paths.
- **To close:** make `ingest`/`propagate` transactional (all-or-nothing per observation),
  add idempotency keys so a retried ingest doesn't double-write, and a quarantine path for
  observations that repeatedly fail classification.

### 3. Latency & cost controls
- **Status:** every ingest fires several capable-model calls; a deep cascade is tens of seconds
  and real per-call cost. Two levers are now BUILT and MEASURED (`gem/cost_eval.py`,
  `runs/cost_eval.txt`): decision **caching** (`GEMConfig.cache_decisions`, default on) and
  **escalation** (`GEMConfig.escalate` + a `cheap_llm`: cheap model first-passes the conflict
  scan, capable model confirms destructive decisions).
- **Measured result (the honest part):** escalation cut capable-model calls **41 → 18 (−56%)**
  BUT dropped accuracy **100% → 81%** on the 11-scenario suite. Cause: escalation only confirms
  the cheap model's *destructive* hits; a cheap **false-UNRELATED** (missed conflict) never
  escalates, so the cascade silently doesn't fire — the exact staleness failure the project
  opposes. **So naive escalation is a cost/accuracy TRADE, not a free win, and for a
  correctness-first system it's the wrong trade.** Caching is the accuracy-free lever (it only
  memoizes) but had a low hit rate on distinct scenarios; it pays off on recurring patterns.
- **The accuracy-preserving lever — BUILT and MEASURED (`gem/cost_eval_sim.py`,
  `runs/cost_eval_sim.txt`): a SIMILARITY GATE on the conflict-scan** (`conflict_sim_threshold`).
  It skips the LLM call for neighbors below a cosine threshold — they're too dissimilar to
  *possibly* be a conflict — and classifies everything above it. Grounded in the measured
  separation (real conflicts >=0.41, distractors <=0.32, so a 0.35 gate is safe). Result on 6
  scenarios buried in 20 distractors (`candidate_k=24`): conflict-scan LLM calls **131 -> 16
  (−88%)** with accuracy **held at 100%** (16/16). This is the cut escalation-to-cheap couldn't
  give, because it uses only embedding distance (no judgment) — the capable model still
  classifies every plausible candidate, so there are no false-negative misses.
- **Conclusion:** the *wasteful* part of the cascade cost (scanning every neighbor) is eliminable
  with no accuracy loss; the *irreducible* part (classifying the real candidates + one capable
  call per actual cascade hop) is inherent to the capability. Decision caching (accuracy-free) is
  the complementary lever for recurring patterns. The naive confirm-destructive escalation remains
  available but is the wrong trade for correctness-first use.

### 4. Concurrency safety
- **Status:** the *restart-correctness* piece is fixed — `FalkorStore` seeds its id counter
  from the **max existing id** on init (not `count(n)`), so a single-writer restart can't
  reissue an id and overwrite a persisted node. What remains a non-goal is true concurrency:
  `new_id` is still per-process and writes are plain Cypher with no transactions/locking, so
  two simultaneous writers could race on ids or interleave a cascade.
- **Why out of scope:** the eval is sequential and single-writer.
- **To close:** server-generated ids (DB sequence / UUIDv7), wrap each observation's
  mutations in a transaction, and take a per-affected-subgraph lock (or an actor/queue model)
  so two cascades touching overlapping nodes can't interleave.

### 5. Nondeterminism guardrails
- **Status:** cascade decisions are LLM judgments and can flip run-to-run. This prototype
  *measures* that variance rather than hiding it. **Measured (gpt-oss:120b-cloud, 5 clean
  passes, 0 rate-limit-degraded — `runs/determinism_120b.txt`): node accuracy min 95.1% /
  mean 99.0% / max 100.0%; 2/19 scenarios flipped, BOTH positive-cascade cases (the 6-hop
  chain and the oblique-trigger case) — none on negatives or boundary pruning.** So the
  variance is small and confined to deep-chain decision points, not the restraint logic.
  **One identified source of that deep-chain instability has since been fixed** (not a
  sampling effect but a deterministic engine bug): when a single trigger directly conflicts
  with several nodes in the *same* `DERIVED_FROM` chain, `ingest` used to fire an independent
  cascade per conflict, so a descendant-conflict re-revised a subchain the root's cascade had
  already corrected — leaving an intermediate node wrongly ACTIVE. The fix shares one cascade
  frontier across an ingest's conflict-actions and processes ancestors first (`engine.py`
  ingest step 5 / `_order_actions_root_first`), guarded by
  `tests/test_engine.py::test_multi_direct_conflict_on_chain_revises_each_node_once`. It was
  surfaced by the deep cross-domain eval (`gem/eval_diverse.py`), which after the fix is
  12/12 scenarios / 37/37 nodes on gpt-oss:120b — the narrow relocation eval could not have
  found it.
  `python -m gem.eval --repeat N` reports per-run accuracy, min/mean/max, and the flip list
  tagged positive (cascade-decision instability) vs negative (boundary instability); a
  per-run integrity counter discards any rate-limit-degraded pass so the number isn't
  silently corrupted. What is NOT built: caching decisions for exact reproducibility,
  confidence thresholds gating destructive actions, or human-in-the-loop on low-confidence
  invalidations. (The small-model work added two of these as opt-in: a conservative fail-safe
  and escalation-triggered self-consistency voting — see the model-tier notes.)
- **Why out of scope:** for the thesis, *measuring and reporting* the variance is the honest
  move; *suppressing* it is a product decision.
- **To close:** cache decisions so a given (fact, change) resolves identically; gate
  destructive actions (SUPERSEDED) behind a confidence threshold, routing low-confidence ones
  to STALE-for-review instead; optionally majority-vote across N samples for high-stakes edges.

### 6. Operational basics (auth, rate limiting, logging, metrics, config)
- **Status:** none. No service to secure (no API yet), no structured logging or metrics, config
  is constructor args + env vars.
- **Why out of scope:** there is no deployed surface to operate. These attach to the API layer
  (plan Unit 8), which is itself optional for the portfolio.
- **To close:** with the FastAPI layer — auth (API keys/OAuth), per-tenant rate limits,
  structured request/cascade logging, metrics (ingest latency, cascade depth/fan-out,
  invalidation counts), and a typed settings object (pydantic-settings).

---

## Why this boundary is the right one

Building auth, locking, and metrics into a thesis prototype spends effort where the problem
*wasn't*, and weakens rather than strengthens the signal — it reads as not knowing what to
prioritize. Demonstrating that the mechanism works, with a fair baseline and honest variance
reporting, is the result. Knowing *exactly* what productionization would take — and having
drawn the line deliberately — is the complementary signal this document is meant to carry.
