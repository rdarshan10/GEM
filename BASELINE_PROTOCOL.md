# Baseline Comparison Protocol (pre-registered)

Written **before** running any baseline, so the comparison can't be accused of tuning the
baselines down after seeing GEM win. The integrity of this comparison matters more than the
result. The goal is NOT to make Mem0/Zep look bad — it's to configure them *competently, per
their own best practices*, and measure whether they can do dependency-aware invalidation on the
propagation axis. If they land at baseline there, it's because their architectures don't model
dependency — not because they were handicapped.

## Fairness commitments (fixed in advance)

1. **Same LLM for all three systems** — `gpt-oss:120b-cloud` via the local Ollama proxy
   (`OLLAMA_HOST`). GEM, Mem0, and Zep/Graphiti all use it. This isolates the variable under
   test (memory *architecture*) from model capability. No system gets a stronger model.
2. **Same scenarios** — the propagation eval scenarios (`gem/eval.py`), same generator, same
   ground truth. Start with the stratified slice (1–2 per template); extend to the full set if
   time permits.
3. **Configured per each system's own docs / recommended settings** — Mem0 with its graph/
   relational config where applicable and default extraction; Zep/Graphiti with its standard
   temporal-graph setup. Give each every fair advantage its own documentation suggests.
4. **Pinned versions** — record `mem0ai` and `graphiti/zep` versions in the results file.
5. **Identical scoring harness.** Because Mem0/Zep have no GEM-style node `status`, the common,
   fair measure is **query-based**: per scenario, after ingesting the facts AND the trigger,
   query each system for the current value of every dependent fact. Score a node:
   - **correct** if the system's answer reflects the change — for a fact that SHOULD be
     invalidated, the system no longer asserts the stale value as current (it returns the new
     value, "unknown", or flags it); for a fact that SHOULD survive (hard negative), the system
     still returns it unchanged.
   - **wrong** if the system asserts the stale value as current on a should-invalidate node, or
     drops a should-survive node.
   This is the same pass/fail semantics as the GEM eval, expressed through queries so every
   system can be measured the same way. GEM is scored through the SAME query harness, not its
   internal status, so the comparison is apples-to-apples.

   **Confound (identified pre-run): query-time re-derivation.** With one strong shared LLM, if a
   system's retrieval surfaces BOTH the dependent fact AND the trigger ("moved to Mumbai"), the
   LLM can re-derive staleness at query time *without any cascade*. On tiny scenarios (5 facts,
   all retrieved) this would let Mem0/Zep "pass" via reasoning, masking the architectural gap and
   *understating* GEM. The cascade's real value is eager invalidation that survives **selective
   retrieval** (when the trigger is NOT co-retrieved). So we report TWO scorings side by side:
   - **(A) Native invalidation** — does the system's OWN state mark the derived fact
     stale/invalid, with no LLM re-derivation? Measures the architecture directly. (Prediction:
     GEM yes; Mem0 no; Zep only on directly-contradicted facts.)
   - **(B) Query-based under SELECTIVE retrieval** — embed each scenario in a larger memory with
     distractors so the trigger isn't always co-retrieved, then query via each system's own
     top-k retrieval + the shared LLM. Measures end-to-end behavior INCLUDING fair re-derivation.
   Reporting both is the honest move: (A) shows the capability, (B) shows the practical effect.
   Neither alone is sufficient; together they can't be gamed in either direction.

## Predictions (registered before running)

Based on the architectures (RELATED_WORK.md), the expected outcome — stated now so it can't be
retrofitted:

- **Negatives / no-op / hard-negatives:** all three correct. The right behavior is "don't
  change the dependent," which every system does by default (none propagates).
- **Multi-hop positives:** GEM correct; **Mem0 ~baseline** (ADD-only — the stale dependent
  persists and is returned as current); **Zep ~baseline** on the *derived* dependents (they are
  not *directly* contradicted by the trigger, so its temporal invalidation never fires on them),
  though Zep may correctly handle the *directly* superseded root.
- **Net prediction:** GEM > Mem0 ≈ Zep on dependent-invalidation, with the gap concentrated on
  multi-hop positives — i.e. exactly the flat-baseline pattern, because neither models
  dependency. If a baseline beats this prediction, that's a real finding and gets reported as-is.

## What would falsify the GEM claim

If Mem0 or Zep, configured per their docs, correctly invalidates *derived* (not directly
contradicted) facts on multi-hop positives at a rate near GEM's, then the "only GEM cascades"
claim is wrong and must be retracted. The protocol is designed to give them the chance to do so.

## Three-column scoring (refined; S1 and S2 pre-registered BEFORE running)

Report all three, because the GAP between them is itself the result:

- **S1 — Native invalidation (the architectural headline).** Read each system's persisted state
  after the trigger, with **zero LLM and zero embedding at scoring time** — a pure field/string
  read. Definition is GENEROUS per system: a node counts "natively invalidated" if the system's
  OWN state marks it stale/updated/removed. GEM: `node.status != ACTIVE` or content changed
  (field read). Mem0: its memories carry NO staleness field (verified by inspecting memory keys);
  a node is natively invalidated only if its original value string no longer appears in
  `get_all` (Mem0's update logic deleted/superseded it). Zep: its `invalid_at` IS a native
  staleness signal — Zep gets FULL credit wherever its direct-contradiction invalidation fires
  (we expect it to win the directly-superseded root and lose only the derived dependents).
- **S2 — Query under SELECTIVE retrieval (the realistic end-to-end number).** Embed each scenario
  in a larger distractor corpus so the trigger isn't reliably co-retrieved, then query via each
  system's own top-k + the shared judge. Pre-committed, knob-free rules (fixed before the run):
  - same distractor corpus (one fixed list), same `k` (=6), same scenarios for all systems;
  - distractors set WITHOUT reference to where the trigger lands;
  - **"each system under its OWN standard bulk-load path"** — GEM ingests distractors via its
    normal `ingest(check_conflicts=False)`; Mem0 loads them via its bulk `add(infer=False)`. The
    fairness claim is "each system operating normally", NOT byte-identical memory — named here so
    the asymmetry can't be raised as a hidden objection;
  - **report the trigger co-retrieval RATE PER SYSTEM** (GEM's and Mem0's retrieval may surface
    the trigger at different rates; if GEM co-retrieves it MORE, that is a confound AGAINST us and
    must be shown). Same `k`, same corpus, rate measured both ways — conditions transparent
    whichever way they cut.
  Predicted shape (the argument is the shape, not one number): GEM ≈ flat across S1/S2/S3 (eager
  invalidation is retrieval-independent); Mem0 declines S3 → S2 (re-derivation degrades as
  co-retrieval drops). GEM-flat vs Mem0-sloping-down is what shows the mechanism's value is
  robustness to retrieval conditions — what eager pre-computation buys and re-derivation can't.
- **S3 — Query under FULL retrieval (the confound, displayed deliberately).** The naive number
  where re-derivation is active. We show it on purpose: "here's the number that can look like a
  wash, here's why it's misleading (the judge re-derives), here's the realistic number (S2)."

Expected pattern: for GEM, S1 ≈ S2 ≈ S3 (eager invalidation survives selective retrieval). For
Mem0/Zep, S2 < S3 (re-derivation collapses when the trigger isn't retrieved) and S1 is the floor.
That divergence is the cleanest demonstration that re-derivation is a crutch that breaks at scale.

Verification commitments (checked before reporting): S1's read path is literally model-free (no
chat, no embedding) — verified the same way the closed-book ablation was verified truly closed.

## Results

All Mem0 runs: same judge/model (gpt-oss:120b), HF MiniLM embedder, 6 representative scenarios.
Mem0 memory fields verified to contain NO staleness signal (`created_at, hash, id, memory,
metadata, updated_at, user_id`) — so S1's "cannot natively invalidate" is checked, not assumed.

- **S1 (native, model-free state read):** GEM **16/16 (100%)** vs Mem0 **7/16 (44%)** —
  `runs/baseline_mem0_native.txt`. Mem0's 44% = the should-SURVIVE facts it correctly keeps, plus
  a few directly-conflicting facts its update-logic happened to remove/rephrase. It cannot mark
  any *derived* dependent stale (no mechanism) — the cascade-shaped gap.
- **S3 (full retrieval query, confound active):** GEM **16/16 (100%)** vs Mem0 **9/16 (56%)** —
  `runs/baseline_mem0.txt`. Mem0 fails even WITH re-derivation available (its retrieval rarely
  co-surfaces the trigger, so the judge has nothing to re-derive from).
- **S2 (selective retrieval, 20 distractors, k=6):** two runs, both informative:
  - *lexical-GEM* (matched co-retrieval 62%/62%, but GEM on the weaker embedder): GEM **88%** vs
    Mem0 **62%** — `runs/baseline_mem0_selective.txt`.
  - *fair-embedder* (GEM + Mem0 both on MiniLM): GEM **100%** vs Mem0 **56%**; co-retrieval **94%
    (GEM) vs 62% (Mem0)** — `runs/baseline_mem0_selective_st.txt`.
- **Zep/Graphiti:** argued-not-measured (see RELATED_WORK.md) — Neo4j setup deferred until asked.

### Honest reading (the predicted shape was WRONG — that's the value)

| | S1 native | S2 selective (fair emb) | S3 full-retr. |
|---|---|---|---|
| GEM  | 100% | 100% | 100% |
| Mem0 |  44% |  56% |  56% |

- **S1 native is the unimpeachable headline** (100% vs 44%): no judge, no retrieval at scoring time
  → no re-derivation confound either way. Mem0 verified to have no staleness field.
- **GEM is NOT retrieval-independent** (S2 caught this). Lexical-GEM scored 88% — its ingest-time
  conflict-check missed the conflict among distractors. Controlled A/B (only the embedder changed,
  lexical→MiniLM) recovered `raise-unknown` 0/2→2/2, **proving the cause was ingest-retrieval, not
  classification** (hypothesis (a), not (b)).
- **Co-retrieval confound surfaced AGAINST us:** fair-embedder GEM co-retrieves 94% vs Mem0 62%
  (different native representations), so GEM's S2 100% is partly re-derivation-aided. The clean
  isolation is S1.

**Refined thesis (replaces the naive "retrieval-independent" claim):** eager invalidation
CONVERTS a per-query dependency (Mem0 re-derives every read) into a per-write one (GEM detects the
conflict once at ingest; the STALE flag is then permanent). Both systems are retrieval-bounded, at
different pipeline stages; GEM's is the one-time-per-write kind, Mem0's the every-query kind — a
better failure profile, not elimination. S2 was worth running precisely because it broke the
prediction and produced this sharper claim.
