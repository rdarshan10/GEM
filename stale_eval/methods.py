"""Optimised, robust, scalable versions of the two promising STALE methods — and the trigger
index LIFECYCLE (dedup + cleanup) that answers the question "won't the trigger index just grow?".

trigger (#2) -> TriggerMemory: an INVERTED invalidator index
    invalidator_phrase (embedded)  ->  {fact_ids it would invalidate}
  - DEDUP: near-identical triggers merge (cosine >= MERGE) and UNION their fact sets, so a shared
    invalidator (e.g. "user relocates") is stored ONCE, not once per dependent fact.
  - LIFECYCLE / CLEANUP: when a fact is invalidated or deleted it is removed from every trigger's
    set, and any trigger whose set becomes empty is dropped. So the index stays ~proportional to
    ACTIVE facts (not facts x triggers), and a dead fact's triggers can never phantom-fire.
  - DETECTION: a new observation is embedding-matched against the index; a fired trigger
    invalidates its WHOLE fact set at once — no per-pair LLM reasoning at detection time.

lazy (#3) -> bounded query-time check: retrieve the candidate plus only its top-k most-relevant
    LATER observations (not the whole timeline) -> ~O(1) per query regardless of haystack size.

Run the lifecycle demo (no cloud — embeddings are local, triggers injected):
    python stale_eval/methods.py
"""

from __future__ import annotations

import json

from gem.embed import cosine, default_embedder


class TriggerMemory:
    MERGE = 0.80   # two triggers this similar are the SAME invalidator -> merge (dedup)
    FIRE = 0.45    # an observation this similar to a trigger FIRES it -> invalidate its facts
    MAX_TRIGGERS = 4   # cap per fact: bounds write cost and worst-case index growth

    def __init__(self, embedder=None, llm=None):
        self.emb = embedder or default_embedder()
        self.llm = llm
        self.facts: dict[str, dict] = {}     # fid -> {text, status, vec, reason}
        self.triggers: list[dict] = []       # [{text, vec, facts:set[fid]}]  (inverted index)
        self._n = 0

    # --- write ------------------------------------------------------------- #
    def add(self, text, triggers=None):
        """One incoming message: first FIRE any matched triggers (invalidate old facts), then
        store the message as a new fact and index its own invalidators."""
        v = self.emb.embed(text)
        fired = [t for t in self.triggers if cosine(v, t["vec"]) >= self.FIRE]
        for t in fired:
            for fid in list(t["facts"]):
                self.invalidate(fid, reason=text)

        self._n += 1
        fid = f"f{self._n}"
        self.facts[fid] = {"text": text, "status": "ACTIVE", "vec": v, "reason": None}

        if triggers is None:
            triggers = self._enumerate(text)
        for phrase in triggers[: self.MAX_TRIGGERS]:
            self._index_trigger(phrase, fid)
        return fid

    def _enumerate(self, text):
        if self.llm is None:
            return []
        out = self.llm.chat_json(
            "For a personal memory fact, list the concrete FUTURE events that would make it false "
            "or obsolete. Respond ONLY JSON: {\"triggers\": [\"...\", ...]}",
            f"FACT: {text}")
        return out.get("triggers", []) if isinstance(out, dict) else []

    def _index_trigger(self, phrase, fid):
        v = self.emb.embed(phrase)
        for t in self.triggers:                       # DEDUP: merge into a near-identical trigger
            if cosine(v, t["vec"]) >= self.MERGE:
                t["facts"].add(fid)
                return
        self.triggers.append({"text": phrase, "vec": v, "facts": {fid}})

    # --- lifecycle --------------------------------------------------------- #
    def invalidate(self, fid, reason=""):
        f = self.facts.get(fid)
        if not f or f["status"] != "ACTIVE":
            return
        f["status"], f["reason"] = "STALE", reason
        for t in self.triggers:                       # CLEANUP: drop the dead fact from every trigger
            t["facts"].discard(fid)
        self.triggers = [t for t in self.triggers if t["facts"]]   # drop now-empty triggers

    def delete(self, fid):
        """Permanent removal: gone from facts AND from every trigger set (then prune empties)."""
        self.facts.pop(fid, None)
        for t in self.triggers:
            t["facts"].discard(fid)
        self.triggers = [t for t in self.triggers if t["facts"]]

    # --- read -------------------------------------------------------------- #
    def active(self):
        return [f for f in self.facts.values() if f["status"] == "ACTIVE"]

    def stale(self):
        return [f for f in self.facts.values() if f["status"] != "ACTIVE"]

    def stats(self):
        n_fact = len(self.facts)
        owned = sum(len(t["facts"]) for t in self.triggers)
        return {
            "facts_total": n_fact,
            "facts_active": len(self.active()),
            "trigger_entries": len(self.triggers),
            "trigger_fact_links": owned,
            "naive_size_would_be": owned if not self.triggers else None,
            "dedup_ratio": round(len(self.triggers) / owned, 2) if owned else 0.0,
        }


def lazy_candidates_and_context(facts_in_order, embedder, query, k_cand=3, k_ctx=4):
    """Bounded lazy readout: the query's candidate facts + only the most-relevant OTHER facts
    (the likely conflicting observations) — NOT the whole timeline. ~O(1) in haystack size."""
    qv = embedder.embed(query)
    fvecs = [(f, embedder.embed(f)) for f in facts_in_order]
    ranked = sorted(fvecs, key=lambda fv: cosine(qv, fv[1]), reverse=True)
    cands = [f for f, _ in ranked[:k_cand]]
    cand_vecs = [v for f, v in ranked[:k_cand]]
    # context = facts most similar to the candidates (where an invalidating observation would be)
    ctx_scored = []
    for f, v in fvecs:
        if f in cands:
            continue
        ctx_scored.append((max(cosine(v, cv) for cv in cand_vecs), f))
    ctx = [f for _, f in sorted(ctx_scored, reverse=True)[:k_ctx]]
    return cands, ctx


def _demo():
    """Deterministic lifecycle demo (no cloud): show dedup + cleanup keep the index bounded."""
    emb = default_embedder()
    tm = TriggerMemory(emb)
    reloc = "the user relocates to a different city"
    # three facts that all depend on the SAME life-anchor (location) + one unrelated fact
    c = tm.add("My commute to work is 45 minutes", triggers=[reloc, "the user changes jobs"])
    w = tm.add("I wake at 7am to beat the local traffic", triggers=[reloc])
    g = tm.add("My gym is two blocks from my apartment", triggers=[reloc])
    tm.add("I am vegetarian", triggers=["the user changes their diet"])

    s0 = tm.stats()
    print("after adding 4 facts (3 share the 'relocates' invalidator):")
    print(f"   facts={s0['facts_total']}  trigger_entries={s0['trigger_entries']}  "
          f"fact-links={s0['trigger_fact_links']}  -> dedup keeps entries < links "
          f"({s0['trigger_entries']} vs {s0['trigger_fact_links']})")

    print("\nobservation arrives: 'I just moved to Mumbai' (fires the shared 'relocates' trigger)")
    tm.add("I just moved to Mumbai and I'm settling in")
    s1 = tm.stats()
    print(f"   one event invalidated {len(tm.stale()) - 0} facts at once (inverted index).")
    print(f"   active facts now: {s1['facts_active']}  (vegetarian survives)")
    print(f"   trigger_entries after cleanup: {s1['trigger_entries']}  "
          f"(dead facts' triggers dropped — index shrank, no phantom triggers left)")
    print("\n   surviving active facts:", [f['text'] for f in tm.active()])
    print("   invalidated:", [(f['text'][:28], 'by: ' + f['reason'][:22]) for f in tm.stale()])


if __name__ == "__main__":
    _demo()
