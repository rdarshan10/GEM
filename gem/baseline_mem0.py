"""Mem0 baseline on the propagation eval — argued -> measured (per BASELINE_PROTOCOL.md).

Same judge LLM (gpt-oss:120b) for BOTH systems, each system's own retrieval, identical per-fact
validity scoring. For each scenario: ingest the facts, ingest the trigger, then for every fact
ask the shared judge — given THIS system's retrieved memories — whether the fact is still VALID
or has CHANGED. Score against ground truth (expect_invalid). This isolates the architecture:
GEM's cascade pre-marks dependents stale; Mem0 (ADD-leaning) keeps them as current memories.

Run:  python -m gem.baseline_mem0 [--limit N]
Needs Ollama (gpt-oss:120b-cloud) reachable; Mem0 configured to use the SAME model.
"""

from __future__ import annotations

import argparse

from .llm import OllamaClient, LLMConfig
from .engine import GEM, GEMConfig
from .embed import cosine
from .store import Status
from .scenarios import SCENARIOS, Scenario

JUDGE = OllamaClient(LLMConfig(model="gpt-oss:120b-cloud"))   # shared judge for ALL systems

# GEM uses the SAME semantic embedder family as Mem0 (MiniLM/384d) — fair retrieval on both
# sides. (The lexical fallback unfairly handicapped GEM's ingest conflict-detection.)
from .embed import STEmbedder as _ST
_GEM_EMB = _ST()

JUDGE_SYS = (
    "You judge whether a statement is STILL CURRENTLY TRUE given the user's memory. Answer "
    "exactly one word: CHANGED if a later memory supersedes / updates / makes it stale or "
    "uncertain, or VALID if it still holds unchanged."
)


def _judge(retrieved_block: str, fact: str) -> bool:
    """Returns True if the judge says the fact CHANGED (is no longer valid as stated)."""
    ans = JUDGE.chat(JUDGE_SYS, f"USER MEMORY:\n{retrieved_block}\n\nStatement: '{fact}'\n"
                                f"Is it VALID or CHANGED?")
    return "CHANGED" in ans.upper()


# --------------------------------------------------------------------------- #
# GEM system (cascade ON) — pre-marks dependents stale
# --------------------------------------------------------------------------- #
def _gem_run(s: Scenario) -> list[bool]:
    g = GEM(llm=JUDGE, embedder=_GEM_EMB, config=GEMConfig(cascade_enabled=True))
    nodes = [g.ingest(f, parents=[], check_conflicts=False) for f in s.facts]
    from .store import EdgeType
    for i, pl in enumerate(s.parents):
        for j in pl:
            g.store.add_edge(nodes[i].id, nodes[j].id, EdgeType.DERIVED_FROM)
    g.ingest(s.trigger, parents=[])      # cascade fires

    def retrieve(fact):
        q = g.embedder.embed(fact)
        scored = sorted(((n, cosine(q, n.embedding)) for n in g.store.all_nodes()
                         if n.embedding is not None), key=lambda t: t[1], reverse=True)[:6]
        out = []
        for n, _ in scored:
            tag = f" [{n.status.value}]" if n.status != Status.ACTIVE else (
                " [NEEDS REVIEW]" if n.meta.get("needs_review") else "")
            out.append(f"- {n.content}{tag}")
        return "\n".join(out)

    return [_judge(retrieve(s.facts[i]), s.facts[i]) for i in range(len(s.facts))]


# --------------------------------------------------------------------------- #
# Mem0 system — configured per its docs (same LLM + a local embedder)
# --------------------------------------------------------------------------- #
def _mem0_factory():
    from mem0 import Memory
    cfg = {
        "llm": {"provider": "ollama", "config": {
            "model": "gpt-oss:120b-cloud", "ollama_base_url": "http://127.0.0.1:11435",
            "temperature": 0}},
        "embedder": {"provider": "huggingface", "config": {
            "model": "all-MiniLM-L6-v2", "embedding_dims": 384}},
        "vector_store": {"provider": "qdrant", "config": {
            "embedding_model_dims": 384, "on_disk": False, "collection_name": "gem_base"}},
    }
    return Memory.from_config(cfg)


def _mem0_run(s: Scenario) -> list[bool]:
    m = _mem0_factory()
    uid = "u"
    for f in s.facts:
        m.add(f, user_id=uid)
    m.add(s.trigger, user_id=uid)

    def retrieve(fact):
        res = m.search(fact, filters={"user_id": uid}, limit=6)
        mems = res.get("results", []) if isinstance(res, dict) else res
        return "\n".join(f"- {x.get('memory')}" for x in mems)

    return [_judge(retrieve(s.facts[i]), s.facts[i]) for i in range(len(s.facts))]


# S1 native value-probes: the distinctive value string of each fact (per scenario name). Used
# for a MODEL-FREE check of whether Mem0's persisted state still asserts the stale value. GEM's
# native check reads node.status (also model-free). No LLM, no embedding at S1 scoring time.
PROBES = {
    "mumbai-commute (4-hop)": ["Bangalore", "45", "7am", "6:45"],
    "rent-follows-city (2-hop)": ["Berlin", "1200"],
    "timezone-survives-same-zone (hard negative)": ["Bangalore", "IST"],
    "charger-survives-move (divergent hard negative)": ["Seattle", "Tesla", "Wall Connector"],
    "raise-unknown-amount (unknown-value)": ["80k", "2000"],
    "api-region-sla (work, 3-hop positive)": ["us-east-1", "20ms", "p99"],
}


def _gem_native(s: Scenario) -> list[bool]:
    """S1 for GEM: pure field read of node state after the cascade. Returns per-fact 'invalidated'."""
    g = GEM(llm=JUDGE, embedder=_GEM_EMB, config=GEMConfig(cascade_enabled=True))
    nodes = [g.ingest(f, parents=[], check_conflicts=False) for f in s.facts]
    from .store import EdgeType
    for i, pl in enumerate(s.parents):
        for j in pl:
            g.store.add_edge(nodes[i].id, nodes[j].id, EdgeType.DERIVED_FROM)
    originals = [n.content for n in nodes]
    g.ingest(s.trigger, parents=[])
    out = []
    for i, n in enumerate(nodes):
        cur = g.store.get(n.id)
        out.append(cur.status != Status.ACTIVE or cur.confidence < 1.0
                   or cur.content != originals[i])     # field read only
    return out


def _mem0_native(m, uid: str, s: Scenario, probes: list[str]) -> list[bool]:
    """S1 for Mem0: model-free string read on a SHARED instance (distinct user_id per scenario
    avoids the Qdrant storage lock). A fact is natively invalidated iff its value probe no
    longer appears in Mem0's persisted memories (Mem0 has no staleness field)."""
    for f in s.facts:
        m.add(f, user_id=uid)
    m.add(s.trigger, user_id=uid)
    g = m.get_all(filters={"user_id": uid})
    mems = g["results"] if isinstance(g, dict) else g
    blob = " ".join(str(x.get("memory", "")) for x in mems).lower()
    # invalidated iff the original value probe is GONE from current memories (no LLM/embedding)
    return [probe.lower() not in blob for probe in probes]


def run_native(scen) -> None:
    print("S1 NATIVE invalidation (model-free state read; no LLM, no embedding)\n")
    m = _mem0_factory()                  # ONE shared instance; user_id isolates scenarios
    m.add("My favorite color is blue", user_id="v")
    _g = m.get_all(filters={"user_id": "v"}); _mems = _g["results"] if isinstance(_g, dict) else _g
    keys = sorted(_mems[0].keys()) if _mems else []
    print(f"  [verified] a Mem0 memory's fields = {keys}  (no 'invalid'/'stale'/'valid_at' field)\n")

    gc = gt = mc = mt = 0
    for k, s in enumerate(scen):
        exp = s.expect_invalid
        gn = _gem_native(s)
        mn = _mem0_native(m, f"s{k}", s, PROBES[s.name])
        g_ok = sum(int(a == e) for a, e in zip(gn, exp))
        m_ok = sum(int(a == e) for a, e in zip(mn, exp))
        gc += g_ok; gt += len(exp); mc += m_ok; mt += len(exp)
        print(f"  {s.name:42} GEM {g_ok}/{len(exp)}   Mem0 {m_ok}/{len(exp)}")
    print("\n" + "=" * 60)
    print(f"GEM  native: {gc}/{gt} ({gc/gt:.0%})    Mem0 native: {mc}/{mt} ({mc/mt:.0%})")
    print("=" * 60)
    print("Mem0 correctly keeps should-SURVIVE facts but cannot mark any should-INVALIDATE")
    print("dependent stale (no mechanism) -> its native score is the survive-fraction floor.")


# S2 distractor corpus — fixed, generic, unrelated to any scenario; set without reference to
# where any trigger lands. Loaded via each system's own bulk path (GEM ingest / Mem0 add infer=False).
DISTRACTORS = [
    "My favorite color is teal", "I have a younger sister named Maya", "I am allergic to peanuts",
    "My gym membership is at FitLife", "I play the guitar on weekends", "My favorite movie is Inception",
    "I drink my coffee black", "My birthday is in March", "I support Arsenal football club",
    "My laptop is a ThinkPad", "I speak conversational Spanish", "I prefer window seats on flights",
    "My houseplant is a monstera", "I jog three times a week", "My favorite cuisine is Thai",
    "I collect vinyl records", "I volunteer at the animal shelter", "My favorite season is autumn",
    "I take vitamin D in winter", "My bookshelf is organized by color",
]
# distinctive NEW-value token the trigger introduces (for the co-retrieval diagnostic)
TRIGGER_TOKEN = {
    "mumbai-commute (4-hop)": "Mumbai", "rent-follows-city (2-hop)": "Munich",
    "timezone-survives-same-zone (hard negative)": "Mumbai",
    "charger-survives-move (divergent hard negative)": "Portland",
    "raise-unknown-amount (unknown-value)": "raise",
    "api-region-sla (work, 3-hop positive)": "ap-south-1",
}


def _gem_selective(s: Scenario):
    g = GEM(llm=JUDGE, embedder=_GEM_EMB, config=GEMConfig(cascade_enabled=True))
    nodes = [g.ingest(f, parents=[], check_conflicts=False) for f in s.facts]
    from .store import EdgeType
    for i, pl in enumerate(s.parents):
        for j in pl:
            g.store.add_edge(nodes[i].id, nodes[j].id, EdgeType.DERIVED_FROM)
    for d in DISTRACTORS:
        g.ingest(d, parents=[], check_conflicts=False)      # GEM's standard bulk-load path
    g.ingest(s.trigger, parents=[])                          # cascade fires

    def retrieve(fact):
        q = g.embedder.embed(fact)
        scored = sorted(((n, cosine(q, n.embedding)) for n in g.store.all_nodes()
                         if n.embedding is not None), key=lambda t: t[1], reverse=True)[:6]
        out = []
        for n, _ in scored:
            tag = f" [{n.status.value}]" if n.status != Status.ACTIVE else (
                " [NEEDS REVIEW]" if n.meta.get("needs_review") else "")
            out.append(f"- {n.content}{tag}")
        return "\n".join(out)

    res, co = [], 0
    tok = TRIGGER_TOKEN[s.name].lower()
    for fact in s.facts:
        block = retrieve(fact)
        res.append(_judge(block, fact))
        co += int(tok in block.lower())
    return res, co, len(s.facts)


def _mem0_selective(m, uid: str, s: Scenario):
    for f in s.facts:
        m.add(f, user_id=uid)                               # normal (infer=True)
    for d in DISTRACTORS:
        m.add(d, user_id=uid, infer=False)                  # Mem0's bulk path, no LLM cost
    m.add(s.trigger, user_id=uid)
    res, co = [], 0
    tok = TRIGGER_TOKEN[s.name].lower()
    for fact in s.facts:
        r = m.search(fact, filters={"user_id": uid}, limit=6)
        mems = r.get("results", []) if isinstance(r, dict) else r
        block = "\n".join(str(x.get("memory", "")) for x in mems)
        res.append(_judge(block, fact))
        co += int(tok in block.lower())
    return res, co, len(s.facts)


def run_selective(scen) -> None:
    print(f"S2 SELECTIVE retrieval ({len(DISTRACTORS)} distractors, k=6, each system's bulk path)\n")
    m = _mem0_factory()
    gc = gt = mc = mt = g_co = g_q = m_co = m_q = 0
    for k, s in enumerate(scen):
        exp = s.expect_invalid
        gres, gco, gq = _gem_selective(s)
        mres, mco, mq = _mem0_selective(m, f"sel{k}", s)
        g_ok = sum(int(a == e) for a, e in zip(gres, exp))
        m_ok = sum(int(a == e) for a, e in zip(mres, exp))
        gc += g_ok; gt += len(exp); mc += m_ok; mt += len(exp)
        g_co += gco; g_q += gq; m_co += mco; m_q += mq
        print(f"  {s.name:42} GEM {g_ok}/{len(exp)}   Mem0 {m_ok}/{len(exp)}")
    print("\n" + "=" * 60)
    print(f"GEM  S2: {gc}/{gt} ({gc/gt:.0%})   trigger co-retrieval rate {g_co}/{g_q} ({g_co/g_q:.0%})")
    print(f"Mem0 S2: {mc}/{mt} ({mc/mt:.0%})   trigger co-retrieval rate {m_co}/{m_q} ({m_co/m_q:.0%})")
    print("=" * 60)
    print("Lower co-retrieval = less chance to re-derive. GEM's eager STALE flag is retrieval-")
    print("independent (flat); Mem0 depends on co-retrieving the trigger (declines as it drops).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mem0 baseline on the propagation eval")
    ap.add_argument("--limit", type=int, default=6, help="number of scenarios (token budget)")
    ap.add_argument("--native", action="store_true", help="S1 native scoring (model-free)")
    ap.add_argument("--selective", action="store_true", help="S2 selective-retrieval scoring")
    args = ap.parse_args(argv)

    # representative subset: positives of varying depth + negatives
    picks = [1, 0, 2, 5, 6, 8][: args.limit]   # mumbai4hop, rent2hop, timezone, charger, raise, api3hop
    scen = [SCENARIOS[i] for i in picks]

    if args.native:
        run_native(scen)
        return 0
    if args.selective:
        run_selective(scen)
        return 0

    gem_correct = gem_total = m0_correct = m0_total = 0
    print(f"Mem0 baseline vs GEM — same judge (gpt-oss:120b), {len(scen)} scenarios\n")
    for s in scen:
        exp = s.expect_invalid
        gem = _gem_run(s)
        m0 = _mem0_run(s)
        gc = sum(int(g == e) for g, e in zip(gem, exp))
        mc = sum(int(x == e) for x, e in zip(m0, exp))
        gem_correct += gc; gem_total += len(exp)
        m0_correct += mc; m0_total += len(exp)
        print(f"  {s.name:42} GEM {gc}/{len(exp)}   Mem0 {mc}/{len(exp)}")

    print("\n" + "=" * 60)
    print(f"GEM  node-accuracy: {gem_correct}/{gem_total} ({gem_correct/gem_total:.0%})")
    print(f"Mem0 node-accuracy: {m0_correct}/{m0_total} ({m0_correct/m0_total:.0%})")
    print("=" * 60)
    print("Scoring: query-based (each system's own retrieval) + shared judge. Per protocol,"
          " this is the end-to-end measure that ALLOWS Mem0 fair re-derivation when its"
          " retrieval co-surfaces the trigger; GEM's advantage is pre-marked staleness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
