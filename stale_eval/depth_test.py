"""The decisive DEPTH test: a 6-hop chain where the change hits the ROOT and the query asks about
the LEAF (semantically distant). This is where the methods should DIVERGE cleanly, not within noise.

Prediction (the architecture claim):
  pure_lazy       FAILS — bounded retrieval around the leaf never surfaces the distant root.
  glazy_pinned    WORKS — walks DERIVED_FROM edges leaf->root, finds the eager-detected stale root.
  glazy_inferred  WORKS *iff* derive_links built the chain (tests the real-dialogue bottleneck).
  eager_pinned    WORKS — cascade pre-propagated root->leaf at write.

Scored with STALE's judge rubric on probes about the LEAF.

Run:  python stale_eval/depth_test.py
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gem import Memory
from gem.embed import default_embedder
from gem.llm import make_llm

from gem.store import Status
from adapter import (judge, gem_context, GEM_ANSWER_SYSTEM, GROUNDED_ANSWER_SYSTEM,
                     glazy_answer, lazy_answer, glazy_memory, _retrieve_nodes, _contradictor)

import json

# Strategy 1 (per-hop lazy): re-derive DOWN the queried path hop-by-hop (the decomposition that
# made eager work), but only along the path the query touches.
PERHOP_SYS = ("A fact's PARENT (which it depends on) has CHANGED or been invalidated. Decide if the "
              "CHILD fact is now invalid. Only the specific changed aspect matters. Respond ONLY "
              "JSON: {\"invalidated\": true/false, \"reason\": \"...\"}")

# Strategy 2 (anchor-stamping): each fact stamped at write with its ROOT anchor's value; staleness
# = the anchor moved (O(1) compare, any depth), then ONE confirm call to preserve semantic-stop.
ANCHOR_SYS = ("A memory's dependency ROOT has changed value. Decide if that change invalidates the "
              "dependent memory (it may not — only the specific changed aspect matters). Respond "
              "ONLY JSON: {\"invalidated\": true/false, \"reason\": \"...\"}")

# 6-hop chain: each fact derived from the previous; dates all anchored to the launch.
CHAIN = [
    "The product launch is scheduled for December 1",                         # f0 root
    "The marketing campaign kicks off November 17, two weeks before launch",  # f1
    "The press embargo lifts on November 17 to match the campaign",           # f2
    "Analyst briefings are booked for November 16, the day before embargo",   # f3
    "The demo video must be finalized by November 14 for the briefings",      # f4
    "Video production starts October 20 to finish the demo by November 14",   # f5 leaf
]
DISTRACTORS = ["The team uses trunk-based development",
               "Our office moved to the third floor last quarter",
               "The brand color was updated to teal"]
SLIP = "The launch has slipped from December 1 to March 1"

ITEM = {
    "uid": "depth-6hop",
    "M_old": "Video production starts October 20 (timed six hops back to the December 1 launch).",
    "M_new": "The launch slipped to March 1, so the entire downstream schedule — including video "
             "production's start — is invalid and must be rescheduled.",
    "explanation": "Video production's October 20 start was derived through a 6-hop chain from the "
                   "December 1 launch; the launch slipped to March 1, invalidating the whole chain.",
    "probing_queries": {
        "dim1_query": "Is video production still starting on October 20?",
        "dim2_query": "Since video production starts October 20, can you reserve the studio for October 20?",
        "dim3_query": "When should video production start?",
    },
    # haystack: establish the chain, then distractors, then the slip
    "haystack_session": [[{"role": "user", "content": c}] for c in CHAIN]
                        + [[{"role": "user", "content": d}] for d in DISTRACTORS]
                        + [[{"role": "user", "content": SLIP}]],
    "relevant_session_index": [0, len(CHAIN) + len(DISTRACTORS)],
}


def build_pinned(cascade):
    """Chain with EXPLICIT edges (isolates architecture from derive_links), then the slip arrives."""
    mem = Memory(cascade=cascade)
    ids = []
    for i, f in enumerate(CHAIN):
        ids.append(mem.load(f, derived_from=([ids[i - 1]] if i else [])))
    for d in DISTRACTORS:
        mem.load(d)
    mem.add(SLIP)                     # eager detection marks the root stale (+ cascade if cascade=True)
    return mem


def answer_eager(mem, q, llm):
    return llm.chat(GEM_ANSWER_SYSTEM, f"MEMORY:\n{gem_context(mem, q)}\n\nQUESTION: {q}")


def _path_to_root(g, leaf):
    path, cur, seen = [], leaf, set()
    while cur and cur.id not in seen:
        seen.add(cur.id)
        path.append(cur)
        parents = g.store.derived_from_targets(cur.id)
        cur = parents[0] if parents else None
    return list(reversed(path))                       # root .. leaf


def answer_perhop(mem, q, llm):
    """Walk to the queried leaf, then re-derive DOWN root->leaf hop-by-hop (lazy eager-on-path)."""
    g = mem._g
    cands = _retrieve_nodes(g, q, k=1)
    if not cands:
        return "I don't have that information."
    path = _path_to_root(g, cands[0])
    invalid = {}
    for i, node in enumerate(path):
        if i == 0:                                    # root: changed if updated/contradicted
            contra = _contradictor(g, node)
            invalid[node.id] = (node.status != Status.ACTIVE) or (contra is not None)
            continue
        parent = path[i - 1]
        if not invalid.get(parent.id):
            invalid[node.id] = False
            continue
        v = llm.chat_json(PERHOP_SYS, f"PARENT (changed/invalid): {parent.content}\n"
                                      f"CHILD (depends on it): {node.content}")
        invalid[node.id] = bool(isinstance(v, dict) and v.get("invalidated"))
    leaf = path[-1]
    status = "INVALID (an upstream fact it depends on changed)" if invalid.get(leaf.id) else "current"
    return llm.chat(GROUNDED_ANSWER_SYSTEM,
                    f"FACT: {leaf.content}\nSTATUS: {status}\n\nQUERY: {q}")


def build_anchor():
    """Chain + STAMP each fact with its root anchor's content (write-time graph walk, no LLM)."""
    mem = Memory(cascade=False)
    ids = []
    for i, f in enumerate(CHAIN):
        ids.append(mem.load(f, derived_from=([ids[i - 1]] if i else [])))
    for d in DISTRACTORS:
        mem.load(d)
    g = mem._g
    stamps = {}
    for n in g.store.all_nodes():
        root, seen = n, set()
        while True:
            parents = g.store.derived_from_targets(root.id)
            if not parents or root.id in seen:
                break
            seen.add(root.id)
            root = parents[0]
        stamps[n.id] = (root.id, root.content)        # anchor + value-at-derivation
    mem.add(SLIP)                                      # eager detection updates the root's content
    return mem, stamps


def answer_anchor(mem, stamps, q, llm):
    g = mem._g
    for n in _retrieve_nodes(g, q, k=1):
        root_id, stamped = stamps.get(n.id, (n.id, n.content))
        root_now = g.store.get(root_id)
        moved = root_now is not None and root_now.content != stamped   # O(1) staleness check
        if not moved:
            return llm.chat(GROUNDED_ANSWER_SYSTEM, f"FACT: {n.content}\nSTATUS: current\n\nQUERY: {q}")
        v = llm.chat_json(ANCHOR_SYS, f"DEPENDENCY ROOT was: {stamped}\nROOT is now: {root_now.content}\n"
                                      f"DEPENDENT MEMORY: {n.content}")
        status = "INVALID (its root dependency changed)" if (isinstance(v, dict) and v.get("invalidated")) else "current"
        return llm.chat(GROUNDED_ANSWER_SYSTEM, f"FACT: {n.content}\nSTATUS: {status}\n\nQUERY: {q}")
    return "I don't have that information."


def run():
    llm, emb = make_llm(), default_embedder()
    facts = [t["content"] for s in ITEM["haystack_session"] for t in s if t["role"] == "user"]
    qs = ITEM["probing_queries"]

    anchor_mem, anchor_stamps = build_anchor()
    methods = {
        "eager_pinned": lambda q: answer_eager(build_pinned(True), q, llm),
        "glazy_pinned": lambda q: glazy_answer(build_pinned(False), q, llm),
        "perhop":       lambda q: answer_perhop(build_pinned(False), q, llm),   # strategy 1
        "anchor":       lambda q: answer_anchor(anchor_mem, anchor_stamps, q, llm),  # strategy 2
        "pure_lazy":    lambda q: lazy_answer(facts, emb, q, llm),
    }
    print("6-HOP DEPTH TEST — change hits the ROOT (launch), query asks about the LEAF (production)\n")
    for name, fn in methods.items():
        resp = {dim.replace("_query", "_response"): fn(qs[dim]) for dim in qs}
        res = judge(ITEM, resp, llm)
        tot = sum(res.values())
        print(f"  {name:15} dim1={'P' if res['dim1'] else 'F'} dim2={'P' if res['dim2'] else 'F'} "
              f"dim3={'P' if res['dim3'] else 'F'}   {tot}/3")
    print("\n(pure_lazy should fail — distant root not retrieved; glazy/eager should pass via the graph)")


if __name__ == "__main__":
    run()
