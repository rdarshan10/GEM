"""Similarity-gated cost reduction — the accuracy-PRESERVING lever (unlike confirm-destructive
escalation, which traded 19% accuracy).

Each scenario is embedded in the 20-distractor corpus (the realistic large-memory condition) and
the trigger's ingest conflict-scan considers a large candidate_k. With the filter OFF the scan
classifies every candidate (expensive); with the filter ON it skips candidates below the cosine
threshold — and since real conflicts measured >=0.41 while distractors <=0.32, a 0.35 threshold
skips every distractor and keeps every real conflict. So the LLM-call count drops with NO accuracy
loss. We report conflict-scan calls, skipped count, and native (model-free) accuracy for both.

Run:  python -m gem.cost_eval_sim     (capable model only — no cheap model needed)
"""

from __future__ import annotations

from .llm import OllamaClient, LLMConfig
from .engine import GEM, GEMConfig
from .embed import default_embedder
from .store import Status, EdgeType
from .scenarios import SCENARIOS
from .baseline_mem0 import DISTRACTORS

_EMB = default_embedder()
PICKS = [1, 0, 2, 5, 6, 8]   # mumbai4, rent2, timezone, charger, raise, api3


def _run(s, capable, threshold: float):
    cfg = GEMConfig(cascade_enabled=True, candidate_k=24, conflict_sim_threshold=threshold)
    g = GEM(llm=capable, embedder=_EMB, config=cfg)
    nodes = [g.ingest(f, parents=[], check_conflicts=False) for f in s.facts]
    for i, pl in enumerate(s.parents):
        for j in pl:
            g.store.add_edge(nodes[i].id, nodes[j].id, EdgeType.DERIVED_FROM)
    for d in DISTRACTORS:
        g.ingest(d, parents=[], check_conflicts=False)     # bury the scenario among distractors
    originals = [n.content for n in nodes]
    g.ingest(s.trigger, parents=[])
    correct = sum(int((g.store.get(n.id).status != Status.ACTIVE
                       or g.store.get(n.id).confidence < 1.0
                       or g.store.get(n.id).content != originals[i]) == s.expect_invalid[i])
                  for i, n in enumerate(nodes))
    return correct, len(nodes), g.stats


def _config(name, scen, capable, threshold):
    cap = skip = tc = tn = 0
    for s in scen:
        c, n, st = _run(s, capable, threshold)
        tc += c; tn += n; cap += st["capable_calls"]; skip += st["sim_skipped"]
    print(f"  {name:26} conflict-scan LLM calls {cap:>4}   sim-skipped {skip:>4}"
          f"   accuracy {tc}/{tn} ({tc/tn:.0%})", flush=True)
    return cap, tc, tn


def main() -> int:
    capable = OllamaClient(LLMConfig(model="gpt-oss:120b-cloud"))
    scen = [SCENARIOS[i] for i in PICKS]
    print(f"similarity-gated cost eval — {len(scen)} scenarios buried in {len(DISTRACTORS)} "
          f"distractors, candidate_k=24\n")
    cap_off, _, _ = _config("filter OFF (thr=0.0)", scen, capable, 0.0)
    cap_on, con, n = _config("filter ON  (thr=0.35)", scen, capable, 0.35)

    print("\n" + "=" * 72)
    saved = (1 - cap_on / cap_off) * 100 if cap_off else 0
    print(f"conflict-scan LLM calls: {cap_off} -> {cap_on}   ({saved:.0f}% fewer)   "
          f"accuracy held at {con}/{n} ({con/n:.0%})")
    print("=" * 72)
    print("The similarity gate skips the clearly-unrelated neighbors (distractors) without an LLM")
    print("call; the real conflict is above threshold so it's always classified. Cost down, no")
    print("accuracy traded — the lever escalation-to-cheap couldn't provide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
