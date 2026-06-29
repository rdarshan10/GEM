"""Measure the cascade-write cost reduction from caching + escalation.

The cascade's expense is capable-model (gpt-oss:120b) calls: the ingest conflict-scan classifies
the trigger against ~k neighbors (most return UNRELATED) plus one call per cascade hop. Escalation
runs a CHEAP model (llama-3.1-8b on Groq) for that first pass and only confirms the DESTRUCTIVE
decisions on the capable model — so the expensive calls drop to the destructive minority. Caching
memoizes repeated (existing, new) decisions.

We report capable-model calls (the cost metric) AND native accuracy (model-free node-status read)
for each config, so any accuracy cost of escalation is visible, not hidden.

Run:  python -m gem.cost_eval         (needs Ollama gpt-oss:120b + GROQ_API_KEY for the cheap model)
"""

from __future__ import annotations

from .llm import OllamaClient, LLMConfig, GroqClient, GroqConfig
from .engine import GEM, GEMConfig
from .embed import default_embedder
from .store import Status, EdgeType
from .scenarios import SCENARIOS, Scenario

_EMB = default_embedder()


def _run(s: Scenario, capable, cheap, escalate: bool):
    cfg = GEMConfig(cascade_enabled=True, escalate=escalate, cache_decisions=True)
    g = GEM(llm=capable, cheap_llm=cheap, embedder=_EMB, config=cfg)
    nodes = [g.ingest(f, parents=[], check_conflicts=False) for f in s.facts]
    for i, pl in enumerate(s.parents):
        for j in pl:
            g.store.add_edge(nodes[i].id, nodes[j].id, EdgeType.DERIVED_FROM)
    originals = [n.content for n in nodes]
    g.ingest(s.trigger, parents=[])                       # cascade (tracked in g.stats)
    correct = 0
    for i, n in enumerate(nodes):
        cur = g.store.get(n.id)
        inv = cur.status != Status.ACTIVE or cur.confidence < 1.0 or cur.content != originals[i]
        correct += int(inv == s.expect_invalid[i])
    return correct, len(nodes), g.stats


def _config(name, scen, capable, cheap, escalate):
    tot_c = tot_n = cap = chp = hits = 0
    for s in scen:
        c, n, st = _run(s, capable, cheap, escalate)
        tot_c += c; tot_n += n
        cap += st["capable_calls"]; chp += st["cheap_calls"]; hits += st["cache_hits"]
    print(f"  {name:28} capable-calls {cap:>4}   cheap-calls {chp:>4}   cache-hits {hits:>3}"
          f"   accuracy {tot_c}/{tot_n} ({tot_c/tot_n:.0%})", flush=True)
    return cap, tot_c, tot_n


def main() -> int:
    capable = OllamaClient(LLMConfig(model="gpt-oss:120b-cloud"))
    cheap = GroqClient(GroqConfig(model="llama-3.1-8b-instant"))
    scen = SCENARIOS                                        # the 11 hand-built scenarios

    print(f"cascade cost eval — {len(scen)} scenarios; capable=gpt-oss:120b, cheap=llama-3.1-8b\n")
    cap_a, ca, na = _config("capable-only (baseline)", scen, capable, None, escalate=False)
    cap_b, cb, nb = _config("escalate (cheap+confirm)", scen, capable, cheap, escalate=True)

    print("\n" + "=" * 72)
    saved = (1 - cap_b / cap_a) * 100 if cap_a else 0
    print(f"capable-model calls: baseline {cap_a}  ->  escalated {cap_b}   "
          f"({saved:.0f}% fewer expensive calls)")
    print(f"accuracy: baseline {ca}/{na} ({ca/na:.0%})  ->  escalated {cb}/{nb} ({cb/nb:.0%})")
    print("=" * 72)
    print("Escalation shifts the high-volume conflict-scan to the cheap model and pays the")
    print("capable model only to CONFIRM destructive decisions. Watch the accuracy column for")
    print("the tradeoff: cheap-model false-UNRELATED can miss a conflict (no escalation fires).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
