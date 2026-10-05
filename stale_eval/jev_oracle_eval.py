"""Jev's own error rate on eval_diverse, with the LLM replaced by a ground-truth oracle.

Real Jev calls make every cheap decision; wherever the engine would call the LLM, an oracle answers
from the scenario's expected labels instead. The LLM path scored 37/37 on this eval, so any miss here
is Jev's (a Jev-only stop/skip that was wrong), and the oracle call count is the LLM work that remains.
Needs TYPESAFE_API_KEY (read from the repo-root .env); no LLM backend required.

    python stale_eval/jev_oracle_eval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jev_smoke import load_dotenv  # noqa: E402

from gem import classify as C  # noqa: E402
from gem.embed import default_embedder  # noqa: E402
from gem.engine import GEM, GEMConfig  # noqa: E402
from gem.eval_diverse import DIVERSE  # noqa: E402
from gem.store import EdgeType, Status  # noqa: E402


class OracleLLM:
    """Answers classify from ground truth: expected-invalid facts -> UPDATES (value unknown)."""

    def __init__(self, truth: dict[str, bool]):
        self.truth = truth
        self.calls = 0

    def chat_json(self, system, user, **kw):
        self.calls += 1
        if "derived_from" in system:
            return {"derived_from": []}
        existing = next((l.split(":", 1)[1].strip() for l in user.splitlines()
                         if l.startswith("EXISTING memory:")), "")
        hit = next((v for k, v in self.truth.items() if k in existing or existing in k), False)
        return {"label": "UPDATES" if hit else "UNRELATED", "revised_content": None}


def run(s, emb, decider: str):
    llm = OracleLLM(dict(zip(s.facts, s.expect_invalid)))
    g = GEM(llm=llm, embedder=emb, config=GEMConfig(decider=decider))
    nodes = [g.ingest(f, parents=[], check_conflicts=False) for f in s.facts]
    for i, pl in enumerate(s.parents):
        for j in pl:
            g.store.add_edge(nodes[i].id, nodes[j].id, EdgeType.DERIVED_FROM)
    originals = [n.content for n in nodes]
    g.trace.clear()
    g.ingest(s.trigger, parents=[])
    misses = []
    for i, n in enumerate(nodes):
        cur = g.store.get(n.id)
        got = cur.status != Status.ACTIVE or cur.confidence < 1.0 or cur.content != originals[i]
        if got != s.expect_invalid[i]:
            misses.append(("MISSED invalidation" if s.expect_invalid[i] else "FALSE invalidation",
                           s.facts[i], cur.status.value))
    return len(nodes) - len(misses), len(nodes), llm.calls, g.stats, misses, g.trace


def main() -> int:
    load_dotenv()
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set (add it to .env).")
        return 2
    emb = default_embedder()
    C.reset_degraded()
    totals = {"llm": [0, 0, 0], "jev": [0, 0, 0, 0, 0]}
    for s in DIVERSE:
        ok_b, n_b, calls_b, _, _, _ = run(s, emb, "llm")
        ok, n, calls, st, misses, trace = run(s, emb, "jev")
        totals["llm"][0] += ok_b; totals["llm"][1] += n_b; totals["llm"][2] += calls_b
        t = totals["jev"]
        t[0] += ok; t[1] += n; t[2] += calls; t[3] += st["jev_calls"]; t[4] += st["jev_decided"]
        flag = "PASS" if not misses else "FAIL"
        print(f"[{flag}] {s.name:42} {ok}/{n}   LLM calls {calls_b:>2} -> {calls:>2}   "
              f"Jev calls {st['jev_calls']:>2}  decided alone {st['jev_decided']:>2}")
        for kind, fact, status in misses:
            print(f"        {kind}: {fact!r} [{status}]")
        if misses:
            for line in trace:
                print("          " + line)
    b, j = totals["llm"], totals["jev"]
    print("\n" + "=" * 78)
    print(f"accuracy   llm-path (oracle) {b[0]}/{b[1]}    jev-path {j[0]}/{j[1]}")
    saved = (1 - j[2] / b[2]) * 100 if b[2] else 0
    print(f"LLM calls  {b[2]} -> {j[2]}  ({saved:.0f}% fewer)   Jev calls {j[3]}   "
          f"decisions made by Jev alone {j[4]}   Jev fallbacks {C.DEGRADED['jev']}")
    print("=" * 78)
    return 0 if j[0] == j[1] else 1


if __name__ == "__main__":
    sys.exit(main())
