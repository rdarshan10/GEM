"""Record real GEM runs for the site's memory explorer -> docs/data/scenarios.json.

Each example is run twice on the same memory: GEM (cascade on, Jev decider) and flat memory (cascade
off). For GEM it records every step in cascade order — the conflict scan, each dependent's Jev
verdict and route, what the LLM decided, and each fact's final state — so the page replays what
actually happened instead of an illustration.

    python docs/build_scenarios.py        (needs TYPESAFE_API_KEY and an LLM; reads .env)
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gem.llm import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from gem import classify as C  # noqa: E402
from gem.decide import TieredJudge  # noqa: E402
from gem.embed import default_embedder  # noqa: E402
from gem.engine import GEM, GEMConfig  # noqa: E402
from gem.llm import make_llm  # noqa: E402
from gem.store import EdgeType, Status  # noqa: E402

SCENARIOS = [
    {
        "key": "relocation", "title": "Moving cities", "domain": "Personal assistant",
        "blurb": "Four facts hang off where you live, one of them four steps away. Two others only "
                 "look related.",
        "facts": ["I live in Bangalore",
                  "My commute to the office is 45 minutes",
                  "My alarm is set for 6:30 am to beat the traffic",
                  "The coffee maker is scheduled to start at 6:45 am",
                  "My timezone is IST",
                  "My rent is 30,000 rupees a month",
                  "I visited Bangalore Palace in 2019",
                  "I prefer aisle seats on flights"],
        "parents": [[], [0], [1], [2], [0], [0], [], []],
        "assoc": [[6, 0]],
        "trigger": "I moved to Mumbai",
    },
    {
        "key": "runtime", "title": "Runtime migration", "domain": "Coding agent project memory",
        "blurb": "A platform decision changes. The limits and workarounds built on it never name "
                 "the platform, so text matching can't find them.",
        "facts": ["The orders service runs on AWS Lambda",
                  "Cold starts add about 800 ms to the first request after idle",
                  "The checkout latency budget is 1.2 seconds to absorb that first-request delay",
                  "Each handler must finish within 15 minutes",
                  "The nightly export is split into 10-minute chunks",
                  "The orders service is written in Go"],
        "parents": [[], [0], [1], [0], [3], []],
        "assoc": [],
        "trigger": "We moved the orders service from AWS Lambda to long-running containers on ECS",
    },
    {
        "key": "launch", "title": "Launch slip", "domain": "Project planning",
        "blurb": "Every date in the plan is relative to the one before it. The launch moves; the "
                 "change has to travel six steps.",
        "facts": ["The product launch date is October 1",
                  "The marketing campaign starts September 15, two weeks before launch",
                  "The press embargo lifts on September 15 to match the campaign",
                  "Analyst briefings are scheduled for September 14, the day before embargo lift",
                  "The demo video must be finalized by September 10 for the briefings",
                  "Video production kicks off August 20 to finish by September 10"],
        "parents": [[], [0], [1], [2], [3], [4]],
        "assoc": [],
        "trigger": "The launch date slipped to December 1",
    },
    {
        "key": "cloud", "title": "Instance change", "domain": "Infrastructure",
        "blurb": "The budget depends on two facts at once, and finance signed off on the budget. "
                 "One parent changes; its sibling facts must stay put.",
        "facts": ["Our servers run in the us-east-1 region",
                  "Our servers use the m5.large instance type",
                  "Our servers boot from image ami-0abc123",
                  "The monthly hosting budget is 400 dollars",
                  "Finance approved the hosting line in this quarter's plan"],
        "parents": [[], [], [], [0, 1], [3]],
        "assoc": [],
        "trigger": "We switched our servers to the m5.xlarge instance type",
    },
    {
        "key": "reorg", "title": "Manager reorg", "domain": "Workplace assistant",
        "blurb": "Three routines fan out from who your manager is. Each one needs its own answer.",
        "facts": ["Alice is my manager",
                  "I send Alice a weekly status report",
                  "Alice approves my time-off requests",
                  "Alice writes my annual performance review"],
        "parents": [[], [0], [0], [0]],
        "assoc": [],
        "trigger": "After the reorg, Bob is now my manager",
    },
    {
        "key": "email", "title": "Email switch", "domain": "Precision check",
        "blurb": "Two small dependency trees sit side by side. The change touches one; the other "
                 "must come through untouched.",
        "facts": ["My personal email provider is Gmail",
                  "Gmail filters forward my receipts to my expense tracker",
                  "My password manager is 1Password",
                  "I store my work 2FA codes in 1Password"],
        "parents": [[], [0], [], [2]],
        "assoc": [],
        "trigger": "I switched my personal email provider to Fastmail",
    },
]

_REVISE = re.compile(r"^(\s*)revise (n\d+): (\w+) .*?(?:\[(\w+)\])?$")
_STOP_JEV = re.compile(r"^(\s*)semantic stop \(jev\): (n\d+)")
_STOP_LLM = re.compile(r"^(\s*)semantic stop: (n\d+)")
_SKIP = re.compile(r"ingest conflict: (n\d+) already revised")


def _build(s, llm, emb, cfg):
    g = GEM(llm=llm, embedder=emb, config=cfg)
    ids = [g.ingest(f, parents=[], check_conflicts=False).id for f in s["facts"]]
    for i, ps in enumerate(s["parents"]):
        for j in ps:
            g.store.add_edge(ids[i], ids[j], EdgeType.DERIVED_FROM)
    for i, j in s["assoc"]:
        g.store.add_edge(ids[i], ids[j], EdgeType.ASSOCIATED)
    return g, ids


def _final(g, ids):
    out = []
    for i in ids:
        n = g.store.get(i)
        out.append({"status": n.status.value, "content": n.content,
                    "review": bool(n.meta.get("needs_review")), "confidence": round(n.confidence, 2)})
    return out


def run_gem(s, llm, emb):
    cfg = GEMConfig(decider="jev")
    g, ids = _build(s, llm, emb, cfg)
    judge = TieredJudge(cfg)
    by_text = {s["facts"][k]: ids[k] for k in range(len(ids))}
    verdicts = {}                                  # node id -> what the decider said
    real_ask = g._ask_decider

    def ask(method, subject, items):
        out = real_ask(method, subject, items)
        for text, v in zip(items, out):
            nid = by_text.get(text)
            if nid is None or v is None:
                continue
            if method == "conflicts":
                top = max(v.probs, key=v.probs.get)
                verdicts.setdefault(nid, {})["scan"] = {
                    "route": judge.conflict(v), "top": top.value, "p_top": round(v.probs[top], 2),
                    "p_no_conflict": round(1 - sum(p for l, p in v.probs.items()
                                                   if l in C.INVALIDATING), 2),
                    "p_covered": round(v.p_covered, 2)}
            elif method == "impact":
                verdicts.setdefault(nid, {})["impact"] = {
                    "route": judge.impact(v), "p_unaffected": round(v.probs.get(0, 0.0), 2),
                    "p_affected": round(1 - v.probs.get(0, 0.0), 2), "p_known": round(v.p_known, 2)}
        return out
    g._ask_decider = ask
    g.trace.clear()
    g.ingest(s["trigger"], parents=[])

    steps = []
    scanned = [nid for nid, v in verdicts.items() if "scan" in v]
    steps.append({"kind": "scan", "checked": [
        {"id": nid, **verdicts[nid]["scan"]} for nid in scanned]})
    for line in g.trace:
        if m := _REVISE.match(line):
            nid, label, status = m.group(2), m.group(3), m.group(4) or "ACTIVE"
            depth = len(m.group(1)) // 2
            v = verdicts.get(nid, {})
            via = "jev" if (depth == 0 and v.get("scan", {}).get("route") == "covered") or \
                (depth > 0 and v.get("impact", {}).get("route") == "stale") else "llm"
            steps.append({"kind": "revise", "id": nid, "depth": depth, "label": label,
                          "status": status, "via": via, "jev": v.get("impact") or v.get("scan"),
                          "content": g.store.get(nid).content})
        elif m := _STOP_JEV.match(line) or _STOP_LLM.match(line):
            nid = m.group(2)
            steps.append({"kind": "stop", "id": nid, "depth": len(m.group(1)) // 2,
                          "via": "jev" if "(jev)" in line else "llm",
                          "jev": verdicts.get(nid, {}).get("impact")})
        elif m := _SKIP.search(line):
            steps.append({"kind": "already", "id": m.group(1)})
    return ids, steps, _final(g, ids), {"llm": g.stats["capable_calls"],
                                        "jev": g.stats["jev_calls"]}, list(g.trace)


def run_flat(s, llm, emb):
    g, ids = _build(s, llm, emb, GEMConfig(decider="llm", cascade_enabled=False))
    g.ingest(s["trigger"], parents=[])
    return _final(g, ids)


def main() -> int:
    llm, emb = make_llm(), default_embedder()
    model = getattr(getattr(llm, "cfg", None), "model", "?")
    out = {"generated": date.today().isoformat(), "llm": model, "decider": "jev", "scenarios": []}
    for s in SCENARIOS:
        C.reset_degraded()
        ids, steps, gem_final, calls, trace = run_gem(s, llm, emb)
        flat_final = run_flat(s, llm, emb)
        if C.degraded_total():
            print(f"!! {s['key']}: {C.degraded_total()} degraded LLM calls — rerun", file=sys.stderr)
            return 1
        out["scenarios"].append({
            **{k: s[k] for k in ("key", "title", "domain", "blurb", "trigger")},
            "nodes": [{"id": ids[i], "text": s["facts"][i], "parents": [ids[j] for j in s["parents"][i]]}
                      for i in range(len(ids))],
            "assoc": [[ids[i], ids[j]] for i, j in s["assoc"]],
            "steps": steps, "gem": gem_final, "flat": flat_final, "calls": calls, "trace": trace,
        })
        print(f"{s['key']:11} llm {calls['llm']}  jev {calls['jev']}  "
              f"gem={[f['status'][0] for f in gem_final]}  flat={[f['status'][0] for f in flat_final]}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scenarios.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
