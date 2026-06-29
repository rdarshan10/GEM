"""Closed-book ablation: does the model already KNOW the answers, or is it reading our
resolved state?

The multi-hop FactConsolidation golds are MQuAKE counterfactual edits — deliberately false
in the real world (e.g. "Blair Walsh plays rugby"). If the model answers from PRETRAINING it
will return the real-world values and MISS the edited golds. So we ask the same questions with
NO facts supplied and SubEM-score: a LOW closed-book score proves the open-book 77% comes from
the maintained state, not memorized knowledge. The gap (open - closed) is the value the memory
system actually adds.

Run:  python -m unit0.closedbook --multi-hop --size 6k
"""

from __future__ import annotations

import argparse

from gem.llm import OllamaClient, LLMConfig
from .data import LoadConfig, load_record
from . import scorer

CLOSED_SYSTEM = (
    "Answer the question from your own knowledge with just the specific entity or value "
    "requested — a name, place, or single word. No explanation."
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Closed-book (no-context) ablation")
    ap.add_argument("--model", default="gpt-oss:120b-cloud")
    ap.add_argument("--size", default="6k", choices=["6k", "32k", "64k", "262k"])
    ap.add_argument("--multi-hop", action="store_true")
    ap.add_argument("--max-questions", type=int, default=None)
    args = ap.parse_args(argv)

    rec = load_record(LoadConfig(size=args.size, multi_hop=args.multi_hop,
                                 max_questions=args.max_questions))
    llm = OllamaClient(LLMConfig(model=args.model))
    print(f"closed-book ablation on {rec.record_id}: {len(rec.questions)} questions, "
          f"model={args.model}\n")

    preds = []
    for i, q in enumerate(rec.questions):
        try:
            ans = llm.chat(CLOSED_SYSTEM, f"Question: {q}\nAnswer:")
        except Exception:
            ans = ""
        preds.append(ans)
        if (i + 1) % 20 == 0:
            print(f"  answered {i + 1}/{len(rec.questions)}", flush=True)

    s = scorer.score(preds, list(rec.answers))
    print("\n" + "=" * 56)
    print(f"CLOSED-BOOK SubEM: {s['subem']:.1%} ({s['correct']}/{s['total']})")
    print("  (open-book multi-hop was 77% — the gap is what the memory state adds)")
    print("=" * 56)
    # show a few where closed-book gave the real-world (wrong) answer
    print("\nsample closed-book answers vs edited gold:")
    shown = 0
    for q, p, g, hit in zip(rec.questions, preds, rec.answers, s["hits"]):
        if not hit and shown < 8:
            print(f"  gold(edit)={g}  closed-book={p[:50]!r}")
            shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
