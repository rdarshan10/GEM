"""Unit 0 baseline runner — the go/no-go gate.

Runs the full chain (chunk stream -> extract -> resolve -> answer) over real
single-hop FactConsolidation data and reports SubEM, the metric the benchmark uses.

Examples:
  # quick smoke test: 10 questions on the smallest single-hop context
  python -m unit0.run --model mistral:7b --size 6k --max-questions 10 --verbose

  # full single-hop gate
  python -m unit0.run --model llama3.1:8b --size 6k

Target: approach / beat the ~60% GPT-4o single-hop baseline. If a local model can't
get within reach after prompt iteration, stop and reconsider (per the plan).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from gem.llm import LLMConfig, OllamaClient
from .data import LoadConfig, load_record
from .pipeline import PipelineConfig, Unit0Pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unit 0 FactConsolidation single-hop gate")
    p.add_argument("--model", default="gpt-oss:120b-cloud",
                   help="Ollama model tag (cloud tags run on Ollama's GPUs)")
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                   help="Ollama host (defaults to OLLAMA_HOST env if set)")
    p.add_argument("--size", default="6k", choices=["6k", "32k", "64k", "262k"],
                   help="single-hop context size (row)")
    p.add_argument("--max-questions", type=int, default=None,
                   help="cap questions for fast iteration")
    p.add_argument("--chunk-chars", type=int, default=2_500,
                   help="chars per ingest chunk; smaller = more calls but faster each")
    p.add_argument("--multi-hop", action="store_true",
                   help="use multi-hop rows (NOT the Unit 0 gate; for later units)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-extractions", action="store_true")
    p.add_argument("--save", default=None, help="path to write JSON results")
    p.add_argument("--probe", default=None,
                   help="comma-separated subjects to check in resolved state "
                        "(e.g. 'goaltender,Watsuki,rugby union')")
    p.add_argument("--extract-only", type=int, default=0, metavar="N",
                   help="print extracted triples from the first N chunks, then exit "
                        "(fast prompt iteration; no answering)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    llm = OllamaClient(LLMConfig(model=args.model, host=args.host))
    if not llm.is_up():
        print(f"ERROR: no Ollama server at {args.host}.", file=sys.stderr)
        print("Start it with `ollama serve` and pull a model, e.g. "
              "`ollama pull mistral:7b`.", file=sys.stderr)
        return 2
    models = llm.available_models()
    if args.model not in models and not any(m.startswith(args.model) for m in models):
        print(f"ERROR: model {args.model!r} not pulled. Available: {models}", file=sys.stderr)
        print(f"Pull it with `ollama pull {args.model}`.", file=sys.stderr)
        return 2

    print(f"Loading FactConsolidation "
          f"{'multi' if args.multi_hop else 'single'}-hop size={args.size} ...")
    rec = load_record(LoadConfig(
        size=args.size,
        multi_hop=args.multi_hop,
        chunk_chars=args.chunk_chars,
        max_questions=args.max_questions,
    ))
    print(f"record {rec.record_id}: {len(rec.chunks)} chunks, "
          f"{len(rec.questions)} questions")

    pipe = Unit0Pipeline(llm, PipelineConfig(
        verbose=args.verbose, log_extractions=args.log_extractions))

    # Fast prompt-iteration path: extract from the first N chunks, print triples, exit.
    if args.extract_only:
        for ci, chunk in enumerate(rec.chunks[: args.extract_only]):
            facts = pipe.extract(chunk)
            print(f"\n--- chunk {ci + 1}: {len(facts)} triples ---")
            for f in facts:
                flag = "  <- EMPTY VALUE" if not f["value"].strip() else ""
                print(f"  {f['entity']} | {f['attribute']} = {f['value']!r}{flag}")
        return 0

    t0 = time.time()
    print("ingesting chunks + answering queries (this is the gate) ...")
    result = pipe.run_record(rec)
    elapsed = time.time() - t0

    s = result.score
    print("\n" + "=" * 60)
    print(f"model:   {args.model}")
    print(f"record:  {result.record_id}")
    print(f"SubEM:   {s['subem']:.1%}  ({s['correct']}/{s['total']})")
    print(f"time:    {elapsed:.0f}s")
    print("=" * 60)

    if args.verbose:
        print("\nfirst 10 misses:")
        shown = 0
        for q, pred, gold, hit in zip(
            rec.questions, result.predictions, result.golds, s["hits"]
        ):
            if not hit and shown < 10:
                print(f"  Q: {q}")
                print(f"    gold: {gold}  pred: {pred!r}")
                shown += 1

    # --probe: did extraction actually capture facts about these subjects?
    # The key diagnostic — extraction problem (fact absent) vs answering problem
    # (fact present but the model ignored it).
    if args.probe:
        print("\nstate probe (extraction check):")
        terms = [t.strip().lower() for t in args.probe.split(",") if t.strip()]
        for term in terms:
            hits = [
                v for v in result.state.values()
                if term in f"{v['entity']} {v['attribute']} {v['value']}".lower()
            ]
            if hits:
                for v in hits:
                    print(f"  [{term}] FOUND: {v['entity']} | {v['attribute']}: {v['value']}")
            else:
                print(f"  [{term}] NOT IN STATE  <- extraction lost it")

    if args.save:
        out = {
            "model": args.model,
            "record_id": result.record_id,
            "subem": s["subem"],
            "correct": s["correct"],
            "total": s["total"],
            "elapsed_sec": elapsed,
            "predictions": result.predictions,
            "golds": result.golds,
            "hits": s["hits"],
            "state": [
                {"entity": v["entity"], "attribute": v["attribute"], "value": v["value"]}
                for v in result.state.values()
            ],
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"saved -> {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
