"""One-off: inspect the real Conflict_Resolution rows.

We do NOT yet know how the dataset encodes the single-hop vs multi-hop distinction,
how the long `context` is chunked, or where the conflicting facts live. This script
prints enough of the real structure to write the loader against fact, not guess.

Run:  python -m unit0.inspect_data
"""

from __future__ import annotations

import json
import textwrap

from datasets import load_dataset

REPO = "ai-hyz/MemoryAgentBench"
SPLIT = "Conflict_Resolution"


def main():
    print(f"Loading {REPO} split={SPLIT} ...")
    ds = load_dataset(REPO, split=SPLIT)
    print(f"\nrows: {len(ds)}")
    print(f"columns: {ds.column_names}\n")

    row = ds[0]
    for col in ds.column_names:
        val = row[col]
        if isinstance(val, str):
            print(f"--- {col}: str, len={len(val)} ---")
            print(textwrap.shorten(val.replace("\n", " "), width=600))
        elif isinstance(val, list):
            print(f"--- {col}: list, len={len(val)} ---")
            for i, item in enumerate(val[:4]):
                s = item if isinstance(item, str) else json.dumps(item)[:300]
                print(f"  [{i}] {textwrap.shorten(str(s), width=300)}")
        elif isinstance(val, dict):
            print(f"--- {col}: dict, keys={list(val.keys())} ---")
            for k, v in val.items():
                desc = (
                    f"list(len={len(v)})" if isinstance(v, list)
                    else f"str(len={len(v)})" if isinstance(v, str)
                    else type(v).__name__
                )
                print(f"  {k}: {desc}")
                # peek into nested structures that likely hold SH/MH labels
                if isinstance(v, list) and v and k in ("qa_pair_ids", "keypoints", "demo"):
                    print(f"      sample: {json.dumps(v[:3])[:400]}")
                elif isinstance(v, str) and k in ("demo",):
                    print(f"      sample: {textwrap.shorten(v, width=400)}")
        else:
            print(f"--- {col}: {type(val).__name__} = {val} ---")
        print()

    # How many questions per row across the split, and a few sample Q/A pairs.
    print("=" * 70)
    print("questions/answers per row:")
    for i in range(len(ds)):
        r = ds[i]
        nq = len(r["questions"]) if "questions" in r else "?"
        na = len(r["answers"]) if "answers" in r else "?"
        ctx = len(r["context"]) if "context" in r else "?"
        print(f"  row {i}: questions={nq} answers={na} context_chars={ctx}")

    print("\nfirst 5 Q/A of row 0:")
    for q, a in list(zip(row["questions"], row["answers"]))[:5]:
        print(f"  Q: {textwrap.shorten(q, width=160)}")
        print(f"  A: {a}\n")


if __name__ == "__main__":
    main()
