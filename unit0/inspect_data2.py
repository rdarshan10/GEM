"""Second inspection pass: find the single-hop rows and the conflict structure.

Row 0's qa_pair_ids are `factconsolidation_mh_6k_*` (multi-hop). Unit 0 needs the
SINGLE-HOP rows. This prints the id prefix / source / sample question for all 8 rows,
then dumps enough of a single-hop context to see how conflicting facts are laid out.
"""

from __future__ import annotations

import re
import textwrap
from collections import Counter

from datasets import load_dataset

ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution")

print("per-row identity:")
for i in range(len(ds)):
    r = ds[i]
    ids = r["metadata"]["qa_pair_ids"]
    # strip trailing _noN to get the family prefix
    prefixes = Counter(re.sub(r"_no\d+$", "", x) for x in ids)
    prefix = prefixes.most_common(1)[0][0]
    print(f"  row {i}: ctx={len(r['context']):>9} chars  source={r['metadata']['source']!r}  "
          f"prefix={prefix}  nq={len(r['questions'])}")
    print(f"          q0: {textwrap.shorten(r['questions'][0], width=110)}")
    print(f"          a0: {r['answers'][0]!r} (type={type(r['answers'][0]).__name__})")

# Find a single-hop row (prefix containing '_sh_') if present.
sh_row = None
for i in range(len(ds)):
    ids = ds[i]["metadata"]["qa_pair_ids"]
    if any("_sh_" in x for x in ids):
        sh_row = i
        break

target = sh_row if sh_row is not None else 0
label = "SINGLE-HOP" if sh_row is not None else "FALLBACK row0 (no _sh_ found)"
print("\n" + "=" * 70)
print(f"context dump for {label}: row {target}")
ctx = ds[target]["context"]
print(f"context length: {len(ctx)} chars")
print("\n--- first 2500 chars ---")
print(ctx[:2500])
print("\n--- last 1500 chars ---")
print(ctx[-1500:])

# Look for repeated subjects (conflicting updates) in a single-hop-style fact list.
print("\n--- searching for duplicated 'born in the city of' style updates ---")
lines = re.split(r"\s*\d+\.\s", ctx)
print(f"approx fact entries: {len(lines)}")
for ln in lines[:8]:
    print("  •", textwrap.shorten(ln.strip(), width=120))
