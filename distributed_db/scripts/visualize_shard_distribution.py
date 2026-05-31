#!/usr/bin/env python3
"""Create a simple visual summary of shard distribution from sh.status output."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("distributed_db/results/shard_distribution_raw.txt")
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("distributed_db/results/shard_distribution_visualization.txt")

text = INPUT.read_text(encoding="utf-8")
match = re.search(r"shardedDataDistribution\s*\n\[(.*)\]\s*\n---", text, re.S)
if not match:
    match = re.search(r"shardedDataDistribution\s*\n\[(.*)\]", text, re.S)
if not match:
    raise SystemExit("Could not locate shardedDataDistribution block in input")

block = "[" + match.group(1).strip() + "]"
# The block is JS-like, not strict JSON; extract shard counts via regex instead of parsing.
shard_entries = re.findall(r"shardName: '([^']+)'[\s\S]*?numOwnedDocuments: (\d+)", block)
if not shard_entries:
    raise SystemExit("Could not parse shard entries")

lines = ["Shard Distribution Visualization", "=" * 32, ""]
max_docs = max(int(count) for _, count in shard_entries)
for shard, count in shard_entries:
    count_int = int(count)
    bar_len = 40 if max_docs == 0 else max(1, round((count_int / max_docs) * 40))
    bar = "█" * bar_len
    lines.append(f"{shard:10} | {bar:<40} | {count_int}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[OK] Wrote {OUTPUT}")
