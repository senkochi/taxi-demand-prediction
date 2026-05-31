#!/usr/bin/env python3
"""Plot shard distribution from mongosh sh.status output saved to shard_distribution_raw.txt

Usage: python plot_shard_distribution.py [input_file] [output_png]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt

INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("distributed_db/results/shard_distribution_raw.txt")
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("distributed_db/results/shard_distribution.png")

if not INPUT.exists():
    print(f"Input file not found: {INPUT}")
    raise SystemExit(1)

text = None
for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
    try:
        text = INPUT.read_text(encoding=enc)
        break
    except Exception:
        continue
if text is None:
    print(f"Failed to read {INPUT} with common encodings")
    raise SystemExit(1)

# Find owned docs per shard
entries = re.findall(r"shardName: '([^']+)'[\s\S]*?numOwnedDocuments: (\d+)", text)
if not entries:
    # fallback: try to find 'docs: N' under 'Shard shardName'
    entries = re.findall(r"Shard (\S+) at [^\n]+\n\{[\s\S]*?docs: (\d+)", text)

if not entries:
    print("Could not parse shard distribution from input file")
    raise SystemExit(1)

shards = [e[0] for e in entries]
counts = [int(e[1]) for e in entries]

# Also attempt to parse orphaned docs
orphans = dict(re.findall(r"shardName: '([^']+)'[\s\S]*?numOrphanedDocs: (\d+)", text))
orphans = {k: int(v) for k, v in orphans.items()}

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
indices = range(len(shards))
bars = ax.bar(indices, counts, color=['#4C72B0','#55A868','#C44E52'][:len(shards)])
ax.set_xticks(indices)
ax.set_xticklabels(shards)
ax.set_ylabel('Owned Documents')
ax.set_title('Shard Distribution (owned docs)')

# Annotate counts and orphan markers
for i, b in enumerate(bars):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + max(counts)*0.01, f"{counts[i]}", ha='center', va='bottom')
    shard = shards[i]
    if shard in orphans and orphans[shard] > 0:
        ax.text(b.get_x() + b.get_width()/2, -max(counts)*0.03, f"orphaned: {orphans[shard]}", ha='center', va='top', color='gray')

plt.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=150)
print(f"Wrote {OUTPUT}")
