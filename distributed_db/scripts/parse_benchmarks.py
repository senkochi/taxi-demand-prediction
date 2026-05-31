#!/usr/bin/env python3
from pathlib import Path
import re
import json

files = {
    'centralized': Path('distributed_db/results/benchmark_centralized_200k.txt'),
    'sharded_pre': Path('distributed_db/results/benchmark_sharded_200k.txt'),
    'sharded_post': Path('distributed_db/results/benchmark_sharded_after_split_200k.txt'),
}

def read_text(p: Path):
    for enc in ('utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1'):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f'Cannot read {p}')

res = {}
for k,p in files.items():
    if not p.exists():
        res[k] = {'error': 'missing'}
        continue
    txt = read_text(p)
    def find_int(name):
        m = re.search(rf"{name}=(\d+)", txt)
        return int(m.group(1)) if m else None
    total = find_int('total_docs') or find_int('total_docs_imported')
    demand = find_int('demand_agg_ms')
    od = find_int('od_agg_ms')
    res[k] = {'total_docs': total, 'demand_agg_ms': demand, 'od_agg_ms': od}

print(json.dumps(res, indent=2))
