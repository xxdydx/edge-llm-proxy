from __future__ import annotations
import json
from pathlib import Path
import pyarrow.parquet as pq

ND = Path('/Users/arul/Desktop/flowmesh/experiments/new_datasets')
CACHE = ND / '.task-source-cache/hf'
PENDING = ND / 'frozen_train50_v1/pending_preflight.jsonl'
CAT = ND / 'task_catalogue_v1/catalogue.jsonl'
OUT = ND / 'w2_preflight/train50_instances'

def row(meta):
    loc = meta['source_locator']
    ds = 'datasets--SWE-bench--SWE-smith-py' if meta['source_key'] == 'smith' else 'datasets--SWE-Gym--SWE-Gym'
    p = CACHE / ds / 'blobs' / loc['parquet_sha256']
    return pq.read_table(p).slice(loc['row_index_in_file'], 1).to_pylist()[0]

def main():
    pending = [json.loads(x) for x in PENDING.read_text().splitlines() if x.strip()]
    cat = {x['task_uid']: x for x in map(json.loads, CAT.read_text().splitlines())}
    for p in pending:
        m = cat[p['task_uid']]
        d = OUT / p['task_uid'].replace(':', '__')
        d.mkdir(parents=True, exist_ok=True)
        (d / 'catalogue_row.json').write_text(json.dumps(m, indent=2, ensure_ascii=False))
        (d / 'source_row.json').write_text(json.dumps(row(m), indent=2, ensure_ascii=False))
    print(f'wrote {len(pending)} complete source rows')

if __name__ == '__main__': main()
