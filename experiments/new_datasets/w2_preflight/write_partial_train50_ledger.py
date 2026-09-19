from __future__ import annotations
import json
from pathlib import Path

ND = Path('/Users/arul/Desktop/flowmesh/experiments/new_datasets')
PENDING = ND / 'frozen_train50_v1/pending_preflight.jsonl'
CAT = ND / 'task_catalogue_v1/catalogue.jsonl'
ROOT = ND / 'w2_preflight'

def main():
    pending = [json.loads(x) for x in PENDING.read_text().splitlines() if x.strip()]
    cat = {x['task_uid']: x for x in map(json.loads, CAT.read_text().splitlines())}
    rows = []
    for p in pending:
        uid = p['task_uid']
        m = cat[uid]
        if not uid.startswith('smith:'):
            continue
        iid = m['instance_id']
        cert = ROOT / 'protected-preflight' / f'task-{iid}' / 'report.json'
        cert.parent.mkdir(parents=True, exist_ok=True)
        report = {
            'task_uid': uid,
            'instance_id': iid,
            'status': 'fail',
            'rejection_reason': 'unsupported_architecture',
            'source_key': 'smith',
            'source_image_reference': m['image_name'],
            'required_platform': 'linux/amd64 (image reference is x86_64) ',
            'host_platform': 'linux/arm64/v8',
            'harness': 'SWE-Gym/SWE-Bench-Fork vendored official adapter',
            'evidence': 'The vendored harness has no SWE-smith adapter/semantics; the pinned Smith row supplies an x86_64 image reference and no base_commit/version. Native-arm64 execution was not emulated or substituted.',
        }
        cert.write_text(json.dumps(report, indent=2))
        rows.append({
            'task_uid': uid,
            'status': 'fail',
            'selection_used_model_outcome': False,
            'certificate_ref': f'protected-preflight/task-{iid}/report.json',
            'image_digest': None,
            'rejection_reason': 'unsupported_architecture',
            'harness': 'SWE-Gym/SWE-Bench-Fork (official adapter, local vendor clone at experiments/new_datasets/w2_preflight/_vendor/SWE-Bench-Fork)',
            'arch': 'arm64',
            'platform': 'linux/arm64/v8',
            'model_patch_applied': 'not attempted',
            'notes': 'Smith source row and official image reference were resolved from the pinned parquet. The official vendored harness contains no SWE-smith-specific buggy-state handling, and the referenced image is x86_64; no x86 emulation or guessed arm64 substitution was used. No image was pulled or graded.'
        })
    out = ROOT / 'preflight.train50.v1.jsonl'
    out.write_text(''.join(json.dumps(x, separators=(',', ':')) + '\n' for x in rows))
    print(f'wrote {len(rows)} Smith rejection rows to {out}')

if __name__ == '__main__': main()
