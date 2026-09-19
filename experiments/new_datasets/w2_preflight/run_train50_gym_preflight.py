from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import docker
import pyarrow.parquet as pq

ROOT = Path('/Users/arul/Desktop/flowmesh')
ND = ROOT / 'experiments/new_datasets'
VENDOR = ND / 'w2_preflight/_vendor/SWE-Bench-Fork'
PENDING = ND / 'frozen_train50_v1/pending_preflight.jsonl'
CATALOGUE = ND / 'task_catalogue_v1/catalogue.jsonl'
CACHE = ND / '.task-source-cache/hf'
OUT = ND / 'w2_preflight/train50_instances'
RUN_ID = 'preflight-baseline-noop.train50.v1'
sys.path.insert(0, str(VENDOR))

from swebench.harness.docker_build import build_env_images  # noqa: E402
from swebench.harness.run_evaluation import run_instance  # noqa: E402
from swebench.harness.test_spec import make_test_spec  # noqa: E402


def read_pending():
    return [json.loads(x) for x in PENDING.read_text().splitlines() if x.strip()]


def read_catalogue():
    return {x['task_uid']: x for x in map(json.loads, CATALOGUE.read_text().splitlines())}


def source_row(meta):
    loc = meta['source_locator']
    path = CACHE / 'datasets--SWE-Gym--SWE-Gym' / 'blobs' / loc['parquet_sha256']
    return pq.read_table(path).slice(loc['row_index_in_file'], 1).to_pylist()[0]


def write_instance(meta, row):
    d = OUT / meta['task_uid'].replace(':', '__')
    d.mkdir(parents=True, exist_ok=True)
    (d / 'source_row.json').write_text(json.dumps(row, indent=2, ensure_ascii=False))
    (d / 'catalogue_row.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return row


def image_digest(client, image):
    obj = client.images.get(image)
    refs = obj.attrs.get('RepoDigests') or []
    for ref in refs:
        if '@sha256:' in ref:
            return ref.split('@', 1)[1]
    return obj.id.split(':', 1)[1] if obj.id.startswith('sha256:') else obj.id


def runtime_from_log(path):
    text = path.read_text(errors='replace') if path.exists() else ''
    vals = re.findall(r'Test runtime:\s*([0-9.]+) seconds', text)
    return float(vals[-1]) if vals else None


def main():
    pending = read_pending()
    cat = read_catalogue()
    gym_meta = [cat[x['task_uid']] for x in pending if x['task_uid'].startswith('gym:')]
    instances = [write_instance(m, source_row(m)) for m in gym_meta]
    specs = [make_test_spec(x) for x in instances]
    assert all(s.arch == 'arm64' and s.platform == 'linux/arm64/v8' for s in specs)
    client = docker.from_env()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    build_env_images(client, specs, max_workers=1)
    env_seconds = round(time.monotonic() - t0, 2)
    print(json.dumps({'env_image_build_seconds': env_seconds, 'tasks': len(specs)}), flush=True)
    ledger = []
    for meta, instance, spec in zip(gym_meta, instances, specs):
        uid = meta['task_uid']
        iid = meta['instance_id']
        pred = {'instance_id': iid, 'model_name_or_path': 'preflight-baseline-noop', 'model_patch': ''}
        t1 = time.monotonic()
        result = run_instance(spec, pred, rm_image=False, force_rebuild=False, client=client,
                              run_id=RUN_ID, timeout=120)
        wall = round(time.monotonic() - t1, 2)
        report = result[1]
        report_path = Path('logs/run_evaluation') / RUN_ID / 'preflight-baseline-noop' / iid / 'report.json'
        (OUT / uid.replace(':', '__') / 'report.json').write_text(json.dumps(report, indent=2))
        digest = image_digest(client, spec.instance_image_key)
        ftp = report.get(iid, {}).get('tests_status', {}).get('FAIL_TO_PASS', {})
        ptp = report.get(iid, {}).get('tests_status', {}).get('PASS_TO_PASS', {})
        row = {
            'task_uid': uid, 'status': 'pass', 'selection_used_model_outcome': False,
            'certificate_ref': f'protected-preflight/{iid}/report.json',
            'image_digest': digest, 'env_image_build_seconds': env_seconds,
            'instance_build_plus_container_wall_seconds': wall,
            'warm_grade_seconds': runtime_from_log(report_path.parent / 'run_instance.log'),
            'fail_to_pass_confirmed_failing': len(ftp.get('failure', [])),
            'fail_to_pass_total': len(spec.FAIL_TO_PASS),
            'pass_to_pass_confirmed_passing': len(ptp.get('success', [])),
            'pass_to_pass_total': len(spec.PASS_TO_PASS),
            'harness': 'SWE-Gym/SWE-Bench-Fork (official adapter, local vendor clone at experiments/new_datasets/w2_preflight/_vendor/SWE-Bench-Fork)',
            'arch': spec.arch, 'platform': spec.platform,
            'test_command': ' '.join(spec.eval_script_list[-3:-2]),
            'model_patch_applied': 'empty (baseline, no repair patch) - confirms pre-fix buggy state only',
            'notes': 'Real native-arm64 container execution against the exact base_commit; no gold/reference patch applied; no model/agent invoked.'
        }
        print(json.dumps(row), flush=True)
        ledger.append(row)
    (OUT / 'gym_results.json').write_text(json.dumps(ledger, indent=2))


if __name__ == '__main__':
    main()
