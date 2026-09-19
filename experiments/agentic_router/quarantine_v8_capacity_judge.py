"""One-off, fail-closed quarantine of an accidental v8 capacity-invalid judge.

Keeps byte-identical original and isolated offending rows. The clean judge
JSONL is then safe for readers that only accept a judge path, while paired
replay readers must still filter non-OK arm outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "results" / "postpilot_v8"
JUDGE = ROOT / "pilot_judge_consistency_v5_order.jsonl"
PAIRS = ROOT / "stage1_replay_examples_pilot_v1.jsonl"
BACKUP = ROOT / "pilot_judge_consistency_v5_order_pre_capacity_quarantine.jsonl"
QUARANTINE = ROOT / "pilot_judge_capacity_invalid_quarantine.jsonl"
MANIFEST = ROOT / "pilot_judge_capacity_invalid_quarantine_manifest.json"
CALL_ID = "2fb85fff-de5a-475d-962a-0ce4256141fd"
EXPECTED_SHA256 = "727430141d0df1a794649cee71451141c343fc6498a52a944e9b3d9cdbf53d05"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exclusive_durable(path: Path, data: bytes) -> None:
    with path.open("xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def main() -> None:
    if any(path.exists() for path in (BACKUP, QUARANTINE, MANIFEST)):
        raise RuntimeError("quarantine artifacts already exist; refusing duplicate rewrite")
    pair_rows = [json.loads(line) for line in PAIRS.read_text().splitlines() if line]
    matched_pairs = [row for row in pair_rows if row["call"]["call_id"] == CALL_ID]
    if len(matched_pairs) != 1 or matched_pairs[0]["local_outcome"]["status"] != "INVALID_CAPACITY" or matched_pairs[0]["cloud_outcome"]["status"] != "OK":
        raise RuntimeError("exact capacity-invalid pair identity/status check failed")
    original = JUDGE.read_bytes()
    if _sha(original) != EXPECTED_SHA256:
        raise RuntimeError("judge source changed; refusing rewrite")
    lines = original.splitlines(keepends=True)
    selected = [line for line in lines if json.loads(line)["call_id"] == CALL_ID]
    if len(selected) != 2 or {json.loads(line)["pass_label"] for line in selected} != {"primary", "reversed"}:
        raise RuntimeError("expected exactly the primary and reversed diagnostic rows")
    clean = b"".join(line for line in lines if json.loads(line)["call_id"] != CALL_ID)
    quarantined = b"".join(selected)
    _exclusive_durable(BACKUP, original)
    _exclusive_durable(QUARANTINE, quarantined)
    fd, temp_name = tempfile.mkstemp(prefix=".judge-clean-", dir=ROOT)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(clean)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, JUDGE)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    manifest = {
        "status": "quarantined_nonlabelable_capacity_invalid_diagnostic",
        "call_id": CALL_ID,
        "reason": "local arm INVALID_CAPACITY; diagnostic judge bypassed normal eligibility guard",
        "paired_local_status": "INVALID_CAPACITY",
        "paired_cloud_status": "OK",
        "pass_labels": ["primary", "reversed"],
        "original_judge_file": BACKUP.name,
        "original_sha256": _sha(original),
        "quarantine_file": QUARANTINE.name,
        "quarantine_sha256": _sha(quarantined),
        "clean_judge_file": JUDGE.name,
        "clean_sha256": _sha(clean),
        "n_original_rows": len(lines),
        "n_quarantined_rows": len(selected),
        "n_clean_rows": len(lines) - len(selected),
        "exclude_from_quality_training_and_judge_audit": True,
    }
    _exclusive_durable(MANIFEST, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({key: manifest[key] for key in
                      ("call_id", "n_original_rows", "n_quarantined_rows",
                       "n_clean_rows", "original_sha256", "clean_sha256")}))


if __name__ == "__main__":
    main()
