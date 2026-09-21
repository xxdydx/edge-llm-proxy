"""Retry every (task, backend) run across the whole dataset that was
affected by the now-fixed stdout-capture data-loss bug (see AUDIT.md):
all 19 runs with claude_timed_out=True, regardless of patch_bytes.
Checked directly: claude_stream.jsonl is 0 bytes for every single one
of them, including the 9 where patch_bytes was nonzero (extract_patch()
reads the container's git diff independently of run_claude()'s stdout
capture, so a real patch could still be recovered even when the
trajectory stdout was lost). Dataset A/C grading for the 9 nonzero-patch
runs is built from the edgeproxy's own trace log, not this local debug
file, so those are not mis-graded -- but rerunning everything anyway per
explicit instruction to be maximally safe rather than rely on that
distinction. With the fix in place (run_smoke_capture.run_claude now
streams to disk and survives a kill), this reruns them for real and
gets a complete captured trajectory for every one.

Only backends that were actually timed-out are included; the other
backend for each of these tasks already has a real, unaffected
completed result and is left alone. Uses the same 2400s budget already
validated as reasonable for this harder batch.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

W3 = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(W3))
import run_smoke_capture as m  # noqa: E402

ND = W3.parent
OUT_ROOT = W3 / "data_loss_retry"
MAX_ACTIVE_SECONDS = 2400

GYM_BATCH2 = ND / "w2_preflight/gym_batch2_14tasks_instances.json"
TRAIN50_BATCH1 = ND / "w2_preflight/train50_5tasks_instances.json"

# (instances_file, instance_id, policy) -- every claude_timed_out=True run
# across the whole dataset (gym_batch2, gym_batch2_retry, train50_batch1).
TARGETS = [
    (GYM_BATCH2, "conan-io__conan-13788", "cloud-only-v1"),
    (GYM_BATCH2, "conan-io__conan-13788", "edge-only-v1"),
    (GYM_BATCH2, "conan-io__conan-14296", "cloud-only-v1"),
    (GYM_BATCH2, "conan-io__conan-14296", "edge-only-v1"),
    (GYM_BATCH2, "facebookresearch__hydra-1551", "edge-only-v1"),
    (GYM_BATCH2, "facebookresearch__hydra-1791", "cloud-only-v1"),
    (GYM_BATCH2, "facebookresearch__hydra-1791", "edge-only-v1"),
    (GYM_BATCH2, "facebookresearch__hydra-1915", "edge-only-v1"),
    (GYM_BATCH2, "facebookresearch__hydra-2290", "edge-only-v1"),
    (GYM_BATCH2, "getmoto__moto-5134", "edge-only-v1"),
    (GYM_BATCH2, "getmoto__moto-6121", "edge-only-v1"),
    (GYM_BATCH2, "iterative__dvc-1661", "edge-only-v1"),
    (GYM_BATCH2, "python__mypy-10401", "edge-only-v1"),
    (GYM_BATCH2, "python__mypy-11420", "edge-only-v1"),
    (GYM_BATCH2, "python__mypy-9629", "edge-only-v1"),
    (TRAIN50_BATCH1, "iterative__dvc-5336", "cloud-only-v1"),
    (TRAIN50_BATCH1, "iterative__dvc-5336", "edge-only-v1"),
    (TRAIN50_BATCH1, "python__mypy-12222", "edge-only-v1"),
]


def main() -> None:
    if "ANTHROPIC_AUTH_TOKEN" not in os.environ:
        for line in (ND.parent.parent / ".env").read_text().splitlines():
            if line.startswith("ANTHROPIC_AUTH_TOKEN="):
                os.environ["ANTHROPIC_AUTH_TOKEN"] = line.split("=", 1)[1].strip()
                break
    assert "ANTHROPIC_AUTH_TOKEN" in os.environ

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_cache: dict[Path, dict] = {}
    manifest = []

    for instances_file, iid, policy in TARGETS:
        if instances_file not in dataset_cache:
            dataset_cache[instances_file] = {
                x["instance_id"]: x for x in m.load_swebench_dataset(str(instances_file))
            }
        instance = dataset_cache[instances_file][iid]
        cfg = m.POLICIES[policy]
        test_spec = m.make_test_spec(instance)
        image = f"sweb.eval.{test_spec.arch}.{iid}:latest"

        run_dir = OUT_ROOT / f"{iid}__{policy}"
        if (run_dir / "run_meta.json").exists():
            print(f"=== skip (already captured): {iid} / {policy} ===", flush=True)
            continue
        if run_dir.exists():
            print(f"=== retrying incomplete stub: {iid} / {policy} ===", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== {iid} / {policy} ===", flush=True)

        name = m.container_name_for(iid, policy) + "-dlretry"
        t0 = time.time()
        m.start_container(image, name)
        m.setup_container(name)
        t1 = time.time()

        session_id = str(uuid.uuid4())
        prompt = m.render_prompt(instance)
        (run_dir / "prompt.txt").write_text(prompt)

        t2 = time.time()
        result = m.run_claude(
            name, prompt, cfg["base_url"], cfg["model"], session_id, MAX_ACTIVE_SECONDS,
            run_dir / "claude_stream.jsonl",
        )
        t3 = time.time()

        patch = m.extract_patch(name)
        (run_dir / "final_patch.diff").write_text(patch)
        m.sh(["docker", "rm", "-f", name])

        record = {
            "instance_id": iid, "policy": policy, "run_dir": run_dir.name, "repeat_index": 0,
            "session_id": session_id, "max_active_seconds": MAX_ACTIVE_SECONDS,
            "retry_reason": "original run was claude_timed_out=True with patch_bytes=0 due to the "
                             "since-fixed stdout-capture data-loss bug (real progress was made and lost)",
            "container_setup_seconds": round(t1 - t0, 2),
            "claude_wall_seconds": round(t3 - t2, 2),
            **result.metadata(),
            "patch_nonempty": bool(patch.strip()), "patch_bytes": len(patch),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "run_meta.json").write_text(json.dumps(record, indent=1))
        manifest.append(record)
        print(json.dumps(record, indent=1), flush=True)

    (OUT_ROOT / "capture_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("DATA_LOSS_RETRY_DONE", flush=True)


if __name__ == "__main__":
    main()
