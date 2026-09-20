"""Retry capture for the 8 gym_batch2 tasks that came back BOTH_FAIL,
with a longer 2400s (2x) active-task budget instead of the original
1200s. Reasoning: successful gym_batch2 runs already used up to 1177s
(98% of the original budget) on cloud and 779s (65%) on edge for
relatively small tasks; the evidenced ~4x edge slowdown (GPU power/
thermal throttling, see AUDIT.md) means 1200s may simply not be enough
wall-clock time for these harder tasks on the local backend, separate
from whether the fix itself is correct. This is a NEW protocol version
(gym-batch2-retry-2400s-v1), not a silent extension of the frozen
gym-batch2-v1 protocol -- kept in its own output dir so the original
1200s-budget captures are untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

W3 = Path(__file__).resolve().parent
sys.path.insert(0, str(W3))
import run_smoke_capture as m  # noqa: E402

INSTANCES_FILE = W3.parent / "w2_preflight/gym_batch2_14tasks_instances.json"
OUT_ROOT = W3 / "gym_batch2_retry"
RETRY_INSTANCE_IDS = {
    "conan-io__conan-13721", "conan-io__conan-13788", "conan-io__conan-14296",
    "facebookresearch__hydra-1791", "facebookresearch__hydra-2290",
    "getmoto__moto-6121", "python__mypy-11420", "python__mypy-9629",
}
RETRY_MAX_ACTIVE_SECONDS = 2400


def main() -> None:
    if "ANTHROPIC_AUTH_TOKEN" not in os.environ:
        for line in (W3.parent.parent.parent / ".env").read_text().splitlines():
            if line.startswith("ANTHROPIC_AUTH_TOKEN="):
                os.environ["ANTHROPIC_AUTH_TOKEN"] = line.split("=", 1)[1].strip()
                break
    assert "ANTHROPIC_AUTH_TOKEN" in os.environ

    all_instances = m.load_swebench_dataset(str(INSTANCES_FILE))
    dataset = [x for x in all_instances if x["instance_id"] in RETRY_INSTANCE_IDS]
    assert len(dataset) == len(RETRY_INSTANCE_IDS), (len(dataset), len(RETRY_INSTANCE_IDS))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []

    for instance in dataset:
        iid = instance["instance_id"]
        test_spec = m.make_test_spec(instance)
        image = f"sweb.eval.{test_spec.arch}.{iid}:latest"

        run_seed = int.from_bytes(hashlib.sha256(f"gymbatch2retry|order|{iid}|{time.time_ns()}".encode()).digest()[:8], "big")
        order = list(m.POLICIES.items())
        random.Random(run_seed).shuffle(order)

        for policy, cfg in order:
            run_dir = OUT_ROOT / f"{iid}__{policy}"
            if (run_dir / "run_meta.json").exists():
                print(f"=== skip (already captured): {iid} / {policy} ===", flush=True)
                continue
            if run_dir.exists():
                print(f"=== retrying incomplete stub: {iid} / {policy} ===", flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"=== {iid} / {policy} ===", flush=True)

            name = m.container_name_for(iid, policy) + "-retry"
            t0 = time.time()
            m.start_container(image, name)
            m.setup_container(name)
            t1 = time.time()

            session_id = __import__("uuid").uuid4().__str__()
            prompt = m.render_prompt(instance)
            (run_dir / "prompt.txt").write_text(prompt)

            t2 = time.time()
            stdout, returncode, timed_out = m.run_claude(name, prompt, cfg["base_url"], cfg["model"], session_id, RETRY_MAX_ACTIVE_SECONDS,
                                                          run_dir / "claude_stream.jsonl")
            t3 = time.time()

            patch = m.extract_patch(name)
            (run_dir / "final_patch.diff").write_text(patch)
            m.sh(["docker", "rm", "-f", name])

            record = {
                "instance_id": iid, "policy": policy, "run_dir": run_dir.name, "repeat_index": 0,
                "session_id": session_id,
                "execution_order": {"seed": run_seed, "policies_in_order": [p for p, _ in order]},
                "max_active_seconds": RETRY_MAX_ACTIVE_SECONDS,
                "container_setup_seconds": round(t1 - t0, 2),
                "claude_wall_seconds": round(t3 - t2, 2),
                "claude_returncode": returncode, "claude_timed_out": timed_out,
                "patch_nonempty": bool(patch.strip()), "patch_bytes": len(patch),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            (run_dir / "run_meta.json").write_text(json.dumps(record, indent=1))
            manifest.append(record)
            print(json.dumps(record, indent=1), flush=True)

    (OUT_ROOT / "capture_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("RETRY_CAPTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
