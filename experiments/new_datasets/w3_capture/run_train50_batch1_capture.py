"""Real capture for the 5 already-preflighted, passing train50 Gym tasks
(mypy-15413, dvc-5336, conan-14177, mypy-12222, conan-15422). Reuses
run_smoke_capture.py's functions (execution-order randomization included)
with a separate instances file and output root -- does not touch the smoke
stage's own w3_capture/*__edge-only-v1 / *__cloud-only-v1 directories.
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

INSTANCES_FILE = W3.parent / "w2_preflight/train50_5tasks_instances.json"
OUT_ROOT = W3 / "train50_batch1"


def main() -> None:
    if "ANTHROPIC_AUTH_TOKEN" not in os.environ:
        for line in (W3.parent.parent.parent / ".env").read_text().splitlines():
            if line.startswith("ANTHROPIC_AUTH_TOKEN="):
                os.environ["ANTHROPIC_AUTH_TOKEN"] = line.split("=", 1)[1].strip()
                break
    assert "ANTHROPIC_AUTH_TOKEN" in os.environ

    dataset = m.load_swebench_dataset(str(INSTANCES_FILE))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []

    for instance in dataset:
        iid = instance["instance_id"]
        test_spec = m.make_test_spec(instance)
        image = f"sweb.eval.{test_spec.arch}.{iid}:latest"

        run_seed = int.from_bytes(hashlib.sha256(f"train50-batch1|order|{iid}|{time.time_ns()}".encode()).digest()[:8], "big")
        order = list(m.POLICIES.items())
        random.Random(run_seed).shuffle(order)

        for policy, cfg in order:
            run_dir = OUT_ROOT / f"{iid}__{policy}"
            # run_meta.json is only written after a real completed attempt;
            # a bare existing dir (prompt.txt only) means a prior crash left
            # an incomplete stub, not a genuine capture -- must not be
            # silently treated as done.
            if (run_dir / "run_meta.json").exists():
                print(f"=== skip (already captured): {iid} / {policy} ===", flush=True)
                continue
            if run_dir.exists():
                print(f"=== retrying incomplete stub: {iid} / {policy} ===", flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"=== {iid} / {policy} ===", flush=True)

            name = m.container_name_for(iid, policy)
            t0 = time.time()
            m.start_container(image, name)
            m.setup_container(name)
            t1 = time.time()

            session_id = __import__("uuid").uuid4().__str__()
            prompt = m.render_prompt(instance)
            (run_dir / "prompt.txt").write_text(prompt)

            t2 = time.time()
            result = m.run_claude(name, prompt, cfg["base_url"], cfg["model"], session_id, m.MAX_ACTIVE_SECONDS,
                                  run_dir / "claude_stream.jsonl")
            t3 = time.time()

            patch = m.extract_patch(name)
            (run_dir / "final_patch.diff").write_text(patch)
            m.sh(["docker", "rm", "-f", name])

            record = {
                "instance_id": iid, "policy": policy, "run_dir": run_dir.name, "repeat_index": 0,
                "session_id": session_id,
                "execution_order": {"seed": run_seed, "policies_in_order": [p for p, _ in order]},
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
    print("BATCH1_CAPTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
