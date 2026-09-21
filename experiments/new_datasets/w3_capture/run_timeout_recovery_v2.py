"""Corrected, resumable rerun of every unique historical timeout pair.

This is an explicitly versioned recovery protocol. It never writes into the
original captures or the first data-loss retry. Edge requests use a separate
proxy endpoint configured for an 8,192-token per-call output cap and disabled
Qwen thinking. Both backends use the four-identical-action loop breaker in
run_smoke_capture.run_claude().
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

W3 = Path(__file__).resolve().parent
sys.path.insert(0, str(W3))
import run_data_loss_retry_capture as historical  # noqa: E402
import run_smoke_capture as m  # noqa: E402

ND = W3.parent
OUT_ROOT = W3 / "timeout_recovery_v2"
MAX_ACTIVE_SECONDS = 1200
PROTOCOL_VERSION = "timeout-recovery-v2-edge-8k-no-thinking-repeat4"

POLICIES = {
    "edge-only-v1": {
        "recovery_policy": "edge-only-v2",
        "base_url": "http://host.docker.internal:18012",
        "model": "local",
        "local_max_output_tokens": 8192,
        "local_thinking": "disabled",
    },
    "cloud-only-v1": {
        "recovery_policy": "cloud-only-v2",
        "base_url": "http://host.docker.internal:18011",
        "model": "deepseek-v4-flash",
        "local_max_output_tokens": None,
        "local_thinking": None,
    },
}

# The historical list already collapses the 19 raw timed-out attempts to the
# 18 unique (dataset, task, backend) pairs that require a corrected rerun.
TARGETS = list(historical.TARGETS)
assert len(TARGETS) == 18
assert len({(str(path), iid, policy) for path, iid, policy in TARGETS}) == 18


def _load_token() -> None:
    if "ANTHROPIC_AUTH_TOKEN" in os.environ:
        return
    for line in (ND.parent.parent / ".env").read_text().splitlines():
        if line.startswith("ANTHROPIC_AUTH_TOKEN="):
            os.environ["ANTHROPIC_AUTH_TOKEN"] = line.split("=", 1)[1].strip()
            break
    assert "ANTHROPIC_AUTH_TOKEN" in os.environ


def main() -> None:
    _load_token()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_cache: dict[Path, dict[str, dict]] = {}

    for instances_file, iid, original_policy in TARGETS:
        cfg = POLICIES[original_policy]
        recovery_policy = cfg["recovery_policy"]
        run_dir = OUT_ROOT / f"{iid}__{recovery_policy}"
        if (run_dir / "run_meta.json").exists():
            print(f"=== skip (already captured): {iid} / {recovery_policy} ===", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)

        if instances_file not in dataset_cache:
            dataset_cache[instances_file] = {
                x["instance_id"]: x
                for x in m.load_swebench_dataset(str(instances_file))
            }
        instance = dataset_cache[instances_file][iid]
        test_spec = m.make_test_spec(instance)
        image = f"sweb.eval.{test_spec.arch}.{iid}:latest"
        name = m.container_name_for(iid, recovery_policy) + "-timeout-v2"
        print(f"=== {iid} / {recovery_policy} ===", flush=True)

        t0 = time.time()
        m.start_container(image, name)
        try:
            m.setup_container(name)
            t1 = time.time()
            session_id = str(uuid.uuid4())
            prompt = m.render_prompt(instance)
            (run_dir / "prompt.txt").write_text(prompt)

            t2 = time.time()
            result = m.run_claude(
                name,
                prompt,
                cfg["base_url"],
                cfg["model"],
                session_id,
                MAX_ACTIVE_SECONDS,
                run_dir / "claude_stream.jsonl",
            )
            t3 = time.time()
            patch = m.extract_patch(name)
            (run_dir / "final_patch.diff").write_text(patch)
        finally:
            m.sh(["docker", "rm", "-f", name])

        record = {
            "instance_id": iid,
            "source_instances_file": str(instances_file.relative_to(ND)),
            "original_timed_out_policy": original_policy,
            "policy": recovery_policy,
            "run_dir": run_dir.name,
            "repeat_index": 0,
            "session_id": session_id,
            "protocol_version": PROTOCOL_VERSION,
            "max_active_seconds": MAX_ACTIVE_SECONDS,
            "identical_action_repeat_limit": m.MAX_IDENTICAL_ACTION_REPEATS,
            "local_max_output_tokens": cfg["local_max_output_tokens"],
            "local_thinking": cfg["local_thinking"],
            "container_setup_seconds": round(t1 - t0, 2),
            "claude_wall_seconds": round(t3 - t2, 2),
            **result.metadata(),
            "patch_nonempty": bool(patch.strip()),
            "patch_bytes": len(patch),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "run_meta.json").write_text(json.dumps(record, indent=1))
        print(json.dumps(record, indent=1), flush=True)

        completed = sorted(OUT_ROOT.glob("*/run_meta.json"))
        manifest = [json.loads(path.read_text()) for path in completed]
        (OUT_ROOT / "capture_manifest.json").write_text(json.dumps(manifest, indent=1))

    completed = sorted(OUT_ROOT.glob("*/run_meta.json"))
    manifest = [json.loads(path.read_text()) for path in completed]
    (OUT_ROOT / "capture_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"TIMEOUT_RECOVERY_V2_DONE runs={len(manifest)}", flush=True)


if __name__ == "__main__":
    main()
