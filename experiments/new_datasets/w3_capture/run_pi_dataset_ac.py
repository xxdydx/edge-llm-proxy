"""Permanent Pi Coder A/C collector for 21 preflighted SWE-Gym tasks (42 cells).

Protocol: pi-dataset-ac-v1
Trajectory timeout: 1800s
Turn count limit: none
Action repeat breaker: 4 identical consecutive actions
Teardown: bounded
Backends:
  - edge: http://host.docker.internal:18012 (model: local)
  - cloud: http://host.docker.internal:18011 (model: deepseek-v4-flash)
Backend order: randomized deterministic per task
Outputs: experiments/new_datasets/w3_capture/pi_dataset_ac_v1
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

W3 = Path(__file__).resolve().parent
ND = W3.parent
REPO_ROOT = W3.parents[2]
sys.path.insert(0, str(W3))

OUT_ROOT = W3 / "pi_dataset_ac_v1"
PROTOCOL_VERSION = "pi-dataset-ac-v1"
MAX_ACTIVE_SECONDS = 1800
MAX_IDENTICAL_ACTION_REPEATS = 4

SOURCE_FILES = (
    ND / "w2_preflight/smoke_instances.json",
    ND / "w2_preflight/train50_5tasks_instances.json",
    ND / "w2_preflight/gym_batch2_14tasks_instances.json",
)

BACKENDS = {
    "edge": {"base_url": "http://host.docker.internal:18012", "model": "local"},
    "cloud": {"base_url": "http://host.docker.internal:18011", "model": "deepseek-v4-flash"},
}
STREAM_FILENAME = "pi_stream.jsonl"


class PiDatasetProtocolError(RuntimeError):
    """Protocol violation or corrupted capture checkpoint."""


class SetupError(RuntimeError):
    """Container startup or environment preparation failed."""


class ProcessError(RuntimeError):
    """Pi execution crashed or returned an unexpected exit status."""


def _capture_api() -> Any:
    """Load the Docker/SWE-bench harness adapter lazily."""
    return importlib.import_module("run_smoke_capture")


def _pi_harness() -> Any:
    """Load the Pi harness adapter lazily."""
    return importlib.import_module("pi_harness")


@dataclass(frozen=True)
class PlannedCell:
    instance: dict[str, Any]
    source_instances_file: Path
    backend: str
    image: str
    prompt: str
    execution_order: dict[str, Any]

    @property
    def instance_id(self) -> str:
        return str(self.instance["instance_id"])

    @property
    def cell_id(self) -> str:
        return f"{self.instance_id}__{self.backend}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_nd(path: Path) -> str:
    return path.resolve().relative_to(ND.resolve()).as_posix()


def _load_raw_instances(source_file: Path) -> list[dict[str, Any]]:
    rows = json.loads(source_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise PiDatasetProtocolError(f"Expected JSON list of task rows in {source_file}")
    return [row for row in rows if isinstance(row, dict) and "instance_id" in row]


def load_all_preflight_tasks() -> list[tuple[Path, dict[str, Any]]]:
    """Load and deduplicate exactly the 21 preflighted tasks from the 3 sources."""
    seen_ids: set[str] = set()
    selected: list[tuple[Path, dict[str, Any]]] = []
    for source_file in SOURCE_FILES:
        if not source_file.is_file():
            raise FileNotFoundError(f"Source instances file missing: {source_file}")
        for instance in _load_raw_instances(source_file):
            iid = str(instance["instance_id"])
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
            selected.append((source_file, instance))
    if len(selected) != 21:
        raise PiDatasetProtocolError(
            f"Expected exactly 21 unique preflighted tasks, got {len(selected)}"
        )
    return selected


def _execution_order(instance_id: str) -> tuple[list[str], dict[str, Any]]:
    seed_material = f"{PROTOCOL_VERSION}|backend-order|{instance_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    backends = list(BACKENDS.keys())
    random.Random(seed).shuffle(backends)
    return backends, {"seed": seed, "backends_in_order": backends}


def build_plan() -> tuple[list[PlannedCell], dict[str, Any], str]:
    """Build the deterministic 42-cell plan and compute protocol fingerprint."""
    tasks = load_all_preflight_tasks()
    task_docs: list[dict[str, Any]] = []
    plan: list[PlannedCell] = []
    api = _capture_api()

    for source_file, instance in tasks:
        instance_id = str(instance["instance_id"])
        prompt = api.render_prompt(instance)
        test_spec = api.make_test_spec(instance)
        image = f"sweb.eval.{test_spec.arch}.{instance_id}:latest"
        task_doc = {
            "instance_id": instance_id,
            "source_instances_file": _relative_to_nd(source_file),
            "source_sha256": _sha256_file(source_file),
            "image": image,
            "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        }
        task_docs.append(task_doc)

        backends_in_order, order_meta = _execution_order(instance_id)
        for position, backend_name in enumerate(backends_in_order, start=1):
            plan.append(
                PlannedCell(
                    instance=instance,
                    source_instances_file=source_file,
                    backend=backend_name,
                    image=image,
                    prompt=prompt,
                    execution_order={**order_meta, "position": position},
                )
            )

    if len(plan) != 42 or len({c.cell_id for c in plan}) != 42:
        raise PiDatasetProtocolError(f"Expected exactly 42 unique cells, got {len(plan)}")

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "task_count": 21,
        "cell_count": 42,
        "tasks": task_docs,
        "backends": BACKENDS,
        "max_active_seconds": MAX_ACTIVE_SECONDS,
        "max_turns": None,
        "identical_action_repeat_limit": MAX_IDENTICAL_ACTION_REPEATS,
        "container_platform": "linux/arm64/v8",
        "official_grading": "separate_post_capture_phase",
    }
    fingerprint = _sha256_bytes(_canonical_json(protocol))
    return plan, protocol, fingerprint


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with temp_path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def _create_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def ensure_output_protocol(out_root: Path, protocol: dict[str, Any], fingerprint: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    protocol_path = out_root / "protocol.json"
    if protocol_path.exists():
        prior = json.loads(protocol_path.read_text(encoding="utf-8"))
        if prior.get("protocol_fingerprint") != fingerprint:
            raise PiDatasetProtocolError(
                f"Output root {out_root} belongs to protocol fingerprint "
                f"{prior.get('protocol_fingerprint')}, expected {fingerprint}"
            )
        return
    if any(out_root.iterdir()):
        raise PiDatasetProtocolError(
            f"Output root {out_root} is nonempty but has no protocol.json; refusing overwrite"
        )
    _create_json_exclusive(
        protocol_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_fingerprint": fingerprint,
            "protocol": protocol,
            "created_at_utc": _utc_now(),
        },
    )


def _cell_identity(cell: PlannedCell, fingerprint: str) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "instance_id": cell.instance_id,
        "source_instances_file": _relative_to_nd(cell.source_instances_file),
        "backend": cell.backend,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "execution_order": cell.execution_order,
        "image": cell.image,
        "prompt_sha256": _sha256_bytes(cell.prompt.encode("utf-8")),
        "max_active_seconds": MAX_ACTIVE_SECONDS,
        "max_turns": None,
    }


def ensure_cell_meta(out_root: Path, cell: PlannedCell, fingerprint: str) -> Path:
    cell_dir = out_root / cell.cell_id
    meta_path = cell_dir / "run_meta.json"
    if not cell_dir.exists():
        cell_dir.mkdir(parents=True, exist_ok=False)
        meta = {
            **_cell_identity(cell, fingerprint),
            "status": "pending",
            "classification": "pending",
            "attempts": [],
            "official_pass": None,
            "official_grade_status": "not_run",
            "output_tokens": None,
            "infrastructure_failure": False,
            "protocol_failure": False,
        }
        _create_json_exclusive(meta_path, meta)
        return meta_path
    if not meta_path.exists():
        raise PiDatasetProtocolError(
            f"Cell directory exists without run_meta.json: {cell_dir}; refusing overwrite"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = _cell_identity(cell, fingerprint)
    for key, expected_value in expected.items():
        if meta.get(key) != expected_value:
            raise PiDatasetProtocolError(
                f"Cell metadata mismatch for {cell.cell_id}: {key} (expected {expected_value}, got {meta.get(key)})"
            )
    return meta_path


def _write_manifest(out_root: Path) -> dict[str, Any]:
    protocol_path = out_root / "protocol.json"
    protocol_doc = json.loads(protocol_path.read_text(encoding="utf-8"))
    cells: list[dict[str, Any]] = []
    for meta_path in sorted(out_root.glob("*/run_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        latest_attempt = meta.get("attempts", [])[-1] if meta.get("attempts") else {}
        cells.append({**latest_attempt, **meta, "attempts": meta.get("attempts", [])})
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_doc["protocol_fingerprint"],
        "total_cells": 42,
        "completed_cells": sum(c.get("status") == "completed" for c in cells),
        "failed_cells": sum(c.get("status") == "failed" for c in cells),
        "pending_cells": sum(c.get("status") == "pending" for c in cells),
        "official_grade_status": "complete" if all_official_grades_recorded(cells) else "pending",
        "cells": cells,
        "updated_at_utc": _utc_now(),
    }
    _atomic_write_json(out_root / "capture_manifest.json", manifest)
    return manifest


def all_official_grades_recorded(cells: list[dict[str, Any]]) -> bool:
    return (
        len(cells) == 42
        and all(cell.get("status") == "completed" for cell in cells)
        and all(cell.get("official_grade_status") == "graded" for cell in cells)
        and all(isinstance(cell.get("official_pass"), bool) for cell in cells)
    )


def record_official_grade(
    out_root: Path,
    cell_id: str,
    official_pass: bool,
    grader_reference: str,
    grade_details: dict[str, Any] | None = None,
) -> None:
    """Record an official SWE-Bench-Fork grade outcome into a completed cell."""
    if not isinstance(official_pass, bool):
        raise TypeError("official_pass must be bool")
    meta_path = out_root / cell_id / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "completed":
        raise PiDatasetProtocolError(f"Cannot grade non-completed cell {cell_id}")
    if meta.get("official_grade_status") != "not_run" or meta.get("official_pass") is not None:
        raise PiDatasetProtocolError(f"Official grade already recorded for {cell_id}; refusing overwrite")
    meta["official_pass"] = official_pass
    meta["official_grade_status"] = "graded"
    meta["official_grader_reference"] = grader_reference
    meta["official_grade_details"] = grade_details or {}
    meta["official_graded_at_utc"] = _utc_now()
    _atomic_write_json(meta_path, meta)
    _write_manifest(out_root)


def _load_auth_token() -> None:
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_AUTH_TOKEN="):
                os.environ["ANTHROPIC_AUTH_TOKEN"] = line.split("=", 1)[1].strip()
                break
    if not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise RuntimeError("missing ANTHROPIC_AUTH_TOKEN")


def _safe_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if token:
        message = message.replace(token, "[REDACTED]")
    return message[:2000]


def _new_attempt(cell: PlannedCell, cell_dir: Path, meta_path: Path) -> tuple[dict[str, Any], Path]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") == "completed":
        return meta, cell_dir
    attempts = list(meta.get("attempts", []))
    if attempts and attempts[-1].get("status") == "running":
        attempts[-1] = {**attempts[-1], "status": "interrupted", "ended_at_utc": _utc_now()}
    attempt_number = len(attempts) + 1
    attempt_dir = cell_dir / f"attempt-{attempt_number:03d}"
    attempt_dir.mkdir(parents=False, exist_ok=False)
    session_id = str(uuid.uuid4())
    attempt = {
        "attempt_number": attempt_number,
        "attempt_dir": attempt_dir.name,
        "session_id": session_id,
        "status": "running",
        "started_at_utc": _utc_now(),
        "infrastructure_failure": False,
        "protocol_failure": False,
    }
    attempts.append(attempt)
    meta["attempts"] = attempts
    meta["status"] = "running"
    meta["active_attempt"] = attempt_number
    meta["session_id"] = session_id
    meta["attempt_dir"] = attempt_dir.name
    meta["termination_reason"] = None
    meta["classification"] = "running"
    meta["timed_out"] = None
    meta["wall_seconds"] = None
    meta["setup_seconds"] = None
    meta["returncode"] = None
    meta["action_count"] = None
    meta["model_turn_count"] = None
    meta["repeated_action_count"] = None
    meta["repeated_action_signature"] = None
    meta["output_tokens"] = None
    meta["patch_nonempty"] = None
    meta["patch_bytes"] = None
    meta["timestamp_utc"] = _utc_now()
    meta["infrastructure_failure"] = False
    meta["protocol_failure"] = False
    _atomic_write_json(meta_path, meta)
    return meta, attempt_dir


def _classify_result(termination_reason: str, returncode: int | None) -> tuple[str, str]:
    """Map outcome to (termination_reason, classification).

    Classifications supported: setup, process, protocol, timeout, repeat, completed.
    """
    if termination_reason == "repeated_action":
        return "repeated_action", "repeat"
    if termination_reason == "trajectory_deadline":
        return "trajectory_deadline", "timeout"
    if termination_reason == "process_error" or (returncode is not None and returncode != 0):
        return "process_error", "process"
    if termination_reason == "completed" and returncode == 0:
        return "completed", "completed"
    return termination_reason, "process"


def execute_cell(out_root: Path, cell: PlannedCell, fingerprint: str) -> dict[str, Any]:
    cell_dir = out_root / cell.cell_id
    meta_path = ensure_cell_meta(out_root, cell, fingerprint)
    initial_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if initial_meta.get("status") == "completed":
        return initial_meta

    meta, attempt_dir = _new_attempt(cell, cell_dir, meta_path)
    attempt_number = int(meta["active_attempt"])
    attempt = meta["attempts"][-1]
    session_id = attempt["session_id"]

    safe_task = cell.instance_id.replace("/", "-").replace("_", "-")
    container_name = f"pi-ac-{safe_task}-{cell.backend}-{attempt_number:03d}".lower()

    stream_path = attempt_dir / STREAM_FILENAME
    prompt_path = attempt_dir / "prompt.txt"
    patch_path = attempt_dir / "final_patch.diff"

    with prompt_path.open("x", encoding="utf-8") as stream:
        stream.write(cell.prompt)

    container_started = False
    setup_seconds = 0.0
    run_started = 0.0
    try:
        api = _capture_api()
        harness = _pi_harness()

        # 1. Start clean ARM64 container
        try:
            api.start_container(cell.image, container_name)
            container_started = True
        except Exception as exc:
            raise SetupError(f"Failed to start container {container_name}: {exc}") from exc

        # 2. Pinned Pi/Node installation inside container
        setup_started = time.monotonic()
        try:
            harness.setup_pi_container(container_name)
        except Exception as exc:
            setup_seconds = time.monotonic() - setup_started
            raise SetupError(f"Pi environment setup failed in {container_name}: {exc}") from exc
        setup_seconds = time.monotonic() - setup_started

        # 3. Execute Pi trajectory (1800s active timeout, no turn limit, 4-repeat breaker)
        backend = BACKENDS[cell.backend]
        run_started = time.monotonic()
        raw_result = harness.run_pi(
            container_name=container_name,
            prompt=cell.prompt,
            base_url=backend["base_url"],
            model=backend["model"],
            session_id=session_id,
            timeout_s=MAX_ACTIVE_SECONDS,
            out_path=stream_path,
            max_turns=None,  # absolutely no turn-count limit
        )
        wall_seconds = time.monotonic() - run_started

        # 4. Classify outcome
        term_reason, classification = _classify_result(
            raw_result.termination_reason, raw_result.returncode
        )
        if classification == "process":
            raise ProcessError(
                f"Pi terminated with process error: returncode={raw_result.returncode} "
                f"stderr={raw_result.stderr[-500:]}"
            )

        # 5. Extract patch
        try:
            patch = api.extract_patch(container_name)
        except Exception as exc:
            raise ProcessError(f"Failed to extract patch from {container_name}: {exc}") from exc

        with patch_path.open("x", encoding="utf-8") as stream:
            stream.write(patch)

        result_record = {
            **_cell_identity(cell, fingerprint),
            "session_id": session_id,
            "status": "completed",
            "classification": classification,
            "setup_seconds": round(setup_seconds, 3),
            "wall_seconds": round(wall_seconds, 3),
            "returncode": raw_result.returncode,
            "termination_reason": term_reason,
            "timed_out": raw_result.timed_out,
            "action_count": raw_result.action_count,
            "model_turn_count": raw_result.model_turn_count,
            "repeated_action_count": raw_result.repeated_action_count,
            "repeated_action_signature": raw_result.repeated_action_signature,
            "output_tokens": raw_result.output_tokens,
            "patch_nonempty": bool(patch.strip()),
            "patch_bytes": len(patch.encode("utf-8")),
            "patch_path": f"{cell.cell_id}/{attempt_dir.name}/{patch_path.name}",
            "stream_path": f"{cell.cell_id}/{attempt_dir.name}/{stream_path.name}",
            "prompt_path": f"{cell.cell_id}/{attempt_dir.name}/{prompt_path.name}",
            "official_pass": None,
            "official_grade_status": "not_run",
            "infrastructure_failure": False,
            "protocol_failure": False,
            "timestamp_utc": _utc_now(),
        }
        _complete_attempt(meta_path, result_record)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    except SetupError as exc:
        _fail_attempt(meta_path, attempt_number, _safe_error(exc), setup_seconds, run_started, "setup")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except ProcessError as exc:
        _fail_attempt(meta_path, attempt_number, _safe_error(exc), setup_seconds, run_started, "process")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except PiDatasetProtocolError as exc:
        _fail_attempt(meta_path, attempt_number, _safe_error(exc), setup_seconds, run_started, "protocol")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail_attempt(meta_path, attempt_number, _safe_error(exc), setup_seconds, run_started, "process")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    finally:
        if container_started:
            try:
                _capture_api().sh(["docker", "rm", "-f", container_name])
            except Exception:
                pass


def _complete_attempt(meta_path: Path, result_record: dict[str, Any]) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    attempts = list(meta.get("attempts", []))
    latest = attempts[-1]
    latest.update({
        **result_record,
        "attempt_number": latest["attempt_number"],
        "attempt_dir": latest["attempt_dir"],
        "status": "completed",
        "started_at_utc": latest["started_at_utc"],
        "ended_at_utc": _utc_now(),
    })
    attempts[-1] = latest
    meta.update(result_record)
    meta["attempts"] = attempts
    meta.pop("active_attempt", None)
    _atomic_write_json(meta_path, meta)


def _fail_attempt(
    meta_path: Path,
    attempt_number: int,
    error: str,
    setup_seconds: float,
    run_started: float,
    failure_kind: str,
) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    attempts = list(meta.get("attempts", []))
    latest = attempts[-1]
    elapsed = time.monotonic() - run_started if run_started else 0.0
    protocol_failure = failure_kind == "protocol"
    infra_failure = failure_kind in {"setup", "infrastructure"}
    term_reason = f"{failure_kind}_error"

    fail_info = {
        "status": "failed",
        "classification": failure_kind,
        "ended_at_utc": _utc_now(),
        "setup_seconds": round(setup_seconds, 3),
        "wall_seconds": round(elapsed, 3),
        "termination_reason": term_reason,
        "timed_out": False,
        "returncode": None,
        "action_count": 0,
        "model_turn_count": 0,
        "repeated_action_count": 0,
        "repeated_action_signature": None,
        "output_tokens": None,
        "patch_nonempty": False,
        "patch_bytes": 0,
        "error": error,
        "infrastructure_failure": infra_failure,
        "protocol_failure": protocol_failure,
    }
    latest.update(fail_info)
    attempts[-1] = latest
    meta.update({
        **fail_info,
        "attempts": attempts,
        "active_attempt": attempt_number,
        "timestamp_utc": _utc_now(),
    })
    _atomic_write_json(meta_path, meta)


def run_all(out_root: Path = OUT_ROOT) -> dict[str, Any]:
    plan, protocol, fingerprint = build_plan()
    ensure_output_protocol(out_root, protocol, fingerprint)
    for cell in plan:
        ensure_cell_meta(out_root, cell, fingerprint)
    _write_manifest(out_root)

    for cell in plan:
        current = json.loads((out_root / cell.cell_id / "run_meta.json").read_text(encoding="utf-8"))
        if current.get("status") == "completed":
            print(f"=== skip captured {cell.cell_id} ===", flush=True)
            continue
        print(f"=== capture {cell.cell_id} ===", flush=True)
        execute_cell(out_root, cell, fingerprint)
        _write_manifest(out_root)
    return _write_manifest(out_root)


def main() -> None:
    _load_auth_token()
    manifest = run_all()
    print(
        json.dumps(
            {
                "total_cells": manifest["total_cells"],
                "completed_cells": manifest["completed_cells"],
                "failed_cells": manifest["failed_cells"],
                "pending_cells": manifest["pending_cells"],
                "official_grade_status": manifest["official_grade_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
