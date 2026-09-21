"""Small, resumable Pi versus Claude Code harness trial.

This runner captures two frozen SWE-Gym tasks under edge/cloud backends and
both harnesses (eight cells total). Capture and official grading are separate
phases: each completed cell starts with ``official_pass: null`` until a grader
records an outcome through :func:`record_official_grade`.

Pi integration contract (implemented in the sibling ``pi_harness.py``):

* ``setup_pi_container(container_name) -> None`` prepares the already-started
  task container.
* ``run_pi(container_name, prompt, base_url, model, session_id, timeout_s,
  out_path) -> result`` runs Pi and returns ``stdout``, ``returncode``,
  ``termination_reason``, ``action_count``, ``model_turn_count``,
  ``repeated_action_count`` and ``repeated_action_signature`` attributes.

The runner owns container lifecycle, the shared prompt/image, timing, patch
collection, checkpointing, and the outer cell manifest. It never invokes the
official grader during capture.
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
from typing import Any, Callable

W3 = Path(__file__).resolve().parent
ND = W3.parent
REPO_ROOT = W3.parents[2]
sys.path.insert(0, str(W3))

OUT_ROOT = W3 / "harness_ab_pi_v1"
PROTOCOL_VERSION = "harness-ab-pi-v1"
MAX_ACTIVE_SECONDS = 1200
# This is frozen for this A/B protocol so a later legacy-runner change cannot
# silently alter the trial fingerprint or one side of the comparison.
MAX_IDENTICAL_ACTION_REPEATS = 4

GYM_BATCH2 = ND / "w2_preflight/gym_batch2_14tasks_instances.json"
TRAIN50_BATCH1 = ND / "w2_preflight/train50_5tasks_instances.json"

# The only task inputs in this protocol. Paths are relative to new_datasets
# when written to metadata, keeping records portable within the repository.
TARGETS = (
    (GYM_BATCH2, "conan-io__conan-13788"),
    (TRAIN50_BATCH1, "iterative__dvc-5336"),
)

BACKENDS = {
    "edge": {"base_url": "http://host.docker.internal:18012", "model": "local"},
    "cloud": {"base_url": "http://host.docker.internal:18011", "model": "deepseek-v4-flash"},
}
HARNESSES = ("claude-code", "pi")
STREAM_FILENAMES = {"claude-code": "claude_stream.jsonl", "pi": "pi_stream.jsonl"}


class HarnessProtocolError(RuntimeError):
    """A harness returned data that does not satisfy the frozen API contract."""


def _capture_api() -> Any:
    """Load the legacy Docker/SWE-bench adapter only when a capture is run.

    Keeping this lazy lets protocol, manifest, and resume tests run without
    installing the legacy adapter's Docker/SWE-bench optional dependencies.
    """
    return importlib.import_module("run_smoke_capture")


@dataclass(frozen=True)
class PlannedCell:
    instance: dict[str, Any]
    source_instances_file: Path
    backend: str
    harness: str
    image: str
    prompt: str
    execution_order: dict[str, Any]

    @property
    def instance_id(self) -> str:
        return str(self.instance["instance_id"])

    @property
    def cell_id(self) -> str:
        return f"{self.instance_id}__{self.backend}__{self.harness}"


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


def _load_selected_instances() -> list[tuple[Path, dict[str, Any]]]:
    cache: dict[Path, dict[str, dict[str, Any]]] = {}
    selected: list[tuple[Path, dict[str, Any]]] = []
    for source_file, instance_id in TARGETS:
        if source_file not in cache:
            rows = _capture_api().load_swebench_dataset(str(source_file))
            cache[source_file] = {str(row["instance_id"]): row for row in rows}
        try:
            instance = cache[source_file][instance_id]
        except KeyError as exc:
            raise RuntimeError(f"Frozen task {instance_id} is absent from {source_file}") from exc
        selected.append((source_file, instance))
    if len(selected) != 2 or len({row[1]["instance_id"] for row in selected}) != 2:
        raise RuntimeError("The frozen Pi A/B protocol must contain exactly two unique tasks")
    return selected


def _execution_order(instance_id: str, backend: str) -> tuple[list[str], dict[str, Any]]:
    seed_material = f"{PROTOCOL_VERSION}|harness-order|{instance_id}|{backend}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    order = list(HARNESSES)
    random.Random(seed).shuffle(order)
    return order, {"seed": seed, "harnesses_in_order": order}


def build_plan() -> tuple[list[PlannedCell], dict[str, Any], str]:
    """Load frozen inputs and return the ordered cells plus fingerprint data."""
    selected = _load_selected_instances()
    prompt_cache: dict[str, str] = {}
    task_docs: list[dict[str, Any]] = []
    plan: list[PlannedCell] = []

    for source_file, instance in selected:
        instance_id = str(instance["instance_id"])
        prompt = _capture_api().render_prompt(instance)
        prompt_cache[instance_id] = prompt
        test_spec = _capture_api().make_test_spec(instance)
        image = f"sweb.eval.{test_spec.arch}.{instance_id}:latest"
        task_doc = {
            "instance_id": instance_id,
            "source_instances_file": _relative_to_nd(source_file),
            "source_sha256": _sha256_file(source_file),
            "image": image,
            "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        }
        task_docs.append(task_doc)

        for backend, backend_cfg in BACKENDS.items():
            order, order_doc = _execution_order(instance_id, backend)
            for position, harness_name in enumerate(order, start=1):
                plan.append(PlannedCell(
                    instance=instance,
                    source_instances_file=source_file,
                    backend=backend,
                    harness=harness_name,
                    image=image,
                    prompt=prompt_cache[instance_id],
                    execution_order={**order_doc, "position": position},
                ))

    if len(plan) != 8 or len({cell.cell_id for cell in plan}) != 8:
        raise RuntimeError("Frozen Pi A/B plan must resolve to exactly eight unique cells")

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "tasks": task_docs,
        "backends": BACKENDS,
        "harnesses": list(HARNESSES),
        "max_active_seconds": MAX_ACTIVE_SECONDS,
        "identical_action_repeat_limit": MAX_IDENTICAL_ACTION_REPEATS,
        "container_platform": "linux/arm64/v8",
        "official_grading": "separate_post_capture_phase",
    }
    fingerprint = _sha256_bytes(_canonical_json(protocol))
    return plan, protocol, fingerprint


def _atomic_write_json(path: Path, value: Any) -> None:
    """Atomically update checkpoint metadata without touching run artifacts."""
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
            raise HarnessProtocolError(
                f"Output root {out_root} belongs to protocol fingerprint "
                f"{prior.get('protocol_fingerprint')}, expected {fingerprint}; refusing to mix runs"
            )
        return
    if any(out_root.iterdir()):
        raise HarnessProtocolError(f"Output root {out_root} is nonempty but has no protocol.json; refusing overwrite")
    _create_json_exclusive(protocol_path, {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "protocol": protocol,
        "created_at_utc": _utc_now(),
    })


def _cell_identity(cell: PlannedCell, fingerprint: str) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "instance_id": cell.instance_id,
        "source_instances_file": _relative_to_nd(cell.source_instances_file),
        "backend": cell.backend,
        "harness": cell.harness,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "execution_order": cell.execution_order,
        "image": cell.image,
        "prompt_sha256": _sha256_bytes(cell.prompt.encode("utf-8")),
        "max_active_seconds": MAX_ACTIVE_SECONDS,
    }


def ensure_cell_meta(out_root: Path, cell: PlannedCell, fingerprint: str) -> Path:
    cell_dir = out_root / cell.cell_id
    meta_path = cell_dir / "run_meta.json"
    if not cell_dir.exists():
        cell_dir.mkdir(parents=True, exist_ok=False)
        meta = {
            **_cell_identity(cell, fingerprint),
            "status": "pending",
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
        raise HarnessProtocolError(f"Cell directory exists without run_meta.json: {cell_dir}; refusing overwrite")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = _cell_identity(cell, fingerprint)
    for key, expected_value in expected.items():
        if meta.get(key) != expected_value:
            raise HarnessProtocolError(f"Cell metadata mismatch for {cell.cell_id}: {key}")
    if meta.get("status") == "completed" and meta.get("official_grade_status") not in {"not_run", "graded"}:
        raise HarnessProtocolError(f"Unexpected grading status for completed cell {cell.cell_id}")
    return meta_path


def _write_manifest(out_root: Path) -> dict[str, Any]:
    protocol_path = out_root / "protocol.json"
    protocol_doc = json.loads(protocol_path.read_text(encoding="utf-8"))
    cells: list[dict[str, Any]] = []
    for meta_path in sorted(out_root.glob("*/run_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        latest_attempt = meta.get("attempts", [])[-1] if meta.get("attempts") else {}
        # Cell metadata owns grading state; attempt metadata owns execution
        # details. This merge order preserves a post-capture official grade
        # instead of letting an older attempt's null grade shadow it.
        cells.append({**latest_attempt, **meta, "attempts": meta.get("attempts", [])})
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_doc["protocol_fingerprint"],
        "official_grade_status": "complete" if all_official_grades_recorded(cells) else "pending",
        "cells": cells,
        "updated_at_utc": _utc_now(),
    }
    _atomic_write_json(out_root / "capture_manifest.json", manifest)
    return manifest


def all_official_grades_recorded(cells: list[dict[str, Any]]) -> bool:
    """True only once every one of the eight captured cells has a grade."""
    return (
        len(cells) == 8
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
    """Record a separately-run official grade exactly once for a captured cell."""
    if not isinstance(official_pass, bool):
        raise TypeError("official_pass must be bool")
    meta_path = out_root / cell_id / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "completed":
        raise HarnessProtocolError(f"Cannot grade uncaptured cell {cell_id}")
    if meta.get("official_grade_status") != "not_run" or meta.get("official_pass") is not None:
        raise HarnessProtocolError(f"Official grade already recorded for {cell_id}; refusing overwrite")
    meta["official_pass"] = official_pass
    meta["official_grade_status"] = "graded"
    meta["official_grader_reference"] = grader_reference
    meta["official_grade_details"] = grade_details or {}
    meta["official_graded_at_utc"] = _utc_now()
    _atomic_write_json(meta_path, meta)
    _write_manifest(out_root)


def _count_claude_events(stdout: str) -> tuple[int, int, int | None]:
    actions = 0
    turns = 0
    output_tokens = 0
    saw_output_tokens = False
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant" or not isinstance(event.get("message"), dict):
            continue
        turns += 1
        usage = event["message"].get("usage")
        if isinstance(usage, dict):
            count = usage.get("output_tokens")
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                output_tokens += count
                saw_output_tokens = True
        content = event["message"].get("content")
        if isinstance(content, list):
            actions += sum(
                1 for block in content
                if isinstance(block, dict) and block.get("type") == "tool_use"
            )
    return actions, turns, output_tokens if saw_output_tokens else None


def _required_result_attr(result: Any, name: str) -> Any:
    if not hasattr(result, name):
        raise HarnessProtocolError(f"Harness result lacks required field {name!r}")
    return getattr(result, name)


def _normalized_result(harness_name: str, result: Any) -> dict[str, Any]:
    stdout = _required_result_attr(result, "stdout")
    returncode = _required_result_attr(result, "returncode")
    termination_reason = _required_result_attr(result, "termination_reason")
    if not isinstance(stdout, str) or not isinstance(termination_reason, str):
        raise HarnessProtocolError("Harness stdout and termination_reason must be strings")
    if harness_name == "claude-code":
        action_count, model_turn_count, output_tokens = _count_claude_events(stdout)
        repeated_count = int(getattr(result, "repeated_action_count", 0))
        repeated_signature = getattr(result, "repeated_action_signature", None)
    else:
        action_count = _required_result_attr(result, "action_count")
        model_turn_count = _required_result_attr(result, "model_turn_count")
        repeated_count = _required_result_attr(result, "repeated_action_count")
        repeated_signature = _required_result_attr(result, "repeated_action_signature")
        output_tokens = getattr(result, "output_tokens", None)
    for field_name, value in (
        ("action_count", action_count),
        ("model_turn_count", model_turn_count),
        ("repeated_action_count", repeated_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise HarnessProtocolError(f"{field_name} must be a nonnegative integer")
    if repeated_signature is not None and not isinstance(repeated_signature, str):
        raise HarnessProtocolError("repeated_action_signature must be str or None")
    if output_tokens is not None and (
        not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 0
    ):
        raise HarnessProtocolError("output_tokens must be a nonnegative integer or None")
    return {
        "stdout": stdout,
        "returncode": returncode,
        "termination_reason": termination_reason,
        "timed_out": termination_reason == "trajectory_deadline",
        "action_count": action_count,
        "model_turn_count": model_turn_count,
        "repeated_action_count": repeated_count,
        "repeated_action_signature": repeated_signature,
        "output_tokens": output_tokens,
    }


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


def _run_harness(cell: PlannedCell, container_name: str, session_id: str, stream_path: Path) -> Any:
    backend = BACKENDS[cell.backend]
    if cell.harness == "claude-code":
        return _capture_api().run_claude(
            container_name,
            cell.prompt,
            backend["base_url"],
            backend["model"],
            session_id,
            MAX_ACTIVE_SECONDS,
            stream_path,
        )
    pi_harness = importlib.import_module("pi_harness")
    return pi_harness.run_pi(
        container_name,
        cell.prompt,
        backend["base_url"],
        backend["model"],
        session_id,
        MAX_ACTIVE_SECONDS,
        stream_path,
    )


def _setup_harness(harness_name: str, container_name: str) -> None:
    if harness_name == "claude-code":
        _capture_api().setup_container(container_name)
        return
    pi_harness = importlib.import_module("pi_harness")
    pi_harness.setup_pi_container(container_name)


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
    container_name = (
        f"pi-ab-{cell.instance_id.replace('/', '-').replace('_', '-')}-"
        f"{cell.backend}-{cell.harness.replace('-', '')}-{attempt_number:03d}"
    ).lower()
    stream_path = attempt_dir / STREAM_FILENAMES[cell.harness]
    prompt_path = attempt_dir / "prompt.txt"
    patch_path = attempt_dir / "final_patch.diff"
    with prompt_path.open("x", encoding="utf-8") as stream:
        stream.write(cell.prompt)

    container_started = False
    setup_seconds = 0.0
    run_started = 0.0
    protocol_failure = False
    try:
        image = cell.image
        api = _capture_api()
        api.start_container(image, container_name)
        container_started = True
        setup_started = time.monotonic()
        _setup_harness(cell.harness, container_name)
        setup_seconds = time.monotonic() - setup_started

        run_started = time.monotonic()
        try:
            raw_result = _run_harness(cell, container_name, session_id, stream_path)
            result = _normalized_result(cell.harness, raw_result)
        except HarnessProtocolError:
            protocol_failure = True
            raise
        wall_seconds = time.monotonic() - run_started
        patch = api.extract_patch(container_name)
        with patch_path.open("x", encoding="utf-8") as stream:
            stream.write(patch)
        result_record = {
            **_cell_identity(cell, fingerprint),
            "session_id": session_id,
            "status": "completed",
            "max_active_seconds": MAX_ACTIVE_SECONDS,
            "setup_seconds": round(setup_seconds, 3),
            "wall_seconds": round(wall_seconds, 3),
            "returncode": result["returncode"],
            "termination_reason": result["termination_reason"],
            "timed_out": result["timed_out"],
            "action_count": result["action_count"],
            "model_turn_count": result["model_turn_count"],
            "repeated_action_count": result["repeated_action_count"],
            "repeated_action_signature": result["repeated_action_signature"],
            "output_tokens": result["output_tokens"],
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
    except HarnessProtocolError as exc:
        failure_kind = "protocol"
        _fail_attempt(meta_path, attempt_number, _safe_error(exc), setup_seconds, run_started, failure_kind)
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # recorded so a later invocation can append a fresh attempt
        _fail_attempt(meta_path, attempt_number, _safe_error(exc), setup_seconds, run_started, "infrastructure")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    finally:
        if container_started:
            try:
                _capture_api().sh(["docker", "rm", "-f", container_name])
            except Exception:
                # The capture outcome is already checkpointed. Container
                # cleanup can be retried by the operator without erasing it.
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
    latest.update({
        "status": "failed",
        "ended_at_utc": _utc_now(),
        "setup_seconds": round(setup_seconds, 3),
        "wall_seconds": round(elapsed, 3),
        "termination_reason": "protocol_error" if protocol_failure else "infrastructure_error",
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
        "infrastructure_failure": not protocol_failure,
        "protocol_failure": protocol_failure,
    })
    attempts[-1] = latest
    meta.update({
        "status": "failed",
        "attempts": attempts,
        "active_attempt": attempt_number,
        "setup_seconds": round(setup_seconds, 3),
        "wall_seconds": round(elapsed, 3),
        "termination_reason": latest["termination_reason"],
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
        "infrastructure_failure": not protocol_failure,
        "protocol_failure": protocol_failure,
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
    counts: dict[str, int] = {}
    for cell in manifest["cells"]:
        counts[cell.get("status", "unknown")] = counts.get(cell.get("status", "unknown"), 0) + 1
    print(json.dumps({"captured_cells": counts, "official_grade_status": manifest["official_grade_status"]}, indent=2))


if __name__ == "__main__":
    main()
