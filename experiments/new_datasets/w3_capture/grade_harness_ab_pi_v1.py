"""Official, resumable SWE-Bench-Fork grading for the Pi/Claude A/B trial.

The grading call matches ``grade_gym_batch2.py`` and
``grade_train50_batch1.py``. It reads the frozen task row and latest patch
from each completed capture cell, runs each cell at most once, stores reports
inside that capture attempt, then records resolved status through the
capture runner's one-time grading hook.

This module does not run as an import side effect. Invoke it explicitly after
all eight capture cells have completed.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

W3 = Path(__file__).resolve().parent
ND = W3.parent
VENDOR = ND / "w2_preflight/_vendor/SWE-Bench-Fork"
OUT_ROOT = W3 / "harness_ab_pi_v1"
RUN_ID_PREFIX = "harness-ab-pi-v1-grade"
GRADE_TIMEOUT_SECONDS = 1800

sys.path.insert(0, str(W3))
import run_harness_ab_pi_v1 as capture_runner  # noqa: E402

SOURCE_BY_INSTANCE = {
    instance_id: source_file
    for source_file, instance_id in capture_runner.TARGETS
}
EXPECTED_CELL_IDS = {
    f"{instance_id}__{backend}__{harness}"
    for instance_id in SOURCE_BY_INSTANCE
    for backend in capture_runner.BACKENDS
    for harness in capture_runner.HARNESSES
}


class GradingProtocolError(RuntimeError):
    """Frozen capture metadata or grading checkpoint is inconsistent."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_new_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # Same-directory hard-link creation is atomic and fails if the final
        # report already exists, so a second grading attempt cannot replace it.
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=True)


def _load_grading_api() -> Any:
    """Import pinned vendored SWE-Bench-Fork code only for an actual grade."""
    vendor = str(VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    docker = importlib.import_module("docker")
    run_evaluation = importlib.import_module("swebench.harness.run_evaluation")
    test_spec = importlib.import_module("swebench.harness.test_spec")
    return type("GradingApi", (), {
        "run_instance": staticmethod(run_evaluation.run_instance),
        "make_test_spec": staticmethod(test_spec.make_test_spec),
        "docker_from_env": staticmethod(docker.from_env),
    })()


def _read_instances(source_file: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(source_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise GradingProtocolError(f"Expected a JSON list of task rows in {source_file}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("instance_id"), str):
            result[row["instance_id"]] = row
    return result


def _load_and_validate_cells(capture_root: Path) -> tuple[str, list[tuple[Path, dict[str, Any]]]]:
    protocol_path = capture_root / "protocol.json"
    if not protocol_path.is_file():
        raise GradingProtocolError(f"Missing protocol checkpoint: {protocol_path}")
    protocol_doc = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol_doc.get("protocol_version") != capture_runner.PROTOCOL_VERSION:
        raise GradingProtocolError("Capture root protocol version does not match the frozen A/B protocol")
    fingerprint = protocol_doc.get("protocol_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise GradingProtocolError("Capture root has no protocol fingerprint")
    protocol = protocol_doc.get("protocol")
    if not isinstance(protocol, dict):
        raise GradingProtocolError("Capture root has no frozen protocol body")
    calculated_fingerprint = capture_runner._sha256_bytes(capture_runner._canonical_json(protocol))
    if calculated_fingerprint != fingerprint:
        raise GradingProtocolError("Capture root protocol body does not match its fingerprint")
    if protocol.get("protocol_version") != capture_runner.PROTOCOL_VERSION:
        raise GradingProtocolError("Frozen protocol body has the wrong version")
    if (
        protocol.get("backends") != capture_runner.BACKENDS
        or protocol.get("harnesses") != list(capture_runner.HARNESSES)
        or protocol.get("max_active_seconds") != capture_runner.MAX_ACTIVE_SECONDS
        or protocol.get("identical_action_repeat_limit") != capture_runner.MAX_IDENTICAL_ACTION_REPEATS
        or protocol.get("container_platform") != "linux/arm64/v8"
        or protocol.get("official_grading") != "separate_post_capture_phase"
    ):
        raise GradingProtocolError("Frozen protocol settings do not match the Pi A/B grading contract")
    protocol_tasks = protocol.get("tasks")
    if not isinstance(protocol_tasks, list):
        raise GradingProtocolError("Frozen protocol task list is missing")
    tasks_by_id = {
        item.get("instance_id"): item for item in protocol_tasks
        if isinstance(item, dict) and isinstance(item.get("instance_id"), str)
    }
    if set(tasks_by_id) != set(SOURCE_BY_INSTANCE):
        raise GradingProtocolError("Frozen protocol tasks do not match the two-task A/B cohort")
    for instance_id, source_file in SOURCE_BY_INSTANCE.items():
        task_doc = tasks_by_id[instance_id]
        expected_source_rel = source_file.resolve().relative_to(ND.resolve()).as_posix()
        if task_doc.get("source_instances_file") != expected_source_rel:
            raise GradingProtocolError(f"Frozen source path mismatch for {instance_id}")
        if task_doc.get("source_sha256") != _sha256_file(source_file):
            raise GradingProtocolError(f"Frozen source file has changed since capture for {instance_id}")

    metas: list[tuple[Path, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for meta_path in sorted(capture_root.glob("*/run_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cell_id = meta.get("cell_id")
        if not isinstance(cell_id, str) or meta_path.parent.name != cell_id:
            raise GradingProtocolError(f"Cell directory and metadata identity differ: {meta_path}")
        if cell_id in seen_ids:
            raise GradingProtocolError(f"Duplicate cell metadata for {cell_id}")
        seen_ids.add(cell_id)
        if cell_id not in EXPECTED_CELL_IDS:
            raise GradingProtocolError(f"Unexpected cell in frozen A/B output: {cell_id}")
        if meta.get("protocol_version") != capture_runner.PROTOCOL_VERSION:
            raise GradingProtocolError(f"Wrong protocol version in {cell_id}")
        if meta.get("protocol_fingerprint") != fingerprint:
            raise GradingProtocolError(f"Protocol fingerprint mismatch in {cell_id}")
        if meta.get("status") != "completed":
            raise GradingProtocolError(f"Refusing to grade non-completed cell {cell_id}: {meta.get('status')}")

        instance_id = meta.get("instance_id")
        expected_source = SOURCE_BY_INSTANCE.get(instance_id)
        if expected_source is None:
            raise GradingProtocolError(f"Task is outside the frozen two-task cohort: {instance_id}")
        expected_source_rel = expected_source.resolve().relative_to(ND.resolve()).as_posix()
        if meta.get("source_instances_file") != expected_source_rel:
            raise GradingProtocolError(f"Wrong source task file in {cell_id}")
        if meta.get("backend") not in capture_runner.BACKENDS or meta.get("harness") not in capture_runner.HARNESSES:
            raise GradingProtocolError(f"Invalid backend/harness in {cell_id}")
        expected_cell_id = f"{instance_id}__{meta['backend']}__{meta['harness']}"
        if cell_id != expected_cell_id:
            raise GradingProtocolError(f"Cell identifier does not match task/backend/harness: {cell_id}")

        attempts = meta.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise GradingProtocolError(f"Cell has no completed capture attempt: {cell_id}")
        latest = attempts[-1]
        if latest.get("status") != "completed":
            raise GradingProtocolError(f"Latest capture attempt is not completed: {cell_id}")
        if not isinstance(latest.get("session_id"), str) or meta.get("session_id") != latest.get("session_id"):
            raise GradingProtocolError(f"Cell session ID does not identify its latest attempt: {cell_id}")
        attempt_dir_name = latest.get("attempt_dir")
        if not isinstance(attempt_dir_name, str) or not re.fullmatch(r"attempt-[0-9]{3,}", attempt_dir_name):
            raise GradingProtocolError(f"Invalid latest attempt directory in {cell_id}")
        attempt_dir = meta_path.parent / attempt_dir_name
        if not attempt_dir.is_dir():
            raise GradingProtocolError(f"Latest attempt directory is missing in {cell_id}")
        patch_ref = latest.get("patch_path") or meta.get("patch_path")
        if not isinstance(patch_ref, str) or Path(patch_ref).is_absolute():
            raise GradingProtocolError(f"Missing or absolute latest patch path in {cell_id}")
        patch_path = (capture_root / patch_ref).resolve()
        if attempt_dir.resolve() not in patch_path.parents or not patch_path.is_file():
            raise GradingProtocolError(f"Latest patch is missing or outside its attempt in {cell_id}")
        if meta.get("patch_path") and meta.get("patch_path") != patch_ref:
            raise GradingProtocolError(f"Cell patch path does not identify its latest attempt: {cell_id}")
        metas.append((meta_path, meta))

    missing = EXPECTED_CELL_IDS - seen_ids
    if missing:
        raise GradingProtocolError(f"Frozen A/B capture is missing cell metadata: {sorted(missing)}")
    if len(metas) != 8:
        raise GradingProtocolError(f"Expected exactly eight frozen cells, found {len(metas)}")
    return fingerprint, metas


def _cell_attempt_paths(capture_root: Path, meta: dict[str, Any]) -> tuple[Path, Path]:
    latest = meta["attempts"][-1]
    attempt_dir = capture_root / meta["cell_id"] / latest["attempt_dir"]
    patch_ref = latest.get("patch_path") or meta["patch_path"]
    return attempt_dir, (capture_root / patch_ref).resolve()


def _new_run_id(meta: dict[str, Any], attempt_number: int) -> str:
    safe_cell = re.sub(r"[^A-Za-z0-9-]+", "-", meta["cell_id"]).strip("-")
    return f"{RUN_ID_PREFIX}-{safe_cell}-a{attempt_number}-{uuid.uuid4().hex[:12]}"


def _reference_from_root(capture_root: Path, report_path: Path) -> str:
    return report_path.resolve().relative_to(capture_root.resolve()).as_posix()


def _record_report_if_complete(
    capture_root: Path,
    meta_path: Path,
    meta: dict[str, Any],
    checkpoint: dict[str, Any],
    report_path: Path,
) -> bool:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol_version") != capture_runner.PROTOCOL_VERSION:
        raise GradingProtocolError(f"Official grade report has the wrong protocol version: {report_path}")
    if report.get("protocol_fingerprint") != meta.get("protocol_fingerprint"):
        raise GradingProtocolError(f"Official grade report fingerprint mismatch: {report_path}")
    if report.get("cell_id") != meta.get("cell_id"):
        raise GradingProtocolError(f"Official grade report belongs to another cell: {report_path}")
    if report.get("grading_run_id") != checkpoint.get("run_id"):
        raise GradingProtocolError(f"Official grade report run ID mismatch: {report_path}")
    if report.get("status") != "complete" or not isinstance(report.get("resolved"), bool):
        return False
    if meta.get("official_grade_status") == "graded":
        return True
    if meta.get("official_grade_status") != "not_run" or meta.get("official_pass") is not None:
        raise GradingProtocolError(f"Unexpected existing official grade state for {meta['cell_id']}")
    details = {
        "grading_run_id": checkpoint["run_id"],
        "grade_seconds": report.get("grade_seconds"),
        "report_path": _reference_from_root(capture_root, report_path),
        "report_sha256": _sha256_file(report_path),
        "patch_sha256": report.get("patch_sha256"),
    }
    capture_runner.record_official_grade(
        capture_root,
        meta["cell_id"],
        report["resolved"],
        _reference_from_root(capture_root, report_path),
        details,
    )
    return True


def _grade_one(
    capture_root: Path,
    meta_path: Path,
    meta: dict[str, Any],
    instance: dict[str, Any],
    api: Any,
    client: Any,
) -> str:
    attempt_dir, patch_path = _cell_attempt_paths(capture_root, meta)
    checkpoint_path = attempt_dir / "official_grade_checkpoint.json"

    if meta.get("official_grade_status") == "graded":
        return "already_graded"
    if meta.get("official_grade_status") != "not_run" or meta.get("official_pass") is not None:
        raise GradingProtocolError(f"Unexpected official grade state in {meta['cell_id']}")

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("protocol_version") != capture_runner.PROTOCOL_VERSION:
            raise GradingProtocolError(f"Wrong grading checkpoint protocol version for {meta['cell_id']}")
        if checkpoint.get("protocol_fingerprint") != meta.get("protocol_fingerprint"):
            raise GradingProtocolError(f"Grading checkpoint fingerprint mismatch for {meta['cell_id']}")
        if checkpoint.get("cell_id") != meta["cell_id"]:
            raise GradingProtocolError(f"Grading checkpoint belongs to another cell: {meta['cell_id']}")
        report_rel = checkpoint.get("report_path")
        if not isinstance(report_rel, str) or Path(report_rel).is_absolute():
            raise GradingProtocolError(f"Invalid report path in grading checkpoint for {meta['cell_id']}")
        report_path = (capture_root / report_rel).resolve()
        if attempt_dir.resolve() not in report_path.parents:
            raise GradingProtocolError(f"Grading report path escaped its attempt directory for {meta['cell_id']}")

        if not report_path.exists() and checkpoint.get("status") == "running":
            # A killed grader may have finished Docker work without persisting
            # its result. Do not repeat the official run: exactly-once takes
            # precedence over guessing the missing outcome.
            return "uncertain_interrupted"
        if report_path.exists():
            recovered = _record_report_if_complete(capture_root, meta_path, meta, checkpoint, report_path)
            if recovered:
                return "recovered_grade" if meta.get("official_grade_status") != "graded" else "already_graded"
            return str(json.loads(report_path.read_text(encoding="utf-8")).get("status", "unscorable"))
        checkpoint_status = checkpoint.get("status")
        if checkpoint_status == "running":
            return "uncertain_interrupted"
        if checkpoint_status in {"complete", "unscorable", "failed"}:
            raise GradingProtocolError(f"Grading checkpoint report is missing for {meta['cell_id']}")
        raise GradingProtocolError(f"Unknown grading checkpoint state for {meta['cell_id']}: {checkpoint_status}")

    report_path = attempt_dir / f"official_grade_{uuid.uuid4().hex[:12]}.json"
    run_id = _new_run_id(meta, int(meta["attempts"][-1]["attempt_number"]))
    checkpoint = {
        "protocol_version": capture_runner.PROTOCOL_VERSION,
        "protocol_fingerprint": meta["protocol_fingerprint"],
        "cell_id": meta["cell_id"],
        "capture_attempt": meta["attempts"][-1]["attempt_number"],
        "status": "running",
        "run_id": run_id,
        "report_path": _reference_from_root(capture_root, report_path),
        "started_at_utc": _now_utc(),
    }
    _write_new_json(checkpoint_path, checkpoint)

    pred = {
        "instance_id": meta["instance_id"],
        "model_name_or_path": f"harness-ab-pi-v1-{meta['backend']}-{meta['harness']}",
        "model_patch": patch_path.read_text(encoding="utf-8"),
    }
    result_payload: dict[str, Any]
    started = time.monotonic()
    try:
        spec = api.make_test_spec(instance)
        result = api.run_instance(
            spec,
            pred,
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id=run_id,
            timeout=GRADE_TIMEOUT_SECONDS,
        )
        elapsed = round(time.monotonic() - started, 2)
        report: dict[str, Any] = {}
        if result is not None and isinstance(result, (tuple, list)) and len(result) >= 2:
            reports = result[1]
            if isinstance(reports, dict) and isinstance(reports.get(meta["instance_id"]), dict):
                report = reports[meta["instance_id"]]
        resolved = report.get("resolved")
        valid_grade = isinstance(resolved, bool)
        result_payload = {
            "protocol_version": capture_runner.PROTOCOL_VERSION,
            "protocol_fingerprint": meta["protocol_fingerprint"],
            "cell_id": meta["cell_id"],
            "instance_id": meta["instance_id"],
            "backend": meta["backend"],
            "harness": meta["harness"],
            "capture_attempt": meta["attempts"][-1]["attempt_number"],
            "capture_session_id": meta.get("session_id"),
            "grading_run_id": run_id,
            "status": "complete" if valid_grade else "unscorable",
            "resolved": resolved if valid_grade else None,
            "report": report,
            "patch_path": _reference_from_root(capture_root, patch_path),
            "patch_sha256": _sha256_file(patch_path),
            "patch_bytes": patch_path.stat().st_size,
            "grade_seconds": elapsed,
            "timestamp_utc": _now_utc(),
        }
        _write_new_json(report_path, result_payload)
        checkpoint.update({
            "status": "complete" if valid_grade else "unscorable",
            "resolved": resolved if valid_grade else None,
            "ended_at_utc": _now_utc(),
            "grade_seconds": elapsed,
        })
        _atomic_json(checkpoint_path, checkpoint)
        if not valid_grade:
            return "unscorable"

        _record_report_if_complete(capture_root, meta_path, meta, checkpoint, report_path)
        return "graded"
    except Exception as exc:
        # Persist the failure and never invoke run_instance again on resume.
        # That keeps official grader calls at-most-once even after errors.
        elapsed = round(time.monotonic() - started, 2)
        result_payload = {
            "protocol_version": capture_runner.PROTOCOL_VERSION,
            "protocol_fingerprint": meta["protocol_fingerprint"],
            "cell_id": meta["cell_id"],
            "instance_id": meta["instance_id"],
            "backend": meta["backend"],
            "harness": meta["harness"],
            "capture_attempt": meta["attempts"][-1]["attempt_number"],
            "grading_run_id": run_id,
            "status": "failed",
            "resolved": None,
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "patch_path": _reference_from_root(capture_root, patch_path),
            "patch_sha256": _sha256_file(patch_path),
            "patch_bytes": patch_path.stat().st_size,
            "grade_seconds": elapsed,
            "timestamp_utc": _now_utc(),
        }
        if not report_path.exists():
            _write_new_json(report_path, result_payload)
        checkpoint.update({"status": "failed", "ended_at_utc": _now_utc(), "grade_seconds": elapsed})
        _atomic_json(checkpoint_path, checkpoint)
        return "failed"


def grade_all(
    capture_root: Path = OUT_ROOT,
    *,
    api: Any | None = None,
    docker_client: Any | None = None,
) -> dict[str, Any]:
    fingerprint, metas = _load_and_validate_cells(capture_root)
    grading_api = api or _load_grading_api()
    # Do not initialize Docker if all cells were already graded.
    needs_docker = any(
        meta.get("official_grade_status") != "graded"
        and not (
            capture_root / meta["cell_id"] / meta["attempts"][-1]["attempt_dir"]
            / "official_grade_checkpoint.json"
        ).exists()
        for _, meta in metas
    )
    client = docker_client
    if needs_docker and client is None:
        client = grading_api.docker_from_env()

    instance_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    results: list[dict[str, str]] = []
    for meta_path, meta in metas:
        cell_id = meta["cell_id"]
        if meta.get("official_grade_status") == "graded":
            results.append({"cell_id": cell_id, "status": "already_graded"})
            continue
        source_file = SOURCE_BY_INSTANCE[meta["instance_id"]]
        if source_file not in instance_cache:
            instance_cache[source_file] = _read_instances(source_file)
        instance = instance_cache[source_file].get(meta["instance_id"])
        if instance is None:
            raise GradingProtocolError(f"Frozen task row is missing from {source_file}: {meta['instance_id']}")
        status = _grade_one(capture_root, meta_path, meta, instance, grading_api, client)
        results.append({"cell_id": cell_id, "status": status})

    return {
        "protocol_version": capture_runner.PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "cells": results,
        "graded_cells": sum(item["status"] in {"graded", "recovered_grade", "already_graded"} for item in results),
        "pending_or_failed_cells": [item["cell_id"] for item in results if item["status"] not in {"graded", "recovered_grade", "already_graded"}],
    }


def main() -> None:
    result = grade_all()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
