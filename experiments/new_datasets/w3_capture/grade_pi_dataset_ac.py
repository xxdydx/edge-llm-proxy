"""Official, resumable SWE-Bench-Fork grader for pi-dataset-ac-v1.

Grades all completed cells from experiments/new_datasets/w3_capture/pi_dataset_ac_v1.
Correctly selects source instance rows across:
  - smoke_instances.json
  - train50_5tasks_instances.json
  - gym_batch2_14tasks_instances.json

Safe against interruption: tracks at-most-once execution via attempt-scoped
checkpoints, skipping already-graded cells and recovering completed reports.
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
OUT_ROOT = W3 / "pi_dataset_ac_v1"
RUN_ID_PREFIX = "pi-ac-v1-grade"
GRADE_TIMEOUT_SECONDS = 1800

sys.path.insert(0, str(W3))
import run_pi_dataset_ac as runner  # noqa: E402


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
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=True)


def _load_grading_api() -> Any:
    """Import pinned vendored SWE-Bench-Fork code only when grading is executed."""
    vendor = str(VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    docker = importlib.import_module("docker")
    run_evaluation = importlib.import_module("swebench.harness.run_evaluation")
    test_spec = importlib.import_module("swebench.harness.test_spec")
    return type(
        "GradingApi",
        (),
        {
            "run_instance": staticmethod(run_evaluation.run_instance),
            "make_test_spec": staticmethod(test_spec.make_test_spec),
            "docker_from_env": staticmethod(docker.from_env),
        },
    )()


def _read_instances(source_file: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(source_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise GradingProtocolError(f"Expected a JSON list of task rows in {source_file}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("instance_id"), str):
            result[row["instance_id"]] = row
    return result


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
    if report.get("protocol_version") != runner.PROTOCOL_VERSION:
        raise GradingProtocolError(
            f"Official grade report has wrong protocol version: {report_path}"
        )
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
    runner.record_official_grade(
        capture_root,
        meta["cell_id"],
        report["resolved"],
        _reference_from_root(capture_root, report_path),
        details,
    )
    return True


def _grade_one_completed_cell(
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
        if checkpoint.get("protocol_version") != runner.PROTOCOL_VERSION:
            raise GradingProtocolError(
                f"Wrong grading checkpoint protocol version for {meta['cell_id']}"
            )
        if checkpoint.get("protocol_fingerprint") != meta.get("protocol_fingerprint"):
            raise GradingProtocolError(
                f"Grading checkpoint fingerprint mismatch for {meta['cell_id']}"
            )
        if checkpoint.get("cell_id") != meta["cell_id"]:
            raise GradingProtocolError(
                f"Grading checkpoint belongs to another cell: {meta['cell_id']}"
            )
        report_rel = checkpoint.get("report_path")
        if not isinstance(report_rel, str) or Path(report_rel).is_absolute():
            raise GradingProtocolError(
                f"Invalid report path in grading checkpoint for {meta['cell_id']}"
            )
        report_path = (capture_root / report_rel).resolve()
        if attempt_dir.resolve() not in report_path.parents:
            raise GradingProtocolError(
                f"Grading report path escaped its attempt directory for {meta['cell_id']}"
            )

        if not report_path.exists() and checkpoint.get("status") == "running":
            return "uncertain_interrupted"
        if report_path.exists():
            recovered = _record_report_if_complete(
                capture_root, meta_path, meta, checkpoint, report_path
            )
            if recovered:
                return (
                    "recovered_grade"
                    if meta.get("official_grade_status") != "graded"
                    else "already_graded"
                )
            return str(
                json.loads(report_path.read_text(encoding="utf-8")).get("status", "unscorable")
            )
        checkpoint_status = checkpoint.get("status")
        if checkpoint_status == "running":
            return "uncertain_interrupted"
        if checkpoint_status in {"complete", "unscorable", "failed"}:
            raise GradingProtocolError(
                f"Grading checkpoint report is missing for {meta['cell_id']}"
            )
        raise GradingProtocolError(
            f"Unknown grading checkpoint state for {meta['cell_id']}: {checkpoint_status}"
        )

    run_id = _new_run_id(meta, int(meta["attempts"][-1]["attempt_number"]))
    report_path = attempt_dir / f"official_grade_{uuid.uuid4().hex[:12]}.json"
    checkpoint = {
        "protocol_version": runner.PROTOCOL_VERSION,
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
        "model_name_or_path": f"{runner.PROTOCOL_VERSION}-{meta['backend']}",
        "model_patch": patch_path.read_text(encoding="utf-8"),
    }
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
            "protocol_version": runner.PROTOCOL_VERSION,
            "protocol_fingerprint": meta["protocol_fingerprint"],
            "cell_id": meta["cell_id"],
            "instance_id": meta["instance_id"],
            "backend": meta["backend"],
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
        checkpoint.update(
            {
                "status": "complete" if valid_grade else "unscorable",
                "resolved": resolved if valid_grade else None,
                "ended_at_utc": _now_utc(),
                "grade_seconds": elapsed,
            }
        )
        _atomic_json(checkpoint_path, checkpoint)
        if not valid_grade:
            return "unscorable"

        _record_report_if_complete(capture_root, meta_path, meta, checkpoint, report_path)
        return "graded"
    except Exception as exc:
        elapsed = round(time.monotonic() - started, 2)
        result_payload = {
            "protocol_version": runner.PROTOCOL_VERSION,
            "protocol_fingerprint": meta["protocol_fingerprint"],
            "cell_id": meta["cell_id"],
            "instance_id": meta["instance_id"],
            "backend": meta["backend"],
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
        checkpoint.update(
            {"status": "failed", "ended_at_utc": _now_utc(), "grade_seconds": elapsed}
        )
        _atomic_json(checkpoint_path, checkpoint)
        return "failed"


def grade_all_completed(
    capture_root: Path = OUT_ROOT,
    *,
    api: Any | None = None,
    docker_client: Any | None = None,
) -> dict[str, Any]:
    """Grade all completed cells under the capture root with official SWE-Bench-Fork.

    Resumable: only un-graded completed cells are run.
    """
    protocol_path = capture_root / "protocol.json"
    if not protocol_path.is_file():
        raise GradingProtocolError(f"Missing protocol checkpoint: {protocol_path}")
    protocol_doc = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol_doc.get("protocol_version") != runner.PROTOCOL_VERSION:
        raise GradingProtocolError(
            f"Protocol version mismatch: expected {runner.PROTOCOL_VERSION}, "
            f"got {protocol_doc.get('protocol_version')}"
        )
    fingerprint = protocol_doc.get("protocol_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise GradingProtocolError("Capture root has no protocol fingerprint")

    # Build source instance lookup across the 3 preflighted files
    source_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    instance_to_source: dict[str, Path] = {}
    for source_file in runner.SOURCE_FILES:
        rows = _read_instances(source_file)
        source_cache[source_file] = rows
        for iid in rows:
            if iid not in instance_to_source:
                instance_to_source[iid] = source_file

    metas: list[tuple[Path, dict[str, Any]]] = []
    skipped_cells: list[dict[str, str]] = []
    for meta_path in sorted(capture_root.glob("*/run_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cell_id = meta.get("cell_id")
        if not isinstance(cell_id, str) or meta_path.parent.name != cell_id:
            raise GradingProtocolError(
                f"Cell directory and metadata identity differ: {meta_path}"
            )
        if meta.get("protocol_version") != runner.PROTOCOL_VERSION:
            raise GradingProtocolError(f"Wrong protocol version in {cell_id}")
        if meta.get("protocol_fingerprint") != fingerprint:
            raise GradingProtocolError(f"Protocol fingerprint mismatch in {cell_id}")

        if meta.get("status") != "completed":
            skipped_cells.append(
                {"cell_id": cell_id, "status": f"skipped_{meta.get('status', 'unknown')}"}
            )
            continue

        iid = meta.get("instance_id")
        if iid not in instance_to_source:
            raise GradingProtocolError(f"Unknown instance ID in {cell_id}: {iid}")

        attempts = meta.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise GradingProtocolError(f"Completed cell has no attempts: {cell_id}")
        latest = attempts[-1]
        if latest.get("status") != "completed":
            raise GradingProtocolError(f"Latest attempt not completed in {cell_id}")
        patch_ref = latest.get("patch_path") or meta.get("patch_path")
        if not isinstance(patch_ref, str) or Path(patch_ref).is_absolute():
            raise GradingProtocolError(f"Invalid patch path in {cell_id}")
        patch_path = (capture_root / patch_ref).resolve()
        if not patch_path.is_file():
            raise GradingProtocolError(f"Patch file missing for {cell_id}: {patch_path}")

        metas.append((meta_path, meta))

    grading_api = api or _load_grading_api()
    needs_docker = any(
        meta.get("official_grade_status") != "graded"
        and not (
            capture_root
            / meta["cell_id"]
            / meta["attempts"][-1]["attempt_dir"]
            / "official_grade_checkpoint.json"
        ).exists()
        for _, meta in metas
    )
    client = docker_client
    if needs_docker and client is None:
        client = grading_api.docker_from_env()

    results: list[dict[str, str]] = []
    for meta_path, meta in metas:
        cell_id = meta["cell_id"]
        if meta.get("official_grade_status") == "graded":
            results.append({"cell_id": cell_id, "status": "already_graded"})
            continue
        iid = meta["instance_id"]
        source_file = instance_to_source[iid]
        instance = source_cache[source_file][iid]
        status = _grade_one_completed_cell(
            capture_root, meta_path, meta, instance, grading_api, client
        )
        results.append({"cell_id": cell_id, "status": status})

    all_results = results + skipped_cells
    graded_count = sum(
        r["status"] in {"graded", "recovered_grade", "already_graded"} for r in results
    )
    return {
        "protocol_version": runner.PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "total_completed_cells_found": len(metas),
        "graded_cells": graded_count,
        "already_graded_cells": sum(r["status"] == "already_graded" for r in results),
        "results": all_results,
        "pending_or_skipped_cells": [
            r["cell_id"]
            for r in all_results
            if r["status"] not in {"graded", "recovered_grade", "already_graded"}
        ],
    }


def main() -> None:
    result = grade_all_completed()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
