"""Offline operational comparison for the frozen Pi vs Claude Code trial.

This reads capture metadata and optional official grading reports. It never
edits a cell, stream, patch, or manifest. The 8-cell trial is too small for a
statistical claim; the gate below is a preregistered operational decision rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = Path(__file__).resolve().parent / "harness_ab_pi_v1"
TASK_IDS = ("conan-io__conan-13788", "iterative__dvc-5336")
BACKENDS = ("edge", "cloud")
HARNESSES = ("claude-code", "pi")
PROTOCOL_VERSION = "harness-ab-pi-v1"
CENSORED_FAILURES = {
    "trajectory_deadline",
    "repeated_action",
    "step_limit",
    "process_error",
    "infrastructure_error",
    "infrastructure_failure",
    "protocol_error",
    "protocol_failure",
    "setup_error",
    "setup_failure",
    "container_error",
    "container_failure",
    "endpoint_error",
    "endpoint_failure",
}
INFRA_PROTOCOL_FAILURES = {
    "process_error",
    "infrastructure_error",
    "infrastructure_failure",
    "protocol_error",
    "protocol_failure",
    "setup_error",
    "setup_failure",
    "container_error",
    "container_failure",
    "endpoint_error",
    "endpoint_failure",
}
NONTERMINAL = {"", "unknown", "pending", "queued", "running", "in_progress"}


def expected_cells() -> list[dict[str, str]]:
    return [
        {"instance_id": task, "backend": backend, "harness": harness}
        for task in TASK_IDS
        for backend in BACKENDS
        for harness in HARNESSES
    ]


def cell_key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    instance_id = row.get("instance_id") or row.get("task_id")
    backend = str(row.get("backend", "")).lower()
    harness = str(row.get("harness", "")).lower().replace("_", "-")
    if backend in {"edge-only", "edge-only-v1", "local"}:
        backend = "edge"
    elif backend in {"cloud-only", "cloud-only-v1", "remote"}:
        backend = "cloud"
    if harness in {"claude", "claude-code", "claude-code-v1"}:
        harness = "claude-code"
    if instance_id is None or backend not in BACKENDS or harness not in HARNESSES:
        return None
    return str(instance_id), backend, harness


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load latest rows from the root manifest, with per-cell fallback."""
    rows: list[dict[str, Any]] = []
    manifest = path / "capture_manifest.json" if path.is_dir() else path
    root = manifest.parent
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            candidates = payload.get("cells", [])
            if isinstance(candidates, dict):
                candidates = list(candidates.values())
            if isinstance(candidates, list):
                rows.extend(item for item in candidates if isinstance(item, dict))
        elif isinstance(payload, list):
            rows.extend(item for item in payload if isinstance(item, dict))

    # Per-cell metadata is checkpointed before the root manifest is refreshed.
    # Always let that newer atomic record replace a stale manifest row so the
    # live monitor does not report a running cell as pending.
    for cell in expected_cells():
        key = (cell["instance_id"], cell["backend"], cell["harness"])
        cell_dir = root / f"{key[0]}__{key[1]}__{key[2]}"
        meta = cell_dir / "run_meta.json"
        if not meta.exists():
            continue
        try:
            row = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            rows = [existing for existing in rows if cell_key(existing) != key]
            rows.append(row)
    return rows


def _termination_reason(row: dict[str, Any]) -> str:
    value = row.get("termination_reason")
    if value is None:
        value = row.get("claude_termination_reason") or row.get("pi_termination_reason")
    if value is None:
        status = str(row.get("status", "")).lower()
        if status in {"completed", "finished", "done", "graded"}:
            return "completed"
        return status or "unknown"
    return str(value).strip().lower().replace("-", "_")


def _terminal(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    reason = _termination_reason(row)
    status = str(row.get("status", "")).lower()
    return reason not in NONTERMINAL or status in {"completed", "finished", "done", "graded", "failed"}


def _as_bool_grade(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pass", "passed", "true", "resolved", "success", "succeeded"}:
            return True
        if normalized in {"fail", "failed", "false", "unresolved", "failure"}:
            return False
    return None


def _grade_from_report(payload: Any) -> bool | None:
    entries = payload if isinstance(payload, list) else [payload]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in ("official_pass", "official_grade", "grade", "grading_status", "resolved"):
            grade = _as_bool_grade(entry.get(key))
            if grade is not None:
                return grade
        report = entry.get("report")
        if isinstance(report, dict):
            for key in ("official_pass", "official_grade", "grade", "grading_status", "resolved"):
                grade = _as_bool_grade(report.get(key))
                if grade is not None:
                    return grade
    return None


def official_pass(row: dict[str, Any], capture_root: Path) -> bool | None:
    for key in ("official_pass", "official_grade", "grade", "grading_status"):
        grade = _as_bool_grade(row.get(key))
        if grade is not None:
            return grade
    for key in ("grading_report_path", "official_grading_report", "grade_report_path"):
        raw_path = row.get(key)
        if not raw_path:
            continue
        report_path = Path(str(raw_path))
        if not report_path.is_absolute():
            report_path = (capture_root / report_path).resolve()
            if not report_path.exists():
                report_path = (REPO_ROOT / str(raw_path)).resolve()
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        grade = _grade_from_report(payload)
        if grade is not None:
            return grade
    return None


def _number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _descriptive(rows: Iterable[dict[str, Any]], capture_root: Path) -> dict[str, Any]:
    rows = list(rows)
    metrics = {
        "wall_seconds": [v for row in rows if (v := _number(row, "wall_seconds", "claude_wall_seconds", "pi_wall_seconds")) is not None],
        "actions": [v for row in rows if (v := _number(row, "action_count", "tool_action_count", "actions")) is not None],
        "model_turns": [v for row in rows if (v := _number(row, "model_turn_count", "turn_count")) is not None],
        "output_tokens": [v for row in rows if (v := _number(row, "output_tokens", "total_output_tokens")) is not None],
    }
    result: dict[str, Any] = {"completed_cells": len(rows)}
    for name, values in metrics.items():
        result[name] = {
            "n": len(values),
            "mean": round(mean(values), 2) if values else None,
            "total": round(sum(values), 2) if values else None,
        }
    grades = [official_pass(row, capture_root) for row in rows]
    known = [grade for grade in grades if grade is not None]
    result["official_passes"] = sum(known)
    result["graded_cells"] = len(known)
    reasons = [_termination_reason(row) for row in rows]
    result["censored_harness_failures"] = sum(reason in CENSORED_FAILURES for reason in reasons)
    result["infrastructure_or_protocol_failures"] = sum(_has_infra_or_protocol_failure(row, reason) for row, reason in zip(rows, reasons))
    return result


def _has_infra_or_protocol_failure(row: dict[str, Any], reason: str) -> bool:
    return (
        reason in INFRA_PROTOCOL_FAILURES
        or bool(row.get("infrastructure_failure"))
        or bool(row.get("protocol_failure"))
        or row.get("protocol_version") != PROTOCOL_VERSION
        or not row.get("protocol_fingerprint")
    )


def analyze(rows: Iterable[dict[str, Any]], capture_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Apply the frozen gate and return a JSON-serializable report."""
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    ignored = 0
    for row in rows:
        key = cell_key(row)
        if key is None or key[0] not in TASK_IDS:
            ignored += 1
            continue
        by_key[key] = row

    keys = expected_cells()
    missing = []
    pending = []
    terminal_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in keys:
        key = (cell["instance_id"], cell["backend"], cell["harness"])
        row = by_key.get(key)
        label = "__".join(key)
        if row is None:
            missing.append(label)
        elif not _terminal(row):
            pending.append(label)
        else:
            terminal_by_key[key] = row

    complete_trial = len(terminal_by_key) == len(keys)
    by_harness = {
        harness: _descriptive(
            [row for key, row in terminal_by_key.items() if key[2] == harness], capture_root
        )
        for harness in HARNESSES
    }

    pair_rows = []
    complete_pairs = 0
    paired_grade_values: list[tuple[bool, bool]] = []
    for task in TASK_IDS:
        for backend in BACKENDS:
            claude_key = (task, backend, "claude-code")
            pi_key = (task, backend, "pi")
            claude_row = terminal_by_key.get(claude_key)
            pi_row = terminal_by_key.get(pi_key)
            pair = {
                "instance_id": task,
                "backend": backend,
                "claude_code_termination": _termination_reason(claude_row) if claude_row else None,
                "pi_termination": _termination_reason(pi_row) if pi_row else None,
                "paired_terminal": bool(claude_row and pi_row),
            }
            if claude_row and pi_row:
                complete_pairs += 1
                cg = official_pass(claude_row, capture_root)
                pg = official_pass(pi_row, capture_root)
                pair["claude_code_official_pass"] = cg
                pair["pi_official_pass"] = pg
                if cg is not None and pg is not None:
                    paired_grade_values.append((cg, pg))
            pair_rows.append(pair)

    claude_failures = by_harness["claude-code"]["censored_harness_failures"]
    pi_failures = by_harness["pi"]["censored_harness_failures"]
    fingerprints = {
        row.get("protocol_fingerprint")
        for row in terminal_by_key.values()
        if row.get("protocol_fingerprint")
    }
    fingerprint_mismatch = len(fingerprints) > 1
    pi_rows = [row for key, row in terminal_by_key.items() if key[2] == "pi"]
    pi_infra_failures = sum(_has_infra_or_protocol_failure(row, _termination_reason(row)) for row in pi_rows)
    if fingerprint_mismatch:
        pi_infra_failures = max(1, pi_infra_failures)
    claude_passes = sum(c for c, _ in paired_grade_values)
    pi_passes = sum(p for _, p in paired_grade_values)
    all_pairs_graded = len(paired_grade_values) == 4

    if not complete_trial:
        recommendation = "inconclusive_incomplete_trial"
        rationale = "Wait for all eight terminal cells before applying the migration gate."
    elif pi_infra_failures:
        recommendation = "do_not_migrate_pi_infrastructure_or_protocol_failure"
        rationale = "Pi introduced an infrastructure or protocol failure in the controlled trial."
    elif not all_pairs_graded:
        recommendation = "inconclusive_waiting_for_paired_official_grades"
        rationale = "The operational failure comparison is complete, but all four matched official grades are not available."
    elif pi_passes < claude_passes:
        recommendation = "do_not_migrate_pi_quality_regression"
        rationale = "Pi has fewer official passes than Claude Code on the four matched task/backend pairs."
    elif claude_failures - pi_failures < 2:
        recommendation = "inconclusive_pi_not_demonstrably_better"
        rationale = "Pi did not reduce censored harness failures by the preregistered minimum of two."
    else:
        recommendation = "recommend_pi_migration"
        rationale = "All eight cells completed; Pi reduced censored failures by at least two, did not lose official passes, and introduced no infrastructure/protocol failure."

    return {
        "protocol_version": PROTOCOL_VERSION,
        "interpretation": "Small-sample operational gate only; this is not a statistical superiority claim.",
        "expected_cells": len(keys),
        "terminal_cells": len(terminal_by_key),
        "complete_pairs": complete_pairs,
        "missing_cells": missing,
        "pending_cells": pending,
        "ignored_rows": ignored,
        "paired_cells": pair_rows,
        "descriptive_by_harness": by_harness,
        "paired_official_passes": {"claude_code": claude_passes, "pi": pi_passes, "graded_pairs": len(paired_grade_values), "required_pairs": 4},
        "censored_failure_reduction_pi_vs_claude": claude_failures - pi_failures,
        "pi_infrastructure_or_protocol_failures": pi_infra_failures,
        "protocol_fingerprints": sorted(fingerprints),
        "protocol_fingerprint_consistent": len(fingerprints) <= 1,
        "recommendation": recommendation,
        "rationale": rationale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", type=Path, default=OUT_ROOT, help="capture root or capture_manifest.json")
    parser.add_argument("--output", type=Path, help="write report JSON outside capture data; defaults to stdout")
    args = parser.parse_args()
    root = args.capture if args.capture.is_dir() else args.capture.parent
    report = analyze(load_rows(args.capture), capture_root=root)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
