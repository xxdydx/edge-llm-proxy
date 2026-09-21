"""60-second monitor and hourly summarizer for permanent Pi collection.

Samples active capture state every 60 seconds and produces concise hourly summaries
with counts and failure rates:
- Capture censored failure rate by reason (trajectory_deadline, repeated_action, etc.)
- After grading, official failure rate strictly among graded cells
- Strictly distinguishes pending, censored, and ungraded cells from task FAIL
- Relay/edge health and driver tmux liveness tracking
- Atomic updates to monitor.json/hourly_summary.json and append-only jsonl logs

Read-only: does not run collection, launch docker containers, or modify capture data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import uuid

logger = logging.getLogger("pi_monitor")

REPO_ROOT = Path(__file__).resolve().parents[2]
ND = REPO_ROOT / "experiments/new_datasets"

POLL_SECONDS_DEFAULT = 60
STALE_AFTER_SECONDS = 180
HOURLY_INTERVAL_SECONDS = 3600

DEFAULT_CAPTURE_ROOT = ND / "w3_capture/pi_dataset_ac_v1"
DEFAULT_MONITOR_ROOT = ND / "campaigns/pi_permanent_v1/monitor"
DEFAULT_DRIVER_SESSION = "pi-collection-campaign"
DEFAULT_RELAY_SESSION = "direct-gpu-relay-timeout-v2"
DEFAULT_EDGE_HEALTH_URL = "http://127.0.0.1:18012/health"

# Canonical tasks from preflight files
TASK_SOURCE_FILES = [
    ND / "w2_preflight/smoke_instances.json",
    ND / "w2_preflight/train50_5tasks_instances.json",
    ND / "w2_preflight/gym_batch2_14tasks_instances.json",
]

BACKENDS = ("edge", "cloud")

CENSORED_REASONS = {
    "trajectory_deadline",
    "repeated_action",
    "step_limit",
    "turn_limit",
    "process_failure",
    "protocol_failure",
    "adapter_error",
    "setup_failure",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with temp_path.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def load_expected_tasks(source_files: list[Path] | None = None) -> list[str]:
    """Load and deduplicate the 21 preflight instance IDs."""
    files = source_files or TASK_SOURCE_FILES
    instance_ids = []
    for file_path in files:
        if file_path.is_file():
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                for row in data:
                    iid = row.get("instance_id")
                    if iid and iid not in instance_ids:
                        instance_ids.append(iid)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read task file %s: %exc", file_path, exc)
    return sorted(instance_ids)


def expected_cells(task_ids: list[str] | None = None) -> list[dict[str, str]]:
    """Return the 42 expected (instance_id, backend) cell specifications."""
    tasks = task_ids or load_expected_tasks()
    cells = []
    for t in tasks:
        for b in BACKENDS:
            cells.append({"instance_id": t, "backend": b, "cell_id": f"{t}__{b}"})
    return cells


def check_tmux_session_alive(session_name: str) -> bool:
    """Check if a tmux session is currently alive."""
    res = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def check_edge_health(health_url: str = DEFAULT_EDGE_HEALTH_URL, timeout: float = 10.0) -> tuple[bool, dict[str, Any] | str]:
    """Query edgeproxy health endpoint."""
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "pi-monitor"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            raw = response.read().decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"raw": raw}
            return (status_code == 200 and data.get("status") == "ok"), data
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Cell Inspection and Analysis
# ---------------------------------------------------------------------------

def _find_cell_directory(capture_root: Path, instance_id: str, backend: str) -> Path | None:
    """Locate output directory for a specific task cell."""
    candidate_names = [
        f"{instance_id}__{backend}",
        f"{instance_id}__{backend}__pi",
        f"{instance_id}__{backend}-only-v1",
        f"{instance_id}__{backend}_only",
    ]
    for name in candidate_names:
        direct = capture_root / name
        if direct.is_dir():
            return direct

    # Fallback: scan directories for run_meta.json matching instance_id and backend
    if capture_root.is_dir():
        for d in capture_root.iterdir():
            if d.is_dir():
                meta_file = d / "run_meta.json"
                if meta_file.is_file():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        m_iid = meta.get("instance_id")
                        m_backend = meta.get("backend") or meta.get("policy", "")
                        if m_iid == instance_id and backend in m_backend:
                            return d
                    except (json.JSONDecodeError, OSError):
                        continue
    return None


def _load_grading_map(capture_root: Path) -> dict[str, dict[str, Any]]:
    """Load any available grading reports into a map keyed by instance_id and backend."""
    grades: dict[str, dict[str, Any]] = {}
    if not capture_root.is_dir():
        return grades

    # Check unified grading_results.json
    unified = capture_root / "grading_results.json"
    if unified.is_file():
        try:
            data = json.loads(unified.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    key = item.get("cell_id") or item.get("session_id") or item.get("instance_id")
                    if key:
                        grades[key] = item
            elif isinstance(data, dict):
                for key, item in data.items():
                    grades[key] = item if isinstance(item, dict) else {"resolved": item}
        except (json.JSONDecodeError, OSError):
            pass

    # Check individual grading_results_*.json files
    for p in capture_root.glob("grading_results_*.json"):
        if p.name == "grading_results.json":
            continue
        try:
            item = json.loads(p.read_text(encoding="utf-8"))
            tag = p.stem.replace("grading_results_", "")
            grades[tag] = item
        except (json.JSONDecodeError, OSError):
            pass

    return grades


def inspect_cell(
    capture_root: Path,
    instance_id: str,
    backend: str,
    now: float,
    grading_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inspect single cell status, distinguishing capture and grading states.

    Returns dict with keys:
    - cell_id
    - instance_id
    - backend
    - capture_status: 'pending' | 'running' | 'completed' | 'censored'
    - censored_reason: str | None
    - grading_status: 'pending' | 'censored' | 'ungraded' | 'graded_pass' | 'graded_fail'
    - resolved: bool | None
    - active_stream: dict | None
    - run_meta: dict | None
    """
    cell_id = f"{instance_id}__{backend}"
    cell_dir = _find_cell_directory(capture_root, instance_id, backend)

    if cell_dir is None or not cell_dir.is_dir():
        return {
            "cell_id": cell_id,
            "instance_id": instance_id,
            "backend": backend,
            "cell_dir": None,
            "capture_status": "pending",
            "censored_reason": None,
            "grading_status": "pending",
            "resolved": None,
            "active_stream": None,
            "run_meta": None,
        }

    meta_file = cell_dir / "run_meta.json"
    run_meta = None
    if meta_file.is_file():
        try:
            run_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Inspect stream file
    streams = list(cell_dir.glob("*stream*.jsonl"))
    stream_info = None
    if streams:
        newest = max(streams, key=lambda s: s.stat().st_mtime)
        stat = newest.stat()
        age = round(max(0.0, now - stat.st_mtime), 1)
        stream_info = {
            "path": str(newest.relative_to(capture_root)),
            "bytes": stat.st_size,
            "age_seconds": age,
            "fresh": age <= STALE_AFTER_SECONDS,
        }

    # Determine capture status and censorship
    if run_meta is None:
        if stream_info and stream_info["fresh"]:
            capture_status = "running"
        else:
            capture_status = "pending"
        censored_reason = None
    else:
        term_reason = run_meta.get("termination_reason", "completed")
        if term_reason in CENSORED_REASONS:
            capture_status = "censored"
            censored_reason = term_reason
        elif term_reason == "completed":
            capture_status = "completed"
            censored_reason = None
        else:
            # Check for error or timeout indication
            if run_meta.get("timeout") or run_meta.get("status") == "timeout":
                capture_status = "censored"
                censored_reason = "trajectory_deadline"
            else:
                capture_status = "completed"
                censored_reason = None

    # Determine grading status
    resolved = None
    grade_data = None

    # Check cell_dir for grading_result.json
    local_grade = cell_dir / "grading_result.json"
    if local_grade.is_file():
        try:
            grade_data = json.loads(local_grade.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Check grading map
    if grade_data is None and grading_map:
        for lookup_key in (cell_id, cell_dir.name, (run_meta or {}).get("session_id"), instance_id):
            if lookup_key and lookup_key in grading_map:
                grade_data = grading_map[lookup_key]
                break

    # Check in run_meta directly
    if grade_data is None and run_meta:
        if "official_pass" in run_meta:
            resolved = run_meta["official_pass"]
            grade_data = {"resolved": resolved}
        elif "resolved" in run_meta:
            resolved = run_meta["resolved"]
            grade_data = {"resolved": resolved}

    if grade_data is not None:
        if isinstance(grade_data, dict):
            # Check for resolved boolean in report or top-level
            if "resolved" in grade_data and isinstance(grade_data["resolved"], bool):
                resolved = grade_data["resolved"]
            elif "report" in grade_data and isinstance(grade_data["report"], dict):
                inst_rep = grade_data["report"].get(instance_id, {})
                if isinstance(inst_rep, dict) and "resolved" in inst_rep:
                    resolved = inst_rep["resolved"]
            elif "tests_status" in grade_data:
                # SWE-Bench status
                f2p = grade_data.get("tests_status", {}).get("FAIL_TO_PASS", {}).get("status")
                resolved = (f2p == "PASSED")

    # Strict separation:
    # pending -> capture not finished
    # censored -> stopped by watchdog/breaker
    # ungraded -> capture finished, but no grade report
    # graded_pass -> tests PASSED
    # graded_fail -> tests FAILED (actual task FAIL!)
    if resolved is True:
        grading_status = "graded_pass"
    elif resolved is False:
        grading_status = "graded_fail"
    else:
        if capture_status == "pending":
            grading_status = "pending"
        elif capture_status == "running":
            grading_status = "running"
        elif capture_status == "censored":
            grading_status = "censored"
        else:
            grading_status = "ungraded"

    return {
        "cell_id": cell_id,
        "instance_id": instance_id,
        "backend": backend,
        "cell_dir": str(cell_dir.relative_to(capture_root)) if cell_dir else None,
        "capture_status": capture_status,
        "censored_reason": censored_reason,
        "grading_status": grading_status,
        "resolved": resolved,
        "active_stream": stream_info if capture_status == "running" else None,
        "run_meta": run_meta,
    }


# ---------------------------------------------------------------------------
# Aggregations, Metrics, and Failure Rates
# ---------------------------------------------------------------------------

def calculate_metrics(cell_inspections: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate counts and failure rates with strict classification.

    Guarantees:
    - Official failure rate is strictly calculated among graded cells (graded_pass + graded_fail).
    - Pending, censored, and ungraded cells are NEVER counted as task FAIL.
    - Censored failure rate is tracked with counts and rates broken down by reason.
    """
    total_cells = len(cell_inspections)
    capture_counts = Counter(c["capture_status"] for c in cell_inspections)
    grading_counts = Counter(c["grading_status"] for c in cell_inspections)

    censored_reasons = Counter(
        c["censored_reason"] for c in cell_inspections if c["censored_reason"] is not None
    )

    pending_count = capture_counts.get("pending", 0)
    running_count = capture_counts.get("running", 0)
    completed_capture = capture_counts.get("completed", 0)
    censored_capture = capture_counts.get("censored", 0)
    attempted_capture = completed_capture + censored_capture

    ungraded_count = grading_counts.get("ungraded", 0)
    graded_pass = grading_counts.get("graded_pass", 0)
    graded_fail = grading_counts.get("graded_fail", 0)
    total_graded = graded_pass + graded_fail

    # Rates
    capture_censored_rate = (
        round(censored_capture / attempted_capture, 4) if attempted_capture > 0 else 0.0
    )
    capture_censored_rates_by_reason = {
        reason: round(count / attempted_capture, 4) if attempted_capture > 0 else 0.0
        for reason, count in sorted(censored_reasons.items())
    }

    official_failure_rate = (
        round(graded_fail / total_graded, 4) if total_graded > 0 else None
    )
    official_pass_rate = (
        round(graded_pass / total_graded, 4) if total_graded > 0 else None
    )

    return {
        "total_cells": total_cells,
        "counts": {
            "pending": pending_count,
            "running": running_count,
            "completed_capture": completed_capture,
            "censored_capture": censored_capture,
            "attempted_capture": attempted_capture,
            "ungraded": ungraded_count,
            "graded_pass": graded_pass,
            "graded_fail": graded_fail,
            "total_graded": total_graded,
        },
        "failure_rates": {
            "capture_censored_failure_rate": capture_censored_rate,
            "capture_censored_by_reason": dict(sorted(censored_reasons.items())),
            "capture_censored_rates_by_reason": capture_censored_rates_by_reason,
            "official_failure_rate_among_graded": official_failure_rate,
            "official_pass_rate_among_graded": official_pass_rate,
        },
    }


def format_concise_summary_message(metrics: dict[str, Any], health: dict[str, Any]) -> str:
    """Format a single concise string summarizing campaign progress and health."""
    total_cells = metrics.get("total_cells", len(metrics.get("counts", {})))
    counts = metrics["counts"]
    rates = metrics["failure_rates"]

    censored_breakdown = ", ".join(
        f"{r}: {c}" for r, c in rates["capture_censored_by_reason"].items()
    ) or "none"

    if counts["total_graded"] > 0:
        fail_pct = f"{rates['official_failure_rate_among_graded'] * 100:.1f}%"
        grading_msg = f"{counts['total_graded']} graded ({counts['graded_pass']} PASS, {counts['graded_fail']} FAIL; official fail rate: {fail_pct})"
    else:
        grading_msg = "0 graded (all pending/ungraded/censored)"

    driver_status = "ALIVE" if health.get("driver_tmux_alive") else "DEAD"
    relay_status = "ALIVE" if health.get("relay_tmux_alive") else "DEAD"
    edge_status = "OK" if health.get("edge_health_ok") else "DOWN"

    return (
        f"Pi permanent collection: {total_cells} cells | "
        f"{counts['running']} running, {counts['completed_capture']} completed, {counts['censored_capture']} censored "
        f"({censored_breakdown}; censored rate: {rates['capture_censored_failure_rate'] * 100:.1f}%) | "
        f"{grading_msg} | {counts['ungraded']} ungraded, {counts['pending']} pending | "
        f"driver: {driver_status}, relay: {relay_status}, edge: {edge_status}"
    )


# ---------------------------------------------------------------------------
# Snapshots and Hourly Summaries
# ---------------------------------------------------------------------------

def build_snapshot(
    capture_root: Path = DEFAULT_CAPTURE_ROOT,
    task_ids: list[str] | None = None,
    *,
    now: float | None = None,
    tmux_checker: Callable[[str], bool] = check_tmux_session_alive,
    edge_health_checker: Callable[[], tuple[bool, Any]] | None = None,
    driver_session: str = DEFAULT_DRIVER_SESSION,
    relay_session: str = DEFAULT_RELAY_SESSION,
) -> dict[str, Any]:
    """Construct complete snapshot of current collection and grading status."""
    now_ts = time.time() if now is None else now
    cells = expected_cells(task_ids)
    grading_map = _load_grading_map(capture_root)

    inspections = [
        inspect_cell(capture_root, cell["instance_id"], cell["backend"], now_ts, grading_map)
        for cell in cells
    ]
    metrics = calculate_metrics(inspections)

    edge_ok, edge_details = (
        edge_health_checker() if edge_health_checker else check_edge_health()
    )
    health = {
        "driver_tmux_alive": tmux_checker(driver_session),
        "relay_tmux_alive": tmux_checker(relay_session),
        "edge_health_ok": edge_ok,
        "edge_health_details": edge_details,
    }

    summary_text = format_concise_summary_message(metrics, health)

    return {
        "timestamp_utc": _utc_now(),
        "protocol_version": "pi-dataset-ac-v1",
        "capture_root": str(capture_root),
        "metrics": metrics,
        "health": health,
        "summary": summary_text,
        "cells": inspections,
    }


def build_hourly_summary(snapshot: dict[str, Any], hour_timestamp_utc: str | None = None) -> dict[str, Any]:
    """Extract a concise hourly summary from a snapshot."""
    hour_ts = hour_timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
    return {
        "hour_timestamp_utc": hour_ts,
        "snapshot_timestamp_utc": snapshot["timestamp_utc"],
        "protocol_version": snapshot.get("protocol_version", "pi-dataset-ac-v1"),
        "total_cells": snapshot["metrics"]["total_cells"],
        "counts": snapshot["metrics"]["counts"],
        "failure_rates": snapshot["metrics"]["failure_rates"],
        "health": {
            "driver_tmux_alive": snapshot["health"]["driver_tmux_alive"],
            "relay_tmux_alive": snapshot["health"]["relay_tmux_alive"],
            "edge_health_ok": snapshot["health"]["edge_health_ok"],
        },
        "concise_message": snapshot["summary"],
    }


def write_snapshot_and_history(snapshot: dict[str, Any], monitor_root: Path) -> None:
    """Atomically write latest snapshot and append to history log."""
    monitor_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = monitor_root / "monitor.json"
    history_path = monitor_root / "monitor.jsonl"

    _atomic_write_json(snapshot_path, snapshot)
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_hourly_summary(hourly_summary: dict[str, Any], monitor_root: Path) -> None:
    """Atomically write current hourly summary and append to hourly log."""
    monitor_root.mkdir(parents=True, exist_ok=True)
    latest_path = monitor_root / "hourly_summary.json"
    log_path = monitor_root / "hourly_summaries.jsonl"

    _atomic_write_json(latest_path, hourly_summary)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(hourly_summary, ensure_ascii=False, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Main Monitoring Loop
# ---------------------------------------------------------------------------

def run_monitor_loop(
    capture_root: Path = DEFAULT_CAPTURE_ROOT,
    monitor_root: Path = DEFAULT_MONITOR_ROOT,
    poll_seconds: int = POLL_SECONDS_DEFAULT,
    once: bool = False,
    task_ids: list[str] | None = None,
    driver_session: str = DEFAULT_DRIVER_SESSION,
    relay_session: str = DEFAULT_RELAY_SESSION,
) -> None:
    """Run persistent 60s sampling loop with concise hourly summaries."""
    logger.info("Starting Pi collection monitor on %s (polling every %ds)...", capture_root, poll_seconds)
    monitor_root.mkdir(parents=True, exist_ok=True)

    last_hourly_bucket: str | None = None

    while True:
        now_dt = datetime.now(timezone.utc)
        current_hour_bucket = now_dt.strftime("%Y-%m-%dT%H:00:00Z")

        snapshot = build_snapshot(
            capture_root=capture_root,
            task_ids=task_ids,
            driver_session=driver_session,
            relay_session=relay_session,
        )
        write_snapshot_and_history(snapshot, monitor_root)

        # Emit hourly summary when hour advances or on first sample
        if last_hourly_bucket != current_hour_bucket:
            hourly = build_hourly_summary(snapshot, current_hour_bucket)
            write_hourly_summary(hourly, monitor_root)
            last_hourly_bucket = current_hour_bucket
            logger.info("Hourly summary written: %s", hourly["concise_message"])

        print(snapshot["summary"], flush=True)

        if once:
            break

        time.sleep(poll_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Permanent Pi Coder Collection Monitor")
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT,
                        help="Root directory where W3 writes Pi captures")
    parser.add_argument("--monitor-root", type=Path, default=DEFAULT_MONITOR_ROOT,
                        help="Root directory for monitor logs and hourly summaries")
    parser.add_argument("--poll-interval", type=int, default=POLL_SECONDS_DEFAULT,
                        help="Sampling interval in seconds (default 60)")
    parser.add_argument("--once", action="store_true",
                        help="Sample once, write snapshot and hourly summary, and exit")
    parser.add_argument("--driver-session", default=DEFAULT_DRIVER_SESSION,
                        help="Tmux session name for campaign driver")
    parser.add_argument("--relay-session", default=DEFAULT_RELAY_SESSION,
                        help="Tmux session name for GPU relay")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_monitor_loop(
        capture_root=args.capture_root,
        monitor_root=args.monitor_root,
        poll_seconds=args.poll_interval,
        once=args.once,
        driver_session=args.driver_session,
        relay_session=args.relay_session,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
