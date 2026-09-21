"""Read-only 60-second monitor for harness-ab-pi-v1.

The monitor writes snapshots into a sibling monitor directory. Capture
manifests, per-cell metadata, streams, and patches are never modified.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_harness_ab_pi_v1 import (
    BACKENDS,
    HARNESSES,
    OUT_ROOT,
    TASK_IDS,
    _terminal,
    cell_key,
    expected_cells,
    load_rows,
)

POLL_SECONDS = 60
STALE_AFTER_SECONDS = 180
DRIVER_SESSION = "harness-ab-pi-v1"
RELAY_SESSION = "direct-gpu-relay-timeout-v2"
EDGE_HEALTH_URL = "http://127.0.0.1:18012/health"
MONITOR_ROOT = OUT_ROOT.parent / "harness_ab_pi_v1_monitor"
SNAPSHOT = MONITOR_ROOT / "monitor.json"
HISTORY = MONITOR_ROOT / "monitor.jsonl"


def _tmux_alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _edge_health() -> tuple[bool, dict[str, Any] | str]:
    try:
        with urllib.request.urlopen(EDGE_HEALTH_URL, timeout=10) as response:
            payload = json.loads(response.read())
        return payload.get("status") == "ok", payload
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _cell_dirs(capture_root: Path, key: tuple[str, str, str]) -> list[Path]:
    root = capture_root / "__".join(key)
    if not root.is_dir():
        return []
    attempts = sorted(path for path in root.glob("attempt-*") if path.is_dir())
    return attempts or [root]


def _active_stream(capture_root: Path, key: tuple[str, str, str], now: float) -> dict[str, Any] | None:
    candidates: list[Path] = []
    for attempt_dir in _cell_dirs(capture_root, key):
        candidates.extend(attempt_dir.glob("*stream*.jsonl"))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    stream = max(existing, key=lambda path: path.stat().st_mtime)
    stat = stream.stat()
    age = round(max(0.0, now - stat.st_mtime), 1)
    return {
        "path": str(stream.relative_to(capture_root)),
        "bytes": stat.st_size,
        "age_seconds": age,
        "fresh": age <= STALE_AFTER_SECONDS,
    }


def build_snapshot(
    capture_root: Path = OUT_ROOT,
    *,
    now: float | None = None,
    tmux_alive=_tmux_alive,
    edge_health=_edge_health,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    rows = load_rows(capture_root)
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = cell_key(row)
        if key is not None:
            rows_by_key[key] = row

    cells = []
    completed = 0
    for cell in expected_cells():
        key = (cell["instance_id"], cell["backend"], cell["harness"])
        row = rows_by_key.get(key)
        terminal = _terminal(row)
        completed += int(terminal)
        active_stream = None if terminal else _active_stream(capture_root, key, now)
        cells.append({
            **cell,
            "status": "terminal" if terminal else (str((row or {}).get("status", "missing"))),
            "termination_reason": (row or {}).get("termination_reason"),
            "active_stream": active_stream,
        })

    edge_ok, edge_details = edge_health()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "harness-ab-pi-v1",
        "completed_cells": completed,
        "expected_cells": len(cells),
        "remaining_cells": len(cells) - completed,
        "cells": cells,
        "driver_tmux_alive": tmux_alive(DRIVER_SESSION),
        "relay_tmux_alive": tmux_alive(RELAY_SESSION),
        "edge_endpoint_ok": edge_ok,
        "edge_health": edge_details,
    }


def write_snapshot(state: dict[str, Any], monitor_root: Path = MONITOR_ROOT) -> None:
    monitor_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = monitor_root / "monitor.json"
    history_path = monitor_root / "monitor.jsonl"
    temporary = monitor_root / "monitor.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(snapshot_path)
    with history_path.open("a") as history:
        history.write(json.dumps(state, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="write one snapshot and exit")
    parser.add_argument("--capture-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--monitor-root", type=Path, default=MONITOR_ROOT)
    args = parser.parse_args()
    while True:
        state = build_snapshot(args.capture_root)
        write_snapshot(state, args.monitor_root)
        print(json.dumps(state, indent=2), flush=True)
        if args.once or state["completed_cells"] >= state["expected_cells"]:
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
