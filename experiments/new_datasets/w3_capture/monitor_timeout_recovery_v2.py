"""Persistently monitor the timeout-recovery-v2 capture campaign.

Writes a replace-in-place current snapshot plus an append-only history. The
monitor is deliberately read-only with respect to the campaign: it reports a
dead driver, unhealthy edge endpoint, or stale active stream without silently
restarting or changing the frozen recovery protocol.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

W3 = Path(__file__).resolve().parent
OUT_ROOT = W3 / "timeout_recovery_v2"
SNAPSHOT = OUT_ROOT / "monitor.json"
HISTORY = OUT_ROOT / "monitor.jsonl"
TOTAL_TARGETS = 18
POLL_SECONDS = 60
EDGE_HEALTH_URL = "http://127.0.0.1:18012/health"


def _tmux_alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _edge_health() -> tuple[bool, dict | str]:
    try:
        with urllib.request.urlopen(EDGE_HEALTH_URL, timeout=15) as response:
            payload = json.loads(response.read())
        expected = (
            payload.get("status") == "ok"
            and payload.get("local_max_output_tokens") == 8192
            and payload.get("local_thinking") == "disabled"
        )
        return expected, payload
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def snapshot() -> dict:
    now = time.time()
    records = []
    for path in sorted(OUT_ROOT.glob("*/run_meta.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue

    active = []
    for run_dir in sorted(path for path in OUT_ROOT.iterdir() if path.is_dir()):
        if (run_dir / "run_meta.json").exists():
            continue
        stream = run_dir / "claude_stream.jsonl"
        if stream.exists():
            stat = stream.stat()
            active.append({
                "run_dir": run_dir.name,
                "stream_bytes": stat.st_size,
                "stream_age_seconds": round(now - stat.st_mtime, 1),
            })

    edge_ok, edge_health = _edge_health()
    reasons = Counter(row.get("claude_termination_reason", "unknown") for row in records)
    policies = Counter(row.get("policy", "unknown") for row in records)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "completed": len(records),
        "total": TOTAL_TARGETS,
        "remaining": max(0, TOTAL_TARGETS - len(records)),
        "termination_counts": dict(sorted(reasons.items())),
        "policy_counts": dict(sorted(policies.items())),
        "last_completed": records[-1].get("run_dir") if records else None,
        "active": active,
        "campaign_tmux_alive": _tmux_alive("timeout-recovery-v2"),
        "relay_tmux_alive": _tmux_alive("direct-gpu-relay-timeout-v2"),
        "edge_endpoint_ok": edge_ok,
        "edge_health": edge_health,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        state = snapshot()
        temporary = SNAPSHOT.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=1) + "\n")
        temporary.replace(SNAPSHOT)
        with HISTORY.open("a") as history:
            history.write(json.dumps(state, separators=(",", ":")) + "\n")
        print(json.dumps(state, indent=1), flush=True)
        if state["completed"] >= TOTAL_TARGETS:
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
