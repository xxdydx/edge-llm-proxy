"""Sequential, durable cloud-trajectory collector for the six pilot dev tasks.

Each job runs in its own process group and has a 20-minute model wall cap.
Official base/gold grader sanity is required before accepting a new task.
The controller will not start a new task while the paired replay/judge of the
previous completed task is pending.  Holdouts are deliberately not launched
here: their model/threshold must be frozen first.

This script is intended for tmux. Its process PID may be used with
``caffeinate -i -w PID``; it does not change persistent power settings.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import collect, pilot_campaign

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "eval-suite" / "swebench" / "results" / pilot_campaign.PILOT_CAMPAIGN
SANITY = RESULTS / "grader_sanity"
STATUS = RESULTS / "pilot_live_status.json"
LOCK = RESULTS / "pilot_live.lock"
SHADOW: Path | None = (
    ROOT / "experiments" / "agentic_router" / "results"
    / "quality_model_baseline205_v5_shadow_logreg.json"
)  # observe-only; 205-call feature parity verified before activation
JOB_TIMEOUT_S = 1200
RUNNER = ROOT / "eval-suite" / "swebench" / "runner" / "run_swebench.py"
GRADER = ROOT / "eval-suite" / "swebench" / "runner" / "verify_official_grader.py"


def _atomic_status(**fields: Any) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    row = {"at_utc": datetime.now(timezone.utc).isoformat(),
           "no_work_after_utc": pilot_campaign.WORK_STOP.astimezone(timezone.utc).isoformat(),
           "requests_drained_by_utc": pilot_campaign.DRAIN.astimezone(timezone.utc).isoformat(),
           "gpu_expiry_utc": "2026-09-17T09:00:00+00:00",
           "job_timeout_s": JOB_TIMEOUT_S, "cloud_model": "deepseek-v4-flash",
           "local_profile": "Qwen3.8-27B-NVFP4/100K", **fields}
    tmp = STATUS.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(row, fh, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, STATUS)


def _remaining() -> float:
    return (pilot_campaign.WORK_STOP - datetime.now(pilot_campaign.SGT)).total_seconds()


def _scoped_container_name(slug: str, condition: str) -> str:
    # Exact same deterministic name as run_swebench.py; never a broad Docker
    # cleanup. A stopped task can leave this name occupied on an interrupted
    # runner, and must be cleaned before a scoped retry.
    if condition == "grader":
        return f"flowmesh-grader-check-{slug}"
    sys.path.insert(0, str(RUNNER.parent))
    import run_swebench  # type: ignore
    return run_swebench.container_name_for(pilot_campaign.PILOT_CAMPAIGN,
                                            f"{slug}__{condition}__seed1")


def _run_bounded(argv: list[str], timeout_s: float, slug: str | None = None,
                 condition: str | None = None) -> tuple[int | None, str]:
    if _remaining() <= 0:
        return None, "drained"
    proc = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    start = time.monotonic()
    try:
        while True:
            remaining = min(timeout_s - (time.monotonic() - start), _remaining())
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout_s)
            try:
                code = proc.wait(timeout=min(10, remaining))
                break
            except subprocess.TimeoutExpired:
                _atomic_status(state="child_running", task=slug, condition=condition,
                               child_pid=proc.pid, elapsed_s=round(time.monotonic() - start, 1))
        if code != 0 and slug and condition:
            name = _scoped_container_name(slug, condition)
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        return code, "completed"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        if slug and condition:
            name = _scoped_container_name(slug, condition)
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        return None, "timeout_or_drain"


def _grade_sanity(slug: str) -> bool:
    if pilot_campaign.POSTPILOT:
        # Postpilot images are pulled one at a time only after a Docker VM headroom
        # check by the supervisor. Never let Docker auto-pull an uncached
        # later task behind that disk gate.
        sys.path.insert(0, str(RUNNER.parent))
        from run_swebench import load_instances  # type: ignore
        from verify_official_grader import fingerprint  # type: ignore
        try:
            fingerprint(next(iter(load_instances(slug))))
        except (OSError, ValueError, RuntimeError):
            _atomic_status(state="image_not_cached", task=slug)
            return False
    path = SANITY / f"{slug}.sanity.json"
    if path.exists():
        sys.path.insert(0, str(RUNNER.parent))
        from run_swebench import load_instances  # type: ignore
        from verify_official_grader import fingerprint  # type: ignore
        try:
            old = json.loads(path.read_text())
            expected = fingerprint(next(iter(load_instances(slug))))
            if old.get("fingerprint") == expected:
                return bool(old.get("sanity_pass"))
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            pass  # stale/unknown image or metadata: rerun instead of trusting
    _atomic_status(state="grader_sanity", task=slug)
    code, _ = _run_bounded([sys.executable, str(GRADER), slug,
                            "--out-dir", str(SANITY)], timeout_s=180,
                           slug=slug, condition="grader")
    return code == 0 and path.exists() and bool(json.loads(path.read_text()).get("sanity_pass"))


def _verdict(slug: str, condition: str) -> tuple[bool | None, str | None]:
    path = RESULTS / "verdicts" / f"{slug}__{condition}__seed1.json"
    if not path.exists():
        return None, None
    return collect.load_verdict(path)


def _retryable_cloud_infra(slug: str) -> bool:
    """Retry only a confirmed upstream outage, never model time exhaustion.

    The runner writes an ungraded placeholder for both 502 circuit breaks
    and its own 20-minute agent deadline.  Both are invalid quality labels,
    but another full attempt at a timed-out task is not a lightweight infra
    retry.
    """
    path = RESULTS / "verdicts" / f"{slug}__cloud__seed1.json"
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return str(row.get("error") or "").startswith(
        "upstream infrastructure failure: 3 consecutive HTTP 502"
    )


def _initial_cloud_launch_needed(slug: str) -> bool:
    """A persisted ungraded verdict is still a completed *attempt*.

    Resume must not confuse its absence of a quality grade with absence of
    the attempt itself, especially after a model time-budget exhaustion.
    """
    return not (RESULTS / "verdicts" / f"{slug}__cloud__seed1.json").exists()


def _launch(slug: str, condition: str) -> tuple[int | None, str]:
    args = [sys.executable, str(RUNNER), "--instances", slug,
            "--conditions", condition, "--seeds", "1",
            "--cloud-parallelism", "1", "--local-parallelism", "1",
            "--claude-model", "deepseek-v4-flash",
            "--vllm-url", "http://127.0.0.1:18004",
            "--max-local-tokens", "100000",
            "--job-timeout-s", str(JOB_TIMEOUT_S),
            "--experiment-id", pilot_campaign.PILOT_CAMPAIGN,
            "--results-dir", str(RESULTS)]
    if SHADOW is not None:
        if not SHADOW.is_file():
            raise RuntimeError(f"Missing frozen shadow artifact: {SHADOW}")
        args.extend(["--agentic-shadow-artifact", str(SHADOW)])
    _atomic_status(state="agent_running", task=slug, condition=condition)
    return _run_bounded(args, timeout_s=JOB_TIMEOUT_S + 600, slug=slug,
                        condition=condition)


def _wait_for_replay(slug: str) -> bool:
    deadline = min(time.monotonic() + 5400, time.monotonic() + max(0, _remaining()))
    while time.monotonic() < deadline:
        manifest = pilot_campaign._load_selection()
        entries = [v for v in manifest["trajectories"].values() if v["task_group"] == slug]
        if entries:
            ids = {call_id for entry in entries for call_id in entry["call_ids"]}
            paired = pilot_campaign._paired_ids()
            judged = pilot_campaign._judged_ids()
            infeasible = {row["call"]["call_id"] for row in pilot_campaign._load_pilot_pairs()
                          if row["local_outcome"]["status"] == "INVALID_CAPACITY"}
            if ids <= paired and ids <= (judged | infeasible):
                return True
        _atomic_status(state="waiting_for_replay", task=slug,
                       selected_calls=sum(len(v["call_ids"]) for v in entries),
                       paired_calls=len(ids & paired) if entries else 0,
                       judged_calls=len(ids & judged) if entries else 0,
                       capacity_invalid_calls=len(ids & infeasible) if entries else 0)
        time.sleep(min(30, max(1, _remaining())))
    return False


def run(only_task: str | None = None) -> None:
    if only_task is not None and only_task not in pilot_campaign.DEVELOPMENT_TASKS:
        raise ValueError("Requested task is not in the frozen development manifest")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for slug in pilot_campaign.DEVELOPMENT_TASKS:
            if only_task is not None and slug != only_task:
                continue
            if _remaining() <= 0:
                break
            if not _grade_sanity(slug):
                _atomic_status(state="grader_invalid", task=slug)
                continue
            _, existing_detail = _verdict(slug, "cloud")
            if _initial_cloud_launch_needed(slug):
                _launch(slug, "cloud")
            outcome, detail = _verdict(slug, "cloud")
            retryable_infra = _retryable_cloud_infra(slug) if outcome is None else False
            if outcome is None and retryable_infra and _remaining() > 0:
                # At most one cloud retry after an infrastructure-invalid
                # verdict. Stop further cloud jobs if repeated 502 persists.
                _atomic_status(state="cloud_infra_retry", task=slug,
                               detail=(detail or existing_detail or "")[:200])
                _launch(slug, "cloud")
                outcome, detail = _verdict(slug, "cloud")
            if outcome is None:
                _atomic_status(state="cloud_infra_blocked" if retryable_infra else "cloud_ungraded_no_retry", task=slug,
                               detail=(detail or "")[:200])
                # Local source trajectories remain useful, but never call
                # them cloud-equivalent. The cloud counterpart is queued for
                # later recovery once the endpoint is healthy.
                if retryable_infra and _remaining() > 0 and _verdict(slug, "local")[0] is None:
                    _launch(slug, "local")
                continue
            _atomic_status(state="graded", task=slug, condition="cloud",
                           task_passed=outcome, detail=(detail or "")[:200])
            if not _wait_for_replay(slug):
                _atomic_status(state="replay_not_finished", task=slug)
                break
        _atomic_status(state="drained" if _remaining() <= 0 else "finished_development")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-task", choices=pilot_campaign.DEVELOPMENT_TASKS,
                        help="run only this preregistered task, preserving normal checkpoints")
    args = parser.parse_args()
    run(only_task=args.only_task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
