"""Frozen four-task live sanity matrix. Prepared only; never auto-starts.

Run explicitly with ``python -m experiments.agentic_router.live_matrix --run``
after the Stage-1 collector has finished. Holds BOTH campaign lock files, so
the pilot collectors cannot launch overlapping GPU work. One task/condition
cell is a bounded runner subprocess; every completed cell gets an atomic
checkpoint and a prompt-free trace-derived metrics summary. This is n=1/cell,
not a quality-rate estimate or a learned-router deployment.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import collect, features, pilot_campaign

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "agentic-router-live-matrix-v1"
RESULTS = ROOT / "eval-suite/swebench/results" / EXPERIMENT
CHECKPOINTS = RESULTS / "matrix_checkpoints"
STATUS = RESULTS / "matrix_status.json"
PAUSE = RESULTS / "PAUSE_NEW_CELLS"
RUNNER = ROOT / "eval-suite/swebench/runner/run_swebench.py"
SHADOW = ROOT / "experiments/agentic_router/results/quality_model_baseline205_v5_shadow_logreg.json"
TASK_MANIFEST = ROOT / "eval-suite/swebench/instances_candidates_verified/pilot_task_manifest_v3.json"
SELECTION_MANIFEST = ROOT / "experiments/agentic_router/results/pilot_selection_manifest.json"
LIVE_LOCK = ROOT / "eval-suite/swebench/results" / pilot_campaign.PILOT_CAMPAIGN / "pilot_live.lock"
REPLAY_LOCK = pilot_campaign.LOCK_PATH
PILOT_LIVE_STATUS = LIVE_LOCK.parent / "pilot_live_status.json"
TASKS = (
    "swebench-flask-70ca03af28",
    "swebench-sympy-66abe976e0",
    "swebench-pytest-f8692a712a",
    "swebench-pylint-1715969d0b",
)
# The 1,865-test Django grader and >180s Astropy base/gold sanity were rejected
# for pilot runtime. Campaign confirmed immutable task manifest v3 and Sympy's
# passing 37.3s official base/gold check. This does not authorize launch while
# the collection controllers are running; locks and live status enforce that.
TASK_SELECTION_FROZEN = True
CONDITIONS = ("cloud", "local", "routing-learned-agentic-heuristic")
SEED = 1
AGENT_CAP_S = 1200
CELL_WALL_CAP_S = 1800
GPU_EXPIRY = datetime(2026, 9, 17, 9, 0, tzinfo=timezone.utc)


def _protocol_fingerprint() -> str:
    """Bind resumable cells to this exact frozen workload and runner code."""
    if not SHADOW.is_file():
        raise FileNotFoundError(SHADOW)
    if not TASK_MANIFEST.is_file():
        raise FileNotFoundError(TASK_MANIFEST)
    payload = {
        "schema_version": "agentic-live-matrix-protocol-v1",
        "experiment_id": EXPERIMENT,
        "tasks": TASKS,
        "conditions": CONDITIONS,
        "seed": SEED,
        "agent_cap_s": AGENT_CAP_S,
        "cell_wall_cap_s": CELL_WALL_CAP_S,
        "cloud_model_alias": "deepseek-v4-flash",
        "upstream": "https://lum.id/claude",
        "local_endpoint": "http://127.0.0.1:18004",
        "max_local_tokens": 100000,
        "local_token_margin": 0.90,
        "shadow_sha256": hashlib.sha256(SHADOW.read_bytes()).hexdigest(),
        "task_manifest_sha256": hashlib.sha256(TASK_MANIFEST.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w") as fh:
        json.dump(value, fh, sort_keys=True, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _cell_key(slug: str, condition: str) -> str:
    return f"{slug}__{condition}__seed{SEED}"


def _scoped_container(slug: str, condition: str) -> str:
    sys.path.insert(0, str(RUNNER.parent))
    import run_swebench  # type: ignore

    return run_swebench.container_name_for(EXPERIMENT, _cell_key(slug, condition))


def _summarize_trace(slug: str, condition: str) -> dict[str, Any]:
    path = collect.locate_trace_file(EXPERIMENT, slug, condition, SEED)
    if path is None:
        return {"trace_available": False}
    rows = []
    for row in collect.load_trace_records(path):
        if row.get("path") == "/v1/messages" and row.get("placement") in ("local", "cloud"):
            rows.append(row)
    by_backend = {name: {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                         "input_missing": 0, "output_missing": 0,
                         "cache_read_tokens": 0, "cache_read_missing": 0}
                  for name in ("local", "cloud")}
    response_models = set()
    shadow_complete = 0
    shadow_reasons: dict[str, int] = {}
    comparable_local = comparable_total = comparable_complete = 0
    for row in rows:
        placement = row["placement"]
        bucket = by_backend[placement]
        bucket["calls"] += 1
        accounting = row.get("token_accounting") or {}
        for key, output, missing in (
            ("input_tokens", "input_tokens", "input_missing"),
            ("output_tokens", "output_tokens", "output_missing"),
            ("cache_read_input_tokens", "cache_read_tokens", "cache_read_missing"),
        ):
            value = accounting.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bucket[output] += value
            else:
                bucket[missing] += 1
        model = (row.get("response") or {}).get("model") if isinstance(row.get("response"), dict) else None
        if model:
            response_models.add(str(model))
        shadow = row.get("agentic_shadow") or {}
        if shadow.get("feature_status") == "complete":
            shadow_complete += 1
        reason = shadow.get("unavailable_reason") or "none"
        shadow_reasons[reason] = shadow_reasons.get(reason, 0) + 1
        reference_input = (row.get("features") or {}).get("local_prompt_tokens")
        selected_output = accounting.get("output_tokens")
        if isinstance(reference_input, (int, float)) and isinstance(selected_output, (int, float)):
            comparable_complete += 1
            selected_load = reference_input + selected_output
            comparable_total += selected_load
            if placement == "local":
                comparable_local += selected_load
    calls = len(rows)
    trajectory = features.derive(collect.build_trajectory(
        collect.load_trace_records(path), task_group=slug, campaign=EXPERIMENT,
        condition=condition, seed=SEED, trace_path=path, task_passed=None,
        verdict_detail=None, verdict_path=Path("unused"),
    ))
    return {
        "trace_available": True,
        "trace_path": str(path),
        "n_calls": calls,
        "backend_usage": by_backend,
        "local_request_share": by_backend["local"]["calls"] / calls if calls else None,
        "comparable_selected_token_share_local_estimate": (
            comparable_local / comparable_total if comparable_total > 0 else None
        ),
        "comparable_token_coverage_calls": comparable_complete,
        "comparable_token_method": "exact local-rendered input prompt plus selected provider output; estimate, not cross-provider reported-token sum",
        "repair_loop_flagged_calls": sum(bool(c.derived.get("repair_loop_flag")) for c in trajectory.calls),
        "prior_truncated_or_invalid_calls": sum(bool(c.derived.get("prior_response_truncated_or_invalid")) for c in trajectory.calls),
        "recent_tool_error_flagged_calls": sum(bool(c.derived.get("recent_tool_error_count")) for c in trajectory.calls),
        "response_model_fingerprints": sorted(response_models),
        "shadow_feature_complete_calls": shadow_complete,
        "shadow_feature_coverage": shadow_complete / calls if calls else None,
        "shadow_unavailable_reasons": shadow_reasons,
    }


def _checkpoint(slug: str, condition: str, *, state: str, **extra: Any) -> dict[str, Any]:
    row = {
        "schema_version": "agentic-live-matrix-cell-v1",
        "experiment_id": EXPERIMENT,
        "slug": slug, "condition": condition, "seed": SEED,
        "state": state, "at_utc": datetime.now(timezone.utc).isoformat(),
        "drain_utc": pilot_campaign.DRAIN.astimezone(timezone.utc).isoformat(),
        "gpu_expiry_utc": GPU_EXPIRY.isoformat(),
        "shadow_artifact": str(SHADOW),
        "protocol_fingerprint": _protocol_fingerprint(),
        **extra,
    }
    _atomic(CHECKPOINTS / f"{_cell_key(slug, condition)}.json", row)
    return row


def _completed(slug: str, condition: str) -> bool:
    path = CHECKPOINTS / f"{_cell_key(slug, condition)}.json"
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    return (row.get("state") == "graded_valid"
            and row.get("protocol_fingerprint") == _protocol_fingerprint())


def _prior_checkpoint(slug: str, condition: str) -> dict[str, Any] | None:
    path = CHECKPOINTS / f"{_cell_key(slug, condition)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _sanity_valid(slug: str) -> bool:
    path = LIVE_LOCK.parent / "grader_sanity" / f"{slug}.sanity.json"
    if not path.is_file():
        return False
    sys.path.insert(0, str(RUNNER.parent))
    from run_swebench import load_instances  # type: ignore
    from verify_official_grader import fingerprint  # type: ignore
    try:
        saved = json.loads(path.read_text())
        instance = next(iter(load_instances(slug)))
        return (saved.get("sanity_pass") is True
                and saved.get("instance_id") == instance["instance_id"]
                and saved.get("fingerprint") == fingerprint(instance))
    except (OSError, RuntimeError, ValueError, StopIteration, json.JSONDecodeError):
        return False


def _pilot_finished() -> bool:
    if not PILOT_LIVE_STATUS.is_file():
        return False
    try:
        state = json.loads(PILOT_LIVE_STATUS.read_text()).get("state")
    except (OSError, json.JSONDecodeError):
        return False
    return state in ("finished_development", "drained")


def _selection_is_v3() -> bool:
    try:
        selected = json.loads(SELECTION_MANIFEST.read_text())
        frozen = json.loads(TASK_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    frozen_development = tuple(row["slug"] for row in frozen.get("tasks", ())
                               if row.get("split") == "development")
    frozen_holdout = tuple(row["slug"] for row in frozen.get("tasks", ())
                           if row.get("split") == "holdout")
    return (selected.get("version") == 3
            and frozen.get("version") == 3
            and frozen_development == pilot_campaign.DEVELOPMENT_TASKS
            and frozen_holdout == pilot_campaign.HOLDOUT_TASKS
            and set(TASKS).issubset(frozen_development)
            and tuple(selected.get("development_tasks", ())) == pilot_campaign.DEVELOPMENT_TASKS
            and tuple(selected.get("holdout_tasks", ())) == pilot_campaign.HOLDOUT_TASKS)


def _run_cell(slug: str, condition: str) -> dict[str, Any]:
    if not SHADOW.is_file():
        raise FileNotFoundError(SHADOW)
    if not _sanity_valid(slug):
        return _checkpoint(slug, condition, state="grader_sanity_unavailable")
    if PAUSE.exists() or (pilot_campaign.DRAIN - datetime.now(pilot_campaign.SGT)).total_seconds() <= 90:
        return _checkpoint(slug, condition, state="paused_or_drained")
    args = [
        sys.executable, str(RUNNER),
        "--instances", slug, "--conditions", condition, "--seeds", "1",
        "--cloud-parallelism", "1", "--local-parallelism", "1",
        "--claude-model", "deepseek-v4-flash",
        "--upstream", "https://lum.id/claude",
        "--vllm-url", "http://127.0.0.1:18004",
        "--max-local-tokens", "100000", "--local-token-margin", "0.90",
        "--job-timeout-s", str(AGENT_CAP_S),
        "--agentic-shadow-artifact", str(SHADOW),
        "--experiment-id", EXPERIMENT, "--results-dir", str(RESULTS),
    ]
    started = time.monotonic()
    _checkpoint(slug, condition, state="running", command=args)
    proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    # Drain means no request may still be running at 16:30 SGT, not merely
    # "do not start a new job". Reserve 60 seconds for process-group/Docker
    # cleanup before that deadline.
    timeout = min(
        CELL_WALL_CAP_S,
        max(1, (pilot_campaign.DRAIN - datetime.now(pilot_campaign.SGT)).total_seconds() - 60),
    )
    timed_out = False
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait(timeout=10)
    if timed_out or code != 0:
        subprocess.run(["docker", "rm", "-f", _scoped_container(slug, condition)],
                       capture_output=True, timeout=30)
    elapsed = time.monotonic() - started
    verdict_path = RESULTS / "verdicts" / f"{_cell_key(slug, condition)}.json"
    passed, detail = collect.load_verdict(verdict_path) if verdict_path.exists() else (None, None)
    trace = _summarize_trace(slug, condition)
    state = "graded_valid" if passed is not None and not timed_out and code == 0 else "infrastructure_invalid"
    return _checkpoint(slug, condition, state=state, passed=passed, verdict_detail=detail,
                       runner_returncode=code, controller_timed_out=timed_out,
                       wall_time_s=round(elapsed, 3), **trace)


def run() -> dict[str, Any]:
    if not TASK_SELECTION_FROZEN:
        return {"state": "task_selection_pending_replacement"}
    protocol_fingerprint = _protocol_fingerprint()
    RESULTS.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (LIVE_LOCK, REPLAY_LOCK):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if PAUSE.exists() or (pilot_campaign.DRAIN - datetime.now(pilot_campaign.SGT)).total_seconds() <= 90:
            status = {"state": "paused_or_drained", "completed_cells": 0}
            _atomic(STATUS, status)
            return status
        if not _pilot_finished():
            status = {"state": "pilot_collector_not_finished",
                      "at_utc": datetime.now(timezone.utc).isoformat()}
            _atomic(STATUS, status)
            return status
        if not _selection_is_v3():
            status = {"state": "selection_manifest_mismatch",
                      "at_utc": datetime.now(timezone.utc).isoformat()}
            _atomic(STATUS, status)
            return status
        for slug in TASKS:
            for condition in CONDITIONS:
                prior = _prior_checkpoint(slug, condition)
                if prior is not None and prior.get("protocol_fingerprint") != protocol_fingerprint:
                    status = {"state": "manual_review_required", "slug": slug,
                              "condition": condition, "prior_state": prior.get("state"),
                              "reason": "checkpoint_protocol_mismatch"}
                    _atomic(STATUS, status)
                    return status
                if prior is not None and prior.get("state") != "graded_valid":
                    status = {"state": "manual_review_required", "slug": slug,
                              "condition": condition, "prior_state": prior.get("state")}
                    _atomic(STATUS, status)
                    return status
        if all(_completed(slug, condition) for slug in TASKS for condition in CONDITIONS):
            status = {"state": "complete", "completed_this_run": 0,
                      "at_utc": datetime.now(timezone.utc).isoformat()}
            _atomic(STATUS, status)
            return status
        # Both model identities must be verified before any 12-cell spend.
        # These bounded preflights run only on explicit --run.
        try:
            pilot_campaign._bounded_flight(pilot_campaign.LOCAL_27B.name)
            pilot_campaign._bounded_flight(pilot_campaign.CLOUD_DEEPSEEK.name)
        except Exception as exc:
            status = {"state": "preflight_failed", "error_type": type(exc).__name__,
                      "at_utc": datetime.now(timezone.utc).isoformat()}
            _atomic(STATUS, status)
            return status
        out = []
        for slug in TASKS:
            for condition in CONDITIONS:
                if PAUSE.exists() or (pilot_campaign.DRAIN - datetime.now(pilot_campaign.SGT)).total_seconds() <= 90:
                    status = {"state": "paused_or_drained", "completed_cells": len(out)}
                    _atomic(STATUS, status)
                    return status
                prior = _prior_checkpoint(slug, condition)
                if prior is not None and prior.get("state") == "graded_valid":
                    continue
                result = _run_cell(slug, condition)
                out.append({"slug": slug, "condition": condition, "state": result["state"]})
                _atomic(STATUS, {"state": "running", "latest": out[-1], "completed_this_run": len(out),
                                 "at_utc": datetime.now(timezone.utc).isoformat()})
                if result["state"] != "graded_valid":
                    return {"state": "halted_on_invalid_cell", "latest": out[-1]}
        status = {"state": "complete", "completed_this_run": len(out),
                  "at_utc": datetime.now(timezone.utc).isoformat()}
        _atomic(STATUS, status)
        return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly start the matrix under both locks")
    args = parser.parse_args()
    if not args.run:
        parser.error("pass --run only after campaign coordination; this script never auto-starts")
    result = run()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] in ("complete", "paused_or_drained") else 1


if __name__ == "__main__":
    raise SystemExit(main())
