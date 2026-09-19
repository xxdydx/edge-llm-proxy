"""One frozen, guarded Matplotlib cloud-vs-static trajectory pair.

Never import this module to start work. `--execute` is required. No retries,
holdouts, router edits, broad Docker cleanup, or model selection occur here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parent.parent
SLUG = "swebench-matplotlib-c968b9e065"
EXPERIMENT = "agentic-router-policy-pair-matplotlib-v1"
IMAGE = "swebench/sweb.eval.x86_64.matplotlib_1776_matplotlib-20676:latest"
IMAGE_ID = "sha256:7f603ecba31ab182ad9b6c2005d4810b6b52c0e238e662820339e21319ca6871"
TASK_SHA256 = "556b6f5a6eca542568b87681f393b2762800f6afa2fdd20782c966930def4e23"
EVAL_SHA256 = "fe45881a5ad5943566d66a6af35d973fb630b64e6541b6affa9eabc13dd1780f"
PROTOCOL = ROOT / "experiments/agentic_router/results/policy_pair_matplotlib_v1_protocol.md"
TASK = ROOT / "eval-suite/swebench/instances" / f"{SLUG}.json"
RUNNER = ROOT / "eval-suite/swebench/runner/run_swebench.py"
GRADER = ROOT / "eval-suite/swebench/runner/verify_official_grader.py"
RESULTS = ROOT / "eval-suite/swebench/results" / EXPERIMENT
RETRY_RESULTS = ROOT / "eval-suite/swebench/results" / f"{EXPERIMENT}-attempt2"
FIRST_STATUS = RESULTS / "controller_status.json"
RETRY_AMENDMENT = ROOT / "experiments/agentic_router/results/policy_pair_matplotlib_v1_disk_gate_retry_amendment.md"
LOCK = ROOT / "experiments/agentic_router/results/pilot_campaign.lock"
SGT = dt.timezone(dt.timedelta(hours=8))
DRAIN = dt.datetime(2026, 9, 17, 16, 30, tzinfo=SGT)
LATEST_PAIR_START = dt.datetime(2026, 9, 17, 15, 0, tzinfo=SGT)
NINE_GIB = 9 * 1024**3
SIX_GIB = 6 * 1024**3
EIGHT_GIB = 8 * 1024**3
MIN_POST_PULL_FREE = SIX_GIB
ORDERS = ("cloud", "routing")
AGENT_CAP_S = 1200
ARM_CAP_S = 2400
GRADER_CAP_S = 180
PULL_CAP_S = 900
DISK_IMAGE = "swebench/sweb.eval.x86_64.pallets_1776_flask-5014:latest"


class GateFailure(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> dt.datetime:
    return dt.datetime.now(SGT)


def _remaining() -> float:
    return (DRAIN - _now()).total_seconds()


def _atomic_status(**fields: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    row = {"schema_version": "matplotlib-policy-pair-status-v1",
           "updated_at_sgt": _now().isoformat(), "drain_sgt": DRAIN.isoformat(),
           "experiment": EXPERIMENT, **fields}
    path = RESULTS / "controller_status.json"
    tmp = RESULTS / f".controller_status.{os.getpid()}.tmp"
    with tmp.open("w") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _run(args: list[str], cap_s: float, log_name: str, *, container: str | None = None) -> int:
    if _remaining() <= cap_s + 30:
        raise GateFailure(f"insufficient_drain_budget_for:{log_name}")
    log_path = RESULTS / log_name
    with log_path.open("x") as log:
        proc = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        started = time.monotonic()
        _atomic_status(stage=log_name, state="running", pid=proc.pid,
                       started_at_sgt=_now().isoformat())
        try:
            while True:
                remaining = min(cap_s - (time.monotonic() - started), _remaining() - 20)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, cap_s)
                try:
                    code = proc.wait(timeout=min(10, remaining))
                    break
                except subprocess.TimeoutExpired:
                    _atomic_status(stage=log_name, state="running", pid=proc.pid,
                                   elapsed_s=round(time.monotonic() - started, 1))
        except subprocess.TimeoutExpired as exc:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            _cleanup_container(container)
            raise GateFailure(f"timeout_or_drain:{log_name}") from exc
    if code != 0:
        _cleanup_container(container)
        raise GateFailure(f"child_exit_{code}:{log_name}")
    if container is not None and _container_exists(container):
        _cleanup_container(container)
        raise GateFailure(f"container_survived_runner:{container}")
    return code


def _container_exists(name: str) -> bool:
    result = subprocess.run(["docker", "container", "inspect", name],
                            capture_output=True, text=True, timeout=15)
    return result.returncode == 0


def _cleanup_container(name: str | None) -> None:
    if name is not None and _container_exists(name):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


def _container_name(condition: str) -> str:
    if condition == "grader":
        return f"flowmesh-grader-check-{SLUG}"
    sys.path.insert(0, str(RUNNER.parent))
    from run_swebench import container_name_for  # type: ignore
    return container_name_for(EXPERIMENT, f"{SLUG}__{condition}__seed1")


def _docker_free() -> int:
    disk_image = IMAGE if _image_id() is not None else DISK_IMAGE
    proc = subprocess.run(
        ["docker", "run", "--rm", "--pull", "never", "--platform", "linux/amd64",
         "--entrypoint", "df", disk_image, "-B1", "/"],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise GateFailure("docker_vm_free_unavailable")
    lines = [line.split() for line in proc.stdout.splitlines() if line.startswith("overlay")]
    if len(lines) != 1 or len(lines[0]) < 4:
        raise GateFailure("docker_vm_free_unparseable")
    return int(lines[0][3])


def _image_id() -> str | None:
    proc = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE],
                          capture_output=True, text=True, timeout=20)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _no_other_task() -> None:
    proc = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=15)
    if proc.returncode != 0 or proc.stdout.strip():
        raise GateFailure("other_docker_container_active_or_uninspectable")
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=15)
    if ps.returncode != 0:
        raise GateFailure("task_process_inventory_unavailable")
    needles = ("run_swebench.py", "pilot_live.py", "pilot_campaign.py",
               "live_matrix.py", "claude -p")
    for line in ps.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or int(parts[0]) == os.getpid():
            continue
        if any(word in parts[1] for word in needles):
            raise GateFailure(f"other_task_process_active:pid={parts[0]}")
    try:
        with httpx.Client(base_url="http://127.0.0.1:18004", timeout=12) as client:
            models = client.get("/v1/models")
            metrics = client.get("/metrics")
        models.raise_for_status()
        metrics.raise_for_status()
        ids = [row.get("id") for row in models.json().get("data", [])]
        if ids != ["local"]:
            raise GateFailure(f"local_model_identity_changed:{ids}")
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            values = [float(line.split()[-1]) for line in metrics.text.splitlines()
                      if line.startswith(name + "{")]
            if not values or any(value != 0 for value in values):
                raise GateFailure(f"local_busy_or_metric_missing:{name}")
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise GateFailure(f"local_relay_preflight_error:{type(exc).__name__}") from exc


def _verify_frozen_inputs() -> None:
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise GateFailure("must_use_repository_venv_python")
    if _now() >= LATEST_PAIR_START:
        raise GateFailure("pair_start_cutoff_passed")
    if RESULTS.exists():
        raise GateFailure("fresh_results_namespace_already_exists")
    if not PROTOCOL.is_file() or _sha(TASK) != TASK_SHA256:
        raise GateFailure("protocol_or_task_hash_mismatch")
    task = json.loads(TASK.read_text())
    if (task.get("instance_id") != "matplotlib__matplotlib-20676"
            or task.get("docker_image") != IMAGE
            or hashlib.sha256(task["eval_script"].encode()).hexdigest() != EVAL_SHA256):
        raise GateFailure("task_instance_or_eval_script_changed")
    if not LOCK.is_file():
        raise GateFailure("campaign_lock_missing")
    try:
        from dotenv import dotenv_values
        credential_present = bool(os.environ.get("ANTHROPIC_AUTH_TOKEN") or
                                  dotenv_values(ROOT / ".env").get("ANTHROPIC_AUTH_TOKEN"))
    except (OSError, ValueError) as exc:
        raise GateFailure("cloud_credential_presence_uncheckable") from exc
    if not credential_present:
        raise GateFailure("cloud_credential_absent")


def _official_sanity() -> None:
    out = RESULTS / "grader_sanity"
    out.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, str(GRADER), SLUG, "--out-dir", str(out)],
         GRADER_CAP_S, "grader_sanity.controller.log", container=_container_name("grader"))
    path = out / f"{SLUG}.sanity.json"
    if not path.is_file():
        raise GateFailure("official_sanity_missing")
    row = json.loads(path.read_text())
    base, gold = row.get("base") or {}, row.get("gold") or {}
    if (not row.get("sanity_pass") or row.get("fingerprint") != {
            "base_commit": "6786f437df54ca7780a047203cbcfaa1db8dc542",
            "eval_script_sha256": EVAL_SHA256, "docker_image_id": IMAGE_ID}
            or base.get("n_fail_to_pass_passed") != 0
            or base.get("n_pass_to_pass_passed") != 32
            or gold.get("n_fail_to_pass_passed") != 2
            or gold.get("n_pass_to_pass_passed") != 32):
        raise GateFailure("official_sanity_or_source_integrity_failed")


def _arm(condition: str) -> dict[str, Any]:
    _no_other_task()
    if _docker_free() < MIN_POST_PULL_FREE:
        raise GateFailure("docker_space_below_frozen_floor_before_arm")
    arm_dir = RESULTS / condition
    arm_dir.mkdir()
    args = [sys.executable, str(RUNNER), "--instances", SLUG, "--conditions", condition,
            "--seeds", "1", "--cloud-parallelism", "1", "--local-parallelism", "1",
            "--claude-model", "deepseek-v4-flash", "--upstream", "https://lum.id/claude",
            "--vllm-url", "http://127.0.0.1:18004", "--max-local-tokens", "100000",
            "--local-token-margin", "0.90", "--job-timeout-s", str(AGENT_CAP_S),
            "--experiment-id", EXPERIMENT, "--results-dir", str(arm_dir)]
    _run(args, ARM_CAP_S, f"{condition}.controller.log", container=_container_name(condition))
    verdict = arm_dir / "verdicts" / f"{SLUG}__{condition}__seed1.json"
    if not verdict.is_file():
        raise GateFailure(f"official_verdict_missing:{condition}")
    row = json.loads(verdict.read_text())
    if row.get("error") is not None or row.get("reason") == "job did not complete":
        raise GateFailure(f"ungraded_or_infrastructure_error:{condition}:{row.get('error')}")
    if type(row.get("passed")) is not bool:
        raise GateFailure(f"official_verdict_invalid:{condition}")
    return {"condition": condition, "passed": row["passed"],
            "reason": row.get("reason"), "verdict_path": str(verdict)}


def main() -> int:
    global RESULTS, MIN_POST_PULL_FREE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="explicit live opt-in")
    parser.add_argument("--disk-gate-retry", action="store_true",
                        help="one separately named retry after attempt 1 stopped before any model arm")
    args = parser.parse_args()
    if not args.execute:
        parser.error("no work started; --execute required")
    if args.disk_gate_retry:
        if not RETRY_AMENDMENT.is_file() or not FIRST_STATUS.is_file():
            raise GateFailure("retry_amendment_or_first_status_missing")
        prior = json.loads(FIRST_STATUS.read_text())
        if (prior.get("state") != "stopped_invalid"
                or not str(prior.get("error", "")).startswith("GateFailure:docker_space_below_6GiB_after_pull:")
                or prior.get("completed") != []):
            raise GateFailure("prior_attempt_not_exact_disk_gate_only_failure")
        RESULTS = RETRY_RESULTS
        MIN_POST_PULL_FREE = EIGHT_GIB
    _verify_frozen_inputs()
    with LOCK.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GateFailure("campaign_lock_busy") from exc
        RESULTS.mkdir(parents=True, exist_ok=False)
        outcomes: list[dict[str, Any]] = []
        try:
            _atomic_status(stage="preflight", state="started",
                           protocol_sha256=_sha(PROTOCOL), task_sha256=_sha(TASK),
                           runner_sha256=_sha(RUNNER), controller_sha256=_sha(Path(__file__)),
                           retry_amendment_sha256=(_sha(RETRY_AMENDMENT) if args.disk_gate_retry else None),
                           attempt=(2 if args.disk_gate_retry else 1), conditions=ORDERS)
            _no_other_task()
            free = _docker_free()
            if free < NINE_GIB:
                raise GateFailure(f"docker_space_below_9GiB_before_pull:{free}")
            if _image_id() is None:
                _run(["docker", "pull", "--platform", "linux/amd64", IMAGE],
                     PULL_CAP_S, "image_pull.controller.log")
            if _image_id() != IMAGE_ID:
                raise GateFailure("pulled_image_id_differs_from_frozen_sanity")
            free = _docker_free()
            if free < MIN_POST_PULL_FREE:
                raise GateFailure(f"docker_space_below_frozen_floor_after_pull:{free}")
            _atomic_status(stage="grader_sanity", state="starting", docker_free_bytes=free,
                           image_id=IMAGE_ID)
            _official_sanity()
            if _docker_free() < MIN_POST_PULL_FREE:
                raise GateFailure("docker_space_below_frozen_floor_after_sanity")
            if _now() >= LATEST_PAIR_START or _remaining() < 2 * ARM_CAP_S + 300:
                raise GateFailure("insufficient_time_for_two_arms")
            for condition in ORDERS:
                _atomic_status(stage=condition, state="starting", completed=outcomes)
                outcomes.append(_arm(condition))
                _atomic_status(stage=condition, state="complete", completed=outcomes)
            _atomic_status(stage="pair", state="complete", completed=outcomes)
            print(json.dumps({"state": "complete", "outcomes": outcomes}))
            return 0
        except Exception as exc:
            _atomic_status(stage="pair", state="stopped_invalid",
                           completed=outcomes, error=f"{type(exc).__name__}:{exc}")
            print(f"stopped_invalid: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
