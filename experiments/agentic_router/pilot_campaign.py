"""Bounded, resumable Stage-1 paired-call pilot.

Only seven preregistered Python tasks are eligible in the frozen v3 plan.
Five are development groups; two are reserved holdouts.  A trajectory contributes at most eight
calls: six spread over its lifetime plus two seeded uniform draws.  Each
backend arm is an isolated subprocess with a hard wall timeout and its own
atomic checkpoint, so a long local stream cannot consume the whole night or
cause a successful cloud arm to be repeated after interruption.

Usage: python -m experiments.agentic_router.pilot_campaign --once
       python -m experiments.agentic_router.pilot_campaign --follow

The follow mode polls newly completed cloud trajectories.  It never launches
after the fixed 2026-09-17 16:30 SGT drain deadline.  No task agents are
launched here: run_swebench.py owns trajectory generation and grading.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from experiments.capability_router import executor as cr_executor
from experiments.capability_router.config import CLOUD_DEEPSEEK, LOCAL_27B

from . import collect, judge, replay, run_judge
from .schema import AgentCallRecord, AgentReplayOutcome, PairedCallExample

SGT = timezone(timedelta(hours=8))
DRAIN = datetime(2026, 9, 17, 16, 30, tzinfo=SGT)
# All requests, including any child process cleanup, must finish by DRAIN.
WORK_STOP = DRAIN - timedelta(seconds=90)
ARM_TIMEOUT_S = 900
POLL_S = 60
PILOT_BATCH = os.environ.get("FLOWMESH_POSTPILOT_BATCH", "v4")
if PILOT_BATCH not in ("v4", "v5", "v6", "v7", "v8", "v9"):
    raise RuntimeError("FLOWMESH_POSTPILOT_BATCH must be v4, v5, v6, v7, v8, or v9")
POSTPILOT_V5 = PILOT_BATCH == "v5"
POSTPILOT_V6 = PILOT_BATCH == "v6"
POSTPILOT_V7 = PILOT_BATCH == "v7"
POSTPILOT_V8 = PILOT_BATCH == "v8"
POSTPILOT_V9 = PILOT_BATCH == "v9"
POSTPILOT = POSTPILOT_V5 or POSTPILOT_V6 or POSTPILOT_V7 or POSTPILOT_V8 or POSTPILOT_V9
PILOT_CAMPAIGN = (f"agentic-router-postpilot-{PILOT_BATCH}" if POSTPILOT
                  else "agentic-router-pilot-v1")
V4_DEVELOPMENT_TASKS = (
    "swebench-flask-70ca03af28",       # flask-5014
    "swebench-sympy-66abe976e0",      # sympy-11618; replaces django-10097
    "swebench-pytest-f8692a712a",     # pytest-10051
    "swebench-pylint-1715969d0b",    # pylint-4604
    "swebench-sphinx-e786135d9b",    # sphinx-10323
)
V5_DEVELOPMENT_TASKS = (
    "swebench-requests-8f6749ed89",       # requests-1142
    "swebench-seaborn-1919a20c17",       # seaborn-3069
    "swebench-scikit-learn-f1b078302b",  # scikit-learn-10297
    "swebench-xarray-405877cffa",        # xarray-3095
)
V6_DEVELOPMENT_TASKS = (
    "swebench-requests-60df5b680e",     # requests-1724
    "swebench-xarray-ae71222844",       # xarray-2905
    "swebench-seaborn-3ca9f75c90",      # seaborn-3187
    "swebench-django-4730fbf7b3",       # django-10554
    "swebench-astropy-5d7dcee7c8",      # astropy-13033
    "swebench-matplotlib-e5d93f5446",   # matplotlib-13989
)
V7_DEVELOPMENT_TASKS = (
    "swebench-pytest-18946c8bcb",      # pytest-7571
    "swebench-pylint-7cebb03cec",      # pylint-6386
    "swebench-scikit-learn-035f33ceae",  # scikit-learn-14053
    "swebench-xarray-13e7d5e1cb",      # xarray-3677
    "swebench-matplotlib-ae2b36dac7",  # matplotlib-24637
    "swebench-sympy-3b20aad08b",       # sympy-24443
    "swebench-sphinx-8a6b6d1307",     # sphinx-9711
    "swebench-astropy-f6db0651f2",    # astropy-14365
)
V8_DEVELOPMENT_TASKS = (
    "swebench-pylint-edcd13a009",       # pylint-6903
    "swebench-scikit-learn-a4a59eac2d",  # scikit-learn-13328
    "swebench-pytest-4c87720313",       # pytest-5631
    "swebench-matplotlib-c968b9e065",  # matplotlib-20676
    "swebench-xarray-ef020e73d9",      # xarray-3151
    "swebench-requests-c040777d5b",   # requests-2931
    "swebench-sphinx-cb0a7e725f",     # sphinx-8721
    "swebench-sympy-e7a894a5c0",      # sympy-20916
)
V9_DEVELOPMENT_TASKS = (
    "swebench-sympy-821769a47e",       # sympy-12481
    "swebench-pylint-2b3eaf8a4a",      # pylint-4970
    "swebench-scikit-learn-771f60c2cb",  # scikit-learn-13779
    "swebench-xarray-bb3811ed23",      # xarray-4629
    "swebench-pytest-6d1623a5d4",      # pytest-8399
    "swebench-requests-ee058aedf2",   # requests-1766
)
DEVELOPMENT_TASKS = (V9_DEVELOPMENT_TASKS if POSTPILOT_V9 else
                     V8_DEVELOPMENT_TASKS if POSTPILOT_V8 else
                     V7_DEVELOPMENT_TASKS if POSTPILOT_V7 else
                     V6_DEVELOPMENT_TASKS if POSTPILOT_V6 else
                     V5_DEVELOPMENT_TASKS if POSTPILOT_V5 else V4_DEVELOPMENT_TASKS)
SUPERSEDED_V1_TASKS = (
    "swebench-flask-70ca03af28", "swebench-django-98fd319fe6",
    "swebench-django-4730fbf7b3", "swebench-pytest-f8692a712a",
    "swebench-pylint-1715969d0b", "swebench-sphinx-e786135d9b",
)
SUPERSEDED_V2_TASKS = (
    "swebench-flask-70ca03af28", "swebench-sympy-66abe976e0",
    "swebench-astropy-29ce4051fa", "swebench-pytest-f8692a712a",
    "swebench-pylint-1715969d0b", "swebench-sphinx-e786135d9b",
)
V4_HOLDOUT_TASKS = (
    "swebench-pytest-c89d81bc3b",     # pytest-10081
    "swebench-pylint-a520f18b1b",    # pylint-4551
)
HOLDOUT_TASKS = () if POSTPILOT else V4_HOLDOUT_TASKS
RESULTS = Path(__file__).resolve().parent / "results"
if POSTPILOT:
    RESULTS = RESULTS / f"postpilot_{PILOT_BATCH}"
ARM_DIR = RESULTS / "pilot_arm_checkpoints"
SELECTION_PATH = RESULTS / "pilot_selection_manifest.json"
PAIRED_PATH = RESULTS / "stage1_replay_examples_pilot_v1.jsonl"
STATUS_PATH = RESULTS / "pilot_campaign_status.json"
LOCK_PATH = RESULTS / "pilot_campaign.lock"
# Keep prior fixed-primary-order judgments immutable. New judgments use the
# per-call counterbalanced v5 order protocol in a separate append-only file.
LEGACY_JUDGE_PATH = RESULTS / "pilot_judge_consistency_v4.jsonl"
JUDGE_PATH = RESULTS / "pilot_judge_consistency_v5_order.jsonl"
PAUSE_PATH = RESULTS / "pilot_replay_paused.json"
JUDGE_FAILURE_PATH = RESULTS / "pilot_last_judge_failure.json"
TERMINAL_PARTIAL_PATH = RESULTS / "pilot_terminal_partial_manifest.json"


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as fh:
        json.dump(data, fh, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(data) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _before_drain() -> bool:
    return datetime.now(SGT) < WORK_STOP


def select_call_indices(n: int, trajectory_id: str) -> list[int]:
    """Six spread anchors plus two seeded uniform draws from non-anchors.

    Anchors have inclusion probability 1. Each other eligible call has
    probability 2/(n-6) for n>8, conditional on the deterministic anchors.
    This is not an unconditional uniform sample of the trajectory.
    """
    if n <= 8:
        return list(range(n))
    anchors = {round(i * (n - 1) / 5) for i in range(6)}
    remaining = sorted(set(range(n)) - anchors)
    seed = int.from_bytes(hashlib.sha256(trajectory_id.encode()).digest()[:8], "big")
    uniform = random.Random(seed).sample(remaining, k=min(2, len(remaining)))
    return sorted(anchors | set(uniform))


V7_SELECTION_PROTOCOL = "v7-one-endpoint-seven-uniform-interior-2026-09-17"
V7_SELECTION_SEED = 20260917
V8_SELECTION_PROTOCOL = "v8-one-endpoint-seven-uniform-interior-2026-09-17"
V8_SELECTION_SEED = 20260917
V9_SELECTION_PROTOCOL = "v9-one-endpoint-seven-uniform-interior-2026-09-17"
V9_SELECTION_SEED = 20260917


def _select_call_indices_one_endpoint_seven(
    call_ids: list[str], trajectory_id: str, batch: str, seed: int
) -> tuple[list[int], dict[str, float]]:
    """Outcome-blind, hash-stable one endpoint + seven interior draw.

    The hash-rank implementation is deterministic, but inclusion probabilities
    describe the preregistered uniform randomization over the hash assignment.
    """
    n = len(call_ids)
    if len(set(call_ids)) != n:
        raise ValueError("Duplicate eligible call ID in trajectory")
    if n <= 8:
        return list(range(n)), {cid: 1.0 for cid in call_ids}

    def rank(index: int, stratum: str) -> str:
        key = f"{batch}|{seed}|one_endpoint7|{trajectory_id}|{stratum}|{call_ids[index]}"
        return hashlib.sha256(key.encode()).hexdigest()

    endpoint = min((0, n - 1), key=lambda i: (rank(i, "endpoint"), call_ids[i]))
    interior = sorted(range(1, n - 1), key=lambda i: (rank(i, "interior"), call_ids[i]))[:7]
    probabilities = {cid: (0.5 if i in (0, n - 1) else 7 / (n - 2))
                     for i, cid in enumerate(call_ids)}
    return sorted([endpoint, *interior]), probabilities


def select_call_indices_v7(call_ids: list[str], trajectory_id: str) -> tuple[list[int], dict[str, float]]:
    return _select_call_indices_one_endpoint_seven(
        call_ids, trajectory_id, "v7", V7_SELECTION_SEED)


def select_call_indices_v8(call_ids: list[str], trajectory_id: str) -> tuple[list[int], dict[str, float]]:
    return _select_call_indices_one_endpoint_seven(
        call_ids, trajectory_id, "v8", V8_SELECTION_SEED)


def select_call_indices_v9(call_ids: list[str], trajectory_id: str) -> tuple[list[int], dict[str, float]]:
    return _select_call_indices_one_endpoint_seven(
        call_ids, trajectory_id, "v9", V9_SELECTION_SEED)


def eligible_main_call(call: AgentCallRecord) -> bool:
    """Exclude clear security/title/child calls from the eight-call pilot.

    Tool-less first turns are genuine but uninformative for a tool-action
    router; their frequency is still observable in the complete trajectory.
    A nonnull agent_id alone is ambiguous: it can identify a root agent, so
    it is recorded in the selection manifest but not excluded by itself.
    """
    return (not bool(call.router_features.get("is_security_monitor"))
            and bool(call.request.get("tools"))
            and not bool(call.causality.get("agent_parent_tool_use_id")))


def _load_selection() -> dict[str, Any]:
    if not SELECTION_PATH.exists():
        return {"version": 9 if POSTPILOT_V9 else 8 if POSTPILOT_V8 else 7 if POSTPILOT_V7 else 6 if POSTPILOT_V6 else 5 if POSTPILOT_V5 else 3,
                "development_tasks": DEVELOPMENT_TASKS,
                "holdout_tasks": HOLDOUT_TASKS, "trajectories": {}}
    data = json.loads(SELECTION_PATH.read_text())
    if POSTPILOT and data.get("version") != (9 if POSTPILOT_V9 else 8 if POSTPILOT_V8 else 7 if POSTPILOT_V7 else 6 if POSTPILOT_V6 else 5):
        raise RuntimeError("Postpilot selection manifest version mismatch")
    if data.get("version") == 1 and tuple(data.get("development_tasks", ())) == SUPERSEDED_V1_TASKS:
        if any(entry["task_group"] != "swebench-flask-70ca03af28"
               for entry in data.get("trajectories", {}).values()):
            raise RuntimeError("Cannot migrate v1 selection: non-Flask calls were already selected")
        archive = RESULTS / "pilot_selection_manifest_v1.json"
        if archive.exists() and json.loads(archive.read_text()) != data:
            raise RuntimeError("v1 selection archive differs; refusing migration")
        if not archive.exists():
            _atomic_json(archive, data)
        data = {**data, "version": 2, "development_tasks": SUPERSEDED_V2_TASKS,
                "supersedes": archive.name,
                "substitution_reason": "Django official grading too slow for preliminary pilot; replaced before Django model generation"}
        _atomic_json(SELECTION_PATH, data)
    if data.get("version") == 2 and tuple(data.get("development_tasks", ())) == SUPERSEDED_V2_TASKS:
        if any(entry["task_group"] != "swebench-flask-70ca03af28"
               for entry in data.get("trajectories", {}).values()):
            raise RuntimeError("Cannot migrate v2 selection: a replacement task already has selected calls")
        archive = RESULTS / "pilot_selection_manifest_v2.json"
        if archive.exists() and json.loads(archive.read_text()) != data:
            raise RuntimeError("v2 selection archive differs; refusing migration")
        if not archive.exists():
            _atomic_json(archive, data)
        data = {**data, "version": 3, "development_tasks": DEVELOPMENT_TASKS,
                "supersedes": archive.name,
                "substitution_reason": "Astropy base+gold preflight exceeded 180s; keep five fast development tasks"}
        _atomic_json(SELECTION_PATH, data)
    if tuple(data["development_tasks"]) != DEVELOPMENT_TASKS or tuple(data["holdout_tasks"]) != HOLDOUT_TASKS:
        raise RuntimeError("Pilot task set changed after selection freeze; refusing to resample")
    return data


def _selected_calls(include_holdout: bool = False) -> list[AgentCallRecord]:
    trajectories, _ = collect.build_trajectories()
    manifest = _load_selection()
    all_tasks = set(DEVELOPMENT_TASKS + (HOLDOUT_TASKS if include_holdout else ()))
    selected: list[AgentCallRecord] = []
    for trajectory in sorted(trajectories, key=lambda t: t.trajectory_id):
        if (trajectory.campaign != PILOT_CAMPAIGN or trajectory.condition != "cloud"
                or trajectory.task_group not in all_tasks or trajectory.task_passed is None
                or not trajectory.calls):
            continue
        eligible = [(i, call) for i, call in enumerate(trajectory.calls) if eligible_main_call(call)]
        if not eligible:
            continue
        chosen = manifest["trajectories"].get(trajectory.trajectory_id)
        if chosen is None:
            eligible_ids = [call.call_id for _, call in eligible]
            if POSTPILOT_V7 or POSTPILOT_V8 or POSTPILOT_V9:
                selector = (select_call_indices_v9 if POSTPILOT_V9 else
                            select_call_indices_v8 if POSTPILOT_V8 else select_call_indices_v7)
                eligible_indices, probabilities = selector(eligible_ids, trajectory.trajectory_id)
            else:
                eligible_indices = select_call_indices(len(eligible), trajectory.trajectory_id)
            indices = [eligible[i][0] for i in eligible_indices]
            chosen = {"task_group": trajectory.task_group,
                      "split": "holdout" if trajectory.task_group in HOLDOUT_TASKS else "development",
                      "n_trajectory_calls": len(trajectory.calls),
                      "n_eligible_main_calls": len(eligible),
                      "n_agent_id_marked_eligible": sum(bool(c.causality.get("agent_id")) for _, c in eligible),
                      "eligibility": "tools_nonempty_and_not_security_and_no_parent_tool_use_id; agent_id_role_ambiguous",
                      "nonanchor_inclusion_probability": (
                          min(1.0, 2 / (len(eligible) - 6)) if len(eligible) > 8 else 1.0
                      ),
                      "indices": indices,
                      "call_ids": [trajectory.calls[i].call_id for i in indices]}
            if POSTPILOT_V7 or POSTPILOT_V8 or POSTPILOT_V9:
                trace_path = Path(eligible[0][1].source_trace_path)
                with trace_path.open("rb") as fh:
                    trace_sha = hashlib.file_digest(fh, "sha256").hexdigest()
                chosen.update({
                    "selection_protocol": V9_SELECTION_PROTOCOL if POSTPILOT_V9 else V8_SELECTION_PROTOCOL if POSTPILOT_V8 else V7_SELECTION_PROTOCOL,
                    "selection_seed": V9_SELECTION_SEED if POSTPILOT_V9 else V8_SELECTION_SEED if POSTPILOT_V8 else V7_SELECTION_SEED,
                    "eligible_call_ids": eligible_ids,
                    "source_trace_sha256": trace_sha,
                    "inclusion_probability_by_call_id": probabilities,
                    "nonanchor_inclusion_probability": None,
                })
            manifest["trajectories"][trajectory.trajectory_id] = chosen
            _atomic_json(SELECTION_PATH, manifest)
        elif POSTPILOT_V7 or POSTPILOT_V8 or POSTPILOT_V9:
            eligible_ids = [call.call_id for _, call in eligible]
            expected_protocol = V9_SELECTION_PROTOCOL if POSTPILOT_V9 else V8_SELECTION_PROTOCOL if POSTPILOT_V8 else V7_SELECTION_PROTOCOL
            expected_seed = V9_SELECTION_SEED if POSTPILOT_V9 else V8_SELECTION_SEED if POSTPILOT_V8 else V7_SELECTION_SEED
            if (chosen.get("selection_protocol") != expected_protocol
                    or chosen.get("selection_seed") != expected_seed
                    or chosen.get("eligible_call_ids") != eligible_ids):
                raise RuntimeError("Frozen eligibility or selection protocol changed")
            with Path(eligible[0][1].source_trace_path).open("rb") as fh:
                trace_sha = hashlib.file_digest(fh, "sha256").hexdigest()
            if chosen.get("source_trace_sha256") != trace_sha:
                raise RuntimeError("Frozen source trace changed")
            selector = (select_call_indices_v9 if POSTPILOT_V9 else
                        select_call_indices_v8 if POSTPILOT_V8 else select_call_indices_v7)
            selected_eligible_indices, expected_probabilities = selector(
                eligible_ids, trajectory.trajectory_id)
            expected_indices = [eligible[i][0] for i in selected_eligible_indices]
            expected_call_ids = [trajectory.calls[i].call_id for i in expected_indices]
            if (chosen.get("indices") != expected_indices
                    or chosen.get("call_ids") != expected_call_ids
                    or chosen.get("inclusion_probability_by_call_id") != expected_probabilities
                    or chosen.get("n_eligible_main_calls") != len(eligible)):
                raise RuntimeError("Frozen selection indices or inclusion probabilities changed")
        by_id = {call.call_id: call for call in trajectory.calls}
        for call_id in chosen["call_ids"]:
            if call_id not in by_id:
                raise RuntimeError(f"Frozen selection changed underneath us: {trajectory.trajectory_id}")
            selected.append(by_id[call_id])
    return selected


def _arm_path(call_id: str, backend_name: str) -> Path:
    safe_key = hashlib.sha256(f"{call_id}::{backend_name}".encode()).hexdigest()
    return ARM_DIR / f"{safe_key}.json"


def _run_arm_worker(call_path: Path, backend_name: str, out_path: Path) -> None:
    cr_executor.load_env()
    backend = LOCAL_27B if backend_name == LOCAL_27B.name else CLOUD_DEEPSEEK
    call = AgentCallRecord(**json.loads(call_path.read_text()))
    outcome, transform = cr_executor.replay_call(replay._teacher_call_for(call), backend)
    _atomic_json(out_path, {"call_id": call.call_id, "backend": backend.name,
                            "outcome": asdict(replay._to_agent_outcome(outcome)),
                            "transform": transform})


def _run_judge_worker(call_id: str) -> None:
    examples = [row for row in _load_pilot_pairs()
                if row["call"]["call_id"] == call_id]
    if len(examples) != 1:
        raise RuntimeError(f"Expected one paired example for {call_id}, found {len(examples)}")
    if (examples[0]["local_outcome"]["status"] != "OK"
            or examples[0]["cloud_outcome"]["status"] != "OK"):
        raise RuntimeError("Judge requires an OK/OK paired example")
    judge.judge_examples_with_consistency(examples, JUDGE_PATH)


def _run_bounded_judge_worker(call_id: str) -> bool:
    """Fail closed on a worker failure without persisting prompts or credentials.

    The judge's JSONL is per-pass durable. A failed worker may have written
    one pass, so the operator can inspect and resume that exact call later.
    """
    timeout = min(ARM_TIMEOUT_S, max(1, (WORK_STOP - datetime.now(SGT)).total_seconds()))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "experiments.agentic_router.pilot_campaign",
             "--judge-worker", call_id],
            cwd=Path(__file__).resolve().parent.parent.parent,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=timeout, check=False,
        )
        if result.returncode == 0:
            return True
        # Only an exception *type* crosses the diagnostic boundary. Never
        # persist its detail, HTTP body, URL, prompt, headers, or full argv.
        tail = (result.stderr or "").splitlines()[-1:] or [""]
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Timeout))(?::|$)", tail[0])
        failure = {"kind": "nonzero_exit", "returncode": result.returncode,
                   "exception_type": match.group(1) if match else "unknown"}
    except subprocess.TimeoutExpired:
        failure = {"kind": "timeout", "timeout_s": timeout,
                   "exception_type": "TimeoutExpired"}
    _atomic_json(JUDGE_FAILURE_PATH, {"call_id": call_id, **failure,
                                      "at_utc": datetime.now(timezone.utc).isoformat()})
    _atomic_json(PAUSE_PATH, {"reason": "judge_worker_failed", "call_id": call_id,
                              "at_utc": datetime.now(timezone.utc).isoformat()})
    return False


def _flight_worker(backend_name: str) -> None:
    cr_executor.load_env()
    backend = LOCAL_27B if backend_name == LOCAL_27B.name else CLOUD_DEEPSEEK
    evidence = cr_executor.preflight(backend)
    if evidence.get("status") != "OK" or evidence.get("response_model") != backend.expected_model_exact:
        raise RuntimeError("Backend identity preflight did not produce the expected model")


def _bounded_flight(backend_name: str) -> None:
    remaining = (WORK_STOP - datetime.now(SGT)).total_seconds()
    if remaining <= 0:
        raise RuntimeError("Drain deadline reached before backend preflight")
    subprocess.run([sys.executable, "-m", "experiments.agentic_router.pilot_campaign",
                    "--flight-worker", backend_name],
                   cwd=Path(__file__).resolve().parent.parent.parent,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=min(90, remaining), check=True)


def _run_arm(call: AgentCallRecord, backend_name: str) -> dict[str, Any] | None:
    out_path = _arm_path(call.call_id, backend_name)
    retry_archive = out_path.with_name(out_path.stem + ".failed_attempt1.json")
    if out_path.exists():
        row = json.loads(out_path.read_text())
        if row.get("call_id") != call.call_id or row.get("backend") != backend_name:
            raise RuntimeError(f"Arm checkpoint identity mismatch: {out_path}")
        if (row.get("outcome", {}).get("status") in ("TRANSPORT_ERROR", "HTTP_ERROR")
                and not POSTPILOT_V9 and not retry_archive.exists() and _before_drain()):
            # Keep the failed first response as a recoverable audit artifact.
            # One bounded retry is allowed; a second failure is terminal
            # infrastructure evidence, never a quality label.
            out_path.replace(retry_archive)
        else:
            return row
    if not _before_drain():
        return None
    input_path = ARM_DIR / (out_path.stem + ".input.json")
    _atomic_json(input_path, asdict(call))
    try:
        subprocess.run([sys.executable, "-m", "experiments.agentic_router.pilot_campaign",
                        "--arm-worker", str(input_path), backend_name, str(out_path)],
                       cwd=Path(__file__).resolve().parent.parent.parent,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=min(ARM_TIMEOUT_S, max(1, (WORK_STOP - datetime.now(SGT)).total_seconds())),
                       check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        _atomic_json(STATUS_PATH, {"state": "arm_failed", "call_id": call.call_id,
                                   "backend": backend_name, "error_type": type(exc).__name__,
                                   "at_sgt": datetime.now(SGT).isoformat()})
        return None
    finally:
        # The input is reproduced from a collected trace on retry; it need
        # not be retained as an extra copy of private request content.
        input_path.unlink(missing_ok=True)
    return json.loads(out_path.read_text()) if out_path.exists() else None


def _paired_ids() -> set[str]:
    rows = run_judge.load_paired_examples(RESULTS) + _load_pilot_pairs()
    ids = [row["call"]["call_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate paired call IDs; refusing further spend")
    return set(ids)


def _load_pilot_pairs() -> list[dict[str, Any]]:
    if not PAIRED_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with PAIRED_PATH.open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Corrupted pilot pair at line {lineno}; refusing duplicate spend") from exc
    return rows


def _selection_digest() -> str | None:
    if not SELECTION_PATH.exists():
        return None
    return hashlib.sha256(SELECTION_PATH.read_bytes()).hexdigest()


def _terminal_partial(selected: list[AgentCallRecord]) -> dict[str, Any] | None:
    """Validate the frozen, explicitly partial v8 replay before any resume.

    The manifest is deliberately not inferred from missing pair rows: a
    missing row is otherwise retried by the normal campaign controller.
    """
    if not TERMINAL_PARTIAL_PATH.exists():
        return None
    data = json.loads(TERMINAL_PARTIAL_PATH.read_text())
    if not POSTPILOT_V8 or data.get("version") != 1 or data.get("batch") != "v8":
        raise RuntimeError("Unexpected terminal partial replay manifest")
    frozen = _load_selection()["trajectories"].get(data.get("trajectory_id"))
    if frozen is None or frozen.get("task_group") != data.get("task_group"):
        raise RuntimeError("Terminal partial task/trajectory mismatch")
    for key in ("indices", "call_ids", "source_trace_sha256", "selection_protocol",
                "selection_seed"):
        if data.get(key) != frozen.get(key):
            raise RuntimeError(f"Terminal partial {key} mismatch")
    expected_probabilities = [frozen["inclusion_probability_by_call_id"][cid]
                              for cid in frozen["call_ids"]]
    if data.get("inclusion_probabilities") != expected_probabilities:
        raise RuntimeError("Terminal partial inclusion probabilities mismatch")
    if data.get("selection_manifest_sha256") != _selection_digest():
        raise RuntimeError("Terminal partial selection manifest hash mismatch")
    records = data.get("outcomes")
    if not isinstance(records, list) or len(records) != len(frozen["call_ids"]):
        raise RuntimeError("Terminal partial outcome count mismatch")
    expected = ["PAIRED_OK_OK"] * 3 + ["LOCAL_TIME_BUDGET_EXHAUSTED_CLOUD_OK"] + [
        "NOT_ATTEMPTED_RESOURCE_CAP"] * 4
    if ([row.get("call_id") for row in records] != frozen["call_ids"]
            or [row.get("status") for row in records] != expected):
        raise RuntimeError("Terminal partial call order/status mismatch")
    selected_ids = {call.call_id for call in selected}
    if not set(frozen["call_ids"]) <= selected_ids:
        raise RuntimeError("Terminal partial calls missing from frozen selection")
    pairs = {row["call"]["call_id"]: row for row in _load_pilot_pairs()}
    for row in records:
        cid = row["call_id"]
        pair = pairs.get(cid)
        if row["status"] == "PAIRED_OK_OK":
            if (pair is None or pair["local_outcome"]["status"] != "OK"
                    or pair["cloud_outcome"]["status"] != "OK"):
                raise RuntimeError("Terminal partial completed pair is missing/non-OK")
        elif pair is not None:
            raise RuntimeError("Terminal partial censored call unexpectedly paired")
        if row["status"] == "LOCAL_TIME_BUDGET_EXHAUSTED_CLOUD_OK":
            if _arm_path(cid, LOCAL_27B.name).exists():
                raise RuntimeError("Terminal partial timed-out local arm has a checkpoint")
            cloud_path = _arm_path(cid, CLOUD_DEEPSEEK.name)
            if not cloud_path.exists():
                raise RuntimeError("Terminal partial cloud checkpoint missing")
            cloud = json.loads(cloud_path.read_text())
            if (cloud.get("call_id") != cid or cloud.get("backend") != CLOUD_DEEPSEEK.name
                    or cloud.get("outcome", {}).get("status") != "OK"):
                raise RuntimeError("Terminal partial cloud checkpoint is not OK")
        elif row["status"] == "NOT_ATTEMPTED_RESOURCE_CAP":
            if (_arm_path(cid, LOCAL_27B.name).exists()
                    or _arm_path(cid, CLOUD_DEEPSEEK.name).exists()):
                raise RuntimeError("Terminal partial unattempted call has an arm checkpoint")
    return data


def _judged_ids() -> set[str]:
    complete: set[str] = set()
    seen_any: set[str] = set()
    for path in (LEGACY_JUDGE_PATH, JUDGE_PATH):
        if not path.exists():
            continue
        pairs: dict[str, set[str]] = {}
        with path.open() as fh:
            for line in fh:
                row = json.loads(line)
                call_id, pass_label = row["call_id"], row["pass_label"]
                if pass_label in pairs.setdefault(call_id, set()):
                    raise RuntimeError(f"Duplicate judge pass for {call_id} in {path.name}")
                pairs[call_id].add(pass_label)
        completed_here = {call_id for call_id, passes in pairs.items()
                          if passes == {"primary", "reversed"}}
        if path == LEGACY_JUDGE_PATH and set(pairs) != completed_here:
            raise RuntimeError("Partial legacy judge record requires manual protocol resolution")
        if seen_any & set(pairs):
            raise RuntimeError("Call judged under both legacy and v5 order protocols")
        seen_any |= set(pairs)
        complete |= completed_here
    return complete


def run_once(include_holdout: bool = False) -> dict[str, Any]:
    if POSTPILOT_V9 and os.environ.get("FLOWMESH_V9_PAIRED_STAGE_APPROVED") != "1":
        raise RuntimeError("V9 paired replay/judge stage is not separately approved")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another pilot_campaign instance owns the lock") from exc
        if PAUSE_PATH.exists():
            status = {"state": "paused", "pause_marker": str(PAUSE_PATH),
                      "at_utc": datetime.now(timezone.utc).isoformat(),
                      "no_work_after_utc": WORK_STOP.astimezone(timezone.utc).isoformat(),
                      "requests_drained_by_utc": DRAIN.astimezone(timezone.utc).isoformat()}
            _atomic_json(STATUS_PATH, status)
            return status
        selected = _selected_calls(include_holdout=include_holdout)
        terminal = _terminal_partial(selected)
        terminal_ids = ({row["call_id"] for row in terminal["outcomes"]}
                        if terminal else set())
        selected_ids = {call.call_id for call in selected}
        done = _paired_ids()
        pending = any(call.call_id not in done and call.call_id not in terminal_ids
                      for call in selected)
        if pending and _before_drain():
            _bounded_flight(LOCAL_27B.name)
            _bounded_flight(CLOUD_DEEPSEEK.name)
        new = 0
        for call in selected:
            if (not _before_drain() or PAUSE_PATH.exists() or call.call_id in done
                    or call.call_id in terminal_ids):
                continue
            local = _run_arm(call, LOCAL_27B.name)
            if POSTPILOT_V9 and local is None:
                _atomic_json(PAUSE_PATH, {"reason": "local_arm_incomplete_no_retry",
                                          "call_id": call.call_id,
                                          "at_utc": datetime.now(timezone.utc).isoformat()})
                break
            cloud = _run_arm(call, CLOUD_DEEPSEEK.name)
            if local is None or cloud is None:
                if POSTPILOT_V9:
                    _atomic_json(PAUSE_PATH, {"reason": "arm_incomplete_no_retry",
                                              "call_id": call.call_id,
                                              "at_utc": datetime.now(timezone.utc).isoformat()})
                    break
                continue
            if local["outcome"]["status"] not in ("OK", "INVALID_CAPACITY") or cloud["outcome"]["status"] != "OK":
                # A backend transport/HTTP/identity failure is infrastructure
                # evidence, never a scored answer. Arm checkpoints retain it
                # for audit; a human must decide if a clean retry is warranted.
                if POSTPILOT_V9:
                    _atomic_json(PAUSE_PATH, {"reason": "arm_status_non_ok_no_retry",
                                              "call_id": call.call_id,
                                              "local_status": local["outcome"]["status"],
                                              "cloud_status": cloud["outcome"]["status"],
                                              "at_utc": datetime.now(timezone.utc).isoformat()})
                    break
                continue
            lo = AgentReplayOutcome(**local["outcome"])
            co = AgentReplayOutcome(**cloud["outcome"])
            example = PairedCallExample(
                call=call, original_backend=call.placement,
                local_outcome=lo, cloud_outcome=co,
                local_label=replay._label(call.placement, LOCAL_27B.name, lo),
                cloud_label=replay._label(call.placement, CLOUD_DEEPSEEK.name, co),
            )
            _append_jsonl(PAIRED_PATH, asdict(example))
            done.add(call.call_id)
            new += 1
            _atomic_json(STATUS_PATH, {"state": "replaying", "selected": len(selected),
                                       "n_pilot_pairs": len(done & selected_ids),
                                       "n_existing_pairs_total": len(done),
                                       "new_this_pass": new,
                                       "last_successful_call_id": call.call_id,
                                       "selection_sha256": _selection_digest(),
                                       "at_utc": datetime.now(timezone.utc).isoformat()})
        judged = _judged_ids()
        for call in selected:
            if not _before_drain() or PAUSE_PATH.exists() or call.call_id not in done or call.call_id in judged:
                continue
            if call.call_id in done:
                pair = next((row for row in _load_pilot_pairs()
                             if row["call"]["call_id"] == call.call_id), None)
                if pair is None or pair["local_outcome"]["status"] != "OK":
                    continue
            if not _run_bounded_judge_worker(call.call_id):
                break
            judged.add(call.call_id)
            _atomic_json(STATUS_PATH, {"state": "judging", "selected": len(selected),
                                       "n_pilot_pairs": len(done & selected_ids),
                                       "n_existing_pairs_total": len(done),
                                       "n_pilot_judged": len(judged & selected_ids),
                                       "last_successful_call_id": call.call_id,
                                       "selection_sha256": _selection_digest(),
                                       "at_utc": datetime.now(timezone.utc).isoformat(),
                                       "no_work_after_utc": WORK_STOP.astimezone(timezone.utc).isoformat(),
                                       "requests_drained_by_utc": DRAIN.astimezone(timezone.utc).isoformat()})
        status = {"state": "paused" if PAUSE_PATH.exists() else "drained" if not _before_drain() else "idle",
                  "selected": len(selected),
                  "n_pilot_pairs": len(done & selected_ids),
                  "n_existing_pairs_total": len(done),
                  "n_pilot_judged": len(judged & selected_ids),
                  "new_this_pass": new,
                  "selection_sha256": _selection_digest(),
                  "at_utc": datetime.now(timezone.utc).isoformat(),
                  "no_work_after_utc": WORK_STOP.astimezone(timezone.utc).isoformat(),
                  "requests_drained_by_utc": DRAIN.astimezone(timezone.utc).isoformat(),
                  "gpu_expiry_utc": "2026-09-17T09:00:00+00:00"}
        _atomic_json(STATUS_PATH, status)
        if new and _before_drain():
            _bounded_flight(LOCAL_27B.name)
            _bounded_flight(CLOUD_DEEPSEEK.name)
        return status


def run_judge_only_terminal_partial() -> dict[str, Any]:
    """Judge only completed OK/OK pairs in a frozen partial replay.

    This path intentionally performs no backend preflight and calls no arm
    worker. The pause marker stays in place to block the normal follower.
    """
    if not PAUSE_PATH.exists():
        raise RuntimeError("Judge-only partial replay requires the pause marker")
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another pilot_campaign instance owns the lock") from exc
        selected = _selected_calls()
        terminal = _terminal_partial(selected)
        if terminal is None:
            raise RuntimeError("No validated terminal partial replay manifest")
        permitted = [row["call_id"] for row in terminal["outcomes"]
                     if row["status"] == "PAIRED_OK_OK"]
        judged = _judged_ids()
        for call_id in permitted:
            if call_id in judged:
                continue
            if not _before_drain():
                raise RuntimeError("Drain deadline reached during partial judge")
            if not _run_bounded_judge_worker(call_id):
                raise RuntimeError("Partial judge worker failed; checkpoint preserved")
            judged = _judged_ids()
            if call_id not in judged:
                raise RuntimeError("Judge worker returned without both passes")
        result = {"state": "terminal_partial_judged", "batch": "v8",
                  "task_group": terminal["task_group"],
                  "n_selected": len(terminal["outcomes"]),
                  "n_paired_ok_ok": len(permitted),
                  "n_judged_ok_ok": len(set(permitted) & judged),
                  "n_local_time_budget_exhausted": 1,
                  "n_not_attempted_resource_cap": 4,
                  "terminal_partial_manifest_sha256": hashlib.sha256(
                      TERMINAL_PARTIAL_PATH.read_bytes()).hexdigest(),
                  "at_utc": datetime.now(timezone.utc).isoformat()}
        _atomic_json(STATUS_PATH, result)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--include-holdout", action="store_true",
                        help="Only after the development model and threshold are frozen")
    parser.add_argument("--arm-worker", nargs=3, metavar=("CALL_JSON", "BACKEND", "OUT_JSON"))
    parser.add_argument("--judge-worker", metavar="CALL_ID")
    parser.add_argument("--judge-only-terminal-partial", action="store_true")
    parser.add_argument("--flight-worker", metavar="BACKEND")
    parser.add_argument("--pause", action="store_true",
                        help="quiesce new replay/judge calls for live matrix")
    parser.add_argument("--resume", action="store_true",
                        help="remove pause marker after live matrix")
    args = parser.parse_args()
    if (POSTPILOT_V9 and os.environ.get("FLOWMESH_V9_PAIRED_STAGE_APPROVED") != "1"
            and (args.arm_worker or args.judge_worker or args.once or args.follow)):
        parser.error("V9 paired replay/judge stage is not separately approved")
    if args.arm_worker:
        call_path, backend_name, out_path = args.arm_worker
        if backend_name not in (LOCAL_27B.name, CLOUD_DEEPSEEK.name):
            parser.error("unsupported backend")
        _run_arm_worker(Path(call_path), backend_name, Path(out_path))
        return 0
    if args.judge_worker:
        _run_judge_worker(args.judge_worker)
        return 0
    if args.judge_only_terminal_partial:
        print(json.dumps(run_judge_only_terminal_partial()), flush=True)
        return 0
    if args.flight_worker:
        if args.flight_worker not in (LOCAL_27B.name, CLOUD_DEEPSEEK.name):
            parser.error("unsupported backend")
        _flight_worker(args.flight_worker)
        return 0
    if args.pause:
        _atomic_json(PAUSE_PATH, {"reason": "live_matrix", "at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"state": "pause_requested", "marker": str(PAUSE_PATH)}))
        return 0
    if args.resume:
        PAUSE_PATH.unlink(missing_ok=True)
        print(json.dumps({"state": "resumed"}))
        return 0
    if args.once == args.follow:
        parser.error("choose exactly one of --once or --follow")
    while True:
        try:
            print(json.dumps(run_once(include_holdout=args.include_holdout)), flush=True)
        except Exception as exc:
            error = {"state": "error", "error_type": type(exc).__name__,
                     "error_message": str(exc)[:300],
                     "at_utc": datetime.now(timezone.utc).isoformat(),
                     "no_work_after_utc": WORK_STOP.astimezone(timezone.utc).isoformat(),
                     "requests_drained_by_utc": DRAIN.astimezone(timezone.utc).isoformat()}
            _atomic_json(STATUS_PATH, error)
            print(json.dumps(error), flush=True)
            if args.once:
                return 1
        if args.once or not _before_drain():
            return 0
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
