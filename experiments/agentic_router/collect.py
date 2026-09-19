"""Assemble AgentTrajectory records from completed SWE-bench Pro campaigns.

Read-only over already-recorded data: raw trace JSONL under ``traces/`` plus
Docker-graded verdict JSON under ``eval-suite/swebench/results/*/verdicts/``.
Does not touch ``edgeproxy/server.py``'s live tracing code, and does not
require a live GPU/cloud connection -- this stage only replays what already
happened, joining the final task outcome onto every call in its trajectory
(edgeproxy records verdicts nowhere per-call today; this is the join that
fixes that, in a batch/offline pass rather than a change to production
tracing).

Generalizes ``capability_router.dataset``'s pattern (same repo, same
verdict/trace-directory conventions) from "cloud-only jobs, cloud is always
the teacher" to "every condition, every call kept regardless of which
backend served it" -- this package needs both local and cloud calls from
mixed-condition trajectories, not only cloud-only ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from edgeproxy.trace import record as trace_record
from edgeproxy.trace.replay import calls as messages_calls

from .schema import AgentCallRecord, AgentTrajectory, TrajectoryManifest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "eval-suite" / "swebench" / "results"
TRACES_ROOT = REPO_ROOT / "traces"


def discover_verdicts(results_root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    """One entry per graded job, any condition, any seed."""
    out: list[dict[str, Any]] = []
    if not results_root.is_dir():
        return out
    for campaign_dir in sorted(results_root.iterdir()):
        verdicts_dir = campaign_dir / "verdicts"
        if not campaign_dir.is_dir() or not verdicts_dir.is_dir():
            continue
        for verdict_path in sorted(verdicts_dir.glob("*.json")):
            base = verdict_path.stem  # "<task_group>__<condition>__seedN"
            parts = base.split("__")
            if len(parts) != 3 or not parts[2].startswith("seed"):
                continue
            task_group, condition, seed_str = parts
            try:
                seed = int(seed_str.removeprefix("seed"))
            except ValueError:
                continue
            out.append(
                {
                    "campaign": campaign_dir.name,
                    "task_group": task_group,
                    "condition": condition,
                    "seed": seed,
                    "verdict_path": verdict_path,
                }
            )
    return out


# Substrings seen in real grading-infra failures (wrong test runner, docker
# pull failure, missing dependency, job crashed before grading ever ran)
# rather than genuine task pass/fail -- confirmed on 2026-09-16 (NodeBB
# instances graded with pytest despite being a JS repo; disk-full docker
# pulls; a container-setup crash leaving the runner's own placeholder
# verdict; pytest exit 4 == pytest's own documented "no tests were
# collected", a harness/environment problem, not a task outcome -- seen on
# a real Python instance where the trajectory has real Claude Code usage
# but zero of 55 known-good pass_to_pass tests were even collected). A
# verdict matching one of these is not real ground truth and must not be
# silently treated as a task outcome.
_GRADER_ERROR_MARKERS = (
    "no junit xml produced",
    "no module named pytest",
    "docker run failed",
    "unable to find image",
    "no space left on device",
    "job did not complete",
    "pytest exit 4",
)


def verdict_is_grading_error(detail: str | None) -> bool:
    if not detail:
        return False
    low = detail.lower()
    return any(marker in low for marker in _GRADER_ERROR_MARKERS)


def _official_eval_unscorable(verdict_path: Path, verdict: dict) -> str | None:
    """Quarantine official test sessions that produced no comparable test result.

    An agent patch can itself crash collection. That is useful failure evidence,
    but SWE-bench's 0/N labels then measure the parser's empty result set, not
    the ordinary task tests. Keep the trajectory and log; do not automatically
    select, replay, or judge its calls as a clean graded task.
    """
    if verdict.get("passed") is not False:
        return None
    log_path = verdict_path.with_suffix(".eval.log")
    try:
        log = log_path.read_text(errors="replace")
    except OSError:
        return None
    if "INTERNALERROR>" in log:
        return "official test session internal error; task counts unscorable"
    if (verdict.get("n_fail_to_pass_passed") == 0
            and verdict.get("n_pass_to_pass_passed") == 0
            and any(marker in log.lower() for marker in (
                "collected 0 items", "no tests ran", "no tests collected", "error collecting"
            ))):
        return "official test session produced no test results; task counts unscorable"
    return None


def load_verdict(verdict_path: Path) -> tuple[bool | None, str | None]:
    """Checks BOTH the `reason` field (the checker's own summary) and the
    `error` field (populated when the job crashed before the checker ever
    ran, e.g. a docker pull failure during container setup) -- a real bug
    (2026-09-16): the runner's placeholder reason on that path is the
    generic "job did not complete", with the actual cause (a docker error)
    only present in `error`; checking `reason` alone missed it."""
    try:
        verdict = json.loads(verdict_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    passed = verdict.get("passed")
    detail = verdict.get("reason")
    error_field = verdict.get("error")
    unscorable = _official_eval_unscorable(verdict_path, verdict)
    if unscorable:
        return None, unscorable
    if verdict_is_grading_error(detail) or verdict_is_grading_error(error_field):
        # Not a real ground-truth outcome -- treat exactly like an
        # unreadable verdict (None), never as a genuine pass/fail.
        combined = detail if not error_field else f"{detail} | error: {error_field}"
        return None, combined
    return (bool(passed) if passed is not None else None), detail


def locate_trace_file(
    campaign: str, task_group: str, condition: str, seed: int,
    traces_root: Path = TRACES_ROOT,
) -> Path | None:
    """Same disambiguation strategy as ``capability_router.dataset.locate_trace_file``
    (run-stamp isn't derivable from the campaign name, so candidates are
    matched by glob then confirmed by reading each candidate's own
    ``experiment_id``), generalized to any condition string and to
    picking the file for the requested seed specifically.

    When more than one candidate matches (a job was killed and relaunched
    under the same campaign/instance/condition/seed -- confirmed to happen
    in practice, e.g. the 2026-09-16 disk-full relaunch), picks the
    MOST RECENTLY MODIFIED one, not the first alphabetically. Directory
    names embed a UTC run-stamp (``...T095202Z...``) so alphabetical sort
    is chronological too, but ascending sort silently picked the OLDEST
    (first, often stale/aborted) attempt -- this produced a real false
    positive once already (a stale 1-record disk-full-attempt trace was
    picked over the real 55-record relaunch, briefly read as an auth
    failure). Mtime is used rather than re-deriving the timestamp from the
    directory name, since that parsing would itself be one more thing to
    get wrong."""
    suffix = task_group.removeprefix("swebench-")
    pattern = f"*-run-{condition}-*-swebench-{suffix}-seed{seed}"
    matches: list[Path] = []
    for candidate_dir in sorted(traces_root.glob(pattern)):
        for jsonl_path in sorted(candidate_dir.glob("*.jsonl")):
            try:
                with jsonl_path.open() as fh:
                    first_line = fh.readline()
            except OSError:
                continue
            if not first_line.strip():
                continue
            try:
                first_record = json.loads(first_line)
            except json.JSONDecodeError:
                continue
            if first_record.get("experiment_id") == campaign:
                matches.append(jsonl_path)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_trace_records(trace_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with trace_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _tool_use_success(tool_use_blocks: list[dict[str, Any]]) -> bool | None:
    """Downstream success for this call's own tool-use output, from the same
    schema-validity check the live ``edgeproxy.call.v1`` view already
    computes (``validate_tool_use_blocks``). None if the call made no tool
    calls at all (not applicable, not a failure)."""
    if not tool_use_blocks:
        return None
    return all(b.get("schema_valid") for b in tool_use_blocks)


def build_trajectory(
    records: list[dict[str, Any]],
    task_group: str,
    campaign: str,
    condition: str,
    seed: int,
    trace_path: Path,
    task_passed: bool | None,
    verdict_detail: str | None,
    verdict_path: Path,
) -> AgentTrajectory:
    trajectory_id = f"{campaign}:{task_group}:{condition}:seed{seed}"
    raw_calls = [r for r in messages_calls(records) if isinstance(r.get("request"), dict)]
    raw_calls.sort(key=lambda r: r.get("ts") or 0.0)

    call_records: list[AgentCallRecord] = []
    for turn_index, rec in enumerate(raw_calls):
        request = rec["request"]
        v1 = trace_record.build_structured_call(rec, request)
        features = rec.get("features") or {}
        response = rec.get("response") if isinstance(rec.get("response"), dict) else None
        call_records.append(
            AgentCallRecord(
                call_id=str(v1.get("call_id") or f"{trajectory_id}:{turn_index}"),
                trajectory_id=trajectory_id,
                task_group=task_group,
                source_campaign=campaign,
                source_trace_path=str(trace_path),
                record_index=turn_index,
                turn_index=turn_index,
                timestamp_unix_s=rec.get("ts"),
                placement=rec.get("placement"),
                policy=rec.get("policy"),
                reason=rec.get("reason"),
                request=request,
                response=response,
                stop_reason=v1.get("stop_reason"),
                tool_names=list(v1.get("tool_names") or []),
                router_features=dict(features),
                tokens=dict(v1.get("tokens") or {}),
                timing=dict(v1.get("timing") or {}),
                cache_probe=v1.get("cache_probe"),
                causality=dict(v1.get("causality") or {}),
                tool_use_blocks=list(v1.get("tool_use_blocks") or []),
            )
        )

    return AgentTrajectory(
        trajectory_id=trajectory_id,
        task_group=task_group,
        campaign=campaign,
        condition=condition,
        seed=seed,
        verdict_path=str(verdict_path),
        task_passed=task_passed,
        verdict_detail=verdict_detail,
        calls=call_records,
    )


def build_trajectories(
    results_root: Path = RESULTS_ROOT, traces_root: Path = TRACES_ROOT,
) -> tuple[list[AgentTrajectory], TrajectoryManifest]:
    trajectories: list[AgentTrajectory] = []
    missing_trace_files: list[str] = []
    unreadable_verdicts: list[str] = []

    for entry in discover_verdicts(results_root):
        task_passed, detail = load_verdict(entry["verdict_path"])
        if task_passed is None:
            unreadable_verdicts.append(str(entry["verdict_path"]))
        trace_path = locate_trace_file(
            entry["campaign"], entry["task_group"], entry["condition"], entry["seed"],
            traces_root=traces_root,
        )
        if trace_path is None:
            missing_trace_files.append(
                f"{entry['campaign']}/{entry['task_group']}/{entry['condition']}/seed{entry['seed']}"
            )
            continue
        records = load_trace_records(trace_path)
        trajectories.append(
            build_trajectory(
                records,
                entry["task_group"],
                entry["campaign"],
                entry["condition"],
                entry["seed"],
                trace_path,
                task_passed,
                detail,
                entry["verdict_path"],
            )
        )

    task_group_counts: dict[str, int] = {}
    condition_counts: dict[str, int] = {}
    n_calls = 0
    for traj in trajectories:
        task_group_counts[traj.task_group] = task_group_counts.get(traj.task_group, 0) + 1
        condition_counts[traj.condition] = condition_counts.get(traj.condition, 0) + 1
        n_calls += len(traj.calls)

    manifest = TrajectoryManifest(
        n_trajectories=len(trajectories),
        n_calls=n_calls,
        task_group_counts=task_group_counts,
        condition_counts=condition_counts,
        missing_trace_files=missing_trace_files,
        unreadable_verdicts=unreadable_verdicts,
    )
    return trajectories, manifest


def save_trajectories(trajectories: list[AgentTrajectory], manifest: TrajectoryManifest, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "trajectories.jsonl"
    with path.open("w") as fh:
        for traj in trajectories:
            fh.write(json.dumps(asdict(traj)) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n")


def load_trajectories(out_dir: Path) -> list[AgentTrajectory]:
    path = out_dir / "trajectories.jsonl"
    trajectories: list[AgentTrajectory] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw["calls"] = [AgentCallRecord(**c) for c in raw["calls"]]
            trajectories.append(AgentTrajectory(**raw))
    return trajectories
