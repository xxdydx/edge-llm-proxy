"""Replay every captured AgentCallRecord against BOTH backends, reusing
``capability_router.executor.replay_call`` directly rather than
reimplementing transport, identity verification, or capacity prechecking --
this experiment's only job is to feed it agentic calls instead of MBPP
prompts or capability_router's own cloud-only teacher-call dataset.

Labeling reuses the same conservative scheme capability_router already
implements (`capability_router.labels`, imported lazily below) is NOT
reused wholesale here because that module's equivalence check is tuned for
short capability-probe replies; agentic tool-loop turns need the SAME kind
of "does the response match well enough" judgment but applied per-call, not
per-final-answer. For V1, labeling is deliberately conservative and simple
(see `_label`) -- refine once real replay data is in hand and it's clear
what a useful call-level label actually looks like for this domain.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from experiments.capability_router import executor as cr_executor
from experiments.capability_router.config import CLOUD_DEEPSEEK, LOCAL_27B, BackendConfig
from experiments.capability_router.schema import ReplayOutcome as CRReplayOutcome
from experiments.capability_router.schema import TeacherCall

from .schema import AgentCallRecord, AgentReplayOutcome, AgentTrajectory, Label, PairedCallExample


def _to_agent_outcome(o: CRReplayOutcome) -> AgentReplayOutcome:
    return AgentReplayOutcome(
        backend=o.backend, status=o.status, response=o.response, latency_s=o.latency_s,
        detail=o.detail, retry_attempts=o.retry_attempts, end_to_end_wall_s=o.end_to_end_wall_s,
    )


def _teacher_call_for(call: AgentCallRecord) -> TeacherCall:
    """Adapts an AgentCallRecord into the TeacherCall shape replay_call
    actually reads from (only `.request` is used by replay_call itself;
    the rest is provenance, carried through unchanged from this call's own
    real fields so it's still traceable after replay)."""
    return TeacherCall(
        call_id=call.call_id,
        task_group=call.task_group,
        source_campaign=call.source_campaign,
        source_trace_path=call.source_trace_path,
        record_index=call.record_index,
        request=call.request,
        teacher_response=call.response or {},
        teacher_placement=call.placement or "unknown",
    )


def _label(original_backend: str | None, target_backend_name: str, outcome: CRReplayOutcome) -> Label:
    """Conservative V1 label: OK on the backend that actually served this
    call live is EQUIVALENT to itself by construction (nothing to compare
    against); OK on the OTHER backend is UNKNOWN until a real quality
    judgment is wired in (agentic correctness needs the downstream task
    verdict, not a single-call text diff -- deferred to the Stage 1
    checkpoint analysis, not decided here). Any non-OK status is
    EXECUTION_ERROR or INVALID_CAPACITY, passed through directly."""
    if outcome.status == "INVALID_CAPACITY":
        return "INVALID_CAPACITY"
    if outcome.status != "OK":
        return "EXECUTION_ERROR"
    if original_backend is not None and target_backend_name.startswith(original_backend):
        return "EQUIVALENT"
    return "UNKNOWN"


def replay_trajectory(
    trajectory: AgentTrajectory,
    local_backend: BackendConfig = LOCAL_27B,
    cloud_backend: BackendConfig = CLOUD_DEEPSEEK,
) -> list[PairedCallExample]:
    out: list[PairedCallExample] = []
    for call in trajectory.calls:
        teacher_call = _teacher_call_for(call)
        local_outcome, _ = cr_executor.replay_call(teacher_call, local_backend)
        cloud_outcome, _ = cr_executor.replay_call(teacher_call, cloud_backend)
        out.append(
            PairedCallExample(
                call=call,
                original_backend=call.placement,
                local_outcome=_to_agent_outcome(local_outcome),
                cloud_outcome=_to_agent_outcome(cloud_outcome),
                local_label=_label(call.placement, local_backend.name, local_outcome),
                cloud_label=_label(call.placement, cloud_backend.name, cloud_outcome),
            )
        )
    return out


def replay_all(
    trajectories: list[AgentTrajectory],
    local_backend: BackendConfig = LOCAL_27B,
    cloud_backend: BackendConfig = CLOUD_DEEPSEEK,
) -> list[PairedCallExample]:
    """Runs preflight/postflight identity checks once for the whole batch
    (same contamination gate capability_router's own batch replay uses),
    not per-trajectory -- an identity failure aborts the whole run rather
    than silently producing partial, unverifiable results."""
    cr_executor.load_env()  # populates ANTHROPIC_AUTH_TOKEN etc. from .env; required before any cloud call
    cr_executor.preflight(local_backend)
    cr_executor.preflight(cloud_backend)
    examples: list[PairedCallExample] = []
    for trajectory in trajectories:
        examples.extend(replay_trajectory(trajectory, local_backend, cloud_backend))
    cr_executor.postflight(local_backend)
    cr_executor.postflight(cloud_backend)
    return examples


def save_examples(examples: list[PairedCallExample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(asdict(ex)) + "\n")
