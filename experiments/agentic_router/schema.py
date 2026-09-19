"""Typed records for the agentic-trace pipeline: collect -> features ->
replay. Mirrors the ``phase1_router``/``capability_router`` convention of
typed dataclasses over raw dicts so a stage-boundary field mismatch fails at
construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class AgentCallRecord:
    """One real call inside one SWE-bench Pro trajectory: everything
    ``edgeproxy`` already recorded for it, plus provenance so it can be
    traced back to the exact trace file and campaign it came from.

    Fields here are read straight off the raw trace record and the
    ``edgeproxy.call.v1`` structured view (``edgeproxy/trace/record.py``'s
    ``build_structured_call``) -- nothing here is invented. Fields that
    genuinely don't exist yet (e.g. "backend used on the previous call in
    this trajectory") are added separately by ``features.derive`` as
    ``AgentCallRecord.derived``, kept apart from what was actually recorded
    live so the two are never confused.
    """

    call_id: str
    trajectory_id: str  # stable id for the owning AgentTrajectory
    task_group: str
    source_campaign: str
    source_trace_path: str
    record_index: int  # position within the raw trace file
    turn_index: int  # position within THIS trajectory, 0-based, ts-sorted
    timestamp_unix_s: float | None
    placement: str | None  # "local" | "cloud" | None if never decided
    policy: str | None
    reason: str | None
    request: dict[str, Any]
    response: dict[str, Any] | None
    stop_reason: str | None
    tool_names: list[str] = field(default_factory=list)
    # request-time features edgeproxy already computed (asdict(CallFeatures)
    # as stored in the raw record) -- includes branch_turn_ordinal,
    # errored_tool_result_density, local_prompt_tokens, local_token_budget,
    # has_agent_tool, is_security_monitor, etc.
    router_features: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, Any] = field(default_factory=dict)  # v1 view "tokens"
    timing: dict[str, Any] = field(default_factory=dict)  # v1 view "timing"
    cache_probe: dict[str, Any] | None = None
    causality: dict[str, Any] = field(default_factory=dict)
    tool_use_blocks: list[dict[str, Any]] = field(default_factory=list)
    # Populated by features.derive(); empty dict until then.
    derived: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrajectory:
    """One completed SWE-bench Pro job: its calls in order, plus the
    Docker-graded final task outcome joined on from the verdict file."""

    trajectory_id: str
    task_group: str
    campaign: str
    condition: str
    seed: int
    verdict_path: str
    task_passed: bool | None  # None if verdict missing/unreadable
    verdict_detail: str | None
    calls: list[AgentCallRecord] = field(default_factory=list)


@dataclass
class TrajectoryManifest:
    """Written alongside a collected dataset so a rerun can prove it found
    the same trajectories, and downstream stages don't need to re-scan
    ``traces/``."""

    n_trajectories: int
    n_calls: int
    task_group_counts: dict[str, int] = field(default_factory=dict)
    condition_counts: dict[str, int] = field(default_factory=dict)
    missing_trace_files: list[str] = field(default_factory=list)
    unreadable_verdicts: list[str] = field(default_factory=list)


# --- Stage 1c: paired call-level replay ------------------------------------

Label = Literal["EQUIVALENT", "NOT_EQUIVALENT", "UNKNOWN", "EXECUTION_ERROR", "INVALID_CAPACITY"]


@dataclass(frozen=True)
class AgentReplayOutcome:
    """Result of replaying one AgentCallRecord's request against one backend.

    Mirrors ``capability_router.schema.ReplayOutcome`` field-for-field (this
    experiment reuses ``capability_router.executor.replay_call`` directly
    rather than re-implementing transport) -- kept as a separate dataclass
    only so this package doesn't import a sibling experiment's schema as a
    load-bearing dependency.
    """

    backend: str
    status: Literal["OK", "IDENTITY_MISMATCH", "TRANSPORT_ERROR", "HTTP_ERROR", "INVALID_CAPACITY"]
    response: dict[str, Any] | None
    latency_s: float | None
    detail: str | None = None
    retry_attempts: int | None = None
    end_to_end_wall_s: float | None = None


@dataclass(frozen=True)
class PairedCallExample:
    """One replayed call: the original trajectory context, both replay
    outcomes, and the label each earned against the ORIGINAL backend's own
    recorded response (the "teacher" for that specific call, whichever
    backend actually served it live)."""

    call: AgentCallRecord
    original_backend: str | None
    local_outcome: AgentReplayOutcome
    cloud_outcome: AgentReplayOutcome
    local_label: Label
    cloud_label: Label
