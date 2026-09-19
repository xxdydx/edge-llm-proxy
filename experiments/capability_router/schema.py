"""Typed records shared across the dataset -> executor -> label -> model
pipeline. Every stage writes/reads these instead of raw dicts so a mistaken
field name fails at construction time, not three stages later in a plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Label = Literal[
    "EQUIVALENT", "NOT_EQUIVALENT", "UNKNOWN", "EXECUTION_ERROR", "INVALID_CAPACITY"
]


@dataclass(frozen=True)
class TeacherCall:
    """One selected historical call: the exact request that was sent, and
    what the recorded cloud trajectory actually did in response (the
    teacher action). Provenance fields exist so every downstream number can
    be traced back to a real trace file and campaign."""

    call_id: str  # stable dedup key, see dataset.request_fingerprint
    task_group: str  # e.g. "swebench-openlibrary-c05ccf2c"
    source_campaign: str
    source_trace_path: str
    record_index: int  # position of this call within its trace file
    request: dict[str, Any]  # the exact Anthropic-format request, as sent
    teacher_response: dict[str, Any]  # the recorded cloud response
    teacher_placement: str  # should always be "cloud" by selection


@dataclass(frozen=True)
class ReplayOutcome:
    """The result of sending one TeacherCall's request to one backend."""

    backend: str
    status: Literal[
        "OK", "IDENTITY_MISMATCH", "TRANSPORT_ERROR", "HTTP_ERROR", "INVALID_CAPACITY"
    ]
    response: dict[str, Any] | None
    latency_s: float | None
    detail: str | None = None  # populated for any non-OK status
    # Bounded fail-fast transport accounting (added 2026-09-10). All optional so
    # older checkpoints and canned test fixtures without them still load.
    #   retry_attempts: same-backend attempts actually made (1 = first try only,
    #     2 = one retry). Capped at executor._TRANSPORT_MAX_ATTEMPTS.
    #   end_to_end_wall_s: wall time across every attempt PLUS inter-attempt
    #     backoff sleeps -- the true cost this (call, backend) pair charged.
    #   final_attempt_latency_s: just the last attempt's own duration (the
    #     attempt whose response/timing is reported in `response._timing`).
    #   attempts_meta: per-attempt [{attempt, transport_err, status_code,
    #     attempt_wall_s}], oldest first.
    retry_attempts: int | None = None
    end_to_end_wall_s: float | None = None
    final_attempt_latency_s: float | None = None
    attempts_meta: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class LabeledExample:
    """One fully-scored dataset row: features, both replay outcomes, and the
    conservative label each earned against the teacher action."""

    call: TeacherCall
    features: dict[str, Any]  # asdict(router.CallFeatures)
    local_outcome: ReplayOutcome
    cloud_outcome: ReplayOutcome
    local_label: Label
    cloud_label: Label
    local_schema_valid: bool | None
    cloud_schema_valid: bool | None
    local_label_detail: str
    cloud_label_detail: str


@dataclass
class DatasetManifest:
    """Written alongside the dataset so a rerun can prove it selected the
    same calls, and so `data/` review doesn't require re-scanning traces."""

    seed: int
    target_examples: int
    selected_count: int
    task_group_counts: dict[str, int] = field(default_factory=dict)
    excluded_dirs: list[str] = field(default_factory=list)
    dedup_dropped: int = 0
