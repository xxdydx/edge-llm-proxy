"""Typed records for the Phase 1 pipeline: dataset -> paired generation ->
grading -> features -> policy -> analysis. Every stage reads/writes these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Problem:
    """One MBPP+ task, self-contained (no agent context).

    Inputs and expected outputs are precomputed on the host from the trusted
    canonical solution (EvalPlus ``trusted_exec``) and stored tuple/set-encoded
    via ``serde`` so the Docker grader reconstructs exact Python objects.
    ``base_input``/``plus_input`` here are the EvalPlus-*deserialised*,
    contract-filtered inputs, encoded.
    """

    task_id: str  # e.g. "Mbpp/2"
    entry_point: str  # function name the solution must define
    prompt: str  # the docstring-style problem statement (with 1 example assert)
    canonical_solution: str  # reference implementation (trusted repo data)
    base_input: list[Any]  # serde-encoded, deserialised, contract-passing base inputs
    plus_input: list[Any]  # serde-encoded, deserialised, contract-passing plus inputs
    expected: list[Any]  # serde-encoded expected outputs from canonical, aligned base+plus
    ref_time: list[float]  # per-input canonical wall time (for EvalPlus time limits)
    atol: float  # float comparison tolerance (0 for exact)
    contract: str  # precondition asserts (kept for provenance / gold-vs-gold test)
    n_base: int  # count of contract-passing base inputs
    n_plus: int  # count of contract-passing plus inputs
    task_group: int  # hash-bucket group id for split isolation
    window: int  # chronological window id (selection order) for the online sim
    order: int  # position in the seeded selection order


@dataclass(frozen=True)
class GenerationOutcome:
    """Result of sending one Problem's prompt to one backend. Mirrors
    capability_router.ReplayOutcome plus the bounded-fail-fast accounting."""

    backend: str
    status: Literal[
        "OK", "IDENTITY_MISMATCH", "TRANSPORT_ERROR", "HTTP_ERROR", "INVALID_CAPACITY"
    ]
    completion: str | None  # extracted code (None on non-OK)
    raw_text: str | None  # full model text (None on non-OK)
    usage: dict[str, Any] | None
    detail: str | None = None
    latency_s: float | None = None
    retry_attempts: int | None = None
    end_to_end_wall_s: float | None = None
    final_attempt_latency_s: float | None = None
    attempts_meta: list[dict[str, Any]] | None = None
    stop_reason: str | None = None
    model_identity: str | None = None


@dataclass(frozen=True)
class GradeResult:
    """Objective isolated-unit-test verdict for one completion. All test
    execution happens inside the Docker grader; model code never runs on the
    host."""

    status: Literal["PASS", "FAIL", "ERROR", "TIMEOUT", "NO_CODE", "SKIPPED"]
    base_pass: bool  # passed the original MBPP inputs
    plus_pass: bool  # passed base + EvalPlus expanded inputs (the strict label)
    n_base: int
    n_base_ok: int
    n_plus: int
    n_plus_ok: int
    detail: str = ""
    grader_wall_s: float | None = None


@dataclass(frozen=True)
class PairedExample:
    """One fully-processed problem: both generations, both grades, and the
    request-only features known before dispatch."""

    problem: Problem
    features: dict[str, Any]
    local_gen: GenerationOutcome
    cloud_gen: GenerationOutcome
    local_grade: GradeResult
    cloud_grade: GradeResult
    # convenience labels (plus_pass is the strict correctness label)
    local_ok: bool
    cloud_ok: bool


@dataclass
class DatasetManifest:
    version: str
    seed: int
    selected: int
    available: int
    task_group_counts: dict[str, int] = field(default_factory=dict)
    window_counts: dict[str, int] = field(default_factory=dict)
