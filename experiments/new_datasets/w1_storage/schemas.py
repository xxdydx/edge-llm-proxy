"""Versioned, JSON-friendly campaign records and validation.

The records deliberately keep nullable values and their explanation in
``missing_reasons``.  This prevents an unknown measurement from becoming a
false zero or an accidentally omitted field.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
import json
import re
from typing import Any, Literal, Mapping

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,254}$")
DATASETS = {"A": "prefix_outcomes", "B": "branch_outcomes", "C": "serving_calls", "preference": "preference_annotations"}
MISSING_REASON = "missing_reasons"

class ValidationError(ValueError):
    pass

def validate_id(value: str, field_name: str = "id") -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ValidationError(f"{field_name} is not a valid campaign ID")
    return value

def _validate_common(obj: Any) -> None:
    for name in ("campaign_id", "task_id", "trajectory_id"):
        validate_id(getattr(obj, name), name)
    if obj.split not in {"train", "validation", "holdout", "synthetic"}:
        raise ValidationError("invalid split")
    if not isinstance(obj.provenance, dict) or not obj.provenance:
        raise ValidationError("provenance must be a non-empty mapping")
    if not isinstance(obj.missing_reasons, dict):
        raise ValidationError("missing_reasons must be a mapping")
    for f in fields(obj):
        if f.name == MISSING_REASON:
            continue
        if getattr(obj, f.name) is None and f.name not in obj.missing_reasons:
            raise ValidationError(f"null field {f.name!r} lacks a missing reason")
    for key in obj.missing_reasons:
        if key not in {f.name for f in fields(obj)}:
            raise ValidationError(f"missing reason refers to unknown field {key!r}")

def _duration_map(value: Mapping[str, Any] | None, name: str) -> None:
    if value is None: return
    for key, number in value.items():
        if key.endswith("_ms") or key.endswith("_seconds"):
            if number is not None and (not isinstance(number, (int, float)) or number < 0):
                raise ValidationError(f"negative/invalid duration {name}.{key}")

def _usage(value: Mapping[str, Any] | None, name: str) -> None:
    if value is None: return
    for key, number in value.items():
        if "token" in key and number is not None and (not isinstance(number, int) or number < 0):
            raise ValidationError(f"invalid token accounting {name}.{key}")

@dataclass(frozen=True)
class PreCallRequest:
    """Only pre-dispatch information; response/future fields are impossible."""
    request_ref: str
    request_hash: str
    features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_ref or not re.fullmatch(r"[a-zA-Z0-9_./:-]+", self.request_ref):
            raise ValidationError("invalid request artifact reference")
        if not re.fullmatch(r"[0-9a-f]{64}", self.request_hash):
            raise ValidationError("request_hash must be sha256 hex")

class Verdict(str, Enum):
    """One orientation's raw judge output: A/B are this pair's swap-mapped slots."""
    A_BETTER = "A_BETTER"; B_BETTER = "B_BETTER"; EQUIVALENT = "EQUIVALENT"
    BOTH_INADEQUATE = "BOTH_INADEQUATE"; UNCERTAIN = "UNCERTAIN"

class AggregatedPreference(str, Enum):
    """The pair's backend-identity outcome after mapping both orientations back
    to edge/cloud -- a different axis than Verdict (A/B slot), so a distinct type."""
    CLOUD_PREFERRED = "CLOUD_PREFERRED"; EDGE_PREFERRED = "EDGE_PREFERRED"
    EQUIVALENT = "EQUIVALENT"; BOTH_INADEQUATE = "BOTH_INADEQUATE"; UNCERTAIN = "UNCERTAIN"

@dataclass(frozen=True)
class RecordBase:
    schema_version: int
    protocol_version: str
    campaign_id: str
    task_id: str
    trajectory_id: str
    source: str
    cohort: str
    split: str
    provenance: Mapping[str, Any]
    validity: str = "valid"
    missing_reasons: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version < 1 or not self.protocol_version or not self.source or not self.cohort:
            raise ValidationError("invalid version/source/cohort")
        _validate_common(self)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum): return value.value
    if is_dataclass(value): return _jsonable(asdict(value))
    if isinstance(value, Mapping): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value

@dataclass(frozen=True)
class PrefixOutcome(RecordBase):
    prefix_id: str = ""
    source_policy: str = ""
    backend_fingerprint_id: str = ""
    call_index: int = 0
    predecision_request_ref: str | None = None
    predecision_request_hash: str | None = None
    tool_schema_ref: str | None = None
    tool_schema_hash: str | None = None
    history_integrity: str = "unknown"
    predecision_features: Mapping[str, Any] = field(default_factory=dict)
    features_version: str = ""
    remaining_budget_at_prefix: Mapping[str, Any] = field(default_factory=dict)
    core_sample_selected: bool = False
    selection_probability: float | None = None
    terminal_grade_ref: str | None = None
    resolved: bool | None = None
    label_valid: bool = False
    termination_reason: str = ""
    outcome_censored: bool = False
    sampling_metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate(); validate_id(self.prefix_id, "prefix_id")
        if self.call_index < 0 or self.selection_probability is not None and not 0 < self.selection_probability <= 1:
            raise ValidationError("invalid prefix sampling/call index")
        for n in ("predecision_request_ref", "tool_schema_ref", "terminal_grade_ref"):
            if getattr(self, n) is not None and not getattr(self, n): raise ValidationError(f"empty {n}")

@dataclass(frozen=True)
class Branch:
    branch_id: str
    initial_backend_fingerprint_id: str
    initial_candidate_ref: str
    initial_invocation_id: str
    continuation_trajectory_ref: str | None
    grader_ref: str | None
    final_patch_ref: str | None
    final_patch_hash: str | None
    resolved: bool | None
    label_valid: bool
    termination_reason: str
    remaining_total_cost_usd: float | None
    remaining_active_seconds: float | None
    input_usage: Mapping[str, Any]
    output_usage: Mapping[str, Any]
    cost_integrity: str
    missing_reasons: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        validate_id(self.branch_id, "branch_id"); validate_id(self.initial_invocation_id, "initial_invocation_id")
        if self.remaining_total_cost_usd is not None and self.remaining_total_cost_usd < 0: raise ValidationError("negative remaining cost")
        if self.remaining_active_seconds is not None and self.remaining_active_seconds < 0: raise ValidationError("negative remaining time")
        _usage(self.input_usage, "input_usage"); _usage(self.output_usage, "output_usage")
        for n in ("continuation_trajectory_ref", "grader_ref", "final_patch_ref", "final_patch_hash"):
            if getattr(self, n) is None and n not in self.missing_reasons: raise ValidationError(f"null branch field {n!r} lacks a missing reason")

@dataclass(frozen=True)
class BranchOutcome(RecordBase):
    branch_pair_id: str = ""
    prefix_id: str = ""
    checkpoint_id: str = ""
    checkpoint_certificate_ref: str = ""
    source_trajectory_id: str = ""
    source_policy: str = ""
    sampling_probability: float | None = None
    intervention: str = "next_main_call_only"
    continuation_policy: str = "edge-only-v1"
    remaining_budget: Mapping[str, Any] = field(default_factory=dict)
    candidate_generation_protocol: str = "fresh_one_per_backend"
    repeat_index: int = 0
    edge_branch: Branch | None = None
    cloud_branch: Branch | None = None
    pair_valid: bool = False
    observed_pair_class: str | None = None
    diagnostic_flags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        super().validate()
        for n in ("branch_pair_id", "prefix_id", "checkpoint_id", "source_trajectory_id"): validate_id(getattr(self, n), n)
        if self.edge_branch is None or self.cloud_branch is None: raise ValidationError("both B branches are required")
        self.edge_branch.validate(); self.cloud_branch.validate()
        if self.intervention != "next_main_call_only" or self.continuation_policy != "edge-only-v1": raise ValidationError("unsupported B protocol")
        if self.sampling_probability is not None and not 0 < self.sampling_probability <= 1: raise ValidationError("invalid sampling probability")

@dataclass(frozen=True)
class ServingCall(RecordBase):
    invocation_id: str = ""
    logical_call_id: str = ""
    prefix_id: str | None = None
    branch_id: str | None = None
    preference_pair_id: str | None = None
    purpose: str = "original"
    invocation_kind: Literal["original", "replay"] = "original"
    backend_fingerprint_id: str = ""
    logical_request_hash: str = ""
    rendered_request_hash: str = ""
    attempt_index: int = 0
    pre_dispatch_snapshot: Mapping[str, Any] = field(default_factory=dict)
    snapshot_timestamp: str | None = None
    snapshot_age_ms: float | None = None
    snapshot_source: str | None = None
    request_start: str | None = None
    first_byte: str | None = None
    first_content: str | None = None
    end: str | None = None
    measured_timings: Mapping[str, Any] = field(default_factory=dict)
    raw_usage: Mapping[str, Any] = field(default_factory=dict)
    normalised_usage: Mapping[str, Any] = field(default_factory=dict)
    usage_integrity: str = "unknown"
    status: str = "unknown"
    response_ref: str | None = None
    error_ref: str | None = None
    cost_basis: str | None = None
    rate_version: str | None = None
    observed_or_estimated_cost: float | None = None
    latency_censored: bool = False
    measurement_quality_flags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        super().validate(); validate_id(self.invocation_id, "invocation_id"); validate_id(self.logical_call_id, "logical_call_id")
        if self.attempt_index < 0 or self.observed_or_estimated_cost is not None and self.observed_or_estimated_cost < 0: raise ValidationError("invalid attempt/cost")
        _duration_map(self.measured_timings, "measured_timings"); _usage(self.raw_usage, "raw_usage"); _usage(self.normalised_usage, "normalised_usage")
        if self.invocation_kind not in {"original", "replay"}: raise ValidationError("invalid invocation kind")

@dataclass(frozen=True)
class PreferenceAnnotation(RecordBase):
    preference_pair_id: str = ""
    prefix_id: str = ""
    candidate_a_invocation_id: str = ""
    candidate_b_invocation_id: str = ""
    judge_invocation_id: str | None = None
    orientation: Literal["A=edge,B=cloud", "A=cloud,B=edge"] = "A=edge,B=cloud"
    verdict: Verdict = Verdict.UNCERTAIN
    reason_codes: list[str] = field(default_factory=list)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    aggregated_verdict: AggregatedPreference | None = None
    aggregation_status: str = "incomplete"

    def validate(self) -> None:
        super().validate(); validate_id(self.preference_pair_id, "preference_pair_id"); validate_id(self.prefix_id, "prefix_id")
        for n in ("candidate_a_invocation_id", "candidate_b_invocation_id"): validate_id(getattr(self, n), n)
        if self.judge_invocation_id is not None: validate_id(self.judge_invocation_id, "judge_invocation_id")
        if not isinstance(self.verdict, Verdict):
            try: object.__setattr__(self, "verdict", Verdict(self.verdict))
            except ValueError as e: raise ValidationError("invalid verdict") from e
        if self.aggregated_verdict is not None and not isinstance(self.aggregated_verdict, AggregatedPreference):
            try: object.__setattr__(self, "aggregated_verdict", AggregatedPreference(self.aggregated_verdict))
            except ValueError as e: raise ValidationError("invalid aggregated verdict") from e

RECORD_TYPES = {"A": PrefixOutcome, "B": BranchOutcome, "C": ServingCall, "preference": PreferenceAnnotation}

def parse_record(dataset: str, value: Mapping[str, Any]) -> RecordBase:
    cls = RECORD_TYPES[dataset]
    data = dict(value)
    if cls is BranchOutcome:
        for n in ("edge_branch", "cloud_branch"):
            if isinstance(data.get(n), dict): data[n] = Branch(**data[n])
    if cls is PreferenceAnnotation:
        if data.get("verdict") is not None: data["verdict"] = Verdict(data["verdict"])
        if data.get("aggregated_verdict") is not None: data["aggregated_verdict"] = AggregatedPreference(data["aggregated_verdict"])
    obj = cls(**data); obj.validate(); return obj
