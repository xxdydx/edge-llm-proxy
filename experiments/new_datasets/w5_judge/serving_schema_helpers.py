"""Synthetic-input helpers for valid W1 ServingCall records."""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, Sequence

from experiments.new_datasets.w1_storage.schemas import ServingCall

_CANONICAL = {
    "input_tokens_exact": ("input_tokens", "prompt_tokens", "input_token_count"),
    "output_tokens_exact": ("output_tokens", "completion_tokens", "output_token_count"),
    "reasoning_tokens_exact": ("reasoning_tokens", "reasoning_token_count"),
    "cache_read_tokens_exact": ("cache_read_tokens", "prompt_cache_hit_tokens", "cached_input_tokens"),
    "cache_write_tokens_exact": ("cache_write_tokens", "prompt_cache_creation_tokens"),
}


def normalize_usage(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    normalized: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for target, aliases in _CANONICAL.items():
        found = next((raw[key] for key in aliases if key in raw), None)
        normalized[target] = found
        if not any(key in raw for key in aliases):
            reasons[target] = "provider_did_not_report_field"
    normalized["missing_reasons"] = reasons
    return normalized, reasons


def decode_speed_measurement(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Do not infer token timing from a buffered single output chunk."""
    if len(chunks) <= 1:
        return {"tokens_per_second": None, "true_decode_tpot_ms": None, "buffering_mode": "buffered_single_chunk", "measurement_quality": "unavailable"}
    timed = [c for c in chunks if c.get("timestamp_ms") is not None and c.get("token_count") is not None]
    if len(timed) < 2:
        return {"tokens_per_second": None, "true_decode_tpot_ms": None, "buffering_mode": "untimed_stream", "measurement_quality": "unavailable"}
    first, last = timed[0], timed[-1]
    duration = float(last["timestamp_ms"]) - float(first["timestamp_ms"])
    tokens = sum(int(c["token_count"]) for c in timed[1:])
    if duration <= 0 or tokens <= 0:
        return {"tokens_per_second": None, "true_decode_tpot_ms": None, "buffering_mode": "invalid_timing", "measurement_quality": "unavailable"}
    return {"tokens_per_second": tokens / (duration / 1000), "true_decode_tpot_ms": duration / tokens, "buffering_mode": "per_token_timing", "measurement_quality": "measured"}


def shared_prefix_cost(shared_cost_usd: float, branch_a_total_usd: float, branch_b_total_usd: float) -> dict[str, float]:
    if min(shared_cost_usd, branch_a_total_usd, branch_b_total_usd) < 0:
        raise ValueError("costs must be non-negative")
    return {
        "shared_cost_usd": shared_cost_usd,
        "branch_a_total_cost_usd": branch_a_total_usd,
        "branch_b_total_cost_usd": branch_b_total_usd,
        "branch_a_marginal_cost_usd": max(0.0, branch_a_total_usd - shared_cost_usd),
        "branch_b_marginal_cost_usd": max(0.0, branch_b_total_usd - shared_cost_usd),
        "combined_allocated_cost_usd": shared_cost_usd + max(0.0, branch_a_total_usd - shared_cost_usd) + max(0.0, branch_b_total_usd - shared_cost_usd),
    }


def build_serving_call(pre_dispatch: Mapping[str, Any], post_dispatch: Mapping[str, Any], *, identity: Mapping[str, Any]) -> ServingCall:
    raw_usage = dict(post_dispatch.get("raw_usage") or post_dispatch.get("usage") or {})
    normalized, usage_reasons = normalize_usage(raw_usage)
    values = {f.name: None for f in fields(ServingCall) if f.name not in {"missing_reasons"}}
    values.update(identity)
    values.update({
        "pre_dispatch_snapshot": dict(pre_dispatch),
        "raw_usage": raw_usage,
        "normalised_usage": normalized,
        "measured_timings": dict(post_dispatch.get("measured_timings") or {}),
        "status": post_dispatch.get("status", "unknown"),
        "invocation_kind": identity.get("invocation_kind", "original"),
        "purpose": identity.get("purpose", "original"),
        "attempt_index": int(post_dispatch.get("attempt_index", 0)),
        "latency_censored": bool(post_dispatch.get("latency_censored", False)),
        "measurement_quality_flags": list(post_dispatch.get("measurement_quality_flags") or []),
    })
    missing: dict[str, str] = {"normalised_usage": "field-level reasons are in normalised_usage.missing_reasons"}
    for name in (f.name for f in fields(ServingCall) if f.name != "missing_reasons"):
        if values.get(name) is None and name not in missing:
            missing[name] = "not_observed_in_synthetic_fixture"
    values["missing_reasons"] = missing
    record = ServingCall(**values)
    record.validate()
    return record
