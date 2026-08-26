"""Anthropic list-price accounting for routed message calls.

Cloud calls are priced from provider-reported usage.  Local calls estimate the
cloud bill that was avoided: the observed local input/output counts supply the
request size while the pre-dispatch Anthropic cache prediction supplies the
counterfactual cache hit.  The latter is necessarily an estimate because the
request was never sent to Anthropic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .cloud_cache import (
    CACHE_TTLS,
    CloudCachePrediction,
    PrefixChain,
    minimum_cacheable_tokens,
)
from .trace.record import build_token_accounting


PRICING_BASIS = "anthropic-standard-global-list-2026-08-26"
PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPricing:
    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float


def pricing_for_model(model: str) -> ModelPricing | None:
    """Return standard global USD/MTok pricing for known Claude families.

    Match dated aliases as well as the short API aliases.  Unknown and
    non-Anthropic models deliberately remain unpriced.
    """
    name = model.lower()
    if any(part in name for part in ("claude-fable-5", "claude-mythos-5")):
        return ModelPricing(10.0, 12.5, 20.0, 1.0, 50.0)
    if "claude-sonnet-5" in name:
        return ModelPricing(2.0, 2.5, 4.0, 0.2, 10.0)
    if any(part in name for part in ("claude-sonnet-4-6", "claude-sonnet-4-5")):
        return ModelPricing(3.0, 3.75, 6.0, 0.3, 15.0)
    if "claude-haiku-4-5" in name:
        return ModelPricing(1.0, 1.25, 2.0, 0.1, 5.0)
    if any(
        part in name
        for part in (
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-opus-4-5",
        )
    ):
        return ModelPricing(5.0, 6.25, 10.0, 0.5, 25.0)
    if "claude-opus-4-1" in name or "claude-opus-4" in name:
        return ModelPricing(15.0, 18.75, 30.0, 1.5, 75.0)
    if "claude-sonnet-4" in name:
        return ModelPricing(3.0, 3.75, 6.0, 0.3, 15.0)
    if "claude-haiku-3-5" in name:
        return ModelPricing(0.8, 1.0, 1.6, 0.08, 4.0)
    if "claude-haiku-3" in name:
        return ModelPricing(0.25, 0.3, 0.5, 0.03, 1.25)
    return None


def _count(value: Any) -> int | None:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _usd(tokens: int, price_per_million: float) -> float:
    return tokens * price_per_million / PER_MILLION


def _priced_result(
    *,
    model: str,
    pricing: ModelPricing,
    placement: str,
    source: str,
    confidence: str,
    uncached: int,
    cache_read: int,
    creation_5m: int,
    creation_1h: int,
    output: int,
) -> dict[str, Any]:
    components = {
        "uncached_input_usd": _usd(uncached, pricing.input),
        "cache_read_input_usd": _usd(cache_read, pricing.cache_read),
        "cache_creation_5m_input_usd": _usd(creation_5m, pricing.cache_write_5m),
        "cache_creation_1h_input_usd": _usd(creation_1h, pricing.cache_write_1h),
        "output_usd": _usd(output, pricing.output),
    }
    cloud_cost = sum(components.values())
    saved = cloud_cost if placement == "local" else 0.0
    return {
        "available": True,
        "reason": None,
        "currency": "USD",
        "pricing_basis": PRICING_BASIS,
        "requested_model": model,
        "price_per_million_tokens": asdict(pricing),
        "source": source,
        "confidence": confidence,
        "counterfactual_cloud_tokens": {
            "uncached_input_tokens": uncached,
            "cache_read_input_tokens": cache_read,
            "cache_creation_5m_input_tokens": creation_5m,
            "cache_creation_1h_input_tokens": creation_1h,
            "output_tokens": output,
        },
        "components_usd": {key: round(value, 12) for key, value in components.items()},
        "cloud_cost_usd": round(cloud_cost, 12),
        "request_saved_usd": round(saved, 12),
        # TraceWriter fills this atomically in JSONL write order.
        "running_saved_usd": None,
    }


def _unavailable(model: str, reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "currency": "USD",
        "pricing_basis": PRICING_BASIS,
        "requested_model": model,
        "price_per_million_tokens": None,
        "source": None,
        "confidence": "unavailable",
        "counterfactual_cloud_tokens": None,
        "components_usd": None,
        "cloud_cost_usd": None,
        "request_saved_usd": None,
        "running_saved_usd": None,
    }


def _actual_cloud_partitions(usage: Mapping[str, Any]) -> tuple[int, int, int, int, int] | None:
    accounting = build_token_accounting(dict(usage))
    output = _count(accounting.get("output_tokens"))
    if output is None:
        return None
    if accounting.get("cache_details_available"):
        uncached = _count(accounting.get("uncached_input_tokens"))
        read = _count(accounting.get("cache_read_input_tokens"))
        created = _count(accounting.get("cache_creation_input_tokens"))
        if None in (uncached, read, created):
            return None
        breakdown = usage.get("cache_creation")
        breakdown = breakdown if isinstance(breakdown, Mapping) else {}
        one_hour = _count(breakdown.get("ephemeral_1h_input_tokens"))
        five_minute = _count(breakdown.get("ephemeral_5m_input_tokens"))
        if one_hour is None and five_minute is None:
            # A combined nonzero creation count cannot be priced correctly
            # without knowing whether it was a 5m or 1h write.
            if created:
                return None
            five_minute, one_hour = 0, 0
        else:
            one_hour = one_hour or 0
            five_minute = five_minute or 0
            if one_hour + five_minute != created:
                return None
        return uncached, read, five_minute, one_hour, output

    total = _count(accounting.get("input_tokens"))
    return (total, 0, 0, 0, output) if total is not None else None


def _scaled_point_tokens(chain: PrefixChain, point_index: int, total_input: int) -> int:
    estimated_total = chain.estimated_total_tokens
    if estimated_total <= 0:
        return 0
    point = chain.points[point_index]
    return min(total_input, round(point.estimated_tokens * total_input / estimated_total))


def _estimated_local_partitions(
    *,
    total_input: int,
    output: int,
    chain: PrefixChain,
    prediction: CloudCachePrediction,
) -> tuple[int, int, int, int, int, str] | None:
    if not chain.valid:
        return None

    if not chain.breakpoints:
        return total_input, 0, 0, 0, output, "no-cloud-cache-breakpoints"
    if prediction.state == "disabled":
        return None

    # A is Anthropic's highest cache hit. Unknown means the tracker has never
    # confirmed this lineage; use a conservative zero-hit estimate and expose
    # the lower confidence in the trace.
    if prediction.estimated_read_tokens is None:
        read = 0
        confidence = "conservative-unconfirmed-cache-state"
    else:
        read = min(total_input, prediction.estimated_read_tokens)
        confidence = "estimated-from-cloud-cache-tracker"

    scaled = [
        (bp, _scaled_point_tokens(chain, bp.point_index, total_input))
        for bp in chain.breakpoints
    ]
    deepest = max(depth for _, depth in scaled)
    minimum = minimum_cacheable_tokens(chain.model)
    if read == 0 and minimum is not None and deepest < minimum:
        return total_input, 0, 0, 0, output, confidence

    # Anthropic's mixed-TTL billing locations are A=highest hit,
    # B=highest later 1h breakpoint, C=last breakpoint. Charges are read A,
    # 1h write B-A, 5m write C-B, then ordinary input after C.
    later = [(bp, depth) for bp, depth in scaled if depth > read]
    one_hour_depths = [
        depth for bp, depth in later if bp.ttl_s == CACHE_TTLS["1h"]
    ]
    b = max(one_hour_depths, default=read)
    c = max([depth for _, depth in later], default=read)
    creation_1h = max(0, b - read)
    creation_5m = max(0, c - b)
    uncached = max(0, total_input - read - creation_1h - creation_5m)
    return uncached, read, creation_5m, creation_1h, output, confidence


def build_cost_savings(
    *,
    placement: str,
    requested_model: str,
    usage: Mapping[str, Any] | None,
    chain: PrefixChain | None = None,
    prediction: CloudCachePrediction | None = None,
) -> dict[str, Any]:
    """Price a cloud call or estimate the cloud bill avoided by a local call."""
    pricing = pricing_for_model(requested_model)
    if pricing is None:
        return _unavailable(requested_model, "unknown-model-pricing")
    usage = usage if isinstance(usage, Mapping) else {}

    if placement == "cloud":
        partitions = _actual_cloud_partitions(usage)
        if partitions is None:
            return _unavailable(requested_model, "provider-usage-unavailable")
        return _priced_result(
            model=requested_model,
            pricing=pricing,
            placement=placement,
            source="provider-reported-usage",
            confidence="actual-token-accounting",
            uncached=partitions[0],
            cache_read=partitions[1],
            creation_5m=partitions[2],
            creation_1h=partitions[3],
            output=partitions[4],
        )

    accounting = build_token_accounting(dict(usage))
    total_input = _count(accounting.get("input_tokens"))
    output = _count(accounting.get("output_tokens"))
    if total_input is None or output is None:
        return _unavailable(requested_model, "local-usage-unavailable")
    if chain is None or prediction is None:
        return _unavailable(requested_model, "cloud-cache-prediction-unavailable")
    partitions = _estimated_local_partitions(
        total_input=total_input,
        output=output,
        chain=chain,
        prediction=prediction,
    )
    if partitions is None:
        return _unavailable(requested_model, "cloud-cache-estimate-disabled")
    return _priced_result(
        model=requested_model,
        pricing=pricing,
        placement=placement,
        source="local-usage-plus-cloud-cache-prediction",
        confidence=partitions[5],
        uncached=partitions[0],
        cache_read=partitions[1],
        creation_5m=partitions[2],
        creation_1h=partitions[3],
        output=partitions[4],
    )
