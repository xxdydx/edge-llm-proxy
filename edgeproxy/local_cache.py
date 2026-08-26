"""Observe vLLM prefix-cache state before placement.

The patched vLLM server exposes a read-only Anthropic-shaped endpoint that
renders the request exactly as generation would, then asks the live KV-cache
manager how many leading tokens are resident.  The probe performs no prefill
or generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx


def _count(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


@dataclass(frozen=True)
class LocalCachePrediction:
    available: bool
    state: str
    input_tokens: int | None = None
    estimated_read_tokens: int | None = None
    estimated_read_fraction: float | None = None
    probe_ms: float | None = None
    source: str = "vllm-live-probe"
    error: str | None = None

    def as_trace(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "state": self.state,
            "source": self.source,
            "input_tokens": self.input_tokens,
            "estimated_read_tokens": self.estimated_read_tokens,
            "estimated_read_fraction": self.estimated_read_fraction,
            "probe_ms": self.probe_ms,
            "error": self.error,
        }


async def probe_local_cache(
    client: httpx.AsyncClient,
    request_json: Mapping[str, Any],
    *,
    timeout_s: float = 5.0,
) -> LocalCachePrediction:
    """Return exact pre-request cache residency from the patched vLLM server."""
    started = time.monotonic()
    try:
        response = await client.post(
            "/v1/messages/count_cached_tokens",
            json=dict(request_json),
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        input_tokens = _count(payload.get("input_tokens"))
        cached_tokens = _count(payload.get("cached_tokens"))
        if input_tokens is None or cached_tokens is None or cached_tokens > input_tokens:
            raise ValueError("invalid count_cached_tokens response")
        fraction = cached_tokens / input_tokens if input_tokens else 0.0
        return LocalCachePrediction(
            available=True,
            state="warm" if cached_tokens > 0 else "cold",
            input_tokens=input_tokens,
            estimated_read_tokens=cached_tokens,
            estimated_read_fraction=round(fraction, 6),
            probe_ms=round((time.monotonic() - started) * 1000, 1),
        )
    except Exception as exc:
        return LocalCachePrediction(
            available=False,
            state="unavailable",
            probe_ms=round((time.monotonic() - started) * 1000, 1),
            error=f"{type(exc).__name__}: {exc}",
        )


def local_cache_trace(
    prediction: LocalCachePrediction,
    usage: Mapping[str, Any] | None = None,
    *,
    selected: bool,
) -> dict[str, Any]:
    usage = usage or {}
    read = _count(usage.get("cache_read_input_tokens"))
    creation = _count(usage.get("cache_creation_input_tokens"))
    uncached = _count(usage.get("input_tokens"))
    details_available = read is not None or creation is not None
    total_input = (
        (read or 0) + (creation or 0) + (uncached or 0)
        if details_available
        else None
    )

    predicted = prediction.estimated_read_tokens
    absolute_error = (
        abs(predicted - read)
        if selected and predicted is not None and read is not None
        else None
    )
    error_fraction_of_input = (
        absolute_error / total_input
        if absolute_error is not None and total_input
        else None
    )
    relative_error = (
        absolute_error / read
        if absolute_error is not None and read
        else (0.0 if absolute_error == 0 else None)
    )

    return {
        "schema_version": 1,
        "mode": "observe",
        "prediction": prediction.as_trace(),
        "actual": {
            "selected_backend": selected,
            "available": selected and details_available,
            "total_input_tokens": total_input if selected else None,
            "cache_read_input_tokens": read if selected else None,
            "cache_creation_input_tokens": creation if selected else None,
            "uncached_input_tokens": uncached if selected else None,
        },
        "agreement": {
            "warm_prediction_correct": (
                (predicted > 0) == (read > 0)
                if selected and predicted is not None and read is not None
                else None
            ),
            "cached_token_error": absolute_error,
            "cached_token_relative_error": (
                round(relative_error, 6) if relative_error is not None else None
            ),
            "cached_token_error_fraction_of_input": (
                round(error_fraction_of_input, 6)
                if error_fraction_of_input is not None
                else None
            ),
            "within_5_percent_of_input": (
                error_fraction_of_input <= 0.05
                if error_fraction_of_input is not None
                else None
            ),
        },
    }
