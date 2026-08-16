"""Split a request's wall-clock into network and server phases.

`ttft_ms` alone conflates RTT, TLS, uploading a ~100KB body, queueing, and
prefill. Routing is a latency tradeoff, so the two have to be separable.

Uses httpx's `extensions={"trace": ...}` hook, which forwards httpcore's
internal phase events. No extra dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ConnTiming:
    """`None` means the phase did not happen, not that it took no time.

    Connection reuse skips connect and TLS entirely; recording those as 0.0
    would drag the average down and understate cold-connection cost.
    """

    connect_ms: float | None = None
    tls_ms: float | None = None
    send_ms: float | None = None
    response_headers_ms: float | None = None

    @property
    def network_ms(self) -> float | None:
        parts = [
            p
            for p in (self.connect_ms, self.tls_ms, self.send_ms, self.response_headers_ms)
            if p is not None
        ]
        return round(sum(parts), 1) if parts else None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "connect_ms": self.connect_ms,
            "tls_ms": self.tls_ms,
            "send_ms": self.send_ms,
            "response_headers_ms": self.response_headers_ms,
        }


# Phases are measured between these event pairs. Ordered as they occur.
_PHASES: tuple[tuple[str, str, str], ...] = (
    ("connect_ms", "connection.connect_tcp.started", "connection.connect_tcp.complete"),
    ("tls_ms", "connection.start_tls.started", "connection.start_tls.complete"),
    ("send_ms", "http11.send_request_headers.started", "http11.send_request_body.complete"),
    (
        "response_headers_ms",
        "http11.send_request_body.complete",
        "http11.receive_response_headers.complete",
    ),
)


def make_trace_extension() -> tuple[dict[str, Any], Callable[[], ConnTiming]]:
    """Return `extensions` for build_request, plus a reader for the result.

    The reader is only meaningful once the response headers have arrived, i.e.
    after `client.send()` returns.
    """
    stamps: dict[str, float] = {}

    async def trace(event_name: str, info: dict[str, Any]) -> None:
        # First occurrence wins: a redirect or retry would otherwise overwrite
        # the phase we actually care about.
        stamps.setdefault(event_name, time.perf_counter())

    def read() -> ConnTiming:
        values: dict[str, float | None] = {}
        for field, start, end in _PHASES:
            if start in stamps and end in stamps:
                values[field] = round((stamps[end] - stamps[start]) * 1000, 2)
            else:
                values[field] = None
        return ConnTiming(**values)  # type: ignore[arg-type]

    return {"trace": trace}, read
