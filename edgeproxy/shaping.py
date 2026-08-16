"""Cloud-path link shaping, and a passive estimate of the real link.

Two separate things that must not be confused:

  LinkShaper   the experiment's independent variable. You set it. The router
               never sees it.
  LinkMonitor  what the router may observe, estimated from its own past cloud
               calls the way a real client would.

Handing the router the shaped value would test nothing.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

# Named scenarios, so a latency number is always attached to a claim about a
# deployment rather than being an unlabelled constant. `none` is the real link.
PRESETS: dict[str, tuple[float, float, float]] = {
    # name:            delay_ms, jitter_ms, mbps
    "none": (0.0, 0.0, 0.0),
    "colo": (5.0, 1.0, 200.0),
    "branch-office": (20.0, 5.0, 50.0),
    "home-broadband": (30.0, 10.0, 5.0),
    "cellular": (80.0, 40.0, 5.0),
}


@dataclass(frozen=True)
class LinkShaper:
    delay_ms: float = 0.0
    jitter_ms: float = 0.0
    bandwidth_mbps: float = 0.0  # 0 = unlimited
    preset: str = "none"

    @classmethod
    def from_preset(cls, name: str) -> LinkShaper:
        if name not in PRESETS:
            raise SystemExit(f"unknown preset {name!r} — one of: {', '.join(PRESETS)}")
        delay, jitter, mbps = PRESETS[name]
        return cls(delay, jitter, mbps, name)

    @property
    def active(self) -> bool:
        return self.delay_ms > 0 or self.bandwidth_mbps > 0

    def transmission_ms(self, n_bytes: int) -> float:
        """Time to push n_bytes onto the link.

        Not negligible for this workload: request bodies run to ~100KB because
        of the tool definitions, which is ~160ms on a 5 Mbps uplink — the same
        order as the RTT. Delay-only shaping would understate cloud cost.
        """
        if self.bandwidth_mbps <= 0:
            return 0.0
        return (n_bytes * 8) / (self.bandwidth_mbps * 1e6) * 1000

    async def apply(self, n_bytes: int = 0) -> float:
        """Sleep for the modelled one-way cost; returns the ms actually waited."""
        if not self.active:
            return 0.0
        wait = self.delay_ms + self.transmission_ms(n_bytes)
        if self.jitter_ms > 0:
            wait += random.uniform(-self.jitter_ms, self.jitter_ms)
        wait = max(0.0, wait)
        await asyncio.sleep(wait / 1000)
        return round(wait, 1)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "preset": self.preset,
            "delay_ms": self.delay_ms,
            "jitter_ms": self.jitter_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
        }


class LinkMonitor:
    """Running estimate of cloud network latency, from observed calls only.

    A real edge client cannot know its RTT before making a request, so neither
    can this. Cold start is a stated property of any policy that consumes it,
    not an oversight.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.rtt_ewma: float | None = None
        self.jitter_ewma: float | None = None
        self.n_samples = 0
        self.last_sample_ts: float | None = None

    def observe(self, network_ms: float | None) -> None:
        if network_ms is None:
            return
        if self.rtt_ewma is None:
            self.rtt_ewma, self.jitter_ewma = network_ms, 0.0
        else:
            # Jitter updated against the *previous* mean, RFC 6298 style —
            # updating it after the mean would understate the deviation.
            self.jitter_ewma = (1 - self.alpha) * (self.jitter_ewma or 0.0) + self.alpha * abs(
                network_ms - self.rtt_ewma
            )
            self.rtt_ewma = (1 - self.alpha) * self.rtt_ewma + self.alpha * network_ms
        self.n_samples += 1
        self.last_sample_ts = time.time()

    def as_dict(self) -> dict[str, float | int | None]:
        stale = (
            round(time.time() - self.last_sample_ts, 1)
            if self.last_sample_ts is not None
            else None
        )
        return {
            "rtt_ewma_ms": round(self.rtt_ewma, 1) if self.rtt_ewma is not None else None,
            "jitter_ewma_ms": (
                round(self.jitter_ewma, 1) if self.jitter_ewma is not None else None
            ),
            "n_samples": self.n_samples,
            # Routing locally starves this estimate of samples; a policy using
            # it needs to know how old it is.
            "staleness_s": stale,
        }
