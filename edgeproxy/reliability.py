"""Rung 1 of the routing policy ladder: verify-then-escalate.

Tracks a rolling local schema-validity rate per tool-suite class and opens a
circuit breaker for a class once its failure rate crosses a threshold,
routing future calls in that class to cloud instead. A small probe fraction
is kept even while the circuit is open, so a class can recover once local
starts succeeding again rather than being permanently exiled.

Deliberately scoped to *future* calls only. Fixing the call that just failed
(buffer the local response, validate before forwarding, replay to cloud on
failure) was considered and set aside: `edgeproxy/server.py`'s streaming
path is a true tee — it forwards each chunk to the client the instant it
arrives, and only reconstructs the full message afterward, purely for
tracing. Buffering long enough to validate before forwarding would hold
back all output on every local call, including the ones we already know run
30+ minutes (see claude-memory/wiki/findings/A qutebrowser replication seed
got stuck in a 55K-token thinking loop on the same decision point.md) — an
unverified risk of breaking exactly the calls that most need to stream,
in exchange for catching a failure mode currently measured at ~2-3% of
calls. See claude-memory/wiki/decisions/Rescope the routing policy ladder
to a 3 to 6 week window.md for the full reasoning.

The breaker itself is impure (in-memory, mutated by every call) by design,
matching the project's existing pattern for `CloudCacheTracker` and
`CohortTracker`: state collection lives outside the pure `Policy.decide()`,
which only ever consumes a precomputed `CallFeatures.local_reliability_blocked`
bool. This keeps `decide()` a pure function of its input, so it stays
replayable and testable exactly like every other policy in `router.py`.
"""

from __future__ import annotations

import random
import threading
from collections import deque

WINDOW = 20
MIN_SAMPLES = 5
DEFAULT_THRESHOLD = 0.25
PROBE_FRACTION = 0.10


class ReliabilityCircuitBreaker:
    """Per-class rolling local schema-validity tracker with a probe stream."""

    def __init__(
        self,
        window: int = WINDOW,
        min_samples: int = MIN_SAMPLES,
        threshold: float = DEFAULT_THRESHOLD,
        probe_fraction: float = PROBE_FRACTION,
        rng: random.Random | None = None,
    ) -> None:
        self._window = window
        self._min_samples = min_samples
        self._threshold = threshold
        self._probe_fraction = probe_fraction
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._classes: dict[str, deque[bool]] = {}

    def record(self, class_key: str | None, success: bool) -> None:
        """Record one local call's outcome for its tool-suite class."""
        if class_key is None:
            return
        with self._lock:
            outcomes = self._classes.setdefault(
                class_key, deque(maxlen=self._window)
            )
            outcomes.append(success)

    def failure_rate(self, class_key: str | None) -> float | None:
        """Rolling failure rate for a class, or None if too few samples."""
        if class_key is None:
            return None
        with self._lock:
            outcomes = self._classes.get(class_key)
            if outcomes is None or len(outcomes) < self._min_samples:
                return None
            failures = sum(1 for ok in outcomes if not ok)
            n = len(outcomes)
        return failures / n

    def should_block(self, class_key: str | None) -> tuple[bool, str]:
        """Return (block_local, note). block_local=True means route cloud.

        `note` explains why, for the trace: "reliability-ok" (below
        threshold or too few samples to judge), "reliability-probe" (circuit
        open, but this call was selected for the recovery probe stream), or
        "reliability-circuit-open" (blocked).
        """
        rate = self.failure_rate(class_key)
        if rate is None or rate <= self._threshold:
            return False, "reliability-ok"
        if self._rng.random() < self._probe_fraction:
            return False, "reliability-probe"
        return True, "reliability-circuit-open"
