"""Async leader/follower dispatch barrier for matched fan-out cohorts.

The per-call policy remains the placement authority.  This component only
coordinates *when* already-local sibling calls are dispatched, plus the one
explicit cold-branch warming exemption approved for a deterministic leader.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from .local_cache import LocalCachePrediction, probe_local_cache


# Every observed sibling edge shared at least 84.1% of the child prompt.  Keep
# four percentage points of margin so a probe must show the cohort-shaped shared
# prefix, rather than merely a small static system/tool prefix, before release.
MIN_SHARED_PREFIX_FRACTION = 0.80
DEFAULT_BARRIER_TIMEOUT_MS = 5_000.0
DEFAULT_POLL_INTERVAL_MS = 25.0


@dataclass
class _CohortDispatchState:
    leader_tool_use_id: str
    first_arrival_monotonic: float
    updated_at_monotonic: float
    leader_call_id: str | None = None
    leader_input_tokens: int | None = None
    leader_dispatched: bool = False


@dataclass(frozen=True)
class DispatchTicket:
    cohort_id: str
    call_id: str
    parent_tool_use_id: str
    leader_tool_use_id: str
    leader_call_id: str | None
    role: str
    within_collection_window: bool
    first_arrival_monotonic: float


class CohortCoordinator:
    """Atomically designate leaders and asynchronously release local followers."""

    def __init__(
        self,
        *,
        window_ms: float,
        timeout_ms: float = DEFAULT_BARRIER_TIMEOUT_MS,
        poll_interval_ms: float = DEFAULT_POLL_INTERVAL_MS,
    ) -> None:
        if window_ms < 0:
            raise ValueError("window_ms must be non-negative")
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        if poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be positive")
        self.window_ms = float(window_ms)
        self.timeout_ms = float(timeout_ms)
        self.poll_interval_ms = float(poll_interval_ms)
        self._lock = threading.Lock()
        self._states: dict[str, _CohortDispatchState] = {}

    def arrive(
        self,
        *,
        call_id: str,
        detection: Mapping[str, Any] | None,
        input_tokens: int | None,
        arrived_at_monotonic: float,
    ) -> DispatchTicket | None:
        """Return an atomic leader/follower designation for a matched child.

        The tracker supplies ``leader_tool_use_id`` from delegation registration
        order.  Only the first request matching that ID can claim leadership;
        retries or duplicate concurrent matches become followers.
        """
        if not detection or detection.get("role") != "child":
            return None
        cohort_id = detection.get("cohort_id")
        tool_use_id = detection.get("parent_tool_use_id")
        leader_tool_use_id = detection.get("leader_tool_use_id")
        if not all(isinstance(value, str) and value for value in (
            cohort_id,
            tool_use_id,
            leader_tool_use_id,
        )):
            return None

        now = float(arrived_at_monotonic)
        with self._lock:
            self._prune_locked(now)
            state = self._states.get(cohort_id)
            if state is None:
                state = _CohortDispatchState(
                    leader_tool_use_id=leader_tool_use_id,
                    first_arrival_monotonic=now,
                    updated_at_monotonic=now,
                )
                self._states[cohort_id] = state
            state.updated_at_monotonic = now

            is_designated_leader = (
                tool_use_id == state.leader_tool_use_id
                and state.leader_call_id is None
            )
            if is_designated_leader:
                state.leader_call_id = str(call_id)
                state.leader_input_tokens = input_tokens
            role = "leader" if is_designated_leader else "follower"
            leader_call_id = state.leader_call_id
            first_arrival = state.first_arrival_monotonic

        return DispatchTicket(
            cohort_id=cohort_id,
            call_id=str(call_id),
            parent_tool_use_id=tool_use_id,
            leader_tool_use_id=leader_tool_use_id,
            leader_call_id=leader_call_id,
            role=role,
            within_collection_window=bool(
                detection.get("within_configured_window")
            ),
            first_arrival_monotonic=first_arrival,
        )

    def mark_leader_dispatched(
        self, ticket: DispatchTicket, *, input_tokens: int | None
    ) -> None:
        if ticket.role != "leader":
            return
        with self._lock:
            state = self._states.get(ticket.cohort_id)
            if state is None or state.leader_call_id != ticket.call_id:
                return
            state.leader_dispatched = True
            if input_tokens is not None:
                state.leader_input_tokens = input_tokens
            state.updated_at_monotonic = time.monotonic()

    def trace_for(
        self,
        ticket: DispatchTicket,
        *,
        policy_placement: str,
        policy_reason: str,
        warming_exemption: bool,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan_version": 1,
            "cohort_id": ticket.cohort_id,
            "role": ticket.role,
            "leader_call_id": (
                ticket.call_id if ticket.role == "leader" else ticket.leader_call_id
            ),
            "leader_tool_use_id": ticket.leader_tool_use_id,
            "parent_tool_use_id": ticket.parent_tool_use_id,
            "policy_placement": policy_placement,
            "policy_reason": policy_reason,
            "warming_exemption": warming_exemption,
            "configured_collection_window_ms": self.window_ms,
            "configured_timeout_ms": self.timeout_ms,
            "poll_interval_ms": self.poll_interval_ms,
            "actual_wait_ms": 0.0,
            "release_reason": (
                "leader-immediate" if ticket.role == "leader" else "pending"
            ),
            "probe_attempts": 0,
            "shared_prefix_threshold_tokens": None,
            "release_cached_tokens": None,
            "realized_outcome": None,
        }

    async def hold_follower(
        self,
        ticket: DispatchTicket,
        *,
        client: httpx.AsyncClient,
        request_json: Mapping[str, Any],
        initial_prediction: LocalCachePrediction | None,
        trace: dict[str, Any],
    ) -> None:
        """Poll without blocking the event loop, bounded by one absolute deadline."""
        if ticket.role != "follower":
            return
        started = time.monotonic()
        deadline = ticket.first_arrival_monotonic + self.timeout_ms / 1000.0

        if not ticket.within_collection_window or started >= deadline:
            trace["release_reason"] = "hold-window-elapsed"
            trace["actual_wait_ms"] = 0.0
            return

        with self._lock:
            state = self._states.get(ticket.cohort_id)
            leader_input_tokens = state.leader_input_tokens if state else None
            leader_dispatched = bool(state and state.leader_dispatched)

        follower_input_tokens = (
            initial_prediction.input_tokens if initial_prediction is not None else None
        )
        threshold = (
            max(
                1,
                math.floor(
                    min(leader_input_tokens, follower_input_tokens)
                    * MIN_SHARED_PREFIX_FRACTION
                ),
            )
            if (
                leader_dispatched
                and leader_input_tokens is not None
                and follower_input_tokens is not None
            )
            else None
        )
        trace["shared_prefix_threshold_tokens"] = threshold

        initial_cached = (
            initial_prediction.estimated_read_tokens
            if initial_prediction is not None and initial_prediction.available
            else None
        )
        if (
            leader_dispatched
            and threshold is not None
            and initial_cached is not None
            and initial_cached >= threshold
        ):
            trace["release_reason"] = "cache-warm-confirmed"
            trace["release_cached_tokens"] = initial_cached
            trace["actual_wait_ms"] = 0.0
            return

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                trace["release_reason"] = "timeout"
                break

            # A follower can beat the designated leader to the proxy.  Refresh
            # the leader snapshot until dispatch is visible, then freeze the
            # threshold: leader token counts cannot change after dispatch, so
            # later cache-probe iterations do not need to take the lock.
            if not leader_dispatched:
                with self._lock:
                    state = self._states.get(ticket.cohort_id)
                    leader_input_tokens = (
                        state.leader_input_tokens if state else None
                    )
                    leader_dispatched = bool(state and state.leader_dispatched)
                if (
                    leader_dispatched
                    and leader_input_tokens is not None
                    and follower_input_tokens is not None
                ):
                    threshold = max(
                        1,
                        math.floor(
                            min(leader_input_tokens, follower_input_tokens)
                            * MIN_SHARED_PREFIX_FRACTION
                        ),
                    )
                    trace["shared_prefix_threshold_tokens"] = threshold

            per_probe_timeout = min(remaining, 0.25)
            try:
                prediction = await asyncio.wait_for(
                    probe_local_cache(
                        client,
                        request_json,
                        timeout_s=per_probe_timeout,
                    ),
                    timeout=remaining,
                )
                trace["probe_attempts"] += 1
            except TimeoutError:
                trace["release_reason"] = "timeout"
                break

            cached = prediction.estimated_read_tokens
            if (
                prediction.available
                and threshold is not None
                and cached is not None
                and cached >= threshold
            ):
                trace["release_reason"] = "cache-warm-confirmed"
                trace["release_cached_tokens"] = cached
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                trace["release_reason"] = "timeout"
                break
            await asyncio.sleep(min(self.poll_interval_ms / 1000.0, remaining))

        trace["actual_wait_ms"] = round((time.monotonic() - started) * 1000, 3)

    def _prune_locked(self, now: float) -> None:
        retention_s = max(60.0, self.timeout_ms / 1000.0 * 2)
        expired = [
            cohort_id
            for cohort_id, state in self._states.items()
            if now - state.updated_at_monotonic > retention_s
        ]
        for cohort_id in expired:
            del self._states[cohort_id]
