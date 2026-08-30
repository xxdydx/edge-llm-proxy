"""Observe fan-out cohorts from proxied Agent calls without changing placement."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
from typing import Any, Callable, Iterable, Mapping


def iter_strings(value: Any) -> Iterable[str]:
    """Yield strings from a JSON-shaped value without retaining a flattened copy."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def request_contains_prompt(request: Any, prompt: str) -> bool:
    """Return whether a delegation prompt occurs verbatim in a child request."""
    return bool(prompt) and any(prompt in text for text in iter_strings(request))


def _cohort_id(session_id: str, parent_call_id: str) -> str:
    digest = sha256()
    digest.update(b"edgeproxy-cohort-v1\0")
    digest.update(session_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(parent_call_id.encode("utf-8"))
    return digest.hexdigest()[:24]


def _agent_delegations(response: Any) -> list[tuple[str, str]]:
    content = response.get("content") if isinstance(response, Mapping) else None
    delegations: list[tuple[str, str]] = []
    for block in content or []:
        if (
            not isinstance(block, Mapping)
            or block.get("type") != "tool_use"
            or block.get("name") != "Agent"
            or not block.get("id")
        ):
            continue
        tool_input = block.get("input")
        prompt = tool_input.get("prompt") if isinstance(tool_input, Mapping) else None
        if isinstance(prompt, str) and prompt:
            delegations.append((str(block["id"]), prompt))
    return delegations


@dataclass
class _ObservedCohort:
    cohort_id: str
    session_id: str
    parent_call_id: str
    parent_backend: str | None
    completed_at_unix_s: float
    prompts_by_tool: dict[str, str]
    first_arrival_unix_s: float | None = None
    arrived_tools: set[str] = field(default_factory=set)


class CohortTracker:
    """Infer child lineage for tracing only; it never delays or routes a call."""

    def __init__(self, *, window_ms: float) -> None:
        if window_ms < 0:
            raise ValueError("window_ms must be non-negative")
        self.window_ms = float(window_ms)
        self._lock = threading.Lock()
        self._cohorts: list[_ObservedCohort] = []

    def observe_parent(
        self,
        *,
        call_id: str,
        session_id: str | None,
        backend: str | None,
        response: Any,
        completed_at_unix_s: float,
    ) -> dict[str, Any] | None:
        """Register one completed response that emitted valid Agent delegations."""
        if not session_id:
            return None
        delegations = _agent_delegations(response)
        if not delegations:
            return None
        cohort = _ObservedCohort(
            cohort_id=_cohort_id(str(session_id), str(call_id)),
            session_id=str(session_id),
            parent_call_id=str(call_id),
            parent_backend=backend,
            completed_at_unix_s=float(completed_at_unix_s),
            prompts_by_tool=dict(delegations),
        )
        with self._lock:
            self._cohorts.append(cohort)
        return {
            "schema_version": 1,
            "cohort_id": cohort.cohort_id,
            "role": "parent",
            "parent_call_id": cohort.parent_call_id,
            "parent_backend": cohort.parent_backend,
            "expected_width": len(cohort.prompts_by_tool),
            "detection_method": "parent_agent_tool_uses",
            "detection_confidence": "exact",
            "observe_only": True,
            "configured_window_ms": self.window_ms,
        }

    def match_child(
        self,
        *,
        session_id: str | None,
        request: Any,
        arrived_at_unix_s: float,
    ) -> dict[str, Any] | None:
        """Match a child request to the most recent compatible delegation prompt."""
        if not session_id:
            return None
        matches: list[tuple[_ObservedCohort, str]] = []
        with self._lock:
            for cohort in self._cohorts:
                if cohort.session_id != str(session_id):
                    continue
                for tool_id, prompt in cohort.prompts_by_tool.items():
                    if request_contains_prompt(request, prompt):
                        matches.append((cohort, tool_id))
            if not matches:
                return None
            # Repeated delegation text can legitimately occur after a retry. The
            # newest completed parent is the only causal candidate available from
            # proxied traffic, and the lower confidence makes that ambiguity visible.
            matches.sort(
                key=lambda item: (
                    item[0].completed_at_unix_s,
                    item[0].cohort_id,
                    item[1],
                ),
                reverse=True,
            )
            cohort, tool_id = matches[0]
            ambiguous = len(matches) > 1
            if cohort.first_arrival_unix_s is None:
                cohort.first_arrival_unix_s = float(arrived_at_unix_s)
            cohort.arrived_tools.add(tool_id)
            offset_ms = max(
                0.0,
                (float(arrived_at_unix_s) - cohort.first_arrival_unix_s) * 1000,
            )
            arrival_index = len(cohort.arrived_tools)

        return {
            "schema_version": 1,
            "cohort_id": cohort.cohort_id,
            "role": "child",
            "parent_call_id": cohort.parent_call_id,
            "parent_tool_use_id": tool_id,
            "parent_backend": cohort.parent_backend,
            "expected_width": len(cohort.prompts_by_tool),
            "ready_width_at_arrival": arrival_index,
            "arrival_offset_ms": round(offset_ms, 3),
            "configured_window_ms": self.window_ms,
            "within_configured_window": offset_ms <= self.window_ms,
            "late_joiner": offset_ms > self.window_ms,
            "detection_method": (
                "content_and_causal_time" if ambiguous else "content_exact"
            ),
            "detection_confidence": "medium" if ambiguous else "high",
            "candidate_count": len(matches),
            "observe_only": True,
            "actual_wait_ms": 0.0,
            "dispatch_irrevocable": True,
        }


def exhaustive_backend_plan(
    call_ids: Iterable[str],
    score: Callable[[dict[str, str]], float],
) -> tuple[dict[str, str], float]:
    """Return the minimum-score local/cloud vector for a bounded ready cohort.

    The scoring model is deliberately injected: this helper adds no guessed
    latency, queue, cache, or cost coefficients and is safe to use in offline
    calibration or observe-only planning.
    """
    calls = tuple(str(call_id) for call_id in call_ids)
    if not calls:
        raise ValueError("at least one call is required")
    best_vector: dict[str, str] | None = None
    best_score: float | None = None
    for placements in product(("cloud", "local"), repeat=len(calls)):
        vector = dict(zip(calls, placements))
        value = float(score(vector))
        key = (value, tuple(placements))
        if best_score is None or key < (
            best_score,
            tuple(best_vector[call_id] for call_id in calls),
        ):
            best_vector = vector
            best_score = value
    assert best_vector is not None and best_score is not None
    return best_vector, best_score
