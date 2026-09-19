"""Bounded pre-dispatch history for shadow-only agentic scoring.

Each proxy episode/session/agent lane owns its own sequence. A decision sees
only earlier calls that *completed before its snapshot*. If an earlier call
is still in flight (or history was truncated), support is withheld rather
than pretending missing prior outcomes are zero. The scorer never mutates
placement.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

FAILURE_MARKERS = ("FAILED", "AssertionError", "Traceback (most recent call last)", "Error:")


def _tool_result_texts(request: dict[str, Any]) -> list[str]:
    for message in reversed(request.get("messages") or []):
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        texts = []
        for block in message["content"]:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            content = block.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(str(x.get("text") or "") for x in content if isinstance(x, dict) and x.get("type") == "text")
        if texts:
            return texts
    return []


@dataclass(frozen=True)
class PriorCall:
    sequence: int
    placement: str
    tool_error: bool
    test_failure: bool
    truncated_or_invalid: bool
    tool_names: tuple[str, ...]


@dataclass
class Lane:
    next_sequence: int = 0
    pending: set[int] = field(default_factory=set)
    calls: deque[PriorCall] = field(default_factory=deque)
    truncated: bool = False


class AgenticHistory:
    def __init__(self, *, max_calls_per_lane: int = 256, max_lanes: int = 128) -> None:
        if max_calls_per_lane < 4 or max_lanes < 1:
            raise ValueError("invalid history bounds")
        self.max_calls_per_lane = max_calls_per_lane
        self.max_lanes = max_lanes
        self.lanes: dict[str, Lane] = {}

    def begin(self, lane_key: str) -> int:
        lane = self.lanes.get(lane_key)
        if lane is None:
            if len(self.lanes) >= self.max_lanes:
                # Never evict a live lane and risk mixing its sequence with a
                # fresh one. The caller must mark scoring unsupported.
                raise RuntimeError("agentic history lane capacity reached")
            lane = self.lanes[lane_key] = Lane()
        sequence = lane.next_sequence
        lane.next_sequence += 1
        lane.pending.add(sequence)
        return sequence

    def snapshot(self, lane_key: str, sequence: int) -> tuple[dict[str, float] | None, str | None]:
        lane = self.lanes.get(lane_key)
        if lane is None or sequence not in lane.pending:
            return None, "history-reservation-missing"
        if lane.truncated:
            return None, "history-truncated"
        if any(i < sequence for i in lane.pending):
            return None, "earlier-call-in-flight"
        prior = sorted(
            (c for c in lane.calls if c.sequence < sequence),
            key=lambda c: c.sequence,
        )
        if len(prior) != sequence:
            return None, "history-gap"
        prev = prior[-1] if prior else None
        consecutive = 0
        if prev is not None:
            for call in reversed(prior):
                if call.placement != prev.placement:
                    break
                consecutive += 1
        recent = prior[-3:]
        signatures = [c.tool_names for c in prior[-4:] if c.tool_names]
        repair = bool(signatures) and signatures.count(signatures[-1]) >= 3
        return {
            "turn_index": float(sequence),
            "recent_tool_error_count": float(sum(c.tool_error for c in recent)),
            "recent_test_failure_count": float(sum(c.test_failure for c in recent)),
            "consecutive_same_backend_turns": float(consecutive),
            "prior_response_truncated_or_invalid": float(bool(prev and prev.truncated_or_invalid)),
            "repair_loop_flag": float(repair),
        }, None

    def complete(
        self, lane_key: str, sequence: int, *, request: dict[str, Any],
        errored_tool_result_density: float, placement: str,
        response: dict[str, Any] | None, tool_use_blocks: list[dict[str, Any]],
    ) -> None:
        lane = self.lanes.get(lane_key)
        if lane is None or sequence not in lane.pending:
            return  # idempotent finalize / already cleared
        lane.pending.remove(sequence)
        if response is None or placement not in ("local", "cloud"):
            lane.truncated = True  # unknown outcome cannot be invented
            return
        texts = _tool_result_texts(request)
        tool_names = tuple(
            str(block.get("name")) for block in (response.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name")
        )
        lane.calls.append(PriorCall(
            sequence=sequence,
            placement=placement,
            tool_error=errored_tool_result_density > 0,
            test_failure=any(marker in text for text in texts for marker in FAILURE_MARKERS),
            truncated_or_invalid=(
                response.get("stop_reason") == "max_tokens"
                or any(block.get("schema_valid") is False for block in tool_use_blocks)
            ),
            tool_names=tool_names,
        ))
        if len(lane.calls) > self.max_calls_per_lane:
            lane.calls.popleft()
            lane.truncated = True

    def abort(self, lane_key: str, sequence: int) -> None:
        """Discard an incomplete attempt without inventing a prior response.

        Future snapshots remain unsupported: the missing outcome is a real
        history gap, not a completed turn with zero errors or tool actions.
        """
        lane = self.lanes.get(lane_key)
        if lane is None or sequence not in lane.pending:
            return
        lane.pending.remove(sequence)
        lane.truncated = True
