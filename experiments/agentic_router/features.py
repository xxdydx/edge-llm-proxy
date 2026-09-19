"""Derive the handful of per-call features that do not already exist in
edgeproxy's recorded traces, from data already present in AgentTrajectory --
no new live instrumentation, no re-running anything.

Everything edgeproxy already computes at request time (turn ordinal, tool
names, errored-tool-result density, exact local prompt tokens/budget, cache
prediction) is read straight from ``AgentCallRecord.router_features`` /
``.tokens`` / ``.cache_probe`` -- reused, not recomputed. Only genuinely
missing fields are derived here:

- ``previous_backend`` / ``consecutive_same_backend_turns`` (prior calls
  only; the current placement is not known until after the decision) -- no
  trajectory-level "last backend used" feature exists anywhere in edgeproxy
  (confirmed by code search before writing this).
- ``recent_tool_error_count`` -- a rolling window over the existing
  per-call ``errored_tool_result_density`` (router.py's own feature,
  the same one ``PredictedRiskPolicy`` already uses), not a new detector.
- ``recent_test_failure_count`` -- a lexical scan over the actual consumed
  tool-result text (edgeproxy records only *which* tool results were
  consumed, not their content-level success/failure, so this is new).
- ``prior_response_truncated_or_invalid`` -- read off the previous call's
  own already-recorded ``stop_reason`` / ``tool_use_blocks`` validity.
- ``repair_loop_flag`` -- a cheap heuristic: the same *previously produced*
  tool_names signature repeated across a short recent window. Current
  response tool_names are never used in a prospective feature.
- ``context_utilization_ratio`` -- ``local_prompt_tokens / local_token_budget``,
  the same ratio ``BranchDriftPolicy``'s validated 65x-truncation
  ``headroom_ratio`` signal uses, just read from already-recorded fields
  instead of recomputed against router.py.
- ``estimated_recompute_cost_if_switched`` -- descriptive retrospective
  proxy from the actual backend flip. It must never enter a prospective
  quality predictor because the current placement is unknown at decision
  time.

Deliberately NOT built here (flagged, not silently dropped -- see the
project plan): a semantic embedding of the request, and an exact
files/snippets-in-context count. Both need either an extra model call or
deeper prompt parsing than is justified before knowing whether these
simpler, already-recorded-data-only features help at all.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .schema import AgentCallRecord, AgentTrajectory

_RECENT_WINDOW = 3
_REPAIR_LOOP_WINDOW = 4
_REPAIR_LOOP_MIN_REPEATS = 3
_SWITCH_COST_MULTIPLIER = 10.0  # same 10x sensitivity value phase1_router reports
_FAILURE_MARKERS = ("FAILED", "AssertionError", "Traceback (most recent call last)", "Error:")
PREDECISION_FEATURE_VERSION = "v4-live-request-extractor-parity-20260916"


def _tool_result_texts(request: dict) -> list[str]:
    """Text content of tool_result blocks in the latest causal user message
    -- same traversal as edgeproxy.trace.record.consumed_tool_result_ids,
    but pulling text instead of just IDs, since that's not recorded
    separately."""
    messages = request.get("messages") if isinstance(request, dict) else None
    for message in reversed(messages or []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        texts: list[str] = []
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            block_content = block.get("content")
            if isinstance(block_content, str):
                texts.append(block_content)
            elif isinstance(block_content, list):
                for sub in block_content:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        texts.append(str(sub.get("text") or ""))
        if texts:
            return texts
    return []


def _looks_like_test_failure(texts: list[str]) -> bool:
    return any(marker in text for text in texts for marker in _FAILURE_MARKERS)


def derive(trajectory: AgentTrajectory) -> AgentTrajectory:
    """Return a new AgentTrajectory with every call's ``.derived`` populated.
    Pure function: does not mutate the input (AgentCallRecord is frozen)."""
    calls = trajectory.calls
    new_calls: list[AgentCallRecord] = []

    for i, call in enumerate(calls):
        prev = calls[i - 1] if i > 0 else None
        previous_backend = prev.placement if prev else None
        # At decision time only the *previous* placements are known.  A
        # streak including this call's placement leaks the historical router's
        # choice into a prospective quality predictor.
        consecutive = 0
        j = i - 1
        while j >= 0 and previous_backend is not None and calls[j].placement == previous_backend:
            consecutive += 1
            j -= 1

        window = calls[max(0, i - _RECENT_WINDOW):i]
        recent_tool_error_count = sum(
            1 for c in window
            if float(c.router_features.get("errored_tool_result_density") or 0.0) > 0.0
        )
        recent_test_failure_count = sum(
            1 for c in window if _looks_like_test_failure(_tool_result_texts(c.request))
        )

        prior_truncated_or_invalid = False
        if prev is not None:
            prior_truncated_or_invalid = prev.stop_reason == "max_tokens" or any(
                b.get("schema_valid") is False for b in prev.tool_use_blocks
            )

        # Current tool_names come from the response and are unavailable before
        # placement.  Only previously produced tool calls may define a loop.
        loop_window = calls[max(0, i - _REPAIR_LOOP_WINDOW):i]
        # AgentCallRecord.tool_names is the OFFERED request tool suite, often
        # constant across turns.  Actual chosen tools are recorded in the
        # preceding response's validated tool_use_blocks instead.
        signatures = [tuple(str(b.get("tool_name")) for b in c.tool_use_blocks if b.get("tool_name"))
                      for c in loop_window]
        signatures = [sig for sig in signatures if sig]
        repair_loop_flag = bool(signatures) and signatures.count(signatures[-1]) >= _REPAIR_LOOP_MIN_REPEATS

        local_prompt_tokens = call.router_features.get("local_prompt_tokens")
        local_token_budget = call.router_features.get("local_token_budget")
        context_utilization_ratio = (
            local_prompt_tokens / local_token_budget
            if isinstance(local_prompt_tokens, (int, float))
            and isinstance(local_token_budget, (int, float))
            and local_token_budget
            else None
        )

        switched = previous_backend is not None and call.placement is not None and previous_backend != call.placement
        input_tokens = call.tokens.get("input_tokens")
        estimated_recompute_cost_if_switched = (
            _SWITCH_COST_MULTIPLIER * float(input_tokens) / 1000.0
            if switched and isinstance(input_tokens, (int, float))
            else 0.0
        )

        derived = {
            "predecision_feature_version": PREDECISION_FEATURE_VERSION,
            "previous_backend": previous_backend,
            "consecutive_same_backend_turns": consecutive,
            "recent_tool_error_count": recent_tool_error_count,
            "recent_test_failure_count": recent_test_failure_count,
            "prior_response_truncated_or_invalid": prior_truncated_or_invalid,
            "repair_loop_flag": repair_loop_flag,
            "context_utilization_ratio": context_utilization_ratio,
            "estimated_recompute_cost_if_switched": estimated_recompute_cost_if_switched,
        }
        new_calls.append(replace(call, derived=derived))

    return replace(trajectory, calls=new_calls)


def derive_all(trajectories: list[AgentTrajectory]) -> list[AgentTrajectory]:
    return [derive(t) for t in trajectories]


def regenerate_replay_calls(replay_rows: list[dict]) -> dict[str, AgentCallRecord]:
    """Recover sampled calls from complete original trajectories.

    A sampled replay file omits intervening calls.  Reading its serialized
    ``derived`` field or deriving over the sample would corrupt prior-backend
    and repair-loop features.  Fail closed if any original trace has gone.
    """
    from . import collect

    by_source: dict[str, dict[str, AgentCallRecord]] = {}
    out: dict[str, AgentCallRecord] = {}
    for row in replay_rows:
        raw_call = row["call"]
        path_str = raw_call["source_trace_path"]
        if path_str not in by_source:
            path = Path(path_str)
            if not path.is_file():
                raise FileNotFoundError(f"source chronology unavailable: {path}")
            trajectory_id = raw_call["trajectory_id"]
            condition = trajectory_id.rsplit(":", 2)[-2]
            seed = int(trajectory_id.rsplit(":seed", 1)[-1])
            traj = collect.build_trajectory(
                collect.load_trace_records(path),
                task_group=raw_call["task_group"],
                campaign=raw_call["source_campaign"],
                condition=condition,
                seed=seed,
                trace_path=path,
                task_passed=None,
                verdict_detail=None,
                verdict_path=Path("unused"),
            )
            by_source[path_str] = {call.call_id: call for call in derive(traj).calls}
        call_id = raw_call["call_id"]
        fresh = by_source[path_str].get(call_id)
        if fresh is None or fresh.request != raw_call["request"]:
            raise ValueError(f"replay call absent or request differs from original trace: {call_id}")
        if call_id in out:
            raise ValueError(f"duplicate replay call_id: {call_id}")
        out[call_id] = fresh
    return out
