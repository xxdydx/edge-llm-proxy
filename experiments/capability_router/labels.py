"""Conservative labeling: does a replayed response do the same *action* the
teacher (recorded cloud) response did, in a way that can be checked without
knowing the actual repository state at that turn?

Only three things are ever checkable without the missing per-turn repo
snapshot:
  - which tool was called (a clean structural fact),
  - whether no tool was called at all when the teacher called one (also
    clean and structural), and
  - whether a tool_use block validates against its own requested schema
    (a deterministic, state-independent proof of failure).

A *different but still schema-valid* argument set for the same tool cannot
be judged right or wrong without the missing repo snapshot -- a different
`old_string` in an `Edit` might be an equally valid fix. That case is
`UNKNOWN` by default, not `NOT_EQUIVALENT`, unless a downstream
deterministic signal (the replay's own tool_use failing schema validation)
proves the divergence really is a failure. Free text is always `UNKNOWN`.
See README.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from edgeproxy.trace.record import validate_tool_use_blocks

from .schema import Label, ReplayOutcome

NO_TOOL_ACTION = {"kind": "text"}


@dataclass(frozen=True)
class RawComponents:
    """Every raw comparison input, preserved so quality_labels.jsonl can be
    audited or relabeled later without replaying the calls again."""

    teacher_kind: str
    teacher_tool_name: str | None
    teacher_tool_input: Any
    replay_kind: str
    replay_tool_name: str | None
    replay_tool_input: Any
    replay_schema_valid: bool | None
    exact_input_match: bool | None


def extract_action(response: dict[str, Any] | None) -> dict[str, Any]:
    """The first tool_use block in a response, or the text-only sentinel.

    First, not "the most interesting", block: teacher forcing compares the
    model's immediate next action, and a model that would have called a
    different tool first has already diverged regardless of what it might
    have called second.
    """
    if not isinstance(response, dict):
        return dict(NO_TOOL_ACTION)
    for block in response.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return {
                "kind": "tool_use",
                "name": str(block.get("name") or ""),
                "input": block.get("input"),
            }
    return dict(NO_TOOL_ACTION)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def action_equivalence(
    request: dict[str, Any],
    teacher_response: dict[str, Any],
    replay_outcome: ReplayOutcome,
) -> tuple[Label, str, RawComponents]:
    teacher_action = extract_action(teacher_response)

    if replay_outcome.status != "OK":
        components = RawComponents(
            teacher_kind=teacher_action["kind"],
            teacher_tool_name=teacher_action.get("name"),
            teacher_tool_input=teacher_action.get("input"),
            replay_kind="error",
            replay_tool_name=None,
            replay_tool_input=None,
            replay_schema_valid=None,
            exact_input_match=None,
        )
        # INVALID_CAPACITY is a scope exclusion (this backend's real context
        # window is smaller than the request), never conflated with an
        # unplanned execution failure -- every other non-OK status collapses
        # to EXECUTION_ERROR.
        if replay_outcome.status == "INVALID_CAPACITY":
            return "INVALID_CAPACITY", replay_outcome.detail or "exceeds backend capacity", components
        return "EXECUTION_ERROR", f"replay did not complete: {replay_outcome.detail}", components

    replay_action = extract_action(replay_outcome.response)
    replay_valid = schema_validity(request, replay_outcome)
    components = RawComponents(
        teacher_kind=teacher_action["kind"],
        teacher_tool_name=teacher_action.get("name"),
        teacher_tool_input=teacher_action.get("input"),
        replay_kind=replay_action["kind"],
        replay_tool_name=replay_action.get("name"),
        replay_tool_input=replay_action.get("input"),
        replay_schema_valid=replay_valid,
        exact_input_match=None,
    )

    if teacher_action["kind"] == "text":
        return "UNKNOWN", "teacher action is free text; not reliably comparable", components

    if replay_action["kind"] == "text":
        return (
            "NOT_EQUIVALENT",
            f"replay produced no tool call; teacher called {teacher_action['name']!r}",
            components,
        )

    if replay_action["name"] != teacher_action["name"]:
        return (
            "NOT_EQUIVALENT",
            f"tool differs: replay={replay_action['name']!r} teacher={teacher_action['name']!r}",
            components,
        )

    exact_match = _canonical(replay_action["input"]) == _canonical(teacher_action["input"])
    components = RawComponents(
        **{**components.__dict__, "exact_input_match": exact_match}
    )
    if exact_match:
        return "EQUIVALENT", f"{teacher_action['name']} input matches exactly", components

    # Same tool, different (but not necessarily wrong) input: this is the
    # state-dependent case that cannot be judged without the missing
    # per-turn repo snapshot -- UNKNOWN by default. The one deterministic,
    # state-independent way a "different input" can still be proven a real
    # failure is the replay's own tool_use failing its requested schema.
    if replay_valid is False:
        return (
            "NOT_EQUIVALENT",
            f"same tool ({teacher_action['name']}) with different input, AND replay's "
            "own tool_use failed schema validation -- deterministic failure proof",
            components,
        )
    return (
        "UNKNOWN",
        f"same tool ({teacher_action['name']}) but different input; equivalence "
        "depends on repository state not recorded in this trace",
        components,
    )


def schema_validity(request: dict[str, Any], replay_outcome: ReplayOutcome) -> bool | None:
    """Validity of the first tool call, matching :func:`extract_action`.

    Later tool calls in the same assistant response are deliberately outside
    this experiment's first-next-action estimand.  Validating every tool call
    here while comparing only the first would let a later invalid call change
    the label assigned to an otherwise valid first action.

    None means there is nothing to validate (execution error, or the reply
    made no tool calls at all -- vacuously not a schema failure, but not
    evidence of validity either, so left unscored rather than True).
    """
    if replay_outcome.status != "OK" or not isinstance(replay_outcome.response, dict):
        return None
    results = validate_tool_use_blocks(request, replay_outcome.response)
    if not results:
        return None
    return results[0]["schema_valid"]
