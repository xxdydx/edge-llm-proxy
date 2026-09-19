"""Offline, predecision-only inventory of deterministic escalation signals.

Reads existing private traces; writes aggregate counts and opaque call IDs only.
No model, grader, or network calls. This is an opportunity inventory, not an
estimate that cloud escalation improves task success.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .switch_audit import TRACE_NAMES

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "eval-suite/swebench/results/agentic-router-live-matrix-v1/matrix_checkpoints"


def _trace_paths() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for checkpoint in sorted(MATRIX.glob("*.json")):
        row = json.loads(checkpoint.read_text())
        path = Path(row.get("trace_path") or "")
        if row.get("trace_available") and path.is_file():
            paths.append(("matrix", path))
    for name in TRACE_NAMES:
        matches = list((ROOT / "traces" / name).glob("*.jsonl"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one mixed trace for {name}: {len(matches)}")
        paths.append(("mixed", matches[0]))
    return paths


def _is_candidate(row: dict[str, Any]) -> bool:
    call = row.get("call") or {}
    causality = call.get("causality") or {}
    features = row.get("features") or {}
    return bool(
        row.get("path", "").rstrip("/") == "/v1/messages"
        and row.get("placement") in {"local", "cloud"}
        and isinstance(features, dict)
        and not features.get("is_security_monitor")
        and not causality.get("agent_id")
        and not causality.get("agent_parent_tool_use_id")
        and isinstance(row.get("ts"), (int, float))
    )


def _history_issue(row: dict[str, Any], ts: float) -> str | None:
    timing = row.get("timing") or {}
    duration = timing.get("total_ms")
    if row.get("status") != 200 or row.get("error") is not None or not isinstance(row.get("response"), dict):
        return "prior_failed_or_no_response"
    if not isinstance(duration, (int, float)):
        return "prior_completion_time_missing"
    if row["ts"] + duration / 1000 > ts:
        return "prior_still_in_flight"
    return None


def _names(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(block["tool_name"])
        for block in ((row.get("call") or {}).get("tool_use_blocks") or [])
        if isinstance(block, dict) and block.get("tool_name")
    )


def _result_ids(row: dict[str, Any]) -> tuple[set[str], set[str], int]:
    """All IDs, explicitly errored IDs, and unidentifiable tool results."""
    ids: set[str] = set()
    errored: set[str] = set()
    missing_id = 0
    for message in ((row.get("request") or {}).get("messages") or []):
        for block in (message.get("content") or []) if isinstance(message, dict) and isinstance(message.get("content"), list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_id = block.get("tool_use_id")
            if not tool_id:
                missing_id += 1
                continue
            ids.add(str(tool_id))
            if block.get("is_error") is True:
                errored.add(str(tool_id))
    return ids, errored, missing_id


def audit() -> dict[str, Any]:
    traces = _trace_paths()
    totals: dict[str, Counter[str]] = {"matrix": Counter(), "mixed": Counter()}
    evidence: list[dict[str, Any]] = []
    for source, path in traces:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [r for r in rows if _is_candidate(r)]
        rows.sort(key=lambda r: (r["ts"], r.get("id") or ""))
        counts = totals[source]
        counts["traces"] += 1
        for row in rows:
            counts["candidate_calls"] += 1
            ts = float(row["ts"])
            headers = row.get("headers") or {}
            session = headers.get("x-claude-code-session-id")
            if not session:
                counts["missing_session"] += 1
                continue
            lane = [r for r in rows if r["ts"] < ts and (r.get("headers") or {}).get("x-claude-code-session-id") == session]
            issues = {_history_issue(r, ts) for r in lane} - {None}
            if issues:
                counts["prior_in_flight_or_unfinished"] += 1
                for issue in issues:
                    counts[issue] += 1
                continue
            counts["history_known"] += 1
            prev = lane[-1] if lane else None
            prevprev = lane[-2] if len(lane) >= 2 else None
            features = row["features"]
            prompt = features.get("local_prompt_tokens")
            budget = features.get("local_token_budget")
            output = features.get("max_tokens")
            feasible = (
                isinstance(prompt, (int, float)) and isinstance(budget, (int, float))
                and isinstance(output, (int, float)) and prompt + output <= budget
                and features.get("local_reliability_blocked") is not True
            )
            if feasible:
                counts["simple_local_headroom_known"] += 1
            else:
                counts["simple_local_headroom_unavailable_or_failed"] += 1
            previous_invalid_local = bool(prev and prev.get("placement") == "local" and (
                (prev.get("response") or {}).get("stop_reason") == "max_tokens"
                or any(b.get("schema_valid") is False for b in ((prev.get("call") or {}).get("tool_use_blocks") or []))
            ))
            current_ids, current_errors, current_unidentifiable = _result_ids(row)
            previous_ids, previous_errors, previous_unidentifiable = _result_ids(prev) if prev else (set(), set(), 0)
            previous_previous_ids = _result_ids(prevprev)[0] if prevprev else set()
            if current_unidentifiable or previous_unidentifiable:
                counts["tool_results_missing_id"] += 1
            new_current_errors = current_errors - previous_ids
            new_previous_errors = previous_errors - previous_previous_ids
            if new_current_errors:
                counts["new_explicit_error_result"] += 1
            repeated_errors = bool(
                prev and new_current_errors and new_previous_errors
            )
            repeated_same_action = bool(
                repeated_errors and prevprev and _names(prev) and _names(prev) == _names(prevprev)
            )
            for key, flag in (
                ("previous_invalid_local", previous_invalid_local),
                ("two_consecutive_new_error_result_requests", repeated_errors),
                ("two_errors_same_prior_action_signature", repeated_same_action),
            ):
                if flag:
                    counts[key] += 1
                    if feasible:
                        counts[key + "_simple_headroom"] += 1
                    evidence.append({
                        "source": source, "trace": path.parent.name,
                        "call_id": row.get("id"), "signal": key,
                        "candidate_placement": row.get("placement"),
                        "simple_headroom": feasible,
                        "previous_call_id": prev.get("id") if prev else None,
                        "new_error_result_ids": sorted(new_current_errors) if repeated_errors else [],
                        "previous_new_error_result_ids": sorted(new_previous_errors) if repeated_errors else [],
                        "prior_action_tool_names": list(_names(prev)) if repeated_same_action else [],
                    })
    return {
        "status": "offline_opportunity_inventory_not_causal",
        "definitions": {
            "candidate": "main-agent /v1/messages placed call with predecision feature record",
            "history_known": "same-session earlier calls completed before candidate timestamp; no prior incomplete/in-flight call",
            "simple_headroom": "recorded local_prompt_tokens + max_tokens <= local_token_budget and breaker not blocked; not full router feasibility",
            "previous_invalid_local": "immediately prior completed same-session local response max_tokens or invalid tool-use schema",
            "two_consecutive_new_error_result_requests": "current and immediately previous requests each introduce a distinct explicitly is_error=true tool_result ID absent from the predecessor request",
            "two_errors_same_prior_action_signature": "above and last two completed responses produced identical nonempty tool-name signatures (names only, not command arguments)",
        },
        "counts": {key: dict(value) for key, value in totals.items()},
        "trigger_evidence": evidence,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
