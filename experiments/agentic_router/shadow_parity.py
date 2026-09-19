"""Read-only parity audit: live-style shadow history vs offline derivation.

Uses full original trace chronology from the frozen artifact's provenance,
not sampled replay rows. Reports aggregate mismatches only; no prompt or
response text leaves the ignored trace files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from edgeproxy.agentic_history import AgenticHistory
from edgeproxy import router

from . import analysis, collect, features

ARTIFACT = Path(__file__).resolve().parent / "results/quality_model_baseline205_v5_shadow_logreg.json"
HISTORY_FEATURES = (
    "turn_index", "recent_tool_error_count", "recent_test_failure_count",
    "consecutive_same_backend_turns", "prior_response_truncated_or_invalid",
    "repair_loop_flag",
)
CURRENT_FEATURES = (
    "local_prompt_tokens", "n_available_tools", "errored_tool_result_density",
    "branch_turn_ordinal", "context_utilization_ratio",
)


def audit(artifact_path: Path = ARTIFACT) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text())
    paths = [Path(value) for value in artifact["training_provenance"]["source_trace_sha256"]]
    missing_paths = [str(p) for p in paths if not p.is_file()]
    if missing_paths:
        return {"status": "missing_source_traces", "n_missing_paths": len(missing_paths)}
    mismatch: Counter[str] = Counter()
    unavailable: Counter[str] = Counter()
    missing_current: Counter[str] = Counter()
    missing_recorded: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    total = scored = 0
    path_counts = {}
    for path in paths:
        raw = collect.load_trace_records(path)
        first = raw[0]
        trajectory = features.derive(collect.build_trajectory(
            raw,
            task_group="parity-audit",
            campaign=str(first.get("experiment_id") or "unknown"),
            condition="historical",
            seed=1,
            trace_path=path,
            task_passed=None,
            verdict_detail=None,
            verdict_path=Path("unused"),
        ))
        history = AgenticHistory(max_calls_per_lane=4096)
        key = "one-historical-trajectory"
        path_counts[path.parent.name] = len(trajectory.calls)
        for call in trajectory.calls:
            total += 1
            sequence = history.begin(key)
            live, reason = history.snapshot(key, sequence)
            offline = analysis.call_features(call)
            current = router.extract_features(call.request)
            for field in ("branch_turn_ordinal", "errored_tool_result_density"):
                if call.router_features.get(field) is None:
                    missing_recorded[field] += 1
            raw_prompt = call.router_features.get("local_prompt_tokens")
            raw_budget = call.router_features.get("local_token_budget")
            current_values = {
                "local_prompt_tokens": float(raw_prompt) if isinstance(raw_prompt, (int, float)) else None,
                "n_available_tools": float(current.n_tools) if isinstance(call.request.get("tools"), list) else None,
                "errored_tool_result_density": float(current.errored_tool_result_density),
                "branch_turn_ordinal": float(current.branch_turn_ordinal or 0),
                "context_utilization_ratio": (
                    float(raw_prompt / raw_budget)
                    if isinstance(raw_prompt, (int, float)) and isinstance(raw_budget, (int, float)) and raw_budget else None
                ),
            }
            for field in CURRENT_FEATURES:
                value = current_values[field]
                if value is None:
                    missing_current[field] += 1
                elif value != offline[field]:
                    mismatch[field] += 1
                    examples.setdefault(field, [])
                    if len(examples[field]) < 3:
                        examples[field].append({
                            "trace": path.parent.name, "turn": call.turn_index,
                            "live": value, "offline": offline[field],
                        })
            if live is None:
                unavailable[reason or "unknown"] += 1
            else:
                scored += 1
                for field in HISTORY_FEATURES:
                    if live[field] != offline[field]:
                        mismatch[field] += 1
                        examples.setdefault(field, [])
                        if len(examples[field]) < 3:
                            examples[field].append({
                                "trace": path.parent.name,
                                "turn": call.turn_index,
                                "live": live[field], "offline": offline[field],
                            })
            history.complete(
                key, sequence,
                request=call.request,
                errored_tool_result_density=float(call.router_features.get("errored_tool_result_density") or 0.0),
                placement=str(call.placement or ""),
                response=call.response,
                tool_use_blocks=call.tool_use_blocks,
            )
    return {
        "status": "parity_checked",
        "n_traces": len(paths), "n_calls": total,
        "n_history_available": scored,
        "n_history_unavailable": total - scored,
        "unavailable_reasons": dict(unavailable),
        "missing_current_features": dict(missing_current),
        "missing_recorded_features": dict(missing_recorded),
        "feature_mismatches": dict(mismatch),
        "example_mismatches": examples,
        "path_call_counts": path_counts,
        "caveat": "Sequential trace replay validates derivation parity when prior calls are complete. Live concurrent calls can be unavailable by design; actual local probe coverage and session headers require separate live measurement.",
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
