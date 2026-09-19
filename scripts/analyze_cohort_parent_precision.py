#!/usr/bin/env python3
"""Reproduce the cohort-parent request-signal retrospective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edgeproxy.cohort import agent_delegation_count
from edgeproxy.cohort_parent import is_cohort_parent_candidate


def _trace_for_graph(graph_path: Path) -> Path:
    if graph_path.name == "trace.graph.json":
        return graph_path.with_name("trace.jsonl")
    return graph_path.with_name(graph_path.name.removesuffix(".graph.json") + ".jsonl")


def _corpus(root: Path) -> list[Path]:
    traces = []
    for graph_path in sorted((root / "traces").rglob("*.graph.json")):
        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not graph.get("cohorts"):
            continue
        trace_path = _trace_for_graph(graph_path)
        if trace_path.is_file():
            traces.append(trace_path)
    return traces


def _eligible(row: dict[str, Any]) -> bool:
    request = row.get("request")
    response = row.get("response")
    detection = row.get("cohort_detection") or {}
    return bool(
        isinstance(request, dict)
        and isinstance(response, dict)
        and detection.get("role") != "child"
        and any(
            isinstance(tool, dict) and tool.get("name") == "Agent"
            for tool in (request.get("tools") or [])
        )
    )


def _metrics(rows: list[dict[str, Any]], selected) -> dict[str, Any]:
    predicted = [row for row in rows if selected(row)]
    true_positives = sum(agent_delegation_count(row["response"]) > 0 for row in predicted)
    positives = sum(agent_delegation_count(row["response"]) > 0 for row in rows)
    return {
        "selected": len(predicted),
        "true_positives": true_positives,
        "false_positives": len(predicted) - true_positives,
        "precision": round(true_positives / len(predicted), 6) if predicted else None,
        "recall": round(true_positives / positives, 6) if positives else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    trace_paths = _corpus(args.repo_root)
    rows = []
    for trace_path in trace_paths:
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _eligible(row):
                rows.append(row)

    new_signal = lambda row: is_cohort_parent_candidate(
        row["request"], row.get("headers") or {}
    )
    local_rows = [row for row in rows if row.get("placement") == "local"]
    result = {
        "cohort_bearing_trace_recordings": len(trace_paths),
        "eligible_calls": len(rows),
        "actual_fan_out_calls": sum(
            agent_delegation_count(row["response"]) > 0 for row in rows
        ),
        "before": _metrics(rows, lambda _row: True),
        "after": _metrics(rows, new_signal),
        "local_override_population": _metrics(local_rows, new_signal),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
