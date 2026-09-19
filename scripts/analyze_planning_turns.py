#!/usr/bin/env python3
"""Characterize early tool-loop turns in recorded SWE-bench Pro traces."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable


def content_blocks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [block for block in value if isinstance(block, dict)]


def turn_ordinal(messages: list[Any]) -> int:
    return 1 + sum(
        1
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and any(
            block.get("type") == "tool_result"
            for block in content_blocks(message.get("content"))
        )
    )


def block_chars(content: Any, block_type: str) -> int:
    total = 0
    for block in content_blocks(content):
        if block.get("type") != block_type:
            continue
        for key in ("text", "thinking", "partial_json"):
            value = block.get(key)
            if isinstance(value, str):
                total += len(value)
    return total


def median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return round(statistics.median(values), 1) if values else None


def pct(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 1) if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        default="traces/swebench-pro-15-run-*/*.jsonl",
        help="trace-file glob relative to the current directory",
    )
    args = parser.parse_args()

    files = sorted(Path(path) for path in glob.glob(args.glob))
    rows: list[dict[str, Any]] = []
    files_with_calls: set[Path] = set()
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                features = record.get("features") or {}
                request = record.get("request") or {}
                response = record.get("response") or {}
                if record.get("path") != "/v1/messages" or not features:
                    continue
                if features.get("is_security_monitor") or not features.get("has_tools"):
                    continue

                content = response.get("content") or []
                blocks = content_blocks(content)
                tools = [
                    block.get("name")
                    for block in blocks
                    if block.get("type") == "tool_use"
                ]
                rows.append(
                    {
                        "ordinal": turn_ordinal(request.get("messages") or []),
                        "tools": tools,
                        "tool_use": bool(tools),
                        "thinking": any(block.get("type") == "thinking" for block in blocks),
                        "text_chars": block_chars(content, "text"),
                        "thinking_chars": block_chars(content, "thinking"),
                        "output_tokens": (record.get("usage") or {}).get("output_tokens"),
                    }
                )
                files_with_calls.add(path)

    print(
        f"files={len(files)} files_with_agent_calls={len(files_with_calls)} "
        f"agent_calls={len(rows)}"
    )
    print(
        "bucket\tn\ttool_use%\tthinking%\tno_tool%\tmedian_text_chars"
        "\tmedian_thinking_chars\tmedian_output_tokens"
    )
    buckets: list[tuple[str, Callable[[int], bool]]] = [
        ("1", lambda value: value == 1),
        ("2", lambda value: value == 2),
        ("3", lambda value: value == 3),
        ("4", lambda value: value == 4),
        ("5", lambda value: value == 5),
        ("6-10", lambda value: 6 <= value <= 10),
        ("11+", lambda value: value >= 11),
    ]
    for label, predicate in buckets:
        selected = [row for row in rows if predicate(row["ordinal"])]
        print(
            label,
            len(selected),
            pct(sum(row["tool_use"] for row in selected), len(selected)),
            pct(sum(row["thinking"] for row in selected), len(selected)),
            pct(sum(not row["tool_use"] for row in selected), len(selected)),
            median(selected, "text_chars"),
            median(selected, "thinking_chars"),
            median(selected, "output_tokens"),
            sep="\t",
        )

    print("\nbucket\tn_tools\texplore%\tmutate%\tbash%")
    grouped = [
        ("1", lambda value: value == 1),
        ("2", lambda value: value == 2),
        ("3", lambda value: value == 3),
        ("4-5", lambda value: 4 <= value <= 5),
        ("6-10", lambda value: 6 <= value <= 10),
        ("11+", lambda value: value >= 11),
    ]
    for label, predicate in grouped:
        tools = [
            tool
            for row in rows
            if predicate(row["ordinal"])
            for tool in row["tools"]
        ]
        counts = Counter(tools)
        explore = sum(counts[name] for name in ("Read", "Glob", "Grep", "LS"))
        mutate = sum(counts[name] for name in ("Edit", "Write", "NotebookEdit"))
        print(
            label,
            len(tools),
            pct(explore, len(tools)),
            pct(mutate, len(tools)),
            pct(counts["Bash"], len(tools)),
            sep="\t",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
