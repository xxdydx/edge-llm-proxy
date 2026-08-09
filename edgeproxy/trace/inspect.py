"""Summarise recorded traces.

    python -m edgeproxy.trace.inspect traces/*.jsonl

This is the deliverable of the reconnaissance step: it answers "what does the
harness actually send?" — which endpoints, how big the prompts are, how much is
reused prefix, and how the traffic splits between cheap sidecalls and main-loop
tool-calling turns.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def load(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _system_text(request: Any) -> str:
    """Anthropic allows `system` as a plain string or a list of blocks."""
    if not isinstance(request, dict):
        return ""
    system = request.get("system")
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "".join(b.get("text", "") for b in system if isinstance(b, dict))
    return ""


def common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    shortest, longest = min(values), max(values)
    for i, ch in enumerate(shortest):
        if ch != longest[i]:
            return shortest[:i]
    return shortest


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def summarise(records: list[dict[str, Any]]) -> None:
    print(f"records            {len(records)}")
    if not records:
        return

    paths = Counter(r.get("path") for r in records)
    print("\npaths")
    for path, n in paths.most_common():
        print(f"  {n:6d}  {path}")

    messages = [r for r in records if r.get("path") == "/v1/messages"]
    if not messages:
        print("\nno /v1/messages records yet")
        return

    print("\nmodels")
    for model, n in Counter(
        (r.get("request") or {}).get("model") for r in messages
    ).most_common():
        print(f"  {n:6d}  {model}")

    statuses = Counter(r.get("status") for r in messages)
    if set(statuses) - {200}:
        print("\nstatuses")
        for status, n in statuses.most_common():
            print(f"  {n:6d}  {status}")

    # Sidecall vs main-loop: the split PLAN.md §1.2 routes on.
    with_tools = [r for r in messages if (r.get("request") or {}).get("tools")]
    print(
        f"\ncall classes       {len(with_tools)} with tools"
        f"  |  {len(messages) - len(with_tools)} without (sidecall candidates)"
    )

    inp = [r.get("usage", {}).get("input_tokens", 0) or 0 for r in messages]
    out = [r.get("usage", {}).get("output_tokens", 0) or 0 for r in messages]
    cached = [r.get("usage", {}).get("cache_read_input_tokens", 0) or 0 for r in messages]
    created = [r.get("usage", {}).get("cache_creation_input_tokens", 0) or 0 for r in messages]

    print("\ntokens")
    print(f"  input            {sum(inp):>10,}   median {statistics.median(inp):>8,.0f}")
    print(f"  output           {sum(out):>10,}   median {statistics.median(out):>8,.0f}")
    print(f"  cache_read       {sum(cached):>10,}")
    print(f"  cache_creation   {sum(created):>10,}")
    billable = sum(inp) + sum(cached)
    if billable:
        print(f"  prefix reuse     {sum(cached) / billable:>10.1%}  (cloud-side reference)")

    ttfts = [
        r["timing"]["ttft_ms"]
        for r in messages
        if isinstance(r.get("timing"), dict) and r["timing"].get("ttft_ms")
    ]
    totals = [
        r["timing"]["total_ms"]
        for r in messages
        if isinstance(r.get("timing"), dict) and r["timing"].get("total_ms")
    ]
    if ttfts:
        print("\nttft_ms          " f"p50 {_pct(ttfts, .5):>8.0f}  p90 {_pct(ttfts, .9):>8.0f}")
    if totals:
        print(f"total_ms         p50 {_pct(totals, .5):>8.0f}  p90 {_pct(totals, .9):>8.0f}")

    systems = [s for s in (_system_text(r.get("request")) for r in messages) if s]
    if systems:
        shared = common_prefix(systems)
        print(
            f"\nsystem prompt      {len(shared):,} chars shared across "
            f"{len(systems)} calls (~{len(shared) // 4:,} tokens)"
        )

    tools = Counter()
    for record in messages:
        for block in (record.get("response") or {}).get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools[block.get("name")] += 1
    if tools:
        print("\ntools called")
        for name, n in tools.most_common(15):
            print(f"  {n:6d}  {name}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="edgeproxy.trace.inspect")
    ap.add_argument("paths", nargs="+", type=Path, help="trace JSONL files")
    args = ap.parse_args()

    existing = [p for p in args.paths if p.exists()]
    if not existing:
        raise SystemExit("no trace files found")
    summarise(load(existing))


if __name__ == "__main__":
    main()
