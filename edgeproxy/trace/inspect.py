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

from .record import build_token_accounting


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


def usage_input_breakdown(usage: Any) -> dict[str, int | bool]:
    """Normalise old and cache-detailed Anthropic usage schemas.

    With prompt-token details enabled, ``input_tokens`` is the uncached
    remainder and total input also includes cache reads and cache creation.
    Historical records without either cache field use ``input_tokens`` as the
    only recoverable total and must not be counted as observed cache misses.
    """
    accounting = build_token_accounting(usage)
    detailed = bool(accounting["cache_details_available"])
    uncached = int(accounting["uncached_input_tokens"] or 0)
    cached = int(accounting["cache_read_input_tokens"] or 0)
    created = int(accounting["cache_creation_input_tokens"] or 0)
    total = int(accounting["input_tokens"] or 0)
    return {
        "detailed": detailed,
        "uncached": uncached,
        "cached": cached,
        "created": created,
        "total": total,
    }


def cache_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Aggregate only records whose provider reported cache detail fields."""
    by_placement: dict[str, dict[str, int]] = {}
    for record in records:
        usage = usage_input_breakdown(record.get("usage"))
        if not usage["detailed"]:
            continue
        placement = str(record.get("placement") or "unknown")
        stats = by_placement.setdefault(
            placement,
            {"requests": 0, "request_hits": 0, "cached": 0, "total": 0},
        )
        stats["requests"] += 1
        stats["request_hits"] += int(usage["cached"] > 0)
        stats["cached"] += int(usage["cached"])
        stats["total"] += int(usage["total"])
    return by_placement


def cache_prediction_stats(
    records: list[dict[str, Any]], backend: str
) -> dict[str, int | float]:
    """Summarise pre-decision estimates against selected-backend usage."""
    key = f"{backend}_cache"
    entries = [r.get(key) for r in records if isinstance(r.get(key), dict)]
    predictions = [
        item
        for item in entries
        if (item.get("prediction") or {}).get("estimated_read_tokens") is not None
    ]
    scored = [item for item in entries if (item.get("actual") or {}).get("available")]
    comparable = [
        item
        for item in scored
        if (item.get("agreement") or {}).get("within_5_percent_of_input") is not None
    ]
    within = sum(
        bool((item.get("agreement") or {}).get("within_5_percent_of_input"))
        for item in comparable
    )
    errors = [
        float((item.get("agreement") or {})["cached_token_error_fraction_of_input"])
        for item in comparable
        if (item.get("agreement") or {}).get("cached_token_error_fraction_of_input")
        is not None
    ]
    return {
        "tracked": len(entries),
        "predicted": len(predictions),
        "actual": len(scored),
        "comparable": len(comparable),
        "within_5pct": within,
        "median_error_fraction": statistics.median(errors) if errors else 0.0,
    }


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

    input_usage = [usage_input_breakdown(r.get("usage")) for r in messages]
    inp = [int(usage["total"]) for usage in input_usage]
    out = [r.get("usage", {}).get("output_tokens", 0) or 0 for r in messages]
    cached = [int(usage["cached"]) for usage in input_usage]
    created = [int(usage["created"]) for usage in input_usage]

    print("\ntokens")
    print(f"  input_total      {sum(inp):>10,}   median {statistics.median(inp):>8,.0f}")
    print(f"  output           {sum(out):>10,}   median {statistics.median(out):>8,.0f}")

    stats_by_placement = cache_stats(messages)
    if stats_by_placement:
        print(f"  cache_read       {sum(cached):>10,}")
        print(f"  cache_creation   {sum(created):>10,}")
        print("\nreported prefix-cache reuse")
        for placement in sorted(stats_by_placement):
            stats = stats_by_placement[placement]
            request_rate = stats["request_hits"] / stats["requests"]
            token_rate = stats["cached"] / stats["total"] if stats["total"] else 0.0
            print(
                f"  {placement:<8} requests {stats['request_hits']:>5,}/{stats['requests']:<5,}"
                f" ({request_rate:>6.1%})  tokens {stats['cached']:>10,}/{stats['total']:<10,}"
                f" ({token_rate:>6.1%})"
            )
        detailed_count = sum(stats["requests"] for stats in stats_by_placement.values())
        if detailed_count != len(messages):
            print(
                f"  note     {len(messages) - detailed_count:,} older/unsupported records "
                "excluded (no cache-detail fields)"
            )
    else:
        print("  cache detail     unavailable (no cache-detail fields)")

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

    tpots = [
        r["timing"]["tpot_ms"]
        for r in messages
        if isinstance(r.get("timing"), dict) and r["timing"].get("tpot_ms") is not None
    ]
    output_rates = [
        r["timing"]["output_tokens_per_s"]
        for r in messages
        if isinstance(r.get("timing"), dict)
        and r["timing"].get("output_tokens_per_s") is not None
    ]
    if tpots:
        print(f"tpot_ms          p50 {_pct(tpots, .5):>8.2f}  p90 {_pct(tpots, .9):>8.2f}")
    if output_rates:
        print(
            "output_tok_s      "
            f"p50 {_pct(output_rates, .5):>8.1f}  p90 {_pct(output_rates, .9):>8.1f}"
        )

    priced = [
        r["cost_savings"]
        for r in messages
        if isinstance(r.get("cost_savings"), dict)
        and r["cost_savings"].get("available")
    ]
    if priced:
        cloud_list_cost = sum(float(row["cloud_cost_usd"]) for row in priced)
        saved = sum(float(row["request_saved_usd"]) for row in priced)
        local_priced = sum(float(row["request_saved_usd"]) > 0 for row in priced)
        print("\nAnthropic list-price accounting")
        print(
            f"  priced requests  {len(priced):>8,}  local savings on {local_priced:,}"
        )
        print(f"  represented cost ${cloud_list_cost:>11.6f}")
        print(f"  routing saved    ${saved:>11.6f}")

    prediction_rows = {
        backend: cache_prediction_stats(messages, backend)
        for backend in ("local", "cloud")
    }
    if any(row["tracked"] for row in prediction_rows.values()):
        print("\ncache prediction versus selected-backend actual")
        for backend, row in prediction_rows.items():
            if not row["tracked"]:
                continue
            print(
                f"  {backend:<8} predicted {row['predicted']:>5,}/{row['tracked']:<5,}"
                f"  actual {row['actual']:>5,}"
                f"  within-5% {row['within_5pct']:>5,}/{row['comparable']:<5,}"
                f"  median error/input {row['median_error_fraction']:>6.2%}"
            )

    cache_predictions = [
        r["cloud_cache"]
        for r in messages
        if isinstance(r.get("cloud_cache"), dict)
    ]
    scored_predictions = [
        item
        for item in cache_predictions
        if (item.get("actual") or {}).get("cache_read_input_tokens") is not None
    ]
    if cache_predictions:
        states = Counter(
            (item.get("prediction") or {}).get("state", "unknown")
            for item in cache_predictions
        )
        state_text = ", ".join(f"{key}={value}" for key, value in sorted(states.items()))
        print(f"\ncloud cache shadow {state_text}")
    if scored_predictions:
        predicted_warm = [
            item
            for item in scored_predictions
            if (item.get("prediction") or {}).get("state") == "warm"
        ]
        true_positive = sum(
            int((item.get("actual") or {}).get("cache_read_input_tokens") or 0) > 0
            for item in predicted_warm
        )
        actual_warm = sum(
            int((item.get("actual") or {}).get("cache_read_input_tokens") or 0) > 0
            for item in scored_predictions
        )
        precision = true_positive / len(predicted_warm) if predicted_warm else None
        recall = true_positive / actual_warm if actual_warm else None
        precision_text = f"{precision:.1%}" if precision is not None else "n/a"
        recall_text = f"{recall:.1%}" if recall is not None else "n/a"
        print(
            f"  provider-scored {len(scored_predictions):,}  "
            f"warm precision {precision_text}  recall {recall_text}"
        )

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
