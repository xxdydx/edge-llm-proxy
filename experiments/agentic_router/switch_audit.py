"""Descriptive switch/cache audit over existing mixed-policy traces only.

No requests are sent. Four observed switches are too few, and adjacent calls
change prompt/output length and backend together; the table is not a causal
estimate of cache loss or a universal switch penalty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRACE_NAMES = (
    "agentic-router-live-recal-run-routing-learned-agentic-heuristic-20260916T040014Z-swebench-ansible-39bd8b99-seed1",
    "agentic-router-live-recal-run-routing-learned-agentic-heuristic-20260916T040014Z-swebench-ansible-5e88cd99-seed1",
)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _view(record: dict[str, Any], turn: int) -> dict[str, Any]:
    call = record.get("call") or {}
    tokens = call.get("tokens") or {}
    timing = call.get("timing") or {}
    input_tokens = _number(tokens.get("input_tokens"))
    cached = _number(tokens.get("cache_read_tokens"))
    return {
        "turn": turn,
        "backend": record.get("placement"),
        "input_tokens": input_tokens,
        "output_tokens": _number(tokens.get("output_tokens")),
        "cache_read_tokens": cached,
        "cache_read_fraction": cached / input_tokens if cached is not None and input_tokens and input_tokens > 0 else None,
        "ttft_ms": _number(timing.get("ttft_ms")),
        "total_ms": _number(timing.get("total_ms")),
        "token_usage_integrity": tokens.get("usage_integrity"),
        "cache_details_available": tokens.get("cache_details_available"),
    }


def audit(paths: list[Path]) -> dict[str, Any]:
    traces = []
    all_switches = []
    all_calls = []
    for path in paths:
        records = []
        with path.open() as fh:
            for line in fh:
                record = json.loads(line)
                if record.get("placement") in ("local", "cloud") and record.get("call"):
                    records.append(record)
        records.sort(key=lambda r: (r.get("ts") or 0, r.get("id") or ""))
        calls = [_view(r, i + 1) for i, r in enumerate(records)]
        switches = []
        for i in range(1, len(calls)):
            before, after = calls[i - 1], calls[i]
            if before["backend"] == after["backend"]:
                continue
            entry = {
                "trace": path.parent.name,
                "from": before["backend"],
                "to": after["backend"],
                "before": before,
                "after": after,
                "input_token_change": (
                    after["input_tokens"] - before["input_tokens"]
                    if before["input_tokens"] is not None and after["input_tokens"] is not None else None
                ),
                "output_token_change": (
                    after["output_tokens"] - before["output_tokens"]
                    if before["output_tokens"] is not None and after["output_tokens"] is not None else None
                ),
            }
            switches.append(entry)
            all_switches.append(entry)
        all_calls.extend(calls)
        traces.append({"trace": path.parent.name, "n_calls": len(calls), "n_switches": len(switches), "switches": switches})
    fields = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_read_fraction", "ttft_ms", "total_ms")
    return {
        "status": "descriptive_not_causal",
        "n_traces": len(traces),
        "n_calls": len(all_calls),
        "n_switches": len(all_switches),
        "direction_counts": {
            direction: sum(s["from"] + "->" + s["to"] == direction for s in all_switches)
            for direction in ("cloud->local", "local->cloud")
        },
        "missingness_all_calls": {field: sum(c[field] is None for c in all_calls) for field in fields},
        "missingness_switch_after": {field: sum(s["after"][field] is None for s in all_switches) for field in fields},
        "traces": traces,
        "interpretation": "Adjacent turns change backend, prompt length, output length, phase, and cache state together. The local cache can remain resident after a cloud turn. No causal switch-cost estimate follows from these four events.",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = []
    for name in TRACE_NAMES:
        matches = list((root / "traces" / name).glob("*.jsonl"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one JSONL trace for {name}, found {len(matches)}")
        paths.append(matches[0])
    result = audit(paths)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
