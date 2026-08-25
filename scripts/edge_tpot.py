#!/usr/bin/env python3
"""Measure edge-model TTFT, TPOT, latency, and decode throughput.

This is the canonical edge-side entry point. It runs the controlled vLLM
benchmark implemented in ``measure_local_throughput`` and gives its output an
``edge-tpot-`` prefix so edge and cloud result files cannot be confused.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

if __package__:
    from scripts.measure_local_throughput import parse_args, run
else:
    from measure_local_throughput import parse_args, run  # type: ignore[no-redef]


def main() -> int:
    args = parse_args()
    if args.output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = Path("results") / f"edge-tpot-{timestamp}.csv"
    return asyncio.run(run(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
