"""Read-only accounting availability audit for the live matrix.

Some cloud adapters report ``usage.input_tokens=0`` despite a nonempty
request. That is not evidence of zero input. Preserve raw values separately
from an exact-provider-usage availability count and the local-render probe
estimate. No prompts or response text are written to the summary.
"""

from __future__ import annotations

import argparse
import json

from . import collect, live_matrix


def audit_cell(slug: str, condition: str) -> dict:
    path = collect.locate_trace_file(live_matrix.EXPERIMENT, slug, condition,
                                     live_matrix.SEED)
    if path is None:
        return {"slug": slug, "condition": condition, "trace_available": False}
    calls = [r for r in collect.load_trace_records(path)
             if r.get("path") == "/v1/messages" and r.get("placement") in ("cloud", "local")]
    by_backend = {}
    for backend in ("cloud", "local"):
        rows = [r for r in calls if r["placement"] == backend]
        exact_input = []
        zero_placeholder = 0
        missing_input = 0
        proxy_estimate = []
        cache_known = 0
        for row in rows:
            accounting = row.get("token_accounting") or {}
            raw = accounting.get("provider_input_tokens", accounting.get("input_tokens"))
            observed = accounting.get("proxy_observed_input_tokens")
            if isinstance(observed, (int, float)) and not isinstance(observed, bool):
                proxy_estimate.append(observed)
            if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
                missing_input += 1
            elif raw == 0 and isinstance(observed, (int, float)) and observed > 0:
                zero_placeholder += 1
            else:
                exact_input.append(raw)
            if accounting.get("cache_details_available") is True:
                cache_known += 1
        by_backend[backend] = {
            "calls": len(rows),
            "provider_input_exact_available_calls": len(exact_input),
            "provider_input_missing_calls": missing_input,
            "provider_input_zero_with_nonempty_proxy_prompt_calls": zero_placeholder,
            "provider_input_exact_sum_only_available_calls": sum(exact_input),
            "proxy_local_render_estimate_available_calls": len(proxy_estimate),
            "proxy_local_render_estimate_sum": sum(proxy_estimate),
            "cache_details_available_calls": cache_known,
        }
    return {"slug": slug, "condition": condition, "trace_available": True,
            "calls": len(calls), "by_backend": by_backend,
            "interpretation": "Provider zero with nonempty proxy-observed prompt is unknown, not measured zero; proxy local-render count is a cross-backend estimate, not provider billing usage."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, help="atomic prompt-free JSON report path")
    args = parser.parse_args()
    rows = [audit_cell(slug, condition) for slug in live_matrix.TASKS
            for condition in live_matrix.CONDITIONS]
    result = {"schema_version": "matrix-accounting-availability-v1", "cells": rows}
    if args.out:
        from pathlib import Path
        live_matrix._atomic(Path(args.out), result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
