#!/usr/bin/env python3
"""Per-policy metrics table for a matched-matrix SWE-bench Pro campaign.

Joins the campaign's `summary.json` (pass/fail, fail_to_pass / pass_to_pass,
wall time, turns) with the edgeproxy trace records for the same
`experiment_id` (placement, schema validity, per-backend tokens, TTFT,
TPOT/throughput, prefix-cache reuse).

Usage:
    python3 scripts/matched_matrix_metrics.py \
        --experiment-id spec-matched-7b-0908 \
        [--results-dir eval-suite/swebench/results/spec-matched-7b-0908] \
        [--traces-root traces] [--format md|json]

One row per (instance, condition). n=1/cell by design here, so treat these as
data points, not rates.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 1) if xs else None


def _sum(xs):
    """Sum the real numbers; None if there were none (distinguish 0 from n/a).

    The DeepSeek 'cloud' gateway returns input_tokens: 0 and null cache
    fields, so a cloud cell has genuinely-unknown input/cache token counts —
    that must render as blank, not as a hard zero.
    """
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) if xs else None


def load_summary(results_dir: Path) -> dict:
    sj = results_dir / "summary.json"
    if not sj.is_file():
        raise SystemExit(f"no summary.json under {results_dir}")
    return json.loads(sj.read_text())


def collect_trace_records(traces_root: Path, experiment_id: str):
    """All trace records whose experiment_id matches, keyed by (slug, condition).

    The episode_id is `<experiment_id>-<slug>__<condition>__seed<n>`; parse the
    slug/condition back out of it so we don't depend on trace-dir naming.
    """
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for jsonl in traces_root.rglob("*.jsonl"):
        try:
            lines = jsonl.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or experiment_id not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("experiment_id") != experiment_id:
                continue
            ep = rec.get("episode_id") or ""
            tail = ep[len(experiment_id) + 1 :] if ep.startswith(experiment_id + "-") else ep
            # tail == "<slug>__<condition>__seed<n>"
            if "__" not in tail:
                continue
            slug, _, rest = tail.partition("__")
            condition = rest.rsplit("__seed", 1)[0]
            by_cell[(slug, condition)].append(rec)
    return by_cell


def cell_metrics(records: list[dict]) -> dict:
    n = len(records)
    local = [r for r in records if r.get("placement") == "local"]
    cloud = [r for r in records if r.get("placement") == "cloud"]

    def toks(rs, field):
        out = []
        for r in rs:
            ta = r.get("token_accounting") or {}
            v = ta.get(field)
            if v is None:
                v = (r.get("call") or {}).get("tokens", {}).get(field)
            out.append(v)
        return _sum(out)

    def reportable_input_tokens(rs):
        values = []
        sources = set()
        for r in rs:
            ta = r.get("token_accounting") or {}
            value = ta.get("reportable_input_tokens")
            source = ta.get("reportable_input_tokens_source")
            if value is None:
                value = ta.get("input_tokens")
                source = "provider_usage" if value is not None and value > 0 else None
            values.append(value)
            if source:
                sources.add(source)
        return _sum(values), "+".join(sorted(sources)) or None

    # schema validity across every tool_use block in the cell
    blocks = [
        b
        for r in records
        for b in ((r.get("call") or {}).get("tool_use_blocks") or [])
    ]
    schema_valid = sum(1 for b in blocks if b.get("schema_valid") is True)

    # local prefix-cache reuse: cache_read / total input, local calls only
    local_in, local_input_source = reportable_input_tokens(local)
    local_cache_read = toks(local, "cache_read_input_tokens")
    cloud_in, cloud_input_source = reportable_input_tokens(cloud)

    def timing(rs, field):
        return [
            ((r.get("call") or {}).get("timing") or {}).get(field) for r in rs
        ]

    probe = [
        (r.get("call") or {}).get("cache_probe") or {} for r in local
    ]
    probe_agree = [
        p.get("agreement", {}).get("warm_prediction_correct")
        for p in probe
        if p
    ]

    return {
        "n_calls": n,
        "n_local": len(local),
        "n_cloud": len(cloud),
        "local_serve_rate": round(len(local) / n, 3) if n else None,
        "tool_blocks": len(blocks),
        "schema_valid": schema_valid,
        "schema_valid_rate": round(schema_valid / len(blocks), 3) if blocks else None,
        "local_input_tokens": local_in,
        "local_input_tokens_source": local_input_source,
        "local_output_tokens": toks(local, "output_tokens"),
        "local_cache_read_tokens": local_cache_read,
        "local_cache_creation_tokens": toks(local, "cache_creation_input_tokens"),
        "local_prefix_reuse": round(local_cache_read / local_in, 3) if local_in else None,
        "cloud_input_tokens": cloud_in,
        "cloud_input_tokens_source": cloud_input_source,
        "cloud_output_tokens": toks(cloud, "output_tokens"),
        "cloud_cache_read_tokens": toks(cloud, "cache_read_input_tokens"),
        "local_ttft_ms_median": _median(timing(local, "ttft_ms")),
        "local_tpot_ms_median": _median(timing(local, "tpot_ms")),
        "local_decode_tok_s_median": _median(timing(local, "decode_tokens_per_s")),
        "cloud_ttft_ms_median": _median(timing(cloud, "ttft_ms")),
        "cloud_tpot_ms_median": _median(timing(cloud, "tpot_ms")),
        "cloud_decode_tok_s_median": _median(timing(cloud, "decode_tokens_per_s")),
        "probe_agree_rate": (
            round(sum(1 for x in probe_agree if x) / len(probe_agree), 3)
            if probe_agree
            else None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--traces-root", type=Path, default=REPO / "traces")
    ap.add_argument("--format", choices=["md", "json"], default="md")
    args = ap.parse_args()

    results_dir = args.results_dir or (
        REPO / "eval-suite/swebench/results" / args.experiment_id
    )
    summary = load_summary(results_dir)
    jobs = {(j["instance_slug"], j["condition"]): j for j in summary["jobs"]}
    traces = collect_trace_records(args.traces_root, args.experiment_id)

    rows = []
    for (slug, cond), job in sorted(jobs.items()):
        m = cell_metrics(traces.get((slug, cond), []))
        rows.append(
            {
                "instance": slug.replace("swebench-", ""),
                "condition": cond,
                "pass": job["passed"],
                "f2p": f'{job["n_fail_to_pass_passed"]}/{job["n_fail_to_pass"]}',
                "p2p": f'{job["n_pass_to_pass_passed"]}/{job["n_pass_to_pass"]}',
                "wall_s": round(job["wall_time_s"], 1),
                "turns": job.get("claude_num_turns"),
                **m,
            }
        )

    if args.format == "json":
        print(json.dumps({"experiment_id": args.experiment_id, "rows": rows}, indent=2))
        return 0

    cols = [
        ("instance", "instance"),
        ("condition", "condition"),
        ("pass", "pass"),
        ("f2p", "f2p"),
        ("p2p", "p2p"),
        ("wall_s", "wall_s"),
        ("turns", "turns"),
        ("n_calls", "calls"),
        ("local_serve_rate", "local_rate"),
        ("schema_valid_rate", "schema_ok"),
        ("local_input_tokens", "loc_in_tok"),
        ("local_input_tokens_source", "loc_in_src"),
        ("local_output_tokens", "loc_out_tok"),
        ("cloud_input_tokens", "cld_in_tok"),
        ("cloud_input_tokens_source", "cld_in_src"),
        ("cloud_output_tokens", "cld_out_tok"),
        ("local_prefix_reuse", "loc_reuse"),
        ("local_ttft_ms_median", "loc_ttft"),
        ("local_tpot_ms_median", "loc_tpot"),
        ("local_decode_tok_s_median", "loc_tok/s"),
        ("cloud_ttft_ms_median", "cld_ttft"),
        ("cloud_tpot_ms_median", "cld_tpot"),
        ("cloud_decode_tok_s_median", "cld_tok/s"),
        ("probe_agree_rate", "probe_ok"),
    ]
    print(f"# Matched-matrix metrics — {args.experiment_id}\n")
    print("| " + " | ".join(h for _, h in cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print(
            "| "
            + " | ".join(str(r.get(k) if r.get(k) is not None else "") for k, _ in cols)
            + " |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
