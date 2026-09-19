"""Roll per-stage summaries up into ``conditions.csv`` and a capacity envelope.

The envelope answers: how much load can this one 27B box carry while holding
the provisional SLO, where is the throughput knee, and what runs out first.
Cost is reported as GPU-active-equivalent seconds per request (no exact
per-request GPU attribution is available from vLLM).
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


_CONDITION_FIELDS = [
    "stage", "mode", "offered_load", "n_requests", "n_ok", "n_error", "error_rate",
    "p50_e2e_s", "p95_e2e_s", "p99_e2e_s", "p50_ttft_s", "p95_ttft_s",
    "stage_wall_s", "realized_rps",
    "completed_input_tokens", "completed_output_tokens", "completed_total_tokens",
    "input_token_tps", "output_token_tps", "total_token_tps",
    "input_tokens_measured_fraction", "mean_per_request_decode_tps",
    "mean_queue_wait_s", "queue_wait_source", "max_num_waiting",
    "mean_gpu_cache_usage", "waiting_grew", "mean_num_running",
    "gpu_active_s", "gpu_active_equiv_s_per_request",
    "slo_ok", "slo_breaches", "schedule_lag_p95_s",
]


def write_conditions_csv(stage_rows: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CONDITION_FIELDS)
        writer.writeheader()
        for row in stage_rows:
            writer.writerow({k: row.get(k, "") for k in _CONDITION_FIELDS})


def _limiting_resource(closed_rows: list[dict], max_num_seqs: int | None) -> str:
    """Best-effort classification of what caps the box, judged at the highest
    offered load that still met the SLO (that is the operating point the
    envelope is about)."""
    if not closed_rows:
        return "unknown"
    ok_rows = [r for r in closed_rows if r.get("slo_ok")] or closed_rows
    peak = max(ok_rows, key=lambda r: r.get("output_token_tps") or 0.0)
    kv = peak.get("mean_gpu_cache_usage") or 0.0
    waiting = peak.get("max_num_waiting") or 0.0
    running = peak.get("mean_num_running") or 0.0
    if kv >= 0.98:
        return "kv_cache_pool"
    if max_num_seqs and running >= 0.9 * max_num_seqs and waiting > 0:
        return f"max_num_seqs ({max_num_seqs}) scheduler slots"
    if waiting > 0:
        return "scheduler queue (decode-bound)"
    return "latency SLO before hardware saturation"


def _by_load(rows: list[dict]) -> dict[float, list[dict]]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[float(row["offered_load"])].append(row)
    return dict(grouped)


def _median_metric(rows: list[dict], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.median(values) if values else None


def _feasible_groups(rows: list[dict], meets) -> dict[float, list[dict]]:
    """A load is feasible when a majority of its independent repetitions meet
    the SLO. This prevents one lucky replicate from defining the envelope."""
    out: dict[float, list[dict]] = {}
    for load, reps in _by_load(rows).items():
        if sum(bool(meets(r)) for r in reps) >= (len(reps) // 2 + 1):
            out[load] = reps
    return out


def _frontier(closed_rows, open_rows, meets) -> dict[str, Any]:
    """Given a predicate ``meets(row)->bool``, return the SLO-feasible frontier."""
    c_ok = _feasible_groups(closed_rows, meets)
    o_ok = _feasible_groups(open_rows, meets)
    return {
        "max_concurrency_under_slo": int(max(c_ok)) if c_ok else None,
        "max_offered_rps_under_slo": max(o_ok) if o_ok else None,
        "max_realized_rps_under_slo": max(
            (_median_metric(reps, "realized_rps") or 0.0 for reps in o_ok.values()),
            default=None,
        ),
        "peak_output_tps_under_slo": max(
            (_median_metric(reps, "output_token_tps") or 0.0 for reps in c_ok.values()),
            default=None,
        ),
    }


def build_envelope(
    stage_rows: list[dict[str, Any]],
    *,
    slo: dict[str, float],
    saturation: dict[str, Any],
    max_num_seqs: int | None,
    provenance: dict[str, Any],
    slo_sensitivity_p95_e2e_s: tuple[float, ...] = (8.0, 12.0, 20.0),
) -> dict[str, Any]:
    from .slo import stage_meets

    closed = [r for r in stage_rows if r["mode"] == "closed_loop"]
    openl = [r for r in stage_rows if r["mode"] == "open_loop"]

    default_frontier = _frontier(closed, openl, lambda r: bool(r["slo_ok"]))
    max_conc_under_slo = default_frontier["max_concurrency_under_slo"]
    max_rps_under_slo = default_frontier["max_offered_rps_under_slo"]
    max_realized_rps_under_slo = default_frontier["max_realized_rps_under_slo"]

    # knee: last closed-loop stage before realized OUTPUT tok/s stops growing >5%
    knee_concurrency = None
    prev = None
    closed_groups = _by_load(closed)
    for load, reps in sorted(closed_groups.items()):
        tps = _median_metric(reps, "output_token_tps") or 0.0
        if prev is not None and prev[1] > 0:
            if (tps - prev[1]) / prev[1] < 0.05:
                knee_concurrency = prev[0]
                break
        prev = (int(load), tps)
    if knee_concurrency is None and closed:
        knee_concurrency = int(max(
            closed_groups,
            key=lambda load: _median_metric(closed_groups[load], "output_token_tps") or 0.0,
        ))

    peak_output_tps = max((r.get("output_token_tps") or 0.0 for r in closed), default=0.0)
    peak_total_tps = max((r.get("total_token_tps") or 0.0 for r in closed), default=0.0)
    peak_input_tps = max((r.get("input_token_tps") or 0.0 for r in closed), default=0.0)
    # GPU-active-equivalent cost at the knee and at low load
    cost_at_knee = (
        _median_metric(closed_groups.get(float(knee_concurrency), []),
                       "gpu_active_equiv_s_per_request")
        if knee_concurrency is not None else None
    )
    cost_at_c1 = _median_metric(closed_groups.get(1.0, []),
                                "gpu_active_equiv_s_per_request")

    # SLO sensitivity: recompute the feasible frontier at several p95 e2e
    # latency cutoffs so the envelope is not tied to one arbitrary number.
    # p95 TTFT and the error-rate ceiling are held at the default SLO.
    slo_sensitivity = []
    for cutoff in slo_sensitivity_p95_e2e_s:
        fr = _frontier(
            closed, openl,
            lambda r, c=cutoff: stage_meets(
                r, p95_e2e_s=c, p95_ttft_s=slo["p95_ttft_s"],
                max_error_rate=slo["max_error_rate"],
            ),
        )
        label = (
            "stricter" if cutoff < slo["p95_e2e_s"]
            else "default" if cutoff == slo["p95_e2e_s"]
            else "looser"
        )
        slo_sensitivity.append({"p95_e2e_s": cutoff, "label": label, **fr})

    return {
        "slo": slo,
        "max_concurrency_under_slo": max_conc_under_slo,
        "max_offered_rps_under_slo": max_rps_under_slo,
        "max_realized_rps_under_slo": max_realized_rps_under_slo,
        "slo_sensitivity": slo_sensitivity,
        "throughput_knee_concurrency": knee_concurrency,
        "peak_realized_output_tps": round(peak_output_tps, 1),
        "peak_realized_input_tps": round(peak_input_tps, 1),
        "peak_realized_total_tps": round(peak_total_tps, 1),
        "limiting_resource": _limiting_resource(closed, max_num_seqs),
        "queue_wait_source": (closed[-1].get("queue_wait_source") if closed else "none"),
        "max_mean_queue_wait_s": max(
            (r.get("mean_queue_wait_s") or 0.0 for r in closed + openl), default=0.0
        ),
        "input_token_coverage_min": min(
            (r.get("input_tokens_measured_fraction") or 0.0 for r in closed + openl),
            default=0.0,
        ),
        "gpu_active_equiv_s_per_request_at_c1": cost_at_c1,
        "gpu_active_equiv_s_per_request_at_knee": cost_at_knee,
        "saturation": saturation,
        "n_closed_loop_stages": len(closed),
        "n_open_loop_stages": len(openl),
        "cost_metric_note": (
            "cost = GPU-active-equivalent seconds per request = "
            "(stage span * fraction of metrics samples with num_requests_running>0) "
            "/ completed OK requests. vLLM exposes no exact per-request GPU-time "
            "attribution, so this is an equivalent-seconds proxy, not a billed number."
        ),
        "provenance": provenance,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
