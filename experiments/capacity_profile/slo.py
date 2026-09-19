"""Per-stage summary statistics, the provisional-SLO verdict, and the
saturation-stop decision that halts ladder escalation.

Saturation is declared when ANY of these hold (each is sufficient):

  1. ``saturation_consecutive_breaches`` consecutive stages violate the SLO.
  2. Throughput knee: realized OUTPUT tokens/s rose by less than
     ``knee_min_relative_gain`` versus the previous stage even though the
     offered load roughly doubled.
  3. KV saturation: mean ``gpu_cache_usage_perc`` >= 0.98 AND the waiting
     queue grew over the stage for ``saturation_consecutive_breaches``
     consecutive stages.

Token throughput is reported as three separate numbers -- input, output, total
-- because prefill (input) and decode (output) load the GPU differently.

Queue wait: if the server exposes ``vllm:request_queue_time_seconds`` the mean
comes from the per-stage histogram delta (`_sum` delta / `_count` delta).
Otherwise a named proxy is reported: ``ttft_inflation_vs_c1`` = this stage's
p50 TTFT minus the c=1 baseline p50 TTFT (prefill-only time), floored at 0.
``max_num_waiting`` is kept too but is a queue DEPTH, not a wait time.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from .driver import RequestResult


def _pct(values: list[float], frac: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * frac
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _round(v: float | None, n: int = 3) -> float | None:
    return round(v, n) if v is not None else None


@dataclass
class StageSummary:
    stage: str
    mode: str
    offered_load: float  # concurrency (closed loop) or target rps (open loop)
    n_requests: int
    n_ok: int
    n_error: int
    error_rate: float
    # latency (seconds), computed over OK requests
    p50_e2e_s: float | None
    p95_e2e_s: float | None
    p99_e2e_s: float | None
    p50_ttft_s: float | None
    p95_ttft_s: float | None
    # realized work
    stage_wall_s: float
    realized_rps: float
    completed_input_tokens: int
    completed_output_tokens: int
    completed_total_tokens: int
    input_token_tps: float
    output_token_tps: float
    total_token_tps: float
    input_tokens_measured_fraction: float  # coverage of usage.input_tokens
    mean_per_request_decode_tps: float | None
    # queue wait
    mean_queue_wait_s: float | None
    queue_wait_source: str  # "vllm_request_queue_time_seconds" | "ttft_inflation_vs_c1" | "none"
    max_num_waiting: float | None  # queue DEPTH, not a wait time
    # server telemetry (from MetricsProbe samples over the stage)
    mean_gpu_cache_usage: float | None
    waiting_grew: bool
    mean_num_running: float | None
    gpu_active_s: float | None
    gpu_active_equiv_s_per_request: float | None
    # verdicts
    slo_ok: bool
    slo_breaches: list[str]
    schedule_lag_p95_s: float | None = None

    def row(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["slo_breaches"] = ";".join(self.slo_breaches)
        return d


def stage_meets(
    row: dict[str, Any],
    *,
    p95_e2e_s: float,
    p95_ttft_s: float,
    max_error_rate: float,
) -> bool:
    """Re-evaluate a persisted stage-summary row against an alternative SLO.
    Used for the capacity-envelope SLO-sensitivity sweep without re-running any
    load. A missing percentile (empty stage) is treated as passing."""
    p95_e2e = row.get("p95_e2e_s")
    p95_ttft = row.get("p95_ttft_s")
    err = row.get("error_rate") or 0.0
    if p95_e2e is not None and p95_e2e > p95_e2e_s:
        return False
    if p95_ttft is not None and p95_ttft > p95_ttft_s:
        return False
    if err > max_error_rate:
        return False
    return True


def _metric_series(samples: list[dict[str, Any]], key: str) -> list[float]:
    return [
        s[key] for s in samples
        if s.get("scrape_ok") and s.get(key) is not None
    ]


def _queue_wait_from_histogram(samples: list[dict[str, Any]]) -> float | None:
    """Mean seconds spent queued over the stage, from the cumulative
    histogram sum/count delta. None if the metric is absent or no request
    completed during the window."""
    sums = _metric_series(samples, "request_queue_time_seconds_sum")
    counts = _metric_series(samples, "request_queue_time_seconds_count")
    if len(sums) < 2 or len(counts) < 2:
        return None
    d_sum = sums[-1] - sums[0]
    d_cnt = counts[-1] - counts[0]
    if d_cnt <= 0:
        return None
    return d_sum / d_cnt


def summarize_stage(
    *,
    stage: str,
    mode: str,
    offered_load: float,
    results: list[RequestResult],
    stage_wall_s: float,
    metric_samples: list[dict[str, Any]],
    slo,
    baseline_ttft_s: float | None = None,
) -> StageSummary:
    ok = [r for r in results if not r.error and r.status == 200]
    err = [r for r in results if r.error or r.status not in (200, None)]
    n = len(results)
    n_ok, n_err = len(ok), len(err)
    error_rate = n_err / n if n else 0.0

    e2e = [r.e2e_s for r in ok if r.e2e_s is not None]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    lags = [r.schedule_lag_s for r in results if r.schedule_lag_s is not None]
    per_req_decode = [r.decode_tokens_per_s for r in ok if r.decode_tokens_per_s is not None]

    in_toks = [r.input_tokens for r in ok if r.input_tokens is not None]
    completed_in = sum(in_toks)
    completed_out = sum(r.output_tokens for r in ok)
    completed_total = completed_in + completed_out
    in_cov = (len(in_toks) / n_ok) if n_ok else 0.0

    realized_rps = n_ok / stage_wall_s if stage_wall_s > 0 else 0.0
    in_tps = completed_in / stage_wall_s if stage_wall_s > 0 else 0.0
    out_tps = completed_out / stage_wall_s if stage_wall_s > 0 else 0.0
    tot_tps = completed_total / stage_wall_s if stage_wall_s > 0 else 0.0

    running = _metric_series(metric_samples, "num_requests_running")
    waiting = _metric_series(metric_samples, "num_requests_waiting")
    kv = _metric_series(metric_samples, "gpu_cache_usage_perc")
    mean_kv = statistics.fmean(kv) if kv else None
    mean_running = statistics.fmean(running) if running else None
    max_waiting = max(waiting) if waiting else None
    waiting_grew = bool(waiting) and waiting[-1] > waiting[0]

    # queue wait: prefer the real histogram, else the named proxy
    q_hist = _queue_wait_from_histogram(metric_samples)
    if q_hist is not None:
        mean_queue_wait = q_hist
        q_source = "vllm_request_queue_time_seconds"
    elif baseline_ttft_s is not None and ttft:
        mean_queue_wait = max(0.0, (_pct(ttft, 0.50) or 0.0) - baseline_ttft_s)
        q_source = "ttft_inflation_vs_c1"
    else:
        mean_queue_wait = None
        q_source = "none"

    # GPU-active seconds: poll span * fraction of samples with >=1 running seq.
    # No exact per-request GPU-time attribution -> equivalent-seconds proxy.
    gpu_active_s = None
    gpu_active_equiv = None
    if len(metric_samples) >= 2:
        span = metric_samples[-1]["ts"] - metric_samples[0]["ts"]
        if span > 0 and running:
            active_frac = sum(1 for r in running if r and r > 0) / len(running)
            gpu_active_s = active_frac * span
            if n_ok:
                gpu_active_equiv = gpu_active_s / n_ok

    breaches: list[str] = []
    p95_e2e = _pct(e2e, 0.95)
    p95_ttft = _pct(ttft, 0.95)
    if p95_e2e is not None and p95_e2e > slo.p95_e2e_s:
        breaches.append(f"p95_e2e {p95_e2e:.2f}s > {slo.p95_e2e_s}s")
    if p95_ttft is not None and p95_ttft > slo.p95_ttft_s:
        breaches.append(f"p95_ttft {p95_ttft:.2f}s > {slo.p95_ttft_s}s")
    if error_rate > slo.max_error_rate:
        breaches.append(f"error_rate {error_rate:.1%} > {slo.max_error_rate:.1%}")

    return StageSummary(
        stage=stage,
        mode=mode,
        offered_load=offered_load,
        n_requests=n,
        n_ok=n_ok,
        n_error=n_err,
        error_rate=round(error_rate, 4),
        p50_e2e_s=_round(_pct(e2e, 0.50)),
        p95_e2e_s=_round(p95_e2e),
        p99_e2e_s=_round(_pct(e2e, 0.99)),
        p50_ttft_s=_round(_pct(ttft, 0.50)),
        p95_ttft_s=_round(p95_ttft),
        stage_wall_s=round(stage_wall_s, 3),
        realized_rps=round(realized_rps, 4),
        completed_input_tokens=completed_in,
        completed_output_tokens=completed_out,
        completed_total_tokens=completed_total,
        input_token_tps=round(in_tps, 2),
        output_token_tps=round(out_tps, 2),
        total_token_tps=round(tot_tps, 2),
        input_tokens_measured_fraction=round(in_cov, 4),
        mean_per_request_decode_tps=_round(
            statistics.fmean(per_req_decode) if per_req_decode else None, 2
        ),
        mean_queue_wait_s=_round(mean_queue_wait, 4),
        queue_wait_source=q_source,
        max_num_waiting=max_waiting,
        mean_gpu_cache_usage=_round(mean_kv, 4),
        waiting_grew=waiting_grew,
        mean_num_running=_round(mean_running, 3),
        gpu_active_s=_round(gpu_active_s, 2),
        gpu_active_equiv_s_per_request=_round(gpu_active_equiv, 4),
        slo_ok=not breaches,
        slo_breaches=breaches,
        schedule_lag_p95_s=_round(_pct(lags, 0.95)),
    )


@dataclass
class SaturationState:
    consecutive_slo_breaches: int = 0
    consecutive_kv_saturated: int = 0
    prev_output_tps: float | None = None
    prev_offered_load: float | None = None
    stopped: bool = False
    stop_reason: str = ""

    def update(self, s: StageSummary, cfg) -> bool:
        """Fold in one finished stage; return True when escalation must stop."""
        if not s.slo_ok:
            self.consecutive_slo_breaches += 1
        else:
            self.consecutive_slo_breaches = 0
        if self.consecutive_slo_breaches >= cfg.saturation_consecutive_breaches:
            self.stopped = True
            self.stop_reason = (
                f"{self.consecutive_slo_breaches} consecutive SLO-breaching stages"
            )
            return True

        kv_sat = (
            s.mean_gpu_cache_usage is not None
            and s.mean_gpu_cache_usage >= 0.98
            and s.waiting_grew
        )
        self.consecutive_kv_saturated = self.consecutive_kv_saturated + 1 if kv_sat else 0
        if self.consecutive_kv_saturated >= cfg.saturation_consecutive_breaches:
            self.stopped = True
            self.stop_reason = "KV pool >=98% with a growing wait queue"
            return True

        if (
            self.prev_output_tps is not None
            and self.prev_offered_load is not None
            and s.offered_load >= 1.8 * self.prev_offered_load
            and self.prev_output_tps > 0
        ):
            gain = (s.output_token_tps - self.prev_output_tps) / self.prev_output_tps
            if gain < cfg.knee_min_relative_gain:
                self.stopped = True
                self.stop_reason = (
                    f"throughput knee: +{gain:.1%} output tok/s for ~2x offered load"
                )
                return True

        self.prev_output_tps = s.output_token_tps
        self.prev_offered_load = s.offered_load
        return False
