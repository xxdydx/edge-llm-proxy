"""Offline tests for experiments/capacity_profile.

No network: the load drivers' HTTP path is exercised by the live smoke run, not
here. These cover the deterministic pieces -- workload selection, Poisson
schedule, metrics parsing, per-stage summarisation (token throughput split,
queue-wait signal + named proxy, SLO verdict), the saturation-stop predicate,
and the envelope rollup.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.capacity_profile.analysis import build_envelope, write_conditions_csv
from experiments.capacity_profile.config import EXPERIMENT, expected_max_requests
from experiments.capacity_profile.driver import (
    RequestResult,
    build_body,
    poisson_offsets,
)
from experiments.capacity_profile.metrics_probe import parse_metrics, _sum_metric
from experiments.capacity_profile.orchestrator import _smoke_cfg
from experiments.capacity_profile.slo import (
    SaturationState,
    StageSummary,
    _queue_wait_from_histogram,
    stage_meets,
    summarize_stage,
)
from experiments.capacity_profile.workload import load_workload

DATASET_GZ = (
    Path(__file__).resolve().parent.parent
    / "experiments" / "phase1_router" / "data" / "MbppPlus-v0.2.0.jsonl.gz"
)


# --- workload -------------------------------------------------------------


@pytest.mark.skipif(not DATASET_GZ.exists(), reason="phase1 dataset gz absent")
def test_load_workload_deterministic_and_provenance():
    a, prov_a = load_workload(DATASET_GZ, size=25, seed=20260910)
    b, prov_b = load_workload(DATASET_GZ, size=25, seed=20260910)
    assert [i.task_id for i in a] == [i.task_id for i in b]
    assert len(a) == 25
    # a different seed reorders the selection
    c, _ = load_workload(DATASET_GZ, size=25, seed=1)
    assert [i.task_id for i in c] != [i.task_id for i in a]
    assert prov_a["dataset_gz_sha256"] == prov_b["dataset_gz_sha256"]
    assert len(prov_a["selected_task_ids"]) == 25
    assert a[0].prompt.startswith("You are given a Python programming problem")
    assert a[0].prompt_chars == len(a[0].prompt)


@pytest.mark.skipif(not DATASET_GZ.exists(), reason="phase1 dataset gz absent")
def test_load_workload_rejects_oversize():
    with pytest.raises(ValueError):
        load_workload(DATASET_GZ, size=10_000, seed=1)


# --- request body ------------------------------------------------------


def test_build_body_disables_thinking_and_keeps_controls():
    item = _fake_item()
    body = build_body(item, EXPERIMENT)
    assert body["model"] == EXPERIMENT.request_model
    assert body["max_tokens"] == EXPERIMENT.max_output_tokens
    assert body["temperature"] == 0.0
    assert body["stream"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0]["content"] == item.prompt

    no_think = replace(EXPERIMENT, disable_thinking=False)
    assert "chat_template_kwargs" not in build_body(item, no_think)


# --- Poisson schedule ------------------------------------------------


def test_poisson_offsets_shape():
    rng = random.Random(42)
    offs = poisson_offsets(rate_rps=4.0, window_s=30.0, rng=rng)
    assert offs == sorted(offs)
    assert all(0 <= o < 30.0 for o in offs)
    # ~120 expected; wide tolerance for RNG
    assert 60 <= len(offs) <= 200
    # deterministic for a given seed
    assert poisson_offsets(4.0, 30.0, random.Random(42)) == offs


# --- metrics parsing -------------------------------------------------


_METRICS_TEXT = """\
# HELP vllm:num_requests_running running
vllm:num_requests_running{model_name="local"} 5.0
vllm:num_requests_waiting{model_name="local"} 3.0
vllm:gpu_cache_usage_perc{model_name="local"} 0.42
vllm:prefix_cache_queries_total{model_name="local"} 1000.0
vllm:prefix_cache_hits_total{model_name="local"} 250.0
vllm:request_queue_time_seconds_sum{model_name="local"} 12.0
vllm:request_queue_time_seconds_count{model_name="local"} 40.0
"""


def test_parse_metrics_reads_gauges_and_counters():
    m = parse_metrics(_METRICS_TEXT)
    assert m["num_requests_running"] == 5.0
    assert m["num_requests_waiting"] == 3.0
    assert m["gpu_cache_usage_perc"] == 0.42
    assert m["request_queue_time_seconds_sum"] == 12.0
    assert m["request_queue_time_seconds_count"] == 40.0
    # absent metric -> None, not 0
    assert m["num_requests_swapped"] is None


def test_sum_metric_sums_across_labels():
    text = 'vllm:x{a="1"} 2.0\nvllm:x{a="2"} 3.0\n'
    assert _sum_metric(text, "vllm:x") == 5.0
    assert _sum_metric(text, "vllm:absent") is None


def test_queue_wait_from_histogram_delta():
    samples = [
        {"scrape_ok": True, "request_queue_time_seconds_sum": 10.0,
         "request_queue_time_seconds_count": 20.0},
        {"scrape_ok": True, "request_queue_time_seconds_sum": 40.0,
         "request_queue_time_seconds_count": 40.0},
    ]
    # (40-10) / (40-20) = 1.5 s mean queue wait
    assert _queue_wait_from_histogram(samples) == pytest.approx(1.5)
    # no count movement -> undefined
    flat = [
        {"scrape_ok": True, "request_queue_time_seconds_sum": 10.0,
         "request_queue_time_seconds_count": 20.0},
        {"scrape_ok": True, "request_queue_time_seconds_sum": 10.0,
         "request_queue_time_seconds_count": 20.0},
    ]
    assert _queue_wait_from_histogram(flat) is None
    assert _queue_wait_from_histogram([]) is None


# --- stage summary --------------------------------------------------


def _fake_item():
    from experiments.capacity_profile.workload import WorkloadItem

    p = "You are given a Python programming problem ... solve it.\n"
    return WorkloadItem(task_id="Mbpp/1", entry_point="f", prompt=p, prompt_chars=len(p))


def _ok(seq, e2e, ttft, in_tok, out_tok):
    return RequestResult(
        seq=seq, task_id="Mbpp/1", mode="closed_loop", stage="s",
        status=200, input_tokens=in_tok, output_tokens=out_tok,
        e2e_s=e2e, ttft_s=ttft, decode_s=max(e2e - ttft, 0.001),
        decode_tokens_per_s=(out_tok - 1) / max(e2e - ttft, 0.001),
    )


def _metric_samples(running, kv, *, with_hist=False, waiting=(0, 0), t0=1000.0, dt=2.0):
    out = []
    for i, r in enumerate(running):
        s = {
            "ts": t0 + i * dt, "stage": "s", "scrape_ok": True,
            "num_requests_running": r,
            "num_requests_waiting": waiting[min(i, len(waiting) - 1)],
            "gpu_cache_usage_perc": kv,
        }
        if with_hist:
            s["request_queue_time_seconds_sum"] = 5.0 * i
            s["request_queue_time_seconds_count"] = 10.0 * i
        out.append(s)
    return out


def test_summarize_stage_splits_token_throughput():
    results = [_ok(i, 4.0, 0.5, 1000, 200) for i in range(10)]
    summ = summarize_stage(
        stage="closed/c4/rep0", mode="closed_loop", offered_load=4.0,
        results=results, stage_wall_s=10.0,
        metric_samples=_metric_samples([4, 4, 4], 0.5),
        slo=EXPERIMENT.slo,
    )
    assert summ.completed_input_tokens == 10_000
    assert summ.completed_output_tokens == 2_000
    assert summ.completed_total_tokens == 12_000
    assert summ.input_token_tps == pytest.approx(1000.0)
    assert summ.output_token_tps == pytest.approx(200.0)
    assert summ.total_token_tps == pytest.approx(1200.0)
    assert summ.input_tokens_measured_fraction == 1.0


def test_summarize_stage_input_coverage_partial():
    results = [_ok(i, 3.0, 0.4, 500, 100) for i in range(8)]
    for r in results[:3]:
        r.input_tokens = None  # gateway/usage gap on some rows
    summ = summarize_stage(
        stage="s", mode="closed_loop", offered_load=2.0, results=results,
        stage_wall_s=5.0, metric_samples=_metric_samples([2, 2], 0.3), slo=EXPERIMENT.slo,
    )
    assert summ.input_tokens_measured_fraction == pytest.approx(5 / 8)
    assert summ.completed_input_tokens == 5 * 500  # only measured rows summed


def test_summarize_stage_queue_wait_prefers_histogram():
    results = [_ok(i, 4.0, 1.2, 900, 150) for i in range(6)]
    summ = summarize_stage(
        stage="s", mode="closed_loop", offered_load=8.0, results=results,
        stage_wall_s=6.0,
        metric_samples=_metric_samples([8, 8, 8, 8], 0.6, with_hist=True),
        slo=EXPERIMENT.slo, baseline_ttft_s=0.3,
    )
    assert summ.queue_wait_source == "vllm_request_queue_time_seconds"
    assert summ.mean_queue_wait_s == pytest.approx(0.5)  # (15-0)/(30-0)


def test_summarize_stage_queue_wait_named_proxy_when_no_histogram():
    results = [_ok(i, 5.0, 1.0, 900, 150) for i in range(6)]  # p50 ttft = 1.0
    summ = summarize_stage(
        stage="s", mode="closed_loop", offered_load=8.0, results=results,
        stage_wall_s=6.0, metric_samples=_metric_samples([8, 8], 0.6),
        slo=EXPERIMENT.slo, baseline_ttft_s=0.3,
    )
    assert summ.queue_wait_source == "ttft_inflation_vs_c1"
    assert summ.mean_queue_wait_s == pytest.approx(0.7)  # 1.0 - 0.3
    # no baseline and no histogram -> explicitly none
    summ2 = summarize_stage(
        stage="s", mode="closed_loop", offered_load=1.0, results=results,
        stage_wall_s=6.0, metric_samples=_metric_samples([1, 1], 0.6),
        slo=EXPERIMENT.slo, baseline_ttft_s=None,
    )
    assert summ2.queue_wait_source == "none"
    assert summ2.mean_queue_wait_s is None


def test_summarize_stage_slo_breach_flags():
    slow = [_ok(i, 20.0, 0.5, 500, 100) for i in range(10)]
    summ = summarize_stage(
        stage="s", mode="closed_loop", offered_load=32.0, results=slow,
        stage_wall_s=25.0, metric_samples=_metric_samples([8, 8], 0.99), slo=EXPERIMENT.slo,
    )
    assert not summ.slo_ok
    assert any("p95_e2e" in b for b in summ.slo_breaches)


def test_summarize_stage_error_rate_breach():
    results = [_ok(i, 3.0, 0.4, 500, 100) for i in range(8)]
    results.append(RequestResult(seq=99, task_id="x", mode="closed_loop", stage="s",
                                 status=500, error="HTTPStatusError: 500"))
    summ = summarize_stage(
        stage="s", mode="closed_loop", offered_load=8.0, results=results,
        stage_wall_s=5.0, metric_samples=_metric_samples([8, 8], 0.5), slo=EXPERIMENT.slo,
    )
    assert summ.n_error == 1
    assert any("error_rate" in b for b in summ.slo_breaches)


def test_gpu_active_equiv_cost():
    results = [_ok(i, 4.0, 0.5, 800, 160) for i in range(4)]
    # 4 samples, span 6s, 3/4 have running>0 -> gpu_active_s = 4.5, /4 ok = 1.125
    samples = _metric_samples([0, 2, 4, 4], 0.5)
    summ = summarize_stage(
        stage="s", mode="closed_loop", offered_load=4.0, results=results,
        stage_wall_s=6.0, metric_samples=samples, slo=EXPERIMENT.slo,
    )
    assert summ.gpu_active_s == pytest.approx(4.5)
    assert summ.gpu_active_equiv_s_per_request == pytest.approx(1.125)


# --- saturation predicate ----------------------------------------


def _stage(**kw) -> StageSummary:
    base = dict(
        stage="s", mode="closed_loop", offered_load=8.0, n_requests=10, n_ok=10,
        n_error=0, error_rate=0.0, p50_e2e_s=3.0, p95_e2e_s=5.0, p99_e2e_s=6.0,
        p50_ttft_s=0.5, p95_ttft_s=0.8, stage_wall_s=10.0, realized_rps=3.0,
        completed_input_tokens=1000, completed_output_tokens=2000,
        completed_total_tokens=3000, input_token_tps=100.0, output_token_tps=200.0,
        total_token_tps=300.0, input_tokens_measured_fraction=1.0,
        mean_per_request_decode_tps=40.0, mean_queue_wait_s=0.1,
        queue_wait_source="none", max_num_waiting=0.0, mean_gpu_cache_usage=0.5,
        waiting_grew=False, mean_num_running=8.0, gpu_active_s=9.0,
        gpu_active_equiv_s_per_request=0.9, slo_ok=True, slo_breaches=[],
    )
    base.update(kw)
    return StageSummary(**base)


def test_saturation_two_consecutive_slo_breaches():
    sat = SaturationState()
    assert not sat.update(_stage(slo_ok=False, slo_breaches=["p95_e2e"]), EXPERIMENT)
    assert sat.update(_stage(slo_ok=False, slo_breaches=["p95_e2e"], offered_load=16.0), EXPERIMENT)
    assert sat.stopped and "consecutive SLO" in sat.stop_reason


def test_saturation_throughput_knee():
    sat = SaturationState()
    assert not sat.update(_stage(offered_load=4.0, output_token_tps=200.0), EXPERIMENT)
    # offered load doubled, output tok/s barely moved -> knee
    assert sat.update(_stage(offered_load=8.0, output_token_tps=205.0), EXPERIMENT)
    assert "knee" in sat.stop_reason


def test_saturation_kv_pool():
    sat = SaturationState()
    hot = dict(mean_gpu_cache_usage=0.99, waiting_grew=True)
    assert not sat.update(_stage(offered_load=8.0, **hot), EXPERIMENT)
    assert sat.update(_stage(offered_load=9.0, **hot), EXPERIMENT)
    assert "KV pool" in sat.stop_reason


def test_saturation_healthy_ladder_does_not_stop():
    sat = SaturationState()
    for load, tps in [(1, 60), (2, 118), (4, 230), (8, 440)]:
        stop = sat.update(_stage(offered_load=float(load), output_token_tps=float(tps)), EXPERIMENT)
        assert not stop
    assert not sat.stopped


# --- envelope + csv ------------------------------------------------


def _row(**kw) -> dict:
    return _stage(**kw).row()


def test_build_envelope_frontier_and_knee():
    rows = [
        _row(stage="closed/c1/rep0", mode="closed_loop", offered_load=1.0,
             output_token_tps=60.0, input_token_tps=300.0, total_token_tps=360.0,
             slo_ok=True, gpu_active_equiv_s_per_request=0.5),
        _row(stage="closed/c4/rep0", mode="closed_loop", offered_load=4.0,
             output_token_tps=230.0, input_token_tps=1100.0, total_token_tps=1330.0,
             slo_ok=True, gpu_active_equiv_s_per_request=0.3),
        _row(stage="closed/c8/rep0", mode="closed_loop", offered_load=8.0,
             output_token_tps=240.0, input_token_tps=1150.0, total_token_tps=1390.0,
             slo_ok=True, mean_gpu_cache_usage=0.99, max_num_waiting=5.0,
             gpu_active_equiv_s_per_request=0.31),
        _row(stage="closed/c16/rep0", mode="closed_loop", offered_load=16.0,
             output_token_tps=242.0, slo_ok=False, slo_breaches=["p95_e2e 14s > 12s"]),
        _row(stage="open/rps2.0", mode="open_loop", offered_load=2.0, slo_ok=True,
             realized_rps=1.95),
        _row(stage="open/rps8.0", mode="open_loop", offered_load=8.0, slo_ok=False,
             slo_breaches=["p95_e2e"]),
    ]
    env = build_envelope(
        rows, slo=EXPERIMENT.slo.as_dict(),
        saturation={"stopped": True, "stop_reason": "knee"},
        max_num_seqs=8, provenance={"workload_seed": 20260910},
    )
    assert env["max_concurrency_under_slo"] == 8
    assert env["max_offered_rps_under_slo"] == 2.0
    assert env["throughput_knee_concurrency"] == 4  # c8 within 5% of c4
    assert env["peak_realized_output_tps"] == 242.0
    assert env["peak_realized_input_tps"] == 1150.0
    assert env["limiting_resource"] == "kv_cache_pool"
    assert "GPU-active-equivalent seconds per request" in env["cost_metric_note"]
    assert env["gpu_active_equiv_s_per_request_at_c1"] == 0.5


def test_build_envelope_aggregates_repetitions_before_finding_knee():
    rows = []
    for c, throughputs in ((1, (58, 60, 62)), (2, (112, 120, 124)),
                           (4, (225, 230, 235)), (8, (231, 234, 238))):
        for rep, tps in enumerate(throughputs):
            rows.append(_row(
                stage=f"closed/c{c}/rep{rep}", mode="closed_loop",
                offered_load=float(c), output_token_tps=float(tps), slo_ok=True,
            ))
    env = build_envelope(
        rows, slo=EXPERIMENT.slo.as_dict(), saturation={},
        max_num_seqs=8, provenance={},
    )
    # Adjacent c=1 replicates are similar by design; they must not be mistaken
    # for two different load rungs. The first <5% gain is c4 -> c8.
    assert env["throughput_knee_concurrency"] == 4
    assert env["max_concurrency_under_slo"] == 8


def test_stage_meets_alternative_slo():
    row = _row(p95_e2e_s=10.0, p95_ttft_s=1.0, error_rate=0.0)
    assert stage_meets(row, p95_e2e_s=12.0, p95_ttft_s=2.0, max_error_rate=0.01)
    assert not stage_meets(row, p95_e2e_s=8.0, p95_ttft_s=2.0, max_error_rate=0.01)
    assert not stage_meets(
        _row(p95_e2e_s=5.0, error_rate=0.05), p95_e2e_s=12.0,
        p95_ttft_s=2.0, max_error_rate=0.01,
    )
    # empty stage (no percentile) passes
    assert stage_meets(
        {"p95_e2e_s": None, "p95_ttft_s": None, "error_rate": 0.0},
        p95_e2e_s=1.0, p95_ttft_s=1.0, max_error_rate=0.0,
    )


def test_build_envelope_slo_sensitivity_monotone():
    rows = [
        _row(stage=f"closed/c{c}/rep0", mode="closed_loop", offered_load=float(c),
             p95_e2e_s=lat, slo_ok=lat <= 12.0, output_token_tps=50.0 * c,
             error_rate=0.0, p95_ttft_s=0.8)
        for c, lat in [(1, 2.0), (2, 4.0), (4, 7.0), (8, 11.0), (16, 18.0), (24, 30.0)]
    ]
    env = build_envelope(
        rows, slo=EXPERIMENT.slo.as_dict(),
        saturation={"stop_reason": "x", "closed_loop_saturation_onset": {}},
        max_num_seqs=8, provenance={},
        slo_sensitivity_p95_e2e_s=(8.0, 12.0, 20.0),
    )
    sens = {v["p95_e2e_s"]: v["max_concurrency_under_slo"] for v in env["slo_sensitivity"]}
    assert sens[8.0] == 4      # only c1,c2,c4 under 8s
    assert sens[12.0] == 8     # + c8 at 11s
    assert sens[20.0] == 16    # + c16 at 18s
    # frontier must be monotone non-decreasing as the cutoff loosens
    ordered = [sens[c] for c in (8.0, 12.0, 20.0)]
    assert ordered == sorted(ordered)
    labels = {v["p95_e2e_s"]: v["label"] for v in env["slo_sensitivity"]}
    assert labels == {8.0: "stricter", 12.0: "default", 20.0: "looser"}


def test_write_conditions_csv_headers(tmp_path):
    out = tmp_path / "conditions.csv"
    write_conditions_csv([_row()], out)
    header = out.read_text().splitlines()[0]
    for col in (
        "input_token_tps", "output_token_tps", "total_token_tps",
        "completed_input_tokens", "mean_queue_wait_s", "queue_wait_source",
        "input_tokens_measured_fraction", "gpu_active_equiv_s_per_request",
    ):
        assert col in header


def test_request_result_row_roundtrips_input_tokens():
    r = _ok(1, 4.0, 0.5, 1234, 200)
    row = r.row()
    assert row["input_tokens"] == 1234
    assert row["output_tokens"] == 200


# --- config helpers ---------------------------------------------


def test_smoke_cfg_shrinks_ladder():
    sc = _smoke_cfg(EXPERIMENT)
    assert sc.concurrency_ladder == (1, 2)
    assert sc.closed_loop_reps == 1
    assert sc.workload_size <= 20


def test_expected_max_requests_positive():
    est = expected_max_requests(EXPERIMENT)
    c = est["closed_loop_if_no_saturation_stop"]
    o = est["open_loop_if_no_saturation_stop"]
    assert c > 0 and o > 0
    assert est["total_if_no_saturation_stop"] == c + o
