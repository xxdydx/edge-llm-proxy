#!/usr/bin/env python3
"""27B edge capacity-profile orchestrator.

  python -m experiments.capacity_profile.orchestrator --preflight
  python -m experiments.capacity_profile.orchestrator --smoke
  python -m experiments.capacity_profile.orchestrator --full
  python -m experiments.capacity_profile.orchestrator --full --resume-run-dir results/run-...
  python -m experiments.capacity_profile.orchestrator --analysis-only --resume-run-dir results/run-...

Load stages are HTTP only. Nothing runs model code on the host. Every stage
appends to ``raw_requests.jsonl`` / ``stage_summaries.jsonl`` /
``metrics_timeseries.jsonl`` so a killed run resumes at the next stage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path

import httpx

from .analysis import build_envelope, load_jsonl, write_conditions_csv
from .config import EXPERIMENT, CapacityConfig, expected_max_requests, new_run_dir
from .driver import run_closed_loop_cell, run_open_loop_stage
from .metrics_probe import QUEUE_TIME_METRICS, MetricsProbe, parse_metrics
from .plots import save_all
from .slo import SaturationState, summarize_stage
from .workload import load_workload

log = logging.getLogger("capacity_profile")


def _client(cfg: CapacityConfig, max_conn: int) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=cfg.base_url.rstrip("/"),
        timeout=httpx.Timeout(cfg.request_read_inactivity_timeout_s, connect=cfg.connect_timeout_s),
        limits=httpx.Limits(max_connections=max_conn + 8, max_keepalive_connections=max_conn + 8),
    )


async def preflight(cfg: CapacityConfig) -> dict:
    """Reachability + served-model identity + scheduler config. Aborts on a
    model-identity mismatch before any load is applied."""
    async with _client(cfg, 4) as client:
        m = await client.get("/metrics", timeout=10.0)
        m.raise_for_status()
        cache_cfg = {}
        for line in m.text.splitlines():
            if line.startswith("vllm:cache_config_info{"):
                import re
                cache_cfg = dict(re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', line))
                break
        try:
            v = await client.get("/version", timeout=10.0)
            version = v.json().get("version", "unknown") if v.status_code == 200 else "unknown"
        except Exception:
            version = "unknown"

        # /v1/models carries the underlying HF path and the served context window
        model_id = model_root = None
        model_max_len = None
        try:
            mm = await client.get("/v1/models", timeout=10.0)
            if mm.status_code == 200:
                data = (mm.json() or {}).get("data") or []
                if data:
                    model_id = data[0].get("id")
                    model_root = data[0].get("root") or data[0].get("parent")
                    model_max_len = data[0].get("max_model_len")
        except Exception:
            pass

        body = {
            "model": cfg.request_model, "max_tokens": 1, "temperature": 0,
            "messages": [{"role": "user", "content": "ping"}], "stream": False,
        }
        if cfg.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = await client.post("/v1/messages", json=body)
        r.raise_for_status()
        served = (r.json() or {}).get("model")
        if served != cfg.expected_model_exact:
            raise SystemExit(
                f"identity mismatch: served {served!r} != expected {cfg.expected_model_exact!r}"
            )

        # HF-path check: hard-abort only on a clear contradiction (a different
        # model family served), warn on merely-unconfirmed.
        hf_seen = model_root or model_id
        hf_ok = None
        if hf_seen and cfg.expected_hf_model:
            hf_ok = cfg.expected_hf_model.split("/")[-1].lower() in str(hf_seen).lower()
            if hf_ok is False and hf_seen not in (served, cfg.expected_model_exact):
                raise SystemExit(
                    f"model mismatch: /v1/models reports {hf_seen!r}, expected "
                    f"{cfg.expected_hf_model!r}"
                )
        ctx_ok = None
        if model_max_len is not None:
            ctx_ok = int(model_max_len) == cfg.expected_max_model_len
            if not ctx_ok:
                raise SystemExit(
                    f"context-window mismatch: /v1/models reports {model_max_len}, "
                    f"expected exactly {cfg.expected_max_model_len}"
                )

        present_metric_names = {
            ln.split("{")[0].split(" ")[0] for ln in m.text.splitlines()
            if ln and not ln.startswith("#")
        }
        queue_time_exposed = all(q in present_metric_names for q in QUEUE_TIME_METRICS)
        info = {
            "base_url": cfg.base_url, "vllm_version": version,
            "served_model": served, "model_id": model_id, "model_root": model_root,
            "model_max_len": model_max_len,
            "expected_hf_model": cfg.expected_hf_model,
            "hf_model_confirmed": hf_ok, "context_window_ok": ctx_ok,
            "cache_config": cache_cfg,
            "max_num_seqs": _int_or_none(cache_cfg.get("max_num_seqs")),
            "queue_time_histogram_exposed": queue_time_exposed,
            "queue_wait_signal": (
                "vllm_request_queue_time_seconds" if queue_time_exposed
                else "ttft_inflation_vs_c1 (proxy: stage p50 TTFT - c=1 p50 TTFT)"
            ),
            "metrics_snapshot": parse_metrics(m.text),
        }
        log.info(
            "preflight OK: model=%s (hf=%s conf=%s) ctx=%s(ok=%s) version=%s "
            "max_num_seqs=%s queue_time_metric=%s",
            served, hf_seen, hf_ok, model_max_len, ctx_ok, version,
            info["max_num_seqs"], queue_time_exposed,
        )
        return info


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _stage_done(run_dir: Path, stage: str) -> bool:
    for row in load_jsonl(run_dir / "stage_summaries.jsonl"):
        if row.get("stage") == stage:
            return True
    return False


def _append(path: Path, obj: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(obj) + "\n")


async def _run_closed_loop(cfg, run_dir, workload, onset: SaturationState) -> float | None:
    """Run the ENTIRE concurrency ladder. Saturation is NOT a stop condition
    here -- the degradation curve past the knee is wanted data. ``onset`` is
    updated purely to record where saturation first appears, for the envelope;
    it never breaks the loop. Returns the concurrency at which closed-loop
    saturation was first detected, or None."""
    onset_at: float | None = None
    raw_path = run_dir / "raw_requests.jsonl"
    sum_path = run_dir / "stage_summaries.jsonl"
    metrics_path = run_dir / "metrics_timeseries.jsonl"
    seq = 0
    baseline_ttft = _baseline_ttft(run_dir)  # c=1 p50 TTFT, for the queue-wait proxy
    for conc in cfg.concurrency_ladder:
        per_cell = min(
            cfg.closed_loop_cell_max_requests,
            max(cfg.closed_loop_cell_min_requests, conc * cfg.closed_loop_cell_multiple),
        )
        for rep in range(cfg.closed_loop_reps):
            stage = f"closed/c{conc}/rep{rep}"
            if _stage_done(run_dir, stage):
                log.info("skip done stage %s", stage); continue
            async with _client(cfg, conc) as client:
                async with MetricsProbe(client, metrics_path, cfg.metrics_poll_s, stage=stage) as probe:
                    t0 = time.perf_counter()
                    results = await run_closed_loop_cell(
                        client, workload, cfg, concurrency=conc,
                        n_requests=per_cell, stage=stage, seq0=seq,
                    )
                    wall = time.perf_counter() - t0
                seq += per_cell
                summ = summarize_stage(
                    stage=stage, mode="closed_loop", offered_load=float(conc),
                    results=results, stage_wall_s=wall,
                    metric_samples=[s for s in probe.samples if s.get("stage") == stage],
                    slo=cfg.slo,
                    baseline_ttft_s=baseline_ttft if conc > 1 else None,
                )
            for r in results:
                _append(raw_path, r.row())
            _append(sum_path, summ.row())
            log.info(
                "%s: n=%d ok=%d p95_e2e=%.2fs out_tps=%.0f in_tps=%.0f kv=%.2f qwait=%s(%s) slo_ok=%s",
                stage, summ.n_requests, summ.n_ok, summ.p95_e2e_s or -1,
                summ.output_token_tps, summ.input_token_tps,
                summ.mean_gpu_cache_usage or -1, summ.mean_queue_wait_s,
                summ.queue_wait_source, summ.slo_ok,
            )
        if conc == 1:
            baseline_ttft = _baseline_ttft(run_dir)
        # record (do NOT act on) where closed-loop saturation first appears
        rung_rows = [
            r for r in load_jsonl(sum_path)
            if r["mode"] == "closed_loop" and r["offered_load"] == float(conc)
        ]
        if rung_rows and not onset.stopped:
            median_row = sorted(rung_rows, key=lambda r: r.get("p95_e2e_s") or 0.0)[len(rung_rows) // 2]
            onset.update(_row_to_summary(median_row), cfg)
            if onset.stopped:
                onset_at = float(conc)
                log.info("closed loop: saturation ONSET at c=%d (%s) -- continuing ladder anyway",
                         conc, onset.stop_reason)
    return onset_at


def _row_to_summary(row: dict):
    from .slo import StageSummary
    return StageSummary(**{k: _coerce(k, v) for k, v in row.items()})


def _coerce(key, val):
    if key == "slo_breaches" and isinstance(val, str):
        return [x for x in val.split(";") if x]
    return val


def _baseline_ttft(run_dir: Path) -> float | None:
    """Median p50 TTFT across finished c=1 closed-loop stages (prefill-only
    time), used as the queue-wait proxy baseline."""
    vals = [
        r["p50_ttft_s"] for r in load_jsonl(run_dir / "stage_summaries.jsonl")
        if r.get("mode") == "closed_loop" and r.get("offered_load") == 1.0
        and r.get("p50_ttft_s") is not None
    ]
    if not vals:
        return None
    return sorted(vals)[len(vals) // 2]


async def _run_open_loop(cfg, run_dir, workload, onset: SaturationState) -> SaturationState:
    """Ramp the Poisson arrival rate. The FULL SaturationState predicate (SLO
    breaches, throughput knee, KV pool + growing backlog) decides when to stop;
    the resulting state/reason is returned so the caller can persist it."""
    raw_path = run_dir / "raw_requests.jsonl"
    sum_path = run_dir / "stage_summaries.jsonl"
    metrics_path = run_dir / "metrics_timeseries.jsonl"
    rng = random.Random(cfg.workload_seed)
    seq = 1_000_000
    sat = SaturationState()
    for rate in cfg.arrival_rates_rps:
        stage = f"open/rps{rate}"
        if _stage_done(run_dir, stage):
            log.info("skip done stage %s", stage)
            row = next(
                (r for r in load_jsonl(sum_path) if r.get("stage") == stage), None
            )
            if row is not None and sat.update(_row_to_summary(row), cfg):
                break
            continue
        max_conn = min(cfg.open_loop_max_inflight, int(rate * 20) + 16)
        async with _client(cfg, max_conn) as client:
            async with MetricsProbe(client, metrics_path, cfg.metrics_poll_s, stage=stage) as probe:
                t0 = time.perf_counter()
                results = await run_open_loop_stage(
                    client, workload, cfg, rate_rps=rate, stage=stage, seq0=seq, rng=rng,
                )
                wall = time.perf_counter() - t0
        seq += len(results) + 10
        summ = summarize_stage(
            stage=stage, mode="open_loop", offered_load=float(rate),
            results=results, stage_wall_s=wall,
            metric_samples=[s for s in probe.samples if s.get("stage") == stage],
            slo=cfg.slo,
            baseline_ttft_s=_baseline_ttft(run_dir),
        )
        for r in results:
            _append(raw_path, r.row())
        _append(sum_path, summ.row())
        log.info(
            "%s: n=%d ok=%d realized_rps=%.2f p95_e2e=%.2fs out_tps=%.0f kv=%.2f "
            "qwait=%s(%s) slo_ok=%s",
            stage, summ.n_requests, summ.n_ok, summ.realized_rps, summ.p95_e2e_s or -1,
            summ.output_token_tps, summ.mean_gpu_cache_usage or -1,
            summ.mean_queue_wait_s, summ.queue_wait_source, summ.slo_ok,
        )
        if sat.update(summ, cfg):
            log.info("open loop: saturation stop after rps%s -- %s", rate, sat.stop_reason)
            break
    return sat


def analyze(run_dir: Path, cfg: CapacityConfig) -> dict:
    stage_rows = load_jsonl(run_dir / "stage_summaries.jsonl")
    if not stage_rows:
        log.warning("no stage summaries in %s; nothing to analyze", run_dir)
        return {}
    write_conditions_csv(stage_rows, run_dir / "conditions.csv")

    pf = {}
    pf_path = run_dir / "preflight.json"
    if pf_path.exists():
        pf = json.loads(pf_path.read_text())
    prov_path = run_dir / "workload_provenance.json"
    provenance = json.loads(prov_path.read_text()) if prov_path.exists() else {}

    sat_path = run_dir / "saturation.json"
    saturation = json.loads(sat_path.read_text()) if sat_path.exists() else {}

    envelope = build_envelope(
        stage_rows, slo=cfg.slo.as_dict(), saturation=saturation,
        max_num_seqs=pf.get("max_num_seqs"), provenance=provenance,
        slo_sensitivity_p95_e2e_s=cfg.slo_sensitivity_p95_e2e_s,
    )
    (run_dir / "capacity_envelope.json").write_text(json.dumps(envelope, indent=2))
    made = save_all(stage_rows, envelope, run_dir, cfg.slo.as_dict())
    _write_envelope_md(run_dir, envelope, stage_rows)
    log.info("wrote capacity_envelope.json + %d plots", len(made))
    log.info(
        "ENVELOPE: max_conc_under_slo=%s max_rps_under_slo=%s knee_c=%s "
        "peak_out_tps=%s limiting=%s slo_sensitivity=%s",
        envelope["max_concurrency_under_slo"], envelope["max_offered_rps_under_slo"],
        envelope["throughput_knee_concurrency"], envelope["peak_realized_output_tps"],
        envelope["limiting_resource"],
        {v["p95_e2e_s"]: v["max_concurrency_under_slo"] for v in envelope["slo_sensitivity"]},
    )
    return envelope


def _write_envelope_md(run_dir: Path, env: dict, stage_rows: list[dict]) -> None:
    lines = [
        "# 27B edge capacity envelope", "",
        f"- SLO: {env['slo']}",
        f"- Max concurrency under SLO: **{env['max_concurrency_under_slo']}**",
        f"- Max offered arrival rate under SLO: **{env['max_offered_rps_under_slo']} req/s** "
        f"(realized {env['max_realized_rps_under_slo']} req/s)",
        f"- Throughput knee: concurrency **{env['throughput_knee_concurrency']}**, "
        f"peak realized **{env['peak_realized_output_tps']} output tok/s** "
        f"(input {env['peak_realized_input_tps']}, total {env['peak_realized_total_tps']} tok/s)",
        f"- Limiting resource: **{env['limiting_resource']}**",
        f"- GPU-active-equiv cost: {env['gpu_active_equiv_s_per_request_at_c1']} s/req at c=1 "
        f"-> {env['gpu_active_equiv_s_per_request_at_knee']} s/req at the knee",
        f"- Queue wait signal: **{env['queue_wait_source']}**, "
        f"max observed mean {env['max_mean_queue_wait_s']} s",
        f"- input_tokens usage coverage (min across stages): {env['input_token_coverage_min']}",
        f"- Saturation stop (open-loop ramp): {env['saturation'].get('stop_reason') or 'not reached'}",
        f"- Closed-loop saturation onset: "
        f"{(env['saturation'].get('closed_loop_saturation_onset') or {}).get('reason') or 'none'}",
        "", f"- {env['cost_metric_note']}", "",
        "## SLO sensitivity (p95 e2e cutoff)", "",
        "| cutoff s | label | max concurrency | max offered req/s | peak out tok/s |",
        "|---|---|---|---|---|",
        *[
            f"| {v['p95_e2e_s']} | {v['label']} | {v['max_concurrency_under_slo']} | "
            f"{v['max_offered_rps_under_slo']} | {v.get('peak_output_tps_under_slo')} |"
            for v in env.get("slo_sensitivity", [])
        ],
        "", "## Stages", "",
        "| stage | offered | n | ok | p95 e2e s | p95 ttft s | in tok/s | out tok/s | "
        "total tok/s | qwait s | KV | SLO |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in stage_rows:
        lines.append(
            f"| {r['stage']} | {r['offered_load']} | {r['n_requests']} | {r['n_ok']} | "
            f"{r.get('p95_e2e_s')} | {r.get('p95_ttft_s')} | {r.get('input_token_tps')} | "
            f"{r.get('output_token_tps')} | {r.get('total_token_tps')} | "
            f"{r.get('mean_queue_wait_s')} | {r.get('mean_gpu_cache_usage')} | "
            f"{'ok' if r['slo_ok'] else 'BREACH'} |"
        )
    (run_dir / "capacity_envelope.md").write_text("\n".join(lines) + "\n")


async def _main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = EXPERIMENT

    if args.analysis_only:
        run_dir = Path(args.resume_run_dir)
        analyze(run_dir, cfg)
        return 0

    if args.preflight and not args.smoke and not args.full:
        info = await preflight(cfg)
        print(json.dumps(info, indent=2))
        return 0

    run_dir = Path(args.resume_run_dir) if args.resume_run_dir else new_run_dir()
    log.info("run dir: %s", run_dir)

    workload, provenance = load_workload(cfg.dataset_gz, cfg.workload_size, cfg.workload_seed)
    (run_dir / "workload_provenance.json").write_text(json.dumps(provenance, indent=2))
    (run_dir / "manifest.json").write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()
                   if not callable(v) and not k.startswith("_")},
        "slo": cfg.slo.as_dict(),
        "expected_max_requests": expected_max_requests(cfg),
    }, indent=2, default=lambda o: getattr(o, "__dict__", str(o))))

    info = await preflight(cfg)
    (run_dir / "preflight.json").write_text(json.dumps(info, indent=2))

    if args.smoke:
        smoke_cfg = _smoke_cfg(cfg)
        await _run_closed_loop(smoke_cfg, run_dir, workload, SaturationState())
        analyze(run_dir, smoke_cfg)
        log.info("smoke complete")
        return 0

    # closed loop runs the WHOLE ladder unconditionally; `onset` only records
    # where saturation first shows up
    onset = SaturationState()
    onset_at = await _run_closed_loop(cfg, run_dir, workload, onset)
    # open loop is the arm the saturation predicate actually stops
    open_sat = await _run_open_loop(cfg, run_dir, workload, onset)
    (run_dir / "saturation.json").write_text(json.dumps({
        "open_loop": open_sat.__dict__,
        "closed_loop_saturation_onset": {
            "detected": onset.stopped,
            "reason": onset.stop_reason,
            "at_concurrency": onset_at,
        },
        "stopped": open_sat.stopped,
        "stop_reason": open_sat.stop_reason,
    }, indent=2))
    analyze(run_dir, cfg)
    return 0


def _smoke_cfg(cfg: CapacityConfig) -> CapacityConfig:
    import dataclasses
    return dataclasses.replace(
        cfg,
        concurrency_ladder=(1, 2),
        closed_loop_reps=1,
        closed_loop_cell_multiple=4,
        closed_loop_cell_min_requests=6,
        workload_size=min(cfg.workload_size, 20),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--smoke", action="store_true", help="closed loop c=1,2 only, then analyze")
    p.add_argument("--full", action="store_true", help="full closed ladder + open-loop ramp")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--resume-run-dir", type=str, default=None)
    return p.parse_args()


def main() -> int:
    return asyncio.run(_main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
