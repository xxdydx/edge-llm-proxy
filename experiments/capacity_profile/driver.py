"""Async request driver + the two load generators.

``measure_request`` streams one Anthropic-style completion and timestamps the
first and last ``content_block_delta`` so TTFT and the decode interval are
distinct measurements (adapted from ``scripts/measure_local_throughput.py``).

``run_closed_loop_cell`` fires a fixed number of requests at a fixed
concurrency. ``run_open_loop_stage`` releases requests on a Poisson schedule at
a target arrival rate. Neither judges output correctness.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

import httpx

from .workload import WorkloadItem


@dataclass
class RequestResult:
    seq: int
    task_id: str
    mode: str  # "closed_loop" | "open_loop"
    stage: str  # e.g. "c8/rep1" or "rps4.0"
    # scheduling
    scheduled_offset_s: float | None = None  # open-loop: intended release time
    release_offset_s: float | None = None  # actual release time, stage-relative
    # outcome
    status: int | None = None
    input_tokens: int | None = None  # from message_start usage.input_tokens
    output_tokens: int = 0
    ttft_s: float | None = None
    e2e_s: float | None = None
    decode_s: float | None = None
    tpot_ms: float | None = None
    decode_tokens_per_s: float | None = None
    stop_reason: str | None = None
    model_identity: str | None = None
    error: str = ""
    # open-loop only: time spent between intended release and actual send
    schedule_lag_s: float | None = None

    def row(self) -> dict[str, Any]:
        return asdict(self)


def build_body(item: WorkloadItem, cfg) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": cfg.request_model,
        "max_tokens": cfg.max_output_tokens,
        "temperature": cfg.temperature,
        "stream": True,
        "messages": [{"role": "user", "content": item.prompt}],
    }
    if cfg.disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


async def measure_request(
    client: httpx.AsyncClient,
    item: WorkloadItem,
    cfg,
    *,
    seq: int,
    mode: str,
    stage: str,
    stage_start: float,
    scheduled_offset_s: float | None = None,
) -> RequestResult:
    res = RequestResult(
        seq=seq,
        task_id=item.task_id,
        mode=mode,
        stage=stage,
        scheduled_offset_s=scheduled_offset_s,
    )
    body = build_body(item, cfg)
    started = time.perf_counter()
    res.release_offset_s = started - stage_start
    if scheduled_offset_s is not None:
        res.schedule_lag_s = max(0.0, res.release_offset_s - scheduled_offset_s)

    first_token_at: float | None = None
    last_token_at: float | None = None
    deltas = 0
    usage_out: int | None = None
    usage_in: int | None = None

    try:
        async with asyncio.timeout(cfg.request_total_timeout_s):
            async with client.stream("POST", "/v1/messages", json=body) as response:
                res.status = response.status_code
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    ptype = payload.get("type")
                    if ptype == "content_block_delta":
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        last_token_at = now
                        deltas += 1
                    elif ptype == "message_start":
                        msg = payload.get("message") or {}
                        res.model_identity = msg.get("model")
                        usage = msg.get("usage") or {}
                        if usage.get("input_tokens") is not None:
                            usage_in = int(usage["input_tokens"])
                        if usage.get("output_tokens") is not None:
                            usage_out = int(usage["output_tokens"])
                    elif ptype in ("message_delta", "message_stop"):
                        delta = payload.get("delta") or {}
                        if delta.get("stop_reason"):
                            res.stop_reason = delta["stop_reason"]
                        usage = payload.get("usage") or {}
                        if usage.get("input_tokens") is not None:
                            usage_in = int(usage["input_tokens"])
                        if usage.get("output_tokens") is not None:
                            usage_out = int(usage["output_tokens"])
    except Exception as exc:  # noqa: BLE001 - any transport/stream fault is a datum
        res.error = f"{type(exc).__name__}: {exc}"

    finished = time.perf_counter()
    res.e2e_s = finished - started
    res.input_tokens = usage_in
    res.output_tokens = usage_out if usage_out is not None else deltas
    if first_token_at is not None:
        res.ttft_s = first_token_at - started
    if first_token_at is not None and last_token_at is not None:
        res.decode_s = max(0.0, last_token_at - first_token_at)
        decode_tokens = res.output_tokens - 1
        if res.decode_s > 0 and decode_tokens > 0:
            res.tpot_ms = res.decode_s * 1000 / decode_tokens
            res.decode_tokens_per_s = decode_tokens / res.decode_s
    if res.status == 200 and res.output_tokens == 0 and not res.error:
        res.error = "stream completed with no output tokens"
    return res


# --- closed loop -----------------------------------------------------------


async def run_closed_loop_cell(
    client: httpx.AsyncClient,
    workload: list[WorkloadItem],
    cfg,
    *,
    concurrency: int,
    n_requests: int,
    stage: str,
    seq0: int,
) -> list[RequestResult]:
    """Keep exactly ``concurrency`` requests in flight until ``n_requests`` have
    completed. This is a saturating closed-loop offered load."""
    sem = asyncio.Semaphore(concurrency)
    stage_start = time.perf_counter()
    results: list[RequestResult] = []
    counter = {"i": 0}

    async def one(idx: int) -> None:
        # Rotate across stages/repetitions. Restarting every cell at item zero
        # would repeatedly benchmark exact prompt-cache hits rather than the
        # representative workload distribution.
        item = workload[(seq0 + idx) % len(workload)]
        async with sem:
            r = await measure_request(
                client, item, cfg,
                seq=seq0 + idx, mode="closed_loop", stage=stage,
                stage_start=stage_start,
            )
        results.append(r)

    await asyncio.gather(*(one(i) for i in range(n_requests)))
    counter["i"] = n_requests
    return results


# --- open loop (Poisson arrivals) ---------------------------------------


def poisson_offsets(rate_rps: float, window_s: float, rng: random.Random) -> list[float]:
    """Exponential inter-arrival times summed into stage-relative release
    offsets, truncated at ``window_s``."""
    offsets: list[float] = []
    t = 0.0
    while True:
        t += rng.expovariate(rate_rps)
        if t >= window_s:
            break
        offsets.append(t)
    return offsets


async def run_open_loop_stage(
    client: httpx.AsyncClient,
    workload: list[WorkloadItem],
    cfg,
    *,
    rate_rps: float,
    stage: str,
    seq0: int,
    rng: random.Random,
) -> list[RequestResult]:
    """Release requests at their scheduled Poisson offsets regardless of whether
    earlier ones have returned (true open loop), then drain in-flight requests
    for up to ``open_loop_drain_s``."""
    offsets = poisson_offsets(rate_rps, cfg.open_loop_window_s, rng)
    # guarantee a floor on sample size for the slow rates
    if len(offsets) < cfg.open_loop_min_requests:
        extra = cfg.open_loop_min_requests - len(offsets)
        step = cfg.open_loop_window_s / (cfg.open_loop_min_requests + 1)
        offsets = sorted(offsets + [step * (k + 1) for k in range(extra)])
    # cap a very fast stage so it cannot run unbounded
    cap = getattr(cfg, "open_loop_max_requests_per_stage", None)
    if cap and len(offsets) > cap:
        offsets = offsets[:cap]

    stage_start = time.perf_counter()
    inflight: list[asyncio.Task] = []
    results: list[RequestResult] = []

    for i, off in enumerate(offsets):
        now = time.perf_counter() - stage_start
        if off > now:
            await asyncio.sleep(off - now)
        if len(inflight) >= cfg.open_loop_max_inflight:
            done, pending = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                results.append(d.result())
            inflight = list(pending)
        item = workload[(seq0 + i) % len(workload)]
        inflight.append(
            asyncio.ensure_future(
                measure_request(
                    client, item, cfg,
                    seq=seq0 + i, mode="open_loop", stage=stage,
                    stage_start=stage_start, scheduled_offset_s=off,
                )
            )
        )

    if inflight:
        done, pending = await asyncio.wait(
            inflight, timeout=cfg.open_loop_drain_s + cfg.request_read_inactivity_timeout_s
        )
        for d in done:
            results.append(d.result())
        for p in pending:
            p.cancel()
            results.append(
                RequestResult(
                    seq=-1, task_id="", mode="open_loop", stage=stage,
                    error="abandoned: still in flight after drain window",
                )
            )
    return results
