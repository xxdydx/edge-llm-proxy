"""Background vLLM ``/metrics`` scraper.

Runs as an asyncio task for the lifetime of a load stage, appending one JSON
object per poll to ``metrics_timeseries.jsonl``. The gauges we care about:

  vllm:num_requests_running   - sequences actively decoding
  vllm:num_requests_waiting   - sequences queued (head-of-line for a KV slot)
  vllm:gpu_cache_usage_perc   - fraction of the KV block pool in use (0..1)
  vllm:prefix_cache_queries_total / _hits_total - cumulative, for reuse deltas

``num_requests_running > 0`` is the GPU-active signal used later to express
cost as GPU-active-equivalent seconds per request (vLLM exposes no exact
per-request GPU-time attribution).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

_METRIC_LINE = re.compile(r"^(?P<name>[^\s{]+)(?:\{[^}]*\})?\s+(?P<value>\S+)$")

_GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    # KV pool utilisation. Older vLLM exposes ``gpu_cache_usage_perc``; this
    # hybrid-attention fork (v0.0.0) exposes ``kv_cache_usage_perc``. Capture
    # both and normalise to ``gpu_cache_usage_perc`` in parse_metrics so the
    # downstream KV-saturation rule / plot have a signal either way.
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_prefix_cache_hit_rate",
)
_COUNTERS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:request_success_total",
    # queue-time histogram: cumulative sum of seconds spent WAITING before
    # prefill, and the request count. Per-stage deltas give the true mean
    # queue wait when the server exposes this (vLLM >= 0.5). Checked live in
    # preflight; if absent, analysis falls back to the named TTFT-inflation
    # proxy (see slo.summarize_stage).
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
)

# names the analysis looks for to decide whether a real queue-time signal
# exists; exported so preflight can report presence/absence.
QUEUE_TIME_METRICS = (
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
)


def _sum_metric(text: str, name: str) -> float | None:
    total = 0.0
    found = False
    for line in text.splitlines():
        m = _METRIC_LINE.match(line)
        if m and m.group("name") == name:
            try:
                total += float(m.group("value"))
            except ValueError:
                continue
            found = True
    return total if found else None


def parse_metrics(text: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name in _GAUGES + _COUNTERS:
        out[name.replace("vllm:", "")] = _sum_metric(text, name)
    # normalise the KV-usage gauge name across vLLM builds
    if out.get("gpu_cache_usage_perc") is None and out.get("kv_cache_usage_perc") is not None:
        out["gpu_cache_usage_perc"] = out["kv_cache_usage_perc"]
    return out


class MetricsProbe:
    def __init__(self, client: httpx.AsyncClient, out_path: Path, poll_s: float, *, stage: str):
        self._client = client
        self._out_path = out_path
        self._poll_s = poll_s
        self._stage = stage
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.samples: list[dict[str, Any]] = []

    async def _loop(self) -> None:
        with self._out_path.open("a") as handle:
            while not self._stop.is_set():
                t0 = time.time()
                sample: dict[str, Any] = {"ts": t0, "stage": self._stage}
                try:
                    resp = await self._client.get("/metrics", timeout=10.0)
                    resp.raise_for_status()
                    sample.update(parse_metrics(resp.text))
                    sample["scrape_ok"] = True
                except Exception as exc:  # noqa: BLE001
                    sample["scrape_ok"] = False
                    sample["error"] = f"{type(exc).__name__}: {exc}"
                handle.write(json.dumps(sample) + "\n")
                handle.flush()
                self.samples.append(sample)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._poll_s)
                except asyncio.TimeoutError:
                    pass

    async def __aenter__(self) -> "MetricsProbe":
        self._task = asyncio.ensure_future(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
