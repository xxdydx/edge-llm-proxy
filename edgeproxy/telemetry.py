"""Cached local-resource telemetry for traces.

Sampling is deliberately out of band: routing a request only copies the most
recent snapshot and never waits for NVML or vLLM's metrics endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import suppress
from typing import Any

import httpx

log = logging.getLogger("edgeproxy.telemetry")

GIB = 1024**3
METRIC_RE = re.compile(r'^([^\s{]+)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$')
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"(?:,|$)')

CLOCK_EVENT_REASONS = (
    (
        "gpu_idle",
        ("nvmlClocksEventReasonGpuIdle", "nvmlClocksThrottleReasonGpuIdle"),
    ),
    (
        "applications_clocks_setting",
        (
            "nvmlClocksEventReasonApplicationsClocksSetting",
            "nvmlClocksThrottleReasonApplicationsClocksSetting",
        ),
    ),
    (
        "software_power_cap",
        ("nvmlClocksEventReasonSwPowerCap", "nvmlClocksThrottleReasonSwPowerCap"),
    ),
    (
        "hardware_slowdown",
        ("nvmlClocksEventReasonHwSlowdown", "nvmlClocksThrottleReasonHwSlowdown"),
    ),
    (
        "sync_boost",
        ("nvmlClocksEventReasonSyncBoost", "nvmlClocksThrottleReasonSyncBoost"),
    ),
    (
        "software_thermal_slowdown",
        (
            "nvmlClocksEventReasonSwThermalSlowdown",
            "nvmlClocksThrottleReasonSwThermalSlowdown",
        ),
    ),
    (
        "hardware_thermal_slowdown",
        (
            "nvmlClocksEventReasonHwThermalSlowdown",
            "nvmlClocksThrottleReasonHwThermalSlowdown",
        ),
    ),
    (
        "hardware_power_brake_slowdown",
        (
            "nvmlClocksEventReasonHwPowerBrakeSlowdown",
            "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
        ),
    ),
    (
        "display_clock_setting",
        (
            "nvmlClocksEventReasonDisplayClockSetting",
            "nvmlClocksThrottleReasonDisplayClockSetting",
        ),
    ),
)


def _gib(value: int | float) -> float:
    return round(value / GIB, 3)


def _labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {
        key: bytes(value, "utf-8").decode("unicode_escape")
        for key, value in LABEL_RE.findall(raw)
    }


def parse_vllm_metrics(text: str, kv_bytes_per_token: int | None) -> dict[str, Any]:
    """Extract the small stable subset needed for placement experiments."""
    values: dict[str, float] = {}
    counter_values: dict[str, float] = {}
    cache_config: dict[str, str] = {}

    counter_names = {
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prompt_tokens_cached_total",
        # Some Prometheus clients expose the counter's base name rather than
        # its rendered ``_total`` name. Accept both without requiring a vLLM
        # version check.
        "vllm:prompt_tokens_cached",
    }

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if name == "vllm:cache_config_info":
            cache_config = _labels(raw_labels)
        elif name in {
            "vllm:kv_cache_usage_perc",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
        }:
            values[name] = value
        elif name in counter_names:
            # Counters can have one series per worker/model. Their useful
            # process-wide value is the sum, not whichever series appeared
            # last in the exposition.
            counter_values[name] = counter_values.get(name, 0.0) + value

    usage_fraction = values.get("vllm:kv_cache_usage_perc")
    size_tokens = _as_int(cache_config.get("kv_cache_size_tokens"))
    pool_bytes = _as_int(cache_config.get("kv_cache_memory_bytes"))
    if pool_bytes is None and size_tokens is not None and kv_bytes_per_token:
        pool_bytes = size_tokens * kv_bytes_per_token

    prefix_queries = counter_values.get("vllm:prefix_cache_queries_total")
    prefix_hits = counter_values.get("vllm:prefix_cache_hits_total")
    prompt_tokens_cached = counter_values.get("vllm:prompt_tokens_cached_total")
    if prompt_tokens_cached is None:
        prompt_tokens_cached = counter_values.get("vllm:prompt_tokens_cached")

    out: dict[str, Any] = {
        "kv_cache_usage_pct": (
            round(usage_fraction * 100, 2) if usage_fraction is not None else None
        ),
        "kv_cache_size_tokens": size_tokens,
        "kv_cache_used_tokens_est": (
            round(size_tokens * usage_fraction)
            if size_tokens is not None and usage_fraction is not None
            else None
        ),
        "kv_cache_pool_gib_est": _gib(pool_bytes) if pool_bytes is not None else None,
        "kv_cache_used_gib_est": (
            _gib(pool_bytes * usage_fraction)
            if pool_bytes is not None and usage_fraction is not None
            else None
        ),
        "kv_bytes_per_token_est": kv_bytes_per_token,
        "num_gpu_blocks": _as_int(cache_config.get("num_gpu_blocks")),
        "block_size": _as_int(cache_config.get("block_size")),
        "cache_dtype": cache_config.get("cache_dtype"),
        "requests_running": _as_number(values.get("vllm:num_requests_running")),
        "requests_waiting": _as_number(values.get("vllm:num_requests_waiting")),
        "prefix_cache_queries_total": _as_number(prefix_queries),
        "prefix_cache_hits_total": _as_number(prefix_hits),
        "prefix_cache_hit_fraction_lifetime": (
            round(prefix_hits / prefix_queries, 6)
            if prefix_queries and prefix_hits is not None
            else None
        ),
        "prompt_tokens_cached_total": _as_number(prompt_tokens_cached),
    }
    return out


def _as_int(value: Any) -> int | None:
    if value is None or value == "None":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_number(value: float | None) -> int | float | None:
    if value is None:
        return None
    return int(value) if value.is_integer() else value


def read_host_ram() -> dict[str, Any]:
    fields: dict[str, int] = {}
    with open("/proc/meminfo", encoding="ascii") as meminfo:
        for line in meminfo:
            name, raw = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable"}:
                fields[name] = int(raw.split()[0]) * 1024
    total = fields["MemTotal"]
    available = fields["MemAvailable"]
    used = total - available
    return {
        "total_gib": _gib(total),
        "used_gib": _gib(used),
        "available_gib": _gib(available),
        "used_pct": round(used / total * 100, 2),
    }


def _nvml_value(nvml: Any, function_names: str | tuple[str, ...], *args: Any) -> Any:
    """Read one optional NVML value without dropping the rest of the snapshot."""
    names = (function_names,) if isinstance(function_names, str) else function_names
    for name in names:
        function = getattr(nvml, name, None)
        if function is None:
            continue
        try:
            return function(*args)
        except Exception:
            continue
    return None


def _nvml_constant(nvml: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = getattr(nvml, name, None)
        if value is not None:
            return int(value)
    return None


def _clock_event_reasons(nvml: Any, mask: int | None) -> list[str] | None:
    if mask is None:
        return None
    reasons: list[str] = []
    known_mask = 0
    for label, constant_names in CLOCK_EVENT_REASONS:
        flag = _nvml_constant(nvml, constant_names)
        if flag is None:
            continue
        known_mask |= flag
        if mask & flag:
            reasons.append(label)
    unknown_mask = mask & ~known_mask
    if unknown_mask:
        reasons.append(f"unknown_0x{unknown_mask:x}")
    return reasons


def _gpu_snapshot(nvml: Any, gpu: Any, gpu_index: int) -> dict[str, Any]:
    """Return stable, best-effort NVML fields for one GPU."""
    info = _nvml_value(nvml, "nvmlDeviceGetMemoryInfo", gpu)
    name = _nvml_value(nvml, "nvmlDeviceGetName", gpu)
    if isinstance(name, bytes):
        name = name.decode(errors="replace")

    utilization = _nvml_value(nvml, "nvmlDeviceGetUtilizationRates", gpu)
    graphics_clock = _nvml_value(
        nvml,
        "nvmlDeviceGetClockInfo",
        gpu,
        getattr(nvml, "NVML_CLOCK_GRAPHICS", None),
    )
    sm_clock = _nvml_value(
        nvml,
        "nvmlDeviceGetClockInfo",
        gpu,
        getattr(nvml, "NVML_CLOCK_SM", None),
    )
    memory_clock = _nvml_value(
        nvml,
        "nvmlDeviceGetClockInfo",
        gpu,
        getattr(nvml, "NVML_CLOCK_MEM", None),
    )
    power_draw_mw = _nvml_value(nvml, "nvmlDeviceGetPowerUsage", gpu)
    power_limit_mw = _nvml_value(
        nvml,
        ("nvmlDeviceGetEnforcedPowerLimit", "nvmlDeviceGetPowerManagementLimit"),
        gpu,
    )
    temperature = _nvml_value(
        nvml,
        "nvmlDeviceGetTemperature",
        gpu,
        getattr(nvml, "NVML_TEMPERATURE_GPU", None),
    )
    clock_event_mask = _nvml_value(
        nvml,
        (
            "nvmlDeviceGetCurrentClocksEventReasons",
            "nvmlDeviceGetCurrentClocksThrottleReasons",
        ),
        gpu,
    )

    return {
        "index": gpu_index,
        "name": name,
        "vram_total_gib": _gib(info.total) if info is not None else None,
        "vram_used_gib": _gib(info.used) if info is not None else None,
        "vram_free_gib": _gib(info.free) if info is not None else None,
        "vram_used_pct": (
            round(info.used / info.total * 100, 2)
            if info is not None and info.total
            else None
        ),
        # NVML calls this ``gpu`` utilization. It is the percentage of the
        # sampling interval with one or more kernels executing—the ``sm``
        # column exposed by nvidia-smi—not an SM occupancy measurement.
        "sm_utilization_pct": (
            int(utilization.gpu) if utilization is not None else None
        ),
        "memory_controller_utilization_pct": (
            int(utilization.memory) if utilization is not None else None
        ),
        "graphics_clock_mhz": _as_int(graphics_clock),
        "sm_clock_mhz": _as_int(sm_clock),
        "memory_clock_mhz": _as_int(memory_clock),
        "power_draw_w": (
            round(power_draw_mw / 1000, 3) if power_draw_mw is not None else None
        ),
        "power_limit_w": (
            round(power_limit_mw / 1000, 3)
            if power_limit_mw is not None
            else None
        ),
        "temperature_c": _as_int(temperature),
        "clock_event_reasons_mask": (
            int(clock_event_mask) if clock_event_mask is not None else None
        ),
        "clock_event_reasons": _clock_event_reasons(nvml, clock_event_mask),
    }


class LocalResourceSampler:
    """Maintain one best-effort resource snapshot for local trace records."""

    def __init__(
        self,
        metrics_client: httpx.AsyncClient,
        *,
        interval_s: float,
        gpu_index: int,
        kv_bytes_per_token: int | None,
    ) -> None:
        self.client = metrics_client
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.kv_bytes_per_token = kv_bytes_per_token
        self.latest: dict[str, Any] | None = None
        self._task: asyncio.Task[None] | None = None
        self._nvml: Any = None
        self._gpu: Any = None

    def start(self) -> None:
        self._init_nvml()
        self._task = asyncio.create_task(self._run(), name="local-resource-sampler")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        if self._nvml is not None:
            with suppress(Exception):
                self._nvml.nvmlShutdown()

    def snapshot(self) -> dict[str, Any] | None:
        if self.latest is None:
            return None
        copy = {**self.latest}
        copy["age_ms"] = round((time.monotonic() - copy.pop("_sampled_mono")) * 1000, 1)
        return copy

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._nvml = pynvml
        except Exception as exc:
            log.warning("NVML unavailable; GPU telemetry disabled: %s", exc)

    async def _run(self) -> None:
        while True:
            await self._sample()
            await asyncio.sleep(self.interval_s)

    async def _sample(self) -> None:
        sampled_at = time.time()
        sample: dict[str, Any] = {
            "sampled_at": sampled_at,
            "_sampled_mono": time.monotonic(),
        }

        try:
            sample["host_ram"] = read_host_ram()
        except Exception as exc:
            sample["host_ram_error"] = repr(exc)

        if self._nvml is not None:
            try:
                sample["gpu"] = _gpu_snapshot(
                    self._nvml, self._gpu, self.gpu_index
                )
            except Exception as exc:
                sample["gpu_error"] = repr(exc)

        try:
            response = await self.client.get("/metrics", timeout=2.0)
            response.raise_for_status()
            sample["vllm"] = parse_vllm_metrics(
                response.text, self.kv_bytes_per_token
            )
        except Exception as exc:
            sample["vllm_metrics_error"] = repr(exc)

        self.latest = sample
