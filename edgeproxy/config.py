"""Settings, resolved from argv then environment."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from .shaping import PRESETS, LinkShaper

# Traffic goes through the Lumid pool rather than api.anthropic.com directly.
DEFAULT_UPSTREAM = "https://lum.id/claude"


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    upstream: str
    trace_dir: Path
    vllm_url: str
    policy: str
    shaping: str        # proxy | netem | none — recorded, so a latency figure
    link_preset: str    # is never ambiguous about whether it was shaped
    cloud_delay_ms: float
    cloud_jitter_ms: float
    cloud_bandwidth_mbps: float
    # vLLM rejects any model name it does not serve, and the harness asks for
    # real Claude names. Local-routed requests get rewritten to this.
    local_model_name: str
    resource_sample_interval_s: float
    gpu_index: int
    kv_bytes_per_token: int | None
    cloud_cache_tracking: str
    experiment_id: str | None = None
    episode_id: str | None = None
    cohort_tracking: str = "off"
    cohort_window_ms: float = 200.0
    local_cache_tracking: str = "off"
    max_local_tokens: int = 60_000
    local_token_margin: float = 0.9
    local_output_reserve_tokens: int = 0

    @property
    def backends(self) -> dict[str, str]:
        """Destination base URLs by placement name.

        `upstream` is the cloud path; `vllm_url` is local. With policy
        "cloud-only" this collapses to the v0 pass-through behaviour.
        """
        return {"cloud": self.upstream, "local": self.vllm_url.rstrip("/")}


def parse_args(argv: list[str] | None = None) -> Config:
    env = os.environ.get
    p = argparse.ArgumentParser(
        prog="edgeproxy",
        description="Forward Anthropic-compatible traffic upstream and record it.",
    )
    p.add_argument("--host", default=env("EDGEPROXY_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(env("EDGEPROXY_PORT", "8000")))
    p.add_argument("--upstream", default=env("EDGEPROXY_UPSTREAM", DEFAULT_UPSTREAM))
    p.add_argument("--trace-dir", default=env("EDGEPROXY_TRACE_DIR", "./traces"))
    p.add_argument(
        "--experiment-id",
        default=env("EDGEPROXY_EXPERIMENT_ID"),
        help="identifier shared by related evaluation conditions",
    )
    p.add_argument(
        "--episode-id",
        default=env("EDGEPROXY_EPISODE_ID"),
        help="identifier shared by every call in one workload condition",
    )
    p.add_argument(
        "--vllm-url",
        default=env("EDGEPROXY_VLLM_URL", "http://localhost:8001"),
        help="local backend base URL",
    )
    p.add_argument(
        "--policy",
        default=env("EDGEPROXY_POLICY", "static"),
        help="placement policy: cloud-only | local-only | static",
    )
    p.add_argument(
        "--local-model-name",
        default=env("EDGEPROXY_LOCAL_MODEL_NAME", "local"),
        help="what vLLM is served as; local-routed requests are rewritten to it",
    )
    p.add_argument(
        "--resource-sample-interval-s",
        type=float,
        default=float(env("EDGEPROXY_RESOURCE_SAMPLE_INTERVAL_S", "1")),
        help="seconds between cached GPU, RAM, and vLLM metric samples",
    )
    p.add_argument(
        "--gpu-index",
        type=int,
        default=int(env("EDGEPROXY_GPU_INDEX", "0")),
        help="NVML GPU index used by the local vLLM server",
    )
    p.add_argument(
        "--kv-bytes-per-token",
        type=int,
        default=(
            int(env("EDGEPROXY_KV_BYTES_PER_TOKEN"))
            if env("EDGEPROXY_KV_BYTES_PER_TOKEN")
            else None
        ),
        help="model KV bytes/token, used to estimate occupied KV-cache GiB",
    )
    p.add_argument(
        "--cloud-cache-tracking",
        default=env("EDGEPROXY_CLOUD_CACHE_TRACKING", "off"),
        choices=["off", "observe"],
        help="predict Anthropic prompt-cache residency without changing placement",
    )
    p.add_argument(
        "--local-cache-tracking",
        default=env("EDGEPROXY_LOCAL_CACHE_TRACKING", "off"),
        choices=["off", "observe"],
        help="probe live vLLM prefix residency before placement",
    )
    p.add_argument(
        "--cohort-tracking",
        default=env("EDGEPROXY_COHORT_TRACKING", "off"),
        choices=["off", "observe"],
        help="infer fan-out cohorts for tracing without changing placement",
    )
    p.add_argument(
        "--cohort-window-ms",
        type=float,
        default=float(env("EDGEPROXY_COHORT_WINDOW_MS", "200")),
        help="measured fan-out collection window recorded by observe mode",
    )
    p.add_argument(
        "--max-local-tokens",
        type=int,
        default=int(env("EDGEPROXY_MAX_LOCAL_TOKENS", "60000")),
        help="local model input-plus-output context limit before safety margin",
    )
    p.add_argument(
        "--local-token-margin",
        type=float,
        default=float(env("EDGEPROXY_LOCAL_TOKEN_MARGIN", "0.9")),
        help="fraction of max-local-tokens available to the static router",
    )
    p.add_argument(
        "--local-output-reserve-tokens",
        type=int,
        default=int(env("EDGEPROXY_LOCAL_OUTPUT_RESERVE_TOKENS", "0")),
        help="extra tokens left unused after the local safety-margin budget",
    )
    p.add_argument(
        "--shaping",
        default=env("EDGEPROXY_SHAPING", "proxy"),
        choices=["proxy", "netem", "none"],
        help="where cloud-path shaping happens; netem means it is applied "
        "outside this process and we only record that",
    )
    p.add_argument(
        "--link-preset",
        default=env("EDGEPROXY_LINK_PRESET", "none"),
        help=f"named scenario: {', '.join(PRESETS)}",
    )
    p.add_argument("--cloud-delay-ms", type=float, default=None, help="overrides preset")
    p.add_argument("--cloud-jitter-ms", type=float, default=None, help="overrides preset")
    p.add_argument("--cloud-bandwidth-mbps", type=float, default=None, help="overrides preset")
    a = p.parse_args(argv)
    if a.max_local_tokens <= 0:
        p.error("--max-local-tokens must be positive")
    if not 0 < a.local_token_margin <= 1:
        p.error("--local-token-margin must be greater than 0 and at most 1")
    if a.local_output_reserve_tokens < 0:
        p.error("--local-output-reserve-tokens must be non-negative")
    if a.cohort_window_ms < 0:
        p.error("--cohort-window-ms must be non-negative")

    # Preset supplies the defaults; the explicit flags win where given.
    base = LinkShaper.from_preset(a.link_preset)
    delay = base.delay_ms if a.cloud_delay_ms is None else a.cloud_delay_ms
    jitter = base.jitter_ms if a.cloud_jitter_ms is None else a.cloud_jitter_ms
    mbps = base.bandwidth_mbps if a.cloud_bandwidth_mbps is None else a.cloud_bandwidth_mbps

    return Config(
        host=a.host,
        port=a.port,
        upstream=a.upstream.rstrip("/"),
        trace_dir=Path(a.trace_dir),
        vllm_url=a.vllm_url,
        policy=a.policy,
        local_model_name=a.local_model_name,
        shaping=a.shaping,
        link_preset=a.link_preset,
        cloud_delay_ms=delay,
        cloud_jitter_ms=jitter,
        cloud_bandwidth_mbps=mbps,
        resource_sample_interval_s=max(a.resource_sample_interval_s, 0.1),
        gpu_index=a.gpu_index,
        kv_bytes_per_token=a.kv_bytes_per_token,
        cloud_cache_tracking=a.cloud_cache_tracking,
        experiment_id=a.experiment_id,
        episode_id=a.episode_id,
        cohort_tracking=a.cohort_tracking,
        cohort_window_ms=a.cohort_window_ms,
        local_cache_tracking=a.local_cache_tracking,
        max_local_tokens=a.max_local_tokens,
        local_token_margin=a.local_token_margin,
        local_output_reserve_tokens=a.local_output_reserve_tokens,
    )
