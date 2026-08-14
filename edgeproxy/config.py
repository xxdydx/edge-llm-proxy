"""Settings, resolved from argv then environment."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

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
    # vLLM rejects any model name it does not serve, and the harness asks for
    # real Claude names. Local-routed requests get rewritten to this.
    local_model_name: str

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
        "--vllm-url",
        default=env("EDGEPROXY_VLLM_URL", "http://localhost:8001"),
        help="local backend base URL",
    )
    p.add_argument(
        "--policy",
        default=env("EDGEPROXY_POLICY", "cloud-only"),
        help="placement policy: cloud-only | local-only | static",
    )
    p.add_argument(
        "--local-model-name",
        default=env("EDGEPROXY_LOCAL_MODEL_NAME", "local"),
        help="what vLLM is served as; local-routed requests are rewritten to it",
    )
    a = p.parse_args(argv)

    return Config(
        host=a.host,
        port=a.port,
        upstream=a.upstream.rstrip("/"),
        trace_dir=Path(a.trace_dir),
        vllm_url=a.vllm_url,
        policy=a.policy,
        local_model_name=a.local_model_name,
    )
