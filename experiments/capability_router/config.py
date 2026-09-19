"""Experiment configuration: endpoints, sample size, seed, thresholds.

Nothing here is a routing decision -- these are the offline experiment's own
knobs, kept separate from `edgeproxy/config.py` on purpose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Correction 2026-09-09: campaign *name* is not a valid exclusion criterion.
# A cloud-only trajectory's teacher action is DeepSeek's real reply,
# independent of what local model that campaign happened to compare
# elsewhere -- reading an already-saved trace file is not "inspecting or
# relaunching" any campaign. This experiment does not exclude any
# completed, passing cloud-only verdict by campaign name.
EXCLUDED_RESULT_DIRS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class BackendConfig:
    """One replay target: a real HTTP endpoint plus how to authenticate to it
    and what model id/name identifies it in both the request and a response
    we can trust as proof the right backend actually answered."""

    name: str  # "local_27b" | "cloud_deepseek"
    base_url: str
    request_model: str  # what this backend's API expects in `model`
    # Exact expected value of a genuine response's `model` field -- an
    # allowlist, not a substring check, confirmed 2026-09-09 against real
    # captured trace records (see the comment block below).
    expected_model_exact: str
    auth_header: str | None = None  # "authorization" | "x-api-key" | None
    auth_env_var: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 300.0
    apply_local_generation_controls: bool = False
    # None = no capacity limit for this backend (cloud). A local model's real
    # context window -- a request whose exact rendered prompt exceeds this is
    # INVALID_CAPACITY for that arm, never silently truncated or rerouted.
    max_context_tokens: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260909
    target_examples: int = 120
    min_task_groups: int = 12
    # Plan explicitly allows adjusting the group target down if achieving it
    # would force including calls with unscorable (UNKNOWN) labels just to
    # hit the count -- never adjust target_examples up past what real,
    # deduplicated, passing-verdict cloud-only trajectories provide.
    allow_group_shortfall: bool = True
    max_calls_per_task_group: int = 20  # caps any one group dominating the sample
    epsilon_grid: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05)
    threshold_grid: tuple[float, ...] = tuple(round(i * 0.01, 2) for i in range(101))
    # Grouped 3-way split by task_group -- disjoint groups across train/val/test.
    train_frac: float = 0.6
    val_frac: float = 0.2  # remainder goes to test
    data_dir: Path = REPO_ROOT / "experiments" / "capability_router" / "data"
    results_root: Path = REPO_ROOT / "experiments" / "capability_router" / "results"


def new_run_dir(results_root: Path | None = None) -> Path:
    """A fresh, non-overwriting, timestamped directory for one full run.
    Never reuse: every real run's raw artifacts stay on disk for review."""
    root = results_root or ExperimentConfig().results_root
    stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    run_dir = root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# Identity values below are not assumed -- confirmed 2026-09-09 against real
# captured trace records: a genuine local/vLLM response's `model` field is
# exactly "local" (the box's `--served-model-name`), and a genuine cloud
# response's `model` field is exactly "deepseek-v4-flash" (the Lumid gateway
# rewrites Claude Code's requested "claude-sonnet-5" to whatever it actually
# served). See `traces/swebench-pro-15-run-cloud-20260907T062340Z-.../` and
# `traces/swebench-pro-15-run-routing-20260908T142148Z-.../` for the records
# this was read from.
LOCAL_27B = BackendConfig(
    name="local_27b",
    base_url="http://127.0.0.1:18004",
    request_model="local",
    expected_model_exact="local",
    auth_header=None,
    auth_env_var=None,
    apply_local_generation_controls=True,
    max_context_tokens=100_000,
)

# 7B repeat: provisioned this session (task tsk-723f06a6-dc6a-42e6-ba08-8ed7ca7207c8,
# alias fmbox-qwen25-7b, per setups/qwen25-7b.env) with a local relay on
# 18002. Its real context window is 60K -- a captured request whose exact
# rendered prompt exceeds that is INVALID_CAPACITY for this arm specifically,
# never silently truncated and never silently sent to cloud as a fallback.
LOCAL_7B = BackendConfig(
    name="local_7b",
    base_url="http://127.0.0.1:18002",
    request_model="local",
    expected_model_exact="local",
    auth_header=None,
    auth_env_var=None,
    apply_local_generation_controls=True,
    max_context_tokens=60_000,
)

CLOUD_DEEPSEEK = BackendConfig(
    name="cloud_deepseek",
    base_url=os.environ.get("EDGEPROXY_UPSTREAM", "https://lum.id/claude"),
    # 2026-09-15 correction: previously left empty deliberately, to reuse the
    # recorded historical request's own `model` field ("claude-sonnet-5")
    # verbatim -- the reasoning was that this is what made the teacher
    # trajectory route to DeepSeek live, so replaying it unchanged would be a
    # like-for-like comparison. That reasoning missed a real mechanism
    # difference: requesting the POOLED Claude model id (`claude-sonnet-5`)
    # draws from the gateway's shared, admin+-gated Claude quota pool, while
    # requesting the NATIVE model id (`deepseek-v4-flash`) directly needs no
    # pool quota and is open to every role. Both ultimately get answered by
    # DeepSeek (confirmed via response identity either way), so rewriting to
    # the native id changes only which quota mechanism is used, not which
    # model answers -- and it does so while making a real, reproduced
    # pool-saturation 429 (documented 2026-09-15: 40+ identical failures
    # across 25 minutes, a fresh API token made no difference, confirming
    # it was pool-level not account-level) disappear immediately in a direct
    # test. Verified: `curl .../v1/messages -d '{"model":"deepseek-v4-flash",...}'`
    # -> HTTP 200, real DeepSeek reply, when the same request with
    # `"model":"claude-sonnet-5"` was still 429ing.
    request_model="deepseek-v4-flash",
    expected_model_exact="deepseek-v4-flash",
    auth_header="authorization",
    auth_env_var="ANTHROPIC_AUTH_TOKEN",
)

# Zero-genuine-local-contamination gate: a cloud reply whose model identity
# is anything other than the exact expected cloud identity -- local-flavored
# or otherwise -- must never be silently accepted as a real cloud response.
# This project has twice found a "cloud" condition silently answered by the
# wrong backend (see wiki: "Live Anthropic cache smoke was routed to
# DeepSeek", "Medium-effort validation ran entirely on cloud, not local").
# Because identity is now an exact allowlist rather than a substring check,
# the contamination gate falls out of the identity check itself -- any
# `model` value other than exactly "deepseek-v4-flash" already fails.

EXPERIMENT = ExperimentConfig()
