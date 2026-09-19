"""Capacity-profile configuration: endpoint, workload, the closed-loop ladder,
the open-loop arrival ramp, the provisional SLO, and the saturation-stop knobs.

The manifest values here are the ones reported to the PI before launch
(2026-09-10, "Manifest OK"). Anything inferred rather than given is flagged in
the field comment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PKG_DIR = Path(__file__).resolve().parent

# Reuse the exact Phase 1 prompt pool so the load distribution matches the
# routing experiment this capacity envelope is meant to bound.
PHASE1_DATASET_GZ = REPO_ROOT / "experiments" / "phase1_router" / "data" / "MbppPlus-v0.2.0.jsonl.gz"


@dataclass(frozen=True)
class SLO:
    """Provisional SLO. Phase 1 local generation ran mean 3.1 s / p95 8.3 s at
    concurrency <= 4, so these are deliberately a little looser than that."""

    p95_e2e_s: float = 12.0
    p95_ttft_s: float = 2.0
    max_error_rate: float = 0.01

    def as_dict(self) -> dict[str, float]:
        return {
            "p95_e2e_s": self.p95_e2e_s,
            "p95_ttft_s": self.p95_ttft_s,
            "max_error_rate": self.max_error_rate,
        }


@dataclass(frozen=True)
class CapacityConfig:
    # --- endpoint under test -------------------------------------------------
    base_url: str = os.environ.get("CAPPROFILE_LOCAL_URL", "http://127.0.0.1:18004")
    request_model: str = "local"
    # served-model identity guard; a mismatch aborts before any load is applied
    expected_model_exact: str = "local"
    expected_hf_model: str = "Inferact/Qwen3.8-27B-NVFP4"
    expected_max_model_len: int = 100_000

    # --- workload ----------------------------------------------------------
    dataset_gz: Path = PHASE1_DATASET_GZ
    # cycle a fixed shuffled subset so prompt variety is bounded but the real
    # length spread is preserved
    workload_size: int = 100
    workload_seed: int = 20260910
    # identical inference controls to Phase 1's local arm
    # Capacity profiling needs enough decode work to expose the throughput knee,
    # without turning each cell into a many-minute generation benchmark.
    max_output_tokens: int = 256
    temperature: float = 0.0
    disable_thinking: bool = True  # Qwen3.8 <think> trap; see phase1_router.generate

    # --- closed-loop concurrency ladder -----------------------------------
    # Required fixed ladder. Always run every rung; do not stop this arm early.
    concurrency_ladder: tuple[int, ...] = (1, 2, 4, 8)
    closed_loop_reps: int = 3
    # requests issued per (concurrency, rep) cell = clamp(concurrency * multiple,
    # min, max). The small cap bounds runtime while retaining repeated samples.
    closed_loop_cell_multiple: int = 4
    closed_loop_cell_min_requests: int = 24
    closed_loop_cell_max_requests: int = 32

    # --- open-loop Poisson arrival ramp ---------------------------------
    arrival_rates_rps: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
    open_loop_window_s: float = 60.0
    open_loop_drain_s: float = 15.0
    # Do not inject requests to meet a floor: doing so changes the offered rate.
    open_loop_min_requests: int = 0
    open_loop_max_requests_per_stage: int = 256
    # a single request may not outlive this many arrival windows before its
    # slot is abandoned (keeps a saturated stage from running forever)
    open_loop_max_inflight: int = 512

    # --- SLO + saturation stop ------------------------------------------
    slo: SLO = field(default_factory=SLO)
    # the capacity envelope is reported at these p95 e2e latency cutoffs too, so
    # it is not tied to one arbitrary number (stricter / default / looser)
    slo_sensitivity_p95_e2e_s: tuple[float, ...] = (8.0, 12.0, 20.0)
    # a stage is a "breach" if it violates the SLO; stop escalating after this
    # many consecutive breaching stages
    saturation_consecutive_breaches: int = 2
    # throughput-knee rule: if a doubling of offered load raises realized
    # tokens/s by less than this fraction, the box is saturated
    knee_min_relative_gain: float = 0.05

    # --- telemetry / timeouts ------------------------------------------
    metrics_poll_s: float = 2.0
    request_read_inactivity_timeout_s: float = 60.0
    # Absolute wall-clock deadline, including a server that keeps a stream
    # technically alive without completing it.  This prevents one bad call
    # from consuming a large fraction of the GPU lease.
    request_total_timeout_s: float = 90.0
    connect_timeout_s: float = 10.0
    stage_wall_cap_s: float = 600.0

    # --- io ------------------------------------------------------------
    results_root: Path = PKG_DIR / "results"


EXPERIMENT = CapacityConfig()


def new_run_dir(results_root: Path | None = None) -> Path:
    root = results_root or EXPERIMENT.results_root
    stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    d = root / stamp
    d.mkdir(parents=True, exist_ok=False)
    return d


def expected_max_requests(cfg: CapacityConfig = EXPERIMENT) -> dict[str, int]:
    """Rough pre-launch volume estimate for the manifest."""
    closed = 0
    for c in cfg.concurrency_ladder:
        per_cell = min(
            cfg.closed_loop_cell_max_requests,
            max(cfg.closed_loop_cell_min_requests, c * cfg.closed_loop_cell_multiple),
        )
        closed += per_cell * cfg.closed_loop_reps
    open_loop = 0
    for r in cfg.arrival_rates_rps:
        open_loop += min(
            cfg.open_loop_max_requests_per_stage,
            max(cfg.open_loop_min_requests, int(r * cfg.open_loop_window_s)),
        )
    return {
        "closed_loop_if_no_saturation_stop": closed,
        "open_loop_if_no_saturation_stop": open_loop,
        "total_if_no_saturation_stop": closed + open_loop,
        "note": "closed loop always completes 1/2/4/8; only the open-loop ramp stops at saturation",
    }
