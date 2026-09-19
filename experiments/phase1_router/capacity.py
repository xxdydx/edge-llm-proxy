"""Capacity gate + joint quality/capacity controller for the Phase 1 router.

Step 3 of the binding execution order in
``claude-memory/wiki/projects/Continuously learned edge-cloud routing project.md``:
the quality gate (``policy.fit_logreg`` + the epsilon threshold picked in
``analysis``) decides whether local is *safe*; this module decides whether the
local 27B box has *room*.

The 300-pair MBPP+ dataset has no arrival timestamps, so capacity is a
*simulated* offered load ``lambda`` in requests/second. For each ``lambda`` we
stream the problems in selection order and admit a request locally only if

    (a) the quality gate already marked it local-eligible, AND
    (b) admitting it keeps the rolling count of local admissions within the
        measured SLO-safe local rate.

Sweeping ``lambda`` produces the headline curve: local load actually admitted
vs offered load, with the quality constraint and any capacity violations shown.

Budget anchor -- ``experiments/capacity_profile/results/run-20260910T135027Z``
(finding "27B edge serving capacity envelope on MBPP+ coding requests"): one
27B box sustains **~2 req/s** of local traffic under a p95-e2e <= 12 s SLO. The
*binding* constraint there was latency, not GPU saturation (KV pool < 8%, no
preemptions), so the primary gate is a direct admitted-local-rate limit at
``cfg.capacity_max_local_req_per_s``. A GPU-service-second model
(``service_seconds``) is kept as a secondary weighting -- it feeds
``local_service_share`` / ``gpu_util`` diagnostics but does not gate.

This is an OFFLINE replay over an already-run paired dataset. It does not touch
the GPU box.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .config import Phase1Config
from .policy import LearnedPolicy, oracle
from .schema import PairedExample

SvcFn = Callable[[PairedExample, Phase1Config], float]


# --- local metric helpers (kept here to avoid an analysis <-> capacity import
#     cycle; identical semantics to analysis._quality / analysis._local_share) -

def _quality(examples: list[PairedExample], route_local: list[bool]) -> float:
    if not examples:
        return 0.0
    ok = sum((ex.local_ok if lo else ex.cloud_ok) for ex, lo in zip(examples, route_local))
    return ok / len(examples)


def _local_share(route_local: list[bool]) -> float:
    return sum(route_local) / len(route_local) if route_local else 0.0


# --- per-request local GPU service time ------------------------------------

def est_output_tokens(ex: PairedExample, cfg: Phase1Config) -> int:
    """Output length used to size local GPU service time. Uses the local arm's
    reported output tokens when present (this is an offline replay, so we have
    the real generation); falls back to a fixed default for rows whose local
    generation was not OK."""
    u = ex.local_gen.usage or {}
    n = u.get("output_tokens")
    if not n or n <= 0:
        return cfg.capacity_default_output_tokens
    return int(n)


def service_seconds(ex: PairedExample, cfg: Phase1Config) -> float:
    """Predicted GPU-active-equivalent seconds to serve this request LOCALLY.

    Affine model: ``svc = floor + slope * output_tokens``.

      - ``floor`` (``cfg.capacity_svc_floor_s``, 0.10 s) is fixed prefill +
        scheduling overhead paid regardless of output length. MBPP+ prompts are
        short (~150-400 tok) so prefill is small and roughly constant.
      - ``slope`` is back-solved from the envelope anchor so a reference-length
        (256-tok) generation totals the measured ``capacity_env_svc_sec_per_req``
        (0.51 s at the c=8 knee). That gives ~0.0016 s/output-token, i.e. an
        effective ~625 GPU-second-equivalent tok/s once c=8 batching is
        amortised -- higher than the ~30 tok/s per-sequence decode, which is
        correct because the 0.51 s figure is already the batched per-request
        cost (2.93 s at c=1 / 5.7x).

    Linear-through-origin is the ``floor = 0`` special case. Pure function of
    ``ex`` + ``cfg``; the only request-derived input is the output-token count
    (known post-hoc here because this is offline replay, but a deployable
    router would use an expected-output-length estimate in its place).
    """
    floor = max(0.0, min(cfg.capacity_svc_floor_s, cfg.capacity_env_svc_sec_per_req * 0.9))
    slope = (cfg.capacity_env_svc_sec_per_req - floor) / max(1, cfg.capacity_env_ref_output_tokens)
    out = est_output_tokens(ex, cfg)
    return max(1e-6, floor + slope * out)


# --- joint route over a simulated offered load ---------------------------

@dataclass
class JointRouteResult:
    lam: float
    n: int
    enforce_capacity: bool
    n_quality_eligible: int
    n_admitted_local: int
    n_shed_by_capacity: int
    local_request_share: float
    local_service_share: float  # local svc-sec / total svc-sec over the stream (diagnostic)
    peak_window_local_rps: float  # max rolling local admissions per second
    peak_window_util: float       # peak_window_local_rps / capacity_max_local_req_per_s
    peak_window_gpu_util: float    # max rolling local svc-sec / (budget_gpu_sec_per_sec * window)
    mean_p_local_admitted: float
    mean_p_local_shed: float       # priority-inversion check: should be <= admitted

    def to_json(self) -> dict:
        return asdict(self)


def joint_route(
    examples: list[PairedExample],
    p_local: list[float] | np.ndarray,
    quality_eligible: list[bool],
    lam: float,
    cfg: Phase1Config,
    *,
    enforce_capacity: bool = True,
    svc_fn: SvcFn = service_seconds,
) -> tuple[list[bool], JointRouteResult]:
    """Stream ``examples`` (selection order) at ``lam`` req/s. Admit local iff
    ``quality_eligible[i]`` AND (``not enforce_capacity`` OR the rolling count
    of local admissions in the trailing ``capacity_window_s`` stays within
    ``capacity_max_local_req_per_s * capacity_window_s``). Returns
    ``(route_local, diagnostics)``.

    ``enforce_capacity=False`` is the quality-gate-only comparator at the same
    offered load: nothing is shed, but ``peak_window_util`` still shows whether
    the SLO-safe local rate was exceeded.
    """
    n = len(examples)
    assert len(p_local) == n and len(quality_eligible) == n
    W = cfg.capacity_window_s
    rate_cap = cfg.capacity_max_local_req_per_s * W  # max local admissions per window
    gpu_budget = cfg.capacity_budget_gpu_sec_per_sec * W
    route = [False] * n
    admitted: list[tuple[float, float]] = []  # (arrival_t, svc_sec)
    shed = 0
    adm_p: list[float] = []
    shed_p: list[float] = []
    total_svc = 0.0
    for i, ex in enumerate(examples):
        t = (i / lam) if lam > 0 else 0.0
        svc = float(svc_fn(ex, cfg))
        total_svc += svc
        if not quality_eligible[i]:
            continue
        cutoff = t - W
        n_in_window = sum(1 for (a, _) in admitted if a > cutoff)
        if (not enforce_capacity) or (n_in_window + 1 <= rate_cap):
            route[i] = True
            admitted.append((t, svc))
            adm_p.append(float(p_local[i]))
        else:
            shed += 1
            shed_p.append(float(p_local[i]))

    local_svc = sum(s for _, s in admitted)
    peak_n = peak_svc = 0.0
    for a, _ in admitted:
        lo = a - W
        peak_n = max(peak_n, sum(1 for (aa, _) in admitted if lo < aa <= a))
        peak_svc = max(peak_svc, sum(s for (aa, s) in admitted if lo < aa <= a))
    peak_rps = peak_n / W
    diag = JointRouteResult(
        lam=lam,
        n=n,
        enforce_capacity=enforce_capacity,
        n_quality_eligible=sum(quality_eligible),
        n_admitted_local=sum(route),
        n_shed_by_capacity=shed,
        local_request_share=_local_share(route),
        local_service_share=(local_svc / total_svc) if total_svc else 0.0,
        peak_window_local_rps=peak_rps,
        peak_window_util=peak_rps / cfg.capacity_max_local_req_per_s if cfg.capacity_max_local_req_per_s else 0.0,
        peak_window_gpu_util=(peak_svc / gpu_budget) if gpu_budget else 0.0,
        mean_p_local_admitted=float(np.mean(adm_p)) if adm_p else 0.0,
        mean_p_local_shed=float(np.mean(shed_p)) if shed_p else 0.0,
    )
    return route, diag


# --- baselines under the same capacity budget ---------------------------

def _eligible_all(n: int) -> list[bool]:
    return [True] * n


def _eligible_oracle(examples: list[PairedExample]) -> list[bool]:
    return list(oracle(examples))  # local iff local actually passes


# --- lambda sweep -------------------------------------------------------

def joint_capacity_sweep(
    test: list[PairedExample],
    p_local: list[float] | np.ndarray,
    threshold: float,
    cfg: Phase1Config,
    q_cloud_test: float,
    *,
    svc_fn: SvcFn = service_seconds,
) -> list[dict]:
    """One row per offered load. Each row compares, at that load:
      - joint          : quality gate then capacity gate
      - quality_only   : quality gate, capacity ignored (violation still shown)
      - capacity_only  : every request local-eligible, capacity gate only
      - oracle_cap     : oracle choice (local iff local passes) under the budget
    """
    p_local = list(p_local)
    n = len(test)
    q_elig = [bool(p >= threshold) for p in p_local]
    rows: list[dict] = []
    for lam in cfg.offered_load_grid:
        variants: dict[str, tuple[list[bool], JointRouteResult]] = {
            "joint": joint_route(test, p_local, q_elig, lam, cfg,
                                 enforce_capacity=True, svc_fn=svc_fn),
            "quality_only": joint_route(test, p_local, q_elig, lam, cfg,
                                        enforce_capacity=False, svc_fn=svc_fn),
            "capacity_only": joint_route(test, p_local, _eligible_all(n), lam, cfg,
                                         enforce_capacity=True, svc_fn=svc_fn),
            "oracle_cap": joint_route(test, p_local, _eligible_oracle(test), lam, cfg,
                                      enforce_capacity=True, svc_fn=svc_fn),
        }
        row: dict = {"offered_load_req_s": lam, "threshold": threshold}
        for name, (route, diag) in variants.items():
            q = _quality(test, route)
            row[name] = {
                **diag.to_json(),
                "quality": q,
                "quality_delta_vs_cloud": q - q_cloud_test,
                "within_epsilon": bool(q >= q_cloud_test - cfg.epsilon),
                "capacity_violated": bool(diag.peak_window_util > 1.0 + 1e-9),
            }
        rows.append(row)
    return rows


def joint_capacity_report(
    stream: list[PairedExample],
    learned: dict[str, LearnedPolicy],
    thresholds: dict[str, float],
    cfg: Phase1Config,
    q_cloud_stream: float,
    *,
    svc_fn: SvcFn = service_seconds,
    stream_label: str = "ordered_all",
) -> dict:
    """Called from analysis.run_analysis. Returns the lambda sweep per learned
    policy. Runs over the full selection-order stream (default) so a per-second
    rate limit has enough requests to bind; quality numbers on that stream
    include train/val rows -- the object of study here is capacity BEHAVIOUR,
    not a clean held-out quality number (see the test-split arms for that)."""
    out: dict = {
        "stream": stream_label,
        "n_stream": len(stream),
        "max_local_req_per_s": cfg.capacity_max_local_req_per_s,
        "window_s": cfg.capacity_window_s,
        "offered_load_grid": list(cfg.offered_load_grid),
        "env_anchor": {
            "slo_safe_local_req_per_s": cfg.capacity_max_local_req_per_s,
            "svc_sec_per_req_256tok": cfg.capacity_env_svc_sec_per_req,
            "source": "capacity_profile/results/run-20260910T135027Z",
        },
        "note": (
            "offline simulation over the already-run paired dataset (no GPU). "
            "primary gate: rolling local-admission rate <= "
            f"{cfg.capacity_max_local_req_per_s} req/s (envelope SLO-safe rate, "
            "latency-bound). gpu_util fields are a secondary GPU-second "
            "diagnostic and do not gate. quality on this stream includes "
            "train/val rows."
        ),
    }
    try:
        for pname, pol in learned.items():
            p_local = pol.proba_local(stream)
            out[pname] = {
                "threshold": thresholds[pname],
                "sweep": joint_capacity_sweep(
                    stream, p_local, thresholds[pname], cfg, q_cloud_stream, svc_fn=svc_fn
                ),
            }
    except NotImplementedError as e:
        out["status"] = "pending_service_seconds_impl"
        out["detail"] = str(e)
    return out
