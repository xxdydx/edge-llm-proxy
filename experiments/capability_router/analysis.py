"""Opportunity / Pareto analysis: how much local traffic can a placement
rule achieve while keeping the *overall system's* success rate within a
fixed tolerance (epsilon) of always routing to cloud?

Every rate below states its own denominator explicitly, because different
metrics here have different natural denominators (all test calls vs.
known-label calls only vs. both-success calls only) and silently mixing
them is exactly the kind of error this package's `UNKNOWN` label exists to
prevent.

Correction 2026-09-09: the routing-quality constraint used for the
epsilon-constrained optimum is Q_router -- the success rate of whichever
backend a policy actually *chose* for each call, evaluated against known
labels only -- compared against Q_cloud (AlwaysCloud's own Q_router on the
same test set). The NOT_EQUIVALENT rate among locally-routed calls alone
(this module's earlier `quality_loss` field, kept for the Pareto plot) is
not Q_router and must never be used as the epsilon constraint on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edgeproxy import router

from .config import ExperimentConfig
from .models import FEATURE_NAMES, TrainedModel, feature_value
from .schema import LabeledExample

SUCCESS_LABELS = ("EQUIVALENT",)
KNOWN_LABELS = ("EQUIVALENT", "NOT_EQUIVALENT")


def _is_success(label: str) -> bool:
    return label in SUCCESS_LABELS


def _is_known(label: str) -> bool:
    return label in KNOWN_LABELS


@dataclass
class ArmResult:
    name: str
    threshold: float | None
    local_count: int
    total_count: int
    local_share: float  # local_count / total_count (over ALL examples)
    known_local_count: int  # of the locally-routed calls, how many have a known local_label
    violation_count: int  # of known_local_count, how many were NOT_EQUIVALENT
    quality_loss: float | None  # violation_count / known_local_count, or None if 0 known


@dataclass
class QRouterResult:
    """Success rate of whichever backend the policy actually chose per call,
    over calls where that chosen backend's label is known. `coverage` is the
    fraction of ALL examples that were scorable at all (excludes UNKNOWN /
    INVALID_CAPACITY / EXECUTION_ERROR on the chosen side)."""

    success_rate: float | None
    known_count: int
    total_count: int
    coverage: float


def score_arm(name: str, examples: list[LabeledExample], route_local: list[bool], threshold: float | None) -> ArmResult:
    assert len(route_local) == len(examples)
    total = len(examples)
    local_idx = [i for i, r in enumerate(route_local) if r]
    known_local = 0
    violations = 0
    for i in local_idx:
        label = examples[i].local_label
        if _is_known(label):
            known_local += 1
            if label == "NOT_EQUIVALENT":
                violations += 1
    return ArmResult(
        name=name,
        threshold=threshold,
        local_count=len(local_idx),
        total_count=total,
        local_share=len(local_idx) / total if total else 0.0,
        known_local_count=known_local,
        violation_count=violations,
        quality_loss=(violations / known_local) if known_local else None,
    )


def q_router(examples: list[LabeledExample], route_local: list[bool]) -> QRouterResult:
    """Whichever backend was chosen for each call, was its action a success?
    Excludes calls where the *chosen* backend's label is unknown/invalid/
    errored -- those cannot support a quality claim either way."""
    known = 0
    successes = 0
    for ex, local in zip(examples, route_local):
        label = ex.local_label if local else ex.cloud_label
        if _is_known(label):
            known += 1
            if _is_success(label):
                successes += 1
    total = len(examples)
    return QRouterResult(
        success_rate=(successes / known) if known else None,
        known_count=known,
        total_count=total,
        coverage=(known / total) if total else 0.0,
    )


def opportunity_2x2(examples: list[LabeledExample]) -> dict[str, int]:
    """Among calls where BOTH backends' labels are known, how many did
    local succeed at, cloud succeed at, both, or neither. Calls where
    either side is UNKNOWN/INVALID_CAPACITY/EXECUTION_ERROR are counted
    separately, not silently folded into "neither"."""
    both = local_only = cloud_only = neither = excluded = 0
    for ex in examples:
        if not (_is_known(ex.local_label) and _is_known(ex.cloud_label)):
            excluded += 1
            continue
        local_ok = _is_success(ex.local_label)
        cloud_ok = _is_success(ex.cloud_label)
        if local_ok and cloud_ok:
            both += 1
        elif local_ok and not cloud_ok:
            local_only += 1
        elif cloud_ok and not local_ok:
            cloud_only += 1
        else:
            neither += 1
    return {
        "both": both,
        "local_only": local_only,
        "cloud_only": cloud_only,
        "neither": neither,
        "excluded_unknown": excluded,
        "total": len(examples),
    }


def p_local_success_given_cloud_success(examples: list[LabeledExample]) -> float | None:
    cloud_success = [ex for ex in examples if _is_known(ex.cloud_label) and _is_success(ex.cloud_label)]
    if not cloud_success:
        return None
    local_known = [ex for ex in cloud_success if _is_known(ex.local_label)]
    if not local_known:
        return None
    return sum(1 for ex in local_known if _is_success(ex.local_label)) / len(local_known)


def router_capture(examples: list[LabeledExample], route_local: list[bool]) -> float | None:
    """(successful local routes among both-success opportunities) / (oracle
    local opportunities). "Both-success opportunities" = calls where local
    AND cloud would both have succeeded; "successful local route" = the
    policy actually chose local for one of those. "Oracle local
    opportunities" = the total count of known-local-success calls (the
    oracle's own routing set), independent of what cloud did on those
    calls -- the denominator this metric is capturing share *of*."""
    oracle_local_opportunities = sum(1 for ex in examples if _is_known(ex.local_label) and _is_success(ex.local_label))
    if oracle_local_opportunities == 0:
        return None
    captured = 0
    for ex, local in zip(examples, route_local):
        if not local:
            continue
        both_succeed = (
            _is_known(ex.local_label) and _is_success(ex.local_label)
            and _is_known(ex.cloud_label) and _is_success(ex.cloud_label)
        )
        if both_succeed:
            captured += 1
    return captured / oracle_local_opportunities


def _token_totals(examples: list[LabeledExample], route_local: list[bool]) -> dict[str, int | None]:
    """Per-arm input/output token totals, split local vs cloud.

    Input-token semantics differ by backend and must not be summed naively:

    * **local (vLLM Anthropic-native)** reports the rendered prompt in THREE
      buckets -- ``input_tokens`` (uncached), ``cache_read_input_tokens``
      (prefix-cache hits), ``cache_creation_input_tokens`` (prefix-cache
      writes). The true prompt size is their sum; ``input_tokens`` alone is
      just the uncached sliver (often <2K of a 70K prompt), so summing only
      that under-reports local input by ~50x. All three buckets are also
      surfaced separately.
    * **cloud (DeepSeek via the Lumid gateway)** returns ``input_tokens: 0``
      with no cache buckets on every call -- a provider "not reported" value,
      not a measured zero. Per this project's null-not-zero rule it is
      rendered ``None`` (unavailable) unless some call actually reported a
      positive input or cache bucket.
    """
    def usage_of(ex: LabeledExample, local: bool) -> dict:
        outcome = ex.local_outcome if local else ex.cloud_outcome
        resp = outcome.response if outcome is not None else None
        return (resp or {}).get("_usage") or {}

    l_uncached = l_cache_read = l_cache_create = local_out = 0
    c_in = cloud_out = 0
    any_local_usage = any_cloud_usage = False
    cloud_input_ever_reported = False
    for ex, local in zip(examples, route_local):
        u = usage_of(ex, local)
        if not u:
            continue
        if local:
            any_local_usage = True
            l_uncached += u.get("input_tokens") or 0
            l_cache_read += u.get("cache_read_input_tokens") or 0
            l_cache_create += u.get("cache_creation_input_tokens") or 0
            local_out += u.get("output_tokens") or 0
        else:
            any_cloud_usage = True
            ci = u.get("input_tokens") or 0
            ccr = u.get("cache_read_input_tokens") or 0
            ccc = u.get("cache_creation_input_tokens") or 0
            if ci or ccr or ccc:
                cloud_input_ever_reported = True
            c_in += ci + ccr + ccc
            cloud_out += u.get("output_tokens") or 0

    local_in_total = l_uncached + l_cache_read + l_cache_create
    return {
        # local: full rendered prompt = uncached + cache-read + cache-creation
        "local_input_tokens": local_in_total if any_local_usage else None,
        "local_input_uncached_tokens": l_uncached if any_local_usage else None,
        "local_input_cache_read_tokens": l_cache_read if any_local_usage else None,
        "local_input_cache_creation_tokens": l_cache_create if any_local_usage else None,
        "local_output_tokens": local_out if any_local_usage else None,
        # cloud: gateway reports input_tokens:0 with no cache buckets -> unavailable, not zero
        "cloud_input_tokens": (
            c_in if (any_cloud_usage and cloud_input_ever_reported) else None
        ),
        "cloud_input_tokens_note": (
            None if cloud_input_ever_reported
            else "gateway returned input_tokens:0 with no cache buckets on all calls; input unavailable, not measured zero"
        ),
        "cloud_output_tokens": cloud_out if any_cloud_usage else None,
    }


def _latency_percentiles(examples: list[LabeledExample], route_local: list[bool]) -> dict[str, float | None]:
    latencies = []
    for ex, local in zip(examples, route_local):
        outcome = ex.local_outcome if local else ex.cloud_outcome
        if outcome is not None and outcome.latency_s is not None:
            latencies.append(outcome.latency_s)
    if not latencies:
        return {"mean_latency_s": None, "p50_latency_s": None, "p95_latency_s": None}
    arr = np.array(latencies)
    return {
        "mean_latency_s": float(arr.mean()),
        "p50_latency_s": float(np.percentile(arr, 50)),
        "p95_latency_s": float(np.percentile(arr, 95)),
    }


@dataclass
class ArmFullMetrics:
    """Everything required for one routing rule's row in baseline_summary.csv."""

    name: str
    threshold: float | None
    traffic_pct_local: float
    traffic_pct_cloud: float
    local_success_count: int
    local_failure_count: int
    q_router: QRouterResult
    router_capture: float | None
    tokens: dict[str, int | str | None]  # int totals + one str note key
    latency: dict[str, float | None]
    cost: None = None  # not configured for this experiment; explicitly null


def full_metrics(name: str, examples: list[LabeledExample], route_local: list[bool], threshold: float | None) -> ArmFullMetrics:
    n = len(examples)
    n_local = sum(route_local)
    local_success = sum(
        1 for ex, local in zip(examples, route_local)
        if local and ex.local_label == "EQUIVALENT"
    )
    local_failure = sum(
        1 for ex, local in zip(examples, route_local)
        if local and ex.local_label == "NOT_EQUIVALENT"
    )
    return ArmFullMetrics(
        name=name,
        threshold=threshold,
        traffic_pct_local=(n_local / n * 100.0) if n else 0.0,
        traffic_pct_cloud=((n - n_local) / n * 100.0) if n else 0.0,
        local_success_count=local_success,
        local_failure_count=local_failure,
        q_router=q_router(examples, route_local),
        router_capture=router_capture(examples, route_local),
        tokens=_token_totals(examples, route_local),
        latency=_latency_percentiles(examples, route_local),
    )


def always_cloud_route(examples: list[LabeledExample]) -> list[bool]:
    return [False] * len(examples)


def always_local_route(examples: list[LabeledExample]) -> list[bool]:
    return [True] * len(examples)


def policy4_route(examples: list[LabeledExample]) -> list[bool]:
    """Existing deployed PlanningEscalationPolicy (Policy 4), replayed
    request-only over the same feature dict every other arm sees --
    identical inputs, no special access. Reads router.py; edits nothing."""
    pol = router.PlanningEscalationPolicy()
    route_local: list[bool] = []
    allowed = set(router.CallFeatures.__dataclass_fields__)
    for ex in examples:
        payload = {k: v for k, v in ex.features.items() if k in allowed}
        feats = router.CallFeatures(**payload)
        decision = pol.decide(feats)
        route_local.append(decision.placement == "local")
    return route_local


def oracle_route(examples: list[LabeledExample]) -> list[bool]:
    """Hindsight upper bound: route local exactly on known-EQUIVALENT calls.
    Not achievable online -- this is a ceiling, not a candidate policy."""
    return [ex.local_label == "EQUIVALENT" for ex in examples]


def threshold_route(model: TrainedModel, examples: list[LabeledExample], threshold: float) -> list[bool]:
    X = np.array([[feature_value(ex.features, n) for n in FEATURE_NAMES] for ex in examples], dtype=float)
    proba = model.predict_proba(X)
    return list(proba >= threshold)


def epsilon_constrained_threshold(
    model: TrainedModel,
    val_examples: list[LabeledExample],
    threshold_grid: tuple[float, ...],
    q_cloud: float,
    epsilon: float,
) -> float | None:
    """Selected on VALIDATION only -- never on test. Among thresholds whose
    validation Q_router is within epsilon of Q_cloud, pick the one with the
    largest local share; ties broken toward the higher threshold (more
    conservative)."""
    best_threshold = None
    best_share = -1.0
    for t in threshold_grid:
        route_local = threshold_route(model, val_examples, t)
        q = q_router(val_examples, route_local)
        if q.success_rate is None:
            continue
        if q.success_rate < q_cloud - epsilon:
            continue
        share = sum(route_local) / len(val_examples) if val_examples else 0.0
        if share > best_share or (share == best_share and (best_threshold is None or t > best_threshold)):
            best_share = share
            best_threshold = t
    return best_threshold


@dataclass
class AnalysisReport:
    train_size: int
    val_size: int
    test_size: int
    q_cloud_test: float | None
    opportunity_2x2_test: dict[str, int] = field(default_factory=dict)
    p_local_given_cloud_test: float | None = None
    baselines_test: dict[str, ArmFullMetrics] = field(default_factory=dict)
    model_test_at_epsilon: dict[str, dict[float, ArmFullMetrics | None]] = field(default_factory=dict)
    model_val_thresholds: dict[str, dict[float, float | None]] = field(default_factory=dict)
    pareto_sweeps_test: dict[str, list[ArmResult]] = field(default_factory=dict)


def run_analysis(
    train: list[LabeledExample],
    val: list[LabeledExample],
    test: list[LabeledExample],
    trained_models: dict[str, TrainedModel],
    cfg: ExperimentConfig,
) -> AnalysisReport:
    q_cloud_val = q_router(val, always_cloud_route(val)).success_rate
    q_cloud_test = q_router(test, always_cloud_route(test)).success_rate

    report = AnalysisReport(
        train_size=len(train), val_size=len(val), test_size=len(test),
        q_cloud_test=q_cloud_test,
    )
    report.opportunity_2x2_test = opportunity_2x2(test)
    report.p_local_given_cloud_test = p_local_success_given_cloud_success(test)

    for name, route_fn in (
        ("always_cloud", always_cloud_route),
        ("always_local", always_local_route),
        ("policy4", policy4_route),
        ("oracle", oracle_route),
    ):
        report.baselines_test[name] = full_metrics(name, test, route_fn(test), None)

    for model_name, model in trained_models.items():
        report.pareto_sweeps_test[model_name] = [
            score_arm(f"{model_name}@{t:.2f}", test, threshold_route(model, test, t), t)
            for t in cfg.threshold_grid
        ]
        report.model_test_at_epsilon[model_name] = {}
        report.model_val_thresholds[model_name] = {}
        for eps in cfg.epsilon_grid:
            if q_cloud_val is None:
                report.model_val_thresholds[model_name][eps] = None
                report.model_test_at_epsilon[model_name][eps] = None
                continue
            t = epsilon_constrained_threshold(model, val, cfg.threshold_grid, q_cloud_val, eps)
            report.model_val_thresholds[model_name][eps] = t
            if t is None:
                report.model_test_at_epsilon[model_name][eps] = None
            else:
                route = threshold_route(model, test, t)
                report.model_test_at_epsilon[model_name][eps] = full_metrics(
                    f"{model_name}@eps={eps}", test, route, t
                )
    return report
