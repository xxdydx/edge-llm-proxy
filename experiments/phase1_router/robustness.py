"""Out-of-fold robustness audit for the step-3 joint controller.

Follows an independent Codex review (2026-09-11) of
`claude-memory/wiki/findings/Joint quality-capacity controller offline sweep on
the Phase 1 pilot.md`, which flagged that the headline sweep in `capacity.py`
is (a) partly in-sample -- it runs over `ordered_all` (TRAIN+VAL+TEST) using a
classifier fit on TRAIN/thresholded on VAL -- and (b) uses periodic arrivals
(`t = i/lambda`) with a window shortened to 10s purely because 300 rows cannot
sustain the envelope's native 60s at high `lambda`. This module addresses both,
matching the governing plan's own step 4 wording ("evaluate the joint optimizer
offline first ... on repeated group-disjoint folds"):

  - repeats the grouped train/val/test split under several seeds (same
    mechanism as `dataset.split_by_group`, applied to already-generated
    `PairedExample`s so no regeneration is needed);
  - refits the quality classifier on TRAIN and its threshold on VAL each time,
    then evaluates capacity behaviour on held-out TEST ONLY -- clean,
    out-of-fold quality, not the full-stream number;
  - uses genuine Poisson arrivals (`poisson_arrival_times`) instead of the
    periodic `t = i/lambda` in `capacity.joint_route`;
  - can exercise the real 60s window by replicating the (small) TEST stream --
    a longer synthetic arrival horizon over the same fixed task distribution,
    which tests the capacity GATE MECHANISM under sustained load, not a claim
    of new task diversity.

This is offline analysis over the already-generated Phase 1 pilot data; it
does not touch the GPU box.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import replace
from typing import Any, Callable

from .capacity import SvcFn, service_seconds
from .config import Phase1Config
from .policy import LearnedPolicy, always_cloud, fit_logreg
from .schema import PairedExample


def poisson_arrival_times(n: int, lam: float, seed: int) -> list[float]:
    """Cumulative sum of Exp(lambda) interarrival gaps -- a genuine Poisson
    process, unlike `capacity.joint_route`'s periodic `t = i/lambda`. All
    requests arrive at t=0 when lam <= 0 (matches joint_route's convention)."""
    rng = random.Random(seed)
    t = 0.0
    times: list[float] = []
    for _ in range(n):
        if lam > 0:
            t += rng.expovariate(lam)
        times.append(t)
    return times


def _quality(examples: list[PairedExample], route_local: list[bool]) -> float:
    if not examples:
        return 0.0
    ok = sum((ex.local_ok if lo else ex.cloud_ok) for ex, lo in zip(examples, route_local))
    return ok / len(examples)


def regroup_split(
    all_examples: list[PairedExample], cfg: Phase1Config, seed: int
) -> tuple[list[PairedExample], list[PairedExample], list[PairedExample]]:
    """Same grouped train/val/test mechanism as `dataset.split_by_group`
    (shuffle task groups, cut by `train_frac`/`val_frac`), applied directly to
    `PairedExample`s so already-generated data can be resplit under a fresh
    seed without regenerating anything."""
    groups = sorted({ex.problem.task_group for ex in all_examples})
    rnd = random.Random(seed)
    rnd.shuffle(groups)
    n = len(groups)
    n_tr = max(1, round(n * cfg.train_frac))
    n_va = max(1, round(n * cfg.val_frac))
    tr_g = set(groups[:n_tr])
    va_g = set(groups[n_tr : n_tr + n_va])
    te_g = set(groups[n_tr + n_va :])
    train = [e for e in all_examples if e.problem.task_group in tr_g]
    val = [e for e in all_examples if e.problem.task_group in va_g]
    test = [e for e in all_examples if e.problem.task_group in te_g]
    return train, val, test


def _best_threshold(policy, val: list[PairedExample], q_cloud_val: float, cfg: Phase1Config) -> float:
    """Duplicated from analysis._best_threshold (not imported, to avoid an
    analysis<->capacity<->robustness import cycle: analysis already imports
    capacity). Identical semantics: highest-local-share threshold among those
    whose VAL quality clears q_cloud_val - epsilon; else the highest-quality
    threshold."""
    best_ok, best_share = None, -1.0
    fallback, fb_q = 1.01, -1.0
    for thr in cfg.threshold_grid:
        route = policy.route(val, thr)
        q = _quality(val, route)
        share = sum(route) / len(route) if route else 0.0
        if q > fb_q:
            fb_q, fallback = q, thr
        if q >= q_cloud_val - cfg.epsilon and share > best_share:
            best_ok, best_share = thr, share
    return best_ok if best_ok is not None else fallback


def joint_route_poisson(
    examples: list[PairedExample],
    p_local: list[float],
    quality_eligible: list[bool],
    lam: float,
    cfg: Phase1Config,
    *,
    enforce_capacity: bool = True,
    svc_fn: SvcFn = service_seconds,
    seed: int = 0,
) -> tuple[list[bool], int]:
    """Same admission logic as `capacity.joint_route` (rolling admitted-local
    count within `cfg.capacity_window_s` capped at
    `cfg.capacity_max_local_req_per_s * window`), but with genuine Poisson
    arrivals instead of periodic `t = i/lambda`. Returns (route_local,
    n_shed_by_capacity)."""
    n = len(examples)
    assert len(p_local) == n and len(quality_eligible) == n
    times = poisson_arrival_times(n, lam, seed)
    W = cfg.capacity_window_s
    rate_cap = cfg.capacity_max_local_req_per_s * W
    route = [False] * n
    admitted: list[float] = []
    shed = 0
    for i, t in enumerate(times):
        if not quality_eligible[i]:
            continue
        cutoff = t - W
        n_in_window = sum(1 for a in admitted if a > cutoff)
        if (not enforce_capacity) or (n_in_window + 1 <= rate_cap):
            route[i] = True
            admitted.append(t)
        else:
            shed += 1
    return route, shed


def repeated_split_capacity_audit(
    all_examples: list[PairedExample],
    cfg: Phase1Config,
    *,
    fit_policy: Callable[[list[PairedExample]], LearnedPolicy | None] = fit_logreg,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    lambdas: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0),
    windows: tuple[float, ...] = (10.0, 60.0),
    replicate_for_window: dict[float, int] | None = None,
) -> list[dict[str, Any]]:
    """One row per (seed, window, lambda). For each seed: a fresh grouped
    train/val/test split, refit on TRAIN, threshold on VAL, evaluate on
    held-out TEST only (out-of-fold) with Poisson arrivals. To let the 60s
    window bind despite a small TEST split, replicate the TEST stream
    `replicate_for_window[window]` times -- see module docstring. Seeds whose
    TEST split has no local-passing example are skipped (no signal to fit a
    meaningful threshold against)."""
    replicate_for_window = replicate_for_window or {10.0: 1, 60.0: 6}
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train, val, test = regroup_split(all_examples, cfg, seed)
        if len(test) < 5 or sum(e.local_ok for e in test) == 0 or len(val) < 2:
            continue
        pol = fit_policy(train)
        if pol is None:
            continue
        q_cloud_val = _quality(val, always_cloud(val))
        thr = _best_threshold(pol, val, q_cloud_val, cfg)
        q_cloud_test = _quality(test, always_cloud(test))
        p_local_test = list(pol.proba_local(test))
        q_elig_test = [bool(p >= thr) for p in p_local_test]
        for w in windows:
            reps = replicate_for_window.get(w, 1)
            stream = test * reps
            p_local = p_local_test * reps
            elig = q_elig_test * reps
            cfg_w = replace(cfg, capacity_window_s=w)
            for lam in lambdas:
                route_j, shed_j = joint_route_poisson(
                    stream, p_local, elig, lam, cfg_w, enforce_capacity=True, seed=seed
                )
                route_q, _ = joint_route_poisson(
                    stream, p_local, elig, lam, cfg_w, enforce_capacity=False, seed=seed
                )
                q = _quality(stream, route_j)
                q_quality_only = _quality(stream, route_q)
                rows.append(
                    {
                        "policy": pol.name,
                        "seed": seed,
                        "window_s": w,
                        "replicate": reps,
                        "lambda": lam,
                        "test_groups": len({e.problem.task_group for e in test}),
                        "test_n": len(test),
                        "threshold": thr,
                        "q_cloud_test": q_cloud_test,
                        "joint_local_share": sum(route_j) / len(route_j),
                        "joint_quality": q,
                        "joint_within_epsilon": bool(q >= q_cloud_test - cfg.epsilon),
                        "joint_shed": shed_j,
                        "quality_only_local_share": sum(route_q) / len(route_q),
                        "quality_only_quality": q_quality_only,
                        "quality_only_within_epsilon": bool(
                            q_quality_only >= q_cloud_test - cfg.epsilon
                        ),
                    }
                )
    return rows


def summarize_by_window_lambda(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate `repeated_split_capacity_audit` rows across seeds: mean/std
    local share and quality, and the fraction of seeds that stayed within
    epsilon, per (window, lambda)."""
    keys = sorted({(r["window_s"], r["lambda"]) for r in rows})
    out = []
    for w, lam in keys:
        group = [r for r in rows if r["window_s"] == w and r["lambda"] == lam]
        shares = [r["joint_local_share"] for r in group]
        quals = [r["joint_quality"] for r in group]
        within = [r["joint_within_epsilon"] for r in group]
        qo_shares = [r["quality_only_local_share"] for r in group]
        qo_quals = [r["quality_only_quality"] for r in group]
        qo_within = [r["quality_only_within_epsilon"] for r in group]
        out.append(
            {
                "window_s": w,
                "lambda": lam,
                "n_seeds": len(group),
                "joint_local_share_mean": statistics.mean(shares),
                "joint_local_share_std": statistics.pstdev(shares) if len(shares) > 1 else 0.0,
                "joint_quality_mean": statistics.mean(quals),
                "joint_quality_std": statistics.pstdev(quals) if len(quals) > 1 else 0.0,
                "within_epsilon_rate": sum(within) / len(within) if within else None,
                "quality_only_local_share_mean": statistics.mean(qo_shares),
                "quality_only_local_share_std": (
                    statistics.pstdev(qo_shares) if len(qo_shares) > 1 else 0.0
                ),
                "quality_only_quality_mean": statistics.mean(qo_quals),
                "quality_only_quality_std": (
                    statistics.pstdev(qo_quals) if len(qo_quals) > 1 else 0.0
                ),
                "quality_only_within_epsilon_rate": (
                    sum(qo_within) / len(qo_within) if qo_within else None
                ),
            }
        )
    return out
