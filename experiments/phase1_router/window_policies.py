"""Step 4 windowing-strategy comparison for the rolling-window online sim.

`analysis.py::_rolling_window` implements exactly one strategy -- an
*expanding* history window that refits from scratch on every example seen so
far, with equal weight regardless of age. The project's governing plan
(`claude-memory/wiki/projects/Continuously learned edge-cloud routing
project.md`, "Sliding-window and cloud-model changes") calls for comparing
that against a fixed-length sliding window, a recency-decay-weighted fit, and
an explicit change-point reset, because a cumulative fit is unsafe once the
served cloud model changes (stale labels stay in the training set forever).

This module implements all four strategies and a synthetic regime-change
injector so they can be compared under an actual quality shift, not just on
stationary MBPP+ data (where there's no real reason for one to beat another).
It is offline analysis over already-generated `PairedExample`s; no GPU.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES
from .policy import LearnedPolicy, _Const, always_cloud, always_local, oracle
from .schema import PairedExample
from .config import Phase1Config


def _quality(examples: list[PairedExample], route_local: list[bool]) -> float:
    if not examples:
        return 0.0
    ok = sum((ex.local_ok if lo else ex.cloud_ok) for ex, lo in zip(examples, route_local))
    return ok / len(examples)


def _local_share(route_local: list[bool]) -> float:
    return sum(route_local) / len(route_local) if route_local else 0.0


def _matrix(examples: list[PairedExample]) -> np.ndarray:
    return np.array([[ex.features.get(f, 0.0) for f in FEATURE_NAMES] for ex in examples], dtype=float)


def _labels(examples: list[PairedExample]) -> np.ndarray:
    return np.array([1 if ex.local_ok else 0 for ex in examples], dtype=int)


def fit_logreg_weighted(examples: list[PairedExample], sample_weight: np.ndarray | None = None) -> LearnedPolicy:
    """Same as `policy.fit_logreg`, but accepts per-example `sample_weight`
    (used by the recency-decay strategy)."""
    from sklearn.linear_model import LogisticRegression

    X = _matrix(examples)
    y = _labels(examples)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    if len(set(y.tolist())) < 2:
        return LearnedPolicy("logreg", _Const(float(y.mean())), mean, std)
    m = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    m.fit(Xs, y, sample_weight=sample_weight)
    return LearnedPolicy("logreg", m, mean, std)


def _best_threshold(pol: LearnedPolicy, train: list[PairedExample], cfg: Phase1Config) -> float:
    """Same semantics as analysis._best_threshold (VAL-quality-constrained,
    max-local-share threshold), applied to the strategy's own training slice
    since the online sim has no separate VAL split per window."""
    q_cloud = _quality(train, always_cloud(train))
    best_ok, best_share = None, -1.0
    fallback, fb_q = 1.01, -1.0
    for thr in cfg.threshold_grid:
        route = pol.route(train, thr)
        q = _quality(train, route)
        share = _local_share(route)
        if q > fb_q:
            fb_q, fallback = q, thr
        if q >= q_cloud - cfg.epsilon and share > best_share:
            best_ok, best_share = thr, share
    return best_ok if best_ok is not None else fallback


def inject_synthetic_regime_change(
    ordered_all: list[PairedExample], from_window: int, flip_rate: float, seed: int
) -> list[PairedExample]:
    """Simulate a cloud model getting WORSE starting at `from_window`: flip a
    `flip_rate` fraction of `cloud_ok=True` examples in windows >= from_window
    to False. This creates a genuine, injectable regime change (the project's
    own "inject a Cloud-A -> Cloud-B switch" idea) without needing a live
    second cloud backend -- a legitimate stand-in for testing the WINDOWING
    MECHANISM's recovery behaviour, not a claim about any real cloud model."""
    import dataclasses as dc
    import random

    rng = random.Random(seed)
    out = []
    for ex in ordered_all:
        if ex.problem.window >= from_window and ex.cloud_ok and rng.random() < flip_rate:
            out.append(dc.replace(ex, cloud_ok=False))
        else:
            out.append(ex)
    return out


def _group_by_window(ordered_all: list[PairedExample]) -> dict[int, list[PairedExample]]:
    by_win: dict[int, list[PairedExample]] = {}
    for ex in ordered_all:
        by_win.setdefault(ex.problem.window, []).append(ex)
    return by_win


def run_window_strategy(
    ordered_all: list[PairedExample],
    cfg: Phase1Config,
    strategy: str,
    *,
    sliding_k: int = 2,
    decay: float = 0.5,
    change_threshold: float = 0.10,
) -> list[dict]:
    """Run one windowing strategy over the chronological windows and report
    per-window routed quality/local-share/epsilon-compliance, mirroring
    analysis._rolling_window's row shape.

    strategy:
      - "expanding": fit on ALL windows seen so far (analysis.py's existing
        baseline, reimplemented here for a uniform comparison harness).
      - "sliding": fit on only the last `sliding_k` windows.
      - "recency_decay": fit on all seen, weighting window `w` by
        `decay ** (current_window - w)` so older evidence fades but is never
        fully dropped.
      - "change_point_reset": like "expanding", but if this window's
        always_cloud quality has moved more than `change_threshold` from the
        previous window's (a stand-in change detector), refit using ONLY the
        current + most recent window instead of full history.

    `change_threshold` default (0.10) is evidence-tuned on the real 300-pair
    run: 0.05 false-triggers on ordinary window-to-window noise under NO
    injected drift (2/5 epsilon-compliant windows, one spurious reset vs 3/5
    with no reset at all); 0.10-0.20 removes the false triggers (3/5,
    matching "expanding"/"sliding") while still detecting and fully
    responding to an injected cloud-quality collapse (2 resets, full local
    share at the drift window) at 0.10-0.15.
    """
    if strategy not in ("expanding", "sliding", "recency_decay", "change_point_reset"):
        raise ValueError(f"unknown strategy: {strategy}")
    by_win = _group_by_window(ordered_all)
    windows = sorted(by_win)
    rows: list[dict] = []
    seen: list[PairedExample] = []
    prev_q_cloud: float | None = None
    resets = 0
    for w in windows:
        cur = by_win[w]
        q_cloud_cur = _quality(cur, always_cloud(cur))
        if not seen or len({e.local_ok for e in seen}) < 2:
            route = always_cloud(cur)
            policy_name = "uninformed(always_cloud)"
        else:
            if strategy == "expanding":
                train = seen
                weight = None
            elif strategy == "sliding":
                keep_windows = set(range(max(0, w - sliding_k), w))
                train = [e for e in seen if e.problem.window in keep_windows] or seen
                weight = None
            elif strategy == "recency_decay":
                train = seen
                weight = np.array([decay ** (w - e.problem.window) for e in seen])
            else:  # change_point_reset
                changed = prev_q_cloud is not None and abs(q_cloud_cur - prev_q_cloud) > change_threshold
                if changed:
                    resets += 1
                    keep_windows = {w - 1, w} if w - 1 in by_win else {w}
                    train = [e for e in seen if e.problem.window in keep_windows] or seen
                else:
                    train = seen
                weight = None
            pol = fit_logreg_weighted(train, sample_weight=weight)
            thr = _best_threshold(pol, train, cfg)
            route = pol.route(cur, thr)
            policy_name = f"{strategy}(n_train={len(train)})"
        q = _quality(cur, route)
        rows.append(
            {
                "window": w,
                "strategy": strategy,
                "n": len(cur),
                "policy": policy_name,
                "routed_quality": q,
                "routed_local_share": _local_share(route),
                "q_always_cloud": q_cloud_cur,
                "q_always_local": _quality(cur, always_local(cur)),
                "q_oracle": _quality(cur, oracle(cur)),
                "within_epsilon": bool(q >= q_cloud_cur - cfg.epsilon),
                "resets_so_far": resets,
            }
        )
        prev_q_cloud = q_cloud_cur
        seen = seen + cur
    return rows


def compare_window_strategies(
    ordered_all: list[PairedExample],
    cfg: Phase1Config,
    strategies: tuple[str, ...] = ("expanding", "sliding", "recency_decay", "change_point_reset"),
    **kw,
) -> dict[str, list[dict]]:
    """One `run_window_strategy` call per strategy, same data. The headline
    comparison metric across strategies is `within_epsilon` rate and recovery
    speed after any injected regime change (see
    `inject_synthetic_regime_change`)."""
    return {s: run_window_strategy(ordered_all, cfg, s, **kw) for s in strategies}
