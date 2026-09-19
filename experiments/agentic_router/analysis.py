"""Stage 1 checkpoint: does a call-level classifier fit on real agentic
trajectories beat MBPP's 47.5% epsilon-compliance baseline
(`experiments/phase1_router/robustness.py`)?

Honest limitation, stated up front rather than glossed over: MBPP had an
exact per-task pass/fail label for both backends on every example, so
epsilon-compliance meant something precise (routed quality vs. cloud
quality on the same held-out tasks). Agentic trajectories don't have that
-- re-simulating a full multi-turn trajectory under a different per-call
routing choice is not something this offline pass can do. What IS available
without spend: every call already carries its trajectory's real Docker-graded
final outcome (`AgentTrajectory.task_passed`). This module trains a
call-level classifier under DISTANT SUPERVISION -- every call in a passing
trajectory is a weak positive, every call in a failing one a weak negative
-- and reports its held-out discrimination (AUC, precision/recall) under the
same grouped train/val/test split methodology `phase1_router.robustness`
uses (group-disjoint by task_group, repeated across seeds). This is a
necessarily coarser signal than MBPP's exact per-task labels, not a
literal apples-to-apples replacement for it -- treat a strong result here as
"worth building the real per-call replay-based label next," not as proof
the routing problem is solved.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from edgeproxy.router import extract_features as live_extract_features

from .schema import AgentCallRecord, AgentTrajectory

FEATURE_NAMES = (
    "turn_index",
    "local_prompt_tokens",
    "n_available_tools",
    "errored_tool_result_density",
    "branch_turn_ordinal",
    "context_utilization_ratio",
    "recent_tool_error_count",
    "recent_test_failure_count",
    "consecutive_same_backend_turns",
    "prior_response_truncated_or_invalid",
    "repair_loop_flag",
    # Observed switch cost depends on the current placement; it is descriptive
    # telemetry, never an input to a counterfactual local-harm prediction.
    # "placement_is_local" REMOVED 2026-09-16 -- data leakage. It's known only
    # AFTER the routing decision this model is meant to predict safety for, so
    # including it let the model partly learn "which calls the live router
    # already trusted," not "which calls are actually safe." See the audit
    # note under wiki/findings for the corrected AUC with this removed.
)

# Sentinel for genuinely-missing numeric features (e.g. context_utilization_ratio
# when local_token_budget wasn't recorded for that call) -- distinct from a real
# zero, never silently coerced to 0.0.
_MISSING = -1.0


def call_features(call: AgentCallRecord) -> dict[str, float]:
    d = call.derived
    # Historical traces sometimes omitted the request-time feature snapshot.
    # Recompute from the saved request with the same extractor used live.
    live = live_extract_features(call.request)
    prompt_tokens = call.router_features.get("local_prompt_tokens")
    offered_tools = call.request.get("tools") if isinstance(call.request, dict) else None
    return {
        "turn_index": float(call.turn_index),
        "local_prompt_tokens": float(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else _MISSING,
        "n_available_tools": float(len(offered_tools)) if isinstance(offered_tools, list) else _MISSING,
        "errored_tool_result_density": float(live.errored_tool_result_density),
        "branch_turn_ordinal": float(live.branch_turn_ordinal) if live.branch_turn_ordinal is not None else _MISSING,
        "context_utilization_ratio": (
            float(d.get("context_utilization_ratio"))
            if d.get("context_utilization_ratio") is not None else _MISSING
        ),
        "recent_tool_error_count": float(d.get("recent_tool_error_count") or 0),
        "recent_test_failure_count": float(d.get("recent_test_failure_count") or 0),
        "consecutive_same_backend_turns": float(d.get("consecutive_same_backend_turns") or 0),
        "prior_response_truncated_or_invalid": float(bool(d.get("prior_response_truncated_or_invalid"))),
        "repair_loop_flag": float(bool(d.get("repair_loop_flag"))),
    }


@dataclass
class CallRow:
    features: dict[str, float]
    label: int  # trajectory.task_passed as a weak per-call label; 0/1
    task_group: str


def build_rows(trajectories: list[AgentTrajectory]) -> list[CallRow]:
    rows: list[CallRow] = []
    for traj in trajectories:
        if traj.task_passed is None:
            continue  # verdict unreadable -- excluded, not coerced to a label
        for call in traj.calls:
            rows.append(CallRow(call_features(call), int(traj.task_passed), traj.task_group))
    return rows


def _matrix(rows: list[CallRow]):
    import numpy as np
    return np.array([[r.features[f] for f in FEATURE_NAMES] for r in rows], dtype=float)


def group_split(rows: list[CallRow], seed: int, train_frac=0.6, val_frac=0.2):
    groups = sorted({r.task_group for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac))
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])
    test_g = set(groups[n_train + n_val:]) or set(groups[-1:])
    train = [r for r in rows if r.task_group in train_g]
    val = [r for r in rows if r.task_group in val_g]
    test = [r for r in rows if r.task_group in test_g]
    return train, val, test


def _fit_model(kind: str, Xtr_s, ytr):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    elif kind == "gbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                min_child_samples=5, class_weight="balanced", verbose=-1)
    else:
        raise ValueError(kind)
    model.fit(Xtr_s, ytr)
    return model


def _group_macro_auc(test: list[CallRow], yte, proba) -> float | None:
    """Unweighted mean of per-task-group AUC within the test set -- exposes
    whether the pooled (call-count-weighted) AUC is dominated by one large
    group (this dataset's worst offender is 41.8% of all calls). None if no
    test group has both classes present (macro AUC undefined)."""
    from sklearn.metrics import roc_auc_score

    by_group: dict[str, list[int]] = {}
    for i, r in enumerate(test):
        by_group.setdefault(r.task_group, []).append(i)
    group_aucs = []
    for idxs in by_group.values():
        y_g = [yte[i] for i in idxs]
        if len(set(y_g)) < 2:
            continue
        p_g = [proba[i] for i in idxs]
        group_aucs.append(roc_auc_score(y_g, p_g))
    return statistics.mean(group_aucs) if group_aucs else None


def _baselines(yte) -> dict:
    """Trivial reference points: majority-class accuracy (== base rate for
    the always-positive rule) and the 0.5 AUC a random/no-signal classifier
    would get by construction -- context for whether a fitted AUC is
    actually informative."""
    # NOTE 2026-09-16: statistics.mean() on a numpy int array silently
    # returns 0 (integer-division coercion), not the real mean -- caught by
    # a sanity check moments after first writing this with that bug. Use
    # numpy's own .mean() throughout this module for numpy inputs.
    base_rate = float(yte.mean()) if len(yte) else 0.0
    return {
        "always_positive_accuracy": max(base_rate, 1 - base_rate),
        "random_auc_reference": 0.5,
    }


def fit_and_eval(train: list[CallRow], val: list[CallRow], test: list[CallRow], kind: str = "logreg") -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    if len({r.label for r in train}) < 2 or not test or len({r.label for r in test}) < 2:
        return {"skipped": True}

    Xtr, ytr = _matrix(train), np.array([r.label for r in train])
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0)
    std[std == 0] = 1.0
    Xtr_s = (Xtr - mean) / std
    try:
        model = _fit_model(kind, Xtr_s, ytr)
    except ImportError:
        return {"skipped": True, "reason": f"{kind} unavailable"}

    Xte, yte = _matrix(test), np.array([r.label for r in test])
    Xte_s = (Xte - mean) / std
    proba = model.predict_proba(Xte_s)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "skipped": False,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "auc": float(roc_auc_score(yte, proba)),
        "group_macro_auc": _group_macro_auc(test, yte, proba),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "base_rate_test": float(yte.mean()),
        **_baselines(yte),
    }


def run_audit(trajectories: list[AgentTrajectory], seeds: range = range(1, 41), kind: str = "logreg") -> dict:
    rows = build_rows(trajectories)
    results = []
    for seed in seeds:
        train, val, test = group_split(rows, seed)
        r = fit_and_eval(train, val, test, kind=kind)
        r["seed"] = seed
        results.append(r)
    valid = [r for r in results if not r["skipped"]]
    macro_valid = [r for r in valid if r.get("group_macro_auc") is not None]
    summary = {
        "model": kind,
        "n_rows": len(rows),
        "n_task_groups": len({r.task_group for r in rows}),
        "n_seeds_requested": len(list(seeds)),
        "n_seeds_valid": len(valid),
        "n_seeds_with_macro_auc": len(macro_valid),
    }
    if valid:
        summary["auc_mean"] = statistics.mean(r["auc"] for r in valid)
        summary["auc_stdev"] = statistics.pstdev(r["auc"] for r in valid) if len(valid) > 1 else 0.0
        summary["precision_mean"] = statistics.mean(r["precision"] for r in valid)
        summary["recall_mean"] = statistics.mean(r["recall"] for r in valid)
        summary["base_rate_mean"] = statistics.mean(r["base_rate_test"] for r in valid)
        summary["always_positive_accuracy_mean"] = statistics.mean(r["always_positive_accuracy"] for r in valid)
    if macro_valid:
        summary["group_macro_auc_mean"] = statistics.mean(r["group_macro_auc"] for r in macro_valid)
        summary["group_macro_auc_stdev"] = (
            statistics.pstdev(r["group_macro_auc"] for r in macro_valid) if len(macro_valid) > 1 else 0.0
        )
    return {"summary": summary, "per_seed": results}
