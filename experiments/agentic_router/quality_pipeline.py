"""Downstream analysis pipeline over judge-labeled paired calls.

ALL RESULTS PRODUCED BY THIS MODULE ARE PROVISIONAL until the full 205-call
replay / 410-row judge snapshot completes (currently 140/205, 280/410) --
every summary dict below carries an explicit `provisional` flag and the
snapshot size it was computed from, so a downstream consumer can never
silently treat a partial-data result as final.

Pipeline, in the agreed order:
  1. Paired-order aggregation: combine each call's primary+reversed judge
     passes into one SAFE / HARM / UNKNOWN label (`aggregate_call_label`).
  2. Distribution reporting (`label_distribution`).
  3. Group-disjoint quality-model training/eval on the aggregated labels
     (`fit_quality_model`, reusing analysis.py's feature set + split
     methodology, but with a REAL per-call label now, not distant
     supervision).
  4. Temporal-safe window-update comparison: frozen / cumulative / sliding
     / EWMA (`window_strategy_comparison`).
  5. Online-policy simulation: contextual bandit (LinUCB), Thompson
     sampling, and periodic-A/B baseline (`bandit_simulation`).
  6. Cache-switching cost analysis (`cache_switch_analysis`) -- LAST, per
     explicit instruction, since it's the least mature piece (the
     underlying `estimated_recompute_cost_if_switched` feature is a
     declared heuristic, not a validated measurement -- see
     experiments/agentic_router/features.py).
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Literal

from . import analysis, features
from .schema import AgentCallRecord

Label = Literal["SAFE", "HARM", "UNKNOWN"]


def _winner(verdict: str, order: list[str] | tuple[str, str]) -> str | None:
    a_arm, b_arm = order
    if verdict == "A_BETTER":
        return a_arm
    if verdict == "B_BETTER":
        return b_arm
    return None  # EQUIVALENT / BOTH_INADEQUATE / UNCERTAIN / PARSE_ERROR


def aggregate_call_label(primary: dict[str, Any], reversed_: dict[str, Any]) -> Label:
    """One label per call from its two judge passes.

    - Both passes agree local wins, OR both say EQUIVALENT (order-invariant
      agreement that local is fine) -> SAFE.
    - Both passes agree cloud wins -> HARM (routing this call local would
      have been the worse choice).
    - Either pass is BOTH_INADEQUATE, UNCERTAIN, or PARSE_ERROR, OR the two
      passes disagree on the winner (an order-sensitive flip -- exactly
      what the reversed pass exists to catch) -> UNKNOWN. Never forced into
      SAFE/HARM.
    """
    for r in (primary, reversed_):
        if r["verdict"] in ("BOTH_INADEQUATE", "UNCERTAIN", "PARSE_ERROR"):
            return "UNKNOWN"

    p_winner = _winner(primary["verdict"], primary["order"])
    r_winner = _winner(reversed_["verdict"], reversed_["order"])

    if p_winner is None and r_winner is None:
        # Both EQUIVALENT (or one EQUIVALENT, the no-winner case only
        # happens when verdict is EQUIVALENT since the others are excluded
        # above) -> local is judged fine.
        return "SAFE"
    if p_winner == r_winner and p_winner is not None:
        return "SAFE" if p_winner == "local" else "HARM"
    # One pass found a winner, the other didn't, or they disagree on who
    # won -- an order-sensitive result, not trustworthy either way.
    return "UNKNOWN"


@dataclass
class LabeledCall:
    call_id: str
    label: Label
    primary_verdict: str
    reversed_verdict: str


def build_call_labels(judge_rows: list[dict[str, Any]]) -> list[LabeledCall]:
    """One LabeledCall per call_id present with both passes. Rows missing a
    pass are skipped (should not happen in a snapshot_clean run -- caller
    should audit first via run_judge.audit_snapshot)."""
    by_call: dict[str, dict[str, dict[str, Any]]] = {}
    for r in judge_rows:
        by_call.setdefault(r["call_id"], {})[r["pass_label"]] = r
    out = []
    for call_id, passes in by_call.items():
        if "primary" not in passes or "reversed" not in passes:
            continue
        label = aggregate_call_label(passes["primary"], passes["reversed"])
        out.append(LabeledCall(call_id, label, passes["primary"]["verdict"], passes["reversed"]["verdict"]))
    return out


def label_distribution(labels: list[LabeledCall], snapshot_size: int) -> dict[str, Any]:
    n = len(labels)
    counts = {"SAFE": 0, "HARM": 0, "UNKNOWN": 0}
    for lc in labels:
        counts[lc.label] += 1
    return {
        "provisional": True,
        "snapshot_size": snapshot_size,
        "n_labeled_calls": n,
        "counts": counts,
        "share": {k: (v / n if n else None) for k, v in counts.items()},
    }


# --- Step 3: group-disjoint quality-model training on judge-derived labels ---

def fit_quality_model(
    replay_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    seeds: range = range(1, 41),
    kind: str = "logreg",
) -> dict[str, Any]:
    """Same grouped train/val/test methodology as analysis.py, but the
    label is now the REAL per-call judge-derived SAFE/HARM signal (UNKNOWN
    calls excluded from train/test, never forced to a side), not distant
    supervision. This is the actual "final-label modeling" step the whole
    pipeline has been building toward."""
    label_by_call = {lc.call_id: lc.label for lc in build_call_labels(judge_rows)}
    call_by_id = features.regenerate_replay_calls(replay_rows)

    rows = []
    for call_id, label in label_by_call.items():
        if label == "UNKNOWN" or call_id not in call_by_id:
            continue
        call = call_by_id[call_id]
        feats = analysis.call_features(call)
        rows.append(analysis.CallRow(feats, 1 if label == "SAFE" else 0, call.task_group))

    n_groups = len({r.task_group for r in rows})
    if n_groups < 3 or len({r.label for r in rows}) < 2:
        return {
            "provisional": True, "skipped": True,
            "reason": f"insufficient data for a real split (n_groups={n_groups}, n_rows={len(rows)})",
            "n_rows": len(rows), "n_task_groups": n_groups,
        }

    results = []
    for seed in seeds:
        train, val, test = analysis.group_split(rows, seed)
        r = analysis.fit_and_eval(train, val, test, kind=kind)
        r["seed"] = seed
        results.append(r)
    valid = [r for r in results if not r["skipped"]]
    summary: dict[str, Any] = {
        "provisional": True, "skipped": False, "model": kind,
        "n_rows": len(rows), "n_task_groups": n_groups,
        "n_seeds_valid": len(valid), "n_seeds_requested": len(list(seeds)),
    }
    if valid:
        summary["auc_mean"] = statistics.mean(r["auc"] for r in valid)
        summary["auc_stdev"] = statistics.pstdev(r["auc"] for r in valid) if len(valid) > 1 else 0.0
        summary["precision_mean"] = statistics.mean(r["precision"] for r in valid)
        summary["recall_mean"] = statistics.mean(r["recall"] for r in valid)
        summary["base_rate_mean"] = statistics.mean(r["base_rate_test"] for r in valid)
    return summary


# --- Step 4: temporal-safe window-update strategy comparison ---

def _chronological_rows(replay_rows: list[dict[str, Any]], label_by_call: dict[str, Label]) -> list[analysis.CallRow]:
    """Rows in real timestamp order (falls back to a stable id-based order
    only if a timestamp is genuinely missing -- never silently reorders
    real chronology)."""
    items = []
    fresh_by_id = features.regenerate_replay_calls(replay_rows)
    for r in replay_rows:
        call_id = r["call"]["call_id"]
        label = label_by_call.get(call_id)
        if label is None or label == "UNKNOWN":
            continue
        call = fresh_by_id[call_id]
        ts = call.timestamp_unix_s if call.timestamp_unix_s is not None else float("inf")
        items.append((ts, call_id, analysis.CallRow(analysis.call_features(call), 1 if label == "SAFE" else 0, call.task_group)))
    items.sort(key=lambda x: (x[0], x[1]))
    return [row for _, _, row in items]


def _fit_logreg_weighted(rows: list[analysis.CallRow], weights: list[float] | None = None):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = analysis._matrix(rows)
    y = np.array([r.label for r in rows])
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    if len(set(y.tolist())) < 2:
        return None, mean, std
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(Xs, y, sample_weight=weights)
    return model, mean, std


def _eval_window(model, mean, std, rows: list[analysis.CallRow]) -> float | None:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    if model is None or not rows:
        return None
    X = analysis._matrix(rows)
    y = np.array([r.label for r in rows])
    if len(set(y.tolist())) < 2:
        return None
    Xs = (X - mean) / std
    proba = model.predict_proba(Xs)[:, 1]
    return float(roc_auc_score(y, proba))


def window_strategy_comparison(
    replay_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    window_size: int = 20,
    ewma_halflife_windows: float = 2.0,
) -> dict[str, Any]:
    """Compares four update strategies chronologically, each window
    evaluated on the NEXT window only (temporal-safe: a strategy's fit at
    window t never sees window t's own labels). Mirrors
    experiments/phase1_router/window_policies.py's four-strategy design,
    adapted from MBPP's selection-order chunks to this dataset's real
    per-call timestamps.
    """
    label_by_call = {lc.call_id: lc.label for lc in build_call_labels(judge_rows)}
    ordered = _chronological_rows(replay_rows, label_by_call)
    n = len(ordered)
    windows = [ordered[i:i + window_size] for i in range(0, n, window_size)]
    if len(windows) < 3:
        return {"provisional": True, "skipped": True, "reason": f"only {len(windows)} windows at size {window_size}, need >=3", "n_rows": n}

    strategies = {"frozen": [], "cumulative": [], "sliding": [], "ewma": []}
    frozen_model = None
    for t in range(1, len(windows)):
        history = windows[:t]
        next_window = windows[t]

        # frozen: fit once on window 0 only, never updates.
        if frozen_model is None:
            frozen_model = _fit_logreg_weighted(windows[0])
        auc = _eval_window(*frozen_model, next_window)
        if auc is not None:
            strategies["frozen"].append(auc)

        # cumulative: fit on everything seen so far.
        cum_rows = [r for w in history for r in w]
        model = _fit_logreg_weighted(cum_rows)
        auc = _eval_window(*model, next_window)
        if auc is not None:
            strategies["cumulative"].append(auc)

        # sliding: fit only on the most recent window.
        model = _fit_logreg_weighted(history[-1])
        auc = _eval_window(*model, next_window)
        if auc is not None:
            strategies["sliding"].append(auc)

        # EWMA: recency-weighted cumulative fit.
        decay = math.log(2) / ewma_halflife_windows
        weighted_rows, weights = [], []
        for age, w in enumerate(reversed(history)):
            wt = math.exp(-decay * age)
            for r in w:
                weighted_rows.append(r)
                weights.append(wt)
        model = _fit_logreg_weighted(weighted_rows, weights)
        auc = _eval_window(*model, next_window)
        if auc is not None:
            strategies["ewma"].append(auc)

    summary = {"provisional": True, "n_rows": n, "n_windows": len(windows), "window_size": window_size}
    for name, aucs in strategies.items():
        summary[name] = {
            "n_evaluated_windows": len(aucs),
            "auc_mean": statistics.mean(aucs) if aucs else None,
            "auc_stdev": statistics.pstdev(aucs) if len(aucs) > 1 else 0.0,
        }
    return summary


# --- Step 5: online-policy simulation (contextual bandit / Thompson / periodic A-B) ---

class LinUCBArm:
    """Ridge-regularized linear bandit arm (one per action: local, cloud)."""

    def __init__(self, d: int, alpha: float = 1.0):
        self.A = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]  # d x d identity
        self.b = [0.0] * d
        self.alpha = alpha
        self.d = d

    def _solve(self) -> list[float]:
        # Small dense solve via numpy for clarity/robustness.
        import numpy as np
        A = np.array(self.A)
        b = np.array(self.b)
        return list(np.linalg.solve(A, b))

    def ucb(self, x: list[float]) -> float:
        import numpy as np
        A = np.array(self.A)
        x_arr = np.array(x)
        theta = np.linalg.solve(A, np.array(self.b))
        mean = float(theta @ x_arr)
        conf = self.alpha * float(math.sqrt(max(0.0, x_arr @ np.linalg.solve(A, x_arr))))
        return mean + conf

    def update(self, x: list[float], reward: float) -> None:
        import numpy as np
        A = np.array(self.A) + np.outer(x, x)
        b = np.array(self.b) + reward * np.array(x)
        self.A = A.tolist()
        self.b = b.tolist()


def _feature_vector(row: analysis.CallRow) -> list[float]:
    return [row.features[f] for f in analysis.FEATURE_NAMES]


def bandit_simulation(
    replay_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    seed: int = 20260916,
) -> dict[str, Any]:
    """Offline replay of three online policies over the chronological
    labeled sequence, each choosing local-vs-cloud per call and receiving
    reward = 1 if that choice matches the judge-derived SAFE/HARM label,
    0 otherwise (HARM calls routed local, or SAFE calls routed cloud, both
    score 0 -- this rewards matching the real safe choice, not blanket
    local preference).

    - `linucb`: contextual bandit, arms scored by ridge-regularized linear
      UCB over the same FEATURE_NAMES the offline classifier uses.
    - `thompson`: Bayesian linear regression per arm, Gaussian posterior,
      Thompson-sampled action selection.
    - `periodic_ab`: fixed-rate exploration baseline (route local exactly
      every Nth call, cloud otherwise) -- no learning, the naive
      alternative these two are meant to beat.
    """
    label_by_call = {lc.call_id: lc.label for lc in build_call_labels(judge_rows)}
    ordered = _chronological_rows(replay_rows, label_by_call)
    if len(ordered) < 10:
        return {"provisional": True, "skipped": True, "reason": f"only {len(ordered)} labeled calls, need >=10"}

    d = len(analysis.FEATURE_NAMES)
    rng = random.Random(seed)

    def reward_for(row: analysis.CallRow, chose_local: bool) -> float:
        is_safe = row.label == 1
        return 1.0 if (chose_local == is_safe) else 0.0

    # LinUCB
    local_arm, cloud_arm = LinUCBArm(d), LinUCBArm(d)
    linucb_rewards = []
    for row in ordered:
        x = _feature_vector(row)
        chose_local = local_arm.ucb(x) >= cloud_arm.ucb(x)
        r = reward_for(row, chose_local)
        linucb_rewards.append(r)
        (local_arm if chose_local else cloud_arm).update(x, r)

    # Thompson sampling: same ridge-posterior machinery, action = whichever
    # arm's SAMPLED (not UCB) score is higher.
    import numpy as np
    local_arm2, cloud_arm2 = LinUCBArm(d), LinUCBArm(d)
    thompson_rewards = []
    for row in ordered:
        x = np.array(_feature_vector(row))
        samples = {}
        for name, arm in (("local", local_arm2), ("cloud", cloud_arm2)):
            A = np.array(arm.A)
            theta_mean = np.linalg.solve(A, np.array(arm.b))
            cov = np.linalg.inv(A)
            theta_sample_vec = np.random.default_rng(rng.randint(0, 2**31 - 1)).multivariate_normal(theta_mean, cov)
            samples[name] = float(theta_sample_vec @ x)
        chose_local = samples["local"] >= samples["cloud"]
        r = reward_for(row, chose_local)
        thompson_rewards.append(r)
        (local_arm2 if chose_local else cloud_arm2).update(list(x), r)

    # Periodic A/B: no learning, fixed exploration rate (route local every
    # 3rd call, cloud otherwise) -- the naive baseline these should beat.
    periodic_rewards = [reward_for(row, i % 3 == 0) for i, row in enumerate(ordered)]

    def cum_mean(rewards: list[float]) -> list[float]:
        out, total = [], 0.0
        for i, r in enumerate(rewards, 1):
            total += r
            out.append(total / i)
        return out

    return {
        "provisional": True,
        "n_calls": len(ordered),
        "linucb": {"mean_reward": statistics.mean(linucb_rewards), "final_cumulative_accuracy": cum_mean(linucb_rewards)[-1]},
        "thompson": {"mean_reward": statistics.mean(thompson_rewards), "final_cumulative_accuracy": cum_mean(thompson_rewards)[-1]},
        "periodic_ab": {"mean_reward": statistics.mean(periodic_rewards), "final_cumulative_accuracy": cum_mean(periodic_rewards)[-1]},
    }


# --- Step 6: cache-switching cost analysis (LAST, least mature) ---

def cache_switch_analysis(replay_rows: list[dict[str, Any]], judge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Explicitly the least mature piece of this pipeline: the underlying
    `estimated_recompute_cost_if_switched` feature (features.py) is a
    DECLARED heuristic (10x token-count multiplier on a backend flip),
    never validated against a real measured cache-miss cost for this
    setup. This function reports what the heuristic says, clearly labeled
    as unvalidated, not a real cost measurement."""
    label_by_call = {lc.call_id: lc.label for lc in build_call_labels(judge_rows)}
    ordered = _chronological_rows(replay_rows, label_by_call)
    switch_costs = [row.features["estimated_recompute_cost_if_switched"] for row in ordered]
    nonzero = [c for c in switch_costs if c > 0]
    return {
        "provisional": True,
        "validated": False,
        "note": "estimated_recompute_cost_if_switched is a declared heuristic (10x token multiplier on backend flip), not a measured cache-miss cost -- see features.py",
        "n_calls": len(ordered),
        "n_with_nonzero_estimated_switch_cost": len(nonzero),
        "mean_estimated_switch_cost_kunits": statistics.mean(switch_costs) if switch_costs else None,
    }
