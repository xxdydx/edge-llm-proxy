"""Learned capability classifiers: predict, from request-only features
alone, whether a local reply would have been action-equivalent to the
teacher (cloud) reply.

Request-only by the plan's fixed scope -- no cache state, cohort signals,
GPU/queue telemetry, embeddings, or bandit history. The feature set below is
a strict subset of `router.CallFeatures`'s pre-dispatch fields, mirroring
`scripts/analyze_call_risk.py`'s existing frozen risk model rather than
inventing a new feature philosophy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .schema import LabeledExample

FEATURE_NAMES = (
    "n_tools",
    "n_messages",
    "est_system_tokens",
    "max_tokens",
    "is_tool_continuation",
    "errored_tool_result_density",
    "branch_turn_ordinal",
    "has_agent_tool",
    "is_security_monitor",
    "local_prompt_tokens",
)

# Sentinel for missing pre-dispatch fields (e.g. branch_turn_ordinal on a
# call outside any tracked branch) -- an out-of-range magnitude value rather
# than 0, so "missing" cannot be confused with a real zero for a tree model
# and stays a large-magnitude outlier for the linear model too.
_MISSING = -1.0


def feature_value(features: dict[str, Any], name: str) -> float:
    value = features.get(name)
    if value is None:
        return _MISSING
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def build_feature_matrix(
    examples: list[LabeledExample], label_field: str
) -> tuple[np.ndarray, np.ndarray, list[LabeledExample]]:
    """Rows with UNKNOWN/EXECUTION_ERROR on `label_field` are excluded
    entirely -- trained and evaluated on known labels only, per the plan."""
    xs: list[list[float]] = []
    ys: list[int] = []
    kept: list[LabeledExample] = []
    for ex in examples:
        label = getattr(ex, label_field)
        if label not in ("EQUIVALENT", "NOT_EQUIVALENT"):
            continue
        xs.append([feature_value(ex.features, name) for name in FEATURE_NAMES])
        ys.append(1 if label == "EQUIVALENT" else 0)
        kept.append(ex)
    return np.array(xs, dtype=float), np.array(ys, dtype=int), kept


@dataclass
class TrainedModel:
    name: str
    predict_proba: Any  # Callable[[np.ndarray], np.ndarray] -- P(equivalent)
    feature_names: tuple[str, ...] = FEATURE_NAMES


def train_logistic_regression(X: np.ndarray, y: np.ndarray, seed: int) -> TrainedModel:
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, y)

    def predict_proba(X_new: np.ndarray) -> np.ndarray:
        return clf.predict_proba(X_new)[:, 1]

    return TrainedModel(name="logistic_regression", predict_proba=predict_proba)


def train_lightgbm(X: np.ndarray, y: np.ndarray, seed: int) -> TrainedModel | None:
    """Returns None (not a fallback substitute) if lightgbm cannot run in
    this environment -- analysis.py must treat a missing model as an
    explicitly skipped comparison arm, never silently omit it from the
    report without saying why."""
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as exc:
        import logging

        logging.getLogger(__name__).warning("lightgbm unavailable, skipping: %s", exc)
        return None

    clf = lgb.LGBMClassifier(
        random_state=seed,
        n_estimators=100,
        max_depth=3,
        min_child_samples=5,
        verbosity=-1,
    )
    clf.fit(X, y)

    def predict_proba(X_new: np.ndarray) -> np.ndarray:
        return clf.predict_proba(X_new)[:, 1]

    return TrainedModel(name="lightgbm", predict_proba=predict_proba)
