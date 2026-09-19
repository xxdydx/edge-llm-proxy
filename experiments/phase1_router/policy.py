"""Routing policies over PairedExample lists.

A policy maps request-only features -> route_local (bool per example).
Baselines are deterministic; the learned policies fit a request-only
classifier of P(local plus_pass) on the TRAIN split, pick a decision
threshold on VAL under the epsilon accuracy constraint, and are reported
once on TEST.

The learned classifiers here (`logreg`, `gbm`) are THIS experiment's own
request-only models -- they are not the routing-ladder policies (Policy
3/4/5/6) and are never labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FEATURE_NAMES
from .schema import PairedExample


def _matrix(examples: list[PairedExample]) -> np.ndarray:
    return np.array([[ex.features.get(f, 0.0) for f in FEATURE_NAMES] for ex in examples], dtype=float)


def _local_label(examples: list[PairedExample]) -> np.ndarray:
    return np.array([1 if ex.local_ok else 0 for ex in examples], dtype=int)


def _no_harm_label(examples: list[PairedExample]) -> np.ndarray:
    """One when routing local does not lose a cloud-solved task.

    This is the complement of quadrant B (cloud passes, local fails), which
    directly matches the router's relative-quality constraint.
    """
    return np.array(
        [0 if ex.cloud_ok and not ex.local_ok else 1 for ex in examples], dtype=int
    )


# --- deterministic baselines ------------------------------------------------

def always_cloud(examples): return [False] * len(examples)
def always_local(examples): return [True] * len(examples)

def oracle(examples: list[PairedExample]) -> list[bool]:
    """Perfect hindsight: route local exactly when local passes. This never
    sacrifices a solvable task (a local pass is a solved task) and maximises
    local share at no quality loss vs always-cloud on the both/cloud-only
    cells; cloud-only and neither cells correctly stay on cloud."""
    return [bool(ex.local_ok) for ex in examples]


def static_heuristic(examples: list[PairedExample], est_token_cutoff: float = 900.0) -> list[bool]:
    """Simple hand rule: keep short-prompt problems local, send the rest to
    cloud. A plausible deployable heuristic with no training."""
    return [float(ex.features.get("prompt_est_tokens", 1e9)) <= est_token_cutoff for ex in examples]


# --- learned request-only classifiers -------------------------------------

@dataclass
class LearnedPolicy:
    name: str
    model: object
    scaler_mean: np.ndarray
    scaler_std: np.ndarray

    def proba_local(self, examples: list[PairedExample]) -> np.ndarray:
        X = (_matrix(examples) - self.scaler_mean) / self.scaler_std
        p = self.model.predict_proba(X)[:, 1]
        return p

    def route(self, examples: list[PairedExample], threshold: float) -> list[bool]:
        return [bool(v >= threshold) for v in self.proba_local(examples)]


def fit_logreg(train: list[PairedExample]) -> LearnedPolicy:
    from sklearn.linear_model import LogisticRegression

    X = _matrix(train)
    y = _local_label(train)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    if len(set(y.tolist())) < 2:
        return LearnedPolicy("logreg", _Const(float(y.mean())), mean, std)
    m = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    m.fit(Xs, y)
    return LearnedPolicy("logreg", m, mean, std)


def fit_no_harm_logreg(train: list[PairedExample]) -> LearnedPolicy:
    """Fit P(not quadrant-B) from pre-dispatch features."""
    from sklearn.linear_model import LogisticRegression

    X = _matrix(train)
    y = _no_harm_label(train)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    if len(set(y.tolist())) < 2:
        return LearnedPolicy("no_harm_logreg", _Const(float(y.mean())), mean, std)
    m = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    m.fit(Xs, y)
    return LearnedPolicy("no_harm_logreg", m, mean, std)


def fit_gbm(train: list[PairedExample]) -> LearnedPolicy | None:
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    X = _matrix(train)
    y = _local_label(train)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    if len(set(y.tolist())) < 2:
        return None
    m = LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                       min_child_samples=5, verbose=-1)
    m.fit(Xs, y)
    return LearnedPolicy("gbm", m, mean, std)


def fit_no_harm_gbm(train: list[PairedExample]) -> LearnedPolicy | None:
    """Fit P(not quadrant-B) with LightGBM from pre-dispatch features."""
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    X = _matrix(train)
    y = _no_harm_label(train)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    if len(set(y.tolist())) < 2:
        return None
    m = LGBMClassifier(
        n_estimators=200, num_leaves=15, learning_rate=0.05,
        min_child_samples=5, class_weight="balanced", verbose=-1,
    )
    m.fit(Xs, y)
    return LearnedPolicy("no_harm_gbm", m, mean, std)


class _Const:
    """Fallback when the train split has only one class: predict its rate."""

    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1.0 - self.p), np.full(len(X), self.p)])
