#!/usr/bin/env python3
"""Fit and validate a request-only local-truncation risk model.

The corpus is byte- and call-ID-deduplicated.  Only successful, local,
tool-equipped message calls with an exact pre-dispatch local token count are
eligible.  The already hard-gated security-monitor class is excluded so its
64-token output ceiling cannot create a trivial classifier.

The holdout is by canonical task instance, never by call.  Its membership and
the corpus timestamp cutoff were frozen before fitting the model committed in
``edgeproxy.router``.

    UV_CACHE_DIR=/tmp/flowmesh-uv-cache uv run python \
      scripts/analyze_call_risk.py traces
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from edgeproxy.router import extract_features


# The live planning A/B was appending to a trace while this analysis ran.
# Freezing its timestamp makes every reported count reproducible even after
# that file grows.
CORPUS_CUTOFF_UNIX_S = 1_788_608_178.0

HOLDOUT_GROUPS = frozenset(
    {
        "swebench-ansible-395e5e20",
        "swebench-openlibrary-53d376b1",
        "swebench-qutebrowser-5fdc83e5",
        "swebench-qutebrowser-e64622cd",
    }
)

FEATURE_NAMES = (
    "local_prompt_tokens",
    "n_messages",
    "errored_tool_result_density",
    "branch_turn_ordinal",
    "est_system_tokens",
    "n_tools",
    "is_tool_continuation",
    "estimated_local_cached_fraction",
    "local_prompt_tokens_squared",
    "local_context_pressure",
)

FEATURE_SETS = {
    "baseline": (0, 1, 2),
    "request_structure": (0, 1, 2, 3, 4, 5, 6),
    "capacity_normalized": (9, 1, 2),
}


@dataclass(frozen=True)
class Example:
    values: tuple[float, ...]
    label: int
    group: str
    model_profile: str = "unknown"
    local_token_budget: int | None = None


def _canonical_group(path: Path) -> str:
    match = re.search(r"swebench-[a-z0-9]+-[0-9a-f]{8}", str(path))
    if match:
        return match.group(0)
    task_names = (
        "explore-call-graph",
        "explore-config-defaults",
        "explore-pricing-api",
        "fanout-architecture-map",
        "fanout-parallel-bugfix",
        "fanout-repo-audit",
        "feature-priority-levels",
        "fix-discount-tier-off-by-one",
        "fix-lru-cache-eviction",
        "fix-null-handling",
        "fix-sorting-ties",
        "refactor-extract-pricing",
        "refactor-remove-duplication",
        "refactor-rename-for-clarity",
    )
    for name in task_names:
        if name in str(path):
            return f"eval-{name}"
    if "fanout" in str(path):
        # traces/fanout/<run-dir>/*.jsonl and traces/fanout-policy-pair/<run-dir>/*.jsonl
        # each hold one independent campaign episode (a distinct date/condition/
        # run-id). A single catch-all "fanout" group used to lump 13 of these
        # together into one 970-call CV fold, which produced 950 false
        # positives under leave-one-group-out (the model trained on everything
        # else fires on a group that is secretly 13 different distributions).
        # Group by the immediate run directory instead.
        parent = path.parent.name
        if parent not in ("fanout", "fanout-policy-pair", "traces"):
            return f"fanout-{parent}"
        # Loose files sitting directly under traces/fanout/ or
        # traces/fanout-policy-pair/ (no run-subdirectory) still encode one
        # episode each in their own filename, e.g.
        # "cloud_20260905T155832Z-55112.jsonl" -- group by the file stem
        # instead of falling back to one shared "fanout" bucket (2026-09-11:
        # this residual bucket held 14 files / 7 positives / 449 false
        # positives in one fold before this fix).
        return f"fanout-{path.stem}"
    return f"legacy-{path.name}"


def _model_profile(record: dict[str, Any]) -> str:
    """Recover the local setup without inspecting prompt or response content."""
    experiment = str(record.get("experiment_id") or "").lower()
    # Check 27B/qwen38 first: the substring "7b" also matches inside "27b", so
    # an experiment id like "risk-corpus-27b-clean-..." must not fall through
    # to the 7B branch.
    if "qwen38" in experiment or "27b" in experiment:
        return "qwen38-27b"
    if "qwen25" in experiment or "7b" in experiment:
        return "qwen25-7b"
    vllm = ((record.get("local_resources") or {}).get("vllm") or {})
    if vllm.get("block_size") == 16 and vllm.get("cache_dtype") == "auto":
        return "qwen25-7b"
    if vllm.get("block_size") == 1568 and vllm.get("cache_dtype") == "fp8":
        return "qwen38-27b"
    return "unknown"


def _configured_budget(record: dict[str, Any], profile: str) -> int | None:
    recorded = (record.get("features") or {}).get("local_token_budget")
    if isinstance(recorded, (int, float)) and recorded > 0:
        return int(recorded)
    # Historical traces predate explicit budget recording. These are the
    # versioned setup-profile budgets, not values inferred from outcomes.
    return {"qwen25-7b": 54_000, "qwen38-27b": 90_000}.get(profile)


def _stop_reason(record: dict[str, Any]) -> str | None:
    return (record.get("response") or {}).get("stop_reason") or (
        record.get("call") or {}
    ).get("stop_reason")


def _placement(record: dict[str, Any]) -> str | None:
    return record.get("placement") or (record.get("call") or {}).get("backend")


def _status(record: dict[str, Any]) -> int | None:
    return record.get("status") or (record.get("call") or {}).get("http_status")


def _errored_tool_result_density(request: dict[str, Any]) -> float:
    total = errored = 0
    for message in request.get("messages") or []:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                total += 1
                errored += bool(block.get("is_error"))
    return errored / total if total else 0.0


def _records(roots: Iterable[Path], cutoff: float) -> tuple[list[tuple[Path, dict[str, Any]]], int, int]:
    seen_files: set[str] = set()
    seen_calls: set[str] = set()
    records: list[tuple[Path, dict[str, Any]]] = []
    invalid_lines = duplicate_files = 0
    paths = sorted(path for root in roots for path in root.rglob("*.jsonl"))
    for path in paths:
        data = path.read_bytes()
        file_digest = hashlib.sha256(data).hexdigest()
        if file_digest in seen_files:
            duplicate_files += 1
            continue
        seen_files.add(file_digest)
        for raw in data.splitlines():
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid_lines += 1
                continue
            if record.get("path") != "/v1/messages" or not isinstance(
                record.get("request"), dict
            ):
                continue
            timestamp = record.get("ts")
            if isinstance(timestamp, (int, float)) and timestamp > cutoff:
                continue
            call_id = record.get("id") or (record.get("call") or {}).get("call_id")
            identity = str(call_id) if call_id else hashlib.sha256(raw).hexdigest()
            if identity in seen_calls:
                continue
            seen_calls.add(identity)
            records.append((path, record))
    return records, invalid_lines, duplicate_files


def _auc(pairs: list[tuple[float, int]]) -> float:
    positives = [score for score, label in pairs if label]
    negatives = [score for score, label in pairs if not label]
    wins = sum(pos > neg for pos in positives for neg in negatives)
    ties = sum(pos == neg for pos in positives for neg in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            scale = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - scale * augmented[column][index]
                for index in range(size + 1)
            ]
    return [augmented[row][-1] for row in range(size)]


def _fit_logistic(
    examples: list[Example], feature_indices: tuple[int, ...], l2: float = 1.0
) -> dict[str, Any]:
    dimensions = len(feature_indices)
    means = [
        statistics.mean(row.values[index] for row in examples)
        for index in feature_indices
    ]
    scales = [
        statistics.pstdev(row.values[index] for row in examples) or 1.0
        for index in feature_indices
    ]
    inputs = [
        [1.0]
        + [
            (row.values[index] - means[j]) / scales[j]
            for j, index in enumerate(feature_indices)
        ]
        for row in examples
    ]
    labels = [row.label for row in examples]
    positives = sum(labels)
    weights = [math.log(positives / (len(labels) - positives))] + [0.0] * dimensions
    for _ in range(50):
        probabilities = []
        for row in inputs:
            logit = sum(w * x for w, x in zip(weights, row))
            probabilities.append(
                1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
            )
        gradient = [
            sum(
                (probabilities[i] - labels[i]) * inputs[i][j]
                for i in range(len(inputs))
            )
            + (l2 * weights[j] if j else 0.0)
            for j in range(dimensions + 1)
        ]
        hessian = [
            [
                sum(
                    probabilities[i]
                    * (1.0 - probabilities[i])
                    * inputs[i][j]
                    * inputs[i][k]
                    for i in range(len(inputs))
                )
                + (l2 if j == k and j else 0.0)
                for k in range(dimensions + 1)
            ]
            for j in range(dimensions + 1)
        ]
        step = _solve(hessian, gradient)
        weights = [weight - delta for weight, delta in zip(weights, step)]
        if max(abs(delta) for delta in step) < 1e-9:
            break
    return {
        "feature_names": [FEATURE_NAMES[index] for index in feature_indices],
        "feature_indices": list(feature_indices),
        "intercept": weights[0],
        "weights": weights[1:],
        "means": means,
        "scales": scales,
    }


def _predict(model: dict[str, Any], values: tuple[float, ...]) -> float:
    logit = model["intercept"] + sum(
        weight * (values[index] - mean) / scale
        for weight, index, mean, scale in zip(
            model["weights"],
            model["feature_indices"],
            model["means"],
            model["scales"],
        )
    )
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))


def _classification(examples: list[Example], scores: list[float], threshold: float) -> dict[str, Any]:
    tp = sum(row.label and score >= threshold for row, score in zip(examples, scores))
    fp = sum(not row.label and score >= threshold for row, score in zip(examples, scores))
    fn = sum(row.label and score < threshold for row, score in zip(examples, scores))
    tn = len(examples) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


def _profile_validation(
    examples: list[Example], scores: list[float], threshold: float
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in sorted({row.model_profile for row in examples}):
        positions = [
            index for index, row in enumerate(examples) if row.model_profile == profile
        ]
        profile_examples = [examples[index] for index in positions]
        profile_scores = [scores[index] for index in positions]
        positives = sum(row.label for row in profile_examples)
        negatives = len(profile_examples) - positives
        result[profile] = {
            "calls": len(profile_examples),
            "positives": positives,
            "groups": len({row.group for row in profile_examples}),
            "auc": (
                _auc(list(zip(profile_scores, (row.label for row in profile_examples))))
                if positives and negatives
                else None
            ),
            **_classification(profile_examples, profile_scores, threshold),
        }
    return result


def _select_threshold(examples: list[Example], scores: list[float]) -> float:
    candidates = []
    for threshold in sorted(set(scores)):
        result = _classification(examples, scores, threshold)
        precision, recall = result["precision"], result["recall"]
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidates.append((f1, precision, threshold))
    return max(candidates)[2]


def _select_threshold_for_recall(
    examples: list[Example], scores: list[float], target_recall: float
) -> float:
    """Highest-precision observed threshold meeting a training recall floor.

    Ties prefer the higher threshold, which avoids escalating extra calls when
    the measured precision and recall are identical.  The threshold is chosen
    from training data only; the held-out task groups remain evaluation-only.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    candidates = []
    for threshold in sorted(set(scores)):
        result = _classification(examples, scores, threshold)
        if result["recall"] >= target_recall:
            candidates.append((result["precision"], threshold))
    if not candidates:
        raise ValueError("no threshold meets the requested recall")
    return max(candidates)[1]


def _select_conjunctive_thresholds_for_recall(
    examples: list[Example],
    first_index: int,
    second_index: int,
    target_recall: float,
) -> tuple[float, float, dict[str, Any]]:
    """Fit a transparent ``x >= a and y >= b`` rule on training data."""
    positives = [row for row in examples if row.label]
    first_thresholds = sorted({row.values[first_index] for row in positives})
    second_thresholds = sorted({row.values[second_index] for row in positives})
    candidates: list[tuple[float, int, float, float, dict[str, Any]]] = []
    for first_threshold in first_thresholds:
        for second_threshold in second_thresholds:
            scores = [
                float(
                    row.values[first_index] >= first_threshold
                    and row.values[second_index] >= second_threshold
                )
                for row in examples
            ]
            result = _classification(examples, scores, 1.0)
            if result["recall"] >= target_recall:
                escalations = result["tp"] + result["fp"]
                candidates.append(
                    (
                        result["precision"],
                        -escalations,
                        first_threshold,
                        second_threshold,
                        result,
                    )
                )
    if not candidates:
        raise ValueError("no conjunctive rule meets the requested recall")
    _, _, first, second, result = max(candidates)
    return first, second, result


def _grouped_cross_validation(
    examples: list[Example],
    feature_indices: tuple[int, ...],
    target_recall: float,
) -> dict[str, Any]:
    """Evaluate on whole unseen groups, stratifying the sparse positive groups.

    Each fold contains one positive-bearing group plus a round-robin share of
    negative-only groups. This prevents the hundreds of correlated calls in a
    single task from appearing on both sides of a fold.
    """
    groups = sorted({row.group for row in examples})
    positive_groups = [
        group for group in groups if any(row.group == group and row.label for row in examples)
    ]
    if len(positive_groups) < 2:
        raise ValueError("grouped validation needs at least two positive groups")
    folds: list[list[str]] = [[group] for group in positive_groups]
    negative_groups = [group for group in groups if group not in positive_groups]
    for index, group in enumerate(negative_groups):
        folds[index % len(folds)].append(group)

    f1_predictions: list[float] = []
    recall_predictions: list[float] = []
    validation_examples: list[Example] = []
    fold_results: list[dict[str, Any]] = []
    for validation_groups in folds:
        validation_group_set = set(validation_groups)
        fold_train = [row for row in examples if row.group not in validation_group_set]
        fold_validation = [row for row in examples if row.group in validation_group_set]
        model = _fit_logistic(fold_train, feature_indices)
        train_scores = [_predict(model, row.values) for row in fold_train]
        validation_scores = [_predict(model, row.values) for row in fold_validation]
        f1_threshold = _select_threshold(fold_train, train_scores)
        recall_threshold = _select_threshold_for_recall(
            fold_train, train_scores, target_recall
        )
        fold_f1_predictions = [float(score >= f1_threshold) for score in validation_scores]
        fold_recall_predictions = [
            float(score >= recall_threshold) for score in validation_scores
        ]
        f1_predictions.extend(fold_f1_predictions)
        recall_predictions.extend(fold_recall_predictions)
        validation_examples.extend(fold_validation)
        fold_results.append(
            {
                "groups": validation_groups,
                "calls": len(fold_validation),
                "positives": sum(row.label for row in fold_validation),
                "f1_validation": _classification(
                    fold_validation, fold_f1_predictions, 1.0
                ),
                "recall_validation": _classification(
                    fold_validation, fold_recall_predictions, 1.0
                ),
            }
        )
    def _macro(key: str) -> dict[str, float]:
        # unweighted mean across folds: every fold counts once regardless of
        # how many calls it has, so one huge fold (e.g. 1,104 calls) cannot
        # silently dominate a pooled, call-count-weighted number the way
        # f1_validation/recall_validation above do.
        precisions = [fold[key]["precision"] for fold in fold_results]
        recalls = [fold[key]["recall"] for fold in fold_results]
        return {
            "n_folds": len(fold_results),
            "precision_mean": statistics.mean(precisions),
            "precision_stdev": statistics.pstdev(precisions) if len(precisions) > 1 else 0.0,
            "recall_mean": statistics.mean(recalls),
            "recall_stdev": statistics.pstdev(recalls) if len(recalls) > 1 else 0.0,
        }

    return {
        "folds": fold_results,
        "f1_validation": _classification(validation_examples, f1_predictions, 1.0),
        "recall_validation": _classification(
            validation_examples, recall_predictions, 1.0
        ),
        "f1_validation_macro": _macro("f1_validation"),
        "recall_validation_macro": _macro("recall_validation"),
    }


def _grouped_cv_by_model_profile(
    examples: list[Example],
    feature_indices: tuple[int, ...],
    target_recall: float,
) -> dict[str, Any]:
    """Leave-one-task-group-out CV run SEPARATELY per local model profile, so a
    27B-only (or 7B-only) generalisation score is not diluted by the other
    model's calls. Uses every group for that profile (train + holdout pool),
    since the frozen train/holdout split is a 7B/27B mixture."""
    result: dict[str, Any] = {}
    for profile in sorted({row.model_profile for row in examples}):
        subset = [row for row in examples if row.model_profile == profile]
        positive_groups = {row.group for row in subset if row.label}
        summary = {
            "calls": len(subset),
            "groups": len({row.group for row in subset}),
            "positives": sum(row.label for row in subset),
            "positive_groups": sorted(positive_groups),
        }
        try:
            summary["cv"] = _grouped_cross_validation(subset, feature_indices, target_recall)
        except ValueError as exc:
            summary["cv"] = None
            summary["skipped"] = str(exc)
        result[profile] = summary
    return result


def analyze(
    roots: list[Path], cutoff: float, target_recall: float = 0.8
) -> dict[str, Any]:
    records, invalid_lines, duplicate_files = _records(roots, cutoff)
    local_successes = [
        (path, record)
        for path, record in records
        if _placement(record) == "local" and _status(record) == 200
    ]
    all_truncations = sum(_stop_reason(record) == "max_tokens" for _, record in local_successes)
    security_truncations = 0
    examples: list[Example] = []
    candidate_values: dict[str, list[tuple[float, int]]] = {
        name: []
        for name in (
            "local_prompt_tokens",
            "estimated_local_cached_tokens",
            "estimated_local_cached_fraction",
            "est_system_tokens",
            "n_cache_breakpoints",
            "n_messages",
            "n_tools",
            "is_tool_continuation",
            "errored_tool_result_density",
            "requested_max_tokens",
        )
    }
    for path, record in local_successes:
        request = record["request"]
        features = extract_features(request)
        label = int(_stop_reason(record) == "max_tokens")
        if features.is_security_monitor:
            security_truncations += label
            continue
        recorded_features = record.get("features") or {}
        prompt_tokens = recorded_features.get("local_prompt_tokens")
        if not features.has_tools or prompt_tokens is None or _stop_reason(record) is None:
            continue
        requested_max = recorded_features.get("max_tokens", features.max_tokens)
        error_density = _errored_tool_result_density(request)
        prompt_tokens_float = float(prompt_tokens)
        profile = _model_profile(record)
        local_token_budget = _configured_budget(record, profile)
        values = (
            prompt_tokens_float,
            float(features.n_messages),
            error_density,
            float(features.branch_turn_ordinal or 0),
            float(features.est_system_tokens),
            float(features.n_tools),
            float(features.is_tool_continuation),
            float(recorded_features.get("estimated_local_cached_fraction") or 0.0),
            prompt_tokens_float * prompt_tokens_float,
            (
                prompt_tokens_float / local_token_budget
                if local_token_budget is not None
                else 0.0
            ),
        )
        examples.append(
            Example(
                values,
                label,
                _canonical_group(path),
                profile,
                local_token_budget,
            )
        )
        observations = (
            float(prompt_tokens),
            float(recorded_features.get("estimated_local_cached_tokens") or 0.0),
            float(recorded_features.get("estimated_local_cached_fraction") or 0.0),
            float(features.est_system_tokens),
            float(features.n_cache_breakpoints),
            float(features.n_messages),
            float(features.n_tools),
            float(features.is_tool_continuation),
            error_density,
            float(requested_max),
        )
        for name, value in zip(candidate_values, observations):
            candidate_values[name].append((value, label))

    train = [row for row in examples if row.group not in HOLDOUT_GROUPS]
    holdout = [row for row in examples if row.group in HOLDOUT_GROUPS]
    group_summary = {
        group: {
            "calls": sum(row.group == group for row in examples),
            "positives": sum(row.label for row in examples if row.group == group),
            "split": "holdout" if group in HOLDOUT_GROUPS else "train",
        }
        for group in sorted({row.group for row in examples})
    }
    model = _fit_logistic(train, FEATURE_SETS["baseline"])
    train_scores = [_predict(model, row.values) for row in train]
    holdout_scores = [_predict(model, row.values) for row in holdout]
    threshold = _select_threshold(train, train_scores)
    recall_threshold = _select_threshold_for_recall(
        train, train_scores, target_recall
    )
    holdout_by_group: dict[str, Any] = {}
    for group in sorted({row.group for row in holdout}):
        positions = [index for index, row in enumerate(holdout) if row.group == group]
        group_examples = [holdout[index] for index in positions]
        group_scores = [holdout_scores[index] for index in positions]
        holdout_by_group[group] = {
            "calls": len(group_examples),
            "positives": sum(row.label for row in group_examples),
            **_classification(group_examples, group_scores, threshold),
        }
    variants: dict[str, Any] = {}
    for name, feature_indices in FEATURE_SETS.items():
        variant_model = _fit_logistic(train, feature_indices)
        variant_train_scores = [_predict(variant_model, row.values) for row in train]
        variant_holdout_scores = [_predict(variant_model, row.values) for row in holdout]
        variant_threshold = _select_threshold(train, variant_train_scores)
        variant_recall_threshold = _select_threshold_for_recall(
            train, variant_train_scores, target_recall
        )
        variants[name] = {
            "model": variant_model,
            "auc": _auc(list(zip(variant_holdout_scores, (row.label for row in holdout)))),
            "f1_threshold": variant_threshold,
            "f1_validation": _classification(
                holdout, variant_holdout_scores, variant_threshold
            ),
            "recall_threshold": variant_recall_threshold,
            "recall_validation": _classification(
                holdout, variant_holdout_scores, variant_recall_threshold
            ),
            "training_group_cv": _grouped_cross_validation(
                train, feature_indices, target_recall
            ),
        }
    univariate_models: dict[str, Any] = {}
    for index, name in enumerate(FEATURE_NAMES):
        if name == "local_prompt_tokens_squared":
            continue
        feature_train_scores = [row.values[index] for row in train]
        feature_holdout_scores = [row.values[index] for row in holdout]
        if _auc(list(zip(feature_train_scores, (row.label for row in train)))) < 0.5:
            continue
        feature_threshold = _select_threshold(train, feature_train_scores)
        feature_recall_threshold = _select_threshold_for_recall(
            train, feature_train_scores, target_recall
        )
        univariate_models[name] = {
            "f1_threshold": feature_threshold,
            "f1_validation": _classification(
                holdout, feature_holdout_scores, feature_threshold
            ),
            "recall_threshold": feature_recall_threshold,
            "recall_validation": _classification(
                holdout, feature_holdout_scores, feature_recall_threshold
            ),
        }
    conjunctive_models: dict[str, Any] = {}
    for first_index, second_index in ((0, 1), (0, 3), (0, 7), (1, 3)):
        first, second, training_result = _select_conjunctive_thresholds_for_recall(
            train, first_index, second_index, target_recall
        )
        conjunctive_holdout_scores = [
            float(
                row.values[first_index] >= first
                and row.values[second_index] >= second
            )
            for row in holdout
        ]
        name = f"{FEATURE_NAMES[first_index]}+{FEATURE_NAMES[second_index]}"
        conjunctive_models[name] = {
            "thresholds": [first, second],
            "training": training_result,
            "validation": _classification(
                holdout, conjunctive_holdout_scores, 1.0
            ),
        }
    return {
        "corpus": {
            "cutoff_unix_s": cutoff,
            "unique_calls": len(records),
            "duplicate_files_skipped": duplicate_files,
            "invalid_json_lines_skipped": invalid_lines,
        },
        "labels": {
            "local_http_200_max_tokens_all_classes": all_truncations,
            "security_monitor_truncations_excluded": security_truncations,
            "eligible_nonsecurity_tool_call_truncations": sum(row.label for row in examples),
            "positive_canonical_groups": len({row.group for row in examples if row.label}),
        },
        "candidate_features": {
            name: {
                "auc": _auc(pairs),
                "positive_mean": statistics.mean(value for value, label in pairs if label),
                "negative_mean": statistics.mean(value for value, label in pairs if not label),
            }
            for name, pairs in candidate_values.items()
        },
        "heldout_feature_auc": {
            name: _auc([(row.values[index], row.label) for row in holdout])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "split": {
            "train_calls": len(train),
            "train_positives": sum(row.label for row in train),
            "train_groups": len({row.group for row in train}),
            "holdout_calls": len(holdout),
            "holdout_positives": sum(row.label for row in holdout),
            "holdout_groups": sorted({row.group for row in holdout}),
        },
        "groups": group_summary,
        "model": model,
        "model_variants": variants,
        "univariate_models": univariate_models,
        "conjunctive_models": conjunctive_models,
        "threshold": threshold,
        "recall_target": {
            "target": target_recall,
            "threshold": recall_threshold,
            "training": _classification(train, train_scores, recall_threshold),
            "validation": _classification(
                holdout, holdout_scores, recall_threshold
            ),
        },
        "holdout_by_group": holdout_by_group,
        "validation_by_model_profile": _profile_validation(
            holdout, holdout_scores, threshold
        ),
        "grouped_cv_by_model_profile": {
            name: _grouped_cv_by_model_profile(examples, indices, target_recall)
            for name, indices in FEATURE_SETS.items()
        },
        "validation": {
            "auc": _auc(list(zip(holdout_scores, (row.label for row in holdout)))),
            "brier": statistics.mean(
                (score - row.label) ** 2 for score, row in zip(holdout_scores, holdout)
            ),
            "mean_predicted_risk": statistics.mean(holdout_scores),
            **_classification(holdout, holdout_scores, threshold),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--cutoff-ts", type=float, default=CORPUS_CUTOFF_UNIX_S)
    parser.add_argument("--target-recall", type=float, default=0.8)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.roots, args.cutoff_ts, args.target_recall),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
