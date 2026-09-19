"""Frozen v4+v5 train-only models scored on distinct corrected-order v6 tasks.

This script intentionally loads the two judge protocols separately. It never
concatenates v6 labels into training or selects a threshold/model from v6.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import analysis, fresh_v4_combined, model_comparison, text_model

RESULTS = Path(__file__).resolve().parent / "results"
V6 = RESULTS / "postpilot_v6"
V6_JUDGE = "pilot_judge_consistency_v5_order.jsonl"
EXPECTED_V6 = "v4-system-messages-tools-2026-09-16|v5-sha256-call-id-2026-09-17"


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if not len(y):
        return {"n": 0}
    return {
        "n": int(len(y)), "harm": int(y.sum()), "harm_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if int(y.sum()) else None,
        "brier": float(brier_score_loss(y, p)),
        "ece_5_bins": model_comparison._ece(y, p),
        "mean_predicted_harm": float(p.mean()),
        "score_min": float(p.min()), "score_max": float(p.max()),
    }


def _cluster_resample(events: list[model_comparison.LabeledEvent], p: np.ndarray) -> dict:
    """Enumerate 3^3 task-cluster bootstrap draws; exploratory, not reliable CI."""
    groups = sorted({e.task_group for e in events})
    y = np.asarray([int(e.label == "HARM") for e in events])
    by_group = {g: [i for i, e in enumerate(events) if e.task_group == g] for g in groups}
    aucs: list[float] = []
    aps: list[float] = []
    for draw in itertools.product(groups, repeat=len(groups)):
        idx = [i for g in draw for i in by_group[g]]
        yy, pp = y[idx], p[idx]
        if len(set(yy)) < 2:
            continue
        aucs.append(float(roc_auc_score(yy, pp)))
        aps.append(float(average_precision_score(yy, pp)))
    return {
        "method": "enumerated task-cluster bootstrap, 3 draws with replacement from 3 tasks",
        "n_total_draws": len(groups) ** len(groups), "n_two_class_draws": len(aucs),
        "auc_percentile_2_5_97_5": np.percentile(aucs, [2.5, 97.5]).tolist() if aucs else None,
        "ap_percentile_2_5_97_5": np.percentile(aps, [2.5, 97.5]).tolist() if aps else None,
        "warning": "Only three independent task clusters; percentile endpoints are sensitivity diagnostics, not a validated 95% confidence interval.",
    }


def run() -> dict:
    train, train_meta = fresh_v4_combined._events_with_meta()
    external, ext_meta = model_comparison.load_events(
        V6, ("stage1_replay_examples_pilot_v1.jsonl",), V6_JUDGE)
    if ext_meta["judge_protocol_versions"] != [EXPECTED_V6] or ext_meta["n_replay"] != 24:
        raise ValueError("external protocol or cohort size changed; freeze a new audit")
    if train_meta["feature_derivation_version"] != ext_meta["feature_derivation_version"]:
        raise ValueError("feature derivation changed between train and external cohort")
    train_groups = {e.task_group for e in train}
    ext_groups = {e.task_group for e in external}
    if train_groups & ext_groups or {model_comparison._repo(g) for g in train_groups} & {model_comparison._repo(g) for g in ext_groups}:
        raise ValueError("task/repository leakage across train and external cohorts")
    reserved = set(model_comparison.RESERVED_PILOT_HOLDOUT_TASKS)
    if (train_groups | ext_groups) & reserved:
        raise ValueError("reserved holdout was included")
    tr = [e for e in train if e.label != "UNKNOWN"]
    te = [e for e in external if e.label != "UNKNOWN"]
    if Counter(e.label for e in tr) != {"SAFE": 34, "HARM": 12} or Counter(e.label for e in te) != {"SAFE": 11, "HARM": 4}:
        raise ValueError("binary label counts changed; freeze a new audit")
    ytr = np.asarray([int(e.label == "HARM") for e in tr])
    yte = np.asarray([int(e.label == "HARM") for e in te])
    Xtr = np.asarray([[e.features[n] for n in analysis.FEATURE_NAMES] for e in tr])
    Xte = np.asarray([[e.features[n] for n in analysis.FEATURE_NAMES] for e in te])
    numeric = model_comparison._model("logreg")
    numeric.fit(Xtr, ytr)
    scores = {
        "numeric_regularized_logreg": np.asarray(numeric.predict_proba(Xte)[:, 1]),
        "word_structural_logreg": text_model._fit_predict(tr, te, "word_structural"),
    }
    support = lambda e: e.features["local_prompt_tokens"] != analysis._MISSING and e.features["context_utilization_ratio"] != analysis._MISSING
    out = {
        "status": "external_diagnostic_only_no_model_or_threshold_selection",
        "train": {"n_pairs": len(train), "n_binary": len(tr), "labels": dict(Counter(e.label for e in train)),
                  "n_tasks": len(train_groups), "n_supported_binary": sum(map(support, tr)),
                  "source_provenance": train_meta["sources"]},
        "external": {"n_pairs": len(external), "n_binary": len(te), "labels": dict(Counter(e.label for e in external)),
                     "n_tasks": len(ext_groups), "n_supported_binary": sum(map(support, te)),
                     "n_supported_all": sum(map(support, external)), "source_provenance": ext_meta},
        "positive_class": "HARM_LOCAL",
        "models": {},
        "caveat": "Blind judge proxy labels and corrected-order protocol shift; 3 task clusters and 4 HARM make calibration and intervals unstable. No live threshold follows.",
    }
    for kind, p in scores.items():
        per_task = {}
        for group in sorted(ext_groups):
            idx = [i for i, e in enumerate(te) if e.task_group == group]
            per_task[group] = _metrics(yte[idx], p[idx])
        out["models"][kind] = {
            "overall_binary": _metrics(yte, p),
            "per_task_binary": per_task,
            "cluster_resample": _cluster_resample(te, p),
            "call_scores": [{"call_id": e.call_id, "task_group": e.task_group, "label": e.label,
                             "harm_score": float(score), "feature_supported": support(e)}
                            for e, score in zip(te, p)],
            "probability_note": "Class-weighted logistic probabilities are not calibrated; Brier/ECE are diagnostics only.",
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "external_v6_from_frozen_dev56_v1.json")
    args = parser.parse_args()
    result = run()
    with args.output.open("x") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(json.dumps({"output": str(args.output), "train": {k: result["train"][k] for k in ("n_pairs", "n_binary", "labels")},
                      "external": {k: result["external"][k] for k in ("n_pairs", "n_binary", "labels")},
                      "metrics": {k: v["overall_binary"] for k, v in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
