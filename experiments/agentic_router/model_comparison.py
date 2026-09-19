"""Small, leakage-checked comparison on blind paired-call labels.

Run with ``python -m experiments.agentic_router.model_comparison``.  Only
aggregate metrics are written; raw prompts and candidate responses stay in
their existing gitignored files.  Every feature is regenerated from the
original chronological trace, never read from a replay row's old ``derived``
dictionary.  All models receive identical leave-task-out and leave-repo-out
folds.  These folds are diagnostic: with four original groups and 18 HARM
labels, no deployment model or threshold is selected here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import analysis, features, quality_pipeline

RESULTS = Path(__file__).resolve().parent / "results"
REPLAY_FILES = (
    "stage1_replay_examples.jsonl",
    "stage1_replay_examples_batch2.jsonl",
    "stage1_replay_examples_batch3.jsonl",
)
JUDGE_FILE = "stage1_judge_consistency.jsonl"
RESERVED_PILOT_HOLDOUT_TASKS = (
    "swebench-pytest-c89d81bc3b",  # pytest 10081
    "swebench-pylint-a520f18b1b",  # pylint 4551
)


@dataclass(frozen=True)
class LabeledEvent:
    call_id: str
    timestamp_unix_s: float | None
    trajectory_id: str
    task_group: str
    label: str  # SAFE | HARM | UNKNOWN
    features: dict[str, float]
    request_text: str = ""


def request_text_for_model(request: dict[str, Any]) -> str:
    """Bounded pre-dispatch user text: task anchor plus latest feedback.

    Assistant candidate responses and gold fixes are absent.  We do not use
    task_group/repository metadata as a feature.  A first/last user window
    keeps the initial issue statement and recent tool feedback without
    serializing the whole codebase into TF-IDF.
    """
    def text_parts(content: Any) -> list[str]:
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            return [part for item in content for part in text_parts(item)]
        if isinstance(content, dict):
            kind = content.get("type")
            if kind == "text":
                return text_parts(content.get("text"))
            if kind == "tool_result":
                return text_parts(content.get("content"))
        return []
    user = ["\n".join(text_parts(msg.get("content"))) for msg in request.get("messages", [])
            if isinstance(msg, dict) and msg.get("role") == "user"]
    if not user:
        return ""
    first = user[0][:4000]
    latest = user[-1][-4000:] if len(user) > 1 else ""
    return first + ("\n[RECENT FEEDBACK]\n" + latest if latest else "")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo(group: str) -> str:
    # SWE-bench group names end in an opaque issue hash; repo is the
    # intervening slug.  Distinct issues in one repo must stay together.
    parts = group.removeprefix("swebench-").rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(f"unparseable task group: {group}")
    return parts[0]


def load_events(
    results: Path = RESULTS,
    replay_files: tuple[str, ...] = REPLAY_FILES,
    judge_file: str = JUDGE_FILE,
) -> tuple[list[LabeledEvent], dict[str, Any]]:
    replay: dict[str, dict[str, Any]] = {}
    excluded_status: set[str] = set()
    excluded_quarantine: set[str] = set()
    capacity_censored: set[str] = set()
    quarantine_path = results / "quarantine_manifest.json"
    quarantined = set()
    if quarantine_path.exists():
        quarantined = set(json.loads(quarantine_path.read_text()).get(
            "paired_call_ids_retained_not_for_quality_training", []))
    capacity_quarantine_path = results / "pilot_judge_capacity_invalid_quarantine_manifest.json"
    if capacity_quarantine_path.exists():
        quarantined.add(json.loads(capacity_quarantine_path.read_text())["call_id"])
    all_replay_ids: set[str] = set()
    for name in replay_files:
        for row in _jsonl(results / name):
            call = row["call"]
            cid = call["call_id"]
            if cid in all_replay_ids:
                raise ValueError(f"duplicate replay call_id: {cid}")
            all_replay_ids.add(cid)
            if row.get("local_outcome", {}).get("status") == "INVALID_CAPACITY":
                capacity_censored.add(cid)
            if cid in quarantined:
                excluded_quarantine.add(cid)
                continue
            if (row.get("local_outcome", {}).get("status") != "OK" or
                    row.get("cloud_outcome", {}).get("status") != "OK"):
                excluded_status.add(cid)
                continue
            replay[cid] = call

    judge = _jsonl(results / judge_file)
    judge_versions = sorted({
        str(r.get("truncation_policy_version", "unspecified")) + "|" +
        str(r.get("primary_order_policy_version", "legacy-fixed-primary-order"))
        for r in judge
    })
    if len(judge_versions) > 1:
        raise ValueError(f"mixed judge protocol versions: {judge_versions}")
    key_counts = Counter((r["call_id"], r["pass_label"]) for r in judge)
    dupes = [key for key, n in key_counts.items() if n != 1]
    if dupes:
        raise ValueError(f"duplicate judge pass, first: {dupes[0]}")
    invalid_judge_ids = {r["call_id"] for r in judge} & (excluded_status | excluded_quarantine)
    if invalid_judge_ids:
        raise ValueError(f"judge rows for {len(invalid_judge_ids)} nonlabelable replay calls")
    labels = {l.call_id: l.label for l in quality_pipeline.build_call_labels(judge)}
    missing_labels = set(replay) - set(labels)
    if missing_labels:
        raise ValueError(f"missing complete judge passes for {len(missing_labels)} replay calls")

    # Sampling scattered replay calls alone cannot recover intervening
    # placements; rebuild complete original trajectories.
    fresh_by_id = features.regenerate_replay_calls([{"call": c} for c in replay.values()])

    events: list[LabeledEvent] = []
    for cid, call in replay.items():
        fresh = fresh_by_id[cid]
        label = labels[cid]
        events.append(LabeledEvent(cid, fresh.timestamp_unix_s, fresh.trajectory_id,
                                   fresh.task_group, label, analysis.call_features(fresh),
                                   request_text_for_model(fresh.request)))
    counts = Counter(labels.values())
    per_group_labels: dict[str, Counter] = {}
    for e in events:
        per_group_labels.setdefault(e.task_group, Counter())[e.label] += 1
    per_group_missing = {
        group: {name: sum(e.features[name] == analysis._MISSING for e in events if e.task_group == group)
                for name in analysis.FEATURE_NAMES}
        for group in per_group_labels
    }
    return events, {
        "n_replay_raw": len(all_replay_ids),
        "n_capacity_censored": len(capacity_censored),
        "n_excluded_arm_status": len(excluded_status),
        "n_excluded_quarantine": len(excluded_quarantine),
        "n_replay": len(replay), "n_judge_passes": len(judge),
        "labels": dict(counts), "n_scored": sum(e.label != "UNKNOWN" for e in events),
        "labels_by_task_group": {g: dict(c) for g, c in per_group_labels.items()},
        "missing_feature_count_by_task_group": per_group_missing,
        "n_task_groups": len({e.task_group for e in events}),
        "n_repos": len({_repo(e.task_group) for e in events}),
        "unknown_share": counts["UNKNOWN"] / len(replay) if replay else None,
        "feature_names": list(analysis.FEATURE_NAMES),
        "feature_source": "whole original trace, chronological reconstruction",
        "request_text_version": "first-user-4000-latest-user-4000-v1",
        "feature_derivation_version": features.PREDECISION_FEATURE_VERSION,
        "judge_protocol_versions": judge_versions,
        "replay_files": list(replay_files), "judge_file": judge_file,
        "replay_sha256": {name: _sha256(results / name) for name in replay_files},
        "judge_sha256": _sha256(results / judge_file),
        "source_trace_sha256": {path: _sha256(Path(path))
                                for path in sorted({c["source_trace_path"] for c in replay.values()})},
        "reserved_pilot_holdout_tasks": list(RESERVED_PILOT_HOLDOUT_TASKS),
    }


def load_rows(
    results: Path = RESULTS,
    replay_files: tuple[str, ...] = REPLAY_FILES,
    judge_file: str = JUDGE_FILE,
) -> tuple[list[analysis.CallRow], dict[str, Any]]:
    events, meta = load_events(results, replay_files, judge_file)
    return [analysis.CallRow(e.features, int(e.label == "HARM"), e.task_group)
            for e in events if e.label != "UNKNOWN"], meta


def _model(kind: str):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=120, max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=0)
    if kind == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))
    if kind == "gbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=80, num_leaves=7, learning_rate=0.05, min_child_samples=8, class_weight="balanced", verbose=-1, random_state=0)
    raise ValueError(kind)


def _risk_curve(y: np.ndarray, p: np.ndarray) -> list[dict[str, float]]:
    # Route the lowest predicted-risk prefix local.  Harm among routed
    # examples is observable in this paired sample; task success is not.
    order = np.argsort(p, kind="stable")
    out = []
    for share in (0.25, 0.5, 0.75, 1.0):
        n = max(1, int(len(y) * share))
        idx = order[:n]
        out.append({"local_share": n / len(y), "harm_among_local": float(y[idx].mean()), "n_local": n})
    return out


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 5) -> float:
    # Descriptive only: sparse positive labels make calibration uncertain.
    err = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (p >= lo) & (p <= hi if hi == 1 else p < hi)
        if mask.any():
            err += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return err


def _fold(train: list[analysis.CallRow], test: list[analysis.CallRow], kind: str) -> dict[str, Any]:
    ytr = np.asarray([r.label for r in train], dtype=int)
    yte = np.asarray([r.label for r in test], dtype=int)
    base = {"n_train": len(train), "n_test": len(test), "harm_train": int(ytr.sum()), "harm_test": int(yte.sum())}
    if not train or not test or len(set(ytr)) < 2:
        return {**base, "skipped": "empty fold or single-class train"}
    Xtr, Xte = analysis._matrix(train), analysis._matrix(test)
    try:
        model = _model(kind)
    except ImportError:
        return {**base, "skipped": "dependency unavailable"}
    if kind == "knn":
        model[-1].n_neighbors = min(5, len(train))
    model.fit(Xtr, ytr)
    p = np.asarray(model.predict_proba(Xte)[:, 1], dtype=float)
    return {
        **base,
        "roc_auc": float(roc_auc_score(yte, p)) if len(set(yte)) == 2 else None,
        "average_precision": float(average_precision_score(yte, p)) if yte.sum() else None,
        "brier": float(brier_score_loss(yte, p)),
        "ece_5_bins": _ece(yte, p),
        "mean_predicted_harm": float(p.mean()),
        "harm_base_rate": float(yte.mean()),
        "local_share_harm_curve": _risk_curve(yte, p),
    }


def compare(rows: list[analysis.CallRow], meta: dict[str, Any]) -> dict[str, Any]:
    heldout_rows = [r for r in rows if r.task_group in RESERVED_PILOT_HOLDOUT_TASKS]
    rows = [r for r in rows if r.task_group not in RESERVED_PILOT_HOLDOUT_TASKS]
    groups = sorted({r.task_group for r in rows})
    repos = sorted({_repo(g) for g in groups})
    out: dict[str, Any] = {
        "status": "diagnostic_only_no_model_selected",
        "meta": meta,
        "caveat": "Blind judge labels are proxies, folds are tiny, and same-task calls are correlated. No task-success guarantee or safe live threshold follows.",
        "holdout_reservation": {
            "task_groups": list(RESERVED_PILOT_HOLDOUT_TASKS),
            "n_labeled_holdout_rows": len(heldout_rows),
            "rule": "excluded from every development fold, model choice, and threshold; final fixed-logreg test only",
        },
        "folds": {},
    }
    for split_name, heldouts, key in (
        ("leave_task_out", groups, lambda r: r.task_group),
        ("leave_repo_out", repos, lambda r: _repo(r.task_group)),
    ):
        out["folds"][split_name] = {}
        for kind in ("logreg", "gbm", "rf", "knn"):
            folds = {}
            for held in heldouts:
                tr = [r for r in rows if key(r) != held]
                te = [r for r in rows if key(r) == held]
                folds[held] = _fold(tr, te, kind)
            aucs = [f["roc_auc"] for f in folds.values() if f.get("roc_auc") is not None]
            out["folds"][split_name][kind] = {
                "folds": folds,
                "unweighted_mean_auc": float(np.mean(aucs)) if aucs else None,
                "fold_auc_min": float(np.min(aucs)) if aucs else None,
                "fold_auc_max": float(np.max(aucs)) if aucs else None,
                "fold_auc_stdev": float(np.std(aucs)) if len(aucs) > 1 else None,
                "n_auc_folds": len(aucs),
            }
    return out


def evaluate_reserved_holdout(rows: list[analysis.CallRow], frozen_artifact: dict[str, Any],
                              artifact_sha256: str) -> dict[str, Any]:
    """Explicit final test against frozen coefficients, never a refit."""
    if frozen_artifact.get("schema_version") != "agentic-harm-shadow-v1":
        raise ValueError("unsupported frozen artifact schema")
    if tuple(frozen_artifact.get("feature_names", ())) != analysis.FEATURE_NAMES:
        raise ValueError("frozen artifact feature mismatch")
    heldout = [r for r in rows if r.task_group in RESERVED_PILOT_HOLDOUT_TASKS]
    if not heldout:
        raise ValueError("no labeled reserved holdout rows")
    X = analysis._matrix(heldout)
    y = np.asarray([r.label for r in heldout], dtype=int)
    means = np.asarray(frozen_artifact["standardization"]["means"])
    stds = np.asarray(frozen_artifact["standardization"]["stds"])
    coef = np.asarray(frozen_artifact["coefficients"])
    z = ((X - means) / stds) @ coef + frozen_artifact["intercept"]
    p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    return {
        "status": "one_time_frozen_artifact_test",
        "frozen_artifact_sha256": artifact_sha256,
        "n": len(heldout), "n_harm": int(y.sum()),
        "n_tasks": len({r.task_group for r in heldout}),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
        "brier": float(brier_score_loss(y, p)),
        "local_share_harm_curve": _risk_curve(y, p),
        "note": "The frozen artifact is shadow-only; these are judge proxy labels, not task success.",
    }


def shadow_logreg_artifact(rows: list[analysis.CallRow], meta: dict[str, Any]) -> dict[str, Any]:
    """Fit a fixed, deliberately non-deployable reference on development rows.

    The threshold is train-only and disables local proposals.  The tiny old
    data and uncalibrated judge proxy cannot certify a useful quality budget;
    the coefficients exist for reproducibility and shadow scoring only.
    """
    train = [r for r in rows if r.task_group not in RESERVED_PILOT_HOLDOUT_TASKS]
    X = analysis._matrix(train)
    y = np.asarray([r.label for r in train], dtype=int)
    if len(set(y)) != 2:
        raise ValueError("shadow fit needs both SAFE and HARM examples")
    model = _model("logreg")
    model.fit(X, y)
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    return {
        "schema_version": "agentic-harm-shadow-v1",
        "status": "shadow_only_no_live_activation",
        "activation_allowed": False,
        "feature_names": list(analysis.FEATURE_NAMES),
        "missing_sentinel": analysis._MISSING,
        "standardization": {"means": scaler.mean_.tolist(), "stds": scaler.scale_.tolist()},
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "positive_class": "HARM_LOCAL",
        "calibration": "none; class_weight=balanced makes raw logistic scores non-calibrated",
        "local_risk_threshold": 0.0,
        "threshold_rule": "training-only conservative fallback; never propose local from this artifact",
        "n_train": len(train), "n_train_harm": int(y.sum()),
        "n_train_groups": len({r.task_group for r in train}),
        "reserved_pilot_holdout_tasks": list(RESERVED_PILOT_HOLDOUT_TASKS),
        "training_provenance": {"replay_files": meta["replay_files"], "judge_file": meta["judge_file"],
                                "replay_sha256": meta["replay_sha256"],
                                "judge_sha256": meta["judge_sha256"],
                                "source_trace_sha256": meta["source_trace_sha256"],
                                "features": meta["feature_source"],
                                "support_filter": meta.get("support_filter"),
                                "n_scored_excluded_by_support_filter": meta.get("n_scored_excluded_by_support_filter"),
                                "feature_derivation_version": meta["feature_derivation_version"],
                                "judge_protocol_versions": meta["judge_protocol_versions"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--replay-file", action="append", dest="replay_files")
    parser.add_argument("--judge-file", default=JUDGE_FILE)
    parser.add_argument("--output-stem", default="quality_model_baseline205_v1")
    parser.add_argument("--evaluate-reserved-holdout", action="store_true")
    parser.add_argument("--frozen-artifact", type=Path)
    parser.add_argument("--ablate-repair-loop", action="store_true",
                        help="set repaired prior-action loop flag to zero for a controlled feature ablation")
    parser.add_argument("--require-complete-context", action="store_true",
                        help="evaluate only calls where exact local prompt and budget were available to the live scorer")
    args = parser.parse_args()
    rows, meta = load_rows(args.results, tuple(args.replay_files or REPLAY_FILES), args.judge_file)
    if args.require_complete_context:
        n_before = len(rows)
        rows = [r for r in rows if r.features["local_prompt_tokens"] != analysis._MISSING
                and r.features["context_utilization_ratio"] != analysis._MISSING]
        meta["support_filter"] = "exact local prompt and context ratio available"
        meta["n_scored_excluded_by_support_filter"] = n_before - len(rows)
        meta["n_scored_after_support_filter"] = len(rows)
    if args.ablate_repair_loop:
        rows = [analysis.CallRow({**r.features, "repair_loop_flag": 0.0}, r.label, r.task_group)
                for r in rows]
        meta["ablation"] = "repair_loop_flag forced to zero after identical label/trace loading"
    output = compare(rows, meta)
    can_fit_shadow = len({r.label for r in rows if r.task_group not in RESERVED_PILOT_HOLDOUT_TASKS}) == 2 and len({r.task_group for r in rows if r.task_group not in RESERVED_PILOT_HOLDOUT_TASKS}) >= 3
    if not can_fit_shadow:
        output["status"] = "insufficient_development_groups_or_harm_examples"
    path = args.results / f"{args.output_stem}_comparison.json"
    artifact_path = args.results / f"{args.output_stem}_shadow_logreg.json"
    if path.exists() or (artifact_path.exists() and not args.evaluate_reserved_holdout and not args.ablate_repair_loop and can_fit_shadow):
        raise FileExistsError("versioned outputs already exist; choose a new --output-stem")
    if args.evaluate_reserved_holdout:
        if args.frozen_artifact is None:
            parser.error("--evaluate-reserved-holdout requires --frozen-artifact")
        raw_artifact = args.frozen_artifact.read_bytes()
        frozen = json.loads(raw_artifact)
        output["final_reserved_holdout"] = evaluate_reserved_holdout(
            rows, frozen, hashlib.sha256(raw_artifact).hexdigest())
    path.write_text(json.dumps(output, indent=2) + "\n")
    if not args.evaluate_reserved_holdout and not args.ablate_repair_loop and can_fit_shadow:
        artifact_path.write_text(json.dumps(shadow_logreg_artifact(rows, meta), indent=2) + "\n")
    print(json.dumps({"path": str(path), "meta": meta, "task_auc": {k: v["unweighted_mean_auc"] for k, v in output["folds"]["leave_task_out"].items()}, "repo_auc": {k: v["unweighted_mean_auc"] for k, v in output["folds"]["leave_repo_out"].items()}}, indent=2))


if __name__ == "__main__":
    main()
