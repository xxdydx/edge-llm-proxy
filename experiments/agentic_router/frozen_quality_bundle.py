"""Frozen v1.1 development predictor bundle and future v9 shadow scorer.

No files are read and no models are fit on import. Fitting is explicitly
limited to the approved terminal v6/v7/v8 snapshot. Scoring v9 requires a
separate approved v9 input-hash manifest; it never trains or tunes on v9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import analysis, features, model_comparison
from .preregistered_corrected_order_quality import (
    Event, HISTORY_SUPPORT_VERSION, JUDGE, KINDS, PROTOCOL, REPLAY, REPO,
    RESULTS, _hash, _json_numeric, _jsonl, _load_namespace, _matrix,
    _repo, snapshot_input_hashes,
)

TRAIN_COUNTS = {"pairs": 82, "binary": 64, "safe": 52, "harm": 12, "unknown": 18,
                "tasks": 14, "repos": 8}
ARTIFACT_NAMES = {
    "numeric_logistic": "numeric_logistic.joblib",
    "numeric_ridge": "numeric_ridge.joblib",
    "word_structural": "word_structural.joblib",
    "shallow_forest": "shallow_forest.joblib",
}


def _fit_all(binary: list[Event]) -> dict[str, Any]:
    y = np.asarray([int(e.label == "HARM") for e in binary])
    if set(y) != {0, 1}:
        raise ValueError("training needs SAFE and HARM")
    x = _matrix(binary)
    numeric_logistic = model_comparison._model("logreg").fit(x, y)
    numeric_ridge = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10, class_weight="balanced")).fit(x, y)
    shallow_forest = RandomForestClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=6, max_features="sqrt",
        class_weight="balanced_subsample", random_state=20260917, n_jobs=1,
    ).fit(x, y)
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                 max_features=4000, sublinear_tf=True)
    text_matrix = vectorizer.fit_transform([e.request_text or "(empty request text)" for e in binary])
    scaler = StandardScaler()
    word_matrix = hstack([text_matrix, csr_matrix(scaler.fit_transform(x))], format="csr")
    word_model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(word_matrix, y)
    return {"numeric_logistic": numeric_logistic, "numeric_ridge": numeric_ridge,
            "word_structural": {"vectorizer": vectorizer, "scaler": scaler, "model": word_model},
            "shallow_forest": shallow_forest}


def _score(models: dict[str, Any], events: list[Event]) -> dict[str, np.ndarray]:
    if not events:
        return {kind: np.asarray([], dtype=float) for kind in KINDS}
    if any(not e.supported or e.exclusion_reason for e in events):
        raise ValueError("unsupported external row cannot be scored")
    if any(any(e.features.get(name) == analysis._MISSING or name not in e.features
               for name in analysis.FEATURE_NAMES) for e in events):
        raise ValueError("missing predecision input cannot be scored")
    x = _matrix(events)
    word = models["word_structural"]
    text = word["vectorizer"].transform([e.request_text or "(empty request text)" for e in events])
    word_x = hstack([text, csr_matrix(word["scaler"].transform(x))], format="csr")
    return {
        "numeric_logistic": np.asarray(models["numeric_logistic"].predict_proba(x)[:, 1]),
        "numeric_ridge": np.asarray(models["numeric_ridge"].decision_function(x)),
        "word_structural": np.asarray(word["model"].predict_proba(word_x)[:, 1]),
        "shallow_forest": np.asarray(models["shallow_forest"].predict_proba(x)[:, 1]),
    }


def _training_audit(events: list[Event]) -> dict:
    binary = [e for e in events if e.supported and e.label in ("SAFE", "HARM")]
    counts = Counter(e.label for e in events)
    audit = {"pairs": len(events), "binary": len(binary), "safe": counts["SAFE"],
             "harm": counts["HARM"], "unknown": counts["UNKNOWN"],
             "tasks": len({e.task_group for e in events}), "repos": len({_repo(e) for e in events})}
    if audit != TRAIN_COUNTS:
        raise ValueError(f"frozen development count changed: {audit}")
    return {"counts": audit,
            "by_namespace": {namespace: dict(Counter(e.label for e in events if e.namespace == namespace))
                             for namespace in sorted({e.namespace for e in events})},
            "task_groups": sorted({e.task_group for e in events}),
            "repos": sorted({_repo(e) for e in events}),
            "binary_label_digest": hashlib.sha256("\n".join(
                f"{e.call_id}:{e.label}" for e in sorted(binary, key=lambda event: event.call_id)
            ).encode()).hexdigest()}


def fit_development(approval_path: Path, output_dir: Path, root: Path = RESULTS) -> dict:
    approval = json.loads(approval_path.read_text())
    if approval.get("terminal") is not True or approval.get("parent_approved") is not True:
        raise ValueError("unapproved development snapshot")
    if approval.get("inputs_sha256") != snapshot_input_hashes(root):
        raise ValueError("development snapshot hash mismatch")
    if output_dir.exists():
        raise FileExistsError(f"immutable bundle path already exists: {output_dir}")
    events, provenance = [], {}
    for version in (6, 7, 8):
        cohort, meta = _load_namespace(version, root)
        events.extend(cohort)
        provenance[f"v{version}"] = meta
    audit = _training_audit(events)
    binary = [e for e in events if e.label in ("SAFE", "HARM")]
    models = _fit_all(binary)
    output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for kind in KINDS:
        path = output_dir / ARTIFACT_NAMES[kind]
        with path.open("xb") as fh:
            joblib.dump(models[kind], fh)
        hashes[kind] = _hash(path)
    manifest = {
        "status": "frozen_shadow_predictors_not_activated",
        "training_audit": audit,
        "training_provenance": provenance,
        "approval_manifest_sha256": _hash(approval_path),
        "approved_input_hashes": approval["inputs_sha256"],
        "judge_protocol": PROTOCOL,
        "predecision_feature_version": features.PREDECISION_FEATURE_VERSION,
        "history_support_version": HISTORY_SUPPORT_VERSION,
        "feature_names": list(analysis.FEATURE_NAMES),
        "request_text_version": "first-user-4000-latest-user-4000-v1",
        "models": {kind: {"file": ARTIFACT_NAMES[kind], "sha256": hashes[kind]} for kind in KINDS},
        "decision_policy": {"quality_gate_active": False, "placement": "cloud_all", "threshold": None},
        "fit_code_sha256": _hash(Path(__file__)),
        "v1_1_runner_code_sha256": _hash(Path(__file__).with_name("preregistered_corrected_order_quality.py")),
        "libraries": {"numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "warning": "Selected blind-judge proxy labels only; causal history supported, live lane-feature parity unproven.",
    }
    with (output_dir / "manifest.json").open("x") as fh:
        json.dump(manifest, fh, indent=2, default=_json_numeric)
        fh.write("\n")
    return manifest


def load_bundle(bundle_dir: Path) -> tuple[dict, dict[str, Any]]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("judge_protocol") != PROTOCOL
            or manifest.get("predecision_feature_version") != features.PREDECISION_FEATURE_VERSION
            or manifest.get("history_support_version") != HISTORY_SUPPORT_VERSION
            or manifest.get("feature_names") != list(analysis.FEATURE_NAMES)
            or manifest.get("fit_code_sha256") != _hash(Path(__file__))
            or manifest.get("v1_1_runner_code_sha256") != _hash(Path(__file__).with_name("preregistered_corrected_order_quality.py"))
            or manifest.get("decision_policy") != {"quality_gate_active": False, "placement": "cloud_all", "threshold": None}):
        raise ValueError("bundle schema or shadow policy changed")
    models = {}
    for kind in KINDS:
        item = manifest["models"][kind]
        if item["file"] != ARTIFACT_NAMES[kind]:
            raise ValueError("unexpected artifact name")
        path = bundle_dir / item["file"]
        if _hash(path) != item["sha256"]:
            raise ValueError(f"model artifact hash mismatch: {kind}")
        # Only load the local, hash-verified joblib; never deserialize an
        # untrusted external artifact.
        models[kind] = joblib.load(path)
    return manifest, models


def _external_input_hashes(root: Path) -> dict[str, str | None]:
    folder = root / "postpilot_v9"
    paths = {folder / name for name in (REPLAY, JUDGE, "pilot_selection_manifest.json",
                                        "pilot_judge_capacity_invalid_quarantine_manifest.json",
                                        "quarantine_manifest.json", "pilot_terminal_partial_manifest.json")}
    selection = json.loads((folder / "pilot_selection_manifest.json").read_text())["trajectories"]
    for item in selection.values():
        group = item["task_group"]
        paths.add(REPO / "eval-suite/swebench/results/agentic-router-postpilot-v9" /
                  "verdicts" / f"{group}__cloud__seed1.json")
    paths.update(Path(row["call"]["source_trace_path"]) for row in _jsonl(folder / REPLAY))
    return {str(path.resolve()): _hash(path) if path.is_file() else None for path in sorted(paths, key=str)}


def score_v9(bundle_dir: Path, approval_path: Path, root: Path = RESULTS) -> dict:
    approval = json.loads(approval_path.read_text())
    if approval.get("terminal") is not True or approval.get("parent_approved") is not True:
        raise ValueError("unapproved v9 snapshot")
    if approval.get("inputs_sha256") != _external_input_hashes(root):
        raise ValueError("v9 snapshot hash mismatch")
    manifest, models = load_bundle(bundle_dir)
    folder = root / "postpilot_v9"
    selection = json.loads((folder / "pilot_selection_manifest.json").read_text())["trajectories"]
    replay = {row["call"]["call_id"]: row["call"] for row in _jsonl(folder / REPLAY)}
    for item in selection.values():
        rows = [replay[cid] for cid in item["call_ids"] if cid in replay]
        if not rows:
            continue
        paths = {row["source_trace_path"] for row in rows}
        if len(paths) != 1 or _hash(Path(next(iter(paths)))) != item.get("source_trace_sha256"):
            raise ValueError("v9 selection source trace SHA-256 mismatch")
    events, provenance = _load_namespace(9, root)  # reuses exact completion-time support gate
    train_tasks = set(manifest["training_audit"]["task_groups"])
    test_tasks = {e.task_group for e in events}
    if train_tasks & test_tasks:
        raise ValueError("v9 task identity overlaps development training")
    train_repos = set(manifest["training_audit"]["repos"])
    test_repos = {_repo(e) for e in events}
    scores = _score(models, events)
    binary = [i for i, e in enumerate(events) if e.label in ("SAFE", "HARM")]
    y = np.asarray([int(events[i].label == "HARM") for i in binary])
    metrics = {}
    for kind, all_scores in scores.items():
        selected = all_scores[binary]
        metrics[kind] = {"n_binary": len(binary), "harm": int(y.sum()),
                         "auc": float(roc_auc_score(y, selected)) if len(set(y)) == 2 else None,
                         "ap": float(average_precision_score(y, selected)) if int(y.sum()) else None}
    return {"status": "external_shadow_diagnostic_no_activation", "v9_provenance": provenance,
            "v9_approval_manifest_sha256": _hash(approval_path),
            "bundle_manifest_sha256": _hash(bundle_dir / "manifest.json"),
            "n_supported": len(events), "labels": dict(Counter(e.label for e in events)),
            "task_overlap": False, "repo_overlap": sorted(train_repos & test_repos),
            "repo_transfer_claim_supported": not bool(train_repos & test_repos),
            "models": metrics, "decision_policy": manifest["decision_policy"],
            "warning": "v9 judged selected calls only; UNKNOWN omitted from binary metrics; no threshold or routing decision changed."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--fit-development", action="store_true")
    modes.add_argument("--score-v9", action="store_true")
    parser.add_argument("--approval-manifest", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, help="Required for immutable v9 score JSON")
    args = parser.parse_args()
    if args.fit_development:
        if args.output:
            parser.error("fit mode writes its own immutable manifest in --bundle-dir")
        manifest = fit_development(args.approval_manifest, args.bundle_dir, args.results_root)
        print(json.dumps({"status": manifest["status"], "training_counts": manifest["training_audit"]["counts"],
                          "bundle_dir": str(args.bundle_dir)}, indent=2))
    else:
        if args.output is None:
            parser.error("--score-v9 requires --output")
        result = score_v9(args.bundle_dir, args.approval_manifest, args.results_root)
        with args.output.open("x") as fh:
            json.dump(result, fh, indent=2, default=_json_numeric)
            fh.write("\n")
        print(json.dumps({"status": result["status"], "n_supported": result["n_supported"],
                          "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
