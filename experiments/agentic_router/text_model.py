"""Request-text and structural baselines on paired agent-call quality labels.

All vectorizers and scalers fit on the training side of each task/repository
fold.  Labels are blind-judge proxies, not execution-backed task outcomes.
The reserved two-task Python pilot holdout is never used for model choice.

Run: ``uv run --no-sync python -m experiments.agentic_router.text_model``.
Use explicit ``--replay-file`` and ``--judge-file`` for a separate v4 pilot;
mixing judge protocol versions is rejected upstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from . import analysis, model_comparison


def _text_matrix(train: list[model_comparison.LabeledEvent], test: list[model_comparison.LabeledEvent],
                 kind: str):
    if kind == "word":
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                     max_features=4000, sublinear_tf=True)
    elif kind == "char":
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1,
                                     max_features=6000, sublinear_tf=True)
    else:
        raise ValueError(kind)
    train_text = [e.request_text or "(empty request text)" for e in train]
    test_text = [e.request_text or "(empty request text)" for e in test]
    return vectorizer.fit_transform(train_text), vectorizer.transform(test_text), vectorizer


def _fit_predict(train: list[model_comparison.LabeledEvent], test: list[model_comparison.LabeledEvent],
                 kind: str) -> np.ndarray:
    text_kind = "char" if kind == "char_structural" else "word"
    Xtr, Xte, _ = _text_matrix(train, test, text_kind)
    ytr = np.asarray([int(e.label == "HARM") for e in train])
    if kind == "lexical_knn":
        similarities = cosine_similarity(Xte, Xtr)
        risks = []
        for sims in similarities:
            idx = np.argsort(sims)[-min(5, len(sims)):]
            weights = np.maximum(sims[idx], 0)
            risks.append(float((weights @ ytr[idx]) / weights.sum()) if weights.sum() else float(ytr.mean()))
        return np.asarray(risks)
    if kind in ("word_structural", "char_structural"):
        numeric_train = np.asarray([[e.features[n] for n in analysis.FEATURE_NAMES] for e in train])
        numeric_test = np.asarray([[e.features[n] for n in analysis.FEATURE_NAMES] for e in test])
        scaler = StandardScaler()
        Xtr = hstack([Xtr, csr_matrix(scaler.fit_transform(numeric_train))], format="csr")
        Xte = hstack([Xte, csr_matrix(scaler.transform(numeric_test))], format="csr")
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    model.fit(Xtr, ytr)
    return np.asarray(model.predict_proba(Xte)[:, 1])


def _fold(train: list[model_comparison.LabeledEvent], test: list[model_comparison.LabeledEvent],
          kind: str) -> dict[str, Any]:
    ytr = np.asarray([int(e.label == "HARM") for e in train])
    yte = np.asarray([int(e.label == "HARM") for e in test])
    base = {"n_train": len(train), "n_test": len(test), "harm_train": int(ytr.sum()), "harm_test": int(yte.sum())}
    if not train or not test or len(set(ytr)) < 2:
        return {**base, "skipped": "empty fold or single-class training"}
    p = _fit_predict(train, test, kind)
    return {**base,
            "roc_auc": float(roc_auc_score(yte, p)) if len(set(yte)) == 2 else None,
            "average_precision": float(average_precision_score(yte, p)) if yte.sum() else None,
            "brier": float(brier_score_loss(yte, p)),
            "local_share_harm_curve": model_comparison._risk_curve(yte, p)}


def compare(events: list[model_comparison.LabeledEvent], meta: dict[str, Any]) -> dict[str, Any]:
    dev = [e for e in events if e.label != "UNKNOWN"
           and e.task_group not in model_comparison.RESERVED_PILOT_HOLDOUT_TASKS]
    groups = sorted({e.task_group for e in dev})
    repos = sorted({model_comparison._repo(g) for g in groups})
    out = {"status": "diagnostic_only_no_model_selected", "meta": meta,
           "n_development": len(dev), "n_development_groups": len(groups),
           "reserved_holdout_tasks": list(model_comparison.RESERVED_PILOT_HOLDOUT_TASKS),
           "models": ["word_only", "word_structural", "char_structural", "lexical_knn"],
           "folds": {},
           "caveat": "Text can encode issue/repository identity; leave-repo-out is essential. Folds are tiny and judge labels are proxy outcomes."}
    for split, heldouts, key in (
        ("leave_task_out", groups, lambda e: e.task_group),
        ("leave_repo_out", repos, lambda e: model_comparison._repo(e.task_group)),
    ):
        out["folds"][split] = {}
        for kind in out["models"]:
            folds = {held: _fold([e for e in dev if key(e) != held],
                                 [e for e in dev if key(e) == held], kind)
                     for held in heldouts}
            aucs = [f["roc_auc"] for f in folds.values() if f.get("roc_auc") is not None]
            out["folds"][split][kind] = {
                "folds": folds, "n_auc_folds": len(aucs),
                "unweighted_mean_auc": float(np.mean(aucs)) if aucs else None,
                "fold_auc_min": float(np.min(aucs)) if aucs else None,
                "fold_auc_max": float(np.max(aucs)) if aucs else None,
                "fold_auc_stdev": float(np.std(aucs)) if len(aucs) > 1 else None,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=model_comparison.RESULTS)
    parser.add_argument("--replay-file", action="append", dest="replay_files")
    parser.add_argument("--judge-file", default=model_comparison.JUDGE_FILE)
    parser.add_argument("--output-stem", default="quality_text_baseline205_v1")
    args = parser.parse_args()
    events, meta = model_comparison.load_events(args.results,
                                                 tuple(args.replay_files or model_comparison.REPLAY_FILES),
                                                 args.judge_file)
    output = compare(events, meta)
    path = args.results / f"{args.output_stem}_comparison.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"path": str(path), "n_dev": output["n_development"],
                      "task_auc": {k: v["unweighted_mean_auc"] for k, v in output["folds"]["leave_task_out"].items()},
                      "repo_auc": {k: v["unweighted_mean_auc"] for k, v in output["folds"]["leave_repo_out"].items()}}, indent=2))


if __name__ == "__main__":
    main()
