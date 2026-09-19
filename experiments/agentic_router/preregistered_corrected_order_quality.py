"""Frozen, offline corrected-order v6/v7/v8 quality comparison.

The module does not read data or train on import. Real execution requires
``--run-real`` after the v8 terminal snapshot and explicit parent approval.
See the v1 memo and its v1.1 pre-run amendment in results/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import analysis, collect, features, model_comparison, quality_pipeline, text_model

RESULTS = Path(__file__).resolve().parent / "results"
REPO = Path(__file__).resolve().parents[2]
REPLAY = "stage1_replay_examples_pilot_v1.jsonl"
JUDGE = "pilot_judge_consistency_v5_order.jsonl"
PROTOCOL = "v4-system-messages-tools-2026-09-16|v5-sha256-call-id-2026-09-17"
EXCLUDED_TASKS = {"swebench-pylint-edcd13a009": "v8 quarantined INVALID_CAPACITY/incomplete judge"}
LOCAL_TOTAL = 100_000
LOCAL_OUTPUT = 32_000
HEADROOM = 4_096
HISTORY_SUPPORT_VERSION = "all-prior-complete-at-dispatch-v1"
KINDS = ("numeric_logistic", "numeric_ridge", "word_structural", "shallow_forest")
MIN_REPOS = 5
MIN_BINARY = 40
MIN_HARM = 10
MIN_HARM_REPOS = 3


@dataclass(frozen=True)
class Event:
    call_id: str
    task_group: str
    label: str
    features: dict[str, float]
    request_text: str
    namespace: str
    inclusion_probability: float | None
    supported: bool
    exclusion_reason: str | None
    selection_stratum: str = "unspecified"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_numeric(value: Any) -> int | float | bool:
    """Serialize NumPy scalar metrics without changing computed values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not a JSON numeric scalar: {type(value).__name__}")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _history_completion_reason(candidate: Any, full_calls: list[Any]) -> str | None:
    """Conservative availability gate before using index-derived history.

    Lane identity is not recoverable in these old traces. Every earlier trace
    call is therefore potentially relevant. A response that overlaps this
    dispatch, or whose completion cannot be proved, makes history unsupported.
    """
    start = _finite(candidate.timestamp_unix_s)
    if start is None:
        return "candidate_dispatch_time_unknown"
    for prior in full_calls:
        if prior.call_id == candidate.call_id:
            break
        prior_start = _finite(prior.timestamp_unix_s)
        total_ms = _finite(prior.timing.get("total_ms")) if isinstance(prior.timing, dict) else None
        if prior_start is None or total_ms is None or total_ms < 0:
            return "prior_completion_time_unknown"
        if (not isinstance(prior.response, dict)
                or not isinstance(prior.response.get("stop_reason"), str)
                or prior.placement not in ("local", "cloud")):
            return "prior_response_incomplete"
        if prior_start >= start or prior_start + total_ms / 1000.0 > start:
            return "earlier_call_in_flight_at_dispatch"
    else:
        return "candidate_absent_from_source_chronology"
    return None


def _repo(event: Event) -> str:
    return model_comparison._repo(event.task_group)


def _support(features_: dict[str, float], requested_output: Any, local_status: str) -> str | None:
    if local_status != "OK":
        return f"local_replay_{local_status.lower()}"
    if any(name not in features_ or features_[name] == analysis._MISSING for name in analysis.FEATURE_NAMES):
        return "missing_predecision_feature"
    prompt = features_["local_prompt_tokens"]
    if not isinstance(requested_output, int) or isinstance(requested_output, bool) or requested_output <= 0:
        return "missing_output_budget"
    if requested_output > LOCAL_OUTPUT or prompt + requested_output + HEADROOM > LOCAL_TOTAL:
        return "local_capacity"
    return None


def _full_calls(raw_call: dict, path: Path) -> list[Any]:
    trajectory_id = raw_call["trajectory_id"]
    condition = trajectory_id.rsplit(":", 2)[-2]
    seed = int(trajectory_id.rsplit(":seed", 1)[-1])
    trajectory = collect.build_trajectory(
        collect.load_trace_records(path), task_group=raw_call["task_group"],
        campaign=raw_call["source_campaign"], condition=condition, seed=seed,
        trace_path=path, task_passed=None, verdict_detail=None,
        verdict_path=Path("unused"),
    )
    return trajectory.calls


def snapshot_input_hashes(root: Path) -> dict[str, str | None]:
    """Hash every mutable real-data input, including absent terminal files.

    Called only after an explicit real-run flag plus approval manifest exists.
    Exact mapping equality prevents silent v6/v7/v8 data or verdict changes.
    """
    paths: set[Path] = set()
    for version in (6, 7, 8):
        folder = root / f"postpilot_v{version}"
        replay = folder / REPLAY
        judge = folder / JUDGE
        selection_path = folder / "pilot_selection_manifest.json"
        quarantine = folder / "pilot_judge_capacity_invalid_quarantine_manifest.json"
        paths.update((replay, judge, selection_path, quarantine))
        if version == 8:
            paths.update(folder / name for name in (
                "pilot_terminal_partial_manifest.json",
                "quarantine_manifest.json",
                "pilot_judge_capacity_invalid_quarantine.jsonl",
                "pilot_judge_consistency_v5_order_pre_capacity_quarantine.jsonl",
                "pilot_campaign_status.json",
            ))
        selection = json.loads(selection_path.read_text())["trajectories"]
        for item in selection.values():
            group = item["task_group"]
            paths.add(REPO / "eval-suite/swebench/results" / f"agentic-router-postpilot-v{version}" /
                      "verdicts" / f"{group}__cloud__seed1.json")
        paths.update(Path(row["call"]["source_trace_path"]) for row in _jsonl(replay))
    return {str(path.resolve()): _hash(path) if path.is_file() else None
            for path in sorted(paths, key=str)}


def _load_namespace(version: int, root: Path) -> tuple[list[Event], dict]:
    namespace = f"postpilot_v{version}"
    folder = root / namespace
    replay_path = folder / REPLAY
    judge_path = folder / JUDGE
    selection_path = folder / "pilot_selection_manifest.json"
    replay_rows, judge_rows = _jsonl(replay_path), _jsonl(judge_path)
    selection = json.loads(selection_path.read_text())["trajectories"]
    by_task_selection = {item["task_group"]: item for item in selection.values()}
    if len(by_task_selection) != len(selection):
        raise ValueError(f"{namespace}: duplicate task in selection manifest")
    replay = {}
    for row in replay_rows:
        cid = row["call"]["call_id"]
        if cid in replay:
            raise ValueError(f"{namespace}: duplicate replay call")
        replay[cid] = row
    passes: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in judge_rows:
        cid, pass_label = row["call_id"], row["pass_label"]
        if pass_label not in ("primary", "reversed") or pass_label in passes[cid]:
            raise ValueError(f"{namespace}: duplicate/unrecognized judge pass")
        protocol = str(row.get("truncation_policy_version")) + "|" + str(row.get("primary_order_policy_version"))
        if protocol != PROTOCOL:
            raise ValueError(f"{namespace}: judge protocol mismatch")
        passes[cid][pass_label] = row
    if set(passes) - set(replay):
        raise ValueError(f"{namespace}: judge ID has no replay")
    source_calls: dict[str, list[Any]] = {}
    for row in replay_rows:
        raw_call = row["call"]
        path_str = raw_call["source_trace_path"]
        if path_str not in source_calls:
            path = Path(path_str)
            if not path.is_file():
                raise FileNotFoundError(f"source chronology unavailable: {path}")
            source_calls[path_str] = _full_calls(raw_call, path)
    if version in (7, 8):
        for group, item in by_task_selection.items():
            selected = [replay[cid] for cid in item["call_ids"] if cid in replay]
            if not selected:
                continue  # incomplete task will be excluded below
            paths = {row["call"]["source_trace_path"] for row in selected}
            if len(paths) != 1 or _hash(Path(next(iter(paths)))) != item.get("source_trace_sha256"):
                raise ValueError(f"{namespace}: selection source trace SHA-256 mismatch for {group}")
    fresh = features.regenerate_replay_calls(replay_rows)
    quarantine: set[str] = set()
    quarantine_path = folder / "pilot_judge_capacity_invalid_quarantine_manifest.json"
    if quarantine_path.exists():
        quarantine.add(str(json.loads(quarantine_path.read_text())["call_id"]))
    events: list[Event] = []
    task_flow: dict[str, dict] = {}
    grade_sha256: dict[str, str] = {}
    for group, item in sorted(by_task_selection.items()):
        if group in model_comparison.RESERVED_PILOT_HOLDOUT_TASKS:
            raise ValueError("reserved holdout appears in development manifest")
        selected_ids = item["call_ids"]
        flow = {"selected": len(selected_ids), "included": 0, "excluded": Counter()}
        task_flow[group] = flow
        if group in EXCLUDED_TASKS:
            flow["excluded"][EXCLUDED_TASKS[group]] = len(selected_ids)
            continue
        verdict = REPO / "eval-suite/swebench/results" / f"agentic-router-postpilot-v{version}" / "verdicts" / f"{group}__cloud__seed1.json"
        if not verdict.exists() or json.loads(verdict.read_text()).get("passed") not in (True, False):
            flow["excluded"]["no_terminal_official_grade"] = len(selected_ids)
            continue
        grade_sha256[group] = _hash(verdict)
        if len(selected_ids) != 8 or len(set(selected_ids)) != 8:
            flow["excluded"]["incomplete_selection"] = len(selected_ids)
            continue
        for cid in selected_ids:
            if cid not in replay:
                flow["excluded"]["unattempted_or_missing_replay"] += 1
                continue
            if cid in quarantine:
                flow["excluded"]["quarantined_invalid_capacity"] += 1
                continue
            row = replay[cid]
            call = fresh[cid]
            chronology_reason = _history_completion_reason(call, source_calls[call.source_trace_path])
            if chronology_reason:
                flow["excluded"][chronology_reason] += 1
                continue
            feature_values = analysis.call_features(call)
            reason = _support(feature_values, call.request.get("max_tokens"), row["local_outcome"].get("status", "MISSING"))
            if row["cloud_outcome"].get("status") != "OK":
                reason = "cloud_replay_not_ok"
            if reason:
                flow["excluded"][reason] += 1
                continue
            if set(passes.get(cid, {})) != {"primary", "reversed"}:
                flow["excluded"]["missing_judge_pass"] += 1
                continue
            label = quality_pipeline.aggregate_call_label(passes[cid]["primary"], passes[cid]["reversed"])
            pi = item.get("inclusion_probability_by_call_id", {}).get(cid)
            if version in (7, 8) and (not isinstance(pi, (int, float)) or not 0 < pi <= 1):
                raise ValueError(f"{namespace}: missing/invalid inclusion probability")
            eligible_ids = item.get("eligible_call_ids", [])
            if version in (7, 8):
                if not eligible_ids or cid not in eligible_ids:
                    raise ValueError(f"{namespace}: selected call missing from eligible frame")
                stratum = "random_endpoint" if cid in (eligible_ids[0], eligible_ids[-1]) else "uniform_interior"
            else:
                stratum = "six_anchor_two_random_design"
            events.append(Event(cid, group, label, feature_values,
                                model_comparison.request_text_for_model(call.request), namespace,
                                float(pi) if pi is not None else None, True, None, stratum))
            flow["included"] += 1
    trace_paths = sorted({row["call"]["source_trace_path"] for row in replay_rows
                          if row["call"]["task_group"] in by_task_selection})
    return events, {"namespace": namespace, "replay_sha256": _hash(replay_path),
                    "judge_sha256": _hash(judge_path), "selection_sha256": _hash(selection_path),
                    "quarantine_sha256": _hash(quarantine_path) if quarantine_path.exists() else None,
                    "official_grade_sha256_by_task": grade_sha256,
                    "source_trace_sha256": {path: _hash(Path(path)) for path in trace_paths},
                    "task_flow": {k: {**v, "excluded": dict(v["excluded"])} for k, v in task_flow.items()}}


def reportability(events: list[Event]) -> dict:
    binary = [e for e in events if e.supported and e.label in ("SAFE", "HARM")]
    harms = [e for e in binary if e.label == "HARM"]
    n_repos = len({_repo(e) for e in binary})
    harm_repos = len({_repo(e) for e in harms})
    passed = n_repos >= MIN_REPOS and len(binary) >= MIN_BINARY and len(harms) >= MIN_HARM and harm_repos >= MIN_HARM_REPOS
    return {"passed": passed, "n_repos": n_repos, "n_binary": len(binary), "n_harm": len(harms),
            "n_harm_repos": harm_repos, "requirements": {"repos": MIN_REPOS, "binary": MIN_BINARY,
            "harm": MIN_HARM, "harm_repos": MIN_HARM_REPOS}}


def _matrix(events: list[Event]) -> np.ndarray:
    return np.asarray([[e.features[n] for n in analysis.FEATURE_NAMES] for e in events], dtype=float)


def _fit_predict(kind: str, train: list[Event], test: list[Event]) -> np.ndarray:
    if kind == "word_structural":
        # Helper fits its vectorizer, scaler and logistic only on training rows.
        return text_model._fit_predict(train, test, "word_structural")
    y = np.asarray([int(e.label == "HARM") for e in train])
    if kind == "numeric_logistic":
        model = model_comparison._model("logreg")
    elif kind == "numeric_ridge":
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10, class_weight="balanced"))
    elif kind == "shallow_forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=6,
                                       max_features="sqrt", class_weight="balanced_subsample",
                                       random_state=20260917, n_jobs=1)
    else:
        raise ValueError(kind)
    model.fit(_matrix(train), y)
    if kind == "numeric_ridge":
        return np.asarray(model.decision_function(_matrix(test)), dtype=float)
    return np.asarray(model.predict_proba(_matrix(test))[:, 1], dtype=float)


def _metrics(events: list[Event], scores: np.ndarray, probability: bool) -> dict:
    y = np.asarray([int(e.label == "HARM") for e in events])
    if not len(events):
        return {"n": 0}
    out = {"n": len(events), "harm": int(y.sum()), "harm_prevalence": float(y.mean()),
           "auc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
           "ap": float(average_precision_score(y, scores)) if int(y.sum()) else None,
           "score_min": float(scores.min()), "score_max": float(scores.max())}
    if probability:
        out["brier_diagnostic"] = float(brier_score_loss(y, scores))
        out["ece_5_bin_diagnostic"] = model_comparison._ece(y, scores)
    return out


def _cluster_sensitivity(scored: list[tuple[Event, float]]) -> dict:
    repos = sorted({_repo(e) for e, _ in scored})
    informative = [repo for repo in repos if len({e.label for e, _ in scored if _repo(e) == repo}) == 2]
    if len(informative) < 5:
        return {"status": "not_reported", "informative_repos": len(informative),
                "reason": "fewer_than_five_two_class_repositories"}
    rng = np.random.default_rng(20260917)
    by_repo = {repo: [(e, s) for e, s in scored if _repo(e) == repo] for repo in repos}
    aucs, aps = [], []
    for _ in range(10_000):
        draw = [pair for repo in rng.choice(repos, len(repos), replace=True) for pair in by_repo[repo]]
        y = np.asarray([int(e.label == "HARM") for e, _ in draw])
        if len(set(y)) < 2:
            continue
        p = np.asarray([score for _, score in draw])
        aucs.append(float(roc_auc_score(y, p)))
        aps.append(float(average_precision_score(y, p)))
    return {"status": "descriptive_only_not_validated_ci", "informative_repos": len(informative),
            "valid_draws": len(aucs), "auc_percentile_2_5_97_5": np.percentile(aucs, [2.5, 97.5]).tolist(),
            "ap_percentile_2_5_97_5": np.percentile(aps, [2.5, 97.5]).tolist()}


def _selection_sensitivity(scored: list[tuple[Event, float]]) -> dict:
    newer = [(e, s) for e, s in scored if e.namespace in ("postpilot_v7", "postpilot_v8")]
    if not newer:
        return {"status": "no_v7_v8_scores"}
    by_task = Counter(e.task_group for e, _ in newer)
    weights = np.asarray([1 / e.inclusion_probability / by_task[e.task_group]
                          for e, _ in newer], dtype=float)
    # Normalize 1/pi weights within each task before task balancing.
    task_totals = defaultdict(float)
    for (event, _), weight in zip(newer, weights):
        task_totals[event.task_group] += weight
    weights = np.asarray([weight / task_totals[event.task_group]
                          for (event, _), weight in zip(newer, weights)])
    y = np.asarray([int(e.label == "HARM") for e, _ in newer])
    scores = np.asarray([score for _, score in newer])
    if len(set(y)) < 2:
        return {"status": "single_class_v7_v8", "n_binary": len(y)}
    return {"status": "diagnostic_known_label_only", "n_binary": len(y),
            "unweighted_auc": float(roc_auc_score(y, scores)),
            "task_balanced_inverse_inclusion_auc": float(roc_auc_score(y, scores, sample_weight=weights)),
            "unweighted_ap": float(average_precision_score(y, scores)),
            "task_balanced_inverse_inclusion_ap": float(average_precision_score(y, scores, sample_weight=weights)),
            "warning": "Conditioned on binary-judged selected calls; unequal UNKNOWN and task selection remain uncorrected."}


def _inner_threshold(train: list[Event]) -> dict:
    check = reportability(train)
    if not (check["n_binary"] >= 60 and check["n_harm"] >= 10 and
            check["n_harm_repos"] >= 3 and check["n_repos"] >= 5):
        return {"status": "cloud_all", "reason": "inner_sample_requirements_not_met"}
    by_repo = sorted({_repo(e) for e in train})
    scores: list[tuple[float, str]] = []
    for repo in by_repo:
        inner_train = [e for e in train if _repo(e) != repo]
        inner_test = [e for e in train if _repo(e) == repo]
        if len({e.label for e in inner_train}) < 2:
            return {"status": "cloud_all", "reason": "single_class_inner_training_fold"}
        scores.extend(zip(_fit_predict("numeric_logistic", inner_train, inner_test),
                          (e.label for e in inner_test)))
    candidates = sorted({float(score) for score, _ in scores})
    qualifying = [t for t in candidates if sum(label == "HARM" and score <= t for score, label in scores) == 0
                  and sum(label == "SAFE" and score <= t for score, label in scores) >= 20]
    if not qualifying:
        return {"status": "cloud_all", "reason": "no_zero_miss_threshold_with_20_safe"}
    return {"status": "illustrative_threshold", "threshold": max(qualifying),
            "inner_binary": len(scores), "inner_safe_routed_local": sum(label == "SAFE" and score <= max(qualifying)
                                                               for score, label in scores)}


def compare(events: list[Event]) -> dict:
    flow = {"labels": dict(Counter(e.label for e in events)), "supported": sum(e.supported for e in events),
            "n_tasks": len({e.task_group for e in events}), "n_repos": len({_repo(e) for e in events})}
    flow["by_namespace"] = {namespace: {"labels": dict(Counter(e.label for e in events if e.namespace == namespace)),
                                       "n_tasks": len({e.task_group for e in events if e.namespace == namespace})}
                            for namespace in sorted({e.namespace for e in events})}
    flow["by_repo"] = {repo: dict(Counter(e.label for e in events if _repo(e) == repo))
                       for repo in sorted({_repo(e) for e in events})}
    flow["by_selection_stratum"] = {
        stratum: dict(Counter(e.label for e in events if e.selection_stratum == stratum))
        for stratum in sorted({e.selection_stratum for e in events})}
    harm = sum(e.label == "HARM" for e in events)
    unknown = sum(e.label == "UNKNOWN" for e in events)
    flow["unknown_extreme_bounds_selected_pairs"] = {
        "all_unknown_safe": harm / len(events) if events else None,
        "all_unknown_harm": (harm + unknown) / len(events) if events else None,
        "warning": "Only the selected, supported pairs; unselected eligible calls remain unknown."}
    flow["binary_label_coverage"] = {
        "binary": sum(e.label in ("SAFE", "HARM") for e in events),
        "unknown": unknown,
        "fraction_binary": (len(events) - unknown) / len(events) if events else None,
    }
    check = reportability(events)
    if not check["passed"]:
        return {"status": "counts_only_reportability_gate_failed", "flow": flow,
                "reportability": check,
                "cost_latency_status": "unavailable_without_true_counterfactual_usage_and_serving_times"}
    binary = [e for e in events if e.supported and e.label in ("SAFE", "HARM")]
    repos = sorted({_repo(e) for e in binary})
    folds: dict[str, Any] = {}
    pooled: dict[str, list[tuple[Event, float]]] = {kind: [] for kind in KINDS}
    for repo in repos:
        train = [e for e in binary if _repo(e) != repo]
        test = [e for e in binary if _repo(e) == repo]
        if len({e.label for e in train}) < 2:
            folds[repo] = {"status": "single_class_training", "n_test": len(test)}
            continue
        result = {"status": "scored", "n_test": len(test), "labels": dict(Counter(e.label for e in test)), "models": {}}
        for kind in KINDS:
            scores = _fit_predict(kind, train, test)
            result["models"][kind] = _metrics(test, scores, kind != "numeric_ridge")
            pooled[kind].extend(zip(test, scores))
        threshold = _inner_threshold(train)
        held_all = [e for e in events if _repo(e) == repo and e.supported]
        if threshold["status"] == "illustrative_threshold":
            held_scores = _fit_predict("numeric_logistic", train, held_all)
            local = held_scores <= threshold["threshold"]
        else:
            local = np.zeros(len(held_all), dtype=bool)
        threshold["outer_supported_n"] = len(held_all)
        threshold["outer_unknown_n"] = sum(e.label == "UNKNOWN" for e in held_all)
        threshold["outer_local_share"] = float(local.mean()) if len(local) else None
        threshold["outer_cloud_fraction"] = float(1 - local.mean()) if len(local) else None
        threshold["outer_harm_routed_local"] = sum(e.label == "HARM" and choose
                                                    for e, choose in zip(held_all, local))
        threshold["outer_unknown_routed_local"] = sum(e.label == "UNKNOWN" and choose
                                                       for e, choose in zip(held_all, local))
        threshold["cost_latency_status"] = "unavailable_without_true_counterfactual_usage_and_serving_times"
        result["threshold_screen"] = threshold
        folds[repo] = result
    pooled_metrics = {}
    for kind, scored in pooled.items():
        if scored:
            pooled_metrics[kind] = _metrics([e for e, _ in scored], np.asarray([s for _, s in scored]),
                                            kind != "numeric_ridge")
            pooled_metrics[kind]["informative_repo_macro"] = {
                "n_informative": sum(fold.get("status") == "scored" and fold["models"][kind]["auc"] is not None
                                     for fold in folds.values()),
                "n_outer_repos": len(repos),
                "auc": float(np.mean([fold["models"][kind]["auc"] for fold in folds.values()
                                      if fold.get("status") == "scored" and fold["models"][kind]["auc"] is not None]))
                if any(fold.get("status") == "scored" and fold["models"][kind]["auc"] is not None
                       for fold in folds.values()) else None,
                "ap": float(np.mean([fold["models"][kind]["ap"] for fold in folds.values()
                                     if fold.get("status") == "scored" and fold["models"][kind]["auc"] is not None]))
                if any(fold.get("status") == "scored" and fold["models"][kind]["auc"] is not None
                       for fold in folds.values()) else None,
            }
            pooled_metrics[kind]["repo_cluster_sensitivity"] = _cluster_sensitivity(scored)
            pooled_metrics[kind]["v7_v8_selection_sensitivity"] = _selection_sensitivity(scored)
    return {"status": "diagnostic_only_no_activation", "flow": flow, "reportability": check,
            "folds": folds, "pooled_out_of_fold": pooled_metrics,
            "cost_latency_status": "unavailable_without_true_counterfactual_usage_and_serving_times",
            "warning": "Selected-call proxy labels, nonuniform v6/v7/v8 sampling, repo-cluster dependence, and uncalibrated scores."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-real", action="store_true", help="Requires terminal v8 snapshot and parent approval")
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--approval-manifest", type=Path,
                        help="Parent-approved terminal JSON with exact snapshot_input_hashes mapping")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_real:
        parser.error("real cohort loading/training is disabled; request approval before --run-real")
    if args.approval_manifest is None:
        parser.error("--approval-manifest is required to freeze the terminal v8 snapshot")
    approval = json.loads(args.approval_manifest.read_text())
    if approval.get("terminal") is not True or approval.get("parent_approved") is not True:
        parser.error("v8 snapshot is not explicitly terminal and parent-approved")
    if approval.get("inputs_sha256") != snapshot_input_hashes(args.results_root):
        parser.error("approved v6/v7/v8 input snapshot missing or changed")
    if args.output.exists():
        parser.error("output exists; immutable diagnostic must use a new path")
    all_events, provenance = [], {}
    for version in (6, 7, 8):
        events, meta = _load_namespace(version, args.results_root)
        all_events.extend(events)
        provenance[f"v{version}"] = meta
    result = compare(all_events)
    result["provenance"] = provenance
    result["protocol"] = PROTOCOL
    result["approval_manifest_sha256"] = _hash(args.approval_manifest)
    result["approved_input_hashes"] = approval["inputs_sha256"]
    result["model_settings"] = {"families": KINDS, "frozen_hyperparameter_grid": "one setting per family"}
    result["feature_derivation_version"] = features.PREDECISION_FEATURE_VERSION
    result["history_support_version"] = HISTORY_SUPPORT_VERSION
    result["feature_semantics"] = {
        "recent_tool_error_count": "number of up to three prior requests with nonzero cumulative errored_tool_result_density; not distinct new errors",
    }
    with args.output.open("x") as fh:
        json.dump(result, fh, indent=2, default=_json_numeric)
        fh.write("\n")
    print(json.dumps({"status": result["status"], "flow": result["flow"],
                      "reportability": result["reportability"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
