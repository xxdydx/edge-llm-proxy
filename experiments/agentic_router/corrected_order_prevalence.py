"""Offline descriptive audit of completed corrected-order v6/v7 and clean v8 calls.

No model is trained. In-flight v8 Matplotlib and quarantined v8 Pylint are
excluded by exact task IDs, even if their files happen to exist later.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import analysis, features, model_comparison, quality_pipeline

ROOT = Path(__file__).resolve().parent / "results"
REPLAY = "stage1_replay_examples_pilot_v1.jsonl"
JUDGE = "pilot_judge_consistency_v5_order.jsonl"
PROTOCOL = "v4-system-messages-tools-2026-09-16|v5-sha256-call-id-2026-09-17"
V8_CLEAN = {"swebench-pytest-4c87720313", "swebench-scikit-learn-a4a59eac2d"}
EXPECTED = {6: 3, 7: 5, 8: 2}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(version: int) -> tuple[list[model_comparison.LabeledEvent], dict]:
    directory = ROOT / f"postpilot_v{version}"
    replay_path, judge_path = directory / REPLAY, directory / JUDGE
    raw = _read_jsonl(replay_path)
    judge = _read_jsonl(judge_path)
    if version == 8:
        raw = [r for r in raw if r["call"]["task_group"] in V8_CLEAN]
        ids = {r["call"]["call_id"] for r in raw}
        judge = [r for r in judge if r["call_id"] in ids]
    else:
        ids = {r["call"]["call_id"] for r in raw}
    if len(ids) != EXPECTED[version] * 8 or len(judge) != 2 * len(ids):
        raise ValueError(f"v{version}: cohort incomplete, duplicated or changed")
    passes = Counter((r["call_id"], r["pass_label"]) for r in judge)
    if set(passes.values()) != {1} or {r["call_id"] for r in judge} != ids:
        raise ValueError(f"v{version}: missing or duplicate judge passes")
    for cid in ids:
        if passes[(cid, "primary")] != 1 or passes[(cid, "reversed")] != 1:
            raise ValueError(f"v{version}: incomplete reverse judgment")
    protocols = {str(r.get("truncation_policy_version")) + "|" + str(r.get("primary_order_policy_version")) for r in judge}
    if protocols != {PROTOCOL}:
        raise ValueError(f"v{version}: wrong judge protocol: {protocols}")
    labels = {r.call_id: r.label for r in quality_pipeline.build_call_labels(judge)}
    if set(labels) != ids:
        raise ValueError(f"v{version}: missing label")
    calls = features.regenerate_replay_calls(raw)
    events = [model_comparison.LabeledEvent(cid, calls[cid].timestamp_unix_s, calls[cid].trajectory_id,
              calls[cid].task_group, labels[cid], analysis.call_features(calls[cid])) for cid in ids]
    return events, {"replay_sha256": _sha(replay_path), "judge_sha256": _sha(judge_path),
                    "protocol": PROTOCOL, "selected_pairs": len(events), "judge_passes": len(judge)}


def _task_metadata(version: int) -> dict[str, dict]:
    directory = ROOT / f"postpilot_v{version}"
    selection = json.loads((directory / "pilot_selection_manifest.json").read_text())["trajectories"]
    out = {}
    for item in selection.values():
        group = item["task_group"]
        if version == 8 and group not in V8_CLEAN:
            continue
        verdict = (Path("eval-suite/swebench/results") /
                   f"agentic-router-postpilot-v{version}" / "verdicts" / f"{group}__cloud__seed1.json")
        grade = json.loads(verdict.read_text())
        if grade["passed"] not in (True, False):
            raise ValueError(f"{group}: no official PASS/FAIL grade")
        out[group] = {"selected_ids": item["call_ids"], "indices": item["indices"],
                      "n_eligible": item["n_eligible_main_calls"], "outcome": "PASS" if grade["passed"] else "FAIL",
                      "selection_protocol": item.get("selection_protocol", "v6-six-fixed-anchors-two-random-interior"),
                      "inclusion_probability_by_call_id": item.get("inclusion_probability_by_call_id"),
                      "nonanchor_inclusion_probability": item.get("nonanchor_inclusion_probability")}
    if len(out) != EXPECTED[version]:
        raise ValueError(f"v{version}: wrong task count")
    return out


def _counts(rows: list[dict]) -> dict:
    labels = Counter(r["label"] for r in rows)
    known = labels["SAFE"] + labels["HARM"]
    return {"pairs": len(rows), "safe": labels["SAFE"], "harm": labels["HARM"],
            "unknown": labels["UNKNOWN"], "known": known,
            "harm_per_known": labels["HARM"] / known if known else None,
            "harm_per_all_lower_bound": labels["HARM"] / len(rows) if rows else None,
            "harm_per_all_upper_bound_if_all_unknown_harm": (labels["HARM"] + labels["UNKNOWN"]) / len(rows) if rows else None}


def _cluster_bootstrap(rows: list[dict], seed: int = 20260917) -> dict:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_group"]].append(row)
    groups = sorted(by_task)
    if len(groups) < 2:
        return {"n_tasks": len(groups), "range": None}
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(10000):
        sampled = [row for group in rng.choice(groups, size=len(groups), replace=True) for row in by_task[group]]
        c = _counts(sampled)
        if c["known"]:
            rates.append(c["harm_per_known"])
    return {"n_tasks": len(groups), "n_valid_resamples": len(rates),
            "percentile_2_5_97_5": np.percentile(rates, [2.5, 97.5]).tolist() if rates else None,
            "warning": "Selected-call, known-label descriptive task bootstrap; not a population-risk or validated coverage interval."}


def run() -> dict:
    rows: list[dict] = []
    source = {}
    for version in (6, 7, 8):
        events, provenance = _events(version)
        task_meta = _task_metadata(version)
        source[f"v{version}"] = provenance
        for event in events:
            meta = task_meta[event.task_group]
            offset = meta["selected_ids"].index(event.call_id)
            index = meta["indices"][offset]
            n = meta["n_eligible"]
            if index < 1 or index > n:
                raise ValueError(f"{event.task_group}: invalid selected index")
            position = "early" if index / n <= 1/3 else "middle" if index / n <= 2/3 else "late"
            f = event.features
            rows.append({"version": version, "task_group": event.task_group, "outcome": meta["outcome"],
                         "label": event.label, "eligible_index": index, "n_eligible": n, "position": position,
                         "endpoint": index in (1, n),
                         "pi": (meta["inclusion_probability_by_call_id"] or {}).get(event.call_id),
                         "prior_tool_error": f["recent_tool_error_count"] > 0,
                         "prior_test_failure": f["recent_test_failure_count"] > 0,
                         "repair_loop": f["repair_loop_flag"] > 0})
    if len(rows) != 80 or len({r["task_group"] for r in rows}) != 10:
        raise ValueError("cohort changed; audit requires re-review")
    groupings = {
        "by_version": lambda r: f"v{r['version']}",
        "by_outcome": lambda r: r["outcome"],
        "by_position": lambda r: r["position"],
        "by_endpoint": lambda r: "endpoint" if r["endpoint"] else "interior",
        "by_prior_tool_error": lambda r: str(r["prior_tool_error"]),
        "by_prior_test_failure": lambda r: str(r["prior_test_failure"]),
        "by_repair_loop": lambda r: str(r["repair_loop"]),
        "by_task": lambda r: r["task_group"],
    }
    out = {"source": source, "overall": _counts(rows), "n_tasks": 10, "groups": {}}
    for name, key in groupings.items():
        buckets = defaultdict(list)
        for row in rows:
            buckets[key(row)].append(row)
        out["groups"][name] = {k: {**_counts(v), "task_cluster_bootstrap": _cluster_bootstrap(v)}
                               for k, v in sorted(buckets.items())}
    out["task_cluster_bootstrap"] = _cluster_bootstrap(rows)
    out["row_metadata"] = [{k: v for k, v in row.items() if k not in ("task_group",)} for row in rows]
    out["limitations"] = ["v6 fixed six anchors plus two random interior; v7/v8 one randomly chosen endpoint plus seven uniform interior, with unequal inclusion probabilities.",
                          "Judge UNKNOWNs are excluded from HARM/known but not presumed missing at random.",
                          "A task's cloud PASS/FAIL is not a counterfactual local trajectory outcome.",
                          "Repeated calls are clustered within only ten tasks; no pre-call flag is activated from this audit."]
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps({"overall": result["overall"], "groups": {k: v for k, v in result["groups"].items() if k != "by_task"}}, indent=2))
