"""Frozen-protocol, offline diagnostics for the five-group pilot plus v5 additions.

Keeps each source snapshot's hashes/provenance separate and never reads v3
judgments or reserved holdouts. This does not fit or activate a live router.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import analysis, model_comparison, text_model

RESULTS = Path(__file__).resolve().parent / "results"
V4 = RESULTS
V5 = RESULTS / "postpilot_v5"
REPLAY = "stage1_replay_examples_pilot_v1.jsonl"
JUDGE = "pilot_judge_consistency_v4.jsonl"
EXPECTED_VERSION = "v4-system-messages-tools-2026-09-16|legacy-fixed-primary-order"


def _events_with_meta():
    first, first_meta = model_comparison.load_events(V4, (REPLAY,), JUDGE)
    second, second_meta = model_comparison.load_events(V5, (REPLAY,), JUDGE)
    if first_meta["judge_protocol_versions"] != [EXPECTED_VERSION] or second_meta["judge_protocol_versions"] != [EXPECTED_VERSION]:
        raise ValueError("source judge protocol is not the frozen v4 version")
    if first_meta["n_replay"] != 40 or second_meta["n_replay"] != 16:
        raise ValueError("unexpected source cohort size; freeze a new version rather than silently append")
    if first_meta["n_task_groups"] != 5 or second_meta["n_task_groups"] != 2:
        raise ValueError("unexpected task-group count")
    if {e.call_id for e in first} & {e.call_id for e in second}:
        raise ValueError("duplicate call IDs across cohorts")
    if {e.task_group for e in first} & {e.task_group for e in second}:
        raise ValueError("duplicate task groups across cohorts")
    events = first + second
    reserved = set(model_comparison.RESERVED_PILOT_HOLDOUT_TASKS)
    if any(e.task_group in reserved for e in events):
        raise ValueError("reserved holdout row appeared in development data")
    groups = {e.task_group for e in events}
    if len(groups) != 7 or len({model_comparison._repo(g) for g in groups}) != 7:
        raise ValueError("expected seven distinct task/repository groups")
    counts = Counter(e.label for e in events)
    meta = {
        "status": "frozen_v4_development_diagnostic_only",
        "sources": {"pilot_five": first_meta, "postpilot_v5_two": second_meta},
        "n_replay": len(events),
        "labels": dict(counts),
        "n_scored": sum(e.label != "UNKNOWN" for e in events),
        "n_task_groups": len(groups),
        "n_repos": 7,
        "labels_by_task_group": {
            g: dict(Counter(e.label for e in events if e.task_group == g))
            for g in sorted(groups)
        },
        "missing_feature_count_by_task_group": {
            g: {name: sum(e.features[name] == analysis._MISSING for e in events if e.task_group == g)
                for name in analysis.FEATURE_NAMES}
            for g in sorted(groups)
        },
        "reserved_pilot_holdout_tasks": sorted(reserved),
        "judge_protocol_versions": [EXPECTED_VERSION],
        "feature_derivation_version": first_meta["feature_derivation_version"],
        "rule": "UNKNOWN excluded only from model folds, still counted in dataset and group coverage",
    }
    if first_meta["feature_derivation_version"] != second_meta["feature_derivation_version"]:
        raise ValueError("mixed feature derivation versions")
    return events, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", default="quality_pilot_v4_plus_postpilot_v5_dev56_v1")
    args = parser.parse_args()
    events, meta = _events_with_meta()
    scored = [e for e in events if e.label != "UNKNOWN"]
    rows = [analysis.CallRow(e.features, int(e.label == "HARM"), e.task_group) for e in scored]
    supported = [e for e in scored if e.features["local_prompt_tokens"] != analysis._MISSING
                 and e.features["context_utilization_ratio"] != analysis._MISSING]
    meta["complete_context_support"] = {
        "n_scored_supported": len(supported),
        "n_scored_unsupported": len(scored) - len(supported),
        "by_task_group": {
            g: {"supported": sum(e.task_group == g for e in supported),
                "scored": sum(e.task_group == g for e in scored)}
            for g in sorted({e.task_group for e in events})
        },
    }
    numeric = model_comparison.compare(rows, meta)
    text = text_model.compare(events, meta)
    supported_rows = [analysis.CallRow(e.features, int(e.label == "HARM"), e.task_group) for e in supported]
    numeric_supported = model_comparison.compare(supported_rows, meta)
    text_supported = text_model.compare(supported, meta)
    outputs = {
        "numeric": numeric,
        "text": text,
        "numeric_supported": numeric_supported,
        "text_supported": text_supported,
    }
    paths = {}
    for kind, result in outputs.items():
        path = RESULTS / f"{args.output_stem}_{kind}.json"
        with path.open("x") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
        paths[kind] = str(path)
    print(json.dumps({"paths": paths, "n_scored": len(scored), "labels": meta["labels"],
                      "support": meta["complete_context_support"]}, indent=2))


if __name__ == "__main__":
    main()
