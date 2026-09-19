"""Deterministic action-validity audit of v3 UNKNOWN calls; no relabeling."""

from __future__ import annotations

import json
from collections import Counter

from edgeproxy.trace.record import validate_tool_use_blocks

from . import model_comparison, quality_pipeline


def main() -> None:
    results = model_comparison.RESULTS
    replay = {}
    for name in model_comparison.REPLAY_FILES:
        for row in model_comparison._jsonl(results / name):
            replay[row["call"]["call_id"]] = row
    labels = quality_pipeline.build_call_labels(model_comparison._jsonl(results / model_comparison.JUDGE_FILE))
    unknown = [x for x in labels if x.label == "UNKNOWN"]
    categories = Counter()
    groups = Counter()
    statuses = Counter()
    details = []
    for entry in unknown:
        row = replay[entry.call_id]
        request = row["call"]["request"]
        local = validate_tool_use_blocks(request, row["local_outcome"].get("response"))
        cloud = validate_tool_use_blocks(request, row["cloud_outcome"].get("response"))
        lv = all(b["schema_valid"] for b in local) if local else None
        cv = all(b["schema_valid"] for b in cloud) if cloud else None
        if lv is False and cv is True or lv is True and cv is False:
            category = "one_invalid_one_valid"
        elif lv is None or cv is None:
            category = "one_or_both_no_tool_calls"
        else:
            category = "both_schema_valid"
        categories[category] += 1
        groups[row["call"]["task_group"]] += 1
        statuses[(row["local_outcome"]["status"], row["cloud_outcome"]["status"])] += 1
        details.append({"call_id": entry.call_id, "task_group": row["call"]["task_group"],
                        "primary_verdict": entry.primary_verdict,
                        "reversed_verdict": entry.reversed_verdict,
                        "local_tool_valid": lv, "cloud_tool_valid": cv, "category": category})
    out = {"status": "no_relabeling", "judge_file": model_comparison.JUDGE_FILE,
           "judge_sha256": model_comparison._sha256(results / model_comparison.JUDGE_FILE),
           "n_unknown": len(unknown), "categories": dict(categories),
           "task_groups": dict(groups),
           "statuses": {f"{a}|{b}": n for (a, b), n in statuses.items()},
           "verdict_pairs": {f"{a}|{b}": n for (a, b), n in Counter((x.primary_verdict, x.reversed_verdict) for x in unknown).items()},
           "calls": details,
           "note": "Tool schema validity alone cannot resolve the observed UNKNOWN calls; manual or execution evidence required."}
    path = results / "unknown_v3_validity_audit.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("n_unknown", "categories", "task_groups", "statuses")}, indent=2))


if __name__ == "__main__":
    main()
