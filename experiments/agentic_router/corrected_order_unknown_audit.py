"""Prompt-free UNKNOWN/reversal audit on frozen v6/v7/clean-v8 pairs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from .corrected_order_prevalence import JUDGE, REPLAY, ROOT, V8_CLEAN, _events, _read_jsonl
from .quality_pipeline import aggregate_call_label


def _arm_choice(row: dict) -> str:
    verdict = row["verdict"]
    if verdict == "A_BETTER":
        return row["order"][0]
    if verdict == "B_BETTER":
        return row["order"][1]
    return verdict


def _unknown_mode(primary: dict, reversed_: dict) -> str:
    verdicts = {primary["verdict"], reversed_["verdict"]}
    if "PARSE_ERROR" in verdicts:
        return "parse_error"
    if "UNCERTAIN" in verdicts:
        return "explicit_uncertain"
    if "BOTH_INADEQUATE" in verdicts:
        return "both_inadequate"
    a, b = _arm_choice(primary), _arm_choice(reversed_)
    if {a, b} == {"local", "cloud"}:
        return "opposite_directional_winners"
    if "EQUIVALENT" in (a, b) and a != b:
        return "directional_vs_equivalent"
    return "other_inconsistency"


def _response_type(outcome: dict) -> str:
    response = outcome.get("response")
    if not isinstance(response, dict):
        return "missing_response"
    types = [block.get("type") for block in response.get("content", []) if isinstance(block, dict)]
    if "tool_use" in types:
        return "tool_call"
    if "text" in types:
        return "text_only"
    return "other_no_tool"


def _pair_type(replay_row: dict) -> str:
    local = _response_type(replay_row["local_outcome"])
    cloud = _response_type(replay_row["cloud_outcome"])
    return f"local_{local}__cloud_{cloud}"


def run() -> dict:
    items = []
    provenance = {}
    for version in (6, 7, 8):
        events, meta = _events(version)  # fail-closed cohort/protocol/completeness check
        directory = ROOT / f"postpilot_v{version}"
        ids = {e.call_id for e in events}
        replay = {r["call"]["call_id"]: r for r in _read_jsonl(directory / REPLAY)
                  if r["call"]["call_id"] in ids}
        passes = defaultdict(dict)
        for row in _read_jsonl(directory / JUDGE):
            if row["call_id"] in ids:
                passes[row["call_id"]][row["pass_label"]] = row
        if set(replay) != ids or set(passes) != ids:
            raise ValueError(f"v{version}: incomplete replay/judge join")
        if version == 8 and {e.task_group for e in events} != V8_CLEAN:
            raise ValueError("v8 clean cohort changed")
        for event in events:
            p, r = passes[event.call_id]["primary"], passes[event.call_id]["reversed"]
            label = aggregate_call_label(p, r)
            if label != event.label:
                raise ValueError("label contract mismatch")
            items.append({"version": version, "task_group": event.task_group,
                          "label": label, "primary_arm_choice": _arm_choice(p),
                          "reverse_arm_choice": _arm_choice(r),
                          "primary_a_arm": p["order"][0],
                          "unknown_mode": _unknown_mode(p, r) if label == "UNKNOWN" else None,
                          "pair_type": _pair_type(replay[event.call_id]),
                          "local_type": _response_type(replay[event.call_id]["local_outcome"]),
                          "cloud_type": _response_type(replay[event.call_id]["cloud_outcome"])})
        provenance[f"v{version}"] = meta
    if len(items) != 80 or Counter(item["label"] for item in items) != {"SAFE": 53, "HARM": 8, "UNKNOWN": 19}:
        raise ValueError("frozen cohort labels changed")
    by_task = defaultdict(list)
    by_pair = defaultdict(list)
    by_version = defaultdict(list)
    for item in items:
        by_task[item["task_group"]].append(item)
        by_pair[item["pair_type"]].append(item)
        by_version[f"v{item['version']}"].append(item)

    def summary(rows: list[dict]) -> dict:
        return {"n": len(rows), "labels": dict(Counter(x["label"] for x in rows)),
                "unknown_modes": dict(Counter(x["unknown_mode"] for x in rows if x["label"] == "UNKNOWN"))}

    harms = [x for x in items if x["label"] == "HARM"]
    unknowns = [x for x in items if x["label"] == "UNKNOWN"]
    return {"provenance": provenance, "overall": summary(items),
            "by_version": {k: summary(v) for k, v in sorted(by_version.items())},
            "by_task": {k: summary(v) for k, v in sorted(by_task.items())},
            "by_candidate_pair_type": {k: summary(v) for k, v in sorted(by_pair.items())},
            "all_harm_both_cloud_winner": all(x["primary_arm_choice"] == x["reverse_arm_choice"] == "cloud" for x in harms),
            "harm_primary_arm_choices": dict(Counter(x["primary_arm_choice"] for x in harms)),
            "harm_reverse_arm_choices": dict(Counter(x["reverse_arm_choice"] for x in harms)),
            "harm_primary_a_arm": dict(Counter(x["primary_a_arm"] for x in harms)),
            "unknown_primary_a_arm": dict(Counter(x["primary_a_arm"] for x in unknowns)),
            "unknown_arm_choice_pairs": dict(Counter(f"{x['primary_arm_choice']}|{x['reverse_arm_choice']}" for x in unknowns)),
            "all_candidate_local_types": dict(Counter(x["local_type"] for x in items)),
            "all_candidate_cloud_types": dict(Counter(x["cloud_type"] for x in items))}


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2))
