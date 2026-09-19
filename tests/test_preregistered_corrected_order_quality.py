"""Synthetic-only tests: never load pending v8 replay/judge files."""

from __future__ import annotations

import sys
import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.agentic_router import analysis
from experiments.agentic_router import preregistered_corrected_order_quality as quality
from experiments.agentic_router.schema import AgentCallRecord
from experiments.agentic_router.preregistered_corrected_order_quality import (
    Event, _history_completion_reason, _inner_threshold, _support, compare, main, reportability,
)


def _event(repo: int, turn: int, label: str) -> Event:
    values = {name: 0.0 for name in analysis.FEATURE_NAMES}
    values.update({"turn_index": float(turn), "local_prompt_tokens": 10_000.0 + 100 * turn,
                   "n_available_tools": 3.0, "context_utilization_ratio": 0.2 + turn / 100,
                   "recent_tool_error_count": float(label == "HARM")})
    return Event(f"synthetic-{repo}-{turn}", f"swebench-repo{repo}-abcdef0123", label,
                 values, f"Fix synthetic case {repo}: test failure {turn}",
                 "postpilot_v7", 0.5, True, None)


def _call(call_id: str, start: float | None, total_ms: float | None = 1000,
          response: dict | None = None) -> AgentCallRecord:
    return AgentCallRecord(
        call_id=call_id, trajectory_id="synthetic:task:cloud:seed1", task_group="swebench-test-abcdef0123",
        source_campaign="synthetic", source_trace_path="synthetic.jsonl", record_index=0,
        turn_index=0, timestamp_unix_s=start, placement="cloud", policy=None, reason=None,
        request={"messages": []}, response={"content": [], "stop_reason": "end_turn"} if response is None else response,
        stop_reason=None, timing={"total_ms": total_ms},
    )


def test_capacity_and_feature_support_are_fail_closed() -> None:
    values = {name: 0.0 for name in analysis.FEATURE_NAMES}
    values["local_prompt_tokens"] = 60_000.0
    assert _support(values, 32_000, "OK") is None
    assert _support(values, 36_000, "OK") == "local_capacity"
    assert _support(values, 32_001, "OK") == "local_capacity"
    assert _support(values, 32_000, "INVALID_CAPACITY") == "local_replay_invalid_capacity"
    values["context_utilization_ratio"] = analysis._MISSING
    assert _support(values, 1_000, "OK") == "missing_predecision_feature"


def test_history_gate_rejects_overlap_unknown_and_out_of_order() -> None:
    candidate = _call("candidate", 10.0)
    complete = _call("prior", 1.0, 8000)
    assert _history_completion_reason(candidate, [complete, candidate]) is None
    assert _history_completion_reason(candidate, [replace(complete, timing={"total_ms": 9001}), candidate]) == "earlier_call_in_flight_at_dispatch"
    assert _history_completion_reason(candidate, [replace(complete, timing={}), candidate]) == "prior_completion_time_unknown"
    assert _history_completion_reason(candidate, [replace(complete, response=None), candidate]) == "prior_response_incomplete"
    assert _history_completion_reason(candidate, [replace(complete, response={"content": []}), candidate]) == "prior_response_incomplete"
    assert _history_completion_reason(candidate, [_call("later", 11.0), candidate]) == "earlier_call_in_flight_at_dispatch"
    assert _history_completion_reason(replace(candidate, timestamp_unix_s=None), [complete, candidate]) == "candidate_dispatch_time_unknown"
    assert _history_completion_reason(candidate, [complete]) == "candidate_absent_from_source_chronology"


def test_counts_only_when_preregistered_reportability_fails() -> None:
    events = [_event(repo, turn, "HARM" if turn == 0 else "SAFE")
              for repo in range(4) for turn in range(8)]
    assert not reportability(events)["passed"]
    result = compare(events)
    assert result["status"] == "counts_only_reportability_gate_failed"
    assert "folds" not in result


def test_four_models_use_repo_heldout_folds_and_inner_gate() -> None:
    events = [_event(repo, turn, "HARM" if turn in (0, 3) else "SAFE")
              for repo in range(6) for turn in range(10)]
    assert reportability(events)["passed"]
    assert _inner_threshold(events[:50])["status"] == "cloud_all"  # below the frozen 60-call minimum
    result = compare(events)
    assert result["status"] == "diagnostic_only_no_activation"
    assert len(result["folds"]) == 6
    assert all(set(fold["models"]) == {"numeric_logistic", "numeric_ridge",
                                       "word_structural", "shallow_forest"}
               for fold in result["folds"].values())
    assert all(fold["n_test"] == 10 for fold in result["folds"].values())
    assert all(fold["threshold_screen"]["status"] == "cloud_all" for fold in result["folds"].values())
    assert result["pooled_out_of_fold"]["numeric_logistic"]["n"] == 60
    json.dumps(result, default=quality._json_numeric)


def test_numpy_metric_scalars_serialize_without_coercing_other_objects() -> None:
    encoded = json.dumps({"n": np.int64(2), "score": np.float64(0.25), "flag": np.bool_(True)},
                         default=quality._json_numeric)
    assert json.loads(encoded) == {"n": 2, "score": 0.25, "flag": True}
    with pytest.raises(TypeError):
        json.dumps({"array": np.asarray([1, 2])}, default=quality._json_numeric)


def test_inner_threshold_uses_safe_score_cutpoint(monkeypatch) -> None:
    events = [_event(repo, turn, "HARM" if turn in (0, 3) else "SAFE")
              for repo in range(6) for turn in range(10)]
    monkeypatch.setattr(quality, "_fit_predict",
                        lambda kind, train, test: np.asarray([0.9 if e.label == "HARM" else 0.1 for e in test]))
    selected = _inner_threshold(events)
    assert selected["status"] == "illustrative_threshold"
    assert selected["threshold"] == 0.1
    assert selected["inner_safe_routed_local"] == 48


def test_cli_cannot_read_real_cohort_without_terminal_approval(monkeypatch, tmp_path) -> None:
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["comparison", "--output", str(output)])
    with pytest.raises(SystemExit):
        main()
    monkeypatch.setattr(sys, "argv", ["comparison", "--run-real", "--output", str(output)])
    with pytest.raises(SystemExit):
        main()
    assert not output.exists()


def test_snapshot_binds_all_namespaces_sources_and_absent_inputs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(quality, "REPO", tmp_path)
    root = tmp_path / "results"
    source = tmp_path / "source.jsonl"
    source.write_text("original\n")
    for version in (6, 7, 8):
        folder = root / f"postpilot_v{version}"
        folder.mkdir(parents=True)
        (folder / quality.REPLAY).write_text(json.dumps({"call": {"source_trace_path": str(source)}}) + "\n")
        (folder / quality.JUDGE).write_text("")
        (folder / "pilot_selection_manifest.json").write_text(json.dumps({
            "trajectories": {"synthetic": {"task_group": f"swebench-repo{version}-abcdef0123"}}
        }))
    before = quality.snapshot_input_hashes(root)
    assert str(source.resolve()) in before
    assert len(before) == 21  # v8 binds five additional terminal/quarantine files
    assert any(path.endswith("quarantine_manifest.json") and value is None for path, value in before.items())
    source.write_text("mutated\n")
    assert quality.snapshot_input_hashes(root) != before


def test_terminal_partial_task_counts_each_selected_opportunity(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(quality, "REPO", tmp_path)
    root = tmp_path / "results"
    folder = root / "postpilot_v8"
    folder.mkdir(parents=True)
    group = "swebench-sympy-abcdef0123"
    source = tmp_path / "source.jsonl"
    source.write_text("synthetic source\n")
    selected = [f"call-{i}" for i in range(8)]
    calls = [replace(_call(cid, float(i + 1), 100), task_group=group,
                     source_trace_path=str(source), trajectory_id=f"synthetic:{group}:cloud:seed1",
                     request={"messages": [], "max_tokens": 1000}, turn_index=i)
             for i, cid in enumerate(selected[:4])]
    replay = [{"call": {"call_id": c.call_id, "task_group": group,
                         "source_trace_path": str(source), "trajectory_id": c.trajectory_id,
                         "request": c.request, "source_campaign": "synthetic"},
               "local_outcome": {"status": "OK" if i < 3 else "TRANSPORT_ERROR"},
               "cloud_outcome": {"status": "OK"}}
              for i, c in enumerate(calls)]
    (folder / quality.REPLAY).write_text("\n".join(map(json.dumps, replay)) + "\n")
    judge = [{"call_id": c.call_id, "pass_label": label, "verdict": "EQUIVALENT",
              "order": ["local", "cloud"] if label == "primary" else ["cloud", "local"],
              "truncation_policy_version": "v4-system-messages-tools-2026-09-16",
              "primary_order_policy_version": "v5-sha256-call-id-2026-09-17"}
             for c in calls[:3] for label in ("primary", "reversed")]
    (folder / quality.JUDGE).write_text("\n".join(map(json.dumps, judge)) + "\n")
    import hashlib
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    (folder / "pilot_selection_manifest.json").write_text(json.dumps({"trajectories": {
        "synthetic": {"task_group": group, "call_ids": selected,
                      "eligible_call_ids": selected, "source_trace_sha256": source_hash,
                      "inclusion_probability_by_call_id": {cid: 0.5 for cid in selected}}
    }}))
    grade = tmp_path / "eval-suite/swebench/results/agentic-router-postpilot-v8/verdicts" / f"{group}__cloud__seed1.json"
    grade.parent.mkdir(parents=True)
    grade.write_text('{"passed": true}')
    monkeypatch.setattr(quality, "_full_calls", lambda raw, path: calls)
    monkeypatch.setattr(quality.features, "regenerate_replay_calls", lambda rows: {c.call_id: c for c in calls})
    monkeypatch.setattr(quality.analysis, "call_features", lambda call: {
        **{name: 0.0 for name in analysis.FEATURE_NAMES},
        "local_prompt_tokens": 10000.0, "context_utilization_ratio": 0.1})
    events, meta = quality._load_namespace(8, root)
    flow = meta["task_flow"][group]
    assert len(events) == 3
    assert flow["selected"] == 8 and flow["included"] == 3
    assert flow["excluded"] == {"unattempted_or_missing_replay": 4,
                                  "local_replay_transport_error": 1}
