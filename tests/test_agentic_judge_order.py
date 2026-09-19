"""Offline regression guards for per-call judge A/B order and resumability."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from experiments.agentic_router import judge


def _example(call_id: str) -> dict:
    return {
        "call": {"call_id": call_id, "request": {
            "system": "system", "messages": [{"role": "user", "content": "task"}], "tools": [],
        }},
        "local_outcome": {"response": {"content": [{"type": "text", "text": "local action"}]}},
        "cloud_outcome": {"response": {"content": [{"type": "text", "text": "cloud action"}]}},
    }


def _fake_judge(bundle, a_text, b_text, backend):
    return "EQUIVALENT", "same", '{"verdict":"EQUIVALENT"}', 100, 25.0, False, None


def test_primary_assignment_is_stable_and_not_singleton_reset():
    ids = [f"call{i}" for i in range(12)]
    assignments = [judge.primary_order_for_call(cid) for cid in ids]
    assert len(set(assignments)) == 2
    assert assignments == [judge.primary_order_for_call(cid) for cid in reversed(ids)][::-1]
    assert judge.primary_order_for_call("call0", seed=1) != judge.primary_order_for_call("call0", seed=2)


def test_singleton_runs_match_batch_and_reverse_exactly():
    examples = [_example(f"call{i}") for i in range(6)]
    with TemporaryDirectory() as tmp, patch.object(judge.cr_executor, "load_env"), patch.object(judge, "_judge_one", _fake_judge):
        batch = Path(tmp) / "batch.jsonl"
        singleton = Path(tmp) / "singleton.jsonl"
        judge.judge_examples_with_consistency(examples, batch)
        for ex in examples:
            judge.judge_examples_with_consistency([ex], singleton)
        batch_rows = [json.loads(line) for line in batch.open()]
        single_rows = [json.loads(line) for line in singleton.open()]
        assert [(r["call_id"], r["pass_label"], r["order"]) for r in batch_rows] == [
            (r["call_id"], r["pass_label"], r["order"]) for r in single_rows
        ]
        for i in range(0, len(batch_rows), 2):
            primary, reversed_ = batch_rows[i:i + 2]
            assert primary["order"] == list(judge.primary_order_for_call(primary["call_id"]))
            assert reversed_["order"] == primary["order"][::-1]
            assert primary["primary_order_policy_version"] == judge.PRIMARY_ORDER_POLICY_VERSION
            assert primary["primary_order_seed"] == 20260916


def test_resume_rejects_legacy_or_wrong_seed_rows():
    with TemporaryDirectory() as tmp, patch.object(judge.cr_executor, "load_env"), patch.object(judge, "_judge_one", _fake_judge):
        path = Path(tmp) / "judgments.jsonl"
        path.write_text(json.dumps({"call_id": "call0", "pass_label": "primary", "order": ["cloud", "local"]}) + "\n")
        with pytest.raises(ValueError, match="different order policy"):
            judge.judge_examples_with_consistency([_example("call0")], path)
        path.unlink()
        judge.judge_examples_with_consistency([_example("call0")], path, seed=1)
        with pytest.raises(ValueError, match="different order policy"):
            judge.judge_examples_with_consistency([_example("call0")], path, seed=2)
