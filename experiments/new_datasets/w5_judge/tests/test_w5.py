from __future__ import annotations

import json
import unittest

from experiments.new_datasets.w1_storage.schemas import Verdict
from experiments.new_datasets.w5_judge.controlled_profile_plan import build_profile_plan
from experiments.new_datasets.w5_judge.judge_client import (
    AggregateLabel,
    OrientationResult,
    ParseStatus,
    anonymization_violations,
    aggregate_orientations,
    build_judge_request,
    invalid_judgment,
    orientation_mappings,
    parse_judge_response,
    primary_mapping,
    verdict_backend,
)
from experiments.new_datasets.w5_judge.serving_schema_helpers import (
    build_serving_call,
    decode_speed_measurement,
    normalize_usage,
    shared_prefix_cost,
)


def parsed(verdict: Verdict):
    return parse_judge_response(json.dumps({
        "verdict": verdict.value,
        "reason_codes": ["SUPPORTED_PROGRESS"],
        "evidence": [{"message_or_candidate_ref": "candidate_A", "explanation": "fixture evidence"}],
        "insufficient_context": False,
    }))


class JudgeTests(unittest.TestCase):
    def test_reversed_mapping_recovers_backend_identity(self):
        pair = "synthetic-pair"
        primary, reverse = orientation_mappings(pair)
        self.assertEqual(reverse, {"A": primary["B"], "B": primary["A"]})
        primary_verdict = Verdict.A_BETTER if primary["A"] == "cloud" else Verdict.B_BETTER
        reverse_verdict = Verdict.A_BETTER if reverse["A"] == "cloud" else Verdict.B_BETTER
        primary_result = OrientationResult("primary", parsed(primary_verdict), primary["A"], primary["B"])
        reverse_result = OrientationResult("reverse", parsed(reverse_verdict), reverse["A"], reverse["B"])
        self.assertEqual(aggregate_orientations(primary_result, reverse_result).label, AggregateLabel.CLOUD_PREFERRED)
        # The same identity recovery works when cloud is primary A.
        other = next(p for p in ("a", "b", "c", "d") if primary_mapping(p)["A"] == "cloud")
        p, r = orientation_mappings(other)
        self.assertEqual(verdict_backend(Verdict.A_BETTER, p), "cloud")
        self.assertEqual(verdict_backend(Verdict.B_BETTER, r), "cloud")

    def test_aggregate_states_are_distinct(self):
        p = orientation_mappings("pair")
        def row(verdict, mapping):
            return OrientationResult("x", parsed(verdict), mapping["A"], mapping["B"])
        self.assertEqual(aggregate_orientations(row(Verdict.EQUIVALENT, p[0]), row(Verdict.EQUIVALENT, p[1])).label, AggregateLabel.EQUIVALENT)
        self.assertEqual(aggregate_orientations(row(Verdict.BOTH_INADEQUATE, p[0]), row(Verdict.BOTH_INADEQUATE, p[1])).label, AggregateLabel.BOTH_INADEQUATE)
        self.assertEqual(aggregate_orientations(row(Verdict.A_BETTER, p[0]), row(Verdict.A_BETTER, p[1])).label, AggregateLabel.UNCERTAIN)
        pending = OrientationResult("x", invalid_judgment("truncated", pending=True), p[0]["A"], p[0]["B"])
        out = aggregate_orientations(pending, row(Verdict.EQUIVALENT, p[1]))
        self.assertEqual(out.status, ParseStatus.PENDING)
        self.assertIsNone(out.label)

    def test_pipe_placeholder_is_rejected(self):
        with self.assertRaises(Exception):
            parse_judge_response('{"verdict":"A_BETTER | B_BETTER | EQUIVALENT | BOTH_INADEQUATE | UNCERTAIN"}')

    def test_anonymization_check(self):
        clean = build_judge_request("fix the bug", [], {"text": "Read the file", "tool_actions": []}, {"text": "Inspect the test", "tool_actions": []})
        self.assertEqual(anonymization_violations(clean), [])
        dirty = build_judge_request("fix the bug", [], {"text": "claude-sonnet-5", "tool_actions": []}, {"text": "deepseek-v4-flash", "tool_actions": []})
        self.assertIn("provider_or_model_identity", anonymization_violations(dirty))


class ServingTests(unittest.TestCase):
    def test_buffered_stream_has_no_decode_speed(self):
        result = decode_speed_measurement([{"text": "all output", "token_count": 20, "timestamp_ms": 1000}])
        self.assertIsNone(result["tokens_per_second"])
        self.assertIsNone(result["true_decode_tpot_ms"])

    def test_shared_prefix_is_not_double_charged(self):
        result = shared_prefix_cost(1.0, 1.5, 1.4)
        self.assertEqual(result["branch_a_marginal_cost_usd"], 0.5)
        self.assertAlmostEqual(result["branch_b_marginal_cost_usd"], 0.4)
        self.assertLessEqual(result["branch_a_marginal_cost_usd"], result["branch_a_total_cost_usd"])
        self.assertLessEqual(result["combined_allocated_cost_usd"], result["branch_a_total_cost_usd"] + result["branch_b_total_cost_usd"])

    def test_usage_absence_is_null_not_zero_and_record_validates(self):
        normalized, reasons = normalize_usage({"prompt_tokens": 12})
        self.assertEqual(normalized["input_tokens_exact"], 12)
        self.assertIsNone(normalized["output_tokens_exact"])
        self.assertIn("output_tokens_exact", reasons)
        record = build_serving_call(
            {"queue_depth": None, "cache_state": "unknown"},
            {"status": "completed", "raw_usage": {"prompt_tokens": 12}},
            identity={"schema_version": 1, "protocol_version": "synthetic-v1", "campaign_id": "syn-campaign", "task_id": "syn-task", "trajectory_id": "syn-traj", "source": "synthetic", "cohort": "fixture", "split": "synthetic", "provenance": {"fixture": True}, "invocation_id": "syn-inv", "logical_call_id": "syn-logical", "backend_fingerprint_id": "syn-edge", "logical_request_hash": "a" * 64, "rendered_request_hash": "b" * 64},
        )
        self.assertIsNone(record.normalised_usage["output_tokens_exact"])


class PlanTests(unittest.TestCase):
    def test_profile_plan_is_40_and_seeded(self):
        refs = [f"fixture:req{i}" for i in range(10)]
        one = build_profile_plan(refs, seed=7)
        two = build_profile_plan(refs, seed=7)
        three = build_profile_plan(refs, seed=8)
        self.assertEqual(one.foreground_count, 40)
        self.assertEqual(one.invocations, two.invocations)
        self.assertNotEqual(one.invocations, three.invocations)
        self.assertEqual({x.condition.value for x in one.invocations}, {"cold_low", "cold_high", "warm_low", "warm_high"})
        self.assertTrue(all(x.output_cap == 256 for x in one.invocations))


if __name__ == "__main__":
    unittest.main()
