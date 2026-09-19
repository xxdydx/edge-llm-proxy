import unittest
from pathlib import Path

from scripts.analyze_call_risk import (
    Example,
    _canonical_group,
    _grouped_cross_validation,
    analyze,
    _model_profile,
    _profile_validation,
    _select_threshold_for_recall,
)


class RecallThresholdTests(unittest.TestCase):
    def test_selects_highest_precision_threshold_meeting_recall_floor(self):
        examples = [
            Example((0.0,), 0, "negative-a"),
            Example((0.0,), 0, "negative-b"),
            Example((0.0,), 1, "positive-a"),
            Example((0.0,), 1, "positive-b"),
        ]
        scores = [0.1, 0.4, 0.9, 0.6]

        self.assertEqual(_select_threshold_for_recall(examples, scores, 1.0), 0.6)

    def test_rejects_invalid_recall_target(self):
        with self.assertRaises(ValueError):
            _select_threshold_for_recall([], [], 0.0)


class GroupedValidationTests(unittest.TestCase):
    def test_entire_groups_stay_in_one_validation_fold(self):
        examples = [
            Example((0.9, 1.0, 0.0), 1, "positive-a"),
            Example((0.8, 1.0, 0.0), 0, "positive-a"),
            Example((0.7, 1.0, 0.0), 1, "positive-b"),
            Example((0.6, 1.0, 0.0), 0, "positive-b"),
            Example((0.2, 1.0, 0.0), 0, "negative-a"),
            Example((0.1, 1.0, 0.0), 0, "negative-b"),
        ]

        result = _grouped_cross_validation(examples, (0,), 0.5)
        validation_groups = [
            group for fold in result["folds"] for group in fold["groups"]
        ]

        self.assertCountEqual(
            validation_groups,
            ["positive-a", "positive-b", "negative-a", "negative-b"],
        )
        self.assertEqual(len(validation_groups), len(set(validation_groups)))

    def test_macro_average_is_not_dominated_by_one_huge_fold(self):
        # one small, perfectly-classified fold ("tiny") and one huge fold
        # ("giant") with a mediocre model: pooled precision is call-count
        # weighted (dominated by "giant"), macro precision weighs both folds
        # equally regardless of call count.
        examples = [
            Example((0.9, 1.0, 0.0), 1, "tiny"),
            Example((0.1, 1.0, 0.0), 0, "tiny"),
        ]
        for i in range(50):
            examples.append(Example((0.9, 1.0, 0.0), 1 if i < 5 else 0, "giant"))
        result = _grouped_cross_validation(examples, (0,), 0.5)
        macro = result["f1_validation_macro"]
        pooled = result["f1_validation"]
        self.assertEqual(macro["n_folds"], 2)
        # macro precision is the unweighted mean of the two folds' precision,
        # not equal to the pooled (call-count-weighted) precision in general
        self.assertNotAlmostEqual(macro["precision_mean"], pooled["precision"])


class ModelProfileTests(unittest.TestCase):
    def test_identifies_profiles_from_experiment_or_versioned_vllm_signature(self):
        self.assertEqual(
            _model_profile({"experiment_id": "risk-corpus-7b-0908"}),
            "qwen25-7b",
        )
        # "7b" is a substring of "27b" -- a 27B experiment id must not fall
        # through to the 7B branch (regression: risk-corpus-27b-clean-... was
        # being labelled qwen25-7b, silently dropping fresh 27B truncations
        # from the 27B slice).
        self.assertEqual(
            _model_profile({"experiment_id": "risk-corpus-27b-clean-20260910T150307Z"}),
            "qwen38-27b",
        )
        self.assertEqual(
            _model_profile(
                {
                    "local_resources": {
                        "vllm": {"block_size": 1568, "cache_dtype": "fp8"}
                    }
                }
            ),
            "qwen38-27b",
        )

    def test_profile_validation_never_pools_models(self):
        examples = [
            Example((0.0,), 1, "a", "qwen25-7b"),
            Example((0.0,), 0, "b", "qwen25-7b"),
            Example((0.0,), 0, "c", "qwen38-27b"),
        ]

        result = _profile_validation(examples, [0.9, 0.1, 0.8], 0.5)

        self.assertEqual(result["qwen25-7b"]["tp"], 1)
        self.assertEqual(result["qwen25-7b"]["fp"], 0)
        self.assertEqual(result["qwen38-27b"]["fp"], 1)
        self.assertIsNone(result["qwen38-27b"]["auc"])


class CanonicalGroupTests(unittest.TestCase):
    def test_fanout_run_directories_are_distinct_groups(self):
        # regression: a single "fanout" catch-all used to lump every fan-out
        # campaign episode (different dates/conditions) into one CV fold and
        # produced 950 false positives under leave-one-group-out.
        a = _canonical_group(Path("traces/fanout/cloud-20260829T054236Z/x.jsonl"))
        b = _canonical_group(Path("traces/fanout-policy-pair/routing-20260905T155122Z-43174/x.jsonl"))
        self.assertNotEqual(a, b)
        self.assertEqual(a, "fanout-cloud-20260829T054236Z")
        self.assertEqual(b, "fanout-routing-20260905T155122Z-43174")

    def test_loose_fanout_files_group_by_filename_stem(self):
        # regression: 14 loose files sitting directly under traces/fanout/ and
        # traces/fanout-policy-pair/ (no run-subdirectory) used to collapse to
        # one shared "fanout" bucket -- 449 false positives in one CV fold.
        # Each filename encodes its own episode; group by stem instead.
        a = _canonical_group(Path("traces/fanout/cloud_20260829T054236Z.jsonl"))
        b = _canonical_group(Path("traces/fanout-policy-pair/routing_20260906T111152Z-40351.jsonl"))
        self.assertNotEqual(a, b)
        self.assertEqual(a, "fanout-cloud_20260829T054236Z")
        self.assertEqual(b, "fanout-routing_20260906T111152Z-40351")

    def test_swebench_instance_still_takes_priority(self):
        p = Path("traces/fanout-policy-pair/routing/swebench-ansible-39bd8b99-seed1/x.jsonl")
        self.assertEqual(_canonical_group(p), "swebench-ansible-39bd8b99")


class AnalysisOutputTests(unittest.TestCase):
    def test_conjunctive_baselines_do_not_overwrite_logistic_validation(self):
        # A compact synthetic trace corpus is awkward because the analyzer's
        # frozen split requires several real task groups. Guard the regression
        # at the source level: the logistic holdout vector must never be the
        # assignment target inside the conjunctive loop again.
        source = Path(analyze.__code__.co_filename).read_text()
        loop = source.split("conjunctive_models: dict[str, Any] = {}", 1)[1]
        loop = loop.split("return {", 1)[0]

        self.assertNotIn("\n        holdout_scores = [", loop)


if __name__ == "__main__":
    unittest.main()
