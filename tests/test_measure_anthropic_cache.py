import unittest

from scripts.measure_anthropic_cache import (
    estimate_planned_tokens,
    planned_request_count,
    summarize,
)


class AnthropicCacheRunnerTests(unittest.TestCase):
    def test_budget_includes_correctness_requests(self):
        self.assertEqual(planned_request_count([2048], 1, "correctness"), 16)
        self.assertGreater(estimate_planned_tokens([2048], 1, "correctness"), 4096)

    def test_summary_preserves_invalid_provider_results(self):
        rows = [
            {
                "target_prefix_tokens": 2048,
                "condition": "warm",
                "valid": False,
                "ttft_ms": 100.0,
                "tpot_ms": 10.0,
                "cache_read_input_tokens": None,
            }
        ]
        result = summarize(rows)
        self.assertEqual(result[0]["valid_samples"], 0)
        self.assertEqual(result[0]["total_samples"], 1)
        self.assertIsNone(result[0]["median_ttft_ms"])


if __name__ == "__main__":
    unittest.main()
