import json
import tempfile
import unittest
from pathlib import Path

from edgeproxy.cloud_cache import CloudCacheTracker, prefix_chain
from edgeproxy.cost import build_cost_savings, pricing_for_model
from edgeproxy.trace.record import TraceWriter


class CostSavingsTests(unittest.TestCase):
    def test_current_model_prices(self):
        self.assertEqual(pricing_for_model("claude-sonnet-5").input, 2.0)
        self.assertEqual(pricing_for_model("claude-opus-5").output, 25.0)
        self.assertEqual(
            pricing_for_model("claude-haiku-4-5-20251001").cache_read, 0.1
        )
        self.assertIsNone(pricing_for_model("deepseek-v4-flash"))

    def test_cloud_usage_prices_each_cache_category(self):
        result = build_cost_savings(
            placement="cloud",
            requested_model="claude-sonnet-5",
            usage={
                "input_tokens": 100,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 300,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 200,
                    "ephemeral_1h_input_tokens": 100,
                },
                "output_tokens": 50,
            },
        )
        # 100*2 + 800*.2 + 200*2.5 + 100*4 + 50*10 = $0.00176
        self.assertEqual(result["cloud_cost_usd"], 0.00176)
        self.assertEqual(result["request_saved_usd"], 0.0)
        self.assertEqual(result["confidence"], "actual-token-accounting")

    def test_local_warm_prediction_estimates_avoided_discounted_bill(self):
        request = {
            "model": "claude-sonnet-5",
            "system": [
                {
                    "type": "text",
                    "text": "stable " * 1000,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "reply"}],
        }
        chain = prefix_chain(request, "scope")
        tracker = CloudCacheTracker()
        cold = tracker.probe(chain, 1)
        tracker.observe_cloud_usage(
            chain,
            cold,
            request_started_at=1,
            response_started_at=2,
            status=200,
            usage={
                "input_tokens": 10,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 1800,
            },
        )
        warm = tracker.probe(chain, 3)
        result = build_cost_savings(
            placement="local",
            requested_model="claude-sonnet-5",
            usage={"input_tokens": 200, "cache_read_input_tokens": 1800, "output_tokens": 50},
            chain=chain,
            prediction=warm,
        )
        tokens = result["counterfactual_cloud_tokens"]
        self.assertEqual(
            tokens["cache_read_input_tokens"], warm.estimated_read_tokens
        )
        self.assertEqual(
            tokens["uncached_input_tokens"]
            + tokens["cache_read_input_tokens"]
            + tokens["cache_creation_5m_input_tokens"]
            + tokens["cache_creation_1h_input_tokens"],
            2000,
        )
        self.assertEqual(result["request_saved_usd"], result["cloud_cost_usd"])

    def test_local_request_without_breakpoint_is_priced_as_uncached(self):
        request = {
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "reply"}],
        }
        chain = prefix_chain(request, "scope")
        prediction = CloudCacheTracker().probe(chain, 1)
        result = build_cost_savings(
            placement="local",
            requested_model="claude-sonnet-5",
            usage={"input_tokens": 1000, "output_tokens": 10},
            chain=chain,
            prediction=prediction,
        )
        self.assertTrue(result["available"])
        self.assertEqual(
            result["counterfactual_cloud_tokens"]["uncached_input_tokens"], 1000
        )
        self.assertEqual(result["confidence"], "no-cloud-cache-breakpoints")

    def test_writer_accumulates_and_recovers_running_savings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = TraceWriter(root)
            for saved in (0.001, 0.0, 0.002):
                writer.write({"cost_savings": {"request_saved_usd": saved}})
            writer = TraceWriter(root)
            writer.write({"cost_savings": {"request_saved_usd": 0.003}})
            records = [
                json.loads(line)
                for path in root.glob("*.jsonl")
                for line in path.read_text().splitlines()
            ]
            self.assertEqual(
                [row["cost_savings"]["running_saved_usd"] for row in records],
                [0.001, 0.001, 0.003, 0.006],
            )


if __name__ == "__main__":
    unittest.main()
