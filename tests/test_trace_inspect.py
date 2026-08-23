import contextlib
import io
import unittest

from edgeproxy.trace.inspect import cache_stats, summarise, usage_input_breakdown


class TraceInspectTests(unittest.TestCase):
    def test_detailed_usage_total_includes_uncached_read_and_created(self):
        usage = usage_input_breakdown(
            {
                "input_tokens": 32,
                "cache_read_input_tokens": 96,
                "cache_creation_input_tokens": 16,
            }
        )

        self.assertTrue(usage["detailed"])
        self.assertEqual(usage["total"], 144)

    def test_historical_usage_keeps_input_as_total_but_not_as_cache_miss(self):
        usage = usage_input_breakdown({"input_tokens": 144})

        self.assertFalse(usage["detailed"])
        self.assertEqual(usage["total"], 144)
        self.assertEqual(
            cache_stats([{"placement": "local", "usage": {"input_tokens": 144}}]),
            {},
        )

    def test_cache_statistics_are_split_by_placement(self):
        records = [
            {
                "placement": "local",
                "usage": {"input_tokens": 16, "cache_read_input_tokens": 48},
            },
            {
                "placement": "local",
                "usage": {"input_tokens": 64, "cache_read_input_tokens": 0},
            },
            {
                "placement": "cloud",
                "usage": {
                    "input_tokens": 8,
                    "cache_read_input_tokens": 80,
                    "cache_creation_input_tokens": 8,
                },
            },
        ]

        stats = cache_stats(records)

        self.assertEqual(
            stats["local"],
            {"requests": 2, "request_hits": 1, "cached": 48, "total": 128},
        )
        self.assertEqual(
            stats["cloud"],
            {"requests": 1, "request_hits": 1, "cached": 80, "total": 96},
        )

    def test_historical_summary_says_cache_detail_is_unavailable(self):
        records = [
            {
                "path": "/v1/messages",
                "placement": "local",
                "request": {"model": "local"},
                "usage": {"input_tokens": 144, "output_tokens": 8},
            }
        ]
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            summarise(records)

        self.assertIn("cache detail     unavailable", output.getvalue())
        self.assertNotIn("cache_read", output.getvalue())


if __name__ == "__main__":
    unittest.main()
