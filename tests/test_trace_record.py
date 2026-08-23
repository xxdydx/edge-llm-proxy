import unittest

from edgeproxy.trace.record import build_token_accounting, reassemble


class TraceRecordTests(unittest.TestCase):
    def test_token_accounting_normalises_detailed_usage(self):
        accounting = build_token_accounting(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 300,
            }
        )

        self.assertEqual(
            accounting,
            {
                "input_tokens": 1200,
                "output_tokens": 50,
                "tokens_processed": 1250,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 300,
                "uncached_input_tokens": 100,
                "cache_details_available": True,
            },
        )

    def test_token_accounting_keeps_missing_cache_breakdown_unknown(self):
        accounting = build_token_accounting(
            {"input_tokens": 1200, "output_tokens": 50}
        )

        self.assertEqual(accounting["input_tokens"], 1200)
        self.assertEqual(accounting["output_tokens"], 50)
        self.assertEqual(accounting["tokens_processed"], 1250)
        self.assertIsNone(accounting["cache_read_input_tokens"])
        self.assertIsNone(accounting["cache_creation_input_tokens"])
        self.assertIsNone(accounting["uncached_input_tokens"])
        self.assertFalse(accounting["cache_details_available"])

    def test_token_accounting_uses_nulls_when_usage_is_unavailable(self):
        accounting = build_token_accounting({})

        self.assertIsNone(accounting["input_tokens"])
        self.assertIsNone(accounting["output_tokens"])
        self.assertIsNone(accounting["tokens_processed"])

    def test_reassemble_preserves_final_cache_usage(self):
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_local",
                    "content": [],
                    "usage": {"input_tokens": 32, "output_tokens": 0},
                },
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "output_tokens": 8,
                    "cache_read_input_tokens": 96,
                    "cache_creation_input_tokens": 16,
                },
            },
        ]

        _, usage = reassemble(events)

        self.assertEqual(
            usage,
            {
                "input_tokens": 32,
                "output_tokens": 8,
                "cache_read_input_tokens": 96,
                "cache_creation_input_tokens": 16,
            },
        )


if __name__ == "__main__":
    unittest.main()
