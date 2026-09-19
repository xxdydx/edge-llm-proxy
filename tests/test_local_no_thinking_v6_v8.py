"""Synthetic mechanism checks; never make network/model calls."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import run_local_no_thinking_v6_v8 as probe


class CountParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.item = {
            "call_id": "synthetic",
            "historical_input_tokens": 18000,
            "original": {"model": "local"},
            "modified": {"model": "local", "chat_template_kwargs": {"enable_thinking": False}},
        }

    def test_exact_amended_offsets(self) -> None:
        with patch.object(probe, "_count", side_effect=[18002, 18004]):
            self.assertEqual(probe._capacity_parity(self.item), (18002, 18004))

    def test_original_offset_mismatch_fails_closed(self) -> None:
        with patch.object(probe, "_count", side_effect=[18003, 18005]):
            with self.assertRaisesRegex(RuntimeError, "count_offset_not_exactly"):
                probe._capacity_parity(self.item)

    def test_modified_offset_mismatch_fails_closed(self) -> None:
        with patch.object(probe, "_count", side_effect=[18002, 18003]):
            with self.assertRaisesRegex(RuntimeError, "count_offset_not_exactly"):
                probe._capacity_parity(self.item)

    def test_exact_capacity_failure(self) -> None:
        item = dict(self.item, historical_input_tokens=58000)
        with patch.object(probe, "_count", side_effect=[58002, 58004]):
            with self.assertRaisesRegex(RuntimeError, "capacity_does_not_fit"):
                probe._capacity_parity(item)


class StreamParseTests(unittest.TestCase):
    def test_schema_valid_tool_only_reports_name_and_boolean(self) -> None:
        raw = b"".join([
            b'data: {"type":"message_start","message":{"model":"local","usage":{"input_tokens":9}}}\n\n',
            b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","name":"Read","input":{}}}\n\n',
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\\"path\\\":\\\"private.py\\\"}"}}\n\n',
            b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}\n\n',
            b'data: {"type":"message_stop"}\n\n',
        ])
        payload = {"tools": [{"name": "Read", "input_schema": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}]}
        row = probe._parse(raw, payload)
        self.assertTrue(row["stream_complete"])
        self.assertEqual(row["tools"], [{"name": "Read", "offered": True, "schema_valid": True}])
        self.assertNotIn("private.py", str(row))
        self.assertEqual(row["stop_reason"], "tool_use")


if __name__ == "__main__":
    unittest.main()
