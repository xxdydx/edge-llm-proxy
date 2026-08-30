import json
import tempfile
import unittest
from pathlib import Path

from edgeproxy.trace.record import (
    TraceWriter,
    build_structured_call,
    build_token_accounting,
    consumed_tool_result_ids,
    reassemble,
    request_identity,
    validate_tool_use_blocks,
)


class TraceRecordTests(unittest.TestCase):
    def test_request_identity_is_stable_and_turn_specific(self):
        first = {
            "model": "claude-sonnet-5",
            "system": "system",
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "one"}],
        }
        second = {**first, "messages": [{"role": "user", "content": "two"}]}

        first_identity = request_identity(first)
        second_identity = request_identity(second)

        self.assertEqual(first_identity["lineage_id"], second_identity["lineage_id"])
        self.assertNotEqual(first_identity["turn_id"], second_identity["turn_id"])
        self.assertEqual(first_identity["tool_names"], ["Read"])
        self.assertIsNotNone(first_identity["tool_suite_hash"])

        cache_variant = {
            **first,
            "tools": [
                {
                    **first["tools"][0],
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        }
        self.assertEqual(
            first_identity["tool_suite_hash"],
            request_identity(cache_variant)["tool_suite_hash"],
        )

    def test_tool_use_validation_records_each_block(self):
        request = {
            "tools": [
                {
                    "name": "Read",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
        response = {
            "content": [
                {"type": "tool_use", "id": "ok", "name": "Read", "input": {"path": "a"}},
                {"type": "tool_use", "id": "bad", "name": "Read", "input": {}},
            ]
        }

        results = validate_tool_use_blocks(request, response)

        self.assertTrue(results[0]["schema_valid"])
        self.assertFalse(results[1]["schema_valid"])
        self.assertIn("required property", results[1]["validation_error"])

    def test_structured_call_uses_symmetric_provider_neutral_fields(self):
        request = {
            "model": "claude-sonnet-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tool-parent"}
                    ],
                }
            ],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        }
        record = {
            "id": "call-1",
            "ts": 1.5,
            "experiment_id": "fanout-1",
            "episode_id": "fanout-1-routing",
            "placement": "cloud",
            "backend": "https://cloud.test",
            "status": 200,
            "requested_max_tokens": 64_000,
            "effective_max_tokens": 20_000,
            "output_reserve_tokens": 0,
            "headers": {
                "x-claude-code-session-id": "session-1",
                "x-claude-code-agent-id": "agent-1",
                "x-claude-code-parent-tool-use-id": "agent-tool-parent",
            },
            "response": {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-child",
                        "name": "Read",
                        "input": {},
                    }
                ],
            },
            "token_accounting": {
                "input_tokens": 120,
                "output_tokens": 8,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 16,
                "uncached_input_tokens": 4,
                "cache_details_available": True,
            },
            "timing": {"ttft_ms": 25.0, "output_tokens_per_s": 80.0},
            "cloud_cache": {"prediction": {}, "actual": {}},
            "cohort_detection": {
                "cohort_id": "cohort-1",
                "detection_confidence": "high",
            },
        }

        call = build_structured_call(record, request)

        self.assertEqual(call["schema_version"], "edgeproxy.call.v1")
        self.assertEqual(call["experiment_id"], "fanout-1")
        self.assertEqual(call["episode_id"], "fanout-1-routing")
        self.assertEqual(call["backend"], "cloud")
        self.assertEqual(call["tokens"]["prompt_tokens_exact"], 120)
        self.assertEqual(call["tokens"]["cache_read_tokens"], 100)
        self.assertEqual(call["tokens"]["cache_write_tokens"], 16)
        self.assertEqual(call["timing"]["decode_tokens_per_s"], 80.0)
        self.assertEqual(call["cache_probe"], record["cloud_cache"])
        self.assertEqual(call["cohort"], record["cohort_detection"])
        self.assertEqual(call["output_limit"]["effective_max_tokens"], 20_000)
        self.assertEqual(call["causality"]["agent_id"], "agent-1")
        self.assertEqual(
            call["causality"]["agent_parent_tool_use_id"], "agent-tool-parent"
        )
        self.assertEqual(
            call["causality"]["consumed_tool_result_ids"], ["tool-parent"]
        )
        self.assertEqual(
            call["causality"]["produced_tool_use_ids"], ["tool-child"]
        )

    def test_consumed_results_come_from_latest_causal_message_only(self):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "old"}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "new-a"},
                        {"type": "tool_result", "tool_use_id": "new-b"},
                    ],
                },
                {"role": "system", "content": [{"type": "text", "text": "meta"}]},
            ]
        }

        self.assertEqual(consumed_tool_result_ids(request), ["new-a", "new-b"])

    def test_writer_links_consumed_tool_result_to_parent_and_root_call(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = TraceWriter(Path(directory))
            parent = {
                "call": {
                    "call_id": "call-parent",
                    "causality": {
                        "consumed_tool_result_ids": [],
                        "produced_tool_use_ids": ["tool-parent"],
                        "parent_call_ids": [],
                        "root_call_id": "call-parent",
                    },
                }
            }
            child = {
                "call": {
                    "call_id": "call-child",
                    "causality": {
                        "consumed_tool_result_ids": ["tool-parent"],
                        "produced_tool_use_ids": [],
                        "parent_call_ids": [],
                        "root_call_id": "call-child",
                    },
                }
            }
            writer.write(parent)
            writer = TraceWriter(Path(directory))
            writer.write(child)
            rows = [
                json.loads(line)
                for path in Path(directory).glob("*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            rows[1]["call"]["causality"]["parent_call_ids"], ["call-parent"]
        )
        self.assertEqual(
            rows[1]["call"]["causality"]["root_call_id"], "call-parent"
        )

    def test_structured_call_does_not_call_gateway_zero_exact(self):
        call = build_structured_call(
            {
                "id": "call-1",
                "placement": "cloud",
                "headers": {},
                "response": {"model": "deepseek-v4-flash", "content": []},
                "token_accounting": {
                    "input_tokens": 0,
                    "output_tokens": 1,
                    "cache_read_input_tokens": None,
                    "cache_creation_input_tokens": None,
                    "uncached_input_tokens": None,
                    "cache_details_available": False,
                },
            },
            {"messages": [{"role": "user", "content": "nonempty"}]},
        )

        self.assertEqual(call["tokens"]["input_tokens"], 0)
        self.assertIsNone(call["tokens"]["prompt_tokens_exact"])
        self.assertEqual(call["tokens"]["usage_integrity"], "partial")

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
