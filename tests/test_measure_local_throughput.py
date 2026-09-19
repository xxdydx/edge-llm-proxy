import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import httpx

from scripts.measure_local_throughput import (
    calculate_tpot_ms,
    exact_prompt_tokens,
    load_prompts,
    measure_request,
    message_body,
    percentile,
)


class FakeTokenizer:
    all_special_ids = [0]

    def __len__(self):
        return 1000

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [10 + (ord(character) % 100) for character in text]

    def decode(self, prompt):
        return "decoded:" + ",".join(str(token_id) for token_id in prompt)


class LocalThroughputHelpersTests(unittest.TestCase):
    def test_load_prompts_ignores_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("# note\n\nfirst\tWrite a guide.\nsecond\tAnalyze this.\n")
            self.assertEqual(
                load_prompts(path),
                [("first", "Write a guide."), ("second", "Analyze this.")],
            )

    def test_load_prompts_requires_tab_separator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("missing separator\n")
            with self.assertRaises(ValueError):
                load_prompts(path)

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([10.0, 20.0], 0.9), 19.0)

    def test_tpot_uses_intervals_after_first_token(self):
        self.assertEqual(calculate_tpot_ms(1270.0, 128), 10.0)
        self.assertIsNone(calculate_tpot_ms(0.0, 128))
        self.assertIsNone(calculate_tpot_ms(10.0, 1))

    def test_exact_prompt_length_and_unique_first_cache_block(self):
        tokenizer = FakeTokenizer()
        first = exact_prompt_tokens(tokenizer, "repeat me", 128, 123, 0)
        second = exact_prompt_tokens(tokenizer, "repeat me", 128, 123, 1)
        self.assertEqual(len(first), 128)
        self.assertEqual(len(second), 128)
        self.assertNotEqual(first[0], second[0])


class MessagesTransportTests(unittest.TestCase):
    def test_message_body_wraps_decoded_tokens_as_user_content(self):
        self.assertEqual(
            message_body("local", [10, 20, 30], 128, True, FakeTokenizer()),
            {
                "model": "local",
                "max_tokens": 128,
                "temperature": 0,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "decoded:10,20,30"}
                ],
            },
        )

    def test_measure_request_times_each_anthropic_output_delta(self):
        requests = []

        def upstream(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            events = [
                {
                    "type": "message_start",
                    "message": {"type": "message", "content": []},
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "a"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "b"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "c"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "max_tokens"},
                    "usage": {"output_tokens": 3},
                },
                {"type": "content_block_stop", "index": 0},
                {"type": "message_stop"},
            ]
            content = b"".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                for event in events
            )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=content
            )

        async def exercise():
            async with httpx.AsyncClient(
                base_url="http://local.test",
                transport=httpx.MockTransport(upstream),
            ) as client:
                return await measure_request(
                    client, "local", [10, 20], 3, 0, "prompt", FakeTokenizer()
                )

        timestamps = iter([10.0, 10.1, 10.2, 10.35, 10.5, 10.6])
        with patch(
            "scripts.measure_local_throughput.time",
            SimpleNamespace(perf_counter=lambda: next(timestamps)),
        ):
            result = asyncio.run(exercise())

        self.assertEqual(result.output_tokens, 3)
        self.assertAlmostEqual(result.ttft_ms, 200.0)
        self.assertAlmostEqual(result.last_token_ms, 500.0)
        self.assertAlmostEqual(result.decode_ms, 300.0)
        self.assertAlmostEqual(result.tpot_ms, 150.0)
        self.assertAlmostEqual(result.decode_tokens_per_s, 1000 / 150.0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/v1/messages")
        self.assertEqual(
            json.loads(requests[0].content),
            message_body("local", [10, 20], 3, True, FakeTokenizer()),
        )


if __name__ == "__main__":
    unittest.main()
