import json
import unittest

import httpx

from scripts.measure_prefix_ttft import detect_match_unit, measure_ttft, message_body


class FakeTokenizer:
    def decode(self, prompt):
        return "decoded:" + ",".join(str(token_id) for token_id in prompt)


class MatchUnitDetectionTests(unittest.TestCase):
    def test_prefers_numeric_prefix_match_unit(self):
        self.assertEqual(
            detect_match_unit({"prefix_match_unit": "32", "block_size": "16"}),
            32,
        )

    def test_falls_back_when_prefix_match_unit_is_none_sentinel(self):
        self.assertEqual(
            detect_match_unit({"prefix_match_unit": "None", "block_size": "16"}),
            16,
        )

    def test_falls_back_when_prefix_match_unit_is_invalid(self):
        self.assertEqual(
            detect_match_unit({"prefix_match_unit": "unknown", "block_size": "16"}),
            16,
        )

    def test_returns_none_without_positive_numeric_value(self):
        self.assertIsNone(
            detect_match_unit({"prefix_match_unit": "None", "block_size": "0"})
        )


class MessagesTransportTests(unittest.TestCase):
    def test_message_body_wraps_decoded_tokens_as_user_content(self):
        self.assertEqual(
            message_body("local", [10, 20, 30], True, FakeTokenizer()),
            {
                "model": "local",
                "max_tokens": 1,
                "temperature": 0,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "decoded:10,20,30"}
                ],
            },
        )

    def test_measure_ttft_reads_anthropic_content_block_delta(self):
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
                    "delta": {"type": "text_delta", "text": "x"},
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

        with httpx.Client(
            base_url="http://local.test", transport=httpx.MockTransport(upstream)
        ) as client:
            ttft_ms, headers_ms = measure_ttft(
                client, "local", [10, 20], FakeTokenizer()
            )

        self.assertGreaterEqual(ttft_ms, headers_ms)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/v1/messages")
        self.assertEqual(
            json.loads(requests[0].content),
            message_body("local", [10, 20], True, FakeTokenizer()),
        )


if __name__ == "__main__":
    unittest.main()
