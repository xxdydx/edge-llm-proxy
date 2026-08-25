import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from edgeproxy.config import Config
from edgeproxy.server import make_app


class SplitSSEStream(httpx.AsyncByteStream):
    def __init__(self, events: list[dict]):
        self.parts = [
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            for event in events
        ]

    async def __aiter__(self):
        for part in self.parts:
            await asyncio.sleep(0.001)
            # Split a data line to exercise the incremental decoder as it runs
            # in the real proxy, not only in the isolated parser unit test.
            middle = max(1, len(part) // 2)
            yield part[:middle]
            yield part[middle:]


def response_events(*, cache_read: int, cache_creation: int) -> list[dict]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": cache_creation,
                        "ephemeral_1h_input_tokens": 0,
                    },
                },
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world"},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


class ServerCloudCacheIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_creation_makes_next_prediction_warm(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            cfg = Config(
                host="127.0.0.1",
                port=0,
                upstream="https://cloud.test",
                trace_dir=trace_dir,
                vllm_url="http://local.test",
                policy="cloud-only",
                shaping="none",
                link_preset="none",
                cloud_delay_ms=0.0,
                cloud_jitter_ms=0.0,
                cloud_bandwidth_mbps=0.0,
                local_model_name="local",
                resource_sample_interval_s=60.0,
                gpu_index=0,
                kv_bytes_per_token=None,
                cloud_cache_tracking="observe",
            )
            app = make_app(cfg)
            calls = 0

            def upstream(request: httpx.Request) -> httpx.Response:
                nonlocal calls
                calls += 1
                values = (0, 1800) if calls == 1 else (1800, 0)
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=SplitSSEStream(
                        response_events(cache_read=values[0], cache_creation=values[1])
                    ),
                )

            async with app.router.lifespan_context(app):
                await app.state.clients["cloud"].aclose()
                app.state.clients["cloud"] = httpx.AsyncClient(
                    base_url="https://cloud.test",
                    transport=httpx.MockTransport(upstream),
                )
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://edge.test"
                ) as client:
                    body = {
                        "model": "claude-sonnet-5",
                        "max_tokens": 8,
                        "stream": True,
                        "system": [
                            {
                                "type": "text",
                                "text": "stable " * 1200,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                        "messages": [{"role": "user", "content": "reply"}],
                    }
                    headers = {"authorization": "Bearer test-only"}
                    first = await client.post("/v1/messages", json=body, headers=headers)
                    second = await client.post("/v1/messages", json=body, headers=headers)
                    self.assertEqual(first.status_code, 200)
                    self.assertEqual(second.status_code, 200)

            records = []
            for path in trace_dir.glob("*.jsonl"):
                records.extend(json.loads(line) for line in path.read_text().splitlines())
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["cloud_cache"]["prediction"]["state"], "unknown")
            self.assertEqual(
                records[0]["cloud_cache"]["actual"]["cache_creation_input_tokens"], 1800
            )
            self.assertEqual(records[1]["cloud_cache"]["prediction"]["state"], "warm")
            self.assertEqual(
                records[1]["cloud_cache"]["actual"]["cache_read_input_tokens"], 1800
            )
            self.assertIsNotNone(records[1]["timing"]["ttft_ms"])
            self.assertIsNotNone(records[1]["timing"]["tpot_ms"])


if __name__ == "__main__":
    unittest.main()
