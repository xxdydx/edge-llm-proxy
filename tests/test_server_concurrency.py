import asyncio
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import httpx

from edgeproxy.config import Config
from edgeproxy.server import make_app


def config(trace_dir: Path, *, policy: str = "local-only") -> Config:
    return Config(
        host="127.0.0.1",
        port=0,
        upstream="https://cloud.test",
        trace_dir=trace_dir,
        vllm_url="http://local.test",
        policy=policy,
        shaping="none",
        link_preset="none",
        cloud_delay_ms=0.0,
        cloud_jitter_ms=0.0,
        cloud_bandwidth_mbps=0.0,
        local_model_name="local",
        resource_sample_interval_s=60.0,
        gpu_index=0,
        kv_bytes_per_token=None,
        cloud_cache_tracking="off",
        local_concurrency_limit=8,
    )


def request_body(*, stream: bool) -> dict:
    return {
        "model": "claude-sonnet-5",
        "max_tokens": 8,
        "stream": stream,
        "messages": [{"role": "user", "content": "work"}],
    }


def records(trace_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in trace_dir.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


def sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


class ConcurrentStream(httpx.AsyncByteStream):
    def __init__(
        self,
        arrived: list[int],
        all_arrived: asyncio.Event,
        release: asyncio.Event,
        target: int,
    ) -> None:
        self.arrived = arrived
        self.all_arrived = all_arrived
        self.release = release
        self.target = target

    async def __aiter__(self):
        self.arrived.append(1)
        if len(self.arrived) == self.target:
            self.all_arrived.set()
        await self.release.wait()
        for event in (
            {
                "type": "message_start",
                "message": {
                    "id": "msg_local",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
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
                "delta": {"type": "text_delta", "text": "done"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ):
            yield sse(event)


class InterruptedStream(httpx.AsyncByteStream):
    def __init__(self, partial_sent: asyncio.Event) -> None:
        self.partial_sent = partial_sent
        self.release = asyncio.Event()

    async def __aiter__(self):
        yield sse({
            "type": "message_start",
            "message": {"id": "partial", "type": "message", "role": "assistant",
                        "content": [], "usage": {"input_tokens": 10, "output_tokens": 0}},
        })
        yield sse({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}})
        yield sse({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "unfinished"}})
        self.partial_sent.set()
        await self.release.wait()


class ServerConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def _clients(self, app, handler):
        for backend in ("cloud", "local"):
            await app.state.clients[backend].aclose()
            app.state.clients[backend] = httpx.AsyncClient(
                base_url=(
                    "https://cloud.test"
                    if backend == "cloud"
                    else "http://local.test"
                ),
                transport=httpx.MockTransport(handler),
            )
        app.state.resource_sampler.snapshot = lambda: {
            "sampled_at": time.time(),
            "age_ms": 0.0,
            "vllm": {"requests_running": 4, "requests_waiting": 2},
        }
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://edge.test",
        )

    async def test_concurrent_streaming_local_requests_reach_peak_and_return_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))
            arrived: list[int] = []
            all_arrived = asyncio.Event()
            release = asyncio.Event()

            async def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=ConcurrentStream(
                        arrived, all_arrived, release, target=3
                    ),
                )

            async with app.router.lifespan_context(app):
                async with await self._clients(app, handler) as client:
                    tasks = [
                        asyncio.create_task(
                            client.post("/v1/messages", json=request_body(stream=True))
                        )
                        for _ in range(3)
                    ]
                    await asyncio.wait_for(all_arrived.wait(), timeout=2)
                    self.assertEqual(
                        app.state.local_backend_state.requests_in_flight, 3
                    )
                    release.set()
                    responses = await asyncio.gather(*tasks)

            self.assertTrue(all(response.status_code == 200 for response in responses))
            self.assertEqual(app.state.local_backend_state.requests_in_flight, 0)
            snapshots = [record["local_resources"] for record in records(trace_dir)]
            self.assertEqual(
                sorted(row["proxy_requests_in_flight"] for row in snapshots),
                [0, 1, 2],
            )
            self.assertTrue(
                all(
                    row["vllm"]
                    == {"requests_running": 4, "requests_waiting": 2}
                    for row in snapshots
                )
            )

    async def test_non_streaming_local_request_decrements_after_body(self):
        with tempfile.TemporaryDirectory() as directory:
            app = make_app(config(Path(directory)))
            entered = asyncio.Event()
            release = asyncio.Event()

            async def handler(request: httpx.Request) -> httpx.Response:
                entered.set()
                await release.wait()
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "content": [{"type": "text", "text": "done"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                )

            async with app.router.lifespan_context(app):
                async with await self._clients(app, handler) as client:
                    task = asyncio.create_task(
                        client.post("/v1/messages", json=request_body(stream=False))
                    )
                    await asyncio.wait_for(entered.wait(), timeout=2)
                    self.assertEqual(
                        app.state.local_backend_state.requests_in_flight, 1
                    )
                    release.set()
                    response = await task

            self.assertEqual(response.status_code, 200)
            self.assertEqual(app.state.local_backend_state.requests_in_flight, 0)

    async def test_local_transport_error_decrements(self):
        with tempfile.TemporaryDirectory() as directory:
            app = make_app(config(Path(directory)))

            async def handler(request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("local unavailable", request=request)

            async with app.router.lifespan_context(app):
                async with await self._clients(app, handler) as client:
                    response = await client.post(
                        "/v1/messages", json=request_body(stream=False)
                    )

            self.assertEqual(response.status_code, 502)
            self.assertEqual(app.state.local_backend_state.requests_in_flight, 0)

    async def test_streaming_local_request_decrements_on_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            app = make_app(config(Path(directory)))
            entered = asyncio.Event()
            release = asyncio.Event()
            arrived: list[int] = []

            async def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=ConcurrentStream(arrived, entered, release, target=1),
                )

            async with app.router.lifespan_context(app):
                async with await self._clients(app, handler) as client:
                    task = asyncio.create_task(
                        client.post("/v1/messages", json=request_body(stream=True))
                    )
                    await asyncio.wait_for(entered.wait(), timeout=2)
                    self.assertEqual(
                        app.state.local_backend_state.requests_in_flight, 1
                    )
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

            self.assertEqual(app.state.local_backend_state.requests_in_flight, 0)

    async def test_cancelled_partial_stream_is_not_a_completed_local_response(self):
        artifact = (Path(__file__).resolve().parents[1] /
                    "experiments/agentic_router/results/quality_model_baseline205_v5_shadow_logreg.json")
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(replace(config(trace_dir), agentic_shadow_artifact=artifact))
            partial_sent = asyncio.Event()
            stream = InterruptedStream(partial_sent)
            reliability_record = Mock(wraps=app.state.reliability.record)
            completion_record = Mock(wraps=app.state.completion_estimator.record)
            app.state.reliability.record = reliability_record
            app.state.completion_estimator.record = completion_record

            async def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

            body = request_body(stream=True)
            body["tools"] = [{"name": "Read", "input_schema": {"type": "object", "properties": {}}}]
            async with app.router.lifespan_context(app):
                async with await self._clients(app, handler) as client:
                    task = asyncio.create_task(client.post(
                        "/v1/messages", json=body,
                        headers={"x-claude-code-session-id": "cancelled-lane"},
                    ))
                    await asyncio.wait_for(partial_sent.wait(), timeout=2)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

            self.assertEqual(app.state.local_backend_state.requests_in_flight, 0)
            reliability_record.assert_not_called()
            completion_record.assert_not_called()
            self.assertEqual(len(records(trace_dir)), 1)
            record = records(trace_dir)[0]
            self.assertEqual(record["error"], "StreamAborted(CancelledError)")
            self.assertFalse(record["stream_complete"])
            self.assertIsInstance(record["response"], dict)  # partial but not valid
            lane = next(iter(app.state.agentic_history.lanes.values()))
            self.assertEqual(list(lane.calls), [])
            self.assertEqual(lane.pending, set())
            self.assertTrue(lane.truncated)

    async def test_early_eof_does_not_turn_partial_message_into_success(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))
            reliability_record = Mock(wraps=app.state.reliability.record)
            completion_record = Mock(wraps=app.state.completion_estimator.record)
            app.state.reliability.record = reliability_record
            app.state.completion_estimator.record = completion_record

            async def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      content=sse({"type": "message_start", "message": {
                                          "id": "partial", "type": "message", "role": "assistant",
                                          "content": [], "usage": {"input_tokens": 10, "output_tokens": 0},
                                      }}))

            body = request_body(stream=True)
            body["tools"] = [{"name": "Read", "input_schema": {"type": "object", "properties": {}}}]
            async with app.router.lifespan_context(app):
                async with await self._clients(app, handler) as client:
                    response = await client.post("/v1/messages", json=body)

            self.assertEqual(response.status_code, 200)  # headers already committed
            record = records(trace_dir)[0]
            self.assertEqual(record["error"], "StreamIncomplete(message_stop missing)")
            self.assertFalse(record["stream_complete"])
            reliability_record.assert_not_called()
            completion_record.assert_not_called()

    async def test_cloud_request_never_increments_local_counter_and_is_traced(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir, policy="cloud-only"))
            entered = asyncio.Event()
            release = asyncio.Event()

            async def handler(request: httpx.Request) -> httpx.Response:
                entered.set()
                await release.wait()
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "content": [{"type": "text", "text": "done"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                )

            async with app.router.lifespan_context(app):
                async with await self._clients(app, handler) as client:
                    task = asyncio.create_task(
                        client.post("/v1/messages", json=request_body(stream=False))
                    )
                    await asyncio.wait_for(entered.wait(), timeout=2)
                    self.assertEqual(
                        app.state.local_backend_state.requests_in_flight, 0
                    )
                    release.set()
                    response = await task

            self.assertEqual(response.status_code, 200)
            self.assertEqual(app.state.local_backend_state.requests_in_flight, 0)
            snapshot = records(trace_dir)[0]["local_resources"]
            self.assertEqual(snapshot["proxy_requests_in_flight"], 0)
            self.assertEqual(snapshot["concurrency_limit"], 8)
            self.assertEqual(snapshot["vllm"]["requests_running"], 4)
            self.assertEqual(snapshot["vllm"]["requests_waiting"], 2)
