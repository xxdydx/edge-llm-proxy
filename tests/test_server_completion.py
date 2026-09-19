import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx

from edgeproxy.config import Config
from edgeproxy.server import make_app
from edgeproxy.trace.record import request_identity


def config(
    trace_dir: Path,
    *,
    policy: str = "static",
    cohort_parent_placement: bool = False,
) -> Config:
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
        local_cache_tracking="observe",
        cohort_parent_placement=cohort_parent_placement,
    )


def request_body(*, tools=None, continuation: bool = False) -> dict:
    content = (
        [{"type": "tool_result", "tool_use_id": "t1", "content": "done"}]
        if continuation
        else "work"
    )
    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": content}],
    }
    if tools is not None:
        body["tools"] = tools
    return body


def records(trace_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in trace_dir.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


class ServerCompletionPredictionTests(unittest.IsolatedAsyncioTestCase):
    async def _post(self, app, body: dict, *, prompt_tokens: int = 1024) -> None:
        def upstream(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/count_cached_tokens"):
                return httpx.Response(
                    200,
                    json={"input_tokens": prompt_tokens, "cached_tokens": 0},
                )
            return httpx.Response(
                200,
                json={
                    "type": "message",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 8},
                },
            )

        async with app.router.lifespan_context(app):
            for backend in ("local", "cloud"):
                await app.state.clients[backend].aclose()
                app.state.clients[backend] = httpx.AsyncClient(
                    base_url=(
                        "http://local.test"
                        if backend == "local"
                        else "https://cloud.test"
                    ),
                    transport=httpx.MockTransport(upstream),
                )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://edge.test",
            ) as client:
                response = await client.post("/v1/messages", json=body)
                self.assertEqual(response.status_code, 200)

    async def test_local_prediction_is_attached_after_class_history(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            # tool_suite_hash is None whenever a request carries no tools (see
            # edgeproxy/trace/record.py's request_identity), matching the same
            # convention the Rung 1 reliability breaker relies on. This test
            # needs a real, non-empty tool suite so the pre-seeded class_key
            # actually matches what the live request computes.
            body = request_body(
                tools=[
                    {
                        "name": "Read",
                        "description": "read a file",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ]
            )
            app = make_app(config(trace_dir))
            class_key = request_identity(body)["tool_suite_hash"]
            app.state.completion_estimator.record(
                class_key, output_tokens=100, tpot_ms=10.0, concurrency=1
            )
            estimator_record = Mock(wraps=app.state.completion_estimator.record)
            app.state.completion_estimator.record = estimator_record

            await self._post(app, body)

            estimator_record.assert_called_once()
            self.assertEqual(estimator_record.call_args.kwargs["concurrency"], 1)

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "local")
            prediction = record["local_completion_prediction"]
            self.assertIsNotNone(prediction)
            self.assertEqual(prediction["expected_output_tokens"], 100.0)
            self.assertEqual(prediction["expected_tpot_ms"], 10.0)
            self.assertEqual(prediction["queue_wait_ms"], 0.0)

    async def test_local_cold_class_is_explicitly_null_not_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))

            await self._post(app, request_body())

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "local")
            self.assertIn("local_completion_prediction", record)
            self.assertIsNone(record["local_completion_prediction"])

    async def test_feasibility_cloud_call_has_no_local_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))
            server_tool = {"type": "web_search_20250305", "name": "web_search"}

            await self._post(app, request_body(tools=[server_tool]))

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "cloud")
            self.assertEqual(record["reason"], "server-side-tool")
            self.assertNotIn("local_completion_prediction", record)

    async def test_headroom_cloud_call_has_no_local_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir, policy="branch-drift"))

            await self._post(
                app,
                request_body(continuation=True),
                prompt_tokens=36_000,
            )

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "cloud")
            self.assertEqual(record["reason"], "branch-headroom-pressure")
            self.assertNotIn("local_completion_prediction", record)

    async def test_cohort_parent_override_has_no_local_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir, cohort_parent_placement=True))
            agent_tool = {
                "name": "Agent",
                "description": "delegate",
                "input_schema": {"type": "object"},
            }

            body = request_body(tools=[agent_tool])
            body["messages"][-1]["content"] = (
                "<system-reminder><total_tokens>14999733 tokens left"
                "</total_tokens></system-reminder>"
            )
            body["messages"][-1]["role"] = "system"
            await self._post(app, body)

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "cloud")
            self.assertEqual(record["reason"], "cohort-parent-placement")
            self.assertNotIn("local_completion_prediction", record)


if __name__ == "__main__":
    unittest.main()
