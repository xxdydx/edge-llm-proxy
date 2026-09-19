import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from edgeproxy.config import Config
from edgeproxy.server import make_app


def config(
    trace_dir: Path,
    *,
    policy: str = "cloud-only",
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
        cohort_tracking="observe",
        cohort_parent_placement=cohort_parent_placement,
    )


def records(trace_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in trace_dir.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


AGENT_TOOL = {
    "name": "Agent",
    "description": "Delegate work",
    "input_schema": {"type": "object"},
}

TOKEN_REMINDER = (
    "<system-reminder>\n"
    "<total_tokens>14999733 tokens left</total_tokens>\n"
    "</system-reminder>"
)


def sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


class PausingAgentStream(httpx.AsyncByteStream):
    def __init__(self, block_processed: asyncio.Event, release: asyncio.Event):
        self.block_processed = block_processed
        self.release = release

    async def __aiter__(self):
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_parent",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "agent-early",
                    "name": "Agent",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"prompt": "inspect early"}),
                },
            },
            {"type": "content_block_stop", "index": 0},
        ]
        for event in events:
            yield sse(event)

        # The proxy requests this next item only after it has processed and
        # forwarded content_block_stop, while the parent stream remains open.
        self.block_processed.set()
        await self.release.wait()
        yield sse(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 5},
            }
        )
        yield sse({"type": "message_stop"})


class ServerCohortIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _post_with_mock_backends(
        self,
        app,
        upstream,
        requests: list[tuple[dict, dict]],
    ) -> None:
        async with app.router.lifespan_context(app):
            for backend in ("cloud", "local"):
                await app.state.clients[backend].aclose()
                app.state.clients[backend] = httpx.AsyncClient(
                    base_url=(
                        "https://cloud.test"
                        if backend == "cloud"
                        else "http://local.test"
                    ),
                    transport=httpx.MockTransport(upstream),
                )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://edge.test",
            ) as client:
                for body, headers in requests:
                    response = await client.post(
                        "/v1/messages", json=body, headers=headers
                    )
                    self.assertEqual(response.status_code, 200)

    async def test_opt_in_off_preserves_policy_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir, policy="local-only"))

            def upstream(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "done"}],
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                )

            await self._post_with_mock_backends(
                app,
                upstream,
                [
                    (
                        {
                            "model": "claude-sonnet-5",
                            "max_tokens": 8,
                            "tools": [AGENT_TOOL],
                            "messages": [{"role": "user", "content": "work"}],
                        },
                        {},
                    )
                ],
            )

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "local")
            self.assertEqual(record["reason"], "policy")
            self.assertNotIn("cohort_parent_placement", record)

    async def test_opt_in_forces_signaled_root_and_records_true_and_false_positives(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(
                config(
                    trace_dir,
                    policy="local-only",
                    cohort_parent_placement=True,
                )
            )

            def upstream(request: httpx.Request) -> httpx.Response:
                prompt = json.loads(request.content)["messages"][0]["content"]
                content = (
                    [
                        {
                            "type": "tool_use",
                            "id": "agent-1",
                            "name": "Agent",
                            "input": {"prompt": "delegated child prompt"},
                        }
                    ]
                    if prompt == "fan out"
                    else [{"type": "text", "text": "done without delegation"}]
                )
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "tool_use" if prompt == "fan out" else "end_turn",
                        "content": content,
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                )

            requests = [
                (
                    {
                        "model": "claude-sonnet-5",
                        "max_tokens": 8,
                        "tools": [AGENT_TOOL],
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": "considering"},
                                {"role": "system", "content": TOKEN_REMINDER},
                            ],
                    },
                    {"x-claude-code-session-id": "session-1"},
                )
                for prompt in ("fan out", "ordinary turn")
            ]
            await self._post_with_mock_backends(app, upstream, requests)

            traced = {row["request"]["messages"][0]["content"]: row for row in records(trace_dir)}
            true_positive = traced["fan out"]
            false_positive = traced["ordinary turn"]
            for record in (true_positive, false_positive):
                self.assertEqual(record["placement"], "cloud")
                self.assertEqual(record["reason"], "cohort-parent-placement")
                self.assertTrue(record["features"]["has_agent_tool"])
            self.assertEqual(
                true_positive["cohort_parent_placement"],
                {
                    "candidate": True,
                    "signal": "root-token-budget-reminder-v1",
                    "outcome": "true_positive",
                    "did_fan_out": True,
                    "agent_delegation_count": 1,
                },
            )
            self.assertEqual(
                false_positive["cohort_parent_placement"],
                {
                    "candidate": True,
                    "signal": "root-token-budget-reminder-v1",
                    "outcome": "false_positive",
                    "did_fan_out": False,
                    "agent_delegation_count": 0,
                },
            )

    async def test_opt_in_does_not_force_ordinary_agent_capable_root(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(
                config(trace_dir, policy="local-only", cohort_parent_placement=True)
            )

            def upstream(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "done"}],
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                )

            await self._post_with_mock_backends(
                app,
                upstream,
                [
                    (
                        {
                            "model": "claude-sonnet-5",
                            "max_tokens": 8,
                            "tools": [AGENT_TOOL],
                            "messages": [{"role": "user", "content": "ordinary turn"}],
                        },
                        {"x-claude-code-session-id": "session-1"},
                    )
                ],
            )

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "local")
            self.assertEqual(record["reason"], "policy")
            self.assertNotIn("cohort_parent_placement", record)

    async def test_agent_header_prevents_nested_parent_override(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(
                config(trace_dir, policy="local-only", cohort_parent_placement=True)
            )

            def upstream(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "done"}],
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                )

            await self._post_with_mock_backends(
                app,
                upstream,
                [
                    (
                        {
                            "model": "claude-sonnet-5",
                            "max_tokens": 8,
                            "tools": [AGENT_TOOL],
                            "messages": [{"role": "system", "content": TOKEN_REMINDER}],
                        },
                        {
                            "x-claude-code-session-id": "session-1",
                            "x-claude-code-agent-id": "child-1",
                        },
                    )
                ],
            )

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "local")
            self.assertNotIn("cohort_parent_placement", record)

    async def test_matched_child_with_agent_tool_is_not_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(
                config(
                    trace_dir,
                    policy="local-only",
                    cohort_parent_placement=True,
                )
            )

            def upstream(request: httpx.Request) -> httpx.Response:
                prompt = json.loads(request.content)["messages"][0]["content"]
                content = (
                    [
                        {
                            "type": "tool_use",
                            "id": "agent-parent",
                            "name": "Agent",
                            "input": {"prompt": "delegated child prompt"},
                        }
                    ]
                    if prompt == "seed parent"
                    else [{"type": "text", "text": "child done"}]
                )
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "tool_use" if prompt == "seed parent" else "end_turn",
                        "content": content,
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                )

            headers = {"x-claude-code-session-id": "session-1"}
            await self._post_with_mock_backends(
                app,
                upstream,
                [
                    (
                        {
                            "model": "claude-sonnet-5",
                            "max_tokens": 8,
                            "tools": [AGENT_TOOL],
                            "messages": [{"role": "user", "content": "seed parent"}],
                        },
                        headers,
                    ),
                    (
                        {
                            "model": "claude-sonnet-5",
                            "max_tokens": 8,
                            "tools": [AGENT_TOOL],
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "delegated child prompt with context",
                                }
                            ],
                        },
                        headers,
                    ),
                ],
            )

            child = next(
                row
                for row in records(trace_dir)
                if row["request"]["messages"][0]["content"].startswith("delegated")
            )
            self.assertEqual(child["cohort_detection"]["role"], "child")
            self.assertTrue(child["features"]["has_agent_tool"])
            self.assertEqual(child["placement"], "local")
            self.assertEqual(child["reason"], "policy")
            self.assertNotIn("cohort_parent_placement", child)

    async def test_child_matches_after_agent_block_stops_before_parent_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))
            block_processed = asyncio.Event()
            release_parent = asyncio.Event()

            def upstream(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content)
                if body.get("stream"):
                    return httpx.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        stream=PausingAgentStream(block_processed, release_parent),
                    )
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "child done"}],
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                )

            headers = {"x-claude-code-session-id": "session-1"}
            parent_body = {
                "model": "claude-sonnet-5",
                "max_tokens": 32,
                "stream": True,
                "messages": [{"role": "user", "content": "delegate"}],
            }
            child_body = {
                "model": "claude-sonnet-5",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "inspect early now"}],
            }

            async with app.router.lifespan_context(app):
                await app.state.clients["cloud"].aclose()
                app.state.clients["cloud"] = httpx.AsyncClient(
                    base_url="https://cloud.test",
                    transport=httpx.MockTransport(upstream),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://edge.test",
                ) as client:
                    parent_task = asyncio.create_task(
                        client.post(
                            "/v1/messages", json=parent_body, headers=headers
                        )
                    )
                    await asyncio.wait_for(block_processed.wait(), timeout=1.0)
                    self.assertFalse(parent_task.done())

                    child_response = await client.post(
                        "/v1/messages", json=child_body, headers=headers
                    )
                    self.assertEqual(child_response.status_code, 200)

                    release_parent.set()
                    parent_response = await parent_task
                    self.assertEqual(parent_response.status_code, 200)

            traced = records(trace_dir)
            parent = next(row for row in traced if row["stream"])
            child = next(row for row in traced if not row["stream"])
            self.assertEqual(child["cohort_detection"]["role"], "child")
            self.assertEqual(
                child["cohort_detection"]["parent_tool_use_id"], "agent-early"
            )
            self.assertEqual(
                child["cohort_detection"]["parent_call_id"], parent["id"]
            )


if __name__ == "__main__":
    unittest.main()
