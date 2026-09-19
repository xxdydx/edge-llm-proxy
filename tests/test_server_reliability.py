import json
import random
import tempfile
import unittest
from pathlib import Path

import httpx

from edgeproxy.config import Config
from edgeproxy.server import make_app


def config(trace_dir: Path, *, policy: str = "static") -> Config:
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
    )


def records(trace_dir: Path) -> list[dict]:
    out = []
    for path in trace_dir.glob("*.jsonl"):
        out.extend(json.loads(line) for line in path.read_text().splitlines())
    out.sort(key=lambda r: r.get("ts", 0))
    return out


TOOLS = [
    {
        "name": "my_tool",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    }
]


def _request(i: int) -> dict:
    return {
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "tools": TOOLS,
        "messages": [{"role": "user", "content": f"call {i}"}],
    }


class ServerReliabilityIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_circuit_opens_after_repeated_schema_failures_then_routes_cloud(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))
            # The open circuit keeps a 10% recovery-probe stream (see
            # reliability.py PROBE_FRACTION), rolled from an unseeded RNG in
            # production. Seed it here so the 6th call — the assertion under
            # test — is deterministically blocked rather than a ~1-in-10
            # probe that stays local and flakes this test. random.Random(0)'s
            # first draws are all well above 0.10.
            app.state.reliability._rng = random.Random(0)

            def invalid_local(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("/count_cached_tokens"):
                    return httpx.Response(
                        200, json={"input_tokens": 100, "cached_tokens": 0}
                    )
                # Missing the required "x" property — always schema-invalid.
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "tool_use",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "my_tool",
                                "input": {},
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                )

            def cloud(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "cloud"}],
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                )

            async with app.router.lifespan_context(app):
                await app.state.clients["local"].aclose()
                await app.state.clients["cloud"].aclose()
                app.state.clients["local"] = httpx.AsyncClient(
                    base_url="http://local.test",
                    transport=httpx.MockTransport(invalid_local),
                )
                app.state.clients["cloud"] = httpx.AsyncClient(
                    base_url="https://cloud.test",
                    transport=httpx.MockTransport(cloud),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://edge.test",
                ) as client:
                    # Default breaker: min_samples=5, threshold=0.25. 5 straight
                    # local failures (rate 1.0) must trip it.
                    for i in range(5):
                        response = await client.post(
                            "/v1/messages", json=_request(i)
                        )
                        self.assertEqual(response.status_code, 200)

                    rows = records(trace_dir)
                    self.assertEqual(len(rows), 5)
                    for row in rows:
                        self.assertEqual(row["placement"], "local")
                        self.assertFalse(
                            row["call"]["tool_use_blocks"][0]["schema_valid"]
                        )

                    # The 6th call in the same tool-suite class must now be
                    # blocked by the open circuit and routed cloud instead.
                    response = await client.post("/v1/messages", json=_request(5))
                    self.assertEqual(response.status_code, 200)

            rows = records(trace_dir)
            self.assertEqual(len(rows), 6)
            sixth = rows[-1]
            self.assertEqual(sixth["placement"], "cloud")
            self.assertEqual(sixth["reason"], "reliability-circuit-open")
            self.assertNotIn("local_completion_prediction", sixth)
            self.assertEqual(sixth["reliability"]["note"], "reliability-circuit-open")
            self.assertAlmostEqual(sixth["reliability"]["failure_rate"], 1.0)

    async def test_healthy_class_never_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))

            def valid_local(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("/count_cached_tokens"):
                    return httpx.Response(
                        200, json={"input_tokens": 100, "cached_tokens": 0}
                    )
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "tool_use",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "my_tool",
                                "input": {"x": "ok"},
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                )

            async with app.router.lifespan_context(app):
                await app.state.clients["local"].aclose()
                app.state.clients["local"] = httpx.AsyncClient(
                    base_url="http://local.test",
                    transport=httpx.MockTransport(valid_local),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://edge.test",
                ) as client:
                    for i in range(8):
                        response = await client.post(
                            "/v1/messages", json=_request(i)
                        )
                        self.assertEqual(response.status_code, 200)

            rows = records(trace_dir)
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r["placement"] == "local" for r in rows))
            self.assertTrue(
                all(r["call"]["tool_use_blocks"][0]["schema_valid"] for r in rows)
            )

    async def test_transport_failures_are_not_recorded_as_false_successes(self):
        # A transport error leaves tool_use_blocks empty, same as a clean
        # text-only response -- without the response_ok guard this would
        # be recorded as a vacuous success, diluting real schema failures
        # enough to keep an unreliable class's circuit closed. Found by
        # Codex code review.
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))

            call_count = {"n": 0}

            def flaky_local(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("/count_cached_tokens"):
                    return httpx.Response(
                        200, json={"input_tokens": 100, "cached_tokens": 0}
                    )
                call_count["n"] += 1
                # Schema-invalid on every real call -- if the interleaved
                # transport failures below were wrongly recorded as
                # successes, they would dilute this below the 0.25
                # threshold and the circuit would never open.
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "tool_use",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": f"t{call_count['n']}",
                                "name": "my_tool",
                                "input": {},
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                )

            def flaky_transport(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("/count_cached_tokens"):
                    return httpx.Response(
                        200, json={"input_tokens": 100, "cached_tokens": 0}
                    )
                raise httpx.ConnectError("connection reset", request=request)

            def cloud(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "cloud"}],
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                )

            async with app.router.lifespan_context(app):
                await app.state.clients["local"].aclose()
                await app.state.clients["cloud"].aclose()
                app.state.clients["cloud"] = httpx.AsyncClient(
                    base_url="https://cloud.test", transport=httpx.MockTransport(cloud)
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://edge.test",
                ) as client:
                    # 5 real schema failures interleaved with 15 transport
                    # failures on the same tool-suite class. Once the 5th
                    # real failure lands (min_samples=5, all failures), the
                    # circuit legitimately opens and later same-class calls
                    # correctly start routing cloud instead -- that's Rung
                    # 1 working as designed, not a test bug; a mocked cloud
                    # backend keeps those calls from erroring on a real
                    # network attempt.
                    for i in range(20):
                        app.state.clients["local"] = httpx.AsyncClient(
                            base_url="http://local.test",
                            transport=httpx.MockTransport(
                                flaky_transport if i % 4 != 0 else flaky_local
                            ),
                        )
                        response = await client.post("/v1/messages", json=_request(i))
                        # 200 for the real (flaky_local) calls; 502 for the
                        # transport failures (matches server.py's existing
                        # httpx.HTTPError handling), or 200 if that call
                        # happened to get routed cloud by an already-open
                        # circuit and the mocked cloud backend served it.
                        self.assertIn(response.status_code, (200, 502))
                        await app.state.clients["local"].aclose()

            rows = records(trace_dir)
            real_failures = [
                r
                for r in rows
                if r["placement"] == "local"
                and r["call"]["tool_use_blocks"]
                and r["call"]["tool_use_blocks"][0]["schema_valid"] is False
            ]
            # Exactly the 5 real schema-invalid calls from the loop (i=0,4,
            # 8,12,16) -- confirms none of the 15 transport failures got
            # mistaken for a real local call.
            self.assertEqual(len(real_failures), 5)
            # rows[16] is the 5th real failure itself (the field on its own
            # record reflects state *before* it, i.e. only 4 samples, still
            # below min_samples=5 -- None). Every row after it reflects the
            # breaker having reached exactly 5/5 failures: if any of the 15
            # transport failures had been wrongly recorded as a false
            # success, this would read below 1.0. This holds regardless of
            # whether any individual later call was itself routed local
            # (probe) or cloud (circuit open) -- the field is written
            # unconditionally at decision time either way.
            for row in rows[17:]:
                self.assertEqual(row["reliability"]["failure_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
