import json
import tempfile
import unittest
from pathlib import Path

import httpx

from edgeproxy.config import Config
from edgeproxy.server import _apply_local_generation_controls, make_app


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
    return out


class ServerLocalCacheIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_controls_translate_claude_high_effort_for_qwen(self):
        request = {
            "temperature": 0.4,
            "output_config": {"effort": "high"},
        }

        original_temperature, strict_tools_added = _apply_local_generation_controls(
            request
        )

        self.assertEqual(original_temperature, 0.4)
        self.assertEqual(strict_tools_added, 0)
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["output_config"]["effort"], "xhigh")

    async def test_local_controls_leave_supported_effort_unchanged(self):
        request = {"output_config": {"effort": "medium"}}

        _apply_local_generation_controls(request)

        self.assertEqual(request["output_config"]["effort"], "medium")

    async def test_probe_failure_falls_back_to_cloud(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))

            def local_probe(request: httpx.Request) -> httpx.Response:
                return httpx.Response(404, json={"error": "probe unavailable"})

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
                    transport=httpx.MockTransport(local_probe),
                )
                app.state.clients["cloud"] = httpx.AsyncClient(
                    base_url="https://cloud.test", transport=httpx.MockTransport(cloud)
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://edge.test",
                ) as client:
                    response = await client.post(
                        "/v1/messages",
                        json={
                            "model": "claude-sonnet-5",
                            "max_tokens": 8,
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )
                    self.assertEqual(response.status_code, 200)

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "cloud")
            self.assertEqual(record["reason"], "local-probe-unavailable")
            self.assertFalse(record["local_cache"]["prediction"]["available"])

    async def test_local_probe_and_actual_are_traced_with_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))
            sent_max_tokens = []

            def local(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("/count_cached_tokens"):
                    return httpx.Response(
                        200, json={"input_tokens": 100, "cached_tokens": 96}
                    )
                sent_max_tokens.append(json.loads(request.content)["max_tokens"])
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "ok"}],
                        "usage": {
                            "input_tokens": 4,
                            "output_tokens": 1,
                            "cache_read_input_tokens": 96,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                )

            async with app.router.lifespan_context(app):
                await app.state.clients["local"].aclose()
                app.state.clients["local"] = httpx.AsyncClient(
                    base_url="http://local.test", transport=httpx.MockTransport(local)
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://edge.test",
                ) as client:
                    response = await client.post(
                        "/v1/messages",
                        json={
                            "model": "claude-sonnet-5",
                            "max_tokens": 64_000,
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )
                    self.assertEqual(response.status_code, 200)

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "local")
            self.assertNotIn("est_prompt_tokens", record["features"])
            self.assertEqual(record["features"]["local_prompt_tokens"], 100)
            self.assertEqual(
                record["local_cache"]["prediction"]["estimated_read_tokens"], 96
            )
            self.assertEqual(
                record["local_cache"]["actual"]["cache_read_input_tokens"], 96
            )
            self.assertTrue(
                record["local_cache"]["agreement"]["within_5_percent_of_input"]
            )
            self.assertEqual(record["call"]["schema_version"], "edgeproxy.call.v1")
            self.assertEqual(record["call"]["backend"], "local")
            self.assertEqual(record["call"]["tokens"]["prompt_tokens_exact"], 100)
            self.assertEqual(record["call"]["tokens"]["cache_read_tokens"], 96)
            self.assertEqual(sent_max_tokens, [53_900])
            self.assertEqual(record["requested_max_tokens"], 64_000)
            self.assertEqual(record["effective_max_tokens"], 53_900)
            self.assertEqual(record["output_reserve_tokens"], 0)

    async def test_exact_probe_length_prevents_local_context_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))

            def local_probe(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200, json={"input_tokens": 55_905, "cached_tokens": 0}
                )

            def cloud(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    json={
                        "type": "message",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "cloud"}],
                        "usage": {"input_tokens": 55_905, "output_tokens": 1},
                    },
                )

            async with app.router.lifespan_context(app):
                await app.state.clients["local"].aclose()
                await app.state.clients["cloud"].aclose()
                app.state.clients["local"] = httpx.AsyncClient(
                    base_url="http://local.test",
                    transport=httpx.MockTransport(local_probe),
                )
                app.state.clients["cloud"] = httpx.AsyncClient(
                    base_url="https://cloud.test", transport=httpx.MockTransport(cloud)
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://edge.test",
                ) as client:
                    response = await client.post(
                        "/v1/messages",
                        json={
                            "model": "claude-sonnet-5",
                            "max_tokens": 4096,
                            # Deliberately tiny text: the legacy character
                            # estimate would fit, while the exact probe does not.
                            "messages": [{"role": "user", "content": "large"}],
                        },
                    )
                    self.assertEqual(response.status_code, 200)

            record = records(trace_dir)[0]
            self.assertEqual(record["placement"], "cloud")
            self.assertEqual(record["reason"], "too-large")
            self.assertEqual(record["features"]["local_prompt_tokens"], 55_905)
            self.assertEqual(
                record["local_cache"]["actual"]["selected_backend"], False
            )


if __name__ == "__main__":
    unittest.main()
