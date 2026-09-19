import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import httpx

from edgeproxy.config import Config
from edgeproxy.server import make_app


ARTIFACT = Path(__file__).resolve().parents[1] / "experiments/agentic_router/results/quality_model_baseline205_v5_shadow_logreg.json"


def config(trace_dir):
    return Config(
        host="127.0.0.1", port=0, upstream="https://cloud.test",
        trace_dir=trace_dir, vllm_url="http://local.test", policy="cloud-only",
        shaping="none", link_preset="none", cloud_delay_ms=0.0,
        cloud_jitter_ms=0.0, cloud_bandwidth_mbps=0.0,
        local_model_name="local", resource_sample_interval_s=60.0,
        gpu_index=0, kv_bytes_per_token=None, cloud_cache_tracking="off",
        local_cache_tracking="observe", episode_id="shadow-test",
        agentic_shadow_artifact=ARTIFACT,
    )


class ServerAgenticShadowTests(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_scores_complete_predecision_history_without_changing_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            app = make_app(config(trace_dir))

            def local(request):
                if request.url.path.endswith("/count_cached_tokens"):
                    body = json.loads(request.content)
                    self.assertEqual(body["output_config"]["effort"], "medium")
                    return httpx.Response(200, json={"input_tokens": 100, "cached_tokens": 0})
                self.fail("shadow must not dispatch to local")

            def cloud(request):
                return httpx.Response(200, json={
                    "type": "message", "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                })

            body = {
                "model": "test", "max_tokens": 10,
                "output_config": {"effort": "high"},
                "tools": [{"name": "Read", "input_schema": {"type": "object", "properties": {}}}],
                "messages": [{"role": "user", "content": "hello"}],
            }
            async with app.router.lifespan_context(app):
                for backend, handler, url in (
                    ("local", local, "http://local.test"),
                    ("cloud", cloud, "https://cloud.test"),
                ):
                    await app.state.clients[backend].aclose()
                    app.state.clients[backend] = httpx.AsyncClient(
                        base_url=url, transport=httpx.MockTransport(handler)
                    )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://edge.test"
                ) as client:
                    for _ in range(2):
                        response = await client.post(
                            "/v1/messages", json=body,
                            headers={"x-claude-code-session-id": "session-one"},
                        )
                        self.assertEqual(response.status_code, 200)

            records = [json.loads(line) for path in trace_dir.glob("*.jsonl") for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual([r["placement"] for r in records], ["cloud", "cloud"])
            self.assertEqual([r["agentic_shadow"]["feature_status"] for r in records], ["complete", "complete"])
            self.assertEqual([r["features"]["agentic_shadow_features"]["turn_index"] for r in records], [0.0, 1.0])
            self.assertEqual(records[1]["features"]["agentic_shadow_features"]["consecutive_same_backend_turns"], 1.0)
            self.assertIn("raw_uncalibrated_harm=", records[0]["agentic_shadow"]["detail"])
            self.assertIn("adaptive-unsupported", records[0]["agentic_shadow"]["detail"])


if __name__ == "__main__":
    unittest.main()
