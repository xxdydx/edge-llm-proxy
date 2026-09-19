import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from edgeproxy.config import Config
from edgeproxy import router
from edgeproxy.server import _leader_warming_exemption_allowed, make_app


HEADERS = {"x-claude-code-session-id": "barrier-session"}


def config(trace_dir: Path, *, timeout_ms: float = 100.0) -> Config:
    return Config(
        host="127.0.0.1",
        port=0,
        upstream="https://cloud.test",
        trace_dir=trace_dir,
        vllm_url="http://local.test",
        policy="static",
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
        cohort_tracking="observe",
        # This is the existing benchmark-only Rung 4 opt-in. The coordinator
        # is adjacent to, and does not alter, its parent-candidate predicate.
        cohort_parent_placement=True,
        cohort_window_ms=300.0,
        cohort_barrier_timeout_ms=timeout_ms,
        cohort_barrier_poll_ms=5.0,
    )


def request(prompt: str) -> dict:
    return {
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    }


def records(trace_dir: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for path in trace_dir.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    rows.sort(key=lambda row: row["ts"])
    return rows


class BarrierBackend:
    def __init__(self, delegations: list[tuple[str, str]]) -> None:
        self.delegations = delegations
        self.warm = False
        self.fail_leader = False
        self.leader_started = asyncio.Event()
        self.release_leader = asyncio.Event()
        self.poll_seen = asyncio.Event()
        self.probe_calls = 0
        self.generations: list[str] = []

    async def __call__(self, backend_request: httpx.Request) -> httpx.Response:
        if backend_request.url.path.endswith("/count_cached_tokens"):
            self.probe_calls += 1
            # Three decision-time probes normally precede the barrier.  Even
            # when a follower arrives first, the fourth call can only occur
            # after that follower has entered barrier polling.
            if self.probe_calls >= 4:
                self.poll_seen.set()
            return httpx.Response(
                200,
                json={
                    "input_tokens": 100,
                    "cached_tokens": 90 if self.warm else 0,
                },
            )

        body = json.loads(backend_request.content)
        prompt = body["messages"][0]["content"]
        self.generations.append(prompt)
        if prompt == "seed":
            return httpx.Response(
                200,
                json={
                    "type": "message",
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": "Agent",
                            "input": {"prompt": child_prompt},
                        }
                        for tool_id, child_prompt in self.delegations
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 4},
                },
            )
        if prompt == self.delegations[0][1]:
            self.leader_started.set()
            await self.release_leader.wait()
            if self.fail_leader:
                raise httpx.ConnectError("leader failed", request=backend_request)

        return httpx.Response(
            200,
            json={
                "type": "message",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 90 if self.warm else 0,
                    "output_tokens": 1,
                },
            },
        )


class ServerCohortBarrierIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def run_app(self, cfg: Config, backend: BarrierBackend, scenario) -> None:
        app = make_app(cfg)

        def cloud(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "unexpected cloud call"})

        async with app.router.lifespan_context(app):
            for name in ("local", "cloud"):
                await app.state.clients[name].aclose()
            app.state.clients["local"] = httpx.AsyncClient(
                base_url="http://local.test",
                transport=httpx.MockTransport(backend),
            )
            app.state.clients["cloud"] = httpx.AsyncClient(
                base_url="https://cloud.test",
                transport=httpx.MockTransport(cloud),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://edge.test",
            ) as client:
                seed = await client.post("/v1/messages", json=request("seed"), headers=HEADERS)
                self.assertEqual(seed.status_code, 200)
                await scenario(client)

    async def test_follower_waits_until_cache_warm_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            backend = BarrierBackend([("agent-a", "leader"), ("agent-b", "follower")])

            async def scenario(client):
                leader = asyncio.create_task(
                    client.post("/v1/messages", json=request("leader"), headers=HEADERS)
                )
                await asyncio.wait_for(backend.leader_started.wait(), 1.0)
                follower = asyncio.create_task(
                    client.post("/v1/messages", json=request("follower"), headers=HEADERS)
                )
                await asyncio.wait_for(backend.poll_seen.wait(), 1.0)
                self.assertEqual(backend.generations, ["seed", "leader"])

                backend.warm = True
                backend.release_leader.set()
                leader_response, follower_response = await asyncio.gather(leader, follower)
                self.assertEqual(leader_response.status_code, 200)
                self.assertEqual(follower_response.status_code, 200)

            await self.run_app(config(trace_dir, timeout_ms=250), backend, scenario)

            traced = {row["request"]["messages"][0]["content"]: row for row in records(trace_dir)}
            self.assertEqual(traced["leader"]["cohort_dispatch"]["role"], "leader")
            dispatch = traced["follower"]["cohort_dispatch"]
            self.assertEqual(dispatch["role"], "follower")
            self.assertEqual(dispatch["release_reason"], "cache-warm-confirmed")
            self.assertGreater(dispatch["actual_wait_ms"], 0)
            self.assertGreaterEqual(dispatch["probe_attempts"], 1)
            self.assertEqual(dispatch["realized_outcome"]["actual_cache_read_tokens"], 90)

    async def test_follower_arriving_before_leader_refreshes_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            backend = BarrierBackend([("agent-a", "leader"), ("agent-b", "follower")])

            async def scenario(client):
                follower = asyncio.create_task(
                    client.post("/v1/messages", json=request("follower"), headers=HEADERS)
                )
                await asyncio.wait_for(backend.poll_seen.wait(), 1.0)
                self.assertEqual(backend.generations, ["seed"])

                leader = asyncio.create_task(
                    client.post("/v1/messages", json=request("leader"), headers=HEADERS)
                )
                await asyncio.wait_for(backend.leader_started.wait(), 1.0)
                backend.warm = True
                backend.release_leader.set()

                leader_response, follower_response = await asyncio.gather(leader, follower)
                self.assertEqual(leader_response.status_code, 200)
                self.assertEqual(follower_response.status_code, 200)

            await self.run_app(config(trace_dir, timeout_ms=250), backend, scenario)

            traced = {row["request"]["messages"][0]["content"]: row for row in records(trace_dir)}
            dispatch = traced["follower"]["cohort_dispatch"]
            self.assertEqual(dispatch["role"], "follower")
            self.assertEqual(dispatch["shared_prefix_threshold_tokens"], 80)
            self.assertEqual(dispatch["release_reason"], "cache-warm-confirmed")
            self.assertGreaterEqual(dispatch["probe_attempts"], 1)
            self.assertLess(dispatch["actual_wait_ms"], 250)

    async def test_failed_leader_cannot_strand_follower(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            backend = BarrierBackend([("agent-a", "leader"), ("agent-b", "follower")])
            backend.fail_leader = True

            async def scenario(client):
                leader = asyncio.create_task(
                    client.post("/v1/messages", json=request("leader"), headers=HEADERS)
                )
                await asyncio.wait_for(backend.leader_started.wait(), 1.0)
                follower = asyncio.create_task(
                    client.post("/v1/messages", json=request("follower"), headers=HEADERS)
                )
                follower_response = await asyncio.wait_for(follower, 1.0)
                self.assertEqual(follower_response.status_code, 200)
                self.assertIn("follower", backend.generations)
                backend.release_leader.set()
                leader_response = await leader
                self.assertEqual(leader_response.status_code, 502)

            await self.run_app(config(trace_dir, timeout_ms=60), backend, scenario)

            traced = {row["request"]["messages"][0]["content"]: row for row in records(trace_dir)}
            dispatch = traced["follower"]["cohort_dispatch"]
            self.assertEqual(dispatch["release_reason"], "timeout")
            self.assertGreaterEqual(dispatch["actual_wait_ms"], 40)

    async def test_two_concurrent_followers_share_exactly_one_leader(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            backend = BarrierBackend(
                [("agent-a", "leader"), ("agent-b", "follower-b"), ("agent-c", "follower-c")]
            )

            async def scenario(client):
                leader = asyncio.create_task(
                    client.post("/v1/messages", json=request("leader"), headers=HEADERS)
                )
                await asyncio.wait_for(backend.leader_started.wait(), 1.0)
                followers = [
                    asyncio.create_task(
                        client.post("/v1/messages", json=request(prompt), headers=HEADERS)
                    )
                    for prompt in ("follower-b", "follower-c")
                ]
                await asyncio.sleep(0.03)
                self.assertEqual(backend.generations, ["seed", "leader"])
                backend.warm = True
                backend.release_leader.set()
                responses = await asyncio.gather(leader, *followers)
                self.assertTrue(all(response.status_code == 200 for response in responses))

            await self.run_app(config(trace_dir, timeout_ms=250), backend, scenario)

            dispatches = [
                row["cohort_dispatch"]
                for row in records(trace_dir)
                if row.get("cohort_dispatch") is not None
            ]
            self.assertEqual(sum(item["role"] == "leader" for item in dispatches), 1)
            self.assertEqual(sum(item["role"] == "follower" for item in dispatches), 2)
            self.assertEqual(len({item["leader_call_id"] for item in dispatches}), 1)

    async def test_elapsed_hold_window_dispatches_without_spurious_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            backend = BarrierBackend([("agent-a", "leader"), ("agent-b", "follower")])

            async def scenario(client):
                leader = asyncio.create_task(
                    client.post("/v1/messages", json=request("leader"), headers=HEADERS)
                )
                await asyncio.wait_for(backend.leader_started.wait(), 1.0)
                await asyncio.sleep(0.06)
                started = time.monotonic()
                follower_response = await client.post(
                    "/v1/messages", json=request("follower"), headers=HEADERS
                )
                elapsed_ms = (time.monotonic() - started) * 1000
                self.assertEqual(follower_response.status_code, 200)
                self.assertLess(elapsed_ms, 30)
                backend.release_leader.set()
                self.assertEqual((await leader).status_code, 200)

            await self.run_app(config(trace_dir, timeout_ms=40), backend, scenario)

            traced = {row["request"]["messages"][0]["content"]: row for row in records(trace_dir)}
            dispatch = traced["follower"]["cohort_dispatch"]
            self.assertEqual(dispatch["release_reason"], "hold-window-elapsed")
            self.assertEqual(dispatch["actual_wait_ms"], 0.0)


class LeaderWarmingExemptionTests(unittest.TestCase):
    def features(self, **overrides) -> router.CallFeatures:
        values = {
            "model": "claude-sonnet-5",
            "has_tools": False,
            "n_tools": 0,
            "has_server_tools": False,
            "n_messages": 1,
            "est_system_tokens": 0,
            "max_tokens": 64,
            "stream": False,
            "is_tool_continuation": False,
            "local_prompt_tokens": 100,
        }
        values.update(overrides)
        return router.CallFeatures(**values)

    def test_only_cold_preference_gate_can_be_exempted(self):
        policy = router.WarmLocalPolicy()
        features = self.features()
        decision = policy.decide(features)

        self.assertEqual(decision.reason, "cold-branch-first-turn")
        self.assertTrue(
            _leader_warming_exemption_allowed(policy, features, decision)
        )

        blocked = self.features(has_server_tools=True)
        self.assertFalse(
            _leader_warming_exemption_allowed(
                policy, blocked, policy.decide(blocked)
            )
        )

    def test_branch_headroom_gate_survives_cold_leader_exemption(self):
        policy = router.BranchDriftPolicy()
        features = self.features(local_prompt_tokens=40_000)
        decision = policy.decide(features)

        # WarmLocalPolicy encounters the cold preference first, but the
        # adjacent exemption must not thereby bypass BranchDrift's hard gate.
        self.assertEqual(decision.reason, "cold-branch-first-turn")
        self.assertFalse(
            _leader_warming_exemption_allowed(policy, features, decision)
        )

    def test_combined_policy_branch_drift_gate_survives_cold_leader_exemption(self):
        # CombinedPolicy does not subclass BranchDriftPolicy, so it needs
        # its own dedicated branch in _leader_warming_exemption_allowed —
        # this would silently pass (wrongly) without that branch.
        policy = router.CombinedPolicy()
        features = self.features(local_prompt_tokens=40_000, branch_turn_ordinal=10)
        decision = policy.decide(features)

        self.assertEqual(decision.reason, "cold-branch-first-turn")
        self.assertFalse(
            _leader_warming_exemption_allowed(policy, features, decision)
        )

    def test_combined_policy_planning_turn_gate_survives_cold_leader_exemption(self):
        policy = router.CombinedPolicy(planning_turns=3)
        features = self.features(local_prompt_tokens=100, branch_turn_ordinal=1)
        decision = policy.decide(features)

        self.assertEqual(decision.reason, "cold-branch-first-turn")
        self.assertFalse(
            _leader_warming_exemption_allowed(policy, features, decision)
        )

    def test_combined_policy_predicted_risk_gate_survives_cold_leader_exemption(self):
        policy = router.CombinedPolicy(risk_threshold=0.4)
        features = self.features(
            local_prompt_tokens=100,
            branch_turn_ordinal=10,
            predicted_local_risk_score=0.9,
        )
        decision = policy.decide(features)

        self.assertEqual(decision.reason, "cold-branch-first-turn")
        self.assertFalse(
            _leader_warming_exemption_allowed(policy, features, decision)
        )

    def test_combined_policy_exemption_allowed_when_no_gate_would_fire(self):
        policy = router.CombinedPolicy()
        features = self.features(
            local_prompt_tokens=100,
            branch_turn_ordinal=10,
            predicted_local_risk_score=0.01,
        )
        decision = policy.decide(features)

        self.assertEqual(decision.reason, "cold-branch-first-turn")
        self.assertTrue(
            _leader_warming_exemption_allowed(policy, features, decision)
        )


if __name__ == "__main__":
    unittest.main()
