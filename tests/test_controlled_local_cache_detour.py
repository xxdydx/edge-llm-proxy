"""Synthetic transport tests only: no GPU, cloud or Docker calls."""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from scripts.controlled_local_cache_detour import AtomicCheckpoint, Config, ProbeInvalid, _orders, run_probe


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.start = dt.datetime(2026, 9, 17, 8, tzinfo=dt.timezone(dt.timedelta(hours=8)))

    def monotonic(self) -> float:
        return self.t

    def now(self) -> dt.datetime:
        return self.start + dt.timedelta(seconds=self.t)

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def stream_response(model: str, cached: int = 100, *, total: int = 5000,
                    missing_usage: bool = False) -> httpx.Response:
    usage = {"input_tokens": 1255, "cache_read_input_tokens": cached,
             "cache_creation_input_tokens": total - 1255 - cached}
    if missing_usage:
        del usage["cache_read_input_tokens"]
    events = [
        {"type": "message_start", "message": {"model": model, "usage": usage}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "OK"}},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    return httpx.Response(200, text="".join("data: " + json.dumps(event) + "\n\n" for event in events),
                          headers={"content-type": "text/event-stream"})


class Harness:
    def __init__(self, *, missing_usage: bool = False, waiting: bool = False,
                 count_usage_mismatch: bool = False) -> None:
        self.clock = Clock()
        self.local_bodies: list[bytes] = []
        self.cloud_bodies: list[dict] = []
        self.seen: set[bytes] = set()
        self.missing_usage = missing_usage
        self.waiting = waiting
        self.count_usage_mismatch = count_usage_mismatch
        self.local = httpx.Client(base_url="http://local", transport=httpx.MockTransport(self.local_handler))
        self.cloud = httpx.Client(base_url="http://cloud", transport=httpx.MockTransport(self.cloud_handler))

    def local_handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text=(
                "vllm:num_requests_running 0\n"
                f"vllm:num_requests_waiting {1 if self.waiting else 0}\n"
                "vllm:kv_cache_usage_perc 0.2\n"
            ))
        body = request.content
        payload = json.loads(body)
        assert payload["stream"] is True
        assert "tools" not in payload
        assert "cache_salt" not in payload
        assert payload["messages"][0]["content"].startswith("Isolation ID ")
        if request.url.path.endswith("count_cached_tokens"):
            return httpx.Response(200, json={"input_tokens": 5000, "cached_tokens": 100 if body in self.seen else 0})
        if request.url.path == "/v1/messages":
            was_seen = body in self.seen
            self.local_bodies.append(body)
            self.seen.add(body)
            self.clock.t += 0.1
            return stream_response(payload["model"], cached=100 if was_seen else 0,
                                   total=4999 if self.count_usage_mismatch else 5000,
                                   missing_usage=self.missing_usage)
        raise AssertionError(f"unexpected local path: {request.url.path}")

    def cloud_handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-v4-flash"
        assert payload["max_tokens"] == 1
        assert "tools" not in payload
        self.cloud_bodies.append(payload)
        self.clock.t += 0.2
        return httpx.Response(200, json={"model": "deepseek-v4-flash", "content": [{"type": "text", "text": "OK"}]})

    def run(self, pairs: int = 2, deadline_seconds: int = 3600) -> dict:
        cfg = Config("local-model", "low", pairs=pairs, interval_s=30,
                     deadline=self.clock.start + dt.timedelta(seconds=deadline_seconds), run_id="synthetic-test-run")
        return run_probe(self.local, self.cloud, cfg, monotonic=self.clock.monotonic,
                         sleep=self.clock.sleep, now=self.clock.now)


class ControlledCacheDetourTests(unittest.TestCase):
    def test_randomized_balanced_orders(self) -> None:
        for seed in range(8):
            orders = _orders(2, seed)
            self.assertEqual(orders[1], list(reversed(orders[0])))
            self.assertEqual(set(orders[0]), {"idle", "cloud"})
        with self.assertRaises(ValueError):
            _orders(4, 1)

    def test_two_matched_pairs_use_identical_bytes_within_arm(self) -> None:
        h = Harness()
        result = h.run()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(h.local_bodies), 8)
        self.assertEqual(len(h.cloud_bodies), 2)
        self.assertEqual(len(set(h.local_bodies)), 4)
        for l0, l1 in zip(h.local_bodies[::2], h.local_bodies[1::2]):
            self.assertIs(l0, l1)
        for pair in result["pairs"]:
            self.assertTrue(pair["valid"])
            for arm in pair["arms"]:
                self.assertEqual(arm["count_cold"]["cached_tokens"], 0)
                self.assertEqual(arm["count_warm"]["cached_tokens"], 100)
                self.assertEqual(arm["count_pre_L1"]["cached_tokens"], 100)
                self.assertAlmostEqual(arm["L0_to_L1_start_s"], 30)
                self.assertEqual(arm["L1"]["cache_read_input_tokens"], 100)

    def test_three_pairs_keep_unique_unsalted_prefixes_and_no_reset(self) -> None:
        h = Harness()
        result = h.run(pairs=3)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(h.local_bodies), 12)
        self.assertEqual(len(set(h.local_bodies)), 6)
        self.assertEqual(len(h.cloud_bodies), 3)
        self.assertEqual(result["orders"][0], result["orders"][2])
        self.assertEqual(result["orders"][1], list(reversed(result["orders"][0])))
        prefixes = [json.loads(body)["messages"][0]["content"].split(". ", 1)[0]
                    for body in h.local_bodies[::2]]
        self.assertEqual(len(set(prefixes)), 6)
        self.assertTrue(all(len(prefix) >= 64 for prefix in prefixes))

    def test_missing_actual_cache_usage_fails_closed(self) -> None:
        h = Harness(missing_usage=True)
        result = h.run()
        self.assertEqual(result["status"], "invalid_stopped")
        self.assertIn("missing_or_invalid_final_usage_bucket", result["pairs"][0]["invalid_reason"])
        self.assertEqual(h.cloud_bodies, [])

    def test_count_usage_mismatch_fails_before_cloud(self) -> None:
        h = Harness(count_usage_mismatch=True)
        result = h.run()
        self.assertEqual(result["status"], "invalid_stopped")
        self.assertIn("count_usage_input_mismatch", result["pairs"][0]["invalid_reason"])
        self.assertEqual(h.cloud_bodies, [])

    def test_no_cloud_after_failed_warm_gate(self) -> None:
        h = Harness()
        h.local_handler_original = h.local_handler

        def never_warms(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("count_cached_tokens"):
                return httpx.Response(200, json={"input_tokens": 5000, "cached_tokens": 0})
            return h.local_handler_original(request)

        h.local.close()
        h.local = httpx.Client(base_url="http://local", transport=httpx.MockTransport(never_warms))
        result = h.run()
        self.assertEqual(result["status"], "invalid_stopped")
        self.assertIn("L0_did_not_warm_prefix", result["pairs"][0]["invalid_reason"])
        self.assertEqual(h.cloud_bodies, [])

    def test_nonidle_queue_fails_before_generation(self) -> None:
        h = Harness(waiting=True)
        result = h.run()
        self.assertEqual(result["status"], "invalid_stopped")
        self.assertIn("local_not_idle", result["pairs"][0]["invalid_reason"])
        self.assertEqual(h.local_bodies, [])

    def test_deadline_fails_before_any_request(self) -> None:
        h = Harness()
        with self.assertRaisesRegex(ProbeInvalid, "insufficient_full_probe_deadline_budget"):
            h.run(deadline_seconds=60)
        self.assertEqual(h.local_bodies, [])
        self.assertEqual(h.cloud_bodies, [])

    def test_atomic_checkpoint_retains_completed_arm_on_interruption(self) -> None:
        h = Harness()
        cfg = Config("local-model", "low", deadline=h.clock.start + dt.timedelta(hours=1),
                     run_id="synthetic-test-run")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "measurement.json"
            writer = AtomicCheckpoint(path, cfg.run_id)
            self.assertEqual(json.loads(path.read_text())["status"], "reserved_not_started")

            def interrupt_after_first_arm(result: dict) -> None:
                writer(result)
                if sum(a["status"] == "complete" for p in result["pairs"] for a in p["arms"]) == 1:
                    raise RuntimeError("synthetic_interruption")

            with self.assertRaisesRegex(RuntimeError, "synthetic_interruption"):
                run_probe(h.local, h.cloud, cfg, monotonic=h.clock.monotonic,
                          sleep=h.clock.sleep, now=h.clock.now, checkpoint=interrupt_after_first_arm)
            retained = json.loads(path.read_text())
            self.assertEqual(retained["status"], "running")
            self.assertEqual(retained["pairs"][0]["arms"][0]["status"], "complete")
            self.assertIn("L1", retained["pairs"][0]["arms"][0])
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])
            with self.assertRaises(FileExistsError):
                AtomicCheckpoint(path, "different-run")

    def test_every_completed_arm_is_checkpointed(self) -> None:
        h = Harness()
        snapshots: list[dict] = []
        cfg = Config("local-model", "low", deadline=h.clock.start + dt.timedelta(hours=1),
                     run_id="synthetic-test-run")
        result = run_probe(h.local, h.cloud, cfg, monotonic=h.clock.monotonic,
                           sleep=h.clock.sleep, now=h.clock.now,
                           checkpoint=lambda row: snapshots.append(json.loads(json.dumps(row))))
        self.assertEqual(result["status"], "complete")
        counts = [sum(a["status"] == "complete" for p in s["pairs"] for a in p["arms"])
                  for s in snapshots]
        for n in range(1, 5):
            self.assertIn(n, counts)
        self.assertEqual(snapshots[-1]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
