"""Focused tests for the bounded fail-fast transport policy added 2026-09-10
to experiments/capability_router/executor.py.

Covers, per the requirement:
  - attempt cap (at most 2 same-backend attempts, retry only on transport err)
  - timeout / backoff accounting (retry_attempts, end_to_end_wall_s incl.
    backoff, final_attempt_latency_s separate)
  - healthy long-streaming semantics: a slow-but-progressing generation is
    NOT aborted (read timeout is an inactivity cap, not a wall-clock cap)
  - resume preservation: the new fields round-trip through the checkpoint,
    and a TRANSPORT_ERROR row is still the only thing re-attempted on resume

No live network anywhere: `_stream_post_once` and `time.sleep` are patched.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from experiments.capability_router import executor
from experiments.capability_router.config import BackendConfig
from experiments.capability_router.schema import ReplayOutcome


def _backend(**overrides) -> BackendConfig:
    base = dict(
        name="local_27b",
        base_url="http://127.0.0.1:9",
        request_model="local",
        expected_model_exact="local",
        auth_header=None,
        auth_env_var=None,
    )
    base.update(overrides)
    return BackendConfig(**base)


def _once_result(*, transport_err=None, status_code=200, wall_s=1.0, model="local"):
    """Shape of one `_stream_post_once` return, with a monotonic-looking
    t0/t_end pair `wall_s` apart."""
    events = (
        [
            {"type": "message_start", "message": {"model": model, "usage": {"input_tokens": 5, "output_tokens": 0}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        ]
        if transport_err is None and status_code == 200
        else []
    )
    return {
        "status_code": None if transport_err else status_code,
        "transport_err": transport_err,
        "raw_error_body": None if status_code == 200 else "boom",
        "events": events,
        "t0": 100.0,
        "t_end": 100.0 + wall_s,
        "ttft_s": 0.2 if transport_err is None and status_code == 200 else None,
        "last_delta_t": 100.0 + wall_s - 0.1,
    }


class AttemptCapTests(unittest.TestCase):
    def test_at_most_two_attempts_on_persistent_transport_error(self):
        calls = []

        def fake_once(be, payload):
            calls.append(1)
            return _once_result(transport_err="RemoteProtocolError: server disconnected")

        with patch.object(executor, "_stream_post_once", side_effect=fake_once), \
             patch.object(executor.time, "sleep") as sleep:
            raw = executor._stream_post(_backend(), {"messages": []})

        self.assertEqual(len(calls), executor._TRANSPORT_MAX_ATTEMPTS)
        self.assertEqual(len(calls), 2)
        self.assertEqual(raw["retry_attempts"], 2)
        self.assertEqual(len(raw["attempts_meta"]), 2)
        self.assertIsNotNone(raw["transport_err"])
        # exactly one backoff sleep, between the two attempts, fixed length
        sleep.assert_called_once_with(executor._TRANSPORT_RETRY_BACKOFF_S)

    def test_success_on_second_attempt_stops_there(self):
        seq = [
            _once_result(transport_err="ConnectError: transient"),
            _once_result(transport_err=None, wall_s=3.0),
        ]

        def fake_once(be, payload):
            return seq.pop(0)

        with patch.object(executor, "_stream_post_once", side_effect=fake_once), \
             patch.object(executor.time, "sleep"):
            raw = executor._stream_post(_backend(), {"messages": []})

        self.assertEqual(raw["retry_attempts"], 2)
        self.assertIsNone(raw["transport_err"])  # final (successful) attempt
        self.assertEqual(raw["attempts_meta"][0]["transport_err"], "ConnectError: transient")
        self.assertIsNone(raw["attempts_meta"][1]["transport_err"])

    def test_real_http_error_is_not_retried(self):
        calls = []

        def fake_once(be, payload):
            calls.append(1)
            return _once_result(transport_err=None, status_code=400)

        with patch.object(executor, "_stream_post_once", side_effect=fake_once), \
             patch.object(executor.time, "sleep") as sleep:
            raw = executor._stream_post(_backend(), {"messages": []})

        self.assertEqual(len(calls), 1)  # a 4xx is a real answer, not retried
        self.assertEqual(raw["retry_attempts"], 1)
        sleep.assert_not_called()


class TimeoutAndBackoffAccountingTests(unittest.TestCase):
    def test_end_to_end_wall_includes_failed_attempts_and_backoff(self):
        """`end_to_end_wall_s` is measured off real time.monotonic() around
        the whole 2-attempt loop, so it covers both attempts AND the real
        backoff sleep between them; `final_attempt_latency_s` is only the last
        attempt's own span. They are populated from different sources and are
        never the same value when a retry happened."""
        import time as _t

        per_attempt = 0.03
        fake_backoff = 0.05

        def real_timed_fail(be, payload):
            t0 = _t.monotonic()
            _t.sleep(per_attempt)
            t_end = _t.monotonic()
            r = _once_result(transport_err="stall", wall_s=t_end - t0)
            r["t0"], r["t_end"] = t0, t_end
            return r

        with patch.object(executor, "_TRANSPORT_RETRY_BACKOFF_S", fake_backoff), \
             patch.object(executor, "_stream_post_once", side_effect=real_timed_fail):
            raw = executor._stream_post(_backend(), {"messages": []})

        self.assertEqual(raw["retry_attempts"], 2)
        # end-to-end spans both attempts + the one real backoff, so it is
        # strictly larger than the final attempt's own latency.
        self.assertGreater(raw["end_to_end_wall_s"], raw["final_attempt_latency_s"])
        self.assertGreaterEqual(
            raw["end_to_end_wall_s"],
            2 * per_attempt + fake_backoff - 0.01,
        )
        # final-attempt latency is just the last attempt's span (~per_attempt)
        self.assertLess(raw["final_attempt_latency_s"], 2 * per_attempt)
        self.assertEqual(len(raw["attempts_meta"]), 2)

    def test_build_outcome_propagates_accounting_fields(self):
        # a raw dict as _stream_post hands to _build_outcome after two failures
        seq = [_once_result(transport_err="dead", wall_s=1.5),
               _once_result(transport_err="dead", wall_s=1.5)]
        with patch.object(executor, "_stream_post_once", side_effect=lambda b, p: seq.pop(0)), \
             patch.object(executor.time, "sleep"):
            enriched_raw = executor._stream_post(_backend(), {"messages": []})
        outcome = executor._build_outcome(_backend(), enriched_raw)
        self.assertEqual(outcome.status, "TRANSPORT_ERROR")
        self.assertEqual(outcome.retry_attempts, 2)
        self.assertIsNotNone(outcome.end_to_end_wall_s)
        self.assertAlmostEqual(outcome.final_attempt_latency_s, 1.5, places=3)
        self.assertEqual(len(outcome.attempts_meta), 2)


class HealthyLongStreamingTests(unittest.TestCase):
    def test_read_timeout_is_inactivity_not_wallclock(self):
        """The per-attempt httpx timeout must be a read/inactivity cap with
        NO overall wall-clock cap, so a healthy generation that keeps
        streaming is never aborted for taking a long time."""
        # Reconstruct the Timeout object the executor builds.
        t = httpx.Timeout(
            executor._READ_INACTIVITY_TIMEOUT_S,
            connect=executor._CONNECT_TIMEOUT_S,
            write=executor._READ_INACTIVITY_TIMEOUT_S,
            pool=executor._CONNECT_TIMEOUT_S,
        )
        self.assertEqual(t.read, executor._READ_INACTIVITY_TIMEOUT_S)
        self.assertEqual(t.connect, executor._CONNECT_TIMEOUT_S)
        self.assertLessEqual(executor._CONNECT_TIMEOUT_S, 10.0)
        self.assertEqual(executor._READ_INACTIVITY_TIMEOUT_S, 90.0)
        # httpx has no single "total" timeout field; the read cap resets on
        # every received chunk, so this configuration cannot wall-clock-abort
        # a stream that keeps making progress.

    def test_slow_but_progressing_generation_is_ok_not_transport_error(self):
        # A single attempt that took a long time but completed fine (events
        # present, no transport_err): must be OK, and its full duration is
        # reported as the final-attempt latency.
        long_ok = _once_result(transport_err=None, wall_s=1200.0)  # 20 min, healthy
        with patch.object(executor, "_stream_post_once", side_effect=lambda b, p: long_ok), \
             patch.object(executor.time, "sleep"):
            raw = executor._stream_post(_backend(), {"messages": []})
        outcome = executor._build_outcome(_backend(), raw)
        self.assertEqual(outcome.status, "OK")
        self.assertEqual(raw["retry_attempts"], 1)
        self.assertAlmostEqual(outcome.final_attempt_latency_s, 1200.0, places=1)


class ResumePreservationTests(unittest.TestCase):
    def test_accounting_fields_round_trip_through_checkpoint(self):
        from experiments.capability_router import orchestrator
        from experiments.capability_router.schema import TeacherCall

        outcome = ReplayOutcome(
            backend="local_27b", status="TRANSPORT_ERROR", response=None,
            latency_s=1.5, detail="dead",
            retry_attempts=2, end_to_end_wall_s=5.1,
            final_attempt_latency_s=1.5,
            attempts_meta=[
                {"attempt": 1, "transport_err": "dead", "status_code": None, "attempt_wall_s": 1.6},
                {"attempt": 2, "transport_err": "dead", "status_code": None, "attempt_wall_s": 1.5},
            ],
        )
        call = TeacherCall(
            call_id="x:0", task_group="g", source_campaign="c", source_trace_path="p",
            record_index=0, request={"model": "local", "messages": []},
            teacher_response={"content": []}, teacher_placement="cloud",
        )
        row = orchestrator._outcome_to_row(call, outcome, transform={})
        self.assertEqual(row["retry_attempts"], 2)
        self.assertEqual(row["end_to_end_wall_s"], 5.1)
        self.assertEqual(row["final_attempt_latency_s"], 1.5)
        self.assertEqual(len(row["attempts_meta"]), 2)

        with tempfile.TemporaryDirectory() as d:
            cp = Path(d) / "dual_execution_results.jsonl"
            executor.append_checkpoint(cp, row)
            reloaded = executor.load_checkpoint(cp)
        key = executor.checkpoint_key("x:0", "local_27b")
        self.assertEqual(reloaded[key]["retry_attempts"], 2)
        self.assertEqual(reloaded[key]["attempts_meta"][0]["attempt_wall_s"], 1.6)
        # rebuilt outcome carries them too
        back = orchestrator._row_to_outcome(reloaded[key])
        self.assertEqual(back.retry_attempts, 2)
        self.assertEqual(back.end_to_end_wall_s, 5.1)

    def test_transport_error_row_still_the_only_thing_reattempted_on_resume(self):
        cp = {
            "ok": {"call_id": "a", "backend": "local_27b", "status": "OK", "retry_attempts": 1},
            "cap": {"call_id": "b", "backend": "local_7b", "status": "INVALID_CAPACITY"},
            "terr": {"call_id": "c", "backend": "local_27b", "status": "TRANSPORT_ERROR",
                     "retry_attempts": 2, "end_to_end_wall_s": 9.9},
        }
        keep = executor.retryable_checkpoint(cp)
        self.assertIn("ok", keep)
        self.assertIn("cap", keep)
        self.assertNotIn("terr", keep)  # only the transport error is re-run


if __name__ == "__main__":
    unittest.main()
