"""Offline tests for run_smoke_capture.run_claude()'s stdout/stderr
capture -- specifically that partial output survives a kill, and that
the whole function is bounded even if the underlying process never
truly exits (the real, reproduced failure mode: a docker-exec client
that outlives SIGKILL). No docker or claude CLI involved -- a fake
subprocess.Popen stands in.
"""
from __future__ import annotations

import subprocess
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

W3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W3))
import run_smoke_capture as m  # noqa: E402


class _Line:
    """A stdout/stderr stand-in: iterating yields lines, optionally
    pausing to simulate a slow/streaming or never-ending process."""

    def __init__(self, lines: list[str], hang_forever: bool = False):
        self._lines = list(lines)
        self._hang_forever = hang_forever

    def __iter__(self):
        for line in self._lines:
            yield line
        if self._hang_forever:
            # A pipe whose writer never closes it (the zombie scenario):
            # iteration just never ends. The reader thread's bounded
            # .join() in run_claude() is what must save us here, not this.
            while True:
                time.sleep(3600)


class _FakeStdin:
    def write(self, _data):
        pass

    def close(self):
        pass


class _FakeProc:
    """A subprocess.Popen stand-in: real Popen semantics that matter
    here (wait()/kill()/returncode), fake everything else."""

    def __init__(self, stdout_lines, *, hang_forever_stdout=False,
                 exit_after_s: float | None = 0.05, survives_kill_for_s: float = 0.0):
        self.stdin = _FakeStdin()
        self.stdout = _Line(stdout_lines, hang_forever=hang_forever_stdout)
        self.stderr = _Line([])
        self._exit_after_s = exit_after_s
        self._survives_kill_for_s = survives_kill_for_s
        self._killed_at: float | None = None
        self._start = time.monotonic()
        self.returncode: int | None = None

    def kill(self):
        self._killed_at = time.monotonic()

    def wait(self, timeout=None):
        deadline = time.monotonic() + (timeout if timeout is not None else 10**9)
        while True:
            now = time.monotonic()
            if self._killed_at is not None:
                if now - self._killed_at >= self._survives_kill_for_s:
                    self.returncode = -9
                    return self.returncode
            elif self._exit_after_s is not None and now - self._start >= self._exit_after_s:
                self.returncode = 0
                return self.returncode
            if now >= deadline:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            time.sleep(0.02)


class RunClaudeCaptureTests(unittest.TestCase):
    def test_normal_completion_writes_full_output_and_no_timeout(self):
        lines = [f"line{i}\n" for i in range(5)]
        with patch.object(m.subprocess, "Popen", return_value=_FakeProc(lines)):
            out_path = Path("/tmp/test_run_claude_normal.jsonl")
            stdout, returncode, timed_out = m.run_claude(
                "fake-container", "prompt", "http://x", "local", "sid", timeout_s=10, out_path=out_path
            )
        self.assertEqual(stdout, "".join(lines))
        self.assertEqual(out_path.read_text(), "".join(lines))
        self.assertFalse(timed_out)
        self.assertEqual(returncode, 0)

    def test_records_max_token_responses(self):
        event = json.dumps({
            "type": "assistant",
            "message": {"stop_reason": "max_tokens", "content": [{"type": "text", "text": "x"}]},
        }) + "\n"
        with patch.object(m.subprocess, "Popen", return_value=_FakeProc([event])):
            result = m.run_claude(
                "fake-container", "prompt", "http://x", "local", "sid", timeout_s=10,
                out_path=Path("/tmp/test_run_claude_max_tokens.jsonl"),
            )
        self.assertEqual(result.max_token_responses, 1)
        self.assertEqual(result.termination_reason, "completed")

    def test_repeated_identical_action_is_stopped_and_classified(self):
        event = json.dumps({
            "type": "assistant",
            "message": {
                # Real Claude Code stream-json tool events have null here;
                # the tool block, not stop_reason, identifies the action.
                "stop_reason": None,
                "content": [{"type": "tool_use", "id": "ignored", "name": "Bash", "input": {"command": "grep x y"}}],
            },
        }) + "\n"
        fake = _FakeProc([event] * m.MAX_IDENTICAL_ACTION_REPEATS, exit_after_s=None)
        with patch.object(m.subprocess, "Popen", return_value=fake):
            result = m.run_claude(
                "fake-container", "prompt", "http://x", "local", "sid", timeout_s=10,
                out_path=Path("/tmp/test_run_claude_repeated_action.jsonl"),
            )
        self.assertEqual(result.termination_reason, "repeated_action")
        self.assertFalse(result.timed_out)
        self.assertEqual(result.repeated_action_count, m.MAX_IDENTICAL_ACTION_REPEATS)
        self.assertEqual(result.returncode, -9)

    def test_partial_output_survives_a_clean_kill(self):
        lines = [f"line{i}\n" for i in range(200)]
        fake = _FakeProc(lines, exit_after_s=None, survives_kill_for_s=0.05)
        with patch.object(m, "CLAUDE_TIMEOUT_EXTRA_S", 0), patch.object(m.subprocess, "Popen", return_value=fake):
            out_path = Path("/tmp/test_run_claude_partial.jsonl")
            stdout, returncode, timed_out = m.run_claude(
                "fake-container", "prompt", "http://x", "local", "sid", timeout_s=0, out_path=out_path
            )
        self.assertTrue(timed_out)
        self.assertGreater(len(stdout), 0)
        self.assertEqual(out_path.read_text(), stdout)

    def test_zombie_process_that_never_exits_after_kill_still_returns_bounded(self):
        # The real, reproduced failure mode: SIGKILL sent, but the
        # process (a docker-exec client) never actually dies, and its
        # pipes never close. The whole function must still return in
        # bounded time with whatever was captured before the kill.
        lines = [f"line{i}\n" for i in range(50)]
        fake = _FakeProc(lines, hang_forever_stdout=False, exit_after_s=None, survives_kill_for_s=10**9)
        with patch.object(m, "CLAUDE_TIMEOUT_EXTRA_S", 0), patch.object(m.subprocess, "Popen", return_value=fake):
            out_path = Path("/tmp/test_run_claude_zombie.jsonl")
            t0 = time.monotonic()
            stdout, returncode, timed_out = m.run_claude(
                "fake-container", "prompt", "http://x", "local", "sid", timeout_s=0, out_path=out_path
            )
            elapsed = time.monotonic() - t0
        self.assertTrue(timed_out)
        self.assertIsNone(returncode)
        self.assertGreater(len(stdout), 0)
        self.assertEqual(out_path.read_text(), stdout)
        # Bounded well under real production timeouts (kill-wait(30) +
        # two 10s reader joins is the real budget) -- this fake uses a
        # 0s primary timeout so the whole thing should be a few seconds.
        self.assertLess(elapsed, 60)


if __name__ == "__main__":
    unittest.main()
