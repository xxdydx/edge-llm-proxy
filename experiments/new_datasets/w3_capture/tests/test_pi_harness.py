"""Offline tests for the pinned Pi adapter; no Docker, install, or inference."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

W3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W3))
import pi_harness as m  # noqa: E402


class _Lines:
    def __init__(self, lines: list[str]):
        self.lines = list(lines)

    def __iter__(self):
        yield from self.lines


class _FakeStdin:
    def __init__(self):
        self.value = ""
        self.closed = False

    def write(self, value: str) -> None:
        self.value += value

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, stdout_lines: list[str], *, exit_after_s: float | None = None):
        self.stdin = _FakeStdin()
        self.stdout = _Lines(stdout_lines)
        self.stderr = _Lines([])
        self._exit_after_s = exit_after_s
        self._started = time.monotonic()
        self._killed = threading.Event()
        self.returncode: int | None = None
        self.kill_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self._killed.set()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = time.monotonic() + (timeout if timeout is not None else 3600)
        while True:
            if self._killed.is_set():
                self.returncode = -9
                return self.returncode
            if self._exit_after_s is not None and time.monotonic() - self._started >= self._exit_after_s:
                self.returncode = 0
                return self.returncode
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(cmd="fake-pi", timeout=timeout)
            time.sleep(0.002)


class _StubbornFakeProcess(_FakeProcess):
    def wait(self, timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired(cmd="stubborn-fake-pi", timeout=timeout)


def _event(event: dict) -> str:
    return json.dumps(event, separators=(",", ":")) + "\n"


def _run_fake(events: list[str], out_path: Path, *, exit_after_s: float | None = None):
    fake = _FakeProcess(events, exit_after_s=exit_after_s)
    captured: dict[str, object] = {}

    def popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["process"] = fake
        return fake

    with patch.dict("os.environ", {"ANTHROPIC_AUTH_TOKEN": "do-not-put-me-in-argv"}):
        with patch.object(m.subprocess, "Popen", side_effect=popen):
            result = m.run_pi(
                "fake-container",
                "sample task prompt",
                "http://host.docker.internal:18010",
                "local-model",
                "ignored-session",
                timeout_s=2,
                out_path=out_path,
            )
    return result, captured, fake


class PiHarnessTests(unittest.TestCase):
    def test_package_and_runtime_pins_are_exact(self):
        self.assertEqual(m.PI_PACKAGE_SPEC, "@earendil-works/pi-coding-agent@0.86.1")
        self.assertEqual(m.NODE_VERSION, "22.23.2")
        self.assertEqual(
            m.NODE_ARCHIVE_SHA256,
            "fff4078c5def658577f92c88db7db3bc0072924bfb93fe52c1e744a54e94abb8",
        )
        self.assertEqual(m.MAX_PI_MODEL_TURNS, 40)
        self.assertEqual(m.MAX_IDENTICAL_ACTION_REPEATS, 4)

    def test_setup_uses_checksum_verified_exact_assets(self):
        captured: dict[str, object] = {}

        def fake_exec(_container_name, script, *, timeout):
            captured["script"] = script
            captured["timeout"] = timeout
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with patch.object(m, "_exec_in", side_effect=fake_exec):
            m.setup_pi_container("test-container")
        script = str(captured["script"])
        self.assertIn(m.NODE_ARCHIVE_URL, script)
        self.assertIn(m.NODE_ARCHIVE_SHA256, script)
        self.assertIn(m.PI_PACKAGE_VERSION, script)
        self.assertIn(m.PI_PACKAGE_INTEGRITY, script)
        self.assertIn("--ignore-scripts", script)
        self.assertIn("useradd -m -s /bin/bash agent", script)
        self.assertIn("chown -R agent:agent /testbed", script)
        self.assertIn("xz-utils", script)
        self.assertNotIn("latest", script)
        self.assertEqual(captured["timeout"], 600)

    def test_command_uses_json_mode_no_session_four_tools_and_stdin_prompt(self):
        event_lines = [
            _event({"type": "agent_start"}),
            _event({"type": "turn_start"}),
            _event({
                "type": "message_end",
                "message": {"role": "assistant", "usage": {"input": 10, "output": 17}, "content": []},
            }),
            _event({
                "type": "turn_end",
                "message": {"role": "assistant", "content": []},
            }),
            _event({"type": "agent_end", "messages": []}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "pi.jsonl"
            result, captured, fake = _run_fake(event_lines, out_path, exit_after_s=0)
            args = captured["args"]
            self.assertIsInstance(args, list)
            self.assertNotIn("do-not-put-me-in-argv", args)
            self.assertNotIn("sample task prompt", args)
            popen_env = captured["kwargs"]["env"]
            self.assertNotIn("ANTHROPIC_AUTH_TOKEN", popen_env)
            self.assertNotIn("ANTHROPIC_API_KEY", popen_env)
            script = str(args[-1])
            self.assertIn("--mode json --no-session", script)
            self.assertIn("--thinking off", script)
            self.assertIn("--tools read,write,edit,bash", script)
            self.assertIn('PROMPT="$(cat)"', script)
            self.assertIn('"$PROMPT"', script)
            self.assertIn("http://host.docker.internal:18010", script)
            self.assertIn("$FLOWMESH_PI_API_KEY", script)
            self.assertNotIn("do-not-put-me-in-argv", script)
            self.assertEqual(fake.stdin.value, "do-not-put-me-in-argv\nsample task prompt")
            self.assertTrue(fake.stdin.closed)
            self.assertEqual(result.termination_reason, "completed")
            self.assertEqual(result.model_turn_count, 1)
            self.assertEqual(result.output_tokens, 17)
            self.assertEqual(out_path.read_text(encoding="utf-8"), "".join(event_lines))

    def test_repeated_canonical_action_stops_after_four_and_keeps_stream(self):
        events = [
            _event({"type": "tool_execution_start", "toolName": "bash", "args": {"command": "pwd", "timeout": 5}}),
            _event({"type": "tool_execution_start", "toolName": "bash", "args": {"timeout": 5, "command": "pwd"}}),
            _event({"type": "tool_execution_start", "toolName": "bash", "args": {"command": "pwd", "timeout": 5}}),
            _event({"type": "tool_execution_start", "toolName": "bash", "args": {"command": "pwd", "timeout": 5}}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "repeat.jsonl"
            result, _captured, fake = _run_fake(events, out_path)
            self.assertEqual(result.termination_reason, "repeated_action")
            self.assertEqual(result.action_count, 4)
            self.assertEqual(result.repeated_action_count, 4)
            self.assertIsNotNone(result.repeated_action_signature)
            self.assertGreaterEqual(fake.kill_calls, 1)
            self.assertEqual(out_path.read_text(encoding="utf-8"), "".join(events))

    def test_step_limit_allows_forty_turns_and_blocks_forty_first(self):
        events = [_event({"type": "turn_start"}) for _ in range(m.MAX_PI_MODEL_TURNS + 1)]
        with tempfile.TemporaryDirectory() as tmp:
            result, _captured, fake = _run_fake(events, Path(tmp) / "steps.jsonl")
            self.assertEqual(result.termination_reason, "step_limit")
            self.assertEqual(result.model_turn_count, m.MAX_PI_MODEL_TURNS)
            self.assertGreaterEqual(fake.kill_calls, 1)

    def test_trajectory_deadline_is_classified_and_process_error_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeProcess([], exit_after_s=None)
            with patch.dict("os.environ", {"ANTHROPIC_AUTH_TOKEN": "token"}):
                with patch.object(m, "PI_TIMEOUT_EXTRA_S", 0):
                    with patch.object(m.subprocess, "Popen", return_value=fake):
                        deadline = m.run_pi(
                            "fake-container", "prompt", "http://edge", "model", "sid",
                            timeout_s=0, out_path=Path(tmp) / "deadline.jsonl",
                        )
            self.assertEqual(deadline.termination_reason, "trajectory_deadline")
            self.assertTrue(deadline.timed_out)
            self.assertEqual(fake.kill_calls, 1)

            with patch.dict("os.environ", {"ANTHROPIC_AUTH_TOKEN": "token"}):
                with patch.object(m.subprocess, "Popen", side_effect=FileNotFoundError("docker missing")):
                    error = m.run_pi(
                        "fake-container", "prompt", "http://edge", "model", "sid",
                        timeout_s=1, out_path=Path(tmp) / "error.jsonl",
                    )
            self.assertEqual(error.termination_reason, "process_error")
            self.assertIn("docker missing", error.stderr)

    def test_kill_teardown_is_bounded_when_docker_exec_does_not_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _StubbornFakeProcess([])
            started = time.monotonic()
            with patch.dict("os.environ", {"ANTHROPIC_AUTH_TOKEN": "token"}):
                with patch.object(m, "PI_TIMEOUT_EXTRA_S", 0):
                    with patch.object(m, "PI_KILL_WAIT_S", 0.02):
                        with patch.object(m, "PI_READER_JOIN_S", 0.02):
                            with patch.object(m.subprocess, "Popen", return_value=fake):
                                result = m.run_pi(
                                    "fake-container", "prompt", "http://edge", "model", "sid",
                                    timeout_s=0, out_path=Path(tmp) / "stubborn.jsonl",
                                )
            elapsed = time.monotonic() - started
            self.assertEqual(result.termination_reason, "trajectory_deadline")
            self.assertIsNone(result.returncode)
            self.assertEqual(fake.kill_calls, 1)
            self.assertLess(elapsed, 1)


if __name__ == "__main__":
    unittest.main()
