"""Pinned, headless Pi Coder adapter for SWE-Gym capture runs.

The adapter only launches Pi inside an already-running ARM64 task container.
Container creation, grading, patch extraction, and experiment scheduling stay
with the caller. Event JSONL is flushed as it arrives so deadline/loop stops
retain partial trajectories.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PI_PACKAGE_NAME = "@earendil-works/pi-coding-agent"
PI_PACKAGE_VERSION = "0.86.1"
PI_PACKAGE_SPEC = f"{PI_PACKAGE_NAME}@{PI_PACKAGE_VERSION}"
PI_PACKAGE_INTEGRITY = (
    "sha512-vZBuNfJnruxZyemZ3O05V0S/Ylze08ahFTIQ1Mik++gVdOevPl89gt/"
    "Uv0U97BPAJaj9cj6Vf9rcIgKtUrd0BA=="
)

NODE_VERSION = "22.23.2"
NODE_ARCHIVE_URL = (
    f"https://nodejs.org/dist/v{NODE_VERSION}/"
    f"node-v{NODE_VERSION}-linux-arm64.tar.xz"
)
NODE_ARCHIVE_SHA256 = "fff4078c5def658577f92c88db7db3bc0072924bfb93fe52c1e744a54e94abb8"
NODE_INSTALL_DIR = f"/opt/node-v{NODE_VERSION}-linux-arm64"

MAX_IDENTICAL_ACTION_REPEATS = 4
PI_MAX_OUTPUT_TOKENS = 8192
PI_TIMEOUT_EXTRA_S = 0
PI_KILL_WAIT_S = 30
PI_READER_JOIN_S = 10
PI_PROVIDER_NAME = "flowmesh-edgeproxy"
PI_TOOL_NAMES = ("read", "write", "edit", "bash")

TerminationReason = Literal[
    "completed",
    "repeated_action",
    "step_limit",
    "trajectory_deadline",
    "process_error",
]


@dataclass(frozen=True)
class PiRunResult:
    stdout: str
    returncode: int | None
    termination_reason: TerminationReason
    action_count: int = 0
    model_turn_count: int = 0
    repeated_action_count: int = 0
    repeated_action_signature: str | None = None
    output_tokens: int | None = None
    stderr: str = ""

    @property
    def timed_out(self) -> bool:
        return self.termination_reason == "trajectory_deadline"

    def metadata(self) -> dict[str, object]:
        return {
            "pi_returncode": self.returncode,
            "pi_termination_reason": self.termination_reason,
            "pi_timed_out": self.timed_out,
            "pi_action_count": self.action_count,
            "pi_model_turn_count": self.model_turn_count,
            "pi_repeated_action_count": self.repeated_action_count,
            "pi_repeated_action_signature": self.repeated_action_signature,
            "pi_output_tokens": self.output_tokens,
        }


def _exec_in(container_name: str, script: str, *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", container_name, "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def setup_pi_container(container_name: str) -> None:
    """Install the checksum-pinned ARM64 Node runtime and Pi CLI in a task container."""
    archive_path = f"/tmp/node-v{NODE_VERSION}-linux-arm64.tar.xz"
    pi_archive_path = f"/tmp/pi-coding-agent-{PI_PACKAGE_VERSION}.tgz"
    npm_tarball_url = (
        "https://registry.npmjs.org/@earendil-works/pi-coding-agent/-/"
        f"pi-coding-agent-{PI_PACKAGE_VERSION}.tgz"
    )
    node_bin = f"{NODE_INSTALL_DIR}/bin"
    script = f"""set -euo pipefail
useradd -m -s /bin/bash agent 2>/dev/null || true
chown -R agent:agent /testbed
if ! command -v curl >/dev/null || ! command -v xz >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq curl ca-certificates xz-utils
fi
command -v sha256sum >/dev/null
trap 'rm -f {shlex.quote(archive_path)} {shlex.quote(pi_archive_path)}' EXIT
curl -fsSL {shlex.quote(NODE_ARCHIVE_URL)} -o {shlex.quote(archive_path)}
printf '%s  %s\\n' {shlex.quote(NODE_ARCHIVE_SHA256)} {shlex.quote(archive_path)} | sha256sum -c -
tar -xJf {shlex.quote(archive_path)} -C /opt --no-same-owner
export PATH={shlex.quote(node_bin)}:"$PATH"
node --version | grep -Fx {shlex.quote('v' + NODE_VERSION)} >/dev/null
curl -fsSL {shlex.quote(npm_tarball_url)} -o {shlex.quote(pi_archive_path)}
node -e 'const fs=require("fs"),crypto=require("crypto"),file=process.argv[1],expected=process.argv[2],actual="sha512-"+crypto.createHash("sha512").update(fs.readFileSync(file)).digest("base64"); if(actual!==expected){{console.error("Pi package integrity mismatch");process.exit(1)}}' {shlex.quote(pi_archive_path)} {shlex.quote(PI_PACKAGE_INTEGRITY)}
npm install --global --ignore-scripts --no-audit --no-fund {shlex.quote(pi_archive_path)}
pi --version | grep -Fx {shlex.quote(PI_PACKAGE_VERSION)} >/dev/null
"""
    try:
        proc = _exec_in(container_name, script, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Pi setup failed for {container_name}: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Pi setup failed for {container_name}: "
            f"{proc.stdout[-2000:]} {proc.stderr[-3000:]}"
        )


def _models_json(base_url: str, model: str, session_id: str = "$FLOWMESH_SESSION_ID") -> str:
    """Return a per-run custom provider config, never including the API token."""
    config = {
        "providers": {
            PI_PROVIDER_NAME: {
                "baseUrl": base_url,
                "api": "anthropic-messages",
                "apiKey": "$FLOWMESH_PI_API_KEY",
                "headers": {
                    "x-claude-code-session-id": session_id,
                },
                "models": [
                    {
                        "id": model,
                        "name": model,
                        "input": ["text"],
                        "contextWindow": 200000,
                        "maxTokens": PI_MAX_OUTPUT_TOKENS,
                        "reasoning": False,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    }
                ],
            }
        }
    }
    return json.dumps(config, ensure_ascii=False, separators=(",", ":"))


def _build_command(container_name: str, base_url: str, model: str) -> list[str]:
    # HOME is isolated so the custom provider cannot overwrite task/user
    # settings; --no-session prevents conversation/session persistence.
    models_config = shlex.quote(_models_json(base_url, model))
    node_bin = shlex.quote(NODE_INSTALL_DIR + "/bin")
    script = "\n".join(
        [
            "set -euo pipefail",
            "IFS= read -r ANTHROPIC_AUTH_TOKEN",
            "IFS= read -r FLOWMESH_SESSION_ID",
            'PROMPT="$(cat)"',
            'export FLOWMESH_PI_API_KEY="$ANTHROPIC_AUTH_TOKEN"',
            'export FLOWMESH_SESSION_ID="$FLOWMESH_SESSION_ID"',
            'export HOME="$(mktemp -d /tmp/pi-harness.XXXXXX)"',
            'mkdir -p "$HOME/.pi/agent"',
            f"printf '%s' {models_config} > \"$HOME/.pi/agent/models.json\"",
            f"{NODE_INSTALL_DIR}/bin/node -e 'const fs=require(\"fs\");const p=process.env.HOME+\"/.pi/agent/models.json\";const c=JSON.parse(fs.readFileSync(p,\"utf8\"));c.providers[\"{PI_PROVIDER_NAME}\"].headers[\"x-claude-code-session-id\"]=process.env.FLOWMESH_SESSION_ID;fs.writeFileSync(p,JSON.stringify(c,null,2));'",
            f"export PATH={node_bin}:\"$PATH\"",
            "exec pi --mode json --no-session "
            f"--provider {shlex.quote(PI_PROVIDER_NAME)} "
            f"--model {shlex.quote(model)} --thinking off "
            f"--tools {shlex.quote(','.join(PI_TOOL_NAMES))} \"$PROMPT\"",
        ]
    )
    return [
        "docker",
        "exec",
        "-i",
        "-u",
        "agent",
        "-w",
        "/testbed",
        "-e",
        "HOME=/home/agent",
        container_name,
        "bash",
        "-lc",
        script,
    ]


def _canonical_action(tool_name: object, args: object) -> str:
    return json.dumps(
        {"name": tool_name, "args": args},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def run_pi(
    container_name: str,
    prompt: str,
    base_url: str,
    model: str,
    session_id: str,
    timeout_s: int,
    out_path: Path,
    *,
    max_turns: int | None = None,
) -> PiRunResult:
    """Run one Pi trajectory and flush every JSON event to the output path.

    session_id is delivered via stdin (line 2) to prevent leaking to host argv,
    and propagated through Pi's models.json as a custom header so proxy traces
    can join to capture cells. Pi runs with --no-session and does not persist
    local conversation state.
    The API token is read from the host environment and delivered as the first
    stdin line; session_id is line 2; the prompt follows on line 3+ and none of
    these values is put in the Docker/Pi argument vector.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")
    try:
        api_token = os.environ["ANTHROPIC_AUTH_TOKEN"]
    except KeyError:
        return PiRunResult(
            stdout="",
            returncode=None,
            termination_reason="process_error",
            stderr="ANTHROPIC_AUTH_TOKEN is not set",
        )

    try:
        child_env = os.environ.copy()
        for secret_name in (
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
            "FLOWMESH_PI_API_KEY",
            "FLOWMESH_SESSION_ID",
        ):
            child_env.pop(secret_name, None)
        proc = subprocess.Popen(
            _build_command(container_name, base_url, model),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PiRunResult(
            stdout="",
            returncode=None,
            termination_reason="process_error",
            stderr=str(exc),
        )

    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    try:
        proc.stdin.write(api_token + "\n" + session_id + "\n" + prompt)
        proc.stdin.close()
    except (BrokenPipeError, OSError, ValueError) as exc:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=PI_KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            pass
        return PiRunResult(
            stdout="",
            returncode=proc.poll(),
            termination_reason="process_error",
            stderr=f"could not send token/prompt to Pi stdin: {exc}",
        )

    stdout_lines: list[str] = []
    stderr_tail: list[str] = []
    stderr_size = 0
    stop_reason: TerminationReason | None = None
    model_turn_count = 0
    action_count = 0
    repeated_action_count = 0
    repeated_action_signature: str | None = None
    output_token_total = 0
    output_tokens_seen = False

    def stop(reason: TerminationReason) -> None:
        nonlocal stop_reason
        if stop_reason is not None:
            return
        stop_reason = reason
        try:
            proc.kill()
        except OSError:
            pass

    def observe(line: str) -> None:
        nonlocal model_turn_count, action_count
        nonlocal repeated_action_count, repeated_action_signature
        nonlocal output_token_total, output_tokens_seen
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(event, dict):
            return
        if event.get("type") == "turn_start":
            if max_turns is not None and model_turn_count >= max_turns:
                stop("step_limit")
            else:
                model_turn_count += 1
            return
        if event.get("type") == "message_end":
            message = event.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                usage = message.get("usage")
                if isinstance(usage, dict):
                    tokens = usage.get("output")
                    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                        output_token_total += tokens
                        output_tokens_seen = True
            return
        if event.get("type") != "tool_execution_start":
            return
        action_count += 1
        signature = _canonical_action(event.get("toolName"), event.get("args"))
        if signature == repeated_action_signature:
            repeated_action_count += 1
        else:
            repeated_action_signature = signature
            repeated_action_count = 1
        if repeated_action_count >= MAX_IDENTICAL_ACTION_REPEATS:
            stop("repeated_action")

    def drain_stdout() -> None:
        try:
            with out_path.open("w", encoding="utf-8") as stream:
                for line in proc.stdout:
                    stdout_lines.append(line)
                    stream.write(line)
                    stream.flush()
                    observe(line)
        except OSError as exc:
            stderr_tail.append(f"failed to stream Pi events to {out_path}: {exc}\n")
            stop("process_error")

    def drain_stderr() -> None:
        nonlocal stderr_size
        for line in proc.stderr:
            stderr_tail.append(line)
            stderr_size += len(line)
            while stderr_size > 12000 and stderr_tail:
                removed = stderr_tail.pop(0)
                stderr_size -= len(removed)

    stdout_reader = threading.Thread(target=drain_stdout, daemon=True)
    stderr_reader = threading.Thread(target=drain_stderr, daemon=True)
    stdout_reader.start()
    stderr_reader.start()

    deadline = time.monotonic() + timeout_s + PI_TIMEOUT_EXTRA_S
    returncode: int | None = None
    termination_reason: TerminationReason
    while True:
        if stop_reason is not None:
            termination_reason = stop_reason
            try:
                returncode = proc.wait(timeout=PI_KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                returncode = None
            break
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            stop("trajectory_deadline")
            termination_reason = stop_reason or "trajectory_deadline"
            try:
                returncode = proc.wait(timeout=PI_KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                returncode = None
            break
        try:
            returncode = proc.wait(timeout=min(remaining_s, 0.25))
            # A final event may already be buffered while the stdout reader is
            # still parsing it, so let that reader finish before classifying.
            stdout_reader.join(timeout=PI_READER_JOIN_S)
            stderr_reader.join(timeout=PI_READER_JOIN_S)
            if stop_reason is not None:
                termination_reason = stop_reason
            elif returncode != 0:
                termination_reason = "process_error"
            else:
                termination_reason = "completed"
            break
        except subprocess.TimeoutExpired:
            continue

    if stdout_reader.is_alive():
        stdout_reader.join(timeout=PI_READER_JOIN_S)
    if stderr_reader.is_alive():
        stderr_reader.join(timeout=PI_READER_JOIN_S)
    return PiRunResult(
        stdout="".join(stdout_lines),
        returncode=returncode,
        termination_reason=termination_reason,
        action_count=action_count,
        model_turn_count=model_turn_count,
        repeated_action_count=repeated_action_count,
        repeated_action_signature=repeated_action_signature,
        output_tokens=output_token_total if output_tokens_seen else None,
        stderr="".join(stderr_tail),
    )
