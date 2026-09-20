"""Real edge-only/cloud-only Claude Code capture for the 2 frozen smoke tasks.

Reuses the official SWE-Bench-Fork harness (already vendored under
experiments/new_datasets/w2_preflight/_vendor/SWE-Bench-Fork, already used
successfully for real preflight of these two exact instances) for
container build/grading, and the project's existing render_prompt/
container-launch conventions from eval-suite/swebench/runner/run_swebench.py
(read, not copied verbatim, because that script hardcodes --platform
linux/amd64 for its own historical jefzda/sweap-images pipeline, which is
wrong for these natively-arm64 SWE-Gym images).

No LLM other than the actual Claude Code harness talking through the
isolated edgeproxy instances is invoked. No gold/reference patch is ever
applied. This produces one real container-executed trajectory per
(instance, policy) pair.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/Users/arul/Desktop/flowmesh")
ND = REPO_ROOT / "experiments/new_datasets"
VENDOR = ND / "w2_preflight/_vendor/SWE-Bench-Fork"
OUT_ROOT = ND / "w3_capture"

sys.path.insert(0, str(VENDOR))
from swebench.harness.test_spec import make_test_spec  # noqa: E402
from swebench.harness.docker_build import build_env_images  # noqa: E402
from swebench.harness.run_evaluation import run_instance  # noqa: E402
from swebench.harness.utils import load_swebench_dataset  # noqa: E402
import docker  # noqa: E402

INSTANCES_FILE = ND / "w2_preflight/smoke_instances.json"

POLICIES = {
    "edge-only-v1": {"base_url": "http://host.docker.internal:18010", "model": "local"},
    "cloud-only-v1": {"base_url": "http://host.docker.internal:18011", "model": "deepseek-v4-flash"},
}

MAX_ACTIVE_SECONDS = 1200
CLAUDE_TIMEOUT_EXTRA_S = 60


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def render_prompt(instance: dict) -> str:
    return (
        f"Your working directory is /testbed, a checkout of the "
        f"{instance['repo']} repository. Fix the following issue:\n\n"
        f"{instance['problem_statement']}\n\n"
        "Investigate the relevant source file(s) yourself and implement a fix. "
        "You may run the repository's own tests with pytest to check your work. "
        "Do not modify any test file. Do not run pip install or access the network."
    )


def container_name_for(instance_id: str, policy: str) -> str:
    return f"smoke-capture-{instance_id.replace('/', '_').replace('.', '_')}-{policy}".lower()


def start_container(image: str, name: str) -> None:
    sh(["docker", "rm", "-f", name])
    proc = sh([
        "docker", "run", "-d", "--name", name,
        "--platform", "linux/arm64/v8",
        "--add-host=host.docker.internal:host-gateway",
        "--entrypoint", "bash",
        image, "-c", "sleep infinity",
    ], timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed: {proc.stderr[-2000:]}")


def exec_in(name: str, script: str, user: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    cmd = ["docker", "exec"]
    if user:
        cmd += ["-u", user, "-e", "HOME=/home/" + user]
    cmd += [name, "bash", "-c", script]
    return sh(cmd, timeout=timeout)


def setup_container(name: str) -> None:
    install = exec_in(
        name,
        "useradd -m -s /bin/bash agent 2>/dev/null; "
        "chown -R agent:agent /testbed; "
        "which curl >/dev/null 2>&1 || apt-get update -qq && apt-get install -y -qq curl ca-certificates; "
        "v=$(curl -fsSL https://downloads.claude.ai/claude-code-releases/latest) && "
        "curl -fsSL -o /usr/local/bin/claude "
        '"https://downloads.claude.ai/claude-code-releases/$v/linux-arm64/claude" && '
        "chmod a+rx /usr/local/bin/claude",
        timeout=300,
    )
    if install.returncode != 0:
        raise RuntimeError(f"claude install failed: {install.stdout[-3000:]} {install.stderr[-3000:]}")


def run_claude(name: str, prompt: str, base_url: str, model: str, session_id: str, timeout_s: int,
               out_path: Path) -> tuple[str, int | None, bool]:
    # The token is supplied through stdin to a shell variable, never in argv
    # -- an argv value is visible to any local `ps aux`, and (as happened
    # once) gets printed verbatim into any TimeoutExpired traceback.
    user_msg_line = json.dumps({
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
        "parent_tool_use_id": None,
    }, ensure_ascii=False) + "\n"
    env_flags = ["-e", f"ANTHROPIC_BASE_URL={base_url}", "-e", "HOME=/home/agent"]
    cmd = [
        "docker", "exec", "-i", "-u", "agent", "-w", "/testbed", *env_flags, name,
        "bash", "-lc",
        "read -r ANTHROPIC_AUTH_TOKEN; export ANTHROPIC_AUTH_TOKEN; exec claude -p "
        f"--model {shlex.quote(model)} --session-id {shlex.quote(session_id)} "
        "--dangerously-skip-permissions "
        "--input-format stream-json --output-format stream-json "
        "--replay-user-messages --verbose",
    ]
    stdin_data = os.environ["ANTHROPIC_AUTH_TOKEN"] + "\n" + user_msg_line
    hard_timeout = timeout_s + CLAUDE_TIMEOUT_EXTRA_S

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc.stdin.write(stdin_data)
    proc.stdin.close()

    # A real trajectory can run 20+ real turns before hitting the wall-clock
    # budget -- that's expensive, genuine progress, not a stuck/empty run.
    # communicate()'s in-memory buffering loses all of it if the process has
    # to be killed and the docker-exec client itself then hangs past its own
    # 30s teardown window (confirmed live: this happens on real, reproducible
    # tasks). Stream to disk line by line as it arrives instead, so whatever
    # was actually produced survives regardless of how the kill goes.
    lines: list[str] = []
    stderr_chunks: list[str] = []

    def _drain_stdout() -> None:
        with out_path.open("w") as f:
            for line in proc.stdout:
                lines.append(line)
                f.write(line)
                f.flush()

    def _drain_stderr() -> None:
        # A zombied docker-exec client (confirmed to happen: it can survive
        # SIGKILL for well past 30s) never closes this pipe either, so an
        # unbounded proc.stderr.read() here can hang indefinitely and defeat
        # every timeout above it. Drain it the same bounded, threaded way.
        for line in proc.stderr:
            stderr_chunks.append(line)

    stdout_reader = threading.Thread(target=_drain_stdout, daemon=True)
    stderr_reader = threading.Thread(target=_drain_stderr, daemon=True)
    stdout_reader.start()
    stderr_reader.start()

    timed_out = False
    try:
        returncode = proc.wait(timeout=hard_timeout)
        stdout_reader.join(timeout=10)
        stderr_reader.join(timeout=10)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        try:
            returncode = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # The docker-exec client can outlive the killed in-container
            # process if the daemon is slow to tear it down; don't let a
            # second timeout crash the whole capture run over one task.
            # Whatever was captured up to this point is already on disk.
            returncode = None
        stdout_reader.join(timeout=10)
        stderr_reader.join(timeout=10)

    stderr = "".join(stderr_chunks)
    if stderr:
        sys.stderr.write(f"[claude stderr for {name}]\n{stderr[-3000:]}\n")
    return "".join(lines), returncode, timed_out


def extract_patch(name: str) -> str:
    # Must run as the "agent" user: /testbed is chown'd to agent:agent, and
    # git's dubious-ownership check makes `git diff` as root silently return
    # empty (not an error we'd notice) instead of the real diff.
    r = exec_in(name, "cd /testbed && git diff", user="agent", timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"git diff failed in {name}: rc={r.returncode} stdout={r.stdout[-1000:]} stderr={r.stderr[-1000:]}")
    return r.stdout


def main() -> None:
    if "ANTHROPIC_AUTH_TOKEN" not in os.environ:
        # load_dotenv_into_environ equivalent: minimal .env loader for this one key
        env_path = REPO_ROOT / ".env"
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_AUTH_TOKEN="):
                os.environ["ANTHROPIC_AUTH_TOKEN"] = line.split("=", 1)[1].strip()
                break
    assert "ANTHROPIC_AUTH_TOKEN" in os.environ, "missing ANTHROPIC_AUTH_TOKEN"

    dataset = load_swebench_dataset(str(INSTANCES_FILE))
    client = docker.from_env()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []

    for instance in dataset:
        iid = instance["instance_id"]
        test_spec = make_test_spec(instance)
        image = f"sweb.eval.{test_spec.arch}.{iid}:latest"

        run_seed = int.from_bytes(hashlib.sha256(f"smoke-v1|order|{iid}|{time.time_ns()}".encode()).digest()[:8], "big")
        order = list(POLICIES.items())
        random.Random(run_seed).shuffle(order)
        for policy, cfg in order:
            repeat_index = 0
            while (OUT_ROOT / f"{iid}__{policy}" if repeat_index == 0
                   else OUT_ROOT / f"{iid}__{policy}__r{repeat_index}").exists():
                repeat_index += 1
            run_dir = (OUT_ROOT / f"{iid}__{policy}" if repeat_index == 0
                       else OUT_ROOT / f"{iid}__{policy}__r{repeat_index}")
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"=== {iid} / {policy} ===", flush=True)

            name = container_name_for(iid, policy)
            t_container_start = time.time()
            start_container(image, name)
            setup_container(name)
            t_setup_done = time.time()

            session_id = str(uuid.uuid4())
            prompt = render_prompt(instance)
            (run_dir / "prompt.txt").write_text(prompt)

            t_claude_start = time.time()
            stdout, returncode, timed_out = run_claude(
                name, prompt, cfg["base_url"], cfg["model"], session_id, MAX_ACTIVE_SECONDS,
                run_dir / "claude_stream.jsonl",
            )
            t_claude_end = time.time()

            patch = extract_patch(name)
            (run_dir / "final_patch.diff").write_text(patch)

            sh(["docker", "rm", "-f", name])

            record = {
                "instance_id": iid,
                "policy": policy,
                "run_dir": run_dir.name,
                "repeat_index": repeat_index,
                "session_id": session_id,
                "execution_order": {"seed": run_seed, "policies_in_order": [p for p, _ in order]},
                "container_setup_seconds": round(t_setup_done - t_container_start, 2),
                "claude_wall_seconds": round(t_claude_end - t_claude_start, 2),
                "claude_returncode": returncode,
                "claude_timed_out": timed_out,
                "patch_nonempty": bool(patch.strip()),
                "patch_bytes": len(patch),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            (run_dir / "run_meta.json").write_text(json.dumps(record, indent=1))
            manifest.append(record)
            print(json.dumps(record, indent=1), flush=True)

    manifest_path = OUT_ROOT / "capture_manifest.json"
    existing_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    existing_run_dirs = {r.get("run_dir") for r in existing_manifest}
    combined = existing_manifest + [r for r in manifest if r["run_dir"] not in existing_run_dirs]
    manifest_path.write_text(json.dumps(combined, indent=1))
    print("CAPTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
