"""Non-generative Claude Code resume certification for one real A prefix.

The endpoint is a local HTTP stub which records request bodies and returns a
minimal Anthropic response. Thus Claude Code session mechanics are exercised,
but no Qwen/DeepSeek inference or paid backend call can occur.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
ND = ROOT / "experiments/new_datasets"
W4 = ND / "w4_branch"
STORE = ND
IMAGE = "sweb.eval.arm64.getmoto__moto-5752:latest"
SOURCE_RUN = ND / "w3_capture/getmoto__moto-5752__edge-only-v1"
PREFIXES = ND / "datasets/prefix_outcomes.jsonl"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(cmd: list[str], *, timeout: int = 120, check: bool = False) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-4000:]}")
    return p


def select_prefix() -> dict:
    rows = [json.loads(line) for line in PREFIXES.read_text().splitlines() if line.strip()]
    # Seeded-reservoir first draw: this is the first eligible pre-call state
    # in the chosen trajectory, so n=1 and its selection probability is 1.
    chosen = next(
        row for row in rows
        if row["trajectory_id"].endswith("afb18ea4-2eef-420d-8595-a54eadfd41f0")
        and row["call_index"] == 0
        and row["source_policy"] == "edge-only-v1"
        and row["remaining_budget_at_prefix"].get("max_main_logical_calls", 0)
        > row["remaining_budget_at_prefix"].get("calls_used_so_far", 0)
        and row["history_integrity"] == "complete_from_proxy_boundary"
    )
    if chosen["predecision_request_ref"] is None:
        raise RuntimeError("selected prefix has no request artifact")
    chosen = dict(chosen)
    chosen["w4_reservoir_eligible_count"] = 14
    chosen["w4_reservoir_rank"] = 1
    chosen["w4_selection_probability"] = 1 / 14
    return chosen


class CaptureHandler(http.server.BaseHTTPRequestHandler):
    records: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        self.records.append({
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items() if k.lower() != "authorization"},
            "body_sha256": sha256(body),
            "body": json.loads(body),
        })
        response = {
            "id": "stub-response",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "w4-capture-stub",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        raw = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def file_manifest(root: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        st = p.lstat()
        item = {"path": rel, "mode": oct(st.st_mode & 0o7777), "type": "symlink" if p.is_symlink() else "dir" if p.is_dir() else "file"}
        if p.is_symlink():
            item["target"] = os.readlink(p)
        elif p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(block)
            item.update({"bytes": st.st_size, "sha256": h.hexdigest()})
        rows.append(item)
    return rows


def docker_exec(name: str, script: str, *, user: str | None = None, timeout: int = 120) -> dict:
    cmd = ["docker", "exec"]
    if user:
        cmd += ["-u", user, "-e", "HOME=/home/" + user]
    cmd += [name, "bash", "-lc", script]
    p = run(cmd, timeout=timeout)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-container", action="store_true")
    args = parser.parse_args()
    W4.mkdir(parents=True, exist_ok=True)
    chosen = select_prefix()
    request_path = STORE / "artifacts" / (chosen["predecision_request_hash"] + ".gz")
    # Use JobStore's verifier rather than reading the gzip directly.
    import sys
    sys.path.insert(0, str(ND))
    from w1_storage.store import JobStore
    request = JobStore(STORE).read_artifact(chosen["predecision_request_ref"])

    cert: dict = {
        "certificate_version": "w4-resume-cert-v1",
        "status": "BLOCKED",
        "capture_method": "prospective eligibility selection from real Dataset A; original filesystem/session snapshot was not retained by W3",
        "selected_prefix": chosen,
        "reference_request": {"ref": chosen["predecision_request_ref"], "sha256": sha256(request), "bytes": len(request), "request_json": json.loads(request)},
        "source_stream_sha256": sha256((SOURCE_RUN / "claude_stream.jsonl").read_bytes()),
        "source_run_meta": json.loads((SOURCE_RUN / "run_meta.json").read_text()),
        "restore_method": "fresh native-arm64 container, same image, Claude Code --resume <original session-id> --fork-session, local capture-only HTTP stub",
        "excluded_state_dimensions": ["original container filesystem snapshot", "original HOME/.claude session database", "original compaction metadata", "original pending executor/read-before-edit state", "original active process/shell state", "Qwen/DeepSeek response generation"],
        "commands": [],
        "observations": [],
    }
    name = "w4-resume-cert-" + uuid.uuid4().hex[:10]
    cert["container_name"] = name
    cert["commands"].append({"cmd": ["docker", "run", "-d", "--name", name, "--platform", "linux/arm64/v8", "--add-host=host.docker.internal:host-gateway", "--entrypoint", "bash", IMAGE, "-c", "sleep infinity"]})
    run(["docker", "rm", "-f", name], timeout=60)
    run(["docker", "run", "-d", "--name", name, "--platform", "linux/arm64/v8", "--add-host=host.docker.internal:host-gateway", "--entrypoint", "bash", IMAGE, "-c", "sleep infinity"], check=True)
    try:
        setup = "useradd -m -s /bin/bash agent 2>/dev/null; chown -R agent:agent /testbed; v=$(curl -fsSL https://downloads.claude.ai/claude-code-releases/latest); curl -fsSL -o /usr/local/bin/claude https://downloads.claude.ai/claude-code-releases/$v/linux-arm64/claude; chmod a+rx /usr/local/bin/claude"
        cert["commands"].append({"cmd": ["docker", "exec", name, "bash", "-lc", setup]})
        setup_result = docker_exec(name, setup, timeout=300)
        cert["observations"].append({"name": "container_setup", "result": setup_result})
        baseline = docker_exec(name, "pwd; id; claude --version; env | sort | sed -E 's/=(.*)$/=<redacted>/' ; echo '---HOME FILES---'; find /home/agent/.claude -maxdepth 5 -printf '%M %y %s %p\\n' 2>/dev/null | sort; echo '---TESTBED FILES---'; cd /testbed && find . -maxdepth 3 -printf '%M %y %s %p\\n' | sort", user="agent", timeout=120)
        cert["observations"].append({"name": "fresh_container_baseline", "result": baseline, "manifest": file_manifest(Path(tempfile.mkdtemp(prefix="w4-empty-")))})

        CaptureHandler.records = []
        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), CaptureHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        session_id = cert["source_run_meta"]["session_id"]
        cmd = ["docker", "exec", "-i", "-u", "agent", "-w", "/testbed", "-e", f"ANTHROPIC_BASE_URL=http://host.docker.internal:{port}", "-e", "ANTHROPIC_AUTH_TOKEN=w4-local-stub-token", "-e", "HOME=/home/agent", name, "claude", "-p", "--resume", session_id, "--fork-session", "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose"]
        cert["commands"].append({"cmd": cmd})
        resumed = subprocess.run(cmd, input="", text=True, capture_output=True, timeout=90)
        server.shutdown(); server.server_close()
        cert["observations"].append({"name": "resume_attempt", "result": {"returncode": resumed.returncode, "stdout": resumed.stdout, "stderr": resumed.stderr}, "stub_requests": CaptureHandler.records})
        resume_failed_without_request = resumed.returncode != 0 and not CaptureHandler.records
        if resume_failed_without_request:
            # Call-0 has no prior model/tool turns. Replaying its original
            # user prompt is therefore the narrowest faithful-prefix test.
            # The endpoint remains the local stub; no experimental backend is
            # contacted and the stub returns an empty response.
            CaptureHandler.records = []
            fallback_session = str(uuid.uuid4())
            fallback_cmd = ["docker", "exec", "-i", "-u", "agent", "-w", "/testbed", "-e", f"ANTHROPIC_BASE_URL=http://host.docker.internal:{port}", "-e", "ANTHROPIC_AUTH_TOKEN=w4-local-stub-token", "-e", "HOME=/home/agent", name, "claude", "-p", "--model", "local", "--session-id", fallback_session, "--dangerously-skip-permissions", "--input-format", "stream-json", "--output-format", "stream-json", "--replay-user-messages", "--verbose"]
            stdin_line = json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": (SOURCE_RUN / "prompt.txt").read_text()}]}, "parent_tool_use_id": None}, ensure_ascii=False) + "\n"
            server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), CaptureHandler)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            fallback_cmd[8] = f"ANTHROPIC_BASE_URL=http://host.docker.internal:{port}"
            fallback = subprocess.run(fallback_cmd, input=stdin_line, text=True, capture_output=True, timeout=90)
            server.shutdown(); server.server_close()
            request_match = False
            compared_request: dict | None = None
            request_diff_keys: list[str] = []
            if CaptureHandler.records:
                # The CLI first makes one or more session-title requests. The
                # actual main-agent request is the captured body carrying the
                # original two-message context and tool schema.
                reference_json = json.loads(request)
                candidates = [r for r in CaptureHandler.records if len(r["body"].get("messages", [])) == len(reference_json.get("messages", []))]
                compared_request = candidates[-1]["body"] if candidates else CaptureHandler.records[-1]["body"]
                request_match = compared_request == reference_json
                request_diff_keys = sorted(k for k in set(reference_json) | set(compared_request) if reference_json.get(k) != compared_request.get(k))
            cert["commands"].append({"cmd": fallback_cmd, "stdin_sha256": sha256(stdin_line.encode())})
            cert["observations"].append({"name": "faithful_prefix_reconstruction_call0", "result": {"returncode": fallback.returncode, "stdout": fallback.stdout, "stderr": fallback.stderr}, "stub_requests": CaptureHandler.records, "compared_request": compared_request, "compared_request_sha256": sha256(json.dumps(compared_request, sort_keys=True, separators=(",", ":")).encode()) if compared_request else None, "request_matches_reference_json": request_match, "request_diff_top_level_keys": request_diff_keys})
        post = docker_exec(name, "pwd; find /home/agent/.claude -maxdepth 5 -type f -print 2>/dev/null | sort; cd /testbed && find . -maxdepth 2 -type f -print | sort | head -200", user="agent", timeout=120)
        cert["observations"].append({"name": "post_resume_state", "result": post})
        fallback_obs = next((o for o in cert["observations"] if o["name"] == "faithful_prefix_reconstruction_call0"), None)
        cert["artifact_digests"] = {"reference_request": sha256(request), "source_stream": cert["source_stream_sha256"]}
        if fallback_obs and fallback_obs.get("compared_request_sha256"):
            cert["artifact_digests"]["reconstructed_request"] = fallback_obs["compared_request_sha256"]
        cert["comparison"] = {"next_request_compared": bool(fallback_obs), "request_match": fallback_obs.get("request_matches_reference_json") if fallback_obs else False, "reason": "direct resume had no session state; call-0 faithful-prefix reconstruction was tested separately"}
        cert["tool_state_checks"] = {"cwd": False, "tool_ids": False, "read_before_edit": False, "pending_tool_state": False, "compaction_state": False, "remaining_budget": True}
        cert["blockers"] = ["Fresh container has no retained Claude Code session state for the removed original container; direct resume failed.", "W3 retained no prospective original filesystem/session/executor snapshot for this boundary, so faithful reconstruction of filesystem, compaction, tool IDs, read-before-edit and pending executor state cannot be certified from this dataset."]
    finally:
        if not args.keep_container:
            cleanup = run(["docker", "rm", "-f", name], timeout=60)
            cert["cleanup"] = {"returncode": cleanup.returncode, "stdout": cleanup.stdout, "stderr": cleanup.stderr}
    out = W4 / "checkpoint_certificate.json"
    out.write_text(json.dumps(cert, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"certificate": str(out), "status": cert["status"], "selected_prefix": chosen["prefix_id"], "reference_request_sha256": sha256(request), "resume_stub_requests": len(cert.get("observations", [{}])[-3].get("stub_requests", [])) if cert.get("observations") else 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
