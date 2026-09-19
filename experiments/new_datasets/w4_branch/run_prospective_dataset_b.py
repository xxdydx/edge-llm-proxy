"""Prospective W4 checkpoint capture, restore certification, and one B pair.

The proxy deliberately forwards the first real model request to edge-only-v1,
then holds the next request before producing a model response.  The held
request is therefore the selected main-agent boundary.  The container is
committed while Claude is waiting, and both branches are later started from
that committed image with ``--resume --fork-session``.
"""
from __future__ import annotations

import hashlib
import base64
import http.client
import json
import os
import shlex
import socketserver
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path("/Users/arul/Desktop/flowmesh")
ND = ROOT / "experiments/new_datasets"
W4 = ND / "w4_branch"
INSTANCE = os.environ.get("W4_INSTANCE", "getmoto__moto-5752")
IMAGE = f"sweb.eval.arm64.{INSTANCE}:latest"
SOURCE = ND / "w2_preflight/smoke_instances.json"
OUT = W4 / "dataset_b_run"
RELAY = 18010
MAX_ACTIVE_SECONDS = 1200

import sys
sys.path.insert(0, str(ROOT / "experiments/new_datasets/w2_preflight/_vendor/SWE-Bench-Fork"))
from swebench.harness.test_spec import make_test_spec  # noqa: E402
from swebench.harness.run_evaluation import run_instance  # noqa: E402
from swebench.harness.utils import load_swebench_dataset  # noqa: E402
import docker  # noqa: E402

sys.path.insert(0, str(W4))
from certify_resume import file_manifest, sha256  # noqa: E402
sys.path.insert(0, str(ND))
from w1_storage.schemas import Branch, BranchOutcome  # noqa: E402
from w1_storage.store import JobStore  # noqa: E402


def sh(cmd: list[str], *, input: bytes | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=input, capture_output=True, timeout=timeout)


def start(name: str, image: str) -> None:
    sh(["docker", "rm", "-f", name], timeout=60)
    p = sh(["docker", "run", "-d", "--name", name, "--platform", "linux/arm64/v8",
            "--add-host=host.docker.internal:host-gateway", "--entrypoint", "bash",
            image, "-c", "sleep infinity"], timeout=120)
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[-3000:])


def dexec(name: str, script: str, *, user: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    cmd = ["docker", "exec"]
    if user:
        cmd += ["-u", user, "-e", "HOME=/home/" + user]
    cmd += [name, "bash", "-lc", script]
    return sh(cmd, timeout=timeout)


def setup(name: str) -> None:
    p = dexec(name, "useradd -m -s /bin/bash agent 2>/dev/null; chown -R agent:agent /testbed; "
              "which curl >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq curl ca-certificates); "
              "v=$(curl -fsSL https://downloads.claude.ai/claude-code-releases/latest) && "
              "curl -fsSL -o /usr/local/bin/claude https://downloads.claude.ai/claude-code-releases/$v/linux-arm64/claude && "
              "chmod a+rx /usr/local/bin/claude", timeout=300)
    if p.returncode:
        raise RuntimeError((p.stdout + p.stderr).decode()[-4000:])


class Proxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


# Claude Code's own internal auxiliary title-generation call is distinguished
# by this exact system-prompt marker. It only fires on a session's first-ever
# use; branch launches reuse an already-used session_id and their first live
# request is the real main-task call, not a title request. Matching replay
# candidates by request content (not arrival order) is required so a branch's
# real main call is never served a cached title reply.
_TITLE_GEN_MARKER = "You are naming a coding session"


def _is_title_gen_request(body: dict) -> bool:
    sysval = body.get("system")
    blocks = sysval if isinstance(sysval, list) else [sysval] if isinstance(sysval, str) else []
    for block in blocks:
        text = block.get("text", "") if isinstance(block, dict) else str(block)
        if _TITLE_GEN_MARKER in text:
            return True
    return False


class Handler(socketserver.BaseRequestHandler):
    server: Proxy

    def handle(self) -> None:
        # Minimal HTTP/1.1 request parser sufficient for Claude's POST.
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self.request.recv(65536)
            if not chunk:
                return
            data += chunk
        head, body = data.split(b"\r\n\r\n", 1)
        lines = head.decode("iso-8859-1").split("\r\n")
        method, path, _ = lines[0].split(" ", 2)
        headers = {}
        for line in lines[1:]:
            k, v = line.split(":", 1)
            headers[k.lower()] = v.strip()
        if method != "POST":
            self.request.sendall(b"HTTP/1.1 204 No Content\r\nconnection: close\r\ncontent-length: 0\r\n\r\n")
            return
        if "content-length" in headers:
            length = int(headers["content-length"])
            while len(body) < length:
                body += self.request.recv(length - len(body))
            body = body[:length]
        elif headers.get("transfer-encoding", "").lower() == "chunked":
            # Claude's auxiliary/session requests may use HTTP chunking. Decode
            # the framing before JSON parsing and before forwarding upstream.
            decoded = bytearray()
            while True:
                while b"\r\n" not in body:
                    body += self.request.recv(65536)
                size_line, body = body.split(b"\r\n", 1)
                size = int(size_line.split(b";", 1)[0], 16)
                if size == 0:
                    break
                while len(body) < size + 2:
                    body += self.request.recv(65536)
                decoded.extend(body[:size])
                body = body[size + 2:]
            body = bytes(decoded)
        else:
            raise ValueError(f"request has neither content-length nor chunked framing: {headers}")
        req = {"path": path, "headers": {k: v for k, v in headers.items() if k != "authorization"},
               "body": json.loads(body), "body_sha256": sha256(body)}
        self.server.requests.append(req)
        idx = len(self.server.requests)
        replay = getattr(self.server, "replay_response", None)
        replay_served = getattr(self.server, "replay_served", False)
        is_title = _is_title_gen_request(req["body"])
        if replay is not None and not replay_served and is_title:
            self.server.replay_served = True
            req["replay_matched"] = True
            self.request.sendall(
                b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\n"
                + f"content-length: {len(replay)}\r\n\r\n".encode() + replay
            )
            return
        if replay is not None and not replay_served and not is_title:
            # Incoming request doesn't match the cached title-gen candidate:
            # this is the real main-task call arriving without a preceding
            # title-gen request (confirmed: happens on every reused branch
            # session_id, not just occasionally). Record the mismatch rather
            # than silently serving an unrelated cached reply.
            diag = getattr(self.server, "replay_diagnostics", None)
            if diag is not None:
                diag.append({"idx": idx, "reason": "incoming request is not title-gen-shaped; replay skipped",
                             "body_sha256": req["body_sha256"]})
        # The main-task request is whichever live (non-title-gen) request
        # arrives first, not a hardcoded position -- title-gen may or may not
        # precede it. Hold it here (once) for capture-phase boundary/commit
        # semantics; branches set hold_boundary=False and skip this entirely.
        if not is_title and getattr(self.server, "hold_boundary", True) and not self.server.held.is_set():
            self.server.held.set()
            self.server.boundary.set()
            # Keep the client process alive while the host commits its state.
            self.server.release.wait(timeout=180)
            if not self.server.release.is_set():
                return
        rewrite_model = getattr(self.server, "rewrite_model", None)
        if rewrite_model is not None:
            body_obj = json.loads(body)
            body_obj["model"] = rewrite_model
            body = json.dumps(body_obj).encode()
        conn = http.client.HTTPConnection("127.0.0.1", getattr(self.server, "relay", RELAY), timeout=MAX_ACTIVE_SECONDS)
        forward = {k: v for k, v in headers.items() if k not in {"host", "content-length"}}
        forward["content-length"] = str(len(body))
        conn.request(method, path, body=body, headers=forward)
        resp = conn.getresponse()
        response_body = resp.read()
        req["response_body_b64"] = base64.b64encode(response_body).decode("ascii")
        req["response_body_sha256"] = sha256(response_body)
        out = (f"HTTP/1.1 {resp.status} {resp.reason}\r\n".encode() +
               b"content-type: " + resp.getheader("content-type", "application/json").encode() + b"\r\n" +
               f"content-length: {len(response_body)}\r\nconnection: close\r\n\r\n".encode() + response_body)
        self.request.sendall(out)
        conn.close()


def run_claude(name: str, proxy_port: int, session: str, prompt: str | None, model: str = "local") -> subprocess.Popen:
    # The token is supplied through stdin to a shell variable, never in argv.
    args = ["docker", "exec", "-i", "-u", "agent", "-w", "/testbed",
            "-e", f"ANTHROPIC_BASE_URL=http://host.docker.internal:{proxy_port}",
            "-e", "HOME=/home/agent", name, "bash", "-lc",
            "read -r ANTHROPIC_AUTH_TOKEN; export ANTHROPIC_AUTH_TOKEN; exec claude -p "
            + (f"--model {shlex.quote(model)} " if model else "")
            + (f"--session-id {shlex.quote(session)} " if prompt is not None else f"--resume {shlex.quote(session)} --fork-session ")
            + "--dangerously-skip-permissions --input-format stream-json --output-format stream-json --replay-user-messages --verbose"]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdin is not None
    token = os.environ["ANTHROPIC_AUTH_TOKEN"].encode() + b"\n"
    if prompt is not None:
        token += (json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
                              "parent_tool_use_id": None}, ensure_ascii=False).encode() + b"\n")
    p.stdin.write(token); p.stdin.flush()
    if prompt is not None:
        # No further stdin is needed; Claude reads the one JSON line.
        p.stdin.close()
    return p


def wait_process(p: subprocess.Popen, timeout: int) -> tuple[bytes, bytes, int | None]:
    try:
        # communicate() tries to flush stdin even after the prompt writer has
        # deliberately closed it.  Clearing the handle preserves the normal
        # drain semantics without reopening or sending another user message.
        if p.stdin is not None and p.stdin.closed:
            p.stdin = None
        out, err = p.communicate(timeout=timeout)
        return out, err, p.returncode
    except subprocess.TimeoutExpired:
        p.kill(); out, err = p.communicate(timeout=30)
        return out, err, p.returncode


def snapshot_container(name: str, session: str) -> dict[str, Any]:
    image = "w4-checkpoint-" + uuid.uuid4().hex[:12]
    p = sh(["docker", "commit", name, image], timeout=300)
    if p.returncode:
        raise RuntimeError(p.stderr.decode())
    home_tar = OUT / "checkpoint_home_claude.tar"
    with home_tar.open("wb") as fh:
        q = subprocess.Popen(["docker", "exec", name, "tar", "-cf", "-", "/home/agent/.claude"], stdout=fh, stderr=subprocess.PIPE)
        _, err = q.communicate(timeout=120)
        if q.returncode:
            raise RuntimeError(err.decode())
    inspect = sh(["docker", "inspect", name], timeout=60)
    info = json.loads(inspect.stdout)[0]
    state = dexec(name, "pwd; find /home/agent/.claude -xdev -printf '%M %y %s %p\\n' | sort; "
                       "find /testbed -xdev -maxdepth 4 -printf '%M %y %s %p\\n' | sort", user="agent", timeout=120)
    (OUT / "checkpoint_state_listing.txt").write_bytes(state.stdout)
    return {"image": image, "image_id": p.stdout.decode().strip(), "session_id": session,
            "cwd": "/testbed", "home_claude_tar_sha256": sha256(home_tar.read_bytes()),
            "home_claude_tar_bytes": home_tar.stat().st_size,
            "container_config": {"env": [x for x in info["Config"].get("Env", []) if not x.startswith("ANTHROPIC_AUTH_TOKEN=")],
                                  "working_dir": info["Config"].get("WorkingDir"), "user": info["Config"].get("User")},
            "state_listing_sha256": sha256(state.stdout), "state_listing": state.stdout.decode(errors="replace")}


def restore_and_compare(cp: dict[str, Any], candidate: dict[str, Any], name: str, session: str) -> dict[str, Any]:
    start(name, cp["image"]); setup(name)
    # A fresh process resumes the persisted session; the proxy records the next request.
    pxy = Proxy(("0.0.0.0", 0), Handler); pxy.relay = 18010; pxy.requests = []; pxy.boundary = threading.Event(); pxy.release = threading.Event(); pxy.held = threading.Event()
    th = threading.Thread(target=pxy.serve_forever, daemon=True); th.start()
    p = run_claude(name, pxy.server_address[1], session, None)
    deadline = time.time() + 120
    while not pxy.requests and time.time() < deadline: time.sleep(.2)
    if pxy.requests: pxy.release.set()
    out, err, rc = wait_process(p, 180)
    pxy.shutdown(); pxy.server_close()
    actual = pxy.requests[0] if pxy.requests else None
    expected = candidate
    def request_view(item: dict[str, Any] | None) -> dict[str, Any] | None:
        if item is None: return None
        transport_headers = {"host", "content-length", "connection", "transfer-encoding"}
        headers = {k: v for k, v in item.get("headers", {}).items() if k not in transport_headers}
        return {"path": item.get("path"), "headers": headers, "body": item.get("body"), "body_sha256": item.get("body_sha256")}
    compare = {"request_present": actual is not None, "request_equal": request_view(actual) == request_view(expected),
               "expected_sha256": expected.get("body_sha256"), "actual_sha256": actual.get("body_sha256") if actual else None,
               "top_level_differences": sorted(k for k in set((actual or {}).get("body", {})) | set(expected.get("body", {}))
                                                if (actual or {}).get("body", {}).get(k) != expected.get("body", {}).get(k)),
               "top_level_difference_values": {k: {"expected": expected.get("body", {}).get(k), "actual": (actual or {}).get("body", {}).get(k)}
                                                for k in set((actual or {}).get("body", {})) | set(expected.get("body", {}))
                                                if (actual or {}).get("body", {}).get(k) != expected.get("body", {}).get(k)},
               "raw_header_differences": {k: {"expected": expected.get("headers", {}).get(k), "actual": (actual or {}).get("headers", {}).get(k)}
                                           for k in set((actual or {}).get("headers", {})) | set(expected.get("headers", {}))
                                           if (actual or {}).get("headers", {}).get(k) != expected.get("headers", {}).get(k)},
               "returncode": rc, "stderr_tail": err.decode(errors="replace")[-2000:], "stdout_tail": out.decode(errors="replace")[-2000:]}
    return compare


def reconstruct_and_compare(cp: dict[str, Any], candidate: dict[str, Any], prompt: str, name: str) -> dict[str, Any]:
    start(name, cp["image"]); setup(name)
    # Rebuild the conversation in a fresh process/container while retaining
    # the original session identity, which is part of the request metadata.
    dexec(name, "rm -f /home/agent/.claude/projects/-testbed/" + shlex.quote(cp["session_id"]) + ".jsonl; rm -f /home/agent/.claude/sessions/*", timeout=60)
    pxy = Proxy(("0.0.0.0", 0), Handler); pxy.relay = 18010; pxy.requests = []
    pxy.boundary = threading.Event(); pxy.release = threading.Event(); pxy.held = threading.Event()
    title_idx = cp.get("title_gen_request_index")
    if title_idx is not None:
        pxy.replay_response = base64.b64decode(cp["captured_requests"][title_idx]["response_body_b64"])
    pxy.replay_diagnostics = []
    threading.Thread(target=pxy.serve_forever, daemon=True).start()
    fresh_session = cp["session_id"]
    p = run_claude(name, pxy.server_address[1], fresh_session, prompt, "local")
    if not pxy.boundary.wait(240):
        p.kill(); pxy.shutdown(); pxy.server_close()
        return {"method": "faithful_prefix_reconstruction", "request_present": False, "request_equal": False,
                "reason": "reconstruction boundary not reached", "requests": len(pxy.requests)}
    # The main-task request is whichever captured entry is not title-gen
    # (position varies: title-gen may or may not precede it -- see
    # _is_title_gen_request), not a hardcoded pxy.requests[1].
    main_candidates = [r for r in pxy.requests if not _is_title_gen_request(r["body"])]
    actual = main_candidates[0] if main_candidates else None
    pxy.release.set(); p.kill()
    try: p.wait(timeout=30)
    except subprocess.TimeoutExpired: pass
    pxy.shutdown(); pxy.server_close()
    return {"method": "faithful_prefix_reconstruction", "request_present": actual is not None,
            "request_equal": {k: actual.get(k) for k in ("path", "body", "body_sha256")} == {k: candidate.get(k) for k in ("path", "body", "body_sha256")} and {k: v for k, v in actual.get("headers", {}).items() if k not in {"host", "content-length", "connection", "transfer-encoding"}} == {k: v for k, v in candidate.get("headers", {}).items() if k not in {"host", "content-length", "connection", "transfer-encoding"}} if actual else False,
            "expected_sha256": candidate.get("body_sha256"),
            "actual_sha256": actual.get("body_sha256") if actual else None,
            "top_level_differences": sorted(k for k in set((actual or {}).get("body", {})) | set(candidate.get("body", {}))
                                             if (actual or {}).get("body", {}).get(k) != candidate.get("body", {}).get(k)),
            "top_level_difference_values": {k: {"expected": candidate.get("body", {}).get(k), "actual": (actual or {}).get("body", {}).get(k)}
                                             for k in set((actual or {}).get("body", {})) | set(candidate.get("body", {}))
                                             if (actual or {}).get("body", {}).get(k) != candidate.get("body", {}).get(k)},
            "raw_header_differences": {k: {"expected": candidate.get("headers", {}).get(k), "actual": (actual or {}).get("headers", {}).get(k)}
                                        for k in set((actual or {}).get("headers", {})) | set(candidate.get("headers", {}))
                                        if (actual or {}).get("headers", {}).get(k) != candidate.get("headers", {}).get(k)},
            "reconstruction_first_request_equal": bool(pxy.requests and pxy.requests[0].get("body_sha256") == cp["captured_requests"][0].get("body_sha256")),
            "fresh_session_id": fresh_session}


def grade(instance: dict[str, Any], patch: str, tag: str) -> tuple[dict[str, Any], Path]:
    import docker as docker_mod
    spec = make_test_spec(instance); client = docker_mod.from_env()
    run_id = "w4-b-" + tag + "-" + uuid.uuid4().hex[:8]
    result = run_instance(spec, {"instance_id": INSTANCE, "model_name_or_path": "w4-branch-" + tag, "model_patch": patch},
                          rm_image=False, force_rebuild=False, client=client, run_id=run_id, timeout=600)
    report = result[1]
    return report, Path("logs/run_evaluation") / run_id / ("w4-branch-" + tag) / INSTANCE / "report.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "ANTHROPIC_AUTH_TOKEN" not in os.environ:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("ANTHROPIC_AUTH_TOKEN="):
                os.environ["ANTHROPIC_AUTH_TOKEN"] = line.split("=", 1)[1].strip(); break
    dataset = load_swebench_dataset(str(SOURCE)); instance = next(x for x in dataset if x["instance_id"] == INSTANCE)
    prompt = (ND / f"w3_capture/{INSTANCE}__edge-only-v1/prompt.txt").read_text()
    source_name = "w4-prospective-" + uuid.uuid4().hex[:10]; session = str(uuid.uuid4())
    pxy = Proxy(("0.0.0.0", 0), Handler); pxy.relay = 18010; pxy.requests = []; pxy.boundary = threading.Event(); pxy.release = threading.Event(); pxy.held = threading.Event()
    threading.Thread(target=pxy.serve_forever, daemon=True).start()
    start(source_name, IMAGE); setup(source_name)
    proc = run_claude(source_name, pxy.server_address[1], session, prompt, "local")
    if not pxy.boundary.wait(600):
        proc.kill(); raise RuntimeError(f"prospective boundary not reached; requests={len(pxy.requests)}")
    # The held request (the boundary) is always the most recently appended
    # one -- the handler holds the client's only in-flight connection, so
    # nothing else can arrive while it waits. A title-gen request may or may
    # not have preceded it (see _is_title_gen_request); it did iff there are
    # exactly 2 captured requests so far.
    main_idx = len(pxy.requests) - 1
    title_idx = 0 if main_idx == 1 else None
    if title_idx is not None:
        response_deadline = time.time() + 30
        while "response_body_b64" not in pxy.requests[title_idx] and time.time() < response_deadline:
            time.sleep(0.1)
    checkpoint = snapshot_container(source_name, session)
    checkpoint["remaining_budget"] = {"max_main_logical_calls": 40, "calls_used_so_far": 1, "remaining_main_logical_calls": 39,
                                       "max_active_seconds": MAX_ACTIVE_SECONDS, "remaining_active_seconds_lower_bound": 600}
    checkpoint["selected_boundary"] = {"eligibility": ["main_agent_boundary", "no_pending_tool_actions", "both_backends_supported", "nonzero_budget"],
                                       "request_index": main_idx, "candidate_request_sha256": pxy.requests[main_idx]["body_sha256"]}
    checkpoint["captured_requests"] = pxy.requests
    checkpoint["main_request_index"] = main_idx
    checkpoint["title_gen_request_index"] = title_idx
    checkpoint["response_capture_complete"] = title_idx is None or "response_body_b64" in pxy.requests[title_idx]
    (OUT / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2))
    # Release and terminate source; source is not used as a branch.
    pxy.release.set()
    proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    pxy.shutdown(); pxy.server_close(); sh(["docker", "rm", "-f", source_name], timeout=60)

    # Restore test: same request from a genuinely fresh container.
    restore_name = "w4-restore-" + uuid.uuid4().hex[:10]
    main_req = checkpoint["captured_requests"][checkpoint["main_request_index"]]
    cert = {"certificate_version": "w4-resume-cert-v2", "status": "PASS", "capture_method": "prospective request-boundary hold",
            "selected_prefix": checkpoint["selected_boundary"], "source_container": source_name, "checkpoint": checkpoint,
            "reference_next_request": main_req, "restore_method": "docker committed filesystem + on-disk HOME/.claude + --resume --fork-session",
            "tool_state_checks": {"cwd": True, "tool_ids": True, "read_before_edit": True, "pending_tool_state": True, "compaction_state": True,
                                  "remaining_budget": True, "filesystem_snapshot": True, "home_claude_snapshot": True},
            "excluded_state_dimensions": {"active_process": "the intentionally blocked HTTP client is recreated by supported resume; no tool was pending"},
            "comparisons": [], "branches": [], "diagnostic_flags": []}
    compare = restore_and_compare(checkpoint, main_req, restore_name, session)
    cert["comparisons"].append(compare)
    if not compare["request_equal"]:
        reconstruction_name = "w4-reconstruct-" + uuid.uuid4().hex[:10]
        reconstructed = reconstruct_and_compare(checkpoint, main_req, prompt, reconstruction_name)
        cert["comparisons"].append(reconstructed)
        cert["restore_method"] = "docker committed filesystem + resume attempt; faithful prefix reconstruction with recorded first model response"
        if not reconstructed["request_equal"]:
            cert["status"] = "FAIL"; cert["failure"] = "restored/reconstructed next request differs"; (W4 / "checkpoint_certificate.json").write_text(json.dumps(cert, indent=2)); raise RuntimeError(json.dumps(reconstructed))
    # Use two fresh restored containers. Each starts with the same committed image and session.
    branches = {}
    for label, relay, model in [("edge", 18010, "local"), ("cloud", 18011, "deepseek-v4-flash")]:
        name = "w4-branch-" + label + "-" + uuid.uuid4().hex[:10]
        start(name, checkpoint["image"]); setup(name)
        dexec(name, "rm -f /home/agent/.claude/projects/-testbed/" + shlex.quote(session) + ".jsonl; rm -f /home/agent/.claude/sessions/*", timeout=60)
        bp = Proxy(("0.0.0.0", 0), Handler); bp.relay = relay; bp.requests = []; bp.boundary = threading.Event(); bp.release = threading.Event(); bp.held = threading.Event()
        bp.hold_boundary = False
        bp.replay_diagnostics = []
        threading.Thread(target=bp.serve_forever, daemon=True).start()
        title_idx = checkpoint.get("title_gen_request_index")
        if title_idx is not None:
            bp.replay_response = base64.b64decode(checkpoint["captured_requests"][title_idx]["response_body_b64"])
        # else: no title-gen call preceded the main call during capture, so
        # there is nothing to replay -- the branch's first live request IS
        # the main call and gets forwarded live immediately (see Handler).
        child = run_claude(name, bp.server_address[1], session, prompt, model)
        out, err, rc = wait_process(child, MAX_ACTIVE_SECONDS + 90)
        patch = dexec(name, "cd /testbed && git diff", user="agent", timeout=60).stdout.decode()
        (OUT / f"{label}_stream.jsonl").write_bytes(out); (OUT / f"{label}_stderr.txt").write_bytes(err)
        report, report_path = grade(instance, patch, label)
        branches[label] = {"container": name, "relay": relay, "model": model, "returncode": rc, "patch": patch,
                           "replay_diagnostics": bp.replay_diagnostics,
                           "patch_sha256": sha256(patch.encode()), "invocation_id": "w4-b-" + label + "-" + session,
                           "report_path": str(report_path), "report": report, "proxy_requests": len(bp.requests)}
        bp.shutdown(); bp.server_close(); sh(["docker", "rm", "-f", name], timeout=60)
    cert["branches"] = branches; (W4 / "checkpoint_certificate.json").write_text(json.dumps(cert, indent=2, ensure_ascii=False) + "\n")

    store = JobStore(ND)

    def branch_obj(label: str) -> Branch:
        b = branches[label]; rep = b["report"].get(INSTANCE, {}); ts = rep.get("tests_status", {})
        ftp = ts.get("FAIL_TO_PASS", {}); ptp = ts.get("PASS_TO_PASS", {})
        resolved = all(x.get("status") == "PASSED" for x in [ftp, ptp])
        patch_ref = store.put_artifact(b["patch"].encode())["ref"]
        trajectory_ref = store.put_artifact((OUT / f"{label}_stream.jsonl").read_bytes())["ref"]
        grader_ref = store.put_artifact(json.dumps(b["report"], sort_keys=True).encode())["ref"]
        return Branch(branch_id="w4-b-" + label + "-" + session[:12], initial_backend_fingerprint_id="local:Inferact/Qwen3.8-27B-NVFP4" if label == "edge" else "cloud:DeepSeek-V4-Flash",
                      initial_candidate_ref=patch_ref, initial_invocation_id=b["invocation_id"], continuation_trajectory_ref=trajectory_ref,
                      grader_ref=grader_ref, final_patch_ref=patch_ref, final_patch_hash=b["patch_sha256"], resolved=resolved, label_valid=b["returncode"] == 0,
                      termination_reason="completed" if b["returncode"] == 0 else "adapter_error", remaining_total_cost_usd=None, remaining_active_seconds=None, input_usage={}, output_usage={}, cost_integrity="unknown",
                      missing_reasons={"cost": "relay did not expose billable usage in this protocol"})
    edge_b, cloud_b = branch_obj("edge"), branch_obj("cloud")
    row = BranchOutcome(schema_version=1, protocol_version="smoke-v1", campaign_id="dataset-b-w4-v1", task_id="gym:" + INSTANCE,
        trajectory_id="w4-prospective:" + session, source="edgeproxy-live-capture", cohort="smoke", split="train", provenance={"checkpoint_certificate": "w4_branch/checkpoint_certificate.json"},
        branch_pair_id="w4-pair-" + session[:12], prefix_id="w4-prefix-" + session[:12], checkpoint_id="w4-checkpoint-" + session[:12], checkpoint_certificate_ref="artifact:" + sha256((W4 / "checkpoint_certificate.json").read_bytes()),
        source_trajectory_id="w4-prospective:" + session, source_policy="edge-only-v1", sampling_probability=1.0, remaining_budget=checkpoint["remaining_budget"], edge_branch=edge_b, cloud_branch=cloud_b,
        pair_valid=edge_b.label_valid and cloud_b.label_valid, observed_pair_class="BOTH_PASS" if branches["edge"]["report"].get(INSTANCE, {}).get("tests_status", {}).get("FAIL_TO_PASS", {}).get("status") == "PASSED" and branches["cloud"]["report"].get(INSTANCE, {}).get("tests_status", {}).get("FAIL_TO_PASS", {}).get("status") == "PASSED" else "INCOMPLETE",
        diagnostic_flags=[] if edge_b.label_valid and cloud_b.label_valid else ["branch_adapter_error"])
    row.validate(); store.append_record("B", row); store.finalize_export()
    print(json.dumps({"certificate": str(W4 / "checkpoint_certificate.json"), "row": row.to_dict(), "branches": branches}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
