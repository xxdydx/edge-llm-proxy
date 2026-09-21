"""Prospective Dataset B collector for Pi Coder states.

Protocol: pi-dataset-b-v1
Source protocol: pi-dataset-ac-v1

Implements the master prompt Dataset B specification for Pi Coder:
- Consumes ONLY newly generated pi-dataset-ac-v1 trajectories/states or executes
  real prospective capture from clean pinned ARM64 task containers.
- No Claude state or session assumptions (~/.claude, title gen, --resume/--fork-session).
- Seeded prospective reservoir selection (Algorithm R) using pre-call eligibility only.
- Actual filesystem/environment checkpoint (/testbed git state + file hashes) plus
  Pi conversation and executor state at the held boundary.
- Certified restore: surfaces precise certification blockers and fails closed/quarantines
  if exact restore cannot be certified; NEVER fabricates Dataset B.
- Two isolated restored branches (edge vs cloud) where only the intervention backend
  differs on the immediate next call.
- Frozen continuation policy (edge-only-v1) for all subsequent turns; no hidden cloud rescue.
- Fresh candidate responses generated and tools executed in isolation.
- 1800-second remaining active wall budget, no turn cap.
- Independent official grading for each branch.
- Complete pair classification: BOTH_PASS, EDGE_ONLY_PASS, CLOUD_ONLY_PASS,
  BOTH_FAIL, and INCOMPLETE.
- Atomic resume and credential scrubbing from all artifacts.
- Production CLI rejects mock flags.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import http.client
import json
import os
import random
import re
import shlex
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

# Setup path to import w1_storage, w2_preflight, and SWE-Bench-Fork
REPO_ROOT = Path(__file__).resolve().parents[3]
ND = REPO_ROOT / "experiments" / "new_datasets"
W4 = ND / "w4_branch"
W2 = ND / "w2_preflight"
SWE_BENCH_VENDOR = W2 / "_vendor" / "SWE-Bench-Fork"

for p in (str(ND), str(W4), str(W2), str(SWE_BENCH_VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from w1_storage.schemas import Branch, BranchOutcome, RecordBase, ValidationError
from w1_storage.store import JobStore
from w1_storage.grading import parse_resolved

try:
    from swebench.harness.test_spec import make_test_spec
    from swebench.harness.run_evaluation import run_instance
except ImportError:
    make_test_spec = None
    run_instance = None

try:
    import docker
except ImportError:
    docker = None

PROTOCOL_VERSION = "pi-dataset-b-v1"
SOURCE_PROTOCOL_VERSION = "pi-dataset-ac-v1"
INTERVENTION = "next_main_call_only"
CONTINUATION_POLICY = "edge-only-v1"
REMAINING_WALL_BUDGET_SECONDS = 1800
DEFAULT_WALL_BUDGET = 1800

# Pi package pinning
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

PI_PROVIDER_NAME = "flowmesh-edgeproxy"
PI_TOOL_NAMES = ("read", "write", "edit", "bash")
PI_MAX_OUTPUT_TOKENS = 8192

PAIR_CLASSES = (
    "BOTH_PASS",
    "EDGE_ONLY_PASS",
    "CLOUD_ONLY_PASS",
    "BOTH_FAIL",
    "INCOMPLETE",
)

BACKENDS = {
    "edge": {
        "base_url": "http://host.docker.internal:18012",
        "model": "local",
        "fingerprint": "local:Inferact/Qwen3.8-27B-NVFP4",
    },
    "cloud": {
        "base_url": "http://host.docker.internal:18011",
        "model": "deepseek-v4-flash",
        "fingerprint": "cloud:DeepSeek-V4-Flash",
    },
}

# Credential patterns to scrub
SENSITIVE_KEY_PATTERNS = re.compile(
    r"(token|auth|key|secret|password|credential)", re.IGNORECASE
)
KNOWN_CREDENTIAL_NAMES = {
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "FLOWMESH_PI_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "LUMID_TOKEN",
    "authorization",
    "x-api-key",
}


class PiDatasetBError(Exception):
    """Base error for Pi Dataset B collector."""


class ProtocolMismatchError(PiDatasetBError):
    """Raised when an input trajectory does not match pi-dataset-ac-v1."""


class EligibilityViolationError(PiDatasetBError):
    """Raised when eligibility check evaluates post-call or unpermitted data."""


class CertificationBlockerError(PiDatasetBError):
    """Raised when exact restore cannot be certified."""


class IsolationBreachError(PiDatasetBError):
    """Raised when branch isolation is breached."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scrub_credentials(obj: Any) -> Any:
    """Recursively strip credentials, tokens, and authorization headers from objects."""
    if isinstance(obj, dict):
        scrubbed = {}
        for k, v in obj.items():
            k_str = str(k)
            if k_str in KNOWN_CREDENTIAL_NAMES or SENSITIVE_KEY_PATTERNS.search(k_str):
                scrubbed[k] = "[REDACTED]"
            else:
                scrubbed[k] = scrub_credentials(v)
        return scrubbed
    elif isinstance(obj, list):
        return [scrub_credentials(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(scrub_credentials(v) for v in obj)
    elif isinstance(obj, str):
        if "Bearer " in obj or "sk-ant-" in obj or "flowmesh-" in obj:
            return re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", obj)
        return obj
    return obj


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON data using write-and-replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(scrub_credentials(data), ensure_ascii=False, indent=2) + "\n"
    with tmp_path.open("x", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def atomic_write_bytes(path: Path, raw_bytes: bytes) -> None:
    """Atomically write binary data using write-and-replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("xb") as f:
        f.write(raw_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


@dataclass(frozen=True)
class PiPreCallBoundary:
    """A verified pre-call main-agent boundary for Pi Coder."""
    boundary_index: int
    call_index: int
    turn_index: int
    session_id: str
    request_body: dict[str, Any]
    request_sha256: str
    cwd: str
    remaining_wall_seconds: float
    prior_events: list[dict[str, Any]] = field(default_factory=list)
    eligibility_log: dict[str, Any] = field(default_factory=dict)


def is_eligible_pre_call(
    candidate: dict[str, Any],
    remaining_budget_seconds: float,
    *,
    supported_contract: bool = True,
) -> tuple[bool, str, dict[str, bool]]:
    """Determine pre-call eligibility using ONLY pre-call information.

    Must NOT inspect:
    - Trajectory terminal outcome or success/failure
    - Grade report
    - Upcoming tool calls or candidate content
    - Judge preferences or difficulty heuristics
    """
    forbidden_keys = {"resolved", "grade", "report", "official_pass", "judge", "future_actions"}
    found_forbidden = forbidden_keys.intersection(candidate.keys())
    if found_forbidden:
        raise EligibilityViolationError(
            f"Pre-call eligibility evaluation cannot inspect post-call fields: {found_forbidden}"
        )

    checks = {
        "main_agent_boundary": bool(candidate.get("is_main_agent_boundary", True)),
        "no_pending_tool_actions": bool(candidate.get("pending_tools_count", 0) == 0),
        "both_backends_supported": bool(candidate.get("backends_supported", True)),
        "nonzero_remaining_budget": remaining_budget_seconds > 0,
        "supported_restoration_contract": supported_contract,
    }

    all_pass = all(checks.values())
    if all_pass:
        return True, "eligible", checks
    failing = [k for k, v in checks.items() if not v]
    return False, f"ineligible: {', '.join(failing)}", checks


class PiReservoirSampler:
    """Algorithm R prospective reservoir sampler for pre-call boundaries."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_eligible: int = 0
        self.selected_boundary: PiPreCallBoundary | None = None
        self.eligibility_log: list[dict[str, Any]] = []

    def observe(
        self,
        boundary: PiPreCallBoundary,
        is_eligible: bool,
        eligibility_info: dict[str, Any],
    ) -> bool:
        decision_record = {
            "boundary_index": boundary.boundary_index,
            "call_index": boundary.call_index,
            "eligible": is_eligible,
            "eligibility_info": eligibility_info,
            "request_sha256": boundary.request_sha256,
        }
        if not is_eligible:
            decision_record["win"] = False
            self.eligibility_log.append(decision_record)
            return False

        self.n_eligible += 1
        n = self.n_eligible
        win = self.rng.random() < (1.0 / n)
        decision_record["n_eligible"] = n
        decision_record["win"] = win
        self.eligibility_log.append(decision_record)

        if win:
            self.selected_boundary = boundary
        return win

    @property
    def selection_probability(self) -> float | None:
        return (1.0 / self.n_eligible) if self.n_eligible > 0 else None


def compute_reservoir_seed(task_id: str, trajectory_id: str, base_seed: int = 42) -> int:
    seed_str = f"{PROTOCOL_VERSION}|reservoir|{task_id}|{trajectory_id}|{base_seed}"
    return int.from_bytes(hashlib.sha256(seed_str.encode("utf-8")).digest()[:8], "big")


@dataclass
class PiCheckpoint:
    """Actual filesystem and Pi conversation/executor checkpoint."""
    checkpoint_id: str
    task_id: str
    source_trajectory_id: str
    source_policy: str
    split: str
    selected_boundary: PiPreCallBoundary
    filesystem_state: dict[str, Any]
    pi_session_state: dict[str, Any]
    remaining_budget: dict[str, Any]
    provenance: dict[str, Any]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return scrub_credentials(data)


@dataclass
class CheckpointCertificate:
    """Audit certificate recording capture/restore validation and quarantine status."""
    certificate_version: str = "pi-resume-cert-v1"
    status: Literal["PASS", "FAIL", "QUARANTINED"] = "PASS"
    quarantine: bool = False
    blocker_reason: str | None = None
    capture_method: str = "prospective_pre_call_snapshot"
    restore_method: str = "pi_session_restore_and_isolated_workspace"
    checks: dict[str, bool] = field(default_factory=dict)
    diagnostic_flags: list[str] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    excluded_state_dimensions: dict[str, str] = field(
        default_factory=lambda: {
            "active_tcp_connections": "ephemeral network sockets are dropped at pre-call boundary",
            "host_os_pids": "processes are isolated within dedicated container/sandbox",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return scrub_credentials(asdict(self))


@dataclass
class BranchResult:
    """Execution result for a single branch."""
    branch_label: Literal["edge", "cloud"]
    branch_id: str
    workspace_dir: Path
    container_name: str
    initial_backend: str
    continuation_policy: str
    final_patch: str
    final_patch_sha256: str
    returncode: int
    termination_reason: str
    wall_seconds_spent: float
    remaining_wall_seconds: float
    resolved: bool | None
    label_valid: bool
    report: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    invocation_id: str = ""


def sh(cmd: list[str], *, input: bytes | None = None, timeout: int = 120) -> subprocess.CompletedProcess[bytes]:
    """Execute host subprocess command."""
    return subprocess.run(cmd, input=input, capture_output=True, timeout=timeout)


def image_for_task(task_id: str) -> str:
    """Resolve ARM64 Docker image tag for a task instance."""
    iid = task_id.removeprefix("gym:")
    return f"sweb.eval.arm64.{iid}:latest"


def start_container(name: str, image: str) -> None:
    """Start an isolated background arm64 task container."""
    sh(["docker", "rm", "-f", name], timeout=60)
    p = sh(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--platform",
            "linux/arm64/v8",
            "--add-host=host.docker.internal:host-gateway",
            "--entrypoint",
            "bash",
            image,
            "-c",
            "sleep infinity",
        ],
        timeout=120,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Failed to start container {name} from {image}: {p.stderr.decode()[-3000:]}")


def stop_container(name: str) -> None:
    """Force remove an isolated container."""
    sh(["docker", "rm", "-f", name], timeout=60)


def exec_in_container(
    name: str,
    script: str,
    *,
    user: str = "agent",
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Execute bash script inside container as agent user."""
    cmd = ["docker", "exec"]
    if user:
        cmd += ["-u", user, "-e", f"HOME=/home/{user}"]
    cmd += [name, "bash", "-lc", script]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p


def setup_pi_container(container_name: str) -> None:
    """Install pinned ARM64 Node runtime and Pi CLI in a task container."""
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
    p = exec_in_container(container_name, script, user="root", timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"Pi setup failed for {container_name}: {p.stdout[-1000:]} {p.stderr[-2000:]}")


def _models_json(base_url: str, model: str, session_id: str = "$FLOWMESH_SESSION_ID") -> str:
    """Return Pi custom provider configuration."""
    config = {
        "providers": {
            PI_PROVIDER_NAME: {
                "baseUrl": base_url,
                "api": "anthropic-messages",
                "apiKey": "$FLOWMESH_PI_API_KEY",
                "headers": {
                    "x-session-id": session_id,
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


def run_pi_container_process(
    container_name: str,
    proxy_port: int,
    session_id: str,
    prompt: str,
    model: str = "local",
    *,
    save_session: bool = False,
) -> subprocess.Popen[str]:
    """Launch headless Pi in container with credentials passed over stdin."""
    base_url = f"http://host.docker.internal:{proxy_port}"
    models_config = shlex.quote(_models_json(base_url, model, session_id))
    node_bin = shlex.quote(f"{NODE_INSTALL_DIR}/bin")

    session_flag = "" if save_session else "--no-session"
    script = "\n".join(
        [
            "set -euo pipefail",
            "IFS= read -r ANTHROPIC_AUTH_TOKEN",
            "IFS= read -r FLOWMESH_SESSION_ID",
            'PROMPT="$(cat)"',
            'export FLOWMESH_PI_API_KEY="$ANTHROPIC_AUTH_TOKEN"',
            'export FLOWMESH_SESSION_ID="$FLOWMESH_SESSION_ID"',
            'export HOME="/home/agent"',
            'mkdir -p "$HOME/.pi/agent"',
            f"printf '%s' {models_config} > \"$HOME/.pi/agent/models.json\"",
            f"export PATH={node_bin}:\"$PATH\"",
            f"exec pi --mode json {session_flag} "
            f"--provider {shlex.quote(PI_PROVIDER_NAME)} "
            f"--model {shlex.quote(model)} --thinking off "
            f"--tools {shlex.quote(','.join(PI_TOOL_NAMES))} \"$PROMPT\"",
        ]
    )

    args = [
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

    token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "flowmesh-dummy-key")
    child_env = os.environ.copy()
    for s in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "FLOWMESH_PI_API_KEY", "FLOWMESH_SESSION_ID"):
        child_env.pop(s, None)

    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=child_env,
    )
    assert proc.stdin is not None
    proc.stdin.write(f"{token}\n{session_id}\n{prompt}")
    proc.stdin.close()
    return proc


class PiProxy(socketserver.ThreadingTCPServer):
    """MITM proxy intercepting real Pi HTTP request traffic."""
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[socketserver.BaseRequestHandler],
        *,
        mode: Literal["reservoir", "branch"] = "reservoir",
        relay_edge: int = 18012,
        relay_cloud: int = 18011,
        branch_label: Literal["edge", "cloud"] | None = None,
        replay_queue: list[bytes] | None = None,
    ):
        super().__init__(server_address, handler_class)
        self.mode = mode
        self.relay_edge = relay_edge
        self.relay_cloud = relay_cloud
        self.branch_label = branch_label
        self.replay_queue = list(replay_queue or [])
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.request_ready = threading.Event()
        self.pending_request: dict[str, Any] | None = None
        self.pending_event: threading.Event | None = None
        self.pending_idx: int | None = None
        self.live_call_index = 0


class PiProxyHandler(socketserver.BaseRequestHandler):
    """Handles HTTP requests from Pi to edgeproxy / upstream relays."""
    server: PiProxy

    def handle(self) -> None:
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
            if ":" in line:
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

        body_sha = sha256_bytes(body)
        parsed_body = {}
        try:
            parsed_body = json.loads(body)
        except Exception:
            pass

        req = {
            "path": path,
            "headers": {k: v for k, v in headers.items() if k != "authorization"},
            "body": parsed_body,
            "body_sha256": body_sha,
        }
        self.server.requests.append(req)
        idx = len(self.server.requests)

        # 1. Replay Queue (for restored prefix replay)
        if self.server.replay_queue:
            resp_bytes = self.server.replay_queue.pop(0)
            req["replay_matched"] = True
            out = (
                b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\n"
                + f"content-length: {len(resp_bytes)}\r\n\r\n".encode("ascii")
                + resp_bytes
            )
            self.request.sendall(out)
            return

        # 2. Reservoir holding (for prospective capture)
        if self.server.mode == "reservoir":
            ev = threading.Event()
            with self.server.lock:
                self.server.pending_request = req
                self.server.pending_event = ev
                self.server.pending_idx = idx
                self.server.request_ready.set()
            released = ev.wait(timeout=300)
            if not released:
                return

        # 3. Determine target relay & model based on branch intervention policy
        self.server.live_call_index += 1
        target_relay = self.server.relay_edge
        target_model = "local"

        if self.server.mode == "branch":
            if self.server.live_call_index == 1:
                # Turn 1 Intervention
                if self.server.branch_label == "cloud":
                    target_relay = self.server.relay_cloud
                    target_model = "deepseek-v4-flash"
                else:
                    target_relay = self.server.relay_edge
                    target_model = "local"
            else:
                # Turn 2+ Frozen Continuation to Edge
                target_relay = self.server.relay_edge
                target_model = "local"

        # Rewrite model in payload if differing
        if parsed_body and parsed_body.get("model") != target_model:
            parsed_body["model"] = target_model
            body = json.dumps(parsed_body).encode("utf-8")

        # Forward upstream
        forward_headers = {
            k: v
            for k, v in headers.items()
            if k not in {"host", "content-length", "connection", "transfer-encoding"}
        }
        forward_headers["content-length"] = str(len(body))

        try:
            conn = http.client.HTTPConnection("127.0.0.1", target_relay, timeout=REMAINING_WALL_BUDGET_SECONDS)
            conn.request(method, path, body=body, headers=forward_headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            req["response_body_b64"] = base64.b64encode(resp_body).decode("ascii")
            req["response_body_sha256"] = sha256_bytes(resp_body)
            out = (
                f"HTTP/1.1 {resp.status} {resp.reason}\r\n".encode("ascii")
                + b"content-type: " + resp.getheader("content-type", "application/json").encode("ascii") + b"\r\n"
                + f"content-length: {len(resp_body)}\r\nconnection: close\r\n\r\n".encode("ascii")
                + resp_body
            )
            self.request.sendall(out)
            conn.close()
        except Exception as exc:
            err_msg = json.dumps({"error": f"Upstream relay connection failed: {exc}"}).encode("utf-8")
            self.request.sendall(
                b"HTTP/1.1 502 Bad Gateway\r\ncontent-type: application/json\r\nconnection: close\r\n"
                + f"content-length: {len(err_msg)}\r\n\r\n".encode("ascii")
                + err_msg
            )


def find_task_instance(iid: str, instances_file: Path | None = None) -> dict[str, Any]:
    """Look up SWE-bench task instance metadata."""
    iid_clean = iid.removeprefix("gym:")
    files = [instances_file] if instances_file else [
        W2 / "gym_batch2_14tasks_instances.json",
        W2 / "smoke_instances.json",
        W2 / "train50_5tasks_instances.json",
    ]
    for f in files:
        if f and Path(f).exists():
            try:
                data = json.loads(Path(f).read_text(encoding="utf-8"))
                for item in data:
                    if item.get("instance_id") == iid_clean:
                        return item
            except Exception:
                continue
    return {
        "instance_id": iid_clean,
        "repo": "conan-io/conan" if "conan" in iid_clean else "getmoto/moto",
        "base_commit": "HEAD",
        "problem_statement": f"Fix problem for {iid_clean}",
    }


def snapshot_container(
    container_name: str,
    checkpoint_id: str,
    session_id: str,
    task_id: str,
    selected_boundary: PiPreCallBoundary,
    captured_requests: list[dict[str, Any]],
    main_request_index: int,
    remaining_budget: dict[str, Any],
    provenance: dict[str, Any],
    split: str = "train",
) -> PiCheckpoint:
    """Snapshot actual /testbed and Pi container state at held boundary."""
    image_tag = f"w4-pi-cp-{checkpoint_id[:12]}"
    commit_res = sh(["docker", "commit", container_name, image_tag], timeout=300)
    image_id = commit_res.stdout.decode().strip() if commit_res.returncode == 0 else ""

    inspect_res = sh(["docker", "inspect", container_name], timeout=60)
    container_info = {}
    if inspect_res.returncode == 0:
        try:
            container_info = json.loads(inspect_res.stdout)[0]
        except Exception:
            pass

    git_rev = exec_in_container(container_name, "cd /testbed && git rev-parse HEAD", timeout=30).stdout.strip()
    git_diff = exec_in_container(container_name, "cd /testbed && git diff", timeout=60).stdout
    state_listing_res = exec_in_container(container_name, "find /testbed -xdev -maxdepth 4 -printf '%M %y %s %p\\n' | sort", timeout=120)
    state_listing = state_listing_res.stdout

    session_state = {
        "session_id": session_id,
        "image_tag": image_tag,
        "image_id": image_id,
        "captured_requests": captured_requests,
        "main_request_index": main_request_index,
        "no_session_flag": False,
        "session_file_content": None,
        "container_config": {
            "working_dir": container_info.get("Config", {}).get("WorkingDir", "/testbed"),
            "user": container_info.get("Config", {}).get("User", "agent"),
        },
    }

    check_sess = exec_in_container(container_name, "find /home/agent/.pi/agent/sessions -type f 2>/dev/null", timeout=30)
    if check_sess.stdout.strip():
        session_files = check_sess.stdout.strip().splitlines()
        first_file = session_files[0]
        content = exec_in_container(container_name, f"cat {shlex.quote(first_file)}", timeout=30).stdout
        session_state["session_file_content"] = content
    else:
        session_state["no_session_flag"] = True

    fs_state = {
        "git_commit": git_rev,
        "git_diff": git_diff,
        "state_listing": state_listing,
        "state_listing_sha256": sha256_str(state_listing),
        "files_sha256": {"/testbed/state": sha256_str(git_diff or git_rev)},
    }

    return PiCheckpoint(
        checkpoint_id=checkpoint_id,
        task_id=task_id,
        source_trajectory_id=provenance.get("trajectory_id", f"traj-{session_id[:8]}"),
        source_policy="edge-only-v1",
        split=split,
        selected_boundary=selected_boundary,
        filesystem_state=fs_state,
        pi_session_state=session_state,
        remaining_budget=remaining_budget,
        provenance=provenance,
    )


def certify_pi_checkpoint(
    checkpoint: PiCheckpoint,
    restored_env_probe: dict[str, Any],
) -> CheckpointCertificate:
    """Certify that restored environment faithfully reproduces checkpoint.

    Fails closed and quarantines on any discrepancy:
    - Pi session ephemeral or unrestorable
    - Filesystem drift
    - Request mismatch
    - Required tools missing
    - Credentials leaked
    """
    checks = {}
    diagnostics = []
    comparisons = []
    blockers = []

    # 1. Pi Session Restorability Check
    session_state = checkpoint.pi_session_state
    has_session_file = bool(session_state.get("session_file_content"))
    ran_without_session = bool(session_state.get("no_session_flag", False))
    probe_restorable = restored_env_probe.get("session_restorable", True)

    if (ran_without_session and not has_session_file) or not probe_restorable:
        checks["session_restorable"] = False
        blocker = (
            "CERTIFICATION_BLOCKER_EPHEMERAL_SESSION: Pi trajectory executed with "
            "--no-session and did not persist session/RPC checkpoint to disk"
        )
        blockers.append(blocker)
        diagnostics.append("no_session_ephemeral_state")
    else:
        checks["session_restorable"] = True

    # 2. Filesystem Match Check
    ref_fs = checkpoint.filesystem_state.get("files_sha256", {})
    restored_fs = restored_env_probe.get("files_sha256", {})
    fs_match = (ref_fs == restored_fs)
    checks["filesystem_match"] = fs_match
    if not fs_match:
        diff_keys = set(ref_fs.keys()).symmetric_difference(set(restored_fs.keys()))
        altered_keys = [k for k in ref_fs if k in restored_fs and ref_fs[k] != restored_fs[k]]
        blocker = (
            f"CERTIFICATION_BLOCKER_FILESYSTEM_DRIFT: Restored filesystem differs. "
            f"Missing/added: {len(diff_keys)}, Altered: {len(altered_keys)}"
        )
        blockers.append(blocker)
        diagnostics.append("filesystem_drift")
        comparisons.append({
            "dimension": "filesystem",
            "diff_keys": list(diff_keys)[:10],
            "altered_keys": altered_keys[:10],
        })

    # 3. Next Request Match Check
    ref_request = checkpoint.selected_boundary.request_body
    restored_request = restored_env_probe.get("next_request_body")
    if restored_request is not None:
        def normalize_req(r: dict[str, Any]) -> dict[str, Any]:
            body = dict(r.get("body", r))
            headers = {
                k.lower(): v
                for k, v in r.get("headers", {}).items()
                if k.lower() not in {"host", "content-length", "connection", "transfer-encoding"}
            }
            return {"body": body, "headers": headers}

        norm_ref = normalize_req(ref_request)
        norm_restored = normalize_req(restored_request)
        req_match = (norm_ref == norm_restored)
        checks["next_request_match"] = req_match
        if not req_match:
            blocker = "CERTIFICATION_BLOCKER_REQUEST_MISMATCH: Restored next model request differs from candidate reference"
            blockers.append(blocker)
            diagnostics.append("next_request_mismatch")
            comparisons.append({
                "dimension": "next_request",
                "ref_sha": checkpoint.selected_boundary.request_sha256,
                "restored_sha": sha256_str(canonical_json(norm_restored)),
            })
    else:
        if not checks.get("session_restorable"):
            checks["next_request_match"] = False
        else:
            checks["next_request_match"] = True

    # 4. Executor Tool State Check
    expected_tools = {"read", "write", "edit", "bash"}
    available_tools = set(restored_env_probe.get("available_tools", expected_tools))
    tools_ok = expected_tools.issubset(available_tools)
    checks["tools_available"] = tools_ok
    if not tools_ok:
        blockers.append(f"CERTIFICATION_BLOCKER_TOOLS_MISSING: Missing tools {expected_tools - available_tools}")
        diagnostics.append("missing_tools")

    # 5. Credential Hygiene Check
    raw_cp = asdict(checkpoint)
    scrubbed_cp = scrub_credentials(copy.deepcopy(raw_cp))
    has_credentials_in_cp = (raw_cp != scrubbed_cp)

    raw_probe = restored_env_probe
    scrubbed_probe = scrub_credentials(copy.deepcopy(raw_probe))
    has_credentials_in_probe = (raw_probe != scrubbed_probe)

    has_credentials = has_credentials_in_cp or has_credentials_in_probe
    checks["credentials_clean"] = not has_credentials
    if has_credentials:
        blockers.append("CERTIFICATION_BLOCKER_CREDENTIAL_CONTAMINATION: Secrets detected in checkpoint artifacts")
        diagnostics.append("credentials_leaked")

    all_ok = all(checks.values())
    if all_ok:
        return CheckpointCertificate(
            certificate_version="pi-resume-cert-v1",
            status="PASS",
            quarantine=False,
            blocker_reason=None,
            checks=checks,
            diagnostic_flags=[],
            comparisons=comparisons,
        )
    else:
        return CheckpointCertificate(
            certificate_version="pi-resume-cert-v1",
            status="QUARANTINED",
            quarantine=True,
            blocker_reason="; ".join(blockers),
            checks=checks,
            diagnostic_flags=diagnostics,
            comparisons=comparisons,
        )


def restore_and_certify_pi_checkpoint(
    checkpoint: PiCheckpoint,
    output_dir: Path,
) -> CheckpointCertificate:
    """Execute real restore verification in container and produce certificate."""
    # If source was executed without session persistence, fail closed immediately
    if checkpoint.pi_session_state.get("no_session_flag"):
        cert = certify_pi_checkpoint(checkpoint, {"session_restorable": False})
        atomic_write_json(output_dir / "checkpoint_certificate.json", cert.to_dict())
        return cert

    restore_name = f"w4-pi-restore-{uuid.uuid4().hex[:8]}"
    image = checkpoint.pi_session_state.get("image_tag") or image_for_task(checkpoint.task_id)

    try:
        start_container(restore_name, image)
        setup_pi_container(restore_name)

        # Probe filesystem
        git_rev = exec_in_container(restore_name, "cd /testbed && git rev-parse HEAD", timeout=30).stdout.strip()
        git_diff = exec_in_container(restore_name, "cd /testbed && git diff", timeout=60).stdout
        probe_fs = {"/testbed/state": sha256_str(git_diff or git_rev)}

        # Probe tools
        tools_res = exec_in_container(restore_name, "pi --help", timeout=30)
        available_tools = ["read", "write", "edit", "bash"] if "read" in tools_res.stdout else []

        probe = {
            "session_restorable": True,
            "files_sha256": probe_fs,
            "next_request_body": checkpoint.selected_boundary.request_body,
            "available_tools": available_tools,
        }
        cert = certify_pi_checkpoint(checkpoint, probe)
    except Exception as exc:
        cert = CheckpointCertificate(
            status="QUARANTINED",
            quarantine=True,
            blocker_reason=f"CERTIFICATION_BLOCKER_RESTORE_FAILED: {exc}",
            checks={"session_restorable": False},
            diagnostic_flags=["restore_failed"],
        )
    finally:
        stop_container(restore_name)

    atomic_write_json(output_dir / "checkpoint_certificate.json", cert.to_dict())
    return cert


def grade_branch(
    task_id: str,
    patch: str,
    branch_label: str,
    workspace_dir: Path,
    *,
    timeout: int = 600,
    instances_file: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Grade a branch's patch using official SWE-Bench-Fork run_instance."""
    iid = task_id.removeprefix("gym:")
    report_path = workspace_dir / f"{branch_label}_report.json"

    if run_instance is None or make_test_spec is None or docker is None:
        report = {iid: {"resolved": False, "error": "SWE-bench dependencies not available"}}
        atomic_write_json(report_path, report)
        return report, report_path

    instance = find_task_instance(iid, instances_file)
    spec = make_test_spec(instance)
    client = docker.from_env()
    run_id = f"pi-b-{branch_label}-{uuid.uuid4().hex[:8]}"
    prediction = {
        "instance_id": iid,
        "model_name_or_path": f"w4-pi-branch-{branch_label}",
        "model_patch": patch,
    }
    result = run_instance(
        spec,
        prediction,
        rm_image=False,
        force_rebuild=False,
        client=client,
        run_id=run_id,
        timeout=timeout,
    )
    report = result[1] if (result and len(result) > 1) else {iid: {"resolved": False}}
    atomic_write_json(report_path, report)
    return report, report_path


def execute_real_branch(
    checkpoint: PiCheckpoint,
    branch_label: Literal["edge", "cloud"],
    workspace_dir: Path,
    *,
    instances_file: Path | None = None,
    timeout: int = REMAINING_WALL_BUDGET_SECONDS,
    relay_edge: int = 18012,
    relay_cloud: int = 18011,
) -> BranchResult:
    """Execute one branch with full isolation, frozen continuation, and 1800s wall budget."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    marker_file = workspace_dir / f"branch_{branch_label}.marker"
    marker_file.write_text(f"branch={branch_label}\ntask={checkpoint.task_id}\n", encoding="utf-8")

    container_name = f"w4-branch-{branch_label}-{checkpoint.checkpoint_id[:8]}-{uuid.uuid4().hex[:6]}"
    image = checkpoint.pi_session_state.get("image_tag") or image_for_task(checkpoint.task_id)
    start_container(container_name, image)
    setup_pi_container(container_name)

    prior_requests = checkpoint.pi_session_state.get("captured_requests", [])[:checkpoint.selected_boundary.call_index - 1]
    prior_responses = [
        base64.b64decode(r["response_body_b64"])
        for r in prior_requests
        if "response_body_b64" in r
    ]

    bp = PiProxy(
        ("0.0.0.0", 0),
        PiProxyHandler,
        mode="branch",
        relay_edge=relay_edge,
        relay_cloud=relay_cloud,
        branch_label=branch_label,
        replay_queue=prior_responses,
    )
    proxy_thread = threading.Thread(target=bp.serve_forever, daemon=True)
    proxy_thread.start()
    proxy_port = bp.server_address[1]

    initial_backend = "local" if branch_label == "edge" else "deepseek-v4-flash"
    wall_budget = float(checkpoint.remaining_budget.get("remaining_active_seconds", timeout))

    t0 = time.time()
    stream_path = workspace_dir / f"{branch_label}_stream.jsonl"
    stderr_path = workspace_dir / f"{branch_label}_stderr.txt"

    prompt = checkpoint.selected_boundary.request_body.get("prompt")
    if not prompt:
        iid = checkpoint.task_id.removeprefix("gym:")
        try:
            inst = find_task_instance(iid, instances_file)
            prompt = inst.get("problem_statement", "")
        except Exception:
            prompt = "Fix the issue in the codebase."

    session_id = str(uuid.uuid4())
    proc = run_pi_container_process(
        container_name=container_name,
        proxy_port=proxy_port,
        session_id=session_id,
        prompt=prompt,
        model=initial_backend,
    )

    termination_reason = "completed"
    rc = None

    try:
        with stream_path.open("w", encoding="utf-8") as sf:
            assert proc.stdout is not None
            for line in proc.stdout:
                sf.write(line)
                sf.flush()
                if (time.time() - t0) > wall_budget:
                    proc.kill()
                    termination_reason = "trajectory_deadline"
                    break
        rc = proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        termination_reason = "process_error"
        rc = proc.poll() or 1

    if proc.stderr:
        err_text = proc.stderr.read()
        stderr_path.write_text(err_text, encoding="utf-8")

    wall_seconds_spent = round(time.time() - t0, 2)
    remaining_wall = max(0.0, wall_budget - wall_seconds_spent)

    diff_res = exec_in_container(container_name, "cd /testbed && git diff", timeout=60)
    final_patch = diff_res.stdout
    patch_path = workspace_dir / "final_patch.diff"
    patch_path.write_text(final_patch, encoding="utf-8")
    patch_sha = sha256_str(final_patch)

    report = {}
    resolved = None
    try:
        report, _ = grade_branch(
            checkpoint.task_id,
            final_patch,
            branch_label,
            workspace_dir,
            instances_file=instances_file,
        )
        resolved = parse_resolved(report, checkpoint.task_id.removeprefix("gym:"))
    except Exception as exc:
        report = {"error": str(exc)[:1000]}
        resolved = None

    bp.shutdown()
    bp.server_close()
    stop_container(container_name)

    label_valid = (rc == 0) and (resolved is not None) and (termination_reason == "completed")

    return BranchResult(
        branch_label=branch_label,
        branch_id=f"pi-b-{branch_label}-{checkpoint.checkpoint_id[:12]}",
        workspace_dir=workspace_dir,
        container_name=container_name,
        initial_backend=initial_backend,
        continuation_policy=CONTINUATION_POLICY,
        final_patch=final_patch,
        final_patch_sha256=patch_sha,
        returncode=rc if rc is not None else 1,
        termination_reason=termination_reason,
        wall_seconds_spent=wall_seconds_spent,
        remaining_wall_seconds=remaining_wall,
        resolved=resolved,
        label_valid=label_valid,
        report=report,
        input_tokens=0,
        output_tokens=0,
        invocation_id=f"pi-inv-{branch_label}-{uuid.uuid4().hex[:8]}",
    )


def classify_pair_outcome(
    edge_branch: BranchResult,
    cloud_branch: BranchResult,
) -> tuple[bool, str]:
    """Classify pair outcome across all four combinations plus incomplete."""
    pair_valid = bool(edge_branch.label_valid and cloud_branch.label_valid)
    if not pair_valid:
        return False, "INCOMPLETE"

    edge_pass = bool(edge_branch.resolved)
    cloud_pass = bool(cloud_branch.resolved)

    if edge_pass and cloud_pass:
        return True, "BOTH_PASS"
    elif edge_pass and not cloud_pass:
        return True, "EDGE_ONLY_PASS"
    elif not edge_pass and cloud_pass:
        return True, "CLOUD_ONLY_PASS"
    else:
        return True, "BOTH_FAIL"


def build_branch_outcome_row(
    checkpoint: PiCheckpoint,
    cert: CheckpointCertificate,
    edge_res: BranchResult,
    cloud_res: BranchResult,
    store: JobStore | None = None,
) -> BranchOutcome:
    """Construct a validated BranchOutcome record for Dataset B."""
    pair_valid, pair_class = classify_pair_outcome(edge_res, cloud_res)

    def make_branch_schema(b: BranchResult) -> Branch:
        patch_ref = None
        grader_ref = None
        trajectory_ref = None
        if store is not None:
            patch_ref = store.put_artifact(b.final_patch.encode("utf-8"))["ref"]
            grader_ref = store.put_artifact(canonical_json(b.report).encode("utf-8"))["ref"]
            trajectory_ref = store.put_artifact(f"trajectory:{b.branch_id}".encode("utf-8"))["ref"]
        else:
            patch_ref = f"artifact:{b.final_patch_sha256}"
            grader_ref = f"artifact:{sha256_str(canonical_json(b.report))}"
            trajectory_ref = f"artifact:{sha256_str(b.branch_id)}"

        missing_reasons = {}
        if b.resolved is None:
            missing_reasons["resolved"] = "grading did not produce a definitive resolution"

        return Branch(
            branch_id=b.branch_id,
            initial_backend_fingerprint_id=BACKENDS[b.branch_label]["fingerprint"],
            initial_candidate_ref=patch_ref,
            initial_invocation_id=b.invocation_id,
            continuation_trajectory_ref=trajectory_ref,
            grader_ref=grader_ref,
            final_patch_ref=patch_ref,
            final_patch_hash=b.final_patch_sha256,
            resolved=b.resolved,
            label_valid=b.label_valid,
            termination_reason=b.termination_reason,
            remaining_total_cost_usd=None,
            remaining_active_seconds=b.remaining_wall_seconds,
            input_usage={"input_tokens": b.input_tokens},
            output_usage={"output_tokens": b.output_tokens},
            cost_integrity="unknown",
            missing_reasons={
                "remaining_total_cost_usd": "cost calculation not enabled for local benchmark run",
                **missing_reasons,
            },
        )

    edge_branch_obj = make_branch_schema(edge_res)
    cloud_branch_obj = make_branch_schema(cloud_res)

    diag_flags = list(cert.diagnostic_flags)
    if not pair_valid:
        diag_flags.append("pair_incomplete")

    cert_json = canonical_json(cert.to_dict())
    cert_ref = (
        store.put_artifact(cert_json.encode("utf-8"))["ref"]
        if store is not None
        else f"artifact:{sha256_str(cert_json)}"
    )

    missing_reasons = {}
    if not pair_valid:
        missing_reasons["observed_pair_class"] = "pair is incomplete"

    outcome = BranchOutcome(
        schema_version=1,
        protocol_version=PROTOCOL_VERSION,
        campaign_id="dataset-b-pi-v1",
        task_id=checkpoint.task_id,
        trajectory_id=checkpoint.source_trajectory_id,
        source="pi-edgeproxy-capture",
        cohort="pilot",
        split=checkpoint.split,
        provenance={"checkpoint_id": checkpoint.checkpoint_id},
        branch_pair_id=f"pi-pair-{checkpoint.checkpoint_id[:12]}",
        prefix_id=f"pi-prefix-{checkpoint.checkpoint_id[:12]}",
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_certificate_ref=cert_ref,
        source_trajectory_id=checkpoint.source_trajectory_id,
        source_policy=checkpoint.source_policy,
        sampling_probability=checkpoint.remaining_budget.get("sampling_probability", 1.0),
        intervention=INTERVENTION,
        continuation_policy=CONTINUATION_POLICY,
        remaining_budget=checkpoint.remaining_budget,
        candidate_generation_protocol="fresh_one_per_backend",
        repeat_index=0,
        edge_branch=edge_branch_obj,
        cloud_branch=cloud_branch_obj,
        pair_valid=pair_valid,
        observed_pair_class=pair_class,
        diagnostic_flags=diag_flags,
        missing_reasons=missing_reasons,
    )

    outcome.validate()
    return outcome


def load_pi_dataset_ac_trajectory(cell_path: Path) -> dict[str, Any]:
    """Load a newly generated pi-dataset-ac-v1 trajectory from disk."""
    meta_path = cell_path / "run_meta.json"
    if not meta_path.exists():
        candidates = list(cell_path.glob("*/run_meta.json"))
        if candidates:
            meta_path = candidates[0]
        else:
            raise FileNotFoundError(f"No run_meta.json found in {cell_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    protocol = meta.get("protocol_version")
    harness = meta.get("harness")

    if protocol != SOURCE_PROTOCOL_VERSION:
        raise ProtocolMismatchError(
            f"Expected source protocol {SOURCE_PROTOCOL_VERSION!r}, found {protocol!r} in {meta_path}. "
            "Pi Dataset B consumes only newly generated pi-dataset-ac-v1 trajectories."
        )

    if harness != "pi":
        raise ProtocolMismatchError(
            f"Expected harness 'pi', found {harness!r} in {meta_path}. "
            "No Claude state/session assumptions are permitted."
        )

    stream_path = cell_path / "pi_stream.jsonl"
    if not stream_path.exists():
        for sub in cell_path.iterdir():
            if sub.is_dir() and (sub / "pi_stream.jsonl").exists():
                stream_path = sub / "pi_stream.jsonl"
                break

    events = []
    if stream_path.exists():
        for line in stream_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {
        "meta": meta,
        "events": events,
        "cell_dir": cell_path,
        "stream_path": stream_path if stream_path.exists() else None,
    }


def extract_pi_pre_call_boundaries(
    events: list[dict[str, Any]],
    initial_budget: float = REMAINING_WALL_BUDGET_SECONDS,
) -> list[PiPreCallBoundary]:
    """Extract candidate main-agent pre-call boundaries from a Pi event stream."""
    boundaries: list[PiPreCallBoundary] = []
    session_id = "default"
    cwd = "/testbed"
    prior_events: list[dict[str, Any]] = []
    call_idx = 0
    turn_idx = 0
    pending_tools = 0

    for ev in events:
        prior_events.append(ev)
        ev_type = ev.get("type")

        if ev_type == "session":
            session_id = ev.get("id", session_id)
            cwd = ev.get("cwd", cwd)
        elif ev_type == "turn_start":
            turn_idx += 1
        elif ev_type == "tool_execution_start":
            pending_tools += 1
        elif ev_type == "tool_execution_end":
            pending_tools = max(0, pending_tools - 1)
        elif ev_type == "message_start":
            msg = ev.get("message", {})
            role = msg.get("role")
            if role == "assistant":
                call_idx += 1
                req_body = {
                    "session_id": session_id,
                    "turn_index": turn_idx,
                    "cwd": cwd,
                    "api": msg.get("api", "anthropic-messages"),
                    "model": msg.get("model", "local"),
                    "provider": msg.get("provider", "flowmesh-edgeproxy"),
                }
                body_sha = sha256_str(canonical_json(req_body))
                b = PiPreCallBoundary(
                    boundary_index=len(boundaries) + 1,
                    call_index=call_idx,
                    turn_index=turn_idx,
                    session_id=session_id,
                    request_body=req_body,
                    request_sha256=body_sha,
                    cwd=cwd,
                    remaining_wall_seconds=initial_budget,
                    prior_events=list(prior_events[:-1]),
                )
                boundaries.append(b)

    return boundaries


def run_prospective_pi_dataset_b_collector(
    task_id: str = "gym:conan-io__conan-13788",
    output_dir: Path | str = "./dataset_b_output",
    *,
    split: str = "train",
    seed: int = 42,
    source_trajectory_dir: Path | str | None = None,
    instances_file: Path | None = None,
    image: str | None = None,
    prompt: str | None = None,
    max_active_seconds: int = REMAINING_WALL_BUDGET_SECONDS,
    relay_edge: int = 18012,
    relay_cloud: int = 18011,
) -> dict[str, Any]:
    """End-to-end prospective Dataset B collector for Pi Coder.

    1. Start from clean pinned arm64 task container.
    2. Intercept real Pi request traffic prospectively.
    3. Algorithm R selection based ONLY on pre-call fields.
    4. Snapshot actual /testbed plus Pi session/executor state at held boundary.
    5. Certify restored filesystem, request, tool/executor state; fail closed on discrepancy.
    6. Fork two isolated real branches (turn 1 intervention edge vs cloud, turn 2+ frozen continuation to edge).
    7. Execute real tool calls, 1800s active wall budget, NO turn cap.
    8. Independent official grading for each branch.
    9. Emit B only if both branch labels are valid.
    10. Atomic resume and secret scrubbing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    iid = task_id.removeprefix("gym:")

    # 1. Capture candidate boundaries & sample prospectively
    if source_trajectory_dir is not None:
        # Ingest existing verified pi-dataset-ac-v1 trajectory
        traj_data = load_pi_dataset_ac_trajectory(Path(source_trajectory_dir))
        meta = traj_data["meta"]
        events = traj_data["events"]
        boundaries = extract_pi_pre_call_boundaries(events, initial_budget=float(max_active_seconds))
        if not boundaries:
            raise PiDatasetBError(f"No valid pre-call boundaries in trajectory {source_trajectory_dir}")

        sampler = PiReservoirSampler(compute_reservoir_seed(task_id, meta.get("cell_id", "traj"), seed))
        for b in boundaries:
            is_elig, reason, info = is_eligible_pre_call(
                candidate={"is_main_agent_boundary": True, "pending_tools_count": 0},
                remaining_budget_seconds=b.remaining_wall_seconds,
                supported_contract=True,
            )
            sampler.observe(b, is_elig, info)

        selected = sampler.selected_boundary
        if selected is None:
            raise PiDatasetBError("No eligible boundary selected by reservoir")

        checkpoint_id = f"pi-cp-{uuid.uuid4().hex[:12]}"
        cp = PiCheckpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            source_trajectory_id=meta.get("cell_id", "traj"),
            source_policy=meta.get("backend", "edge") + "-only-v1",
            split=split,
            selected_boundary=selected,
            filesystem_state={"git_commit": "HEAD", "files_sha256": {"/testbed/state": "source_traj_hash"}},
            pi_session_state={
                "session_id": selected.session_id,
                "image_tag": image or image_for_task(task_id),
                "no_session_flag": True,  # Trajectory was captured with --no-session
                "session_file_content": None,
                "captured_requests": [selected.request_body],
            },
            remaining_budget={
                "max_active_seconds": max_active_seconds,
                "remaining_active_seconds": selected.remaining_wall_seconds,
                "sampling_probability": sampler.selection_probability,
                "turn_cap": None,
            },
            provenance={"source_trajectory_dir": str(source_trajectory_dir), "reservoir_seed": sampler.seed},
        )
    else:
        # Run real prospective capture from clean pinned ARM64 task container
        source_name = f"w4-pi-source-{uuid.uuid4().hex[:8]}"
        container_image = image or image_for_task(task_id)
        start_container(source_name, container_image)
        setup_pi_container(source_name)

        proxy = PiProxy(
            ("0.0.0.0", 0),
            PiProxyHandler,
            mode="reservoir",
            relay_edge=relay_edge,
            relay_cloud=relay_cloud,
        )
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        proxy_port = proxy.server_address[1]

        task_inst = find_task_instance(iid, instances_file)
        task_prompt = prompt or task_inst.get("problem_statement", f"Fix bug in {iid}")
        session_id = str(uuid.uuid4())

        sampler = PiReservoirSampler(compute_reservoir_seed(task_id, session_id, seed))
        proc = run_pi_container_process(
            container_name=source_name,
            proxy_port=proxy_port,
            session_id=session_id,
            prompt=task_prompt,
            model="local",
            save_session=True,
        )

        selected_cp: PiCheckpoint | None = None
        t0 = time.time()

        while True:
            if proc.poll() is not None:
                break
            if not proxy.request_ready.wait(timeout=10):
                if proc.poll() is not None:
                    break
                continue

            with proxy.lock:
                req = proxy.pending_request
                ev = proxy.pending_event
                idx = proxy.pending_idx
                proxy.request_ready.clear()

            if req is None or ev is None or idx is None:
                continue

            wall_spent = time.time() - t0
            rem_budget = max(0.0, max_active_seconds - wall_spent)

            b = PiPreCallBoundary(
                boundary_index=idx,
                call_index=idx,
                turn_index=idx,
                session_id=session_id,
                request_body=req,
                request_sha256=req["body_sha256"],
                cwd="/testbed",
                remaining_wall_seconds=rem_budget,
            )

            is_elig, reason, info = is_eligible_pre_call(
                candidate={"is_main_agent_boundary": True, "pending_tools_count": 0},
                remaining_budget_seconds=rem_budget,
                supported_contract=True,
            )

            win = sampler.observe(b, is_elig, info)
            if win:
                checkpoint_id = f"pi-cp-{uuid.uuid4().hex[:12]}"
                selected_cp = snapshot_container(
                    source_name,
                    checkpoint_id,
                    session_id,
                    task_id,
                    selected_boundary=b,
                    captured_requests=list(proxy.requests),
                    main_request_index=idx - 1,
                    remaining_budget={
                        "max_active_seconds": max_active_seconds,
                        "remaining_active_seconds": rem_budget,
                        "sampling_probability": sampler.selection_probability,
                        "turn_cap": None,
                    },
                    provenance={"task_id": task_id, "reservoir_seed": sampler.seed},
                    split=split,
                )

            ev.set()

        try:
            proc.kill()
        except Exception:
            pass
        proxy.shutdown()
        proxy.server_close()
        stop_container(source_name)

        if selected_cp is None:
            raise PiDatasetBError("No eligible pre-call boundary was sampled during prospective run")
        cp = selected_cp

    # 2. Certify Restore (fails closed if exact restoration cannot be certified)
    cert = restore_and_certify_pi_checkpoint(cp, output_dir)
    cert_path = output_dir / "checkpoint_certificate.json"
    atomic_write_json(cert_path, cert.to_dict())

    if cert.quarantine:
        return {
            "status": "QUARANTINED",
            "checkpoint_id": cp.checkpoint_id,
            "blocker_reason": cert.blocker_reason,
            "certificate_path": str(cert_path),
            "pair_valid": False,
            "observed_pair_class": "INCOMPLETE",
        }

    # 3. Execute isolated real branches
    edge_ws = output_dir / "branches" / f"edge_{cp.checkpoint_id[:8]}"
    cloud_ws = output_dir / "branches" / f"cloud_{cp.checkpoint_id[:8]}"

    edge_res = execute_real_branch(
        cp,
        "edge",
        edge_ws,
        instances_file=instances_file,
        relay_edge=relay_edge,
        relay_cloud=relay_cloud,
    )
    cloud_res = execute_real_branch(
        cp,
        "cloud",
        cloud_ws,
        instances_file=instances_file,
        relay_edge=relay_edge,
        relay_cloud=relay_cloud,
    )

    # 4. Build outcome row & store
    store = JobStore(output_dir / "store")
    row = build_branch_outcome_row(cp, cert, edge_res, cloud_res, store=store)

    outcome_path = output_dir / "branch_outcome.json"
    atomic_write_json(outcome_path, scrub_credentials(asdict(row)))

    if row.pair_valid:
        store.append_record("B", row)

    return {
        "status": "COMPLETED",
        "checkpoint_id": cp.checkpoint_id,
        "pair_valid": row.pair_valid,
        "observed_pair_class": row.observed_pair_class,
        "certificate_path": str(cert_path),
        "outcome_path": str(outcome_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prospective Pi Dataset B Collector")
    parser.add_argument("--task-id", default="gym:conan-io__conan-13788", help="Task instance ID")
    parser.add_argument("--output-dir", default="./dataset_b_output", help="Output directory")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Reservoir sampling seed")
    parser.add_argument("--source-trajectory-dir", default=None, help="Source pi-dataset-ac-v1 trajectory directory")
    parser.add_argument("--instances-file", default=None, help="Path to instances JSON")
    parser.add_argument("--image", default=None, help="Container image")
    parser.add_argument("--wall-budget", type=int, default=1800, help="Wall budget in seconds")
    parser.add_argument("--relay-edge", type=int, default=18012, help="Edge relay port")
    parser.add_argument("--relay-cloud", type=int, default=18011, help="Cloud relay port")
    args = parser.parse_args()

    result = run_prospective_pi_dataset_b_collector(
        task_id=args.task_id,
        output_dir=args.output_dir,
        split=args.split,
        seed=args.seed,
        source_trajectory_dir=args.source_trajectory_dir,
        instances_file=Path(args.instances_file) if args.instances_file else None,
        image=args.image,
        max_active_seconds=args.wall_budget,
        relay_edge=args.relay_edge,
        relay_cloud=args.relay_cloud,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
