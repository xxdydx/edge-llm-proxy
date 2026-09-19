#!/usr/bin/env python3
"""One guarded, local-only replay of a saved Matplotlib model request.

No code or tools from the response are executed. Raw request/stream stay in
gitignored traces. This is a model-behavior diagnostic, not a task replay.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from jsonschema import validate as validate_schema

from edgeproxy.trace.record import reassemble


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "traces/swebench-pro-15-run-routing-20260917T012600Z-swebench-matplotlib-c968b9e065-seed1/2026-09-17.jsonl"
PRIVATE = ROOT / "traces/offline-no-think-matplotlib-call-4e873d1c-v1"
PUBLIC = ROOT / "experiments/agentic_router/results/offline_no_think_matplotlib_call_v1.json"
PROTOCOL = ROOT / "experiments/agentic_router/results/offline_no_think_matplotlib_call_v1_protocol.md"
LOCK = ROOT / "experiments/agentic_router/results/pilot_campaign.lock"
CALL_ID = "4e873d1c-0842-48d9-a074-d143c31c3271"
SOURCE_SHA = "1cb33af74bcadaf38216412f5e4563a887843698e758bb855fb5a1b204a30495"
ORIGINAL_SHA = "29cc9f6f106ec57e34869f424bf5177d76912dc0d7b8f7200ba1bba3ff01f1f8"
MODIFIED_SHA = "7a42c88067576d89c894f2a458489acdb58b251e9a709bfa72b50fd965387c2e"
URL = "http://127.0.0.1:18004/v1/messages"
MAX_WALL_S = 180.0


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def gauges() -> tuple[float, float]:
    with httpx.Client(timeout=10.0) as client:
        models = client.get("http://127.0.0.1:18004/v1/models")
        metrics = client.get("http://127.0.0.1:18004/metrics")
    models.raise_for_status()
    metrics.raise_for_status()
    data = models.json().get("data", [])
    if len(data) != 1 or data[0].get("id") != "local" or data[0].get("root") != "Inferact/Qwen3.8-27B-NVFP4" or data[0].get("max_model_len") != 100000:
        raise RuntimeError("served_model_identity_changed")
    values = []
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        hits = [float(line.split()[-1]) for line in metrics.text.splitlines()
                if line.startswith(name + "{")]
        if len(hits) != 1:
            raise RuntimeError("missing_or_ambiguous_metric:" + name)
        values.append(hits[0])
    return values[0], values[1]


def exclusive_private(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())


def worker() -> int:
    request = json.loads((PRIVATE / "request.json").read_text())
    if canonical_sha(request) != MODIFIED_SHA:
        raise RuntimeError("private_request_hash_mismatch")
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", URL, json=request, headers={"content-type": "application/json"}) as response:
            with (PRIVATE / "response.sse").open("xb") as out:
                for chunk in response.iter_bytes():
                    out.write(chunk)
                    out.flush()
            exclusive_private(PRIVATE / "http_status.json", json.dumps({"status": response.status_code}).encode())
    return 0


def parse_stream(raw: bytes, request: dict) -> dict:
    events = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: "):
            continue
        value = line[6:]
        if value == "[DONE]":
            continue
        try:
            event = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    message, usage = reassemble(events)
    blocks = message.get("content") or []
    offered = {tool.get("name"): tool.get("input_schema") for tool in request["tools"]}
    tool_validity = []
    for block in blocks:
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        input_value = block.get("input")
        valid = isinstance(input_value, dict) and name in offered
        if valid:
            try:
                validate_schema(input_value, offered[name])
            except Exception:
                valid = False
        tool_validity.append({"offered_name": name in offered, "schema_valid": valid})
    return {
        "complete_message_stop": any(event.get("type") == "message_stop" for event in events),
        "stop_reason": message.get("stop_reason"),
        "content_types": [block.get("type") for block in blocks],
        "tool_validity": tool_validity,
        "input_tokens": usage.get("input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker()
    if not args.execute:
        parser.error("--execute required; no call made")
    if PRIVATE.exists() or PUBLIC.exists() or not PROTOCOL.is_file():
        raise RuntimeError("private_or_public_output_exists_or_protocol_missing")
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("source_trace_hash_changed")
    matches = [row for row in (json.loads(line) for line in SOURCE.open())
               if row.get("call", {}).get("call_id") == CALL_ID]
    if len(matches) != 1:
        raise RuntimeError("saved_call_not_unique")
    original = matches[0]["request"]
    if canonical_sha(original) != ORIGINAL_SHA:
        raise RuntimeError("saved_request_hash_changed")
    if (original.get("model") != "local" or original.get("max_tokens") != 32000
            or original.get("temperature") != 0 or original.get("stream") is not True
            or original.get("output_config") != {"effort": "medium"}
            or len(original.get("tools") or []) != 22
            or "chat_template_kwargs" in original):
        raise RuntimeError("saved_request_contract_changed")
    request = {**original, "chat_template_kwargs": {"enable_thinking": False}}
    if canonical_sha(request) != MODIFIED_SHA:
        raise RuntimeError("modified_request_hash_changed")
    lock = LOCK.open("r+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("campaign_lock_busy") from exc
    if gauges() != (0.0, 0.0):
        raise RuntimeError("local_engine_not_idle_before_call")

    os.umask(0o077)
    PRIVATE.mkdir(mode=0o700)
    exclusive_private(PRIVATE / "request.json", json.dumps(request, ensure_ascii=False).encode())
    started = time.monotonic()
    with (PRIVATE / "worker.stderr").open("xb") as stderr:
        proc = subprocess.Popen([sys.executable, str(Path(__file__)), "--worker"],
                                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=stderr,
                                start_new_session=True)
        timed_out = False
        try:
            code = proc.wait(timeout=MAX_WALL_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait(timeout=5)
    elapsed = time.monotonic() - started
    drained = False
    final_gauges = None
    for _ in range(15):
        try:
            final_gauges = gauges()
        except Exception:
            final_gauges = None
        if final_gauges == (0.0, 0.0):
            drained = True
            break
        time.sleep(1)
    status_path = PRIVATE / "http_status.json"
    http_status = json.loads(status_path.read_text()).get("status") if status_path.is_file() else None
    stream_path = PRIVATE / "response.sse"
    parsed = parse_stream(stream_path.read_bytes(), request) if stream_path.is_file() else {}
    result = {
        "schema_version": "offline-no-think-saved-call-v1",
        "source_call_id": CALL_ID,
        "source_trace_sha256": SOURCE_SHA,
        "original_request_sha256": ORIGINAL_SHA,
        "modified_request_sha256": MODIFIED_SHA,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "only_request_change": "chat_template_kwargs.enable_thinking=false",
        "endpoint": URL,
        "wall_s": round(elapsed, 3),
        "watchdog_timeout": timed_out,
        "worker_exit_code": code,
        "http_status": http_status,
        "engine_drained": drained,
        "final_running_waiting": final_gauges,
        **parsed,
    }
    exclusive_private(PUBLIC, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result, sort_keys=True))
    lock.close()
    return 0 if not timed_out and code == 0 and drained and http_status == 200 and parsed.get("complete_message_stop") else 2


if __name__ == "__main__":
    raise SystemExit(main())
