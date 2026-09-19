#!/usr/bin/env python3
"""Guarded local-only saved-call ablation; never executes emitted tools."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from jsonschema import validate as validate_schema

from edgeproxy.trace.record import SSEDecoder, reassemble
from experiments.capability_router.config import LOCAL_27B
from experiments.capability_router.executor import _sanitize_for_backend


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "experiments/agentic_router/results"
PROTOCOL = BASE / "local_no_thinking_v6_v8_preoutcome_protocol_20260917.md"
AMENDMENT = BASE / "local_no_thinking_v6_v8_preoutcome_amendment_v1_1_20260917.md"
PUBLIC = BASE / "local_no_thinking_v6_v8_results_v1_1.json"
PRIVATE = ROOT / "traces/local-no-thinking-v6-v8-v1-1"
LOCK = BASE / "pilot_campaign.lock"
SGT = dt.timezone(dt.timedelta(hours=8))
DRAIN = dt.datetime(2026, 9, 17, 16, 30, tzinfo=SGT)
LATEST_START = dt.datetime(2026, 9, 17, 15, 54, tzinfo=SGT)
GLOBAL_CAP_S = 36 * 60
PER_CALL_CAP_S = 160
COUNT_OFFSET = 2
ENDPOINT = "http://127.0.0.1:18004"
MODEL_ROOT = "Inferact/Qwen3.8-27B-NVFP4"
MODEL_SNAPSHOT = "6128240ebaf4eaa7bad2b3d1c72c37d677c5f462"
TEMPLATE_SHA = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
TRANSFORM_SHA = {
    "experiments/capability_router/executor.py": "be166cd4be014e35a08ae12defb41274604c205b99713f459e9bc4021715ab70",
    "edgeproxy/server.py": "1a6fb17f43e43c00a7c43b6fde3d9a379c4be81b9caf625b2aef6dde64ce95cb",
}
REPLAY_SHA = {
    6: "28807891c493fec4580aaa07d3a6a874e868754746d0d2925bdd0ad63fd2f6e8",
    7: "8c2cb36f3038657215f20e6b1a3fafc427d95e271da6c10963f4dd034e39206a",
    8: "4c7d4f91a4e830ca2c3adabd85b49eacaffe83980635e320b841098b50caad98",
}
SOURCE_SHA = {
    "swebench-xarray-13e7d5e1cb": "322d75a43e88522a9afd44ea1cc68a40660e36bdfaca999679211c95559915bb",
    "swebench-seaborn-3ca9f75c90": "bbd2a8cb0d45ddaa5afffa5dfa53f46454ca0839041f7874a2939ebfe6cd9630",
    "swebench-scikit-learn-a4a59eac2d": "71a7ffd84aa8f4764b199b861bcb5070f24c71d102d37877ea93e391dd762656",
    "swebench-pylint-7cebb03cec": "f7c727f3ffe5b8578eabf48e9d01edc372f85ae75c7d5232d1e979a103dd0d51",
    "swebench-sympy-3b20aad08b": "9c9b6b174cc287de927de85b18cd11179bc23d8f23cea5797cae8a80cd495264",
    "swebench-pytest-18946c8bcb": "4cd4609cea1060eb4b02df1419861e340bae58a28671effe1f7b73dab89b9818",
}
# (call_id, task_group, stratum, v6/v7/v8), frozen before this ablation.
SELECTION = (
    ("6f599993-dca6-4604-9cce-5f50c61c3b24", "swebench-xarray-13e7d5e1cb", "ordinary", 7),
    ("b5f0b9ce-fb77-4020-92fb-d7f03152a315", "swebench-seaborn-3ca9f75c90", "tail", 6),
    ("b0b1c0c1-1a54-40b7-a4ad-073b6b9b5909", "swebench-scikit-learn-a4a59eac2d", "ordinary", 8),
    ("418083f0-30b2-4334-a9ec-45049670f726", "swebench-pylint-7cebb03cec", "tail", 7),
    ("c2666183-4ca2-4a7d-bcb9-78c84d8b965b", "swebench-sympy-3b20aad08b", "ordinary", 7),
    ("f50a7d97-7029-48f8-918f-9897e08cd390", "swebench-pytest-18946c8bcb", "tail", 7),
    ("1bdba237-54d7-4bf8-a0dc-8f5758c70b62", "swebench-xarray-13e7d5e1cb", "tail", 7),
    ("bce854bc-a93d-4027-8614-43adfd118e67", "swebench-seaborn-3ca9f75c90", "ordinary", 6),
    ("59ea24d7-3491-46e6-9d19-660759de7473", "swebench-scikit-learn-a4a59eac2d", "tail", 8),
    ("5078490d-86f1-4031-846a-502fb7ec515c", "swebench-pylint-7cebb03cec", "ordinary", 7),
    ("f9d5841f-8b47-4a01-895c-1847a29c99f1", "swebench-sympy-3b20aad08b", "tail", 7),
    ("e86bf390-919f-46ab-82f8-efe153b90760", "swebench-pytest-18946c8bcb", "ordinary", 7),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _now() -> dt.datetime:
    return dt.datetime.now(SGT)


def _safe_write(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def _checkpoint(state: dict[str, Any]) -> None:
    temp = PUBLIC.with_name(f".{PUBLIC.name}.{os.getpid()}.tmp")
    with temp.open("w") as out:
        json.dump(state, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, PUBLIC)


def _identity_and_idle(timeout_s: float = 10) -> None:
    with httpx.Client(timeout=timeout_s) as client:
        models = client.get(ENDPOINT + "/v1/models")
        metrics = client.get(ENDPOINT + "/metrics")
    models.raise_for_status()
    metrics.raise_for_status()
    rows = models.json().get("data", [])
    if (len(rows) != 1 or rows[0].get("id") != "local"
            or rows[0].get("root") != MODEL_ROOT
            or rows[0].get("max_model_len") != 100000):
        raise RuntimeError("model_identity_changed")
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        values = [float(line.split()[-1]) for line in metrics.text.splitlines()
                  if line.startswith(name + "{")]
        if len(values) != 1 or values[0] != 0:
            raise RuntimeError("local_queue_busy_or_missing:" + name)


def _no_other_work() -> None:
    proc = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=10)
    if proc.returncode != 0 or proc.stdout.strip():
        raise RuntimeError("other_docker_work_or_unavailable")
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10)
    if ps.returncode != 0:
        raise RuntimeError("process_inventory_unavailable")
    needles = ("run_swebench.py", "pilot_live.py", "pilot_campaign.py", "live_matrix.py", "claude -p")
    for line in ps.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) != os.getpid() and any(n in parts[1] for n in needles):
            raise RuntimeError("other_task_process_active")


def _frozen_hashes() -> None:
    if not PROTOCOL.is_file() or not AMENDMENT.is_file():
        raise RuntimeError("protocol_or_amendment_missing")
    for rel, expected in TRANSFORM_SHA.items():
        if _sha(ROOT / rel) != expected:
            raise RuntimeError("transform_hash_changed:" + rel)
    for version, expected in REPLAY_SHA.items():
        path = BASE / f"postpilot_v{version}/stage1_replay_examples_pilot_v1.jsonl"
        if _sha(path) != expected:
            raise RuntimeError(f"replay_hash_changed:v{version}")


def _count(payload: dict[str, Any]) -> int:
    with httpx.Client(timeout=20) as client:
        response = client.post(ENDPOINT + "/v1/messages/count_cached_tokens", json=payload)
    response.raise_for_status()
    value = response.json().get("input_tokens")
    if type(value) is not int or value <= 0:
        raise RuntimeError("exact_token_count_unavailable")
    return value


def _load_selection() -> list[dict[str, Any]]:
    _frozen_hashes()
    ids = {item[0] for item in SELECTION}
    replay_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in ids}
    checkpoint_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in ids}
    for version in REPLAY_SHA:
        root = BASE / f"postpilot_v{version}"
        for line in (root / "stage1_replay_examples_pilot_v1.jsonl").open():
            row = json.loads(line)
            call = row.get("call") or {}
            if call.get("call_id") in ids:
                replay_rows[call["call_id"]].append({"version": version, "row": row})
        for path in (root / "pilot_arm_checkpoints").glob("*.json"):
            row = json.loads(path.read_text())
            if row.get("backend") == "local_27b" and row.get("call_id") in ids:
                checkpoint_rows[row["call_id"]].append({"version": version, "row": row})
    source_cache: dict[Path, list[dict[str, Any]]] = {}
    prepared = []
    for order, (call_id, task_group, stratum, version) in enumerate(SELECTION, 1):
        if len(replay_rows[call_id]) != 1 or len(checkpoint_rows[call_id]) != 1:
            raise RuntimeError("nonunique_replay_or_checkpoint:" + call_id)
        replay = replay_rows[call_id][0]
        checkpoint = checkpoint_rows[call_id][0]
        if replay["version"] != version or checkpoint["version"] != version:
            raise RuntimeError("version_assignment_changed:" + call_id)
        row = replay["row"]
        call = row["call"]
        old = checkpoint["row"]
        source = Path(call["source_trace_path"])
        if call.get("task_group") != task_group or _sha(source) != SOURCE_SHA[task_group]:
            raise RuntimeError("task_or_source_hash_mismatch:" + call_id)
        source_cache.setdefault(source, [json.loads(line) for line in source.open()])
        matching = [r for r in source_cache[source] if (r.get("call") or {}).get("call_id") == call_id]
        if len(matching) != 1 or matching[0].get("request") != call.get("request"):
            raise RuntimeError("source_request_parity_failed:" + call_id)
        if row.get("local_outcome", {}).get("status") != "OK" or old.get("outcome", {}).get("status") != "OK":
            raise RuntimeError("original_local_not_ok:" + call_id)
        response = old["outcome"].get("response") or {}
        if response.get("stop_reason") != "tool_use" or not any(
            b.get("type") == "thinking" for b in response.get("content") or []
        ):
            raise RuntimeError("original_stratum_response_changed:" + call_id)
        original_output = (response.get("_usage") or {}).get("output_tokens")
        original_wall = old["outcome"].get("latency_s")
        if (stratum == "ordinary" and not (original_wall <= 10 and original_output <= 250)
                or stratum == "tail" and not (original_wall >= 25 and original_output >= 700)):
            raise RuntimeError("original_stratum_boundary_changed:" + call_id)
        payload, transform = _sanitize_for_backend(LOCAL_27B, call["request"])
        if (transform.get("strict_tools_added") != 22
                or len(payload.get("tools") or []) != 22
                or payload.get("model") != "local" or payload.get("max_tokens") != 32000
                or payload.get("temperature") != 0 or payload.get("stream") is not True
                or payload.get("output_config") != {"effort": "medium"}
                or "chat_template_kwargs" in payload):
            raise RuntimeError("local_native_transform_changed:" + call_id)
        offered_names = [tool.get("name") for tool in payload["tools"]]
        if offered_names != call.get("tool_names") or any(
            tool.get("strict") is not True or not isinstance(tool.get("input_schema"), dict)
            for tool in payload["tools"]
        ):
            raise RuntimeError("tool_names_or_schema_changed:" + call_id)
        precheck = old.get("transform", {}).get("capacity_precheck") or {}
        if (precheck.get("budget") != 90000 or precheck.get("fits") is not True
                or precheck.get("detail") != "fits"
                or old.get("transform", {}).get("capacity_clamp") is not None
                or type(precheck.get("input_tokens")) is not int):
            raise RuntimeError("historical_capacity_precheck_changed:" + call_id)
        modified = {**payload, "chat_template_kwargs": {"enable_thinking": False}}
        control = dict(modified)
        if control.pop("chat_template_kwargs") != {"enable_thinking": False} or control != payload:
            raise RuntimeError("nonadditive_request_change:" + call_id)
        prepared.append({"order": order, "call_id": call_id, "task_group": task_group,
                         "stratum": stratum, "source": source, "original": payload,
                         "modified": modified, "historical_input_tokens": precheck["input_tokens"],
                         "original_wall_s": original_wall,
                         "original_ttft_s": (response.get("_timing") or {}).get("ttft_s"),
                         "original_output_tokens": original_output,
                         "original_tool_names": [b.get("name") for b in response.get("content") or []
                                                 if b.get("type") == "tool_use"]})
    return prepared


def _capacity_parity(item: dict[str, Any]) -> tuple[int, int]:
    old = _count(item["original"])
    new = _count(item["modified"])
    historical = item["historical_input_tokens"]
    if old != historical + COUNT_OFFSET or new != old + COUNT_OFFSET:
        raise RuntimeError("count_offset_not_exactly_frozen_pattern:" + item["call_id"])
    if new + 32000 > 90000:
        raise RuntimeError("current_exact_capacity_does_not_fit:" + item["call_id"])
    return old, new


def _parse(raw: bytes, payload: dict[str, Any]) -> dict[str, Any]:
    decoder = SSEDecoder()
    events = decoder.feed(raw) + decoder.finish()
    message, usage = reassemble(events)
    offered = {t["name"]: t["input_schema"] for t in payload["tools"]}
    tools = []
    for block in message.get("content") or []:
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        valid = name in offered and isinstance(block.get("input"), dict)
        if valid:
            try:
                validate_schema(block["input"], offered[name])
            except Exception:
                valid = False
        tools.append({"name": name, "offered": name in offered, "schema_valid": valid})
    return {"stream_complete": any(e.get("type") == "message_stop" for e in events),
            "response_model": message.get("model"),
            "stop_reason": message.get("stop_reason"),
            "content_types": [b.get("type") for b in message.get("content") or []],
            "tools": tools,
            "output_tokens": usage.get("output_tokens"),
            "input_tokens": usage.get("input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens")}


def _worker(private_call: Path) -> int:
    request = json.loads((private_call / "request.json").read_text())
    started = time.monotonic()
    ttft = None
    decoder = SSEDecoder()
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", ENDPOINT + "/v1/messages", json=request,
                           headers={"content-type": "application/json", "anthropic-version": "2023-06-01"}) as resp:
            _safe_write(private_call / "http_status.json", json.dumps({"http_status": resp.status_code}).encode())
            with (private_call / "response.sse").open("xb") as out:
                for chunk in resp.iter_bytes():
                    out.write(chunk)
                    out.flush()
                    for event in decoder.feed(chunk):
                        if event.get("type") == "content_block_delta" and ttft is None:
                            ttft = time.monotonic() - started
            _safe_write(private_call / "worker.json", json.dumps({"ttft_s": ttft}).encode())
    return 0


def _drain() -> bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            remaining = max(0.2, deadline - time.monotonic())
            with httpx.Client(timeout=min(2, remaining)) as client:
                metrics = client.get(ENDPOINT + "/metrics")
            metrics.raise_for_status()
            values = []
            for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
                matches = [float(line.split()[-1]) for line in metrics.text.splitlines()
                           if line.startswith(name + "{")]
                if len(matches) != 1:
                    raise RuntimeError("drain_metric_missing")
                values.append(matches[0])
            if values == [0.0, 0.0]:
                return True
        except (httpx.HTTPError, RuntimeError):
            pass
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    return False


def _one(item: dict[str, Any], counts: tuple[int, int]) -> dict[str, Any]:
    number = item["order"]
    private_call = PRIVATE / f"{number:02d}-{item['call_id']}"
    private_call.mkdir(mode=0o700)
    _safe_write(private_call / "request.json", json.dumps(item["modified"], ensure_ascii=False).encode())
    started = time.monotonic()
    with (private_call / "worker.stderr").open("xb") as stderr:
        proc = subprocess.Popen([sys.executable, str(Path(__file__)), "--worker", str(private_call)],
                                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=stderr,
                                start_new_session=True,
                                env={k: v for k, v in os.environ.items()
                                     if k not in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")})
        timeout = False
        try:
            code = proc.wait(timeout=PER_CALL_CAP_S)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait(timeout=5)
    wall = time.monotonic() - started
    drained = _drain()
    meta = json.loads((private_call / "worker.json").read_text()) if (private_call / "worker.json").is_file() else {}
    status = json.loads((private_call / "http_status.json").read_text()) if (private_call / "http_status.json").is_file() else {}
    parsed = _parse((private_call / "response.sse").read_bytes(), item["modified"]) if (private_call / "response.sse").is_file() else {}
    return {"order": number, "call_id": item["call_id"], "task_group": item["task_group"],
            "stratum": item["stratum"], "original_wall_s": item["original_wall_s"],
            "original_ttft_s": item["original_ttft_s"],
            "original_output_tokens": item["original_output_tokens"],
            "original_tool_names": item["original_tool_names"],
            "historical_input_tokens": item["historical_input_tokens"],
            "current_original_input_tokens": counts[0],
            "current_modified_input_tokens": counts[1],
            "original_native_body_sha256": _canonical_sha(item["original"]),
            "modified_body_sha256": _canonical_sha(item["modified"]),
            "http_status": status.get("http_status"), "ttft_s": meta.get("ttft_s"),
            "wall_s": round(wall, 3), "watchdog_timeout": timeout,
            "worker_exit_code": code, "engine_drained": drained, **parsed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        return _worker(args.worker)
    if not args.execute:
        parser.error("--execute required; no model call made")
    if _now() >= LATEST_START:
        raise RuntimeError("latest_start_cutoff_passed")
    if PUBLIC.exists() or PRIVATE.exists():
        raise RuntimeError("fresh_output_namespace_required")
    os.umask(0o077)
    started = time.monotonic()
    with LOCK.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("campaign_lock_busy") from exc
        _no_other_work()
        _identity_and_idle()
        prepared = _load_selection()
        # All 12 requests must pass reconstruction and the amended exact
        # current-count pattern BEFORE the first generation.
        initial_counts = [_capacity_parity(item) for item in prepared]
        PRIVATE.mkdir(mode=0o700)
        state: dict[str, Any] = {
            "schema_version": "local-no-thinking-v6-v8-v1.1",
            "state": "running", "started_at_sgt": _now().isoformat(),
            "protocol_sha256": _sha(PROTOCOL), "amendment_sha256": _sha(AMENDMENT),
            "controller_sha256": _sha(Path(__file__)),
            "replay_sha256": REPLAY_SHA, "source_sha256": SOURCE_SHA,
            "transform_sha256": TRANSFORM_SHA,
            "model": {"served_id": "local", "root": MODEL_ROOT, "max_model_len": 100000,
                      "snapshot": MODEL_SNAPSHOT, "chat_template_sha256": TEMPLATE_SHA,
                      "vllm_version": "0.0.0-patched"},
            "selection": [{"order": item["order"], "call_id": item["call_id"],
                           "task_group": item["task_group"], "stratum": item["stratum"],
                           "historical_input_tokens": item["historical_input_tokens"],
                           "current_original_input_tokens": initial_counts[i][0],
                           "current_modified_input_tokens": initial_counts[i][1]}
                          for i, item in enumerate(prepared)],
            "completed": [], "stop_reason": None,
        }
        _checkpoint(state)
        try:
            for item in prepared:
                if (_now() + dt.timedelta(seconds=PER_CALL_CAP_S + 20) >= DRAIN
                        or time.monotonic() - started + PER_CALL_CAP_S + 20 > GLOBAL_CAP_S):
                    raise RuntimeError("insufficient_remaining_watchdog_budget")
                _frozen_hashes()
                if _sha(item["source"]) != SOURCE_SHA[item["task_group"]]:
                    raise RuntimeError("source_hash_changed_before_call")
                _no_other_work()
                _identity_and_idle()
                counts = _capacity_parity(item)
                if counts != initial_counts[item["order"] - 1]:
                    raise RuntimeError("count_changed_since_initial_preflight")
                state["active_order"] = item["order"]
                state["active_call_id"] = item["call_id"]
                _checkpoint(state)
                result = _one(item, counts)
                state["completed"].append(result)
                state.pop("active_order", None)
                state.pop("active_call_id", None)
                _checkpoint(state)
                print(f"completed {item['order']}/12 {item['stratum']} status={result.get('http_status')} stop={result.get('stop_reason')} wall_s={result['wall_s']} drained={result['engine_drained']}", flush=True)
                if not result["engine_drained"]:
                    raise RuntimeError("engine_failed_to_drain_after_call")
                if result["watchdog_timeout"]:
                    if result.get("http_status") not in (None, 200):
                        raise RuntimeError("local_endpoint_rejected_flag")
                    continue
                if (result.get("http_status") != 200 or result.get("response_model") != "local"
                        or result["worker_exit_code"] != 0 or not result.get("stream_complete")):
                    raise RuntimeError("local_endpoint_incompatible_or_identity_changed")
            state["state"] = "complete"
            state["finished_at_sgt"] = _now().isoformat()
            _checkpoint(state)
            return 0
        except Exception as exc:
            state["state"] = "stopped_invalid"
            state["stop_reason"] = f"{type(exc).__name__}:{exc}"
            state["finished_at_sgt"] = _now().isoformat()
            _checkpoint(state)
            print(state["stop_reason"], file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
