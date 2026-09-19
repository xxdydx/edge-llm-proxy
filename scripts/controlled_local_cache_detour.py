"""Bounded, opt-in local prefix-cache detour microprobe.

This is a mechanism probe, not an agent-task evaluation. It never resets the
global cache and never sends tools. Importing this module makes no requests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import httpx


class ProbeInvalid(RuntimeError):
    """A measurement cannot be interpreted as a matched cache comparison."""


@dataclass(frozen=True)
class Config:
    local_model: str
    effort: str
    pairs: int = 2
    seed: int = 17
    interval_s: float = 30.0
    kv_tolerance: float = 0.05
    deadline: dt.datetime = dt.datetime(2026, 9, 17, 16, 30, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    run_id: str = ""


def _body(cfg: Config, prompt: str) -> bytes:
    payload = {
        "model": cfg.local_model,
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
        "output_config": {"effort": cfg.effort},
        "messages": [{"role": "user", "content": prompt}],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _prompt() -> str:
    # Synthetic and non-sensitive; deterministic vocabulary avoids repetitive
    # blocks while keeping the request identical across all arms.
    rng = random.Random(749203)
    syllables = ("amber", "basil", "cedar", "delta", "ember", "fable", "glade", "harbor", "ivory", "juniper")
    words = [syllables[rng.randrange(len(syllables))] for _ in range(4600)]
    return "Return only the word OK. The following is inert synthetic context: " + " ".join(words)


def _orders(pairs: int, seed: int) -> list[list[str]]:
    if pairs not in (2, 3):
        raise ValueError("only 2 or 3 matched pairs are permitted")
    first = "idle" if random.Random(seed).randrange(2) == 0 else "cloud"
    second = "cloud" if first == "idle" else "idle"
    return [[first, second] if i % 2 == 0 else [second, first] for i in range(pairs)]


def _remaining(cfg: Config, now: Callable[[], dt.datetime]) -> float:
    seconds = (cfg.deadline - now()).total_seconds()
    if seconds <= 0:
        raise ProbeInvalid("drain_deadline_reached")
    return seconds


def _check_request_budget(cfg: Config, now: Callable[[], dt.datetime], maximum_s: float = 25.0) -> float:
    remaining = _remaining(cfg, now)
    if remaining <= maximum_s + 2:
        raise ProbeInvalid("insufficient_drain_deadline_budget")
    return min(maximum_s, remaining - 2)


def _metric_value(raw: str, name: str) -> float:
    values: list[float] = []
    for line in raw.splitlines():
        part = line.split("#", 1)[0].strip().split()
        if len(part) != 2 or part[0].split("{", 1)[0] != name:
            continue
        try:
            value = float(part[1])
        except ValueError as exc:
            raise ProbeInvalid(f"invalid_metric:{name}") from exc
        if not math.isfinite(value):
            raise ProbeInvalid(f"nonfinite_metric:{name}")
        values.append(value)
    if not values:
        raise ProbeInvalid(f"missing_metric:{name}")
    return sum(values)


def _metrics(client: httpx.Client, cfg: Config, now: Callable[[], dt.datetime]) -> dict[str, float]:
    response = client.get("/metrics", timeout=_check_request_budget(cfg, now, 8))
    response.raise_for_status()
    raw = response.text
    running = _metric_value(raw, "vllm:num_requests_running")
    waiting = _metric_value(raw, "vllm:num_requests_waiting")
    try:
        kv = _metric_value(raw, "vllm:gpu_cache_usage_perc")
    except ProbeInvalid as exc:
        if str(exc) != "missing_metric:vllm:gpu_cache_usage_perc":
            raise
        kv = _metric_value(raw, "vllm:kv_cache_usage_perc")
    if running != 0 or waiting != 0:
        raise ProbeInvalid(f"local_not_idle:running={running},waiting={waiting}")
    if not 0 <= kv <= 1:
        raise ProbeInvalid("invalid_kv_occupancy")
    return {"running": running, "waiting": waiting, "kv_fraction": kv}


def _count(client: httpx.Client, body: bytes, cfg: Config, now: Callable[[], dt.datetime]) -> dict[str, int]:
    response = client.post(
        "/v1/messages/count_cached_tokens", content=body,
        headers={"content-type": "application/json"}, timeout=_check_request_budget(cfg, now, 8),
    )
    response.raise_for_status()
    data = response.json()
    input_tokens, cached_tokens = data.get("input_tokens"), data.get("cached_tokens")
    if (type(input_tokens) is not int or type(cached_tokens) is not int
            or input_tokens <= 0 or not 0 <= cached_tokens <= input_tokens):
        raise ProbeInvalid("missing_or_invalid_count_cached_tokens")
    return {"input_tokens": input_tokens, "cached_tokens": cached_tokens}


def _stream(client: httpx.Client, body: bytes, cfg: Config,
            monotonic: Callable[[], float], now: Callable[[], dt.datetime]) -> dict[str, Any]:
    started = monotonic()
    ttft_ms: float | None = None
    complete = False
    usage: dict[str, Any] = {}
    model: str | None = None
    stop_reason: str | None = None
    with client.stream(
        "POST", "/v1/messages", content=body,
        headers={"content-type": "application/json"}, timeout=_check_request_budget(cfg, now),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            _remaining(cfg, now)
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                continue
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "message_start":
                message = event.get("message") or {}
                if isinstance(message, dict):
                    model = message.get("model")
                    usage.update(message.get("usage") or {})
            elif kind == "message_delta":
                usage.update(event.get("usage") or {})
                delta = event.get("delta") or {}
                if isinstance(delta, dict) and isinstance(delta.get("stop_reason"), str):
                    stop_reason = delta["stop_reason"]
            elif kind == "content_block_delta" and ttft_ms is None:
                ttft_ms = (monotonic() - started) * 1000
            elif kind == "message_stop":
                complete = True
    if not complete or ttft_ms is None:
        raise ProbeInvalid("local_stream_incomplete_or_no_content_delta")
    required = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
    if any(type(usage.get(name)) is not int or usage[name] < 0 for name in required):
        raise ProbeInvalid("missing_or_invalid_final_usage_bucket")
    if stop_reason is None:
        raise ProbeInvalid("missing_stop_reason")
    return {
        "ttft_ms": round(ttft_ms, 3), "elapsed_s": round(monotonic() - started, 3),
        "cache_read_input_tokens": usage["cache_read_input_tokens"],
        "cache_creation_input_tokens": usage["cache_creation_input_tokens"],
        "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
        "total_input_tokens": sum(usage[name] for name in required[:3]),
        "stop_reason": stop_reason, "message_stop": True, "model": model,
    }


def _require_count_usage_parity(count: dict[str, int], generation: dict[str, Any]) -> None:
    if count["input_tokens"] != generation["total_input_tokens"]:
        raise ProbeInvalid(
            f"count_usage_input_mismatch:count={count['input_tokens']},"
            f"generation={generation['total_input_tokens']}"
        )


def _cloud_detour(client: httpx.Client, cfg: Config, prompt: str,
                  now: Callable[[], dt.datetime]) -> dict[str, Any]:
    # The answer is deliberately discarded: no agent state or local prompt changes.
    body = json.dumps({
        "model": "deepseek-v4-flash", "max_tokens": 1, "temperature": 0,
        "messages": [{"role": "user", "content": "Reply OK. " + prompt[:48]}],
    }, sort_keys=True, separators=(",", ":")).encode()
    response = client.post(
        "/v1/messages", content=body, headers={"content-type": "application/json"},
        timeout=_check_request_budget(cfg, now),
    )
    response.raise_for_status()
    data = response.json()
    if data.get("model") != "deepseek-v4-flash":
        raise ProbeInvalid("cloud_model_identity_mismatch_or_missing")
    if not isinstance(data.get("content"), list):
        raise ProbeInvalid("cloud_response_incomplete")
    return {"status_code": response.status_code, "model": data["model"], "request_sha256": hashlib.sha256(body).hexdigest()}


def run_probe(
    local: httpx.Client, cloud: httpx.Client, cfg: Config, *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if cfg.pairs not in (2, 3) or cfg.interval_s <= 0 or not cfg.run_id:
        raise ValueError("2–3 pairs, positive interval and unique run_id required")
    # Conservative capacity check before a single request: each arm may take
    # two 25s local streams, one 25s cloud call, and the fixed interval.
    required = cfg.pairs * 2 * (cfg.interval_s + 75) + 30
    if _remaining(cfg, now) < required:
        raise ProbeInvalid("insufficient_full_probe_deadline_budget")
    prompt = _prompt()
    result: dict[str, Any] = {
        "schema_version": "controlled-local-cache-detour-v1.1", "status": "running",
        "config": {**asdict(cfg), "deadline": cfg.deadline.isoformat()},
        "orders": _orders(cfg.pairs, cfg.seed), "pairs": [],
        "note": "Mechanism probe only; no task-quality or significance inference.",
    }
    if checkpoint is not None:
        checkpoint(result)
    for pair_index, order in enumerate(result["orders"]):
        pair: dict[str, Any] = {"index": pair_index, "arms": [], "valid": False}
        result["pairs"].append(pair)
        try:
            for condition in order:
                # The pinned count endpoint silently drops cache_salt. Put a
                # unique nonce at the START of user content instead, so the
                # count and generation render the same unsalted token stream.
                arm_prefix = hashlib.sha256(f"{cfg.run_id}:{pair_index}:{condition}".encode()).hexdigest()
                body = _body(cfg, f"Isolation ID {arm_prefix}. {prompt}")
                arm: dict[str, Any] = {
                    "condition": condition, "arm_prefix_sha256": hashlib.sha256(arm_prefix.encode()).hexdigest(),
                    "local_body_sha256": hashlib.sha256(body).hexdigest(), "status": "running",
                }
                pair["arms"].append(arm)
                arm["metrics_before"] = _metrics(local, cfg, now)
                arm["count_cold"] = _count(local, body, cfg, now)
                if arm["count_cold"]["cached_tokens"] != 0:
                    raise ProbeInvalid("fresh_prefix_not_cold")
                arm["L0"] = _stream(local, body, cfg, monotonic, now)
                _require_count_usage_parity(arm["count_cold"], arm["L0"])
                completed_at = monotonic()
                arm["count_warm"] = _count(local, body, cfg, now)
                if arm["count_warm"]["cached_tokens"] <= 0:
                    raise ProbeInvalid("L0_did_not_warm_prefix")
                if condition == "cloud":
                    arm["cloud"] = _cloud_detour(cloud, cfg, prompt, now)
                    arm["cloud_finished_elapsed_s"] = monotonic() - completed_at
                    if arm["cloud_finished_elapsed_s"] >= cfg.interval_s:
                        raise ProbeInvalid("cloud_detour_exceeded_interval")
                # Leave time for pre-L1 gauges/probe, then start L1 at t+30s.
                target = completed_at + cfg.interval_s
                sleep(max(0, target - 2 - monotonic()))
                arm["metrics_pre_L1"] = _metrics(local, cfg, now)
                arm["count_pre_L1"] = _count(local, body, cfg, now)
                sleep(max(0, target - monotonic()))
                arm["L0_to_L1_start_s"] = monotonic() - completed_at
                if abs(arm["L0_to_L1_start_s"] - cfg.interval_s) > 1:
                    raise ProbeInvalid("unmatched_L0_to_L1_interval")
                if abs(arm["metrics_before"]["kv_fraction"] - arm["metrics_pre_L1"]["kv_fraction"]) > cfg.kv_tolerance:
                    raise ProbeInvalid("within_arm_kv_occupancy_drift")
                arm["L1"] = _stream(local, body, cfg, monotonic, now)
                _require_count_usage_parity(arm["count_pre_L1"], arm["L1"])
                arm["metrics_after"] = _metrics(local, cfg, now)
                arm["L0_L1_bytes_equal"] = True  # exactly the same immutable bytes object passed twice
                arm["status"] = "complete"
                if checkpoint is not None:
                    checkpoint(result)
            idle, detour = (next(a for a in pair["arms"] if a["condition"] == c) for c in ("idle", "cloud"))
            if abs(idle["metrics_pre_L1"]["kv_fraction"] - detour["metrics_pre_L1"]["kv_fraction"]) > cfg.kv_tolerance:
                raise ProbeInvalid("between_arm_kv_occupancy_mismatch")
            pair["valid"] = True
            pair["difference_cloud_minus_idle"] = {
                "cached_probe_tokens": detour["count_pre_L1"]["cached_tokens"] - idle["count_pre_L1"]["cached_tokens"],
                "actual_cache_read_tokens": detour["L1"]["cache_read_input_tokens"] - idle["L1"]["cache_read_input_tokens"],
                "ttft_ms": round(detour["L1"]["ttft_ms"] - idle["L1"]["ttft_ms"], 3),
            }
            if checkpoint is not None:
                checkpoint(result)
        except (ProbeInvalid, httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            pair["invalid_reason"] = f"{type(exc).__name__}:{exc}"
            if pair["arms"] and pair["arms"][-1]["status"] == "running":
                pair["arms"][-1]["status"] = "invalid"
            result["status"] = "invalid_stopped"
            if checkpoint is not None:
                checkpoint(result)
            return result
    result["status"] = "complete"
    if checkpoint is not None:
        checkpoint(result)
    return result


class AtomicCheckpoint:
    """Reserve one fresh output, then atomically replace durable snapshots."""

    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.tmp = path.with_name(f".{path.name}.{run_id}.tmp")
        self.last_result: dict[str, Any] = {
            "schema_version": "controlled-local-cache-detour-v1.1",
            "status": "reserved_not_started", "run_id": run_id,
        }
        # O_EXCL is important: an earlier run's results must never be replaced.
        with path.open("x") as handle:
            json.dump(self.last_result, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_parent()

    def _fsync_parent(self) -> None:
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def __call__(self, result: dict[str, Any]) -> None:
        # This file contains only synthetic prompts' hashes and measurements;
        # the cloud bearer token is never included in the result object.
        with self.tmp.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(self.tmp, self.path)
        self._fsync_parent()
        self.last_result = json.loads(json.dumps(result))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-live", action="store_true", help="required opt-in; no dry-run network")
    parser.add_argument("--local-url", required=True)
    parser.add_argument("--cloud-url", required=True)
    parser.add_argument("--local-model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--cloud-token-env", default="ANTHROPIC_AUTH_TOKEN")
    parser.add_argument("--lock-file", type=Path, required=True, help="existing campaign-coordinated lock path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, choices=(2, 3), default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--drain-deadline", default="2026-09-17T16:30:00+08:00")
    args = parser.parse_args()
    if not args.execute_live:
        parser.error("no network was used; add --execute-live only after parent approval and campaign quiescence")
    if not args.lock_file.is_file():
        parser.error("lock file must already exist; coordinate its exact path with campaign")
    if args.output.exists():
        parser.error("output exists; refusing overwrite")
    token = os.environ.get(args.cloud_token_env)
    if not token:
        parser.error(f"missing cloud credential in {args.cloud_token_env}; not sent or logged")
    deadline = dt.datetime.fromisoformat(args.drain_deadline)
    if deadline.tzinfo is None:
        parser.error("drain deadline must have timezone offset")
    cfg = Config(args.local_model, args.effort, args.pairs, args.seed,
                 deadline=deadline, run_id=uuid.uuid4().hex)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("campaign lock busy; no requests sent")
        checkpoint = AtomicCheckpoint(args.output, cfg.run_id)
        with httpx.Client(base_url=args.local_url.rstrip("/")) as local, httpx.Client(
            base_url=args.cloud_url.rstrip("/"), headers={"authorization": f"Bearer {token}"}
        ) as cloud:
            try:
                result = run_probe(local, cloud, cfg, checkpoint=checkpoint)
            except Exception as exc:
                result = dict(checkpoint.last_result)
                result["status"] = "invalid_stopped"
                result["fatal_error"] = f"{type(exc).__name__}:{exc}"
                checkpoint(result)
    print(f"{result['status']}: {args.output}")
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
