#!/usr/bin/env python3
"""Measure proxy-side cloud-link shaping with controlled A/B requests.

The runner starts one isolated cloud-only edgeproxy per configured delay,
randomizes requests across conditions, preserves each proxy's raw JSONL trace,
and writes sample and summary CSVs.  Two upstream modes are supported:

* ``stub``: a local deterministic Anthropic-SSE stub; validates timing
  boundaries and arithmetic without GPU access, cloud variance, or API cost.
* ``cloud``: the configured real Anthropic-compatible upstream; measures the
  end-to-end effect under realistic network and provider variance.

The live proxy currently computes ``server_ttft_ms = ttft_ms - network_ms``.
Because shaping happens before the measured HTTP phases, the script also emits
``effective_network_ms = network_ms + shaped_ms`` and
``corrected_server_ttft_ms = ttft_ms - effective_network_ms``.  Keeping both
values makes the known accounting problem visible instead of silently fixing
historical trace fields during analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

import httpx

from edgeproxy.config import DEFAULT_UPSTREAM


@dataclass
class ProxyCondition:
    delay_ms: float
    port: int
    trace_dir: Path
    log_path: Path
    process: subprocess.Popen[bytes]
    client: httpx.Client


def comma_floats(value: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("provide one or more non-negative numbers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("delay conditions must be unique")
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def corrected_timing(record: dict[str, Any]) -> dict[str, float | None]:
    timing = record.get("timing") or {}
    link = record.get("link") or {}
    network = timing.get("network_ms")
    shaped = link.get("shaped_ms") or 0.0
    ttft = timing.get("ttft_ms")
    effective = network + shaped if network is not None else None
    corrected = ttft - effective if ttft is not None and effective is not None else None
    return {
        "effective_network_ms": round(effective, 3) if effective is not None else None,
        "corrected_server_ttft_ms": round(corrected, 3) if corrected is not None else None,
    }


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.median(values) if values else None


def summarize(rows: list[dict[str, Any]], tolerance_ms: float) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["configured_delay_ms"]), []).append(row)

    medians: dict[float, dict[str, float | None]] = {}
    keys = (
        "shaped_ms",
        "ttft_ms",
        "total_ms",
        "transport_network_ms",
        "effective_network_ms",
        "raw_server_ttft_ms",
        "corrected_server_ttft_ms",
        "response_headers_ms",
    )
    for delay, group in grouped.items():
        valid = [row for row in group if row.get("valid") is True]
        medians[delay] = {key: _median(valid, key) for key in keys}

    baseline = medians.get(0.0)
    output: list[dict[str, Any]] = []
    for delay, group in sorted(grouped.items()):
        valid = [row for row in group if row.get("valid") is True]
        current = medians[delay]

        def delta(key: str) -> float | None:
            if not baseline or baseline[key] is None or current[key] is None:
                return None
            return float(current[key]) - float(baseline[key])

        ttft_delta = delta("ttft_ms")
        shaped_delta = delta("shaped_ms")
        corrected_delta = delta("corrected_server_ttft_ms")
        output.append(
            {
                "configured_delay_ms": delay,
                "valid_samples": len(valid),
                "total_samples": len(group),
                "median_shaped_ms": current["shaped_ms"],
                "median_ttft_ms": current["ttft_ms"],
                "p10_ttft_ms": percentile(
                    [float(row["ttft_ms"]) for row in valid], 0.1
                ) if valid else None,
                "p90_ttft_ms": percentile(
                    [float(row["ttft_ms"]) for row in valid], 0.9
                ) if valid else None,
                "median_total_ms": current["total_ms"],
                "median_transport_network_ms": current["transport_network_ms"],
                "median_effective_network_ms": current["effective_network_ms"],
                "median_raw_server_ttft_ms": current["raw_server_ttft_ms"],
                "median_corrected_server_ttft_ms": current["corrected_server_ttft_ms"],
                "median_response_headers_ms": current["response_headers_ms"],
                "delta_ttft_vs_zero_ms": ttft_delta,
                "delta_shaped_vs_zero_ms": shaped_delta,
                "delta_raw_server_vs_zero_ms": delta("raw_server_ttft_ms"),
                "delta_corrected_server_vs_zero_ms": corrected_delta,
                "ttft_tracks_shaping": (
                    abs(ttft_delta - shaped_delta) <= tolerance_ms
                    if ttft_delta is not None and shaped_delta is not None
                    else None
                ),
                "corrected_server_stable": (
                    abs(corrected_delta) <= tolerance_ms
                    if corrected_delta is not None
                    else None
                ),
            }
        )
    return output


class StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LinkShapingStub/1"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        delay_ms = getattr(self.server, "response_delay_ms", 0.0)
        if delay_ms:
            time.sleep(delay_ms / 1000)

        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stub",
                    "type": "message",
                    "role": "assistant",
                    "model": "stub",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 8, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "PONG"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        body = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, format: str, *args: Any) -> None:
        return


def start_stub(response_delay_ms: float) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    server.response_delay_ms = response_delay_ms  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(url: str, process: subprocess.Popen[bytes], timeout_s: float = 15) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"edgeproxy exited early with status {process.returncode}")
        try:
            if httpx.get(url, timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"edgeproxy did not become healthy at {url}")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def trace_records(trace_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def wait_for_record(trace_dir: Path, prior_count: int, timeout_s: float = 5) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        records = trace_records(trace_dir)
        if len(records) > prior_count:
            return records[-1]
        time.sleep(0.05)
    raise RuntimeError(f"trace record was not written under {trace_dir}")


def auth_headers(mode: str) -> dict[str, str]:
    headers = {"anthropic-version": "2023-06-01"}
    if mode == "stub":
        return headers
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if token:
        headers["authorization"] = f"Bearer {token}"
    elif api_key:
        headers["x-api-key"] = api_key
    else:
        raise SystemExit(
            "cloud mode needs ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY in the environment"
        )
    return headers


def start_proxy(
    delay_ms: float,
    upstream: str,
    trace_root: Path,
    jitter_ms: float,
    bandwidth_mbps: float,
    stack: ExitStack,
) -> ProxyCondition:
    label = str(delay_ms).replace(".", "p")
    trace_dir = trace_root / f"delay-{label}ms"
    trace_dir.mkdir(parents=True, exist_ok=True)
    log_path = trace_root / f"proxy-{label}ms.log"
    log_handle = stack.enter_context(log_path.open("wb"))
    port = free_port()
    command = [
        sys.executable,
        "-m",
        "edgeproxy.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--upstream",
        upstream,
        "--vllm-url",
        "http://127.0.0.1:9",
        "--trace-dir",
        str(trace_dir),
        "--policy",
        "cloud-only",
        "--shaping",
        "proxy",
        "--link-preset",
        "none",
        "--cloud-delay-ms",
        str(delay_ms),
        "--cloud-jitter-ms",
        str(jitter_ms),
        "--cloud-bandwidth-mbps",
        str(bandwidth_mbps),
    ]
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
    stack.callback(stop_process, process)
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_health(f"{base_url}/health", process)
    except Exception:
        log_handle.flush()
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        stop_process(process)
        raise RuntimeError(f"could not start proxy for {delay_ms} ms:\n{detail}")
    return ProxyCondition(
        delay_ms=delay_ms,
        port=port,
        trace_dir=trace_dir,
        log_path=log_path,
        process=process,
        client=stack.enter_context(httpx.Client(base_url=base_url, timeout=600.0)),
    )


def request_once(
    condition: ProxyCondition,
    payload: dict[str, Any],
    headers: dict[str, str],
    repetition: int,
    sequence: int,
    mode: str,
) -> dict[str, Any]:
    before = len(trace_records(condition.trace_dir))
    client_started = time.monotonic()
    with condition.client.stream("POST", "/v1/messages", json=payload, headers=headers) as response:
        response_body = b"".join(response.iter_bytes())
        status = response.status_code
    client_total_ms = (time.monotonic() - client_started) * 1000
    record = wait_for_record(condition.trace_dir, before)
    timing = record.get("timing") or {}
    link = record.get("link") or {}
    corrected = corrected_timing(record)
    ttft = timing.get("ttft_ms")
    valid = (
        status == 200
        and record.get("status") == 200
        and record.get("placement") == "cloud"
        and ttft is not None
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "sequence": sequence,
        "repetition": repetition,
        "configured_delay_ms": condition.delay_ms,
        "configured_jitter_ms": link.get("jitter_ms"),
        "configured_bandwidth_mbps": link.get("bandwidth_mbps"),
        "request_bytes": len(json.dumps(payload).encode()),
        "response_bytes": len(response_body),
        "placement": record.get("placement"),
        "status": record.get("status"),
        "shaped_ms": link.get("shaped_ms") or 0.0,
        "ttft_ms": ttft,
        "total_ms": timing.get("total_ms"),
        "client_total_ms": round(client_total_ms, 3),
        "transport_network_ms": timing.get("network_ms"),
        "effective_network_ms": corrected["effective_network_ms"],
        "raw_server_ttft_ms": timing.get("server_ttft_ms"),
        "corrected_server_ttft_ms": corrected["corrected_server_ttft_ms"],
        "connect_ms": timing.get("connect_ms"),
        "tls_ms": timing.get("tls_ms"),
        "send_ms": timing.get("send_ms"),
        "response_headers_ms": timing.get("response_headers_ms"),
        "valid": valid,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stub", "cloud"), default="stub")
    parser.add_argument("--delays-ms", type=comma_floats, default=comma_floats("0,80"))
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--bandwidth-mbps", type=float, default=0.0)
    parser.add_argument("--stub-server-delay-ms", type=float, default=10.0)
    parser.add_argument("--upstream", help=f"cloud upstream (default: {DEFAULT_UPSTREAM})")
    parser.add_argument(
        "--model",
        default=os.environ.get("LINK_BENCH_MODEL", "claude-haiku-4-5-20251001"),
    )
    parser.add_argument("--prompt", default="Reply with exactly: PONG")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--tolerance-ms", type=float, default=25.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trace-root",
        type=Path,
        help="raw JSONL/log directory (default: gitignored traces/link-shaping/<run>)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if 0.0 not in args.delays_ms:
        raise SystemExit("--delays-ms must include 0 as the baseline")
    if args.repetitions <= 0 or args.warmups < 0:
        raise SystemExit("repetitions must be positive and warmups non-negative")
    if args.jitter_ms < 0 or args.bandwidth_mbps < 0 or args.tolerance_ms < 0:
        raise SystemExit("jitter, bandwidth, and tolerance must be non-negative")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("results") / f"link-shaping-{args.mode}-{stamp}.csv"
    summary_path = output.with_name(f"{output.stem}-summary.csv")
    trace_root = args.trace_root or Path("traces") / "link-shaping" / output.stem
    occupied = [path for path in (output, summary_path) if path.exists()]
    if trace_root.exists() and any(trace_root.iterdir()):
        occupied.append(trace_root)
    if occupied:
        raise SystemExit(
            "refusing to mix with existing results: "
            + ", ".join(str(path) for path in occupied)
            + "; choose a new --output path"
        )
    trace_root.mkdir(parents=True, exist_ok=True)

    stub: ThreadingHTTPServer | None = None
    if args.mode == "stub":
        stub, _ = start_stub(args.stub_server_delay_ms)
        upstream = f"http://127.0.0.1:{stub.server_port}"
    else:
        upstream = args.upstream or DEFAULT_UPSTREAM

    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "stream": True,
        "messages": [{"role": "user", "content": args.prompt}],
    }
    headers = auth_headers(args.mode)
    rows: list[dict[str, Any]] = []

    try:
        with ExitStack() as stack:
            conditions = {
                delay: start_proxy(
                    delay,
                    upstream,
                    trace_root,
                    args.jitter_ms,
                    args.bandwidth_mbps,
                    stack,
                )
                for delay in args.delays_ms
            }
            for condition in conditions.values():
                for warmup in range(args.warmups):
                    request_once(
                        condition, payload, headers, -(warmup + 1), -1, args.mode
                    )

            schedule = [
                (delay, repetition)
                for repetition in range(args.repetitions)
                for delay in args.delays_ms
            ]
            random.Random(args.seed).shuffle(schedule)
            for sequence, (delay, repetition) in enumerate(schedule, 1):
                row = request_once(
                    conditions[delay], payload, headers, repetition, sequence, args.mode
                )
                rows.append(row)
                print(
                    f"[{sequence:>3}/{len(schedule)}] delay={delay:>6g}ms "
                    f"shaped={row['shaped_ms']!s:>6} "
                    f"ttft={row['ttft_ms']!s:>7} "
                    f"raw_server={row['raw_server_ttft_ms']!s:>7} "
                    f"corrected_server={row['corrected_server_ttft_ms']!s:>7} "
                    f"valid={row['valid']}"
                )
    finally:
        if stub is not None:
            stub.shutdown()
            stub.server_close()

    summaries = summarize(rows, args.tolerance_ms)
    write_csv(output, rows)
    write_csv(summary_path, summaries)
    print(f"samples: {output}")
    print(f"summary: {summary_path}")
    print(f"raw traces and proxy logs: {trace_root}")
    for row in summaries:
        print(
            f"delay={row['configured_delay_ms']:g}ms valid="
            f"{row['valid_samples']}/{row['total_samples']} "
            f"delta_ttft={row['delta_ttft_vs_zero_ms']} "
            f"delta_shaped={row['delta_shaped_vs_zero_ms']} "
            f"delta_corrected_server={row['delta_corrected_server_vs_zero_ms']} "
            f"tracks={row['ttft_tracks_shaping']} "
            f"server_stable={row['corrected_server_stable']}"
        )
    return 0 if all(row["valid_samples"] == row["total_samples"] for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
