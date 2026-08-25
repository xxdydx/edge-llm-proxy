#!/usr/bin/env python3
"""Validate edgeproxy's cloud-cache shadow with real Anthropic-backed calls.

The runner starts an isolated cloud-only edgeproxy in observe mode, sends a
controlled workload through it, and compares pre-request predictions with the
provider's cache usage.  Live calls are impossible without --confirm-live and
are bounded by --max-input-token-budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgeproxy.config import DEFAULT_UPSTREAM
from edgeproxy.trace.record import SSEDecoder, reassemble


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def auth_headers(cache_diagnostics: bool = False) -> dict[str, str]:
    headers = {"anthropic-version": "2023-06-01"}
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if token:
        headers["authorization"] = f"Bearer {token}"
    elif api_key:
        headers["x-api-key"] = api_key
    else:
        raise SystemExit("set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY")
    if cache_diagnostics:
        headers["anthropic-beta"] = "cache-diagnosis-2026-04-07"
    return headers


def comma_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_health(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"edgeproxy exited with {process.returncode}")
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError("edgeproxy did not become healthy")


def trace_records(trace_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def wait_for_record(trace_dir: Path, count: int, benchmark_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        records = trace_records(trace_dir)
        for record in records[count:]:
            if (record.get("headers") or {}).get("x-flowmesh-benchmark-id") == benchmark_id:
                return record
        time.sleep(0.05)
    raise RuntimeError("proxy trace was not written")


def start_proxy(
    upstream: str, trace_dir: Path, log_path: Path, stack: ExitStack
) -> tuple[str, httpx.Client]:
    port = free_port()
    log_handle = stack.enter_context(log_path.open("wb"))
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
        "none",
        "--cloud-cache-tracking",
        "observe",
    ]
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
    stack.callback(stop_process, process)
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_health(f"{base_url}/health", process)
    except Exception:
        log_handle.flush()
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"could not start edgeproxy:\n{detail}")
    client = stack.enter_context(httpx.Client(base_url=base_url, timeout=600.0))
    return base_url, client


def output_instruction() -> str:
    return "Output the word CACHE exactly 64 times, separated by single spaces."


def system_block(text: str, ttl: str = "5m", cache: bool = True) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": text}
    if cache:
        control: dict[str, str] = {"type": "ephemeral"}
        if ttl != "5m":
            control["ttl"] = ttl
        block["cache_control"] = control
    return block


def payload(
    model: str,
    prefix: str,
    *,
    suffix: str = "",
    ttl: str = "5m",
    blocks_after: int = 0,
    automatic: bool = False,
    diagnostics_previous_id: str | None = None,
) -> dict[str, Any]:
    first = system_block(prefix, ttl=ttl, cache=blocks_after == 0 and not automatic)
    system = [first]
    for index in range(blocks_after):
        system.append(system_block(f"stable suffix block {index}: {suffix}", cache=False))
    if blocks_after:
        system[-1] = system_block(system[-1]["text"], ttl=ttl)
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 128,
        "temperature": 0,
        "stream": True,
        "system": system,
        "messages": [{"role": "user", "content": output_instruction() + suffix}],
    }
    if automatic:
        body["cache_control"] = {"type": "ephemeral", **({"ttl": ttl} if ttl != "5m" else {})}
    if diagnostics_previous_id is not None:
        body["diagnostics"] = {"previous_message_id": diagnostics_previous_id}
    return body


def planned_request_count(
    lengths: list[int], repetitions: int, suite: str, concurrency: int = 2
) -> int:
    count = len(lengths) * repetitions * 2
    if suite == "correctness":
        count += 12 + concurrency
    elif suite == "ttl":
        count += 5
    return count


def estimate_planned_tokens(
    lengths: list[int], repetitions: int, suite: str, concurrency: int = 2
) -> int:
    pairs = sum(lengths) * repetitions * 2
    if suite == "correctness":
        pairs += max(lengths[0], 2048) * (12 + concurrency)
    elif suite == "ttl":
        pairs += max(lengths[0], 2048) * 5
    return pairs


def count_tokens(client: httpx.Client, headers: dict[str, str], body: dict[str, Any]) -> int | None:
    count_body = {key: value for key, value in body.items() if key not in ("stream", "max_tokens")}
    try:
        response = client.post("/v1/messages/count_tokens", json=count_body, headers=headers)
        if response.status_code != 200:
            return None
        return int(response.json().get("input_tokens"))
    except (httpx.HTTPError, TypeError, ValueError, json.JSONDecodeError):
        return None


def fitted_prefix(
    client: httpx.Client,
    headers: dict[str, str],
    model: str,
    target_tokens: int,
    run_id: str,
) -> tuple[str, int | None]:
    # Four characters/token is only a starting point; use the provider's Count
    # Tokens endpoint to correct the generated length when it is available.
    chars = max(128, target_tokens * 4)
    measured: int | None = None
    for _ in range(4):
        marker = f"FlowMesh cache validation {run_id}. Preserve this reference text. "
        text = marker + ("alpha beta gamma delta " * ((chars // 23) + 1))
        text = text[:chars]
        measured = count_tokens(client, headers, payload(model, text))
        if measured is None or abs(measured - target_tokens) <= max(32, target_tokens // 100):
            return text, measured
        chars = max(128, int(chars * target_tokens / measured))
    return text, measured


def send_request(
    client: httpx.Client,
    trace_dir: Path,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    condition: str,
    target_prefix_tokens: int,
    repetition: int,
    start_barrier: threading.Barrier | None = None,
) -> tuple[dict[str, Any], str | None]:
    prior = len(trace_records(trace_dir))
    benchmark_id = uuid.uuid4().hex
    request_headers = dict(headers)
    request_headers["x-flowmesh-benchmark-id"] = benchmark_id
    decoder = SSEDecoder()
    events: list[dict[str, Any]] = []
    first_at: float | None = None
    last_at: float | None = None
    started = time.monotonic()
    if start_barrier is not None:
        start_barrier.wait(timeout=10)
    with client.stream(
        "POST", "/v1/messages", json=body, headers=request_headers
    ) as response:
        for chunk in response.iter_bytes():
            now = time.monotonic()
            new_events = decoder.feed(chunk)
            events.extend(new_events)
            if any(event.get("type") == "content_block_delta" for event in new_events):
                first_at = first_at or now
                last_at = now
        status = response.status_code
    events.extend(decoder.finish())
    message, usage = reassemble(events)
    trace = wait_for_record(trace_dir, prior, benchmark_id)
    output_tokens = usage.get("output_tokens")
    try:
        output_tokens = int(output_tokens)
    except (TypeError, ValueError):
        output_tokens = None
    ttft_ms = (first_at - started) * 1000 if first_at is not None else None
    output_duration_ms = (
        (last_at - first_at) * 1000 if first_at is not None and last_at is not None else None
    )
    tpot_ms = (
        output_duration_ms / (output_tokens - 1)
        if output_duration_ms is not None and output_tokens is not None and output_tokens > 1
        else None
    )
    cloud_cache = trace.get("cloud_cache") or {}
    prediction = cloud_cache.get("prediction") or {}
    accounting = trace.get("token_accounting") or {}
    diagnostics = message.get("diagnostics") if isinstance(message, dict) else None
    response_model = message.get("model") if isinstance(message, dict) else None
    anthropic_backend = isinstance(response_model, str) and response_model.startswith("claude-")
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "repetition": repetition,
        "requested_model": body.get("model"),
        "response_model": response_model,
        "anthropic_backend": anthropic_backend,
        "target_prefix_tokens": target_prefix_tokens,
        "status": status,
        "prediction_state": prediction.get("state"),
        "prediction_reason": prediction.get("reason"),
        "estimated_cache_read_tokens": prediction.get("estimated_read_tokens"),
        "cache_read_input_tokens": accounting.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": accounting.get("cache_creation_input_tokens"),
        "uncached_input_tokens": accounting.get("uncached_input_tokens"),
        "total_input_tokens": accounting.get("input_tokens"),
        "output_tokens": accounting.get("output_tokens"),
        "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
        "total_ms": trace.get("timing", {}).get("total_ms"),
        "connect_ms": trace.get("timing", {}).get("connect_ms"),
        "response_headers_ms": trace.get("timing", {}).get("response_headers_ms"),
        "diagnostics": json.dumps(diagnostics, separators=(",", ":")) if diagnostics else None,
        "valid": (
            status == 200
            and anthropic_backend
            and accounting.get("cache_details_available") is True
        ),
    }
    return row, message.get("id") if isinstance(message, dict) else None


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for length in sorted({int(row["target_prefix_tokens"]) for row in rows}):
        subset = [row for row in rows if row["target_prefix_tokens"] == length]
        for condition in sorted({str(row["condition"]) for row in subset}):
            all_group = [row for row in subset if row["condition"] == condition]
            group = [row for row in all_group if row["valid"]]
            ttft = [float(row["ttft_ms"]) for row in group if row["ttft_ms"] is not None]
            tpot = [float(row["tpot_ms"]) for row in group if row["tpot_ms"] is not None]
            output.append(
                {
                    "target_prefix_tokens": length,
                    "condition": condition,
                    "valid_samples": len(group),
                    "total_samples": len(all_group),
                    "median_ttft_ms": round(statistics.median(ttft), 3) if ttft else None,
                    "median_tpot_ms": round(statistics.median(tpot), 3) if tpot else None,
                    "cache_hit_samples": sum(
                        int(row["cache_read_input_tokens"] or 0) > 0 for row in group
                    ),
                    "median_cache_read_tokens": (
                        round(
                            statistics.median(
                                int(row["cache_read_input_tokens"] or 0) for row in group
                            ),
                            1,
                        )
                        if group
                        else None
                    ),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("smoke", "performance", "correctness", "ttl"),
        default="smoke",
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_UPSTREAM)
    )
    parser.add_argument("--model", default=os.environ.get("CACHE_BENCH_MODEL", "claude-sonnet-5"))
    parser.add_argument("--prefix-tokens", type=comma_ints, default=comma_ints("2048"))
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="simultaneous requests in the correctness availability test",
    )
    parser.add_argument("--max-input-token-budget", type=int, default=250_000)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument("--ttl-refresh-at-s", type=float, default=240.0)
    parser.add_argument("--ttl-after-original-s", type=float, default=330.0)
    parser.add_argument("--cache-diagnostics", action="store_true")
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-root", type=Path)
    return parser.parse_args()


def main() -> int:
    load_env(Path(".env"))
    args = parse_args()
    if not args.confirm_live:
        raise SystemExit(
            "live cloud calls are disabled; inspect the budget, then add --confirm-live"
        )
    if args.repetitions <= 0 or args.output_tokens <= 1 or args.concurrency < 2:
        raise SystemExit(
            "repetitions must be positive, output-tokens must exceed one, "
            "and concurrency must be at least two"
        )
    planned = estimate_planned_tokens(
        args.prefix_tokens, args.repetitions, args.suite, args.concurrency
    )
    if planned > args.max_input_token_budget:
        raise SystemExit(
            f"planned input estimate {planned:,} exceeds --max-input-token-budget "
            f"{args.max_input_token_budget:,}"
        )
    estimated_cost: float | None = None
    if args.input_usd_per_million is not None and args.output_usd_per_million is not None:
        calls = planned_request_count(
            args.prefix_tokens, args.repetitions, args.suite, args.concurrency
        )
        estimated_cost = (
            planned * args.input_usd_per_million
            + calls * args.output_tokens * args.output_usd_per_million
        ) / 1_000_000
        if args.max_cost_usd is not None and estimated_cost > args.max_cost_usd:
            raise SystemExit(
                f"estimated cost ${estimated_cost:.4f} exceeds "
                f"--max-cost-usd ${args.max_cost_usd:.4f}"
            )
    elif args.max_cost_usd is not None:
        raise SystemExit("--max-cost-usd requires both per-million price arguments")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("results") / f"anthropic-cache-{args.suite}-{stamp}.csv"
    summary_path = output.with_name(f"{output.stem}-summary.csv")
    trace_root = args.trace_root or Path("traces") / "anthropic-cache" / output.stem
    if output.exists() or summary_path.exists() or trace_root.exists():
        raise SystemExit("refusing to mix with existing output; choose a new --output")
    trace_root.mkdir(parents=True)
    headers = auth_headers(args.cache_diagnostics)
    print(f"endpoint={args.base_url} model={args.model} suite={args.suite}")
    print(f"planned_input_token_estimate={planned:,} max={args.max_input_token_budget:,}")
    cost_label = estimated_cost if estimated_cost is not None else "not-configured"
    print(f"estimated_cost_usd={cost_label}")

    rows: list[dict[str, Any]] = []
    incompatible_upstream = False
    with ExitStack() as stack:
        _, client = start_proxy(args.base_url, trace_root, trace_root / "proxy.log", stack)
        for target in args.prefix_tokens:
            for repetition in range(args.repetitions):
                run_id = uuid.uuid4().hex
                prefix, counted = fitted_prefix(client, headers, args.model, target, run_id)
                print(f"prefix target={target:,} counted={counted}")
                body = payload(args.model, prefix)
                body["max_tokens"] = args.output_tokens
                cold, message_id = send_request(
                    client,
                    trace_root,
                    headers,
                    body,
                    condition="cold",
                    target_prefix_tokens=target,
                    repetition=repetition,
                )
                rows.append(cold)
                if not cold["valid"]:
                    incompatible_upstream = True
                    print(
                        "aborting matrix: preflight response was not a Claude backend "
                        "with Anthropic cache-detail usage"
                    )
                    break
                warm_body = dict(body)
                if args.cache_diagnostics and message_id:
                    warm_body["diagnostics"] = {"previous_message_id": message_id}
                warm, _ = send_request(
                    client,
                    trace_root,
                    headers,
                    warm_body,
                    condition="warm",
                    target_prefix_tokens=target,
                    repetition=repetition,
                )
                rows.append(warm)
                print(
                    f"target={target:>6} cold read={cold['cache_read_input_tokens']} "
                    f"ttft={cold['ttft_ms']} | warm prediction={warm['prediction_state']} "
                    f"read={warm['cache_read_input_tokens']} ttft={warm['ttft_ms']}"
                )
            if incompatible_upstream:
                break

        if not incompatible_upstream and args.suite == "correctness":
            target = args.prefix_tokens[0]
            prefix, _ = fitted_prefix(client, headers, args.model, target, uuid.uuid4().hex)
            base = payload(args.model, prefix, suffix="A")
            for condition, body in (
                ("partial-create", base),
                ("partial-reuse", payload(args.model, prefix, suffix="B")),
                ("system-changed", payload(args.model, prefix + " changed", suffix="B")),
                ("automatic-create", payload(args.model, prefix + " auto", automatic=True)),
                ("automatic-reuse", payload(args.model, prefix + " auto", automatic=True)),
                ("one-hour-create", payload(args.model, prefix + " hour", ttl="1h")),
                ("one-hour-reuse", payload(args.model, prefix + " hour", ttl="1h")),
                ("below-minimum", payload(args.model, "short cache prefix")),
            ):
                row, _ = send_request(
                    client,
                    trace_root,
                    headers,
                    body,
                    condition=condition,
                    target_prefix_tokens=target,
                    repetition=0,
                )
                rows.append(row)
                print(
                    f"{condition}: prediction={row['prediction_state']} "
                    f"read={row['cache_read_input_tokens']} "
                    f"created={row['cache_creation_input_tokens']}"
                )

            for distance in (19, 20):
                lookback_prefix = prefix + f" independent-lookback-{distance}"
                for condition, body in (
                    (f"lookback-{distance}-create", payload(args.model, lookback_prefix)),
                    (
                        f"lookback-{distance}-probe",
                        payload(args.model, lookback_prefix, blocks_after=distance),
                    ),
                ):
                    row, _ = send_request(
                        client,
                        trace_root,
                        headers,
                        body,
                        condition=condition,
                        target_prefix_tokens=target,
                        repetition=0,
                    )
                    rows.append(row)
                    print(
                        f"{condition}: prediction={row['prediction_state']} "
                        f"read={row['cache_read_input_tokens']} "
                        f"created={row['cache_creation_input_tokens']}"
                    )

            concurrent_prefix = prefix + " concurrent-first-write"
            concurrent_body = payload(args.model, concurrent_prefix)
            barrier = threading.Barrier(args.concurrency)
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [
                    executor.submit(
                        send_request,
                        client,
                        trace_root,
                        headers,
                        concurrent_body,
                        condition=f"concurrent-{index}",
                        target_prefix_tokens=target,
                        repetition=0,
                        start_barrier=barrier,
                    )
                    for index in range(args.concurrency)
                ]
                for future in futures:
                    row, _ = future.result()
                    rows.append(row)
                    print(
                        f"{row['condition']}: prediction={row['prediction_state']} "
                        f"read={row['cache_read_input_tokens']} "
                        f"created={row['cache_creation_input_tokens']}"
                    )

        if not incompatible_upstream and args.suite == "ttl":
            target = args.prefix_tokens[0]
            refresh_prefix, _ = fitted_prefix(
                client, headers, args.model, target, uuid.uuid4().hex
            )
            expiry_prefix, _ = fitted_prefix(
                client, headers, args.model, target, uuid.uuid4().hex
            )
            refresh_body = payload(args.model, refresh_prefix)
            expiry_body = payload(args.model, expiry_prefix)
            refresh_create, _ = send_request(
                client,
                trace_root,
                headers,
                refresh_body,
                condition="ttl-refresh-create",
                target_prefix_tokens=target,
                repetition=0,
            )
            rows.append(refresh_create)
            expiry_create, _ = send_request(
                client,
                trace_root,
                headers,
                expiry_body,
                condition="ttl-expiry-create",
                target_prefix_tokens=target,
                repetition=0,
            )
            rows.append(expiry_create)
            time.sleep(args.ttl_refresh_at_s)
            refresh, _ = send_request(
                client,
                trace_root,
                headers,
                refresh_body,
                condition="ttl-refresh",
                target_prefix_tokens=target,
                repetition=0,
            )
            rows.append(refresh)
            remaining = max(0.0, args.ttl_after_original_s - args.ttl_refresh_at_s)
            time.sleep(remaining)
            after, _ = send_request(
                client,
                trace_root,
                headers,
                refresh_body,
                condition="ttl-refreshed-after-original-expiry",
                target_prefix_tokens=target,
                repetition=0,
            )
            rows.append(after)
            expired, _ = send_request(
                client,
                trace_root,
                headers,
                expiry_body,
                condition="ttl-no-read-after-expiry",
                target_prefix_tokens=target,
                repetition=0,
            )
            rows.append(expired)

    write_csv(output, rows)
    write_csv(summary_path, summarize(rows))
    print(f"samples={output}")
    print(f"summary={summary_path}")
    print(f"private_traces={trace_root}")
    return 0 if all(row["valid"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
