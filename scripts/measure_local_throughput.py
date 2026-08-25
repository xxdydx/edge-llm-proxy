#!/usr/bin/env python3
"""Measure edge vLLM latency, TPOT, and throughput across a controlled matrix.

The benchmark varies input length, requested output length, concurrency, and
cold/full-warm prefix state. It writes one raw row per request and one summary
row per condition. Cache state is checked from vLLM prefix-cache counter deltas.
``scripts/edge_tpot.py`` is the canonical user-facing entry point; this module
retains the original name for compatibility and contains the implementation.

This is a serving-performance benchmark, not a quality benchmark. Prompts are
expanded to exact token lengths and ``ignore_eos`` forces the requested decode
length so early EOS does not confound throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

if __package__:
    from scripts.measure_prefix_ttft import (
        HIT_METRIC,
        QUERY_METRIC,
        cache_config,
        detect_match_unit,
        metric_total,
    )
else:
    from measure_prefix_ttft import (  # type: ignore[no-redef]
        HIT_METRIC,
        QUERY_METRIC,
        cache_config,
        detect_match_unit,
        metric_total,
    )


@dataclass(frozen=True)
class Condition:
    prompt_tokens: int
    requested_output_tokens: int
    concurrency: int
    cache_state: str
    repetition: int


@dataclass
class RequestResult:
    request_index: int
    prompt_id: str
    status: int | None = None
    output_tokens: int = 0
    response_headers_ms: float | None = None
    ttft_ms: float | None = None
    last_token_ms: float | None = None
    e2e_ms: float | None = None
    decode_ms: float | None = None
    tpot_ms: float | None = None
    decode_tokens_per_s: float | None = None
    error: str = ""


def comma_ints(value: str) -> list[int]:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(number <= 0 for number in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def cache_states(value: str) -> list[str]:
    values = [part.strip().lower() for part in value.split(",")]
    allowed = {"cold", "warm"}
    if not values or any(item not in allowed for item in values):
        raise argparse.ArgumentTypeError("cache states must be cold,warm")
    return values


def load_prompts(path: Path) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            raise ValueError(f"{path}:{line_number}: expected prompt_id<TAB>prompt")
        prompt_id, text = line.split("\t", 1)
        if not prompt_id.strip() or not text.strip():
            raise ValueError(f"{path}:{line_number}: prompt id and text must be non-empty")
        prompts.append((prompt_id.strip(), text.strip()))
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to construct controlled prompts") from exc
    return AutoTokenizer.from_pretrained(name)


def exact_prompt_tokens(
    tokenizer: Any,
    prompt_text: str,
    length: int,
    condition_seed: int,
    request_index: int,
) -> list[int]:
    marker = (
        f"Controlled benchmark request {condition_seed}-{request_index}. "
        "Treat the following as the task context. "
    )
    marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
    seed_tokens = tokenizer.encode(prompt_text + "\n", add_special_tokens=False)
    if not seed_tokens:
        raise RuntimeError("prompt seed tokenized to an empty sequence")

    # A distinct non-special token in position zero prevents nominally cold
    # concurrent requests from sharing even the first cache block. A textual
    # marker alone is insufficient because its differing digits may occur only
    # after a common opening block. The same sequence is used by the warmer.
    special = set(tokenizer.all_special_ids)
    unique_id = (condition_seed + request_index * 104729) % len(tokenizer)
    while unique_id in special:
        unique_id = (unique_id + 1) % len(tokenizer)
    combined = [unique_id]
    combined.extend(marker_tokens[: max(0, length - 1)])
    while len(combined) < length:
        remaining = length - len(combined)
        combined.extend(seed_tokens[:remaining])
    return combined


def completion_body(
    model: str, prompt: list[int], output_tokens: int, stream: bool
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": stream,
        "return_token_ids": True,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


async def fetch_metrics(client: httpx.AsyncClient) -> tuple[float, float, str]:
    response = await client.get("/metrics")
    response.raise_for_status()
    text = response.text
    return metric_total(text, QUERY_METRIC), metric_total(text, HIT_METRIC), text


async def reset_cache(client: httpx.AsyncClient) -> None:
    response = await client.post("/reset_prefix_cache")
    if response.status_code == 404:
        raise RuntimeError(
            "/reset_prefix_cache is unavailable; vLLM 0.27 exposes it only when "
            "the server is started with VLLM_SERVER_DEV_MODE=1"
        )
    response.raise_for_status()


async def warm_prompt(client: httpx.AsyncClient, model: str, prompt: list[int]) -> None:
    response = await client.post(
        "/v1/completions", json=completion_body(model, prompt, 1, False)
    )
    response.raise_for_status()


async def server_version(client: httpx.AsyncClient) -> str:
    try:
        response = await client.get("/version")
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("version") or payload)
    except Exception:
        return "unknown"


async def measure_request(
    client: httpx.AsyncClient,
    model: str,
    prompt: list[int],
    output_tokens: int,
    request_index: int,
    prompt_id: str,
) -> RequestResult:
    result = RequestResult(request_index=request_index, prompt_id=prompt_id)
    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    counted_token_ids = 0
    usage_output_tokens: int | None = None

    try:
        async with client.stream(
            "POST",
            "/v1/completions",
            json=completion_body(model, prompt, output_tokens, True),
        ) as response:
            result.status = response.status_code
            result.response_headers_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    continue
                payload = json.loads(data)
                usage = payload.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    usage_output_tokens = int(usage["completion_tokens"])

                for choice in payload.get("choices") or []:
                    token_ids = choice.get("token_ids") or []
                    has_token = bool(token_ids) or bool(choice.get("text"))
                    if not has_token:
                        continue
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    counted_token_ids += len(token_ids)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    finished = time.perf_counter()
    result.e2e_ms = (finished - started) * 1000
    result.output_tokens = usage_output_tokens or counted_token_ids
    if first_token_at is not None:
        result.ttft_ms = (first_token_at - started) * 1000
    if last_token_at is not None:
        result.last_token_ms = (last_token_at - started) * 1000
    if first_token_at is not None and last_token_at is not None:
        result.decode_ms = max(0.0, (last_token_at - first_token_at) * 1000)
        result.tpot_ms = calculate_tpot_ms(result.decode_ms, result.output_tokens)
        if result.tpot_ms is not None:
            result.decode_tokens_per_s = 1000 / result.tpot_ms
    if result.status == 200 and result.output_tokens == 0 and not result.error:
        result.error = "stream completed without countable output tokens"
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def calculate_tpot_ms(decode_ms: float | None, output_tokens: int) -> float | None:
    """Return mean time between generated tokens after the first token.

    TTFT ends when token one arrives. The remaining ``output_tokens - 1``
    inter-token intervals make up decode time, so a one-token response has no
    defined TPOT rather than a misleading zero.
    """
    decode_tokens = output_tokens - 1
    if decode_ms is None or decode_ms <= 0 or decode_tokens <= 0:
        return None
    return decode_ms / decode_tokens


def optional_round(value: float | None) -> float | str:
    return round(value, 3) if value is not None else ""


def write_summary(output: Path, rows: list[dict[str, Any]]) -> Path:
    summary_path = output.with_name(f"{output.stem}-summary.csv")
    grouped: dict[tuple[int, int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["prompt_tokens"]),
            int(row["requested_output_tokens"]),
            int(row["concurrency"]),
            str(row["cache_state"]),
        )
        grouped.setdefault(key, []).append(row)

    fields = [
        "backend",
        "prompt_tokens",
        "requested_output_tokens",
        "concurrency",
        "cache_state",
        "median_realized_cached_fraction",
        "cache_valid_batches",
        "total_batches",
        "valid_requests",
        "total_requests",
        "median_ttft_ms",
        "p90_ttft_ms",
        "median_tpot_ms",
        "p90_tpot_ms",
        "median_e2e_ms",
        "p90_e2e_ms",
        "median_request_decode_tokens_per_s",
        "p10_request_decode_tokens_per_s",
        "median_batch_output_tokens_per_s",
        "p10_batch_output_tokens_per_s",
        "measured_output_tokens",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, group in sorted(grouped.items()):
            valid = [row for row in group if not row["error"] and row["status"] == 200]
            ttfts = [float(row["ttft_ms"]) for row in valid if row["ttft_ms"] != ""]
            e2es = [float(row["e2e_ms"]) for row in valid if row["e2e_ms"] != ""]
            decode_rates = [
                float(row["decode_tokens_per_s"])
                for row in valid
                if row["decode_tokens_per_s"] != ""
            ]
            tpots = [float(row["tpot_ms"]) for row in valid if row["tpot_ms"] != ""]
            batches = {str(row["batch_id"]): row for row in group}.values()
            batch_rates = [float(row["batch_output_tokens_per_s"]) for row in batches]
            cache_fractions = [float(row["realized_cached_fraction"]) for row in batches]
            writer.writerow(
                {
                    "backend": "edge",
                    "prompt_tokens": key[0],
                    "requested_output_tokens": key[1],
                    "concurrency": key[2],
                    "cache_state": key[3],
                    "median_realized_cached_fraction": round(
                        statistics.median(cache_fractions), 4
                    ),
                    "cache_valid_batches": sum(bool(row["cache_state_valid"]) for row in batches),
                    "total_batches": len(cache_fractions),
                    "valid_requests": len(valid),
                    "total_requests": len(group),
                    "median_ttft_ms": optional_round(statistics.median(ttfts) if ttfts else None),
                    "p90_ttft_ms": optional_round(percentile(ttfts, 0.9) if ttfts else None),
                    "median_tpot_ms": optional_round(
                        statistics.median(tpots) if tpots else None
                    ),
                    "p90_tpot_ms": optional_round(
                        percentile(tpots, 0.9) if tpots else None
                    ),
                    "median_e2e_ms": optional_round(statistics.median(e2es) if e2es else None),
                    "p90_e2e_ms": optional_round(percentile(e2es, 0.9) if e2es else None),
                    "median_request_decode_tokens_per_s": optional_round(
                        statistics.median(decode_rates) if decode_rates else None
                    ),
                    "p10_request_decode_tokens_per_s": optional_round(
                        percentile(decode_rates, 0.1) if decode_rates else None
                    ),
                    "median_batch_output_tokens_per_s": optional_round(
                        statistics.median(batch_rates) if batch_rates else None
                    ),
                    "p10_batch_output_tokens_per_s": optional_round(
                        percentile(batch_rates, 0.1) if batch_rates else None
                    ),
                    "measured_output_tokens": sum(int(row["output_tokens"]) for row in valid),
                }
            )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="local")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", type=Path, default=Path("prompts.txt"))
    parser.add_argument(
        "--prompt-lengths", type=comma_ints, default=comma_ints("1024,8192,24576")
    )
    parser.add_argument(
        "--output-lengths", type=comma_ints, default=comma_ints("128,512,2048")
    )
    parser.add_argument("--concurrency", type=comma_ints, default=comma_ints("1,2,4,8"))
    parser.add_argument("--cache-states", type=cache_states, default=cache_states("cold,warm"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--match-unit", type=int)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    prompts = load_prompts(args.prompts)
    tokenizer = load_tokenizer(args.tokenizer)
    output = args.output or Path("results") / (
        f"local-throughput-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    limits = httpx.Limits(max_connections=max(args.concurrency) + 4)
    timeout = httpx.Timeout(args.timeout, connect=10.0)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), timeout=timeout, limits=limits
    ) as client:
        _, _, metrics_text = await fetch_metrics(client)
        config = cache_config(metrics_text)
        match_unit = args.match_unit or detect_match_unit(config)
        if not match_unit:
            raise RuntimeError(
                "could not detect prefix_match_unit/block_size; pass --match-unit"
            )
        version = await server_version(client)

        print(f"server={args.base_url} version={version} model={args.model}")
        print(f"tokenizer={args.tokenizer} match_unit={match_unit}")
        print(f"prompts={args.prompts} output={output}")

        conditions = [
            Condition(prompt, generated, concurrency, state, repetition)
            for prompt in args.prompt_lengths
            for generated in args.output_lengths
            for concurrency in args.concurrency
            for state in args.cache_states
            for repetition in range(args.repetitions)
        ]
        random.Random(args.seed).shuffle(conditions)

        raw_fields = [
            "backend",
            "timestamp_utc",
            "batch_id",
            "vllm_version",
            "model",
            "tokenizer",
            "prompt_id",
            "request_index",
            "prompt_tokens",
            "requested_output_tokens",
            "concurrency",
            "cache_state",
            "repetition",
            "actual_queried_tokens",
            "actual_cached_tokens",
            "realized_cached_fraction",
            "cache_state_valid",
            "status",
            "output_tokens",
            "response_headers_ms",
            "ttft_ms",
            "last_token_ms",
            "decode_ms",
            "tpot_ms",
            "decode_tokens_per_s",
            "e2e_ms",
            "batch_wall_ms",
            "batch_output_tokens_per_s",
            "error",
        ]
        rows: list[dict[str, Any]] = []
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=raw_fields)
            writer.writeheader()

            # One unreported request settles imports, kernels, and connection setup.
            smoke_prompt = exact_prompt_tokens(tokenizer, prompts[0][1], 256, args.seed, 0)
            await warm_prompt(client, args.model, smoke_prompt)
            await reset_cache(client)

            for condition_number, condition in enumerate(conditions, 1):
                condition_seed = args.seed + condition_number * 1000
                selected: list[tuple[str, list[int]]] = []
                for request_index in range(condition.concurrency):
                    prompt_id, prompt_text = prompts[
                        (condition.repetition + request_index) % len(prompts)
                    ]
                    tokens = exact_prompt_tokens(
                        tokenizer,
                        prompt_text,
                        condition.prompt_tokens,
                        condition_seed,
                        request_index,
                    )
                    selected.append((prompt_id, tokens))

                await reset_cache(client)
                if condition.cache_state == "warm":
                    for _, tokens in selected:
                        await warm_prompt(client, args.model, tokens)

                queries_before, hits_before, _ = await fetch_metrics(client)
                batch_started = time.perf_counter()
                results = await asyncio.gather(
                    *[
                        measure_request(
                            client,
                            args.model,
                            tokens,
                            condition.requested_output_tokens,
                            request_index,
                            prompt_id,
                        )
                        for request_index, (prompt_id, tokens) in enumerate(selected)
                    ]
                )
                batch_wall_ms = (time.perf_counter() - batch_started) * 1000
                queries_after, hits_after, _ = await fetch_metrics(client)

                actual_queries = max(0, round(queries_after - queries_before))
                actual_hits = max(0, round(hits_after - hits_before))
                realized_fraction = actual_hits / actual_queries if actual_queries else 0.0
                if condition.cache_state == "cold":
                    cache_valid = actual_hits <= match_unit * condition.concurrency
                else:
                    expected_hits = max(
                        0,
                        (condition.prompt_tokens - match_unit) * condition.concurrency,
                    )
                    cache_valid = abs(actual_hits - expected_hits) <= (
                        match_unit * condition.concurrency
                    )

                total_output = sum(result.output_tokens for result in results)
                batch_rate = total_output / (batch_wall_ms / 1000) if batch_wall_ms else 0.0
                batch_id = f"batch-{condition_number:04d}"
                for result in results:
                    row = {
                        "backend": "edge",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "batch_id": batch_id,
                        "vllm_version": version,
                        "model": args.model,
                        "tokenizer": args.tokenizer,
                        "prompt_id": result.prompt_id,
                        "request_index": result.request_index,
                        "prompt_tokens": condition.prompt_tokens,
                        "requested_output_tokens": condition.requested_output_tokens,
                        "concurrency": condition.concurrency,
                        "cache_state": condition.cache_state,
                        "repetition": condition.repetition,
                        "actual_queried_tokens": actual_queries,
                        "actual_cached_tokens": actual_hits,
                        "realized_cached_fraction": round(realized_fraction, 6),
                        "cache_state_valid": cache_valid,
                        "status": result.status or "",
                        "output_tokens": result.output_tokens,
                        "response_headers_ms": optional_round(result.response_headers_ms),
                        "ttft_ms": optional_round(result.ttft_ms),
                        "last_token_ms": optional_round(result.last_token_ms),
                        "decode_ms": optional_round(result.decode_ms),
                        "tpot_ms": optional_round(result.tpot_ms),
                        "decode_tokens_per_s": optional_round(
                            result.decode_tokens_per_s
                        ),
                        "e2e_ms": optional_round(result.e2e_ms),
                        "batch_wall_ms": round(batch_wall_ms, 3),
                        "batch_output_tokens_per_s": round(batch_rate, 3),
                        "error": result.error,
                    }
                    writer.writerow(row)
                    rows.append(row)
                handle.flush()

                valid_requests = sum(not result.error for result in results)
                print(
                    f"[{condition_number:>3}/{len(conditions)}] "
                    f"N={condition.prompt_tokens:>5} O={condition.requested_output_tokens:>4} "
                    f"C={condition.concurrency} {condition.cache_state:<4} "
                    f"cache={realized_fraction:>6.1%} "
                    f"batch={batch_rate:>7.1f} tok/s valid={valid_requests}/{len(results)}"
                )

    summary = write_summary(output, rows)
    failures = sum(bool(row["error"]) for row in rows)
    print(f"\nwrote {len(rows)} request rows to {output}")
    print(f"wrote condition summary to {summary}")
    print(f"request failures={failures}")
    return 1 if failures else 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
