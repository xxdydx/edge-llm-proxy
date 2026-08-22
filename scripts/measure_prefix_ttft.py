#!/usr/bin/env python3
"""Measure vLLM TTFT as a function of total and cached-prefix tokens.

For every sample this benchmark resets the prefix cache, sends a warmer prompt,
then sends a target prompt sharing exactly the requested token prefix.  It
measures the target's first-token latency and verifies the actual cache state
using vLLM's prefix-cache counter deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


METRIC_LINE = re.compile(r"^(?P<name>[^\s{]+)(?:\{[^}]*\})?\s+(?P<value>\S+)$")
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')
QUERY_METRIC = "vllm:prefix_cache_queries_total"
HIT_METRIC = "vllm:prefix_cache_hits_total"


@dataclass(frozen=True)
class Condition:
    total_tokens: int
    prefix_fraction: float
    repetition: int


def comma_ints(value: str) -> list[int]:
    try:
        out = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not out or any(number <= 0 for number in out):
        raise argparse.ArgumentTypeError("lengths must be positive")
    return out


def comma_floats(value: str) -> list[float]:
    try:
        out = [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not out or any(number < 0 or number > 1 for number in out):
        raise argparse.ArgumentTypeError("fractions must be between 0 and 1")
    return out


def metric_total(text: str, metric: str) -> float:
    total = 0.0
    found = False
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if match and match.group("name") == metric:
            total += float(match.group("value"))
            found = True
    if not found:
        raise RuntimeError(f"metric {metric!r} not found at /metrics")
    return total


def cache_config(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            return {
                key: bytes(value, "utf-8").decode("unicode_escape")
                for key, value in LABEL.findall(line)
            }
    return {}


def detect_match_unit(config: dict[str, str]) -> int | None:
    """Return the first positive cache match unit reported by vLLM.

    Some vLLM versions expose ``prefix_match_unit="None"`` alongside a valid
    ``block_size``.  Treat non-numeric sentinel values as absent so detection
    falls through to the block size instead of failing during integer parsing.
    """
    for key in ("prefix_match_unit", "block_size"):
        raw = config.get(key)
        try:
            value = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            continue
        if value is not None and value > 0:
            return value
    return None


def cache_counters(client: httpx.Client) -> tuple[float, float, str]:
    response = client.get("/metrics")
    response.raise_for_status()
    text = response.text
    return metric_total(text, QUERY_METRIC), metric_total(text, HIT_METRIC), text


def reset_cache(client: httpx.Client) -> None:
    response = client.post("/reset_prefix_cache")
    response.raise_for_status()


def completion_body(model: str, prompt: list[int], stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "stream": stream,
        "return_token_ids": True,
    }


def warm(client: httpx.Client, model: str, prompt: list[int]) -> None:
    response = client.post("/v1/completions", json=completion_body(model, prompt, False))
    response.raise_for_status()


def measure_ttft(client: httpx.Client, model: str, prompt: list[int]) -> tuple[float, float]:
    started = time.perf_counter()
    headers_ms: float | None = None
    ttft_ms: float | None = None

    with client.stream(
        "POST", "/v1/completions", json=completion_body(model, prompt, True)
    ) as response:
        headers_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            payload = json.loads(data)
            for choice in payload.get("choices") or []:
                if choice.get("text") or choice.get("token_ids"):
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000

    if headers_ms is None or ttft_ms is None:
        raise RuntimeError("stream ended without a generated token")
    return ttft_ms, headers_ms


def token_sequences(tokenizer_name: str, length: int, seed: int) -> tuple[list[int], ...]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to construct valid token IDs") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    special = set(tokenizer.all_special_ids)
    pool = [token_id for token_id in range(len(tokenizer)) if token_id not in special]
    if len(pool) < 3:
        raise RuntimeError("tokenizer has too few non-special tokens")

    streams: list[list[int]] = []
    for offset in range(4):
        rng = random.Random(seed + offset)
        streams.append([rng.choice(pool) for _ in range(length)])

    # A target must diverge at the first token after every possible shared
    # prefix.  Otherwise an accidental match could make the real hit exceed R.
    warm_tail, target_tail, unrelated = streams[1], streams[2], streams[3]
    replacement_rng = random.Random(seed + 10)
    for index in range(length):
        while target_tail[index] == warm_tail[index]:
            target_tail[index] = replacement_rng.choice(pool)
    while unrelated[0] == target_tail[0]:
        unrelated[0] = replacement_rng.choice(pool)

    return tuple(streams)


def server_version(client: httpx.Client) -> str:
    try:
        response = client.get("/version")
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("version") or payload)
    except Exception:
        return "unknown"


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


def write_summary(output: Path, rows: list[dict[str, Any]]) -> Path:
    summary_path = output.with_name(f"{output.stem}-summary.csv")
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["total_tokens"]), int(row["intended_cached_tokens"]))
        groups.setdefault(key, []).append(row)

    cold_medians: dict[int, float] = {}
    for (total, cached), group in groups.items():
        valid = [float(row["ttft_ms"]) for row in group if row["valid_hit"]]
        if cached == 0 and valid:
            cold_medians[total] = statistics.median(valid)

    fields = [
        "total_tokens",
        "intended_cached_tokens",
        "cached_fraction",
        "median_ttft_ms",
        "p10_ttft_ms",
        "p90_ttft_ms",
        "saved_vs_cold_ms",
        "valid_samples",
        "total_samples",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (total, cached), group in sorted(groups.items()):
            valid = [float(row["ttft_ms"]) for row in group if row["valid_hit"]]
            if not valid:
                continue
            median = statistics.median(valid)
            cold = cold_medians.get(total)
            writer.writerow(
                {
                    "total_tokens": total,
                    "intended_cached_tokens": cached,
                    "cached_fraction": cached / total,
                    "median_ttft_ms": round(median, 3),
                    "p10_ttft_ms": round(percentile(valid, 0.1), 3),
                    "p90_ttft_ms": round(percentile(valid, 0.9), 3),
                    "saved_vs_cold_ms": round(cold - median, 3) if cold is not None else "",
                    "valid_samples": len(valid),
                    "total_samples": len(group),
                }
            )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="local")
    parser.add_argument("--tokenizer", required=True, help="HF model/tokenizer name or path")
    parser.add_argument(
        "--lengths",
        type=comma_ints,
        default=comma_ints("1024,2048,4096,8192,12288,16384,24576"),
    )
    parser.add_argument(
        "--fractions",
        type=comma_floats,
        default=comma_floats("0,0.25,0.5,0.75,0.9,1"),
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--match-unit", type=int, help="override auto-detected prefix unit")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")

    output = args.output or Path("results") / (
        f"prefix-ttft-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        _, _, metrics_text = cache_counters(client)
        config = cache_config(metrics_text)
        match_unit = args.match_unit or detect_match_unit(config)
        if not match_unit or match_unit <= 0:
            raise SystemExit(
                "could not detect prefix_match_unit/block_size from /metrics; "
                "pass --match-unit explicitly"
            )

        max_length = max(args.lengths)
        shared, warm_tail, target_tail, unrelated = token_sequences(
            args.tokenizer, max_length, args.seed
        )
        version = server_version(client)

        print(f"server={args.base_url} version={version}")
        print(f"model={args.model} tokenizer={args.tokenizer} match_unit={match_unit}")
        print(f"cache_config={json.dumps(config, sort_keys=True)}")
        print(f"output={output}")

        # Settle imports, kernels, and the HTTP connection with unrelated data,
        # then clear the prefix cache before the actual randomized sweep.
        warm(client, args.model, unrelated[: min(512, max_length)])
        reset_cache(client)

        conditions = [
            Condition(total, fraction, repetition)
            for total in args.lengths
            for fraction in args.fractions
            for repetition in range(args.repetitions)
        ]
        random.Random(args.seed).shuffle(conditions)

        fields = [
            "timestamp_utc",
            "vllm_version",
            "model",
            "tokenizer",
            "kv_cache_dtype",
            "num_gpu_blocks",
            "kv_cache_size_tokens",
            "match_unit",
            "total_tokens",
            "prefix_fraction",
            "intended_cached_tokens",
            "actual_queried_tokens",
            "actual_cached_tokens",
            "repetition",
            "ttft_ms",
            "response_headers_ms",
            "valid_hit",
        ]
        rows: list[dict[str, Any]] = []

        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()

            for index, condition in enumerate(conditions, 1):
                total = condition.total_tokens
                cached = math.floor(total * condition.prefix_fraction / match_unit) * match_unit
                warm_prompt = shared[:cached] + warm_tail[: total - cached]
                target_prompt = shared[:cached] + target_tail[: total - cached]
                if cached == 0:
                    warm_prompt = unrelated[:total]

                reset_cache(client)
                warm(client, args.model, warm_prompt)
                queries_before, hits_before, _ = cache_counters(client)
                ttft_ms, headers_ms = measure_ttft(client, args.model, target_prompt)
                queries_after, hits_after, _ = cache_counters(client)

                actual_queries = round(queries_after - queries_before)
                actual_hits = round(hits_after - hits_before)
                valid_hit = abs(actual_hits - cached) <= match_unit
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "vllm_version": version,
                    "model": args.model,
                    "tokenizer": args.tokenizer,
                    "kv_cache_dtype": config.get("cache_dtype", config.get("kv_cache_dtype", "")),
                    "num_gpu_blocks": config.get("num_gpu_blocks", ""),
                    "kv_cache_size_tokens": config.get("kv_cache_size_tokens", ""),
                    "match_unit": match_unit,
                    "total_tokens": total,
                    "prefix_fraction": condition.prefix_fraction,
                    "intended_cached_tokens": cached,
                    "actual_queried_tokens": actual_queries,
                    "actual_cached_tokens": actual_hits,
                    "repetition": condition.repetition,
                    "ttft_ms": round(ttft_ms, 3),
                    "response_headers_ms": round(headers_ms, 3),
                    "valid_hit": valid_hit,
                }
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(
                    f"[{index:>3}/{len(conditions)}] N={total:>5} R={cached:>5} "
                    f"hit={actual_hits:>5} ttft={ttft_ms:>8.1f}ms "
                    f"{'ok' if valid_hit else 'HIT-MISMATCH'}"
                )

    invalid = sum(not row["valid_hit"] for row in rows)
    summary = write_summary(output, rows)
    print(f"\nwrote {len(rows)} raw rows to {output}")
    print(f"wrote curve summary to {summary}")
    print(f"cache-hit mismatches={invalid}")
    return 1 if invalid else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
