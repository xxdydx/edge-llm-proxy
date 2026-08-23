"""Write one JSONL record per proxied call.

Nothing in here may raise into the request path — edgeproxy sits in front of
daily Claude Code use, and a recording bug must never break a session.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

log = logging.getLogger("edgeproxy.trace")

# Dropped before anything is written. Traces are gitignored, but they should
# not contain credentials regardless.
REDACT = {
    "authorization",
    "x-api-key",
    "proxy-authorization",
    "cookie",
    "set-cookie",
}


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items() if k.lower() not in REDACT}


def _token_count(value: Any) -> int | None:
    if value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def build_token_accounting(usage: Any) -> dict[str, int | bool | None]:
    """Build one provider-neutral token summary for every trace record.

    Anthropic-compatible detailed usage partitions total input into uncached,
    cache-read, and cache-creation tokens. Without cache-detail fields, the
    provider's input total remains useful but its cache breakdown is unknown.
    """
    usage = usage if isinstance(usage, dict) else {}
    raw_input = _token_count(usage.get("input_tokens"))
    output = _token_count(usage.get("output_tokens"))
    cache_details_available = (
        "cache_read_input_tokens" in usage
        or "cache_creation_input_tokens" in usage
    )

    if cache_details_available:
        cache_read = _token_count(usage.get("cache_read_input_tokens")) or 0
        cache_creation = _token_count(usage.get("cache_creation_input_tokens")) or 0
        uncached = raw_input
        total_input = (
            raw_input + cache_read + cache_creation
            if raw_input is not None
            else None
        )
    else:
        cache_read = None
        cache_creation = None
        uncached = None
        total_input = raw_input

    tokens_processed = (
        total_input + output
        if total_input is not None and output is not None
        else None
    )
    return {
        "input_tokens": total_input,
        "output_tokens": output,
        "tokens_processed": tokens_processed,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
        "uncached_input_tokens": uncached,
        "cache_details_available": cache_details_available,
    }


# --------------------------------------------------------------------- SSE ---


def parse_sse(raw: bytes) -> list[dict[str, Any]]:
    """Pull the JSON payloads out of an accumulated SSE byte stream."""
    events: list[dict[str, Any]] = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def reassemble(events: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fold Anthropic stream events back into (message, usage).

    Deltas arrive per content block: text_delta appends text, input_json_delta
    appends a JSON fragment that only parses once the block is complete.
    """
    message: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}

    for e in events:
        etype = e.get("type")

        if etype == "message_start":
            m = e.get("message") or {}
            message.update({k: v for k, v in m.items() if k != "content"})
            usage.update(m.get("usage") or {})

        elif etype == "content_block_start":
            blocks[e.get("index", len(blocks))] = dict(e.get("content_block") or {})

        elif etype == "content_block_delta":
            block = blocks.setdefault(e.get("index", 0), {})
            delta = e.get("delta") or {}
            match delta.get("type"):
                case "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                case "input_json_delta":
                    block["_partial_json"] = block.get("_partial_json", "") + delta.get(
                        "partial_json", ""
                    )
                case "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
                case "signature_delta":
                    block["signature"] = block.get("signature", "") + delta.get("signature", "")

        elif etype == "message_delta":
            message.update(e.get("delta") or {})
            usage.update(e.get("usage") or {})

    content = []
    for index in sorted(blocks):
        block = blocks[index]
        partial = block.pop("_partial_json", None)
        if partial is not None:
            try:
                block["input"] = json.loads(partial)
            except json.JSONDecodeError:
                block["input"] = {"_unparsed": partial}
        content.append(block)

    message["content"] = content
    return message, usage


# ------------------------------------------------------------------ writer ---


class TraceWriter:
    """Append-only JSONL, one file per UTC day."""

    def __init__(self, trace_dir: Path) -> None:
        self.dir = trace_dir
        self._lock = threading.Lock()
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            log.exception("could not create trace dir %s — recording disabled", self.dir)

    def _path(self) -> Path:
        return self.dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"

    def write(self, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock, self._path().open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            log.exception("trace write failed (ignored)")
