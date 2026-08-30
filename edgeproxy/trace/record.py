"""Write one JSONL record per proxied call.

Nothing in here may raise into the request path — edgeproxy sits in front of
daily Claude Code use, and a recording bug must never break a session.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

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


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _stable_hash(namespace: bytes, value: Any) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(b"\0")
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _without_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_cache_control(child)
            for key, child in value.items()
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [_without_cache_control(child) for child in value]
    return value


def request_identity(request: Any) -> dict[str, Any]:
    """Return stable, non-secret lineage, turn, and tool-suite identities."""
    if not isinstance(request, dict):
        return {
            "lineage_id": None,
            "turn_id": None,
            "tool_suite_hash": None,
            "tool_names": [],
        }
    tools = request.get("tools") or []
    semantic_tools = _without_cache_control(tools)
    tool_names = [
        str(tool.get("name") or tool.get("type"))
        for tool in tools
        if isinstance(tool, dict) and (tool.get("name") or tool.get("type"))
    ]
    static_prefix = {
        "model": request.get("model"),
        "system": _without_cache_control(request.get("system") or []),
        "tools": semantic_tools,
    }
    return {
        "lineage_id": _stable_hash(b"edgeproxy-lineage-v1", static_prefix),
        "turn_id": _stable_hash(
            b"edgeproxy-turn-v1",
            {"lineage": static_prefix, "messages": request.get("messages") or []},
        ),
        "tool_suite_hash": (
            _stable_hash(b"edgeproxy-tool-suite-v1", semantic_tools)
            if tools
            else None
        ),
        "tool_names": tool_names,
    }


def consumed_tool_result_ids(request: Any) -> list[str]:
    """Return tool results introduced by the latest causal user message.

    Anthropic requests contain the whole conversation, so collecting every
    ``tool_result`` would incorrectly make each call depend on all earlier
    tools.  The newest message containing tool results is the causal input for
    this inference turn.  Preserve block order while removing duplicates.
    """
    messages = request.get("messages") if isinstance(request, dict) else None
    for message in reversed(messages or []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        ids = [
            str(block.get("tool_use_id"))
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "tool_result"
            and block.get("tool_use_id")
        ]
        if ids:
            return list(dict.fromkeys(ids))
    return []


def _agent_parent_tool_use_id(request: Any, headers: Mapping[str, Any]) -> str | None:
    """Read an explicit agent-parent ID when a client provides one."""
    candidates: list[Any] = [
        headers.get("x-claude-code-parent-tool-use-id"),
        headers.get("x-parent-tool-use-id"),
    ]
    if isinstance(request, dict):
        metadata = request.get("metadata")
        candidates.append(request.get("parent_tool_use_id"))
        if isinstance(metadata, dict):
            candidates.append(metadata.get("parent_tool_use_id"))
    return next((str(value) for value in candidates if value), None)


def validate_tool_use_blocks(request: Any, response: Any) -> list[dict[str, Any]]:
    """Validate each returned tool-use input against its requested JSON schema."""
    tools = request.get("tools") if isinstance(request, dict) else None
    schemas = {
        str(tool.get("name")): tool.get("input_schema")
        for tool in (tools or [])
        if isinstance(tool, dict)
        and tool.get("name")
        and isinstance(tool.get("input_schema"), dict)
    }
    content = response.get("content") if isinstance(response, dict) else None
    results: list[dict[str, Any]] = []
    for index, block in enumerate(content or []):
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = str(block.get("name") or "")
        schema = schemas.get(name)
        tool_input = block.get("input")
        valid = False
        error: str | None = None
        if schema is None:
            error = "tool schema not present in request"
        elif isinstance(tool_input, dict) and "_unparsed" in tool_input:
            error = "tool input JSON was incomplete or malformed"
        else:
            try:
                validator = Draft202012Validator(schema)
                first_error = next(iter(validator.iter_errors(tool_input)), None)
                if first_error is None:
                    valid = True
                else:
                    path = ".".join(str(part) for part in first_error.absolute_path)
                    error = f"{path + ': ' if path else ''}{first_error.message}"
            except SchemaError as exc:
                error = f"invalid requested tool schema: {exc.message}"
        results.append(
            {
                "content_block_index": index,
                "tool_use_id": block.get("id"),
                "tool_name": name or None,
                "schema_valid": valid,
                "validation_error": error,
            }
        )
    return results


def build_structured_call(
    record: Mapping[str, Any], original_request: Any
) -> dict[str, Any]:
    """Build the stable v1 provider-neutral per-call trace view."""
    identity = request_identity(original_request)
    accounting = record.get("token_accounting") or {}
    timing = record.get("timing") or {}
    placement = record.get("placement")
    selected_cache = (
        record.get(f"{placement}_cache")
        if placement in {"local", "cloud"}
        else None
    )
    response = record.get("response")
    headers = record.get("headers") or {}
    session_id = headers.get("x-claude-code-session-id")
    agent_id = headers.get("x-claude-code-agent-id")
    if session_id:
        # A real client session is a stronger lineage key than similarity of
        # model/system/tools. Keep it hashed so downstream figures need not
        # expose the raw client identifier.
        identity["lineage_id"] = _stable_hash(
            b"edgeproxy-session-lineage-v1", str(session_id)
        )
    total_input = accounting.get("input_tokens")
    cache_details_available = accounting.get("cache_details_available", False)
    has_prompt = bool(
        isinstance(original_request, dict)
        and (original_request.get("system") or original_request.get("messages"))
    )
    # Several Anthropic-compatible gateways return a literal zero when they do
    # not expose input accounting. Do not label that value exact for a nonempty
    # prompt. Detailed Anthropic/vLLM cache buckets or a positive provider total
    # are sufficient evidence that the field is real.
    prompt_tokens_exact = (
        total_input
        if total_input is not None
        and (cache_details_available or total_input > 0 or not has_prompt)
        else None
    )
    tool_use_blocks = validate_tool_use_blocks(original_request, response)
    return {
        "schema_version": "edgeproxy.call.v1",
        "timestamp_unix_s": record.get("ts"),
        "call_id": record.get("id"),
        "experiment_id": record.get("experiment_id"),
        "episode_id": record.get("episode_id"),
        "session_id": session_id,
        **identity,
        "backend": placement,
        "backend_url": record.get("backend"),
        "http_status": record.get("status"),
        "tokens": {
            "input_tokens": total_input,
            "prompt_tokens_exact": prompt_tokens_exact,
            "output_tokens": accounting.get("output_tokens"),
            "cache_read_tokens": accounting.get("cache_read_input_tokens"),
            "cache_write_tokens": accounting.get("cache_creation_input_tokens"),
            "uncached_input_tokens": accounting.get("uncached_input_tokens"),
            "cache_details_available": cache_details_available,
            "usage_integrity": (
                "complete"
                if prompt_tokens_exact is not None and cache_details_available
                else "partial"
            ),
        },
        "stop_reason": (
            response.get("stop_reason") if isinstance(response, dict) else None
        ),
        "output_limit": {
            "requested_max_tokens": record.get("requested_max_tokens"),
            "effective_max_tokens": record.get("effective_max_tokens"),
            "reserve_tokens": record.get("output_reserve_tokens"),
        },
        "tool_use_blocks": tool_use_blocks,
        "cohort": record.get("cohort_detection"),
        "causality": {
            "agent_id": str(agent_id) if agent_id else None,
            "agent_parent_tool_use_id": _agent_parent_tool_use_id(
                original_request, headers
            ),
            "consumed_tool_result_ids": consumed_tool_result_ids(original_request),
            "produced_tool_use_ids": [
                block["tool_use_id"]
                for block in tool_use_blocks
                if block.get("tool_use_id")
            ],
            # TraceWriter fills these from prior records in the same JSONL.
            "parent_call_ids": [],
            "root_call_id": record.get("id"),
        },
        "timing": {
            "ttft_ms": timing.get("ttft_ms"),
            "decode_tokens_per_s": timing.get("output_tokens_per_s"),
            "tpot_ms": timing.get("tpot_ms"),
            "total_ms": timing.get("total_ms"),
        },
        "cache_probe": selected_cache,
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


class SSEDecoder:
    """Incrementally decode SSE JSON without assuming network chunk boundaries."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        self._buffer.extend(chunk)
        events: list[dict[str, Any]] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(self._buffer[:newline]).strip()
            del self._buffer[: newline + 1]
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

    def finish(self) -> list[dict[str, Any]]:
        if not self._buffer:
            return []
        tail = bytes(self._buffer) + b"\n"
        self._buffer.clear()
        return self.feed(tail)


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
        self._running_path: Path | None = None
        self._running_saved_usd = 0.0
        self._tool_producers: dict[str, str] = {}
        self._call_roots: dict[str, str] = {}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            log.exception("could not create trace dir %s — recording disabled", self.dir)

    def _path(self) -> Path:
        return self.dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"

    def _load_running_total(self, path: Path) -> float:
        """Recover today's cumulative saving after a proxy restart."""
        if not path.exists():
            return 0.0
        last_total: float | None = None
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                        value = (record.get("cost_savings") or {}).get(
                            "running_saved_usd"
                        )
                        if value is not None:
                            last_total = float(value)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        except OSError:
            log.exception("could not recover running cost saving from %s", path)
        return last_total or 0.0

    def _load_causality_index(self, path: Path) -> None:
        self._tool_producers = {}
        self._call_roots = {}
        if not path.exists():
            return
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    call = record.get("call") or {}
                    call_id = call.get("call_id")
                    causality = call.get("causality") or {}
                    if not call_id:
                        continue
                    self._call_roots[str(call_id)] = str(
                        causality.get("root_call_id") or call_id
                    )
                    for tool_id in causality.get("produced_tool_use_ids") or []:
                        self._tool_producers[str(tool_id)] = str(call_id)
        except OSError:
            log.exception("could not recover causality index from %s", path)

    def _link_causality(self, record: dict[str, Any]) -> None:
        call = record.get("call")
        if not isinstance(call, dict) or not call.get("call_id"):
            return
        causality = call.get("causality")
        if not isinstance(causality, dict):
            return
        call_id = str(call["call_id"])
        parents = list(
            dict.fromkeys(
                self._tool_producers[str(tool_id)]
                for tool_id in causality.get("consumed_tool_result_ids") or []
                if str(tool_id) in self._tool_producers
            )
        )
        causality["parent_call_ids"] = parents
        roots = list(
            dict.fromkeys(self._call_roots.get(parent, parent) for parent in parents)
        )
        causality["root_call_id"] = roots[0] if len(roots) == 1 else call_id
        self._call_roots[call_id] = str(causality["root_call_id"])
        for tool_id in causality.get("produced_tool_use_ids") or []:
            self._tool_producers[str(tool_id)] = call_id

    def write(self, record: dict[str, Any]) -> None:
        try:
            with self._lock:
                path = self._path()
                if path != self._running_path:
                    self._running_saved_usd = self._load_running_total(path)
                    self._load_causality_index(path)
                    self._running_path = path
                self._link_causality(record)
                cost = record.get("cost_savings")
                if isinstance(cost, dict):
                    saved = cost.get("request_saved_usd")
                    if saved is not None:
                        self._running_saved_usd += float(saved)
                    cost["running_saved_usd"] = round(self._running_saved_usd, 12)
                line = json.dumps(record, ensure_ascii=False, default=str)
                fh = path.open("a", encoding="utf-8")
                try:
                    fh.write(line + "\n")
                finally:
                    fh.close()
        except Exception:
            log.exception("trace write failed (ignored)")
