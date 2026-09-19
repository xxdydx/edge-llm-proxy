"""Strict dual executor: replay one captured request against one real
backend and prove, from the response itself, which backend actually
answered. No fallback exists anywhere in this module -- a failed identity
check is a recorded outcome, never a silently-accepted guess.

Three checks, all mandatory:
  - preflight(): a cheap probe against each backend before the batch starts,
    with its raw evidence persisted to disk.
  - every real replay: the response's own `model` field is checked against
    an *exact* allowlist value before the outcome is trusted as "OK" -- an
    identity mismatch marks that one sample INVALID, and is never silently
    accepted.
  - postflight(): the same probe repeated after the batch and persisted, so
    a relay/endpoint that changed identity mid-run is caught, not assumed.

Streaming is used (not a single JSON POST) specifically to recover real
TTFT/TPOT/TPS -- a non-streaming call cannot distinguish "slow to start"
from "slow to finish", and this experiment's artifacts require both,
explicitly null rather than guessed when genuinely unavailable.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from edgeproxy.server import _apply_local_generation_controls
from edgeproxy.trace.record import SSEDecoder, reassemble

from .config import REPO_ROOT, BackendConfig
from .schema import ReplayOutcome, TeacherCall

# Reuse the project's own safe .env loader (declares key *names* into
# os.environ, never prints values) instead of re-implementing it.
sys.path.insert(0, str(REPO_ROOT / "eval-suite" / "runner"))
from run_eval import load_dotenv_into_environ  # noqa: E402


class BackendIdentityError(RuntimeError):
    """Raised when a backend's response identity cannot be trusted -- at
    preflight/postflight this aborts the run outright rather than proceeding
    on an unverified endpoint."""


def _auth_headers(backend: BackendConfig) -> dict[str, str]:
    if backend.auth_header is None or backend.auth_env_var is None:
        return {}
    token = os.environ.get(backend.auth_env_var)
    if not token:
        return {}
    if backend.auth_header == "authorization":
        return {"authorization": f"Bearer {token}"}
    return {backend.auth_header: token}


def _identity_ok(backend: BackendConfig, response: dict[str, Any]) -> tuple[bool, str]:
    """Exact allowlist match, not a substring check -- a response whose
    `model` is anything other than exactly the expected value fails,
    including e.g. `"local-something"` that a substring check would have
    let through."""
    model = str(response.get("model") or "")
    if not model:
        return False, "response carried no model field"
    if model != backend.expected_model_exact:
        return False, f"model={model!r} != expected {backend.expected_model_exact!r}"
    return True, f"model={model!r} confirmed"


def _probe_input_tokens(backend: BackendConfig, request: dict[str, Any]) -> int | None:
    """Exact pre-request input-token count from the same patched vLLM
    endpoint production routing uses (`edgeproxy.local_cache.probe_local_cache`'s
    `/v1/messages/count_cached_tokens` contract, called synchronously here
    rather than importing the async version). None on any probe failure --
    the caller falls back to sending the historical `max_tokens` unclamped,
    and a genuinely oversized request still gets caught deterministically by
    `_identity`/capacity error detection in `_build_outcome`."""
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            resp = client.post(f"{backend.base_url}/v1/messages/count_cached_tokens", json=request)
        resp.raise_for_status()
        value = resp.json().get("input_tokens")
        return int(value) if value is not None else None
    except (httpx.HTTPError, ValueError, TypeError):
        return None


# Same 0.9 safety margin `edgeproxy/config.py`'s `EDGEPROXY_LOCAL_TOKEN_MARGIN`
# default uses in every production/campaign launch this project runs.
_LOCAL_TOKEN_MARGIN = 0.9


@dataclass
class CapacityPrecheck:
    input_tokens: int | None
    budget: int | None
    fits: bool  # False only when the precheck is conclusive and negative;
                # True also covers "probe unavailable, cannot conclude" so
                # the real call still runs and vLLM's own error is the
                # deterministic backstop.
    clamp: dict[str, Any] | None
    detail: str


def capacity_precheck(backend: BackendConfig, payload: dict[str, Any]) -> CapacityPrecheck:
    """Exact rendered prompt token count where feasible, checked *before*
    the real generation POST -- not inferred only from vLLM's error text on
    a failed call. Two outcomes when the probe succeeds:
      - the prompt ALONE already exceeds the budget: conclusively
        INVALID_CAPACITY, and the real generation call is never sent;
      - it fits, but the historical `max_tokens` combined with it would not:
        clamp `max_tokens` down to the real remaining headroom (same
        computation as `WarmLocalPolicy.effective_max_tokens`; the prompt
        itself is never touched).
    When the probe itself fails, this cannot conclude either way, so
    `fits=True` and the real call still runs -- `_build_outcome`'s
    capacity-error-text detection remains the deterministic backstop for
    exactly that case."""
    if backend.max_context_tokens is None:
        return CapacityPrecheck(None, None, True, None, "backend has no context limit configured")
    input_tokens = _probe_input_tokens(backend, payload)
    if input_tokens is None:
        return CapacityPrecheck(None, None, True, None, "token-count probe unavailable; deferring to live capacity error detection")
    budget = int(backend.max_context_tokens * _LOCAL_TOKEN_MARGIN)
    if input_tokens > budget:
        return CapacityPrecheck(
            input_tokens, budget, False, None,
            f"prompt alone ({input_tokens} tokens) exceeds {backend.name}'s budget ({budget} tokens)",
        )
    headroom = max(0, budget - input_tokens)
    requested = payload.get("max_tokens")
    clamp = None
    if isinstance(requested, int) and requested > headroom:
        payload["max_tokens"] = headroom
        clamp = {"input_tokens": input_tokens, "budget": budget, "original_max_tokens": requested, "clamped_max_tokens": headroom}
    return CapacityPrecheck(input_tokens, budget, True, clamp, "fits")


def _sanitize_for_backend(backend: BackendConfig, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (payload, transform_record). Only the local backend gets
    `_apply_local_generation_controls` -- the exact same transform real
    production local traffic goes through (imported from `edgeproxy.server`,
    never reimplemented), so replay compatibility issues (e.g. a reasoning
    `effort` value this box doesn't support) are fixed the same way live
    traffic already handles them, not papered over ad hoc."""
    payload = copy.deepcopy(request)
    payload.pop("stream", None)
    transform: dict[str, Any] = {}
    if backend.apply_local_generation_controls:
        original_temperature, strict_tools_added = _apply_local_generation_controls(payload)
        transform = {
            "original_temperature": original_temperature,
            "strict_tools_added": strict_tools_added,
        }
    if backend.request_model:
        payload["model"] = backend.request_model
    payload["stream"] = True
    return payload, transform


# Diagnosed 2026-09-09: the run-20260909T072022Z concurrent replay hit ~40%
# TRANSPORT_ERROR across all three backends the moment cloud+local_27b+
# local_7b were dispatched simultaneously per call. Root cause: every one of
# these connections is short-lived and freshly established per request --
# cloud_deepseek makes a fresh httpx connection (fresh DNS lookup for
# "lum.id") per call, and BOTH local relays tunnel through `socat ...
# fork EXEC:ssh <alias> stdio_relay.py`, which spawns a brand-new `ssh`
# process per TCP connection. Concurrent replay put up to 3 such connections
# up at once, multiplying how often the already-documented flaky local DNS
# resolver / SSH-relay drop gets hit simultaneously. Observed failure
# signatures: `ConnectError: [Errno 8] nodename nor servname provided` for
# cloud, and `RemoteProtocolError: Server disconnected without sending a
# response` for both locals.
#
# Bounded fail-fast policy (2026-09-10, replaces the earlier 4-attempt /
# httpx.Timeout(300) design that let a single wedged call block the batch for
# ~20 minutes). A transport failure must resolve fast and be checkpointed for
# a later `--resume-run-dir` pass, never wait:
#   - connect: <=10s to establish the TCP/TLS/SSH-relay pipe.
#   - read: 90s of NO network progress (no bytes received) ends the attempt.
#     This is an INACTIVITY cap, not a total wall-clock cap -- every received
#     SSE chunk resets it, so a healthy generation that keeps streaming
#     tokens is never cut off even if the full turn runs many minutes.
#   - at most 2 same-backend attempts total, one short fixed backoff between.
# Retries fire ONLY on a transport-level failure, never on a real HTTP
# response (even a 4xx/5xx -- that is a genuine answer), and never by falling
# back to a different backend.
_CONNECT_TIMEOUT_S = 10.0
_READ_INACTIVITY_TIMEOUT_S = 90.0
_TRANSPORT_MAX_ATTEMPTS = 2
_TRANSPORT_RETRY_BACKOFF_S = 2.0

# Rate-limit (HTTP 429) retry is deliberately separate from the transport
# retry above: a 429 is a real HTTP response (the "retries fire only on a
# transport-level failure, never on a real HTTP response" rule above is about
# ordinary 4xx/5xx application errors, which ARE a genuine answer never to be
# retried) -- but 429 specifically MEANS "retry later", so it gets its own
# longer-patience exponential backoff, honoring the server's own Retry-After
# header when present instead of guessing. Capped both in attempts and in
# per-wait duration so a persistently saturated pool fails clearly rather
# than blocking indefinitely.
_RATE_LIMIT_MAX_ATTEMPTS = 5
_RATE_LIMIT_BASE_BACKOFF_S = 5.0
_RATE_LIMIT_MAX_BACKOFF_S = 60.0


def _stream_post(backend: BackendConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Calls `_stream_post_once`, retrying on two independent conditions:
      - a transport-level failure: up to `_TRANSPORT_MAX_ATTEMPTS` times,
        short fixed backoff (unchanged from before rate-limit handling);
      - an HTTP 429: up to `_RATE_LIMIT_MAX_ATTEMPTS` times, exponential
        backoff honoring the response's own `Retry-After` header when given,
        else `_RATE_LIMIT_BASE_BACKOFF_S * 2**n` capped at
        `_RATE_LIMIT_MAX_BACKOFF_S`.
    A real non-429 HTTP status (2xx/4xx/5xx) is still never retried -- that
    remains a genuine answer. Returns the last attempt's raw dict, enriched
    with:
      retry_attempts          -- attempts actually made
      attempts_meta           -- per-attempt {attempt, transport_err,
                                 status_code, attempt_wall_s}, oldest first
      end_to_end_wall_s       -- wall across all attempts + backoff sleeps
      final_attempt_latency_s -- just the last attempt's own duration
    """
    attempts_meta: list[dict[str, Any]] = []
    overall_t0 = time.monotonic()
    result: dict[str, Any] | None = None
    rate_limit_attempt = 0
    attempt = 0
    while True:
        attempt += 1
        result = _stream_post_once(backend, payload)
        attempts_meta.append(
            {
                "attempt": attempt,
                "transport_err": result["transport_err"],
                "status_code": result["status_code"],
                "attempt_wall_s": round(result["t_end"] - result["t0"], 4),
            }
        )
        if result["transport_err"] is not None:
            if len(attempts_meta) < _TRANSPORT_MAX_ATTEMPTS:
                time.sleep(_TRANSPORT_RETRY_BACKOFF_S)
                continue
            break
        if result["status_code"] == 429:
            rate_limit_attempt += 1
            if rate_limit_attempt < _RATE_LIMIT_MAX_ATTEMPTS:
                backoff = result.get("retry_after_s")
                if backoff is None:
                    backoff = min(
                        _RATE_LIMIT_BASE_BACKOFF_S * (2 ** (rate_limit_attempt - 1)),
                        _RATE_LIMIT_MAX_BACKOFF_S,
                    )
                time.sleep(backoff)
                continue
        break
    overall_t_end = time.monotonic()
    result["retry_attempts"] = len(attempts_meta)
    result["rate_limit_attempts"] = rate_limit_attempt
    result["attempts_meta"] = attempts_meta
    result["end_to_end_wall_s"] = overall_t_end - overall_t0
    result["final_attempt_latency_s"] = result["t_end"] - result["t0"]
    return result


def _stream_post_once(
    backend: BackendConfig, payload: dict[str, Any]
) -> dict[str, Any]:
    """POST with streaming, timing TTFT and decode window from the raw SSE
    events. Returns a dict the caller turns into a ReplayOutcome; never
    raises for an ordinary HTTP/transport failure -- those are captured in
    the returned dict's `transport_err`/`status_code` fields."""
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        **backend.extra_headers,
        **_auth_headers(backend),
    }
    decoder = SSEDecoder()
    events: list[dict[str, Any]] = []
    t0 = time.monotonic()
    ttft_s: float | None = None
    last_delta_t: float | None = None
    status_code: int | None = None
    transport_err: str | None = None
    raw_error_body: str | None = None
    retry_after_s: float | None = None

    # read/write are INACTIVITY caps (no bytes for this long -> ReadTimeout),
    # not a total wall-clock cap: httpx resets them on every received chunk,
    # so a healthy long-running stream is never aborted for taking a while --
    # only genuine no-progress stalls trip it. There is deliberately no
    # overall timeout here.
    stream_timeout = httpx.Timeout(
        _READ_INACTIVITY_TIMEOUT_S,
        connect=_CONNECT_TIMEOUT_S,
        write=_READ_INACTIVITY_TIMEOUT_S,
        pool=_CONNECT_TIMEOUT_S,
    )
    try:
        with httpx.Client(timeout=stream_timeout) as client:
            with client.stream("POST", f"{backend.base_url}/v1/messages", json=payload, headers=headers) as resp:
                status_code = resp.status_code
                if status_code != 200:
                    raw_error_body = resp.read().decode(errors="replace")[:2000]
                    if status_code == 429:
                        header_val = resp.headers.get("retry-after")
                        if header_val is not None:
                            try:
                                retry_after_s = float(header_val)
                            except ValueError:
                                pass  # non-numeric (HTTP-date form) -- fall back to our own backoff schedule
                else:
                    for chunk in resp.iter_bytes():
                        now = time.monotonic()
                        new_events = decoder.feed(chunk)
                        for ev in new_events:
                            if ev.get("type") == "content_block_delta":
                                if ttft_s is None:
                                    ttft_s = now - t0
                                last_delta_t = now
                        events.extend(new_events)
                    events.extend(decoder.finish())
    except httpx.HTTPError as exc:
        transport_err = f"{type(exc).__name__}: {exc}"

    t_end = time.monotonic()
    return {
        "status_code": status_code,
        "transport_err": transport_err,
        "raw_error_body": raw_error_body,
        "retry_after_s": retry_after_s,
        "events": events,
        "t0": t0,
        "t_end": t_end,
        "ttft_s": ttft_s,
        "last_delta_t": last_delta_t,
    }


def _transport_meta(raw: dict[str, Any]) -> dict[str, Any]:
    """Bounded fail-fast accounting fields, pulled off the raw dict `_stream_post`
    enriched. `.get` with None defaults so canned test fixtures / older
    checkpoints that predate these keys still build a valid ReplayOutcome."""
    return {
        "retry_attempts": raw.get("retry_attempts"),
        "end_to_end_wall_s": raw.get("end_to_end_wall_s"),
        "final_attempt_latency_s": raw.get(
            "final_attempt_latency_s",
            (raw["t_end"] - raw["t0"]) if "t_end" in raw and "t0" in raw else None,
        ),
        "attempts_meta": raw.get("attempts_meta"),
    }


def _build_outcome(backend: BackendConfig, raw: dict[str, Any]) -> ReplayOutcome:
    meta = _transport_meta(raw)
    if raw["transport_err"] is not None:
        return ReplayOutcome(
            backend=backend.name, status="TRANSPORT_ERROR", response=None,
            latency_s=raw["t_end"] - raw["t0"], detail=raw["transport_err"], **meta,
        )
    if raw["status_code"] != 200:
        body = raw["raw_error_body"] or ""
        # vLLM's own authoritative rejection for a request whose exact
        # rendered prompt exceeds this backend's real context window --
        # distinguished from a generic HTTP error so a capacity-excluded
        # sample is never conflated with a genuine execution failure, silently
        # truncated, or (per the fixed "no fallback" rule) silently retried
        # against a different backend.
        capacity_signatures = (
            "maximum context length",
            "cannot be greater than max_model_len",
            "max_total_tokens",
        )
        if raw["status_code"] == 400 and any(sig in body.lower() for sig in capacity_signatures):
            return ReplayOutcome(
                backend=backend.name, status="INVALID_CAPACITY", response=None,
                latency_s=raw["t_end"] - raw["t0"],
                detail=f"exceeds {backend.name}'s real context window: {body}", **meta,
            )
        return ReplayOutcome(
            backend=backend.name, status="HTTP_ERROR", response=None,
            latency_s=raw["t_end"] - raw["t0"],
            detail=f"HTTP {raw['status_code']}: {body}", **meta,
        )
    message, usage = reassemble(raw["events"])
    total_latency_s = raw["t_end"] - raw["t0"]
    output_tokens = usage.get("output_tokens")
    decode_s = (
        raw["t_end"] - raw["ttft_s"] - raw["t0"]
        if raw["ttft_s"] is not None
        else None
    )
    # Fold timing/usage into the response dict so downstream code (labels,
    # artifact writers) has one place to read everything from.
    enriched = dict(message)
    enriched["_usage"] = usage
    enriched["_timing"] = {
        "total_latency_s": total_latency_s,
        "ttft_s": raw["ttft_s"],
        "tpot_ms": (
            (decode_s * 1000.0 / output_tokens)
            if decode_s is not None and output_tokens
            else None
        ),
        "tps": (decode_s and output_tokens and output_tokens / decode_s) or None,
    }

    ok, detail = _identity_ok(backend, enriched)
    if not ok:
        return ReplayOutcome(
            backend=backend.name, status="IDENTITY_MISMATCH", response=enriched,
            latency_s=total_latency_s, detail=detail, **meta,
        )
    return ReplayOutcome(
        backend=backend.name, status="OK", response=enriched,
        latency_s=total_latency_s, detail=detail, **meta,
    )


def preflight(backend: BackendConfig) -> dict[str, Any]:
    """Cheap identity probe. Raises BackendIdentityError -- caller does not
    proceed to the real batch on failure; there is no fallback backend.
    Returns the raw evidence dict for persistence."""
    payload = {
        "model": backend.request_model or "claude-sonnet-5",
        "messages": [{"role": "user", "content": "reply with the single word: ping"}],
        "max_tokens": 4,
        "stream": True,
    }
    raw = _stream_post(backend, payload)
    outcome = _build_outcome(backend, raw)
    evidence = {
        "backend": backend.name,
        "status": outcome.status,
        "detail": outcome.detail,
        "response_model": (outcome.response or {}).get("model") if outcome.response else None,
        "timestamp_monotonic": raw["t0"],
    }
    if outcome.status != "OK":
        raise BackendIdentityError(f"{backend.name} preflight failed: {outcome.status} {outcome.detail}")
    return evidence


def postflight(backend: BackendConfig) -> dict[str, Any]:
    """Identical check, repeated after the batch."""
    return preflight(backend)


def replay_call(call: TeacherCall, backend: BackendConfig) -> tuple[ReplayOutcome, dict[str, Any]]:
    payload, transform = _sanitize_for_backend(backend, call.request)

    precheck = capacity_precheck(backend, payload)
    transform["capacity_precheck"] = {
        "input_tokens": precheck.input_tokens, "budget": precheck.budget,
        "fits": precheck.fits, "detail": precheck.detail,
    }
    if precheck.clamp is not None:
        transform["capacity_clamp"] = precheck.clamp
    if not precheck.fits:
        # Conclusive precheck failure: the real generation POST is never
        # sent -- this is not inferred after the fact from an error string.
        outcome = ReplayOutcome(
            backend=backend.name, status="INVALID_CAPACITY", response=None,
            latency_s=0.0, detail=precheck.detail,
        )
        return outcome, transform

    raw = _stream_post(backend, payload)
    outcome = _build_outcome(backend, raw)
    return outcome, transform


def load_env() -> None:
    """Populate os.environ from the repo's .env (key names only, values
    never leave the process) -- required before any cloud replay call."""
    load_dotenv_into_environ(REPO_ROOT / ".env")


# --------------------------------------------------------------- checkpoint --


def checkpoint_key(call_id: str, backend_name: str) -> str:
    return f"{call_id}::{backend_name}"


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Existing raw records keyed by (call_id, backend) -- resuming a run
    skips any pair already present here instead of spending a duplicate
    call.

    Append safety: a process killed mid-write can leave the checkpoint
    file's final line truncated (a single `write()` of a short line is not
    guaranteed atomic across a hard kill). A malformed line -- the last one
    or, defensively, any other -- is skipped rather than raising, so one
    interrupted write cannot corrupt every already-completed row before it.
    """
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "call_id" not in row or "backend" not in row:
                continue
            done[checkpoint_key(row["call_id"], row["backend"])] = row
    return done


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    """One `write()` call per row: on POSIX, a `write()` of a size this
    small either lands in the file whole or (only under a hard kill
    mid-syscall) leaves a truncated trailing line -- `load_checkpoint`
    skips exactly that case. Never rewrites or reorders earlier lines, so a
    crash here cannot corrupt rows already durably appended."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


# Statuses NOT in this set are treated as retryable on resume: only a
# transient, connection-level failure is worth re-attempting against the
# SAME backend. A real HTTP answer (OK, INVALID_CAPACITY, HTTP_ERROR) or an
# identity failure is never re-attempted -- retrying those would either
# waste a call on a deterministic outcome or risk masking a real problem
# behind a lucky retry.
TERMINAL_STATUSES = frozenset({"OK", "INVALID_CAPACITY", "HTTP_ERROR", "IDENTITY_MISMATCH"})


def retryable_checkpoint(checkpoint: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Rows to treat as "already done" when resuming: everything except
    TRANSPORT_ERROR. A TRANSPORT_ERROR row is left out of the returned
    dict entirely -- it is not deleted from disk (the file is append-only,
    see `append_checkpoint`), just excluded from the "skip this" set, so
    the caller re-attempts it and appends a new, superseding row rather
    than mutating or removing the old one."""
    return {key: row for key, row in checkpoint.items() if row.get("status") in TERMINAL_STATUSES}


def validate_resume_checkpoint(calls: list[TeacherCall], checkpoint: dict[str, dict[str, Any]]) -> None:
    """Resume/fingerprint mismatch rejection: a checkpoint file is only
    safe to resume against the *same* dataset it was built from -- if it
    references call_ids that are not in the current dataset (e.g. the
    dataset was rebuilt with a different sample), resuming would silently
    skip calls that were never actually replayed for this run. Raises
    rather than silently proceeding; the caller must not catch this to
    fall back to a fresh run automatically -- that decision is the
    operator's, not this function's."""
    if not checkpoint:
        return
    current_ids = {c.call_id for c in calls}
    checkpoint_ids = {row["call_id"] for row in checkpoint.values()}
    unknown = checkpoint_ids - current_ids
    if unknown:
        sample = sorted(unknown)[:5]
        raise BackendIdentityError(
            f"checkpoint references {len(unknown)} call_id(s) not present in the current "
            f"dataset (e.g. {sample}) -- this checkpoint does not match the current dataset "
            "and cannot be safely resumed against it"
        )
