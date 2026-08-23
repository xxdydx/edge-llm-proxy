"""Pass-through proxy that records every call.

v0 makes no decisions: it forwards everything upstream unchanged and writes a
copy to disk. Credentials are relayed as headers and never read or stored, so
this works with an API key, an OAuth subscription token, or a Lumid PAT without
knowing which is in play.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Config, parse_args
from . import router
from .shaping import LinkMonitor, LinkShaper
from .telemetry import LocalResourceSampler
from .timing import make_trace_extension
from .trace.record import (
    TraceWriter,
    build_token_accounting,
    parse_sse,
    reassemble,
    redact_headers,
)

log = logging.getLogger("edgeproxy")

# Connection-scoped headers must not be forwarded. accept-encoding is dropped
# too so upstream replies uncompressed and the stream tee stays simple.
HOP_BY_HOP = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "te",
    "trailer",
    "accept-encoding",
    "proxy-authorization",
}

# Re-emitting these would contradict the body we actually send back.
RESPONSE_STRIP = {"content-length", "content-encoding", "transfer-encoding", "connection"}

LOCAL_TEMPERATURE = 0


def _apply_local_generation_controls(request_json: dict[str, Any]) -> tuple[Any, int]:
    """Make local sampling deterministic and opt client tools into constraints.

    vLLM only enables schema-constrained decoding for automatic tool choice when
    at least one tool declares ``strict: true``.  Claude Code does not currently
    send that opt-in, so add it at the edge boundary.  Server-side tools never
    reach this function because the router sends them to cloud.

    Returns the original temperature and the number of tools changed so both
    rewrites are visible in the trace.
    """
    original_temperature = request_json.get("temperature")
    request_json["temperature"] = LOCAL_TEMPERATURE

    strict_tools_added = 0
    for tool in request_json.get("tools") or []:
        if (
            isinstance(tool, dict)
            and isinstance(tool.get("input_schema"), dict)
            and tool.get("strict") is not True
        ):
            tool["strict"] = True
            strict_tools_added += 1

    return original_temperature, strict_tools_added


def _usage_of(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
        return payload["usage"]
    return {}


def make_app(cfg: Config) -> FastAPI:
    writer = TraceWriter(cfg.trace_dir)

    policy = router.build(cfg.policy)

    # `netem` means shaping happens outside this process; we record the claim
    # but must not also apply it, or the delay would be counted twice.
    shaper = LinkShaper(
        delay_ms=cfg.cloud_delay_ms if cfg.shaping == "proxy" else 0.0,
        jitter_ms=cfg.cloud_jitter_ms if cfg.shaping == "proxy" else 0.0,
        bandwidth_mbps=cfg.cloud_bandwidth_mbps if cfg.shaping == "proxy" else 0.0,
        preset=cfg.link_preset,
    )
    monitor = LinkMonitor()

    # Last-seen wall clock per Claude Code session, so a call knows how long the
    # gap was. Sessions are few and short-lived; leaking a handful of float
    # entries is cheaper than expiring them.
    last_seen: dict[str, float] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One client per destination. Both speak the Anthropic API, so placement
        # is purely a choice of base URL — nothing downstream needs to know which
        # was picked.
        app.state.clients = {
            name: httpx.AsyncClient(
                base_url=url,
                # Long generations are normal; only connect should be brisk.
                timeout=httpx.Timeout(600.0, connect=10.0),
                follow_redirects=True,
            )
            for name, url in cfg.backends.items()
        }
        app.state.resource_sampler = LocalResourceSampler(
            app.state.clients["local"],
            interval_s=cfg.resource_sample_interval_s,
            gpu_index=cfg.gpu_index,
            kv_bytes_per_token=cfg.kv_bytes_per_token,
        )
        app.state.resource_sampler.start()
        log.info(
            "policy=%s cloud=%s local=%s traces=%s",
            policy.name, cfg.upstream, cfg.vllm_url, cfg.trace_dir,
        )
        try:
            yield
        finally:
            await app.state.resource_sampler.close()
            for client in app.state.clients.values():
                await client.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "upstream": cfg.upstream, "trace_dir": str(cfg.trace_dir)}

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(path: str, request: Request) -> Response:
        started = time.monotonic()
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}

        request_json: Any = None
        if body:
            try:
                request_json = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        streaming = isinstance(request_json, dict) and bool(request_json.get("stream"))

        # Only /v1/messages is a routable call; everything else (count_tokens,
        # /v1/models, health probes) goes to cloud so behaviour is unchanged.
        placement = "cloud"
        reason = "not-routable"
        detail: str | None = None
        clamped_to: int | None = None
        original_model: str | None = None
        original_temperature: Any = None
        strict_tools_added = 0
        feature_dict: dict[str, Any] | None = None
        if path.rstrip("/") == "v1/messages" and isinstance(request_json, dict):
            try:
                session = request.headers.get("x-claude-code-session-id")
                gap = None
                if session:
                    now = time.time()
                    prev = last_seen.get(session)
                    gap = round(now - prev, 1) if prev is not None else None
                    last_seen[session] = now
                features = router.extract_features(request_json, gap)
                feature_dict = asdict(features)
                decision = policy.decide(features)
                placement, reason, detail = decision.placement, decision.reason, decision.detail

                # Local-only rewrites; cloud gets the request exactly as sent.
                if placement == "local":
                    original_temperature, strict_tools_added = (
                        _apply_local_generation_controls(request_json)
                    )

                    if request_json.get("model") != cfg.local_model_name:
                        original_model = request_json.get("model")
                        request_json["model"] = cfg.local_model_name

                    if hasattr(policy, "effective_max_tokens"):
                        want = policy.effective_max_tokens(features)
                        if want != request_json.get("max_tokens"):
                            request_json["max_tokens"] = want
                            clamped_to = want

                    # Always re-serialise: temperature and strict-tool controls
                    # are local-only rewrites even when model/token limits were
                    # already in their desired form.
                    body = json.dumps(request_json).encode()
            except Exception:
                log.exception("router failed — falling back to cloud")
                placement = "cloud"

        record: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "path": "/" + path,
            "method": request.method,
            "stream": streaming,
            "placement": placement,
            "reason": reason,
            "reason_detail": detail,
            "policy": policy.name,
            "clamped_max_tokens": clamped_to,
            "original_model": original_model,
            "original_temperature": original_temperature,
            "strict_tools_added": strict_tools_added,
            "backend": cfg.backends[placement],
            "features": feature_dict,
            "headers": redact_headers(request.headers),
            "request": request_json,
            # Present even on transport/provider errors so downstream analysis
            # can distinguish unavailable usage (null) from measured zero.
            "usage": {},
            "token_accounting": build_token_accounting({}),
        }
        if placement == "local":
            record["local_resources"] = request.app.state.resource_sampler.snapshot()

        client: httpx.AsyncClient = request.app.state.clients[placement]
        trace_ext, read_timing = make_trace_extension()
        upstream_request = client.build_request(
            request.method,
            "/" + path,
            content=body,
            headers=headers,
            params=request.query_params,
            extensions=trace_ext,
        )

        # Uplink cost, cloud only. Local is loopback and gets nothing.
        shaped_ms = await shaper.apply(len(body)) if placement == "cloud" else 0.0

        try:
            upstream = await client.send(upstream_request, stream=streaming)
        except httpx.HTTPError as exc:
            record |= {
                "status": 502,
                "error": repr(exc),
                "timing": {"total_ms": round((time.monotonic() - started) * 1000, 1)},
            }
            writer.write(record)
            log.warning("upstream error on %s: %s", path, exc)
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {"type": "upstream_error", "message": str(exc)},
                },
            )

        out_headers = {
            k: v for k, v in upstream.headers.items() if k.lower() not in RESPONSE_STRIP
        }
        record["status"] = upstream.status_code

        # send() has returned, so response headers are in and the phase stamps
        # are complete. The body has not been read yet.
        conn = read_timing()
        net_ms = conn.network_ms
        if placement == "cloud":
            monitor.observe(net_ms)
        record["link"] = {
            "shaping": cfg.shaping,
            **shaper.as_dict(),
            "shaped_ms": shaped_ms or None,
            **monitor.as_dict(),
        }

        if not streaming:
            payload = await upstream.aread()
            await upstream.aclose()
            try:
                parsed = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            usage = _usage_of(parsed)
            record |= {
                "response": parsed,
                "usage": usage,
                "token_accounting": build_token_accounting(usage),
                "timing": {
                    "total_ms": round((time.monotonic() - started) * 1000, 1),
                    "network_ms": net_ms,
                    **conn.as_dict(),
                },
            }
            writer.write(record)
            return Response(
                content=payload,
                status_code=upstream.status_code,
                headers=out_headers,
                media_type=upstream.headers.get("content-type"),
            )

        async def relay():
            """Forward chunks the instant they arrive, keeping a copy for the trace.

            `record` is mutated via .update() rather than |= — an augmented
            assignment would rebind the name and make it local to this generator.
            """
            accumulated = bytearray()
            ttft_ms: float | None = None
            try:
                async for chunk in upstream.aiter_bytes():
                    if ttft_ms is None and b"content_block_delta" in chunk:
                        ttft_ms = round((time.monotonic() - started) * 1000, 1)
                    accumulated.extend(chunk)
                    yield chunk
            except httpx.HTTPError as exc:
                record["error"] = repr(exc)
                log.warning("stream interrupted on %s: %s", path, exc)
            finally:
                await upstream.aclose()
                try:
                    message, usage = reassemble(parse_sse(bytes(accumulated)))
                    record.update({
                        "response": message,
                        "usage": usage,
                        "token_accounting": build_token_accounting(usage),
                        "timing": {
                            "ttft_ms": ttft_ms,
                            "total_ms": round((time.monotonic() - started) * 1000, 1),
                            "network_ms": net_ms,
                            # Queueing + prefill, with the link taken out. This
                            # is the term a cost model gets fitted against.
                            "server_ttft_ms": (
                                round(ttft_ms - net_ms, 1)
                                if ttft_ms is not None and net_ms is not None
                                else None
                            ),
                            **conn.as_dict(),
                        },
                    })
                except Exception:
                    log.exception("SSE reassembly failed (recording raw length only)")
                    record["response_bytes"] = len(accumulated)
                writer.write(record)

        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            headers=out_headers,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    return app


def main() -> None:
    import uvicorn

    cfg = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )
    uvicorn.run(make_app(cfg), host=cfg.host, port=cfg.port, access_log=False)


if __name__ == "__main__":
    main()
