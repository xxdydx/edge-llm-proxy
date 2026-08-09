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
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Config, parse_args
from .trace.record import TraceWriter, parse_sse, reassemble, redact_headers

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


def _usage_of(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
        return payload["usage"]
    return {}


def make_app(cfg: Config) -> FastAPI:
    writer = TraceWriter(cfg.trace_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=cfg.upstream,
            # Long generations are normal; only the connect phase should be brisk.
            timeout=httpx.Timeout(600.0, connect=10.0),
            follow_redirects=True,
        )
        log.info("upstream=%s traces=%s", cfg.upstream, cfg.trace_dir)
        try:
            yield
        finally:
            await app.state.client.aclose()

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

        record: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "path": "/" + path,
            "method": request.method,
            "stream": streaming,
            "headers": redact_headers(request.headers),
            "request": request_json,
        }

        client: httpx.AsyncClient = request.app.state.client
        upstream_request = client.build_request(
            request.method,
            "/" + path,
            content=body,
            headers=headers,
            params=request.query_params,
        )

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

        if not streaming:
            payload = await upstream.aread()
            await upstream.aclose()
            try:
                parsed = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            record |= {
                "response": parsed,
                "usage": _usage_of(parsed),
                "timing": {"total_ms": round((time.monotonic() - started) * 1000, 1)},
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
                        "timing": {
                            "ttft_ms": ttft_ms,
                            "total_ms": round((time.monotonic() - started) * 1000, 1),
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
