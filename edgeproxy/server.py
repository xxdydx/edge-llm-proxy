"""Pass-through proxy that records every call.

v0 makes no decisions: it forwards everything upstream unchanged and writes a
copy to disk. Credentials are relayed as headers and never read or stored, so
this works with an API key, an OAuth subscription token, or a Lumid PAT without
knowing which is in play.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import router
from .cloud_cache import (
    CloudCacheObservation,
    CloudCachePrediction,
    CloudCacheTracker,
    PrefixChain,
    cache_scope,
    cloud_cache_trace,
    prefix_chain,
)
from .cost import build_cost_savings
from .config import Config, parse_args
from .local_cache import LocalCachePrediction, local_cache_trace, probe_local_cache
from .shaping import LinkMonitor, LinkShaper
from .telemetry import LocalResourceSampler
from .timing import make_trace_extension
from .trace.record import (
    TraceWriter,
    SSEDecoder,
    build_token_accounting,
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
# Claude Code uses Anthropic's "high" effort spelling. Qwen3.8's chat template
# calls the equivalent highest setting "xhigh" and rejects "high" outright.
LOCAL_REASONING_EFFORT_ALIASES = {"high": "xhigh"}


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

    # This function runs on both the copied pre-routing probe body and the
    # selected local request. Keep the transformation here so both render the
    # same prefix and therefore query the same vLLM cache blocks. Cloud bodies
    # never reach this function and retain Anthropic's original spelling.
    output_config = request_json.get("output_config")
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if effort in LOCAL_REASONING_EFFORT_ALIASES:
            output_config["effort"] = LOCAL_REASONING_EFFORT_ALIASES[effort]

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
    cloud_tracker = CloudCacheTracker()

    policy = router.build(
        cfg.policy,
        max_local_tokens=cfg.max_local_tokens,
        margin=cfg.local_token_margin,
    )

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
        return {
            "status": "ok",
            "upstream": cfg.upstream,
            "trace_dir": str(cfg.trace_dir),
            "cloud_cache_tracking": cfg.cloud_cache_tracking,
            "local_cache_tracking": cfg.local_cache_tracking,
            "max_local_tokens": cfg.max_local_tokens,
            "local_token_margin": cfg.local_token_margin,
            "effective_local_token_budget": int(
                cfg.max_local_tokens * cfg.local_token_margin
            ),
        }

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
        requested_model = (
            str(request_json.get("model") or "")
            if isinstance(request_json, dict)
            else ""
        )
        original_temperature: Any = None
        strict_tools_added = 0
        feature_dict: dict[str, Any] | None = None
        cloud_chain: PrefixChain | None = None
        cloud_prediction: CloudCachePrediction | None = None
        local_prediction: LocalCachePrediction | None = None
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
                if cfg.local_cache_tracking == "observe":
                    # Probe the exact prompt that vLLM would receive. Generation
                    # controls are local-only and the original cloud request
                    # must remain untouched until placement is known.
                    local_probe_request = copy.deepcopy(request_json)
                    _apply_local_generation_controls(local_probe_request)
                    local_probe_request["model"] = cfg.local_model_name
                    local_prediction = await probe_local_cache(
                        request.app.state.clients["local"], local_probe_request
                    )
                    features = replace(
                        features,
                        local_prompt_tokens=local_prediction.input_tokens,
                        local_cache_state=local_prediction.state,
                        estimated_local_cached_tokens=(
                            local_prediction.estimated_read_tokens
                        ),
                        estimated_local_cached_fraction=(
                            local_prediction.estimated_read_fraction
                        ),
                        local_cache_prediction_confidence=(
                            "live-ground-truth"
                            if local_prediction.available
                            else "unavailable"
                        ),
                    )
                if cfg.cloud_cache_tracking == "observe":
                    try:
                        scope = cache_scope(cfg.upstream, headers)
                        cloud_chain = prefix_chain(request_json, scope)
                        cloud_prediction = cloud_tracker.probe(cloud_chain, started)
                        features = replace(
                            features,
                            cloud_cache_state=cloud_prediction.state,
                            estimated_cloud_cached_tokens=cloud_prediction.estimated_read_tokens,
                            estimated_cloud_cached_fraction=(
                                cloud_prediction.estimated_read_fraction
                            ),
                            cloud_cache_expires_in_s=cloud_prediction.expires_in_s,
                            cloud_cache_prediction_confidence=(
                                "confirmed"
                                if cloud_prediction.state == "warm"
                                else "conservative"
                            ),
                        )
                    except Exception:
                        # Observability must never influence placement or break
                        # the request path.
                        log.exception("cloud cache prediction failed (ignored)")
                        cloud_chain = None
                        cloud_prediction = None
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
        if local_prediction is not None:
            record["local_cache"] = local_cache_trace(
                local_prediction, selected=placement == "local"
            )
        if cloud_prediction is not None:
            record["cloud_cache"] = cloud_cache_trace(
                cloud_prediction, selected=placement == "cloud"
            )
        if path.rstrip("/") == "v1/messages":
            # Ensure transport errors and malformed upstream responses still
            # carry an explicit unavailable cost record rather than omitting
            # the field. Successful responses replace this after usage arrives.
            record["cost_savings"] = build_cost_savings(
                placement=placement,
                requested_model=requested_model,
                usage={},
                chain=cloud_chain,
                prediction=cloud_prediction,
            )

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
        response_started_at = conn.response_started_at or time.monotonic()
        net_ms = conn.network_ms
        if placement == "cloud":
            monitor.observe(net_ms)
        record["link"] = {
            "shaping": cfg.shaping,
            **shaper.as_dict(),
            "shaped_ms": shaped_ms or None,
            **monitor.as_dict(),
        }

        cloud_observation: CloudCacheObservation | None = None
        cloud_observed = False

        def observe_cloud_cache(usage: dict[str, Any]) -> None:
            nonlocal cloud_observation, cloud_observed
            if (
                cloud_observed
                or placement != "cloud"
                or cloud_chain is None
                or cloud_prediction is None
            ):
                return
            if not (
                "cache_read_input_tokens" in usage
                or "cache_creation_input_tokens" in usage
            ):
                return
            try:
                cloud_observation = cloud_tracker.observe_cloud_usage(
                    cloud_chain,
                    cloud_prediction,
                    request_started_at=started,
                    response_started_at=response_started_at,
                    status=upstream.status_code,
                    usage=usage,
                )
                cloud_observed = cloud_observation.applied
                record["cloud_cache"] = cloud_cache_trace(
                    cloud_prediction,
                    usage,
                    cloud_observation,
                    selected=True,
                )
            except Exception:
                log.exception("cloud cache observation failed (ignored)")

        if not streaming:
            payload = await upstream.aread()
            await upstream.aclose()
            try:
                parsed = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            usage = _usage_of(parsed)
            observe_cloud_cache(usage)
            if local_prediction is not None:
                record["local_cache"] = local_cache_trace(
                    local_prediction, usage, selected=placement == "local"
                )
            if cloud_prediction is not None:
                record["cloud_cache"] = cloud_cache_trace(
                    cloud_prediction,
                    usage,
                    cloud_observation,
                    selected=placement == "cloud",
                )
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
            if path.rstrip("/") == "v1/messages":
                record["cost_savings"] = build_cost_savings(
                    placement=placement,
                    requested_model=requested_model,
                    usage=usage,
                    chain=cloud_chain,
                    prediction=cloud_prediction,
                )
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
            first_output_at: float | None = None
            last_output_at: float | None = None
            decoder = SSEDecoder()
            decoded_events: list[dict[str, Any]] = []
            try:
                async for chunk in upstream.aiter_bytes():
                    now = time.monotonic()
                    events = decoder.feed(chunk)
                    decoded_events.extend(events)
                    for event in events:
                        if event.get("type") == "message_start":
                            observe_cloud_cache(_usage_of(event.get("message")))
                        if event.get("type") == "content_block_delta":
                            if first_output_at is None:
                                first_output_at = now
                                ttft_ms = round((now - started) * 1000, 1)
                            last_output_at = now
                    accumulated.extend(chunk)
                    yield chunk
            except httpx.HTTPError as exc:
                record["error"] = repr(exc)
                log.warning("stream interrupted on %s: %s", path, exc)
            finally:
                await upstream.aclose()
                try:
                    decoded_events.extend(decoder.finish())
                    message, usage = reassemble(decoded_events)
                    observe_cloud_cache(usage)
                    if local_prediction is not None:
                        record["local_cache"] = local_cache_trace(
                            local_prediction, usage, selected=placement == "local"
                        )
                    output_tokens = usage.get("output_tokens")
                    try:
                        output_tokens = int(output_tokens)
                    except (TypeError, ValueError):
                        output_tokens = None
                    output_duration_ms = (
                        round((last_output_at - first_output_at) * 1000, 1)
                        if first_output_at is not None and last_output_at is not None
                        else None
                    )
                    tpot_ms = (
                        round(output_duration_ms / (output_tokens - 1), 3)
                        if output_duration_ms is not None
                        and output_tokens is not None
                        and output_tokens > 1
                        else None
                    )
                    output_tokens_per_s = (
                        round((output_tokens - 1) * 1000 / output_duration_ms, 3)
                        if output_duration_ms is not None
                        and output_duration_ms > 0
                        and output_tokens is not None
                        and output_tokens > 1
                        else None
                    )
                    if cloud_prediction is not None:
                        record["cloud_cache"] = cloud_cache_trace(
                            cloud_prediction,
                            usage,
                            cloud_observation,
                            selected=placement == "cloud",
                        )
                    record.update({
                        "response": message,
                        "usage": usage,
                        "token_accounting": build_token_accounting(usage),
                        "timing": {
                            "ttft_ms": ttft_ms,
                            "output_duration_ms": output_duration_ms,
                            "tpot_ms": tpot_ms,
                            "output_tokens_per_s": output_tokens_per_s,
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
                    if path.rstrip("/") == "v1/messages":
                        record["cost_savings"] = build_cost_savings(
                            placement=placement,
                            requested_model=requested_model,
                            usage=usage,
                            chain=cloud_chain,
                            prediction=cloud_prediction,
                        )
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
