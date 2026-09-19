"""Pass-through proxy that records every call.

v0 makes no decisions: it forwards everything upstream unchanged and writes a
copy to disk. Credentials are relayed as headers and never read or stored, so
this works with an API key, an OAuth subscription token, or a Lumid PAT without
knowing which is in play.
"""

from __future__ import annotations

import asyncio
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
from .agentic_history import AgenticHistory
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
from .cohort import CohortTracker, agent_delegation_count
from .cohort_parent import SIGNAL_NAME, is_cohort_parent_candidate
from .completion import CompletionEstimator, predict_local_completion
from .coordinator import CohortCoordinator, DispatchTicket
from .local_cache import LocalCachePrediction, local_cache_trace, probe_local_cache
from .reliability import ReliabilityCircuitBreaker
from .shaping import LinkMonitor, LinkShaper
from .telemetry import LocalBackendState, LocalResourceSampler
from .timing import make_trace_extension
from .trace.record import (
    SSEDecoder,
    TraceWriter,
    build_structured_call,
    build_token_accounting,
    redact_headers,
    reassemble,
    request_identity,
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
# Claude Code uses Anthropic's "high" effort spelling; Qwen3.8's chat template
# rejects "high" outright, so it always needs remapping to a value Qwen
# accepts. Originally mapped to "xhigh" (Qwen's actual top tier) to preserve
# rough parity with what "high" means on Claude's own scale. Changed to
# "medium" 2026-09-04 at Arul's request: local decode throughput is a fixed
# ~30 tok/s regardless of effort level (measured, stable across 27+ hours and
# multiple box provisions -- see claude-memory/wiki), so xhigh's extra
# reasoning tokens don't decode any faster, they just add more tokens to
# decode. This is a live hypothesis, not yet validated against task quality
# -- see claude-memory/wiki/decisions/Lower local reasoning effort from
# xhigh to medium.md. Applies to newly-launched jobs only; a running proxy
# process keeps whatever alias was loaded when it started.
LOCAL_REASONING_EFFORT_ALIASES = {"high": "medium"}


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


def _leader_warming_exemption_allowed(
    policy: router.Policy,
    features: router.CallFeatures,
    decision: router.Decision,
) -> bool:
    """Allow only the approved cold-branch preference exemption.

    Re-run StaticPolicy's hard gates directly, then explicitly preserve Rung
    3's headroom gate because BranchDriftPolicy normally evaluates it only
    after WarmLocalPolicy has accepted a call.  No other cloud decision can be
    changed back to local here.
    """
    if decision.reason != "cold-branch-first-turn":
        return False
    if not isinstance(policy, router.WarmLocalPolicy):
        return False
    if router.StaticPolicy.decide(policy, features).placement != "local":
        return False
    if isinstance(policy, router.CombinedPolicy):
        # CombinedPolicy composes all three escalation gates directly (it
        # does not subclass BranchDriftPolicy/PlanningEscalationPolicy/
        # PredictedRiskPolicy), so none of the isinstance branches below
        # would otherwise catch it — re-check all three explicitly here.
        if router.planning_turn_escalation(features, policy.planning_turns) is not None:
            return False
        if router.branch_drift_escalation(
            features, policy.headroom_threshold, policy.budget()
        ) is not None:
            return False
        if router.predicted_risk_escalation(features, policy.risk_threshold) is not None:
            return False
    elif isinstance(policy, router.BranchDriftPolicy):
        if features.local_prompt_tokens is not None:
            ratio = features.local_prompt_tokens / policy.budget()
            if ratio >= policy.headroom_threshold:
                return False
    return True


def make_app(cfg: Config) -> FastAPI:
    writer = TraceWriter(cfg.trace_dir)
    cloud_tracker = CloudCacheTracker()
    cohort_tracker = CohortTracker(window_ms=cfg.cohort_window_ms)
    cohort_coordinator = CohortCoordinator(
        window_ms=cfg.cohort_window_ms,
        timeout_ms=cfg.cohort_barrier_timeout_ms,
        poll_interval_ms=cfg.cohort_barrier_poll_ms,
    )
    cohort_detection_enabled = (
        cfg.cohort_tracking == "observe" or cfg.cohort_parent_placement
    )
    reliability = ReliabilityCircuitBreaker()
    completion_estimator = CompletionEstimator()
    local_backend_state = LocalBackendState(
        concurrency_limit=cfg.local_concurrency_limit
    )
    # vLLM mixes cache_salt into the first prefix-block hash without changing
    # prompt rendering or token length.  Episode scope prevents sequential A/B
    # conditions from warming each other; request scope is the bag-of-requests
    # ablation in which even byte-identical prefixes cannot share KV blocks.
    local_cache_namespace = cfg.episode_id or f"process-{uuid.uuid4()}"

    policy = router.build(
        cfg.policy,
        max_local_tokens=cfg.max_local_tokens,
        margin=cfg.local_token_margin,
        output_reserve_tokens=cfg.local_output_reserve_tokens,
    )
    agentic_history = AgenticHistory() if cfg.agentic_shadow_artifact is not None else None
    agentic_shadow = None
    if cfg.agentic_shadow_artifact is not None:
        scorer = router.ArtifactHarmScorer.from_json(
            cfg.agentic_shadow_artifact,
            lambda f: f.agentic_shadow_features or {},
        )
        agentic_shadow = router.AdaptiveAgenticPolicy(
            max_local_tokens=cfg.max_local_tokens,
            margin=cfg.local_token_margin,
            output_reserve_tokens=cfg.local_output_reserve_tokens,
            harm_scorer=scorer,
            baseline_policy=policy,
            shadow=True,
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
    # Stable, cheap request-time state for the future cohort planner.  The
    # resource sampler itself is attached during lifespan startup.
    app.state.local_backend_state = local_backend_state
    app.state.completion_estimator = completion_estimator
    app.state.agentic_history = agentic_history
    app.state.cohort_coordinator = cohort_coordinator
    # Exposed so a test (or a future operator endpoint) can inspect or, for
    # determinism, reseed the recovery-probe RNG. Nothing on the request path
    # reads it back from here — `proxy()` closes over `reliability` directly.
    app.state.reliability = reliability

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "upstream": cfg.upstream,
            "trace_dir": str(cfg.trace_dir),
            "experiment_id": cfg.experiment_id,
            "episode_id": cfg.episode_id,
            "cohort_tracking": cfg.cohort_tracking,
            "cohort_window_ms": cfg.cohort_window_ms,
            "cohort_parent_placement": cfg.cohort_parent_placement,
            "cohort_barrier_timeout_ms": cfg.cohort_barrier_timeout_ms,
            "cohort_barrier_poll_ms": cfg.cohort_barrier_poll_ms,
            "cloud_cache_tracking": cfg.cloud_cache_tracking,
            "local_cache_tracking": cfg.local_cache_tracking,
            "local_cache_salt_scope": cfg.local_cache_salt_scope,
            "max_local_tokens": cfg.max_local_tokens,
            "local_token_margin": cfg.local_token_margin,
            "local_output_reserve_tokens": cfg.local_output_reserve_tokens,
            "local_concurrency_limit": cfg.local_concurrency_limit,
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
        call_id = str(uuid.uuid4())
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}

        request_json: Any = None
        if body:
            try:
                request_json = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        original_request_json = copy.deepcopy(request_json)

        streaming = isinstance(request_json, dict) and bool(request_json.get("stream"))

        # Only /v1/messages is a routable call; everything else (count_tokens,
        # /v1/models, health probes) goes to cloud so behaviour is unchanged.
        placement = "cloud"
        reason = "not-routable"
        detail: str | None = None
        clamped_to: int | None = None
        requested_max_tokens = (
            int(request_json.get("max_tokens") or 0)
            if isinstance(request_json, dict)
            else None
        )
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
        cohort_detection: dict[str, Any] | None = None
        tool_suite_hash: str | None = None
        reliability_note: str | None = None
        features: router.CallFeatures | None = None
        local_resources_at_decision: dict[str, Any] | None = None
        local_completion_prediction: dict[str, Any] | None = None
        local_completion_prediction_eligible = False
        local_dispatch_concurrency: int | None = None
        local_probe_request: dict[str, Any] | None = None
        local_cache_salt: str | None = None
        agentic_history_key: str | None = None
        agentic_history_sequence: int | None = None
        agentic_shadow_record: dict[str, Any] | None = None
        dispatch_ticket: DispatchTicket | None = None
        cohort_dispatch: dict[str, Any] | None = None
        if path.rstrip("/") == "v1/messages" and isinstance(request_json, dict):
            try:
                if cfg.local_cache_salt_scope == "condition":
                    local_cache_salt = local_cache_namespace
                elif cfg.local_cache_salt_scope == "request":
                    local_cache_salt = f"{local_cache_namespace}:{call_id}"
                session = request.headers.get("x-claude-code-session-id")
                if cohort_detection_enabled:
                    cohort_detection = cohort_tracker.match_child(
                        session_id=session,
                        request=original_request_json,
                        arrived_at_unix_s=time.time(),
                    )
                gap = None
                if session:
                    now = time.time()
                    prev = last_seen.get(session)
                    gap = round(now - prev, 1) if prev is not None else None
                    last_seen[session] = now
                features = replace(
                    router.extract_features(request_json, gap),
                    local_token_budget=int(
                        cfg.max_local_tokens * cfg.local_token_margin
                    ),
                )
                if agentic_history is not None:
                    # The proxy is per episode in the SWE-bench harness. Keep
                    # client sessions and explicit subagent IDs in separate
                    # bounded lanes; do not share their previous placements.
                    # Without a client session ID, a shared "no-session"
                    # bucket could mix unrelated roots. Withhold the score.
                    if session:
                        metadata = request_json.get("metadata")
                        parent_id = (
                            request.headers.get("x-claude-code-parent-tool-use-id")
                            or request.headers.get("x-parent-tool-use-id")
                            or request_json.get("parent_tool_use_id")
                            or (metadata.get("parent_tool_use_id") if isinstance(metadata, dict) else None)
                        )
                        agentic_history_key = "|".join((
                            str(cfg.episode_id or "process"),
                            str(session),
                            str(request.headers.get("x-claude-code-agent-id") or "main"),
                            str(parent_id or "no-parent"),
                        ))
                        try:
                            agentic_history_sequence = agentic_history.begin(agentic_history_key)
                        except RuntimeError:
                            agentic_history_key = None

                # Rung 1: resolve the circuit-breaker check to a plain bool
                # here, where I/O and mutable state are allowed, so decide()
                # stays a pure function of features. class_key/reliability_note
                # are closed over below by finalize_structured_call() and the
                # record dict.
                tool_suite_hash = request_identity(request_json).get(
                    "tool_suite_hash"
                )
                reliability_blocked, reliability_note = reliability.should_block(
                    tool_suite_hash
                )
                features = replace(
                    features, local_reliability_blocked=reliability_blocked
                )

                if cfg.local_cache_tracking == "observe":
                    # Probe the exact prompt that vLLM would receive. Generation
                    # controls are local-only and the original cloud request
                    # must remain untouched until placement is known.
                    local_probe_request = copy.deepcopy(request_json)
                    _apply_local_generation_controls(local_probe_request)
                    local_probe_request["model"] = cfg.local_model_name
                    if local_cache_salt is not None:
                        local_probe_request["cache_salt"] = local_cache_salt
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
                    features = replace(
                        features,
                        predicted_local_risk_score=(
                            router.predict_local_risk_score(features)
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
                if agentic_history is not None and agentic_shadow is not None:
                    snapshot = None
                    unsupported = "history-lane-unavailable"
                    if agentic_history_key is not None and agentic_history_sequence is not None:
                        snapshot, unsupported = agentic_history.snapshot(
                            agentic_history_key, agentic_history_sequence
                        )
                    if snapshot is not None:
                        if features.local_prompt_tokens is None or not features.local_token_budget:
                            unsupported = "exact-local-prompt-unavailable"
                        else:
                            shadow_features = {
                                **snapshot,
                                "local_prompt_tokens": float(features.local_prompt_tokens),
                                "n_available_tools": float(features.n_tools),
                                "errored_tool_result_density": float(features.errored_tool_result_density),
                                "branch_turn_ordinal": float(features.branch_turn_ordinal or 0),
                                "context_utilization_ratio": float(features.local_prompt_tokens / features.local_token_budget),
                            }
                            features = replace(features, agentic_shadow_features=shadow_features)
                            unsupported = None
                    shadow_decision = agentic_shadow.decide(features)
                    agentic_shadow_record = {
                        "artifact": str(cfg.agentic_shadow_artifact),
                        "baseline_policy": policy.name,
                        "baseline_placement": shadow_decision.placement,
                        "reason": shadow_decision.reason,
                        "detail": shadow_decision.detail,
                        "feature_status": "complete" if unsupported is None else "unavailable",
                        "unavailable_reason": unsupported,
                        "feature_provenance": "current-request-and-completed-prior-calls-before-dispatch",
                    }
                    feature_dict = asdict(features)
                if cfg.cohort_parent_placement and cohort_detection is not None:
                    dispatch_ticket = cohort_coordinator.arrive(
                        call_id=call_id,
                        detection=cohort_detection,
                        input_tokens=(
                            local_prediction.input_tokens
                            if local_prediction is not None
                            else None
                        ),
                        arrived_at_monotonic=started,
                    )
                # This is a synchronous cached read immediately before the
                # placement decision: no metrics I/O is added to routing.
                local_resources_at_decision = local_backend_state.snapshot(
                    request.app.state.resource_sampler.snapshot()
                )
                decision = policy.decide(features)
                placement, reason, detail = decision.placement, decision.reason, decision.detail
                if (
                    cfg.cohort_parent_placement
                    and cohort_detection is None
                    and is_cohort_parent_candidate(original_request_json, headers)
                ):
                    placement = "cloud"
                    reason = "cohort-parent-placement"
                    detail = None

                cohort_ready = bool(
                    dispatch_ticket is not None
                    and int(cohort_detection.get("expected_width") or 0) >= 2
                )
                warming_exemption = bool(
                    cohort_ready
                    and dispatch_ticket is not None
                    and dispatch_ticket.role == "leader"
                    and _leader_warming_exemption_allowed(policy, features, decision)
                )
                if warming_exemption:
                    placement = "local"
                    reason = "cohort-warming-leader"
                    detail = "exempted cold-branch-first-turn preference gate"
                if cohort_ready and dispatch_ticket is not None:
                    cohort_dispatch = cohort_coordinator.trace_for(
                        dispatch_ticket,
                        policy_placement=decision.placement,
                        policy_reason=decision.reason,
                        warming_exemption=warming_exemption,
                    )

                # Local-only rewrites; cloud gets the request exactly as sent.
                if placement == "local":
                    original_temperature, strict_tools_added = (
                        _apply_local_generation_controls(request_json)
                    )

                    if request_json.get("model") != cfg.local_model_name:
                        original_model = request_json.get("model")
                        request_json["model"] = cfg.local_model_name
                    if local_cache_salt is not None:
                        request_json["cache_salt"] = local_cache_salt

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

        # Rung 4 step 4 is observe-only. Gate on the final placement, after
        # the pure policy chain, the separate benchmark cohort-parent override,
        # and any safe router fallback. Cloud calls never receive a local
        # completion estimate. A local class with insufficient live history is
        # traced explicitly as null rather than guessed.
        if placement == "local" and features is not None:
            local_completion_prediction_eligible = True
            prediction = predict_local_completion(
                features,
                tool_suite_hash,
                local_backend_state,
                completion_estimator,
            )
            if prediction is not None:
                local_completion_prediction = prediction.as_trace()

        # The adjacent coordinator changes only dispatch timing after the final
        # placement is known.  No lock is held across these awaits: cache probes
        # and sleeps yield normally to unrelated FastAPI requests.
        if cohort_dispatch is not None and dispatch_ticket is not None:
            if placement != "local":
                cohort_dispatch["release_reason"] = "not-held-cloud-gate"
            elif dispatch_ticket.role == "leader":
                cohort_coordinator.mark_leader_dispatched(
                    dispatch_ticket,
                    input_tokens=(
                        local_prediction.input_tokens
                        if local_prediction is not None
                        else None
                    ),
                )
            elif local_probe_request is not None:
                await cohort_coordinator.hold_follower(
                    dispatch_ticket,
                    client=request.app.state.clients["local"],
                    request_json=local_probe_request,
                    initial_prediction=local_prediction,
                    trace=cohort_dispatch,
                )
            else:
                cohort_dispatch["release_reason"] = "timeout"

        if local_resources_at_decision is None:
            local_resources_at_decision = local_backend_state.snapshot(
                request.app.state.resource_sampler.snapshot()
            )

        record: dict[str, Any] = {
            "id": call_id,
            "ts": time.time(),
            "path": "/" + path,
            "method": request.method,
            "stream": streaming,
            "placement": placement,
            "reason": reason,
            "reason_detail": detail,
            "policy": policy.name,
            "experiment_id": cfg.experiment_id,
            "episode_id": cfg.episode_id,
            "clamped_max_tokens": clamped_to,
            "requested_max_tokens": requested_max_tokens,
            "effective_max_tokens": (
                request_json.get("max_tokens")
                if isinstance(request_json, dict)
                else None
            ),
            "output_reserve_tokens": cfg.local_output_reserve_tokens,
            "original_model": original_model,
            "original_temperature": original_temperature,
            "strict_tools_added": strict_tools_added,
            "backend": cfg.backends[placement],
            "features": feature_dict,
            "agentic_shadow": agentic_shadow_record,
            "reliability": {
                "class_key": tool_suite_hash,
                "note": reliability_note,
                "failure_rate": reliability.failure_rate(tool_suite_hash),
            },
            "cohort_detection": cohort_detection,
            "cohort_dispatch": cohort_dispatch,
            "headers": redact_headers(request.headers),
            "request": request_json,
            # Present even on transport/provider errors so downstream analysis
            # can distinguish unavailable usage (null) from measured zero.
            "usage": {},
            "token_accounting": build_token_accounting({}),
            # Always present, including cloud placements.  The exact proxy
            # count and sampled vLLM running/waiting values describe the state
            # seen immediately before policy.decide() for routable calls.
            "local_resources": local_resources_at_decision,
            "local_cache_salt_scope": cfg.local_cache_salt_scope,
        }
        if reason == "cohort-parent-placement":
            record["cohort_parent_placement"] = {
                "candidate": True,
                "signal": SIGNAL_NAME,
                "outcome": "pending",
                "did_fan_out": None,
                "agent_delegation_count": None,
            }
        if local_completion_prediction_eligible:
            record["local_completion_prediction"] = local_completion_prediction

        def finalize_structured_call() -> None:
            try:
                if cohort_detection_enabled:
                    # Streaming Agent blocks are normally registered at their
                    # content_block_stop event. Keep this reconciliation for
                    # non-streaming responses and unusual/incomplete streams;
                    # CohortTracker deduplicates the shared tool-use IDs.
                    parent_detection = cohort_tracker.observe_parent(
                        call_id=str(record["id"]),
                        session_id=(record.get("headers") or {}).get(
                            "x-claude-code-session-id"
                        ),
                        backend=record.get("placement"),
                        response=record.get("response"),
                        completed_at_unix_s=time.time(),
                    )
                    if parent_detection is not None:
                        record["cohort_detection"] = parent_detection
                record["call"] = build_structured_call(record, original_request_json)
            except Exception:
                # Trace enrichment must never interrupt a proxied response.
                log.exception("structured call trace failed (ignored)")
            if agentic_history is not None and agentic_history_key is not None and agentic_history_sequence is not None:
                try:
                    if record.get("error") is not None or not isinstance(record.get("response"), dict):
                        agentic_history.abort(agentic_history_key, agentic_history_sequence)
                    else:
                        agentic_history.complete(
                            agentic_history_key,
                            agentic_history_sequence,
                            request=original_request_json,
                            errored_tool_result_density=(
                                features.errored_tool_result_density if features is not None else 0.0
                            ),
                            placement=str(record.get("placement") or ""),
                            response=record["response"],
                            tool_use_blocks=list(((record.get("call") or {}).get("tool_use_blocks")) or []),
                        )
                except Exception:
                    log.exception("agentic shadow history update failed (ignored)")
            try:
                parent_placement = record.get("cohort_parent_placement")
                response = record.get("response")
                if isinstance(parent_placement, dict):
                    if isinstance(response, dict):
                        delegation_count = agent_delegation_count(response)
                        did_fan_out = delegation_count > 0
                        parent_placement.update(
                            {
                                "outcome": (
                                    "true_positive"
                                    if did_fan_out
                                    else "false_positive"
                                ),
                                "did_fan_out": did_fan_out,
                                "agent_delegation_count": delegation_count,
                            }
                        )
                    else:
                        parent_placement["outcome"] = "unknown"
            except Exception:
                log.exception("cohort parent-placement outcome failed (ignored)")
            try:
                # Rung 1: close the loop for *future* calls in this class.
                # A local call with no tool_use blocks (pure text/thinking,
                # or ended via end_turn) has no schema-validity evidence
                # either way and vacuously counts as a success -- but only
                # when the call actually completed. A transport failure
                # (record["error"] set) or an SSE reassembly failure (no
                # "response" dict; relay() falls back to recording only
                # "response_bytes") also leaves tool_use_blocks empty, and
                # without this guard both would be recorded as false
                # successes, diluting real failures enough to keep a
                # genuinely unreliable class's circuit closed. Found by
                # Codex code review, see claude-memory/wiki/decisions/Codex
                # code review of Rung 1 and Rung 3 diff.md.
                response_ok = (
                    isinstance(record.get("response"), dict)
                    and record.get("error") is None
                )
                if (
                    record.get("placement") == "local"
                    and tool_suite_hash is not None
                    and response_ok
                ):
                    blocks = ((record.get("call") or {}).get("tool_use_blocks")) or []
                    success = not any(
                        block.get("schema_valid") is False for block in blocks
                    )
                    reliability.record(tool_suite_hash, success)
            except Exception:
                log.exception("reliability circuit breaker update failed (ignored)")
            try:
                # Advance deployment-specific EWMAs only from real, completed
                # local outcomes. Output length is available for non-streaming
                # calls too; TPOT advances only when the existing stream timing
                # instrumentation measured it.
                response_ok = (
                    isinstance(record.get("response"), dict)
                    and record.get("error") is None
                )
                if (
                    record.get("placement") == "local"
                    and tool_suite_hash is not None
                    and response_ok
                ):
                    raw_output_tokens = (record.get("usage") or {}).get(
                        "output_tokens"
                    )
                    try:
                        measured_output_tokens = int(raw_output_tokens)
                    except (TypeError, ValueError):
                        measured_output_tokens = None
                    completion_estimator.record(
                        tool_suite_hash,
                        output_tokens=measured_output_tokens,
                        tpot_ms=(record.get("timing") or {}).get("tpot_ms"),
                        concurrency=local_dispatch_concurrency,
                    )
            except Exception:
                log.exception("completion estimator update failed (ignored)")
            try:
                if cohort_dispatch is not None:
                    actual_read = ((record.get("local_cache") or {}).get("actual") or {}).get(
                        "cache_read_input_tokens"
                    )
                    cohort_dispatch["realized_outcome"] = {
                        "status": record.get("status"),
                        "error": record.get("error"),
                        "actual_cache_read_tokens": actual_read,
                        "leader_completed": (
                            (
                                isinstance(record.get("response"), dict)
                                and record.get("error") is None
                            )
                            if dispatch_ticket is not None
                            and dispatch_ticket.role == "leader"
                            else None
                        ),
                    }
            except Exception:
                log.exception("cohort dispatch outcome tracing failed (ignored)")

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

        local_request_lease = (
            local_backend_state.begin_request() if placement == "local" else None
        )
        if local_request_lease is not None:
            local_dispatch_concurrency = (
                local_request_lease.requests_in_flight_at_dispatch
            )
        request_task = asyncio.current_task()
        if local_request_lease is not None and request_task is not None:
            # Final backstop for an unexpected exception anywhere after
            # acquisition. Normal paths release earlier; the lease is
            # idempotent, so task completion cannot double-decrement.
            request_task.add_done_callback(
                lambda _task: local_request_lease.release()
            )

        upstream: httpx.Response | None = None
        try:
            upstream = await client.send(upstream_request, stream=streaming)
        except httpx.HTTPError as exc:
            record |= {
                "status": 502,
                "error": repr(exc),
                "timing": {"total_ms": round((time.monotonic() - started) * 1000, 1)},
            }
            finalize_structured_call()
            writer.write(record)
            log.warning("upstream error on %s: %s", path, exc)
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {"type": "upstream_error", "message": str(exc)},
                },
            )
        finally:
            # Includes the handled HTTPError path, unexpected exceptions, and
            # task cancellation while waiting for response headers.
            if local_request_lease is not None and upstream is None:
                local_request_lease.release()

        assert upstream is not None

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
            try:
                payload = await upstream.aread()
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
                    "token_accounting": build_token_accounting(
                        usage,
                        observed_input_tokens=(feature_dict or {}).get(
                            "local_prompt_tokens"
                        ),
                        observed_input_source="local_render_probe",
                    ),
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
                finalize_structured_call()
                writer.write(record)
                return Response(
                    content=payload,
                    status_code=upstream.status_code,
                    headers=out_headers,
                    media_type=upstream.headers.get("content-type"),
                )
            finally:
                try:
                    await upstream.aclose()
                finally:
                    if local_request_lease is not None:
                        local_request_lease.release()

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
            block_events: dict[int, list[dict[str, Any]]] = {}
            saw_message_stop = False

            def process_stream_event(event: dict[str, Any], now: float) -> None:
                nonlocal first_output_at, last_output_at, ttft_ms, saw_message_stop
                event_type = event.get("type")
                if event_type == "message_stop":
                    saw_message_stop = True
                if event_type == "message_start":
                    observe_cloud_cache(_usage_of(event.get("message")))
                if event_type == "content_block_start":
                    block_events[event.get("index", 0)] = [event]
                elif event_type == "content_block_delta":
                    block_events.setdefault(event.get("index", 0), []).append(event)
                    if first_output_at is None:
                        first_output_at = now
                        ttft_ms = round((now - started) * 1000, 1)
                    last_output_at = now
                elif event_type == "content_block_stop":
                    events_for_block = block_events.pop(event.get("index", 0), None)
                    if not events_for_block or not cohort_detection_enabled:
                        return
                    try:
                        block_message, _ = reassemble(events_for_block)
                        content = block_message.get("content") or []
                        block = content[0] if content else None
                        tool_input = (
                            block.get("input") if isinstance(block, dict) else None
                        )
                        prompt = (
                            tool_input.get("prompt")
                            if isinstance(tool_input, dict)
                            else None
                        )
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_use"
                            and block.get("name") == "Agent"
                            and block.get("id")
                            and isinstance(prompt, str)
                            and prompt
                        ):
                            parent_detection = cohort_tracker.register_delegation(
                                call_id=str(record["id"]),
                                session_id=(record.get("headers") or {}).get(
                                    "x-claude-code-session-id"
                                ),
                                backend=record.get("placement"),
                                tool_use_id=str(block["id"]),
                                prompt=prompt,
                                observed_at_unix_s=time.time(),
                            )
                            if parent_detection is not None:
                                record["cohort_detection"] = parent_detection
                    except Exception:
                        # Cohort observation must never interrupt the stream.
                        log.exception(
                            "streamed cohort delegation registration failed (ignored)"
                        )

            try:
                async for chunk in upstream.aiter_bytes():
                    now = time.monotonic()
                    events = decoder.feed(chunk)
                    decoded_events.extend(events)
                    for event in events:
                        process_stream_event(event, now)
                    accumulated.extend(chunk)
                    yield chunk
            except httpx.HTTPError as exc:
                record["error"] = repr(exc)
                log.warning("stream interrupted on %s: %s", path, exc)
            except (asyncio.CancelledError, GeneratorExit) as exc:
                record["error"] = f"StreamAborted({type(exc).__name__})"
                raise
            except Exception as exc:
                record["error"] = f"StreamAborted({type(exc).__name__}: {exc})"
                raise
            finally:
                try:
                    try:
                        await upstream.aclose()
                        tail_events = decoder.finish()
                        decoded_events.extend(tail_events)
                        for event in tail_events:
                            process_stream_event(event, time.monotonic())
                        # Reassembly can produce a plausible response from a
                        # prefix. Anthropic's message_stop is the only end-of-
                        # message proof for a normally exhausted SSE stream.
                        if not saw_message_stop and record.get("error") is None:
                            record["error"] = "StreamIncomplete(message_stop missing)"
                        record["stream_complete"] = saw_message_stop and record.get("error") is None
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
                            "token_accounting": build_token_accounting(
                                usage,
                                observed_input_tokens=(feature_dict or {}).get(
                                    "local_prompt_tokens"
                                ),
                                observed_input_source="local_render_probe",
                            ),
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
                    finalize_structured_call()
                    writer.write(record)
                finally:
                    if local_request_lease is not None:
                        local_request_lease.release()

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
