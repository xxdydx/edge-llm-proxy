"""Observe-only shadow of Anthropic's prompt cache.

The provider does not expose cache residency before a request.  This module
therefore models only entries that the proxy has later seen confirmed by
Anthropic usage.  Prefix identity is independent of token estimation: hashes
are exact over the request structure we receive, while token depths are
explicitly estimates used to reconcile provider usage with a candidate.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit


KEY_SCHEMA_VERSION = 1
LOOKBACK_BLOCKS = 20
CACHE_TTLS = {"5m": 300.0, "1h": 3600.0}
DEFAULT_CACHE_TTL_S = CACHE_TTLS["5m"]


def _sha256(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _semantic_json(value: Any) -> bytes:
    """Canonical JSON for broad lineage grouping.

    Object order is not semantically meaningful for the tools/system cohort
    key.  Array order remains meaningful and is preserved by json.dumps.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _ordered_json(value: Any) -> bytes:
    """Stable serialization that preserves order-sensitive payload maps."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")


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


def _ttl_of(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    control = value.get("cache_control")
    if not isinstance(control, dict) or control.get("type") != "ephemeral":
        return None
    return CACHE_TTLS.get(str(control.get("ttl") or "5m"))


def _cache_config(request: Mapping[str, Any]) -> dict[str, Any]:
    """Conservative cache-affecting configuration seed.

    Some fields invalidate only later prompt sections on Anthropic.  Seeding
    the entire chain with them can create false-cold predictions, but cannot
    create the more dangerous false-warm prediction.
    """
    keys = (
        "thinking",
        "effort",
        "tool_choice",
        "disable_parallel_tool_use",
        "citations",
        "speed",
    )
    return {key: request[key] for key in keys if key in request}


def cache_scope(upstream: str, headers: Mapping[str, str]) -> str:
    """Return an opaque cache namespace without retaining a credential."""
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    credential = lowered.get("x-api-key") or lowered.get("authorization") or "anonymous"
    split = urlsplit(upstream)
    origin = f"{split.scheme.lower()}://{split.netloc.lower()}"
    return _sha256(b"edgeproxy-cloud-scope-v1", origin.encode(), credential.encode())


def static_lineage_key(request: Mapping[str, Any]) -> str:
    """Group calls with the same semantic tools/system static prefix."""
    static = {
        "tools": _without_cache_control(request.get("tools") or []),
        "system": _without_cache_control(request.get("system") or []),
    }
    return _sha256(b"edgeproxy-static-lineage-v1", _semantic_json(static))


@dataclass(frozen=True)
class PromptElement:
    section: str
    position: int
    payload: Any
    breakpoint_ttl_s: float | None = None


@dataclass(frozen=True)
class PrefixPoint:
    index: int
    key: str
    estimated_tokens: int
    section: str


@dataclass(frozen=True)
class CacheBreakpoint:
    point_index: int
    ttl_s: float
    automatic: bool = False


@dataclass(frozen=True)
class PrefixChain:
    scope_hash: str
    lineage_key: str
    model: str
    points: tuple[PrefixPoint, ...]
    breakpoints: tuple[CacheBreakpoint, ...]
    valid: bool = True
    disabled_reason: str | None = None

    @property
    def estimated_total_tokens(self) -> int:
        return self.points[-1].estimated_tokens if self.points else 0


@dataclass(frozen=True)
class CloudCachePrediction:
    state: str
    reason: str
    scope_hash: str
    lineage_key: str
    matched_prefix_hash: str | None = None
    matched_point_index: int | None = None
    estimated_read_tokens: int | None = None
    estimated_read_fraction: float | None = None
    expires_in_s: float | None = None
    ttl_s: float | None = None

    def as_trace(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "matched_prefix_hash": self.matched_prefix_hash,
            "matched_point_index": self.matched_point_index,
            "estimated_read_tokens": self.estimated_read_tokens,
            "estimated_read_fraction": self.estimated_read_fraction,
            "expires_in_s": self.expires_in_s,
            "ttl_s": self.ttl_s,
        }


@dataclass
class CloudCacheEntry:
    key: str
    estimated_prefix_tokens: int
    ttl_s: float
    created_at: float
    last_accessed_at: float
    expires_at: float
    source: str
    lineage_key: str


@dataclass(frozen=True)
class CloudCacheObservation:
    applied: bool
    outcome: str
    mapped_prefix_hash: str | None = None
    mapped_estimated_tokens: int | None = None
    invalidated_prefix_hash: str | None = None
    entries_created: int = 0
    reason: str | None = None

    def as_trace(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "outcome": self.outcome,
            "mapped_prefix_hash": self.mapped_prefix_hash,
            "mapped_estimated_tokens": self.mapped_estimated_tokens,
            "invalidated_prefix_hash": self.invalidated_prefix_hash,
            "entries_created": self.entries_created,
            "reason": self.reason,
        }


def prompt_elements(request: Mapping[str, Any]) -> list[PromptElement]:
    """Flatten cacheable request content in Anthropic render order."""
    elements: list[PromptElement] = []

    def append(section: str, payload: Any) -> None:
        elements.append(
            PromptElement(
                section=section,
                position=len(elements),
                payload=_without_cache_control(payload),
                breakpoint_ttl_s=_ttl_of(payload),
            )
        )

    for tool in request.get("tools") or []:
        if isinstance(tool, dict):
            append("tools", tool)

    system = request.get("system")
    if isinstance(system, str):
        append("system", {"type": "text", "text": system})
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                append("system", block)

    for message_index, message in enumerate(request.get("messages") or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if not isinstance(blocks, list):
            continue
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            # Role and message position prevent identical text in different
            # conversational locations from sharing an identity accidentally.
            wrapped = {
                "message_index": message_index,
                "block_index": block_index,
                "role": role,
                "content": block,
            }
            append("messages", wrapped)
            # cache_control belongs to the content block, not our wrapper.
            if _ttl_of(block) is not None:
                elements[-1] = PromptElement(
                    section="messages",
                    position=elements[-1].position,
                    payload=_without_cache_control(wrapped),
                    breakpoint_ttl_s=_ttl_of(block),
                )
    return elements


def minimum_cacheable_tokens(model: str) -> int | None:
    """Current documented Anthropic minimums, matched conservatively."""
    name = model.lower()
    if any(part in name for part in ("fable-5", "mythos-5")):
        return 512
    if any(part in name for part in ("opus-4-7", "mythos-preview")):
        return 2048
    if any(part in name for part in ("opus-4-6", "opus-4-5", "haiku-4-5")):
        return 4096
    if "sonnet" in name or "opus-4-8" in name:
        return 1024
    return None


def prefix_chain(request: Mapping[str, Any], scope: str) -> PrefixChain:
    elements = prompt_elements(request)
    model = str(request.get("model") or "")
    lineage = static_lineage_key(request)
    seed = _sha256(
        b"edgeproxy-anthropic-prefix-v1",
        scope.encode(),
        model.encode(),
        _ordered_json(_cache_config(request)),
    )
    points: list[PrefixPoint] = []
    breakpoints: list[CacheBreakpoint] = []
    cumulative_chars = 0
    previous = seed
    for element in elements:
        encoded = _ordered_json(
            {
                "section": element.section,
                "position": element.position,
                "payload": element.payload,
            }
        )
        cumulative_chars += len(encoded)
        previous = _sha256(previous.encode(), encoded)
        points.append(
            PrefixPoint(
                index=element.position,
                key=previous,
                estimated_tokens=max(1, cumulative_chars // 4),
                section=element.section,
            )
        )
        if element.breakpoint_ttl_s is not None:
            breakpoints.append(CacheBreakpoint(element.position, element.breakpoint_ttl_s))

    top_ttl = _ttl_of(request)
    if top_ttl is not None and points:
        target = points[-1].index
        existing = next((bp for bp in breakpoints if bp.point_index == target), None)
        if existing is None:
            breakpoints.append(CacheBreakpoint(target, top_ttl, automatic=True))
        elif existing.ttl_s != top_ttl:
            return PrefixChain(
                scope,
                lineage,
                model,
                tuple(points),
                tuple(breakpoints),
                False,
                "conflicting-automatic-breakpoint-ttl",
            )

    breakpoints.sort(key=lambda bp: bp.point_index)
    if len(breakpoints) > 4:
        return PrefixChain(
            scope, lineage, model, tuple(points), tuple(breakpoints), False, "too-many-breakpoints"
        )
    # Anthropic requires longer-lived entries before shorter-lived ones.
    seen_short = False
    for bp in breakpoints:
        if bp.ttl_s == DEFAULT_CACHE_TTL_S:
            seen_short = True
        elif seen_short:
            return PrefixChain(
                scope, lineage, model, tuple(points), tuple(breakpoints), False, "invalid-ttl-order"
            )
    return PrefixChain(scope, lineage, model, tuple(points), tuple(breakpoints))


def _usage_count(usage: Mapping[str, Any], key: str) -> int | None:
    if key not in usage:
        return None
    try:
        value = int(usage[key])
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _candidate_points(chain: PrefixChain) -> Iterable[tuple[PrefixPoint, CacheBreakpoint]]:
    seen: set[str] = set()
    for bp in reversed(chain.breakpoints):
        low = max(0, bp.point_index - (LOOKBACK_BLOCKS - 1))
        for index in range(bp.point_index, low - 1, -1):
            point = chain.points[index]
            if point.key in seen:
                continue
            seen.add(point.key)
            yield point, bp


@dataclass
class CloudCacheTracker:
    entries: dict[str, CloudCacheEntry] = field(default_factory=dict)
    seen_lineages: set[tuple[str, str, str]] = field(default_factory=set)
    token_scale_samples: dict[tuple[str, str], list[float]] = field(default_factory=dict)

    def _token_scale(self, chain: PrefixChain) -> float:
        samples = self.token_scale_samples.get((chain.scope_hash, chain.model), [])
        return statistics.median(samples) if samples else 1.0

    def _learn_token_scale(self, chain: PrefixChain, point: PrefixPoint, actual: int) -> None:
        if point.estimated_tokens <= 0 or actual <= 0:
            return
        ratio = actual / point.estimated_tokens
        # Bad mappings should not poison later estimates indefinitely.
        if not 0.1 <= ratio <= 10.0:
            return
        key = (chain.scope_hash, chain.model)
        samples = self.token_scale_samples.setdefault(key, [])
        samples.append(ratio)
        del samples[:-100]

    def prune(self, now: float) -> int:
        expired = [key for key, entry in self.entries.items() if entry.expires_at <= now]
        for key in expired:
            self.entries.pop(key, None)
        return len(expired)

    def probe(self, chain: PrefixChain, now: float) -> CloudCachePrediction:
        if not chain.valid:
            return CloudCachePrediction(
                "disabled",
                chain.disabled_reason or "invalid-chain",
                chain.scope_hash,
                chain.lineage_key,
            )
        if not chain.breakpoints:
            return CloudCachePrediction(
                "disabled", "no-cache-breakpoints", chain.scope_hash, chain.lineage_key
            )

        expired_candidate = False
        matches: list[tuple[PrefixPoint, CloudCacheEntry]] = []
        for point, _ in _candidate_points(chain):
            entry = self.entries.get(point.key)
            if entry is None:
                continue
            if entry.expires_at <= now:
                expired_candidate = True
                self.entries.pop(point.key, None)
                continue
            matches.append((point, entry))

        if matches:
            point, entry = max(matches, key=lambda item: item[0].index)
            total = chain.estimated_total_tokens * self._token_scale(chain)
            cached_tokens = entry.estimated_prefix_tokens
            fraction = cached_tokens / total if total else None
            return CloudCachePrediction(
                state="warm",
                reason="confirmed-active-entry",
                scope_hash=chain.scope_hash,
                lineage_key=chain.lineage_key,
                matched_prefix_hash=point.key,
                matched_point_index=point.index,
                estimated_read_tokens=cached_tokens,
                estimated_read_fraction=round(fraction, 6) if fraction is not None else None,
                expires_in_s=round(max(0.0, entry.expires_at - now), 3),
                ttl_s=entry.ttl_s,
            )

        lineage_scope = (chain.scope_hash, chain.model, chain.lineage_key)
        known = lineage_scope in self.seen_lineages
        return CloudCachePrediction(
            "cold" if (known or expired_candidate) else "unknown",
            (
                "expired-entry"
                if expired_candidate
                else ("known-lineage-no-entry" if known else "first-seen")
            ),
            chain.scope_hash,
            chain.lineage_key,
            estimated_read_tokens=0 if known or expired_candidate else None,
            estimated_read_fraction=0.0 if known or expired_candidate else None,
        )

    def observe_cloud_usage(
        self,
        chain: PrefixChain,
        prediction: CloudCachePrediction,
        *,
        request_started_at: float,
        response_started_at: float,
        status: int,
        usage: Mapping[str, Any],
    ) -> CloudCacheObservation:
        if status < 200 or status >= 300:
            return CloudCacheObservation(False, "unavailable", reason="unsuccessful-response")
        cache_read = _usage_count(usage, "cache_read_input_tokens")
        cache_creation = _usage_count(usage, "cache_creation_input_tokens")
        if cache_read is None and cache_creation is None:
            return CloudCacheObservation(False, "unavailable", reason="cache-usage-unavailable")

        self.seen_lineages.add((chain.scope_hash, chain.model, chain.lineage_key))
        invalidated: str | None = None
        mapped: PrefixPoint | None = None

        if cache_read and cache_read > 0:
            candidates = list(_candidate_points(chain))
            if prediction.matched_prefix_hash:
                mapped = next(
                    (
                        point
                        for point, _ in candidates
                        if point.key == prediction.matched_prefix_hash
                    ),
                    None,
                )
            if mapped is None and candidates:
                mapped, mapped_bp = min(
                    candidates,
                    key=lambda item: (
                        abs(
                            item[0].estimated_tokens * self._token_scale(chain) - cache_read
                        ),
                        -item[0].index,
                    ),
                )
                self._learn_token_scale(chain, mapped, cache_read)
                entry = self.entries.get(mapped.key)
                ttl_s = entry.ttl_s if entry else mapped_bp.ttl_s
                self.entries[mapped.key] = CloudCacheEntry(
                    mapped.key,
                    cache_read,
                    ttl_s,
                    request_started_at,
                    request_started_at,
                    request_started_at + ttl_s,
                    "confirmed-read-adopted",
                    chain.lineage_key,
                )
            elif mapped is not None:
                self._learn_token_scale(chain, mapped, cache_read)
                entry = self.entries.get(mapped.key)
                if entry is not None:
                    # A read reports the provider's exact token depth for this
                    # confirmed prefix. Retain it for later predictions instead
                    # of freezing the rough structural-size estimate.
                    entry.estimated_prefix_tokens = cache_read
                    entry.last_accessed_at = request_started_at
                    entry.expires_at = request_started_at + entry.ttl_s
                    entry.source = "confirmed-read"
        elif prediction.matched_prefix_hash:
            invalidated = prediction.matched_prefix_hash
            self.entries.pop(invalidated, None)

        created = 0
        if cache_creation and cache_creation > 0 and chain.valid:
            for bp in chain.breakpoints:
                point = chain.points[bp.point_index]
                if point.key in self.entries:
                    # A mixed read/write response must not relabel or extend a
                    # confirmed read from the later response-start timestamp.
                    continue
                calibrated_tokens = max(
                    1, round(point.estimated_tokens * self._token_scale(chain))
                )
                self.entries[point.key] = CloudCacheEntry(
                    point.key,
                    calibrated_tokens,
                    bp.ttl_s,
                    response_started_at,
                    response_started_at,
                    response_started_at + bp.ttl_s,
                    "confirmed-creation",
                    chain.lineage_key,
                )
                created += 1

        if cache_read and cache_creation:
            outcome = "read-and-write"
        elif cache_read:
            outcome = "read"
        elif cache_creation:
            outcome = "write"
        else:
            outcome = "uncached"
        mapped_tokens = (
            self.entries[mapped.key].estimated_prefix_tokens
            if mapped is not None and mapped.key in self.entries
            else (mapped.estimated_tokens if mapped else None)
        )
        return CloudCacheObservation(
            True,
            outcome,
            mapped.key if mapped else None,
            mapped_tokens,
            invalidated,
            created,
        )


def cloud_cache_trace(
    prediction: CloudCachePrediction,
    usage: Mapping[str, Any] | None = None,
    observation: CloudCacheObservation | None = None,
) -> dict[str, Any]:
    usage = usage or {}
    read = _usage_count(usage, "cache_read_input_tokens")
    creation = _usage_count(usage, "cache_creation_input_tokens")
    details_available = read is not None or creation is not None
    uncached = _usage_count(usage, "input_tokens") if details_available else None
    breakdown = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else {}
    actual_warm = read is not None and read > 0
    predicted_warm = prediction.state == "warm"
    token_error = (
        abs((prediction.estimated_read_tokens or 0) - read) if read is not None else None
    )
    return {
        "schema_version": KEY_SCHEMA_VERSION,
        "mode": "observe",
        "scope_hash": prediction.scope_hash,
        "static_lineage_key": prediction.lineage_key,
        "prediction": prediction.as_trace(),
        "actual": {
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": creation,
            "uncached_input_tokens": uncached,
            "creation_5m_input_tokens": _usage_count(breakdown, "ephemeral_5m_input_tokens"),
            "creation_1h_input_tokens": _usage_count(breakdown, "ephemeral_1h_input_tokens"),
        },
        "agreement": {
            "warm_prediction_correct": predicted_warm == actual_warm if read is not None else None,
            "cached_token_error": token_error,
        },
        "observation": observation.as_trace() if observation else None,
    }
