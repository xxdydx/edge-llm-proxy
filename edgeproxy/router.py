"""Per-call placement: local (vLLM) or cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

Placement = Literal["local", "cloud"]


@dataclass(frozen=True)
class Decision:
    placement: Placement
    reason: str
    detail: str | None = None

CHARS_PER_TOKEN = 4  # just a rough estimate
# Claude Code's security monitor sends a long transcript but asks for only a
# 64-token decision. Qwen3.8 repeatedly exhausts that output budget before
# returning its decision, so this special control-plane call stays cloud-side.
SECURITY_MONITOR_MARKER = "You are a security monitor for autonomous"


@dataclass(frozen=True)
class CallFeatures:
    """What the router knows at decision time."""

    model: str
    has_tools: bool
    n_tools: int
    has_server_tools: bool
    n_messages: int
    est_system_tokens: int
    max_tokens: int
    stream: bool
    is_tool_continuation: bool
    is_security_monitor: bool = False
    seconds_since_last_call: float | None = None
    # Exact local rendering and live prefix residency from the patched vLLM
    # probe. The static policy uses exact input length when it is available;
    # cache-aware preference remains a later policy decision.
    local_prompt_tokens: int | None = None
    local_cache_state: str | None = None
    estimated_local_cached_tokens: int | None = None
    estimated_local_cached_fraction: float | None = None
    local_cache_prediction_confidence: str | None = None
    # TTL this request asked for, and how many breakpoints it set. Both are
    # properties of the request body, so they are recoverable from any trace.
    cloud_cache_ttl_s: float | None = None
    n_cache_breakpoints: int = 0
    # Observe-only cloud cache estimates. Current policies deliberately ignore
    # these until live validation supports a separate promotion decision.
    cloud_cache_state: str | None = None
    estimated_cloud_cached_tokens: int | None = None
    estimated_cloud_cached_fraction: float | None = None
    cloud_cache_expires_in_s: float | None = None
    cloud_cache_prediction_confidence: str | None = None


def _text_len(content: Any) -> int:
    """Character count of Anthropic content, which may be a string or blocks."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            for key in ("text", "thinking", "partial_json"):
                if isinstance(block.get(key), str):
                    total += len(block[key])
            if isinstance(block.get("content"), (str, list)):
                total += _text_len(block["content"])
            if isinstance(block.get("input"), dict):
                total += len(str(block["input"]))
        return total
    return 0


def _text_contains(content: Any, marker: str) -> bool:
    """Whether an Anthropic text/block value contains ``marker``."""
    if isinstance(content, str):
        return marker in content
    if isinstance(content, list):
        return any(_text_contains(block, marker) for block in content)
    if isinstance(content, dict):
        return any(_text_contains(value, marker) for value in content.values())
    return False


# Anthropic prompt caching. A breakpoint is `cache_control: {"type":
# "ephemeral"}`, which lives 5 minutes; `{"ttl": "1h"}` lives an hour. Those are
# the only two values, the 1h form is GA (no beta header), and the choice is
# per-breakpoint — so the TTL of a call is read off the request, not assumed.
# Claude Code sets no explicit ttl on any breakpoint in the recorded corpus,
# which means the 5-minute default; other clients need not.
CACHE_TTLS = {"5m": 300.0, "1h": 3600.0}
DEFAULT_CACHE_TTL_S = 300.0


def _breakpoint_ttls(request: dict[str, Any]) -> list[float]:
    """Seconds-to-live of every cache_control breakpoint in the request."""
    out: list[float] = []

    def scan(obj: Any) -> None:
        if isinstance(obj, dict):
            cc = obj.get("cache_control")
            if isinstance(cc, dict) and cc.get("type") == "ephemeral":
                out.append(CACHE_TTLS.get(cc.get("ttl"), DEFAULT_CACHE_TTL_S))
            for value in obj.values():
                scan(value)
        elif isinstance(obj, list):
            for value in obj:
                scan(value)

    # Render order, which is also the order breakpoints appear in the prefix.
    for key in ("tools", "system", "messages"):
        scan(request.get(key))

    # Top-level cache_control is the auto-caching shorthand: one breakpoint on
    # the last cacheable block. It is a sibling of `messages`, so scanning the
    # three keys above would miss it.
    top = request.get("cache_control")
    if isinstance(top, dict) and top.get("type") == "ephemeral":
        out.append(CACHE_TTLS.get(top.get("ttl"), DEFAULT_CACHE_TTL_S))

    return out


def cloud_cache_ttl_s(request: dict[str, Any]) -> float | None:
    """How long this request's cloud-side prefix stays warm, in seconds.

    `None` when the request sets no breakpoints at all — nothing is cached, so
    there is no gap after which it goes cold.
    """
    ttls = _breakpoint_ttls(request)
    if not ttls:
        return None
    # Breakpoints nest: each caches the whole span from the start of the prompt
    # to its own position, as one entry that stands alone. Entries expire
    # independently, so the deepest one still alive is what gets reused and the
    # prefix is fully cold only once the longest-lived breakpoint expires.
    return max(ttls)


def extract_features(
    request: dict[str, Any], seconds_since_last_call: float | None = None
) -> CallFeatures:
    messages = request.get("messages") or []
    tools = request.get("tools") or []
    ttls = _breakpoint_ttls(request)

    system_chars = _text_len(request.get("system"))
    tool_chars = len(str(tools)) if tools else 0

    is_continuation = False
    if messages and isinstance(messages[-1], dict):
        last = messages[-1].get("content")
        if isinstance(last, list):
            is_continuation = any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in last
            )

    # Client tools carry {name, description, input_schema}; server tools carry a
    # versioned `type` instead, e.g. {"type": "web_search_20250305", ...}.
    server_tools = any(
        isinstance(t, dict) and ("type" in t or "input_schema" not in t) for t in tools
    )

    return CallFeatures(
        model=str(request.get("model", "")),
        has_tools=bool(tools),
        n_tools=len(tools),
        has_server_tools=server_tools,
        n_messages=len(messages),
        est_system_tokens=(system_chars + tool_chars) // CHARS_PER_TOKEN,
        max_tokens=int(request.get("max_tokens") or 0),
        stream=bool(request.get("stream")),
        is_tool_continuation=is_continuation,
        is_security_monitor=_text_contains(
            request.get("system"), SECURITY_MONITOR_MARKER
        ),
        seconds_since_last_call=seconds_since_last_call,
        cloud_cache_ttl_s=max(ttls) if ttls else None,
        n_cache_breakpoints=len(ttls),
    )


class Policy(Protocol):
    """Swappable placement rule. Implementations must not do I/O."""

    name: str

    def decide(self, f: CallFeatures) -> Decision: ...


class CloudOnly:
    """Baseline. Also the safe default before a policy is chosen."""

    name = "cloud-only"

    def decide(self, f: CallFeatures) -> Decision:
        return Decision("cloud", "policy")


class LocalOnly:
    """Baseline. Expect tool-calling turns to fail."""

    name = "local-only"

    def decide(self, f: CallFeatures) -> Decision:
        return Decision("local", "policy")


class StaticPolicy:
    name = "static"

    def __init__(
        self,
        max_local_tokens: int = 60_000,
        margin: float = 0.9,
        local_can_tool_call: bool = True,
        clamp_max_tokens: int | None = 4096,
    ) -> None:
        self.max_local_tokens = max_local_tokens
        self.margin = margin
        self.local_can_tool_call = local_can_tool_call
        self.clamp_max_tokens = clamp_max_tokens

    def budget(self) -> int:
        return int(self.max_local_tokens * self.margin)

    def effective_max_tokens(self, f: CallFeatures) -> int:
        """What max_tokens should become if this call is served locally.

        The harness asks for 64000 on nearly every call — a reservation, not a
        prediction; observed local outputs median 125 tokens. vLLM checks
        prompt + max_tokens against max_model_len up front, so that reservation
        alone excludes calls whose prompts fit comfortably (47 of 159 in the
        recorded corpus). Clamping is what an edge proxy is for: fitting a
        cloud-shaped request to local capacity.
        """
        if self.clamp_max_tokens is None:
            return f.max_tokens
        if f.local_prompt_tokens is None:
            return max(1, min(f.max_tokens, self.clamp_max_tokens))
        prompt_tokens = f.local_prompt_tokens
        headroom = max(0, self.budget() - prompt_tokens)
        return max(1, min(f.max_tokens, self.clamp_max_tokens, headroom))

    def decide(self, f: CallFeatures) -> Decision:
        # This is a Claude Code control-plane sidecall, not an ordinary agent
        # turn. The client grants it 64 output tokens; cloud completes this
        # call class while local Qwen3.8 often emits reasoning until vLLM cuts
        # it off at that limit. Keep all other feasible traffic local-first.
        if f.is_security_monitor:
            return Decision("cloud", "security-monitor-cloud")

        if f.has_server_tools:
            return Decision("cloud", "server-side-tool")

        # When live probing was requested, failure leaves both hard capacity
        # and cache state unknown. Do not repeat the old unsafe behaviour by
        # silently substituting the character estimate.
        if f.local_cache_prediction_confidence == "unavailable":
            return Decision("cloud", "local-probe-unavailable")

        if f.local_prompt_tokens is None:
            return Decision("cloud", "local-token-count-unavailable")

        prompt_tokens = f.local_prompt_tokens
        need = prompt_tokens + self.effective_max_tokens(f)
        if need > self.budget():
            return Decision("cloud", "too-large", f"{need} > {self.budget()} tokens")

        if f.has_tools and not self.local_can_tool_call:
            return Decision("cloud", "tools-unsupported")

        return Decision("local", "fits")


POLICIES: dict[str, type] = {
    "cloud-only": CloudOnly,
    "local-only": LocalOnly,
    "static": StaticPolicy,
}


def build(
    name: str,
    *,
    max_local_tokens: int = 60_000,
    margin: float = 0.9,
) -> Policy:
    try:
        policy_type = POLICIES[name]
    except KeyError:
        raise SystemExit(f"unknown policy {name!r} — one of: {', '.join(POLICIES)}")
    if policy_type is StaticPolicy:
        return StaticPolicy(max_local_tokens=max_local_tokens, margin=margin)
    return policy_type()
