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


@dataclass(frozen=True)
class CallFeatures:
    """What the router knows at decision time."""

    model: str
    has_tools: bool
    n_tools: int
    has_server_tools: bool
    n_messages: int
    est_prompt_tokens: int
    est_system_tokens: int
    max_tokens: int
    stream: bool
    is_tool_continuation: bool


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


def extract_features(request: dict[str, Any]) -> CallFeatures:
    messages = request.get("messages") or []
    tools = request.get("tools") or []

    system_chars = _text_len(request.get("system"))
    body_chars = sum(_text_len(m.get("content")) for m in messages if isinstance(m, dict))
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
        est_prompt_tokens=(system_chars + body_chars + tool_chars) // CHARS_PER_TOKEN,
        est_system_tokens=(system_chars + tool_chars) // CHARS_PER_TOKEN,
        max_tokens=int(request.get("max_tokens") or 0),
        stream=bool(request.get("stream")),
        is_tool_continuation=is_continuation,
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
        max_local_tokens: int = 32_000,
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
        headroom = max(0, self.budget() - f.est_prompt_tokens)
        return max(1, min(f.max_tokens, self.clamp_max_tokens, headroom))

    def decide(self, f: CallFeatures) -> Decision:
        if f.has_server_tools:
            return Decision("cloud", "server-side-tool")

        need = f.est_prompt_tokens + self.effective_max_tokens(f)
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


def build(name: str) -> Policy:
    try:
        return POLICIES[name]()
    except KeyError:
        raise SystemExit(f"unknown policy {name!r} — one of: {', '.join(POLICIES)}")
