"""Per-call placement: local (vLLM) or cloud.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol

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
    # Fraction of request-visible prior tool_result blocks explicitly marked
    # is_error. Unlike response-derived reliability, this is known before the
    # current call is dispatched.
    errored_tool_result_density: float = 0.0
    # One-based position in the current tool loop, derived only from request
    # history. None is reserved for old externally-built feature payloads that
    # predate the signal; the live extractor always supplies an integer.
    branch_turn_ordinal: int | None = None
    # Request-time availability of Claude Code's delegation tool. Defaulting
    # false keeps older recorded feature dictionaries replayable.
    has_agent_tool: bool = False
    is_security_monitor: bool = False
    seconds_since_last_call: float | None = None
    # Exact local rendering and live prefix residency from the patched vLLM
    # probe. The static policy uses exact input length when it is available;
    # cache-aware preference remains a later policy decision.
    local_prompt_tokens: int | None = None
    # Usable input-plus-output budget for the configured local setup after
    # applying its safety margin. This lets future learned scores compare the
    # 7B (54K) and 27B (90K) profiles by capacity pressure rather than treating
    # the same absolute prompt length as equivalent on both models.
    local_token_budget: int | None = None
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
    # Rung 1 (verify-then-escalate): computed by server.py from a
    # ReliabilityCircuitBreaker's rolling per-tool-suite-class local
    # schema-validity rate, *before* decide() runs — this keeps decide()
    # a pure function of features, matching every other policy here. See
    # edgeproxy/reliability.py.
    local_reliability_blocked: bool = False
    # Frozen offline logistic prediction of stop_reason=max_tokens. None means
    # the call is outside the model's training domain or lacks an exact local
    # prompt count; it must not be interpreted as zero risk.
    predicted_local_risk_score: float | None = None
    # Optional pre-dispatch state for the experimental adaptive policy. These
    # are deliberately absent from old traces and are never inferred from the
    # placement of the current call.
    previous_backend: Placement | None = None
    expected_local_service_ms: float | None = None
    expected_cloud_service_ms: float | None = None
    # Exact versioned input to the read-only shadow scorer. None means some
    # required pre-dispatch feature was unavailable; never impute it to zero.
    agentic_shadow_features: dict[str, float] | None = None


# Frozen 2026-09-05 training-only parameters from
# scripts/analyze_call_risk.py. The held-out instances were never used to fit
# these values or select the threshold.
LOCAL_RISK_INTERCEPT = -4.795043462543514
LOCAL_RISK_MEANS = (36775.372277227725, 23.242574257425744, 0.025218441145327872)
LOCAL_RISK_SCALES = (14420.982843011754, 28.033953411323353, 0.09298056847149268)
LOCAL_RISK_WEIGHTS = (1.297817761997748, 0.13570384947006683, 0.23094825371486538)
DEFAULT_LOCAL_RISK_THRESHOLD = 0.3516231494107525
# 2026-09-16: recalibrated 0.70 -> 0.55 after a live campaign found the
# scorer's max real-traffic score was 0.641 (36 real calls,
# agentic-router-live-campaign-20260916) -- 0.70 was unreachable
# regardless of call quality, not merely conservative, because the
# turn_safety factor (0.5x for branch_turn_ordinal<=3) caps most early-turn
# scores well below it. 0.55 sits just under the observed real-traffic
# median (0.574) so the routing condition actually exercises local
# placement instead of degenerating to always-cloud.
DEFAULT_AGENTIC_HEURISTIC_THRESHOLD = 0.55


def predict_local_risk_score(f: CallFeatures) -> float | None:
    """Predict local truncation risk from pre-dispatch request features only.

    The currently deployed coefficients are the frozen legacy model and do not
    yet consume ``local_token_budget``. The budget is nevertheless recorded so
    the replacement model can learn model-normalized context pressure.
    Returning None outside the fitted population is an explicit
    out-of-distribution result, not a low-risk prediction.
    """
    if not f.has_tools or f.is_security_monitor or f.local_prompt_tokens is None:
        return None
    values = (
        float(f.local_prompt_tokens),
        float(f.n_messages),
        float(f.errored_tool_result_density),
    )
    logit = LOCAL_RISK_INTERCEPT + sum(
        weight * (value - mean) / scale
        for weight, value, mean, scale in zip(
            LOCAL_RISK_WEIGHTS, values, LOCAL_RISK_MEANS, LOCAL_RISK_SCALES
        )
    )
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))


def default_agentic_scorer(f: CallFeatures) -> float:
    """Return a conservative, dependency-free estimate of local safety.

    This is deliberately a heuristic rather than the agentic experiment's
    fitted GBM. That model's labels describe eventual trajectory success, not
    the causal safety of routing this individual call locally, so its AUC does
    not justify shipping its probabilities as routing probabilities.

    Four bounded safety factors are multiplied: remaining local-context
    headroom, absence of request-visible tool errors, inverse predicted local
    truncation risk, and being past the first three exploration/planning
    turns. Multiplication prevents a strong signal from compensating for an
    unsafe one. Missing capacity makes the score zero; missing truncation risk
    and an unknown/early turn each apply a 0.5 uncertainty penalty. The result
    is always in [0, 1], where higher means safer to route local.
    """

    def bounded(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    if (
        f.local_prompt_tokens is None
        or f.local_token_budget is None
        or f.local_token_budget <= 0
    ):
        return 0.0

    headroom_safety = bounded(
        (f.local_token_budget - f.local_prompt_tokens) / f.local_token_budget
    )
    error_safety = 1.0 - bounded(float(f.errored_tool_result_density))
    if f.predicted_local_risk_score is None:
        risk_safety = 0.5
    else:
        risk_safety = 1.0 - bounded(float(f.predicted_local_risk_score))
    turn_safety = (
        1.0
        if f.branch_turn_ordinal is not None and f.branch_turn_ordinal > 3
        else 0.5
    )

    return bounded(headroom_safety * error_safety * risk_safety * turn_safety)


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


def _errored_tool_result_density(messages: list[Any]) -> float:
    """Share of prior tool results explicitly marked as errors."""
    total = errored = 0
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                total += 1
                errored += bool(block.get("is_error"))
    return errored / total if total else 0.0


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

    # Claude Code appends a trailing role="system" <system-reminder> message
    # after most turns -- including immediately after tool results -- so the
    # literal last message is essentially always role=system, never the
    # tool_result itself. Skip past any trailing system-role message(s) to
    # find the actual last conversational turn before checking it. Confirmed
    # live 2026-09-02: without this skip, is_tool_continuation was False on
    # 100% of a real multi-turn episode's calls, including calls that
    # immediately followed real tool_result blocks.
    is_continuation = False
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            is_continuation = any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            )
        break

    # Each completed tool-loop turn contributes one user message containing
    # one or more tool_result blocks. Count messages, not blocks: parallel tool
    # results are returned together and lead to one subsequent model turn.
    branch_turn_ordinal = 1 + sum(
        1
        for msg in messages
        if isinstance(msg, dict)
        and msg.get("role") == "user"
        and isinstance(msg.get("content"), list)
        and any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in msg["content"]
        )
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
        has_agent_tool=any(
            isinstance(tool, dict) and tool.get("name") == "Agent"
            for tool in tools
        ),
        has_server_tools=server_tools,
        n_messages=len(messages),
        est_system_tokens=(system_chars + tool_chars) // CHARS_PER_TOKEN,
        max_tokens=int(request.get("max_tokens") or 0),
        stream=bool(request.get("stream")),
        is_tool_continuation=is_continuation,
        errored_tool_result_density=_errored_tool_result_density(messages),
        branch_turn_ordinal=branch_turn_ordinal,
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


class StaticPolicy:
    name = "static"

    def __init__(
        self,
        max_local_tokens: int = 60_000,
        margin: float = 0.9,
        local_can_tool_call: bool = True,
        output_reserve_tokens: int = 0,
    ) -> None:
        self.max_local_tokens = max_local_tokens
        self.margin = margin
        self.local_can_tool_call = local_can_tool_call
        self.output_reserve_tokens = output_reserve_tokens

    def budget(self) -> int:
        return int(self.max_local_tokens * self.margin)

    def effective_max_tokens(self, f: CallFeatures) -> int:
        """What max_tokens should become if this call is served locally.

        Use the exact vLLM-rendered prompt length and consume all available
        headroom after the configured safety-margin budget and explicit output
        reserve. This removes the fixed 4,096-token truncation class while
        retaining a hard input-plus-output capacity bound.
        """
        if f.local_prompt_tokens is None:
            return 0
        headroom = max(
            0,
            self.budget() - f.local_prompt_tokens - self.output_reserve_tokens,
        )
        return min(f.max_tokens, headroom)

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
        effective_max_tokens = self.effective_max_tokens(f)
        if effective_max_tokens < 1:
            return Decision(
                "cloud",
                "too-large",
                f"budget={self.budget()} prompt={prompt_tokens} "
                f"reserve={self.output_reserve_tokens}",
            )
        need = prompt_tokens + effective_max_tokens + self.output_reserve_tokens
        if need > self.budget():
            return Decision("cloud", "too-large", f"{need} > {self.budget()} tokens")

        if f.has_tools and not self.local_can_tool_call:
            return Decision("cloud", "tools-unsupported")

        # Rung 1: this tool-suite class has been failing schema validation
        # locally often enough to trip the circuit breaker. Route this call
        # cloud instead, unless it was selected for the recovery probe
        # stream (server.py already resolved that coin flip into this bool).
        if f.local_reliability_blocked:
            return Decision("cloud", "reliability-circuit-open")

        return Decision("local", "fits")


class LocalOnly(StaticPolicy):
    """Baseline for the spec's three-way comparison: force every call local
    regardless of feasibility. Tool-calling turns are expected to degrade or
    fail — that is the point of the baseline, not a bug in it.

    It still subclasses ``StaticPolicy`` for one reason: ``max_tokens`` must be
    clamped to the local context window before the request reaches vLLM.
    Claude Code's non-tool sidecalls and main-loop turns request
    ``max_tokens`` up to 64,000 by default; on a local model whose window is
    smaller (the 7B setup is 60,000) vLLM rejects that outright with HTTP 400
    ``max_completion_tokens cannot be greater than max_model_len``, and every
    call in the session fails. ``StaticPolicy`` avoids this by routing
    probe-unavailable calls to cloud, but ``LocalOnly`` has no cloud escape,
    so it clamps here instead.
    """

    name = "local-only"

    def decide(self, f: CallFeatures) -> Decision:
        return Decision("local", "policy")

    def effective_max_tokens(self, f: CallFeatures) -> int:
        budget = self.budget()
        if f.local_prompt_tokens is not None:
            headroom = budget - f.local_prompt_tokens - self.output_reserve_tokens
        else:
            # Probe unavailable: cannot size headroom exactly, but must still
            # keep max_tokens strictly below the context window. budget already
            # carries the safety margin.
            headroom = budget - 1
        return max(1, min(f.max_tokens, headroom))


class WarmLocalPolicy(StaticPolicy):
    """``StaticPolicy`` plus one preference gate: send a branch's first turn
    to cloud, to skip local's one-time ~20-30s cold-start prefill of the
    system prompt (see
    claude-memory/wiki/findings/Local prefix cache hit rate holds at 59
    percent under Claude Code.md). Every other call — including warm
    continuations regardless of requested output size — stays local. The
    goal is maximizing local share subject to not eating that one
    avoidable cost, not hedging broadly toward cloud.

    Gates on ``is_tool_continuation``, not ``estimated_local_cached_fraction``
    — this is a fix, not the original design. The cache-fraction version
    shipped first and broke live: `estimated_local_cached_fraction` is only
    non-zero for content local has actually served, so escalating a call on
    a low fraction stops local from ever warming that branch, which makes
    every later call in the same branch also read cold and escalate —
    permanently, regardless of the threshold value. First live A/B run put
    100% of one job's calls on cloud. See
    claude-memory/wiki/problems/WarmLocalPolicy escalation is a
    self-reinforcing cloud lock.md. ``is_tool_continuation`` is immune to
    this: it's computed purely from whether the request's last message
    carries a `tool_result` block, so it depends only on conversation shape,
    never on which backend served anything before.

    A second predicate (escalate only when ALSO "reasoning-heavy" by
    ``max_tokens``) was designed, then dropped before being written: real
    traffic (n=184 calls, 2026-09-02 SWE-bench Pro campaigns) shows 97.8%
    of calls request ``max_tokens`` > 4,000 (median 44,417 of a 54,000
    budget) — Claude Code requests a large output ceiling on nearly every
    call regardless of what it actually needs, so ``max_tokens`` does not
    separate "will generate a lot" from "asked for headroom as usual."
    Using it would have made this policy functionally a plain cold-only
    gate while reading like a conjunction. See
    claude-memory/wiki/findings/max_tokens does not discriminate call
    weight in real traffic.md.

    Cloud's own prompt cache lives only 5 minutes (``DEFAULT_CACHE_TTL_S``)
    and this policy is local-first by design, so cloud-cache-state fields
    are deliberately not read here — routing on cloud warmth as a hard gate
    was already tried and falsified (see the "cache warmth is a weight,
    not a gate" note in claude-memory/wiki/topics/Routing Policy.md).

    # RESOLVED (Rung 4 step 5): a deliberate first-sibling warming
    # placement is by definition a fresh branch, which would otherwise be
    # escalated to cloud here, defeating the warming intent. The leader
    # barrier now handles this explicitly:
    # ``_leader_warming_exemption_allowed`` in server.py re-checks
    # ``decision.reason == "cold-branch-first-turn"`` and exempts only the
    # designated cohort leader from this exact gate, before any hard
    # feasibility/reliability/headroom gate. See
    # claude-memory/wiki/findings/Leader barrier dispatch is async and
    # timeout bounded.md.
    """

    name = "warm-local"

    def decide(self, f: CallFeatures) -> Decision:
        decision = super().decide(f)
        if decision.placement != "local":
            return decision  # a feasibility gate already said no; don't override

        # `estimated_local_cached_fraction` is logged for diagnostic context
        # only — it is NOT the decision input (see docstring for why).
        frac = f.estimated_local_cached_fraction

        if not f.is_tool_continuation:
            return Decision(
                "cloud",
                "cold-branch-first-turn",
                f"is_tool_continuation=False cached_fraction={frac}",
            )

        # Annotate the non-escalating path too, so trace analysis doesn't
        # need to cross-reference `features` separately to see why a call
        # passed the cold-branch check.
        return Decision(
            "local", "fits", f"is_tool_continuation=True cached_fraction={frac}"
        )


def branch_drift_escalation(
    f: CallFeatures, headroom_threshold: float, budget: int
) -> Decision | None:
    """Shared incremental check behind ``BranchDriftPolicy`` and
    ``CombinedPolicy``. Returns the escalation ``Decision`` if this call is
    deep enough into a branch to be low on local headroom, else ``None``
    (caller keeps whatever decision it already had)."""
    if f.local_prompt_tokens is not None:
        headroom_ratio = f.local_prompt_tokens / budget
        if headroom_ratio >= headroom_threshold:
            return Decision(
                "cloud",
                "branch-headroom-pressure",
                f"headroom_ratio={headroom_ratio:.3f} >= {headroom_threshold}",
            )
    return None


def predicted_risk_escalation(f: CallFeatures, risk_threshold: float) -> Decision | None:
    """Shared incremental check behind ``PredictedRiskPolicy`` and
    ``CombinedPolicy``. Returns the escalation ``Decision`` if the learned
    request-only risk score is at or above threshold, else ``None``."""
    score = (
        f.predicted_local_risk_score
        if f.predicted_local_risk_score is not None
        else predict_local_risk_score(f)
    )
    if score is not None and score >= risk_threshold:
        return Decision(
            "cloud",
            "predicted-local-risk",
            f"risk_score={score:.3f} >= {risk_threshold:.3f}",
        )
    return None


def planning_turn_escalation(f: CallFeatures, planning_turns: int) -> Decision | None:
    """Shared incremental check behind ``PlanningEscalationPolicy`` and
    ``CombinedPolicy``. Returns the escalation ``Decision`` if this call is
    at or before the observed early exploration/planning phase, else
    ``None``."""
    ordinal = f.branch_turn_ordinal
    if ordinal is not None and ordinal <= planning_turns:
        return Decision(
            "cloud",
            "early-planning-turn",
            f"branch_turn_ordinal={ordinal} <= {planning_turns}",
        )
    return None


class BranchDriftPolicy(WarmLocalPolicy):
    """Rung 3 (routing policy ladder): ``WarmLocalPolicy``'s cold-branch
    gate, plus a second, independent escalation for calls deep enough into
    a branch that local is running low on headroom.

    Motivated by an offline finding that ``WarmLocalPolicy``'s existing
    gate protects the wrong end: on 387 real SWE-bench Pro local calls (18
    truncations), calls it already keeps local (``is_tool_continuation``
    True) truncate at 6.7%, while the calls it escalates
    (``is_tool_continuation`` False) truncate at only 2.2% — the cold-start
    calls it targets were never the truncation risk; the risk concentrates
    in the later, "warm" continuations it always keeps local. See
    claude-memory/wiki/decisions/Tune Rung 3 branch drift policy against
    real trace data.md for the full sweep.

    ``headroom_threshold`` default 0.65 is not a round-number guess — it is
    the point-estimate boundary found by sweeping thresholds against that
    corpus: 0.65 is the LARGEST threshold that still catches 16/18 (89%) of
    real truncations, maximizing local share (49.6%) within that recall
    tier. **But a cluster-bootstrap follow-up (resampling by trace
    directory, since truncations cluster within branches rather than being
    387 independent draws) found 0.65 is NOT statistically distinguishable
    from 0.60 or 0.70** — only 4 of 27 trace directories contain any
    truncation at all, and the entire apparent 0.65-vs-0.70 advantage
    traces to two truncations inside a single cluster. Treat 0.65 as a
    defensible, preregistered candidate inside a wide, unresolved
    0.60–0.70 band, not a validated optimum. See claude-memory/wiki/
    decisions/Tune Rung 3 branch drift policy against real trace data.md
    for both the original sweep and the clustered correction.

    ``estimated_local_cached_fraction`` was tested as a second input and
    deliberately excluded: it is non-monotonic with truncation on the same
    corpus (fully-cached calls truncate at 36.8%, *higher* than uncached
    calls at 0%), because cache fraction and headroom both rise together as
    a branch grows deeper. Combining them would be redundant with headroom,
    not complementary, and risks reproducing the already-falsified "binary
    cache gate collapses local share" failure mode (see the "cache warmth
    is a weight, not a gate" note in claude-memory/wiki/topics/Routing
    Policy.md). A third proposed signal (predicted local-vs-cloud cost
    ratio) has no working predictor yet and is not implemented here.
    """

    name = "branch-drift"

    def __init__(self, *args: Any, headroom_threshold: float = 0.65, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.headroom_threshold = headroom_threshold

    def decide(self, f: CallFeatures) -> Decision:
        decision = super().decide(f)
        if decision.placement != "local":
            return decision  # a feasibility/reliability/cold-branch gate already said no

        escalated = branch_drift_escalation(f, self.headroom_threshold, self.budget())
        return escalated if escalated is not None else decision


class LearnedAgenticPolicy(StaticPolicy):
    """Apply an injected learned preference score after every hard gate.

    The scorer is deliberately external to the router so this policy does not
    assume a classifier implementation or artifact format. Scores at or above
    ``threshold`` prefer local placement. An absent or failing scorer routes
    cloud safely until a trained model is explicitly configured.
    """

    name = "learned-agentic"

    def __init__(
        self,
        *args: Any,
        scorer: Callable[[CallFeatures], float] | None = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scorer = scorer
        self.threshold = threshold

    def decide(self, f: CallFeatures) -> Decision:
        decision = super().decide(f)
        if decision.placement != "local":
            return decision  # a hard feasibility/reliability gate already said no

        if self.scorer is None:
            return Decision("cloud", "learned-scorer-unavailable")

        try:
            score = float(self.scorer(f))
        except Exception as exc:
            return Decision(
                "cloud",
                "learned-scorer-error",
                f"{type(exc).__name__}: {exc}",
            )

        if score >= self.threshold:
            return Decision(
                "local",
                "learned-score-local",
                f"score={score} threshold={self.threshold}",
            )
        return Decision(
            "cloud",
            "learned-score-cloud",
            f"score={score} threshold={self.threshold}",
        )


@dataclass(frozen=True)
class HarmEstimate:
    """A pre-dispatch estimate, not an observed outcome or task-success rate."""

    probability: float
    supported: bool
    model_version: str
    detail: str | None = None


class ArtifactHarmScorer:
    """Strict, dependency-free scorer for the versioned JSON linear artifact.

    The feature extractor must return exactly the artifact's *pre-dispatch*
    features. It is intentionally injected, since today's CallFeatures does
    not contain every trajectory-history feature. Missing fields do not become
    the training sentinel. The current 205-call artifact is shadow-only and
    cannot mark an estimate supported for active routing.
    """

    SCHEMA = "agentic-harm-shadow-v1"
    FEATURE_DERIVATION = "v4-live-request-extractor-parity-20260916"

    def __init__(
        self,
        artifact: Mapping[str, Any],
        feature_extractor: Callable[[CallFeatures], Mapping[str, float]],
    ) -> None:
        if artifact.get("schema_version") != self.SCHEMA:
            raise ValueError("unsupported harm artifact schema")
        if artifact.get("positive_class") != "HARM_LOCAL":
            raise ValueError("artifact class must be HARM_LOCAL")
        provenance = artifact.get("training_provenance")
        if not isinstance(provenance, dict) or provenance.get("feature_derivation_version") != self.FEATURE_DERIVATION:
            raise ValueError("harm artifact feature derivation version mismatch")
        if artifact.get("activation_allowed") is not False:
            raise ValueError("shadow artifact must explicitly forbid live activation")
        names = artifact.get("feature_names")
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise ValueError("invalid artifact feature names")
        scaling = artifact.get("standardization")
        if not isinstance(scaling, dict):
            raise ValueError("missing artifact standardization")
        try:
            means = tuple(float(v) for v in scaling["means"])
            stds = tuple(float(v) for v in scaling["stds"])
            coefficients = tuple(float(v) for v in artifact["coefficients"])
            intercept = float(artifact["intercept"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid artifact parameters") from exc
        if not (len(names) == len(means) == len(stds) == len(coefficients)):
            raise ValueError("artifact feature/parameter length mismatch")
        if not all(math.isfinite(v) for v in means + stds + coefficients + (intercept,)) or any(v <= 0 for v in stds):
            raise ValueError("artifact parameters must be finite; stds positive")
        self.names = tuple(names)
        self.means = means
        self.stds = stds
        self.coefficients = coefficients
        self.intercept = intercept
        self.status = str(artifact.get("status", ""))
        self.version = str(artifact.get("artifact_version") or artifact["schema_version"])
        self.feature_extractor = feature_extractor

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        feature_extractor: Callable[[CallFeatures], Mapping[str, float]],
    ) -> "ArtifactHarmScorer":
        with Path(path).open() as fh:
            artifact = json.load(fh)
        if not isinstance(artifact, dict):
            raise ValueError("artifact must be a JSON object")
        return cls(artifact, feature_extractor)

    def __call__(self, f: CallFeatures) -> HarmEstimate:
        values = self.feature_extractor(f)
        if set(values) != set(self.names):
            return HarmEstimate(1.0, False, self.version, "feature-name-mismatch")
        try:
            x = tuple(float(values[name]) for name in self.names)
        except (TypeError, ValueError, KeyError):
            return HarmEstimate(1.0, False, self.version, "feature-value-invalid")
        if not all(math.isfinite(v) for v in x):
            return HarmEstimate(1.0, False, self.version, "feature-value-invalid")
        logit = self.intercept + sum(
            coefficient * (value - mean) / std
            for coefficient, value, mean, std in zip(self.coefficients, x, self.means, self.stds)
        )
        probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        return HarmEstimate(
            probability,
            False,  # this schema is explicitly shadow-only, regardless of status text
            self.version,
            f"artifact_status={self.status} raw_uncalibrated_harm={probability:.6g}",
        )


class AdaptiveAgenticPolicy(StaticPolicy):
    """Conservative, opt-in quality gate with shadow placement by default.

    The injected scorer owns feature extraction and model loading. It must
    return an explicitly supported estimate; missing or invalid estimates do
    not silently become zero risk. No state is retained across calls: the
    caller supplies the *previous* backend when known, so independent task
    trajectories cannot contaminate one another.

    In shadow mode this policy preserves an explicitly injected baseline
    policy's placement (StaticPolicy if omitted) and records the proposed
    adaptive decision in ``detail``. Live use requires both a scorer and an
    explicit ``shadow=False`` construction; it is not registered as a build()
    choice while held-out evidence is inadequate.
    """

    name = "adaptive-agentic-shadow"

    def __init__(
        self,
        *args: Any,
        harm_scorer: Callable[[CallFeatures], HarmEstimate | None] | None = None,
        baseline_policy: Policy | None = None,
        cloud_threshold: float = 0.25,
        return_local_threshold: float = 0.10,
        shadow: bool = True,
        max_local_latency_ratio: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= return_local_threshold < cloud_threshold <= 1:
            raise ValueError("require 0 <= return_local_threshold < cloud_threshold <= 1")
        if max_local_latency_ratio is not None and (
            not math.isfinite(max_local_latency_ratio) or max_local_latency_ratio <= 0
        ):
            raise ValueError("max_local_latency_ratio must be finite and positive")
        self.harm_scorer = harm_scorer
        self.baseline_policy = baseline_policy
        self.cloud_threshold = cloud_threshold
        self.return_local_threshold = return_local_threshold
        self.shadow = shadow
        self.max_local_latency_ratio = max_local_latency_ratio

    def _proposal(self, f: CallFeatures) -> Decision:
        if self.harm_scorer is None:
            return Decision("cloud", "adaptive-model-unavailable")
        try:
            estimate = self.harm_scorer(f)
        except Exception as exc:
            return Decision("cloud", "adaptive-model-error", type(exc).__name__)
        if estimate is None or not isinstance(estimate, HarmEstimate) or not estimate.supported:
            version = estimate.model_version if isinstance(estimate, HarmEstimate) else "none"
            detail = estimate.detail if isinstance(estimate, HarmEstimate) else None
            return Decision("cloud", "adaptive-unsupported", f"model={version} {detail or ''}".strip())
        risk = estimate.probability
        if not isinstance(risk, (int, float)) or not math.isfinite(risk) or not 0 <= risk <= 1:
            return Decision("cloud", "adaptive-invalid-score", f"model={estimate.model_version}")

        audit = f"harm={risk:.6g} model={estimate.model_version}"
        if estimate.detail:
            audit += f" {estimate.detail}"
        # Asymmetric hysteresis: escalation is immediate at the upper bound;
        # returning from cloud needs the lower bound. Unknown prior placement
        # gets the same conservative lower bound.
        if risk >= self.cloud_threshold:
            return Decision("cloud", "adaptive-quality-escalation", audit)
        if risk > self.return_local_threshold and f.previous_backend != "local":
            return Decision("cloud", "adaptive-return-hysteresis", audit)

        # Optional serving gate uses measured/predicted times supplied by the
        # caller. A warm-cache hint alone never overrides quality. Missing
        # telemetry simply leaves this optional gate inactive.
        local_ms = f.expected_local_service_ms
        cloud_ms = f.expected_cloud_service_ms
        if (
            self.max_local_latency_ratio is not None
            and local_ms is not None
            and cloud_ms is not None
            and math.isfinite(local_ms)
            and math.isfinite(cloud_ms)
            and local_ms >= 0
            and cloud_ms > 0
            and local_ms > self.max_local_latency_ratio * cloud_ms
        ):
            return Decision("cloud", "adaptive-serving-gate", audit + f" local_ms={local_ms} cloud_ms={cloud_ms}")
        return Decision("local", "adaptive-quality-eligible", audit)

    def decide(self, f: CallFeatures) -> Decision:
        base = super().decide(f)
        if base.placement != "local":
            if self.shadow:
                baseline = self.baseline_policy.decide(f) if self.baseline_policy is not None else base
                return Decision(
                    baseline.placement,
                    "adaptive-shadow",
                    f"baseline={baseline.reason} proposed=cloud proposal_reason={base.reason}",
                )
            return base  # feasibility/reliability gates precede model and cache
        proposed = self._proposal(f)
        if self.shadow:
            baseline = self.baseline_policy.decide(f) if self.baseline_policy is not None else base
            return Decision(
                baseline.placement,
                "adaptive-shadow",
                f"baseline={baseline.reason} proposed={proposed.placement} "
                f"proposal_reason={proposed.reason} {proposed.detail or ''}".strip(),
            )
        return proposed


class HeuristicAgenticPolicy(LearnedAgenticPolicy):
    """Usable learned-agentic variant with the conservative heuristic wired in.

    The threshold remains constructor-tunable for offline sweeps, while the
    registered default is intentionally selective because the available
    agentic classifier labels are trajectory-level distant supervision.
    """

    name = "learned-agentic-heuristic"

    def __init__(
        self,
        *args: Any,
        threshold: float = DEFAULT_AGENTIC_HEURISTIC_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            scorer=default_agentic_scorer,
            threshold=threshold,
            **kwargs,
        )


class PredictedRiskPolicy(WarmLocalPolicy):
    """Escalate calls whose request-only predicted local risk is high.

    The score estimates the probability of a local ``max_tokens`` stop in the
    fitted trace population. It is separate from turn ordinal and preserves
    every feasibility, reliability, and cold-branch decision inherited from
    ``WarmLocalPolicy``.
    """

    name = "predicted-risk"

    def __init__(
        self,
        *args: Any,
        risk_threshold: float = DEFAULT_LOCAL_RISK_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be between 0 and 1")
        self.risk_threshold = risk_threshold

    def decide(self, f: CallFeatures) -> Decision:
        decision = super().decide(f)
        if decision.placement != "local":
            return decision

        escalated = predicted_risk_escalation(f, self.risk_threshold)
        return escalated if escalated is not None else decision


class PlanningEscalationPolicy(WarmLocalPolicy):
    """Quality-experiment variant that sends the first tool-loop turns cloud.

    ``WarmLocalPolicy`` remains the baseline and still owns its validated
    cold-branch escalation. This subclass adds cloud placement only when that
    baseline would otherwise choose local and ``branch_turn_ordinal`` is at
    most ``planning_turns``. Therefore every later turn, every hard gate, and
    every later non-tool continuation has exactly WarmLocalPolicy's decision
    and reason.

    The default of three is trace-grounded rather than guessed. Across 1,306
    tool-equipped calls in 58 real SWE-bench Pro traces, turns 2 and 3 devoted
    77.0% and 60.8% of tool calls to Read/Glob/Grep/LS and only 4.0% and 6.8%
    to Edit/Write. At turns 4-5, exploration fell to 15.2% and mutation rose
    to 22.4%. The experiment therefore treats turns 1-3 as the observed early
    exploration/planning phase and leaves the implementation phase unchanged.
    """

    name = "planning-escalation"

    def __init__(
        self, *args: Any, planning_turns: int = 3, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if planning_turns < 1:
            raise ValueError("planning_turns must be at least 1")
        self.planning_turns = planning_turns

    def decide(self, f: CallFeatures) -> Decision:
        decision = super().decide(f)
        if decision.placement != "local":
            return decision

        escalated = planning_turn_escalation(f, self.planning_turns)
        return escalated if escalated is not None else decision


class CombinedPolicy(WarmLocalPolicy):
    """Rung 4+ (routing policy ladder): composes every validated escalation
    gate on top of ``StaticPolicy``'s hard feasibility/reliability gates and
    ``WarmLocalPolicy``'s cold-branch gate — ``BranchDriftPolicy``'s headroom
    pressure, ``PlanningEscalationPolicy``'s early-turn escalation, and
    ``PredictedRiskPolicy``'s learned risk score. Any one of the three may
    independently escalate a call to cloud; the base gates always run first
    (via ``super().decide()``) and their decision is never overridden here.

    Precedence when more than one of the three would fire on the same call
    (checked in this order, first match wins):

    1. **Planning-turn ordinal** — cheapest to evaluate (a plain counter),
       purely structural, and backed by this project's strongest replicated
       evidence for any of the three (a 3-seed A/B, 12/12 pass vs.
       WarmLocalPolicy's 11/12).
    2. **Branch-drift headroom** — a direct local-capacity constraint check.
    3. **Predicted risk** — the one model-based, least selective signal of
       the three (a logistic regression whose own held-out validation found
       it dominated by prompt length rather than true difficulty),
       evaluated last as a catch-all over whatever the two rule-based gates
       already let through.

    This ordering is a deliberate, documented design choice, not an
    artifact of implementation order — see claude-memory/wiki/decisions/
    Compose Policy 6 as the all-improvements combined policy.md for the
    full rationale and the live audit this policy was built from.

    ``_leader_warming_exemption_allowed`` in server.py must additionally
    re-check all three of this policy's gates (not just branch-drift's, as
    it already did for ``BranchDriftPolicy``) before granting a cohort
    leader's cold-branch warming exemption — cohort coordination must never
    override a cloud/safety decision this policy already made.
    """

    name = "all-improvements"

    def __init__(
        self,
        *args: Any,
        headroom_threshold: float = 0.65,
        planning_turns: int = 3,
        risk_threshold: float = DEFAULT_LOCAL_RISK_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.headroom_threshold = headroom_threshold
        if planning_turns < 1:
            raise ValueError("planning_turns must be at least 1")
        self.planning_turns = planning_turns
        if not 0.0 <= risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be between 0 and 1")
        self.risk_threshold = risk_threshold

    def decide(self, f: CallFeatures) -> Decision:
        decision = super().decide(f)
        if decision.placement != "local":
            return decision  # a feasibility/reliability/cold-branch gate already said no

        for escalated in (
            planning_turn_escalation(f, self.planning_turns),
            branch_drift_escalation(f, self.headroom_threshold, self.budget()),
            predicted_risk_escalation(f, self.risk_threshold),
        ):
            if escalated is not None:
                return escalated

        return decision


POLICIES: dict[str, type] = {
    "cloud-only": CloudOnly,
    "local-only": LocalOnly,
    "static": StaticPolicy,
    "warm-local": WarmLocalPolicy,
    "branch-drift": BranchDriftPolicy,
    "learned-agentic": LearnedAgenticPolicy,
    "learned-agentic-heuristic": HeuristicAgenticPolicy,
    "predicted-risk": PredictedRiskPolicy,
    "planning-escalation": PlanningEscalationPolicy,
    "all-improvements": CombinedPolicy,
}


def build(
    name: str,
    *,
    max_local_tokens: int = 60_000,
    margin: float = 0.9,
    output_reserve_tokens: int = 0,
) -> Policy:
    try:
        policy_type = POLICIES[name]
    except KeyError:
        raise SystemExit(f"unknown policy {name!r} — one of: {', '.join(POLICIES)}")
    if issubclass(policy_type, StaticPolicy):
        return policy_type(
            max_local_tokens=max_local_tokens,
            margin=margin,
            output_reserve_tokens=output_reserve_tokens,
        )
    return policy_type()
