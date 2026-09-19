"""Pure request/response helpers for the offline, blinded preference judge.

This module deliberately has no HTTP client.  The caller owns transport and
must persist every request and response, including invalid responses.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.new_datasets.w1_storage.schemas import Verdict


class AggregateLabel(str, Enum):
    CLOUD_PREFERRED = "CLOUD_PREFERRED"
    EDGE_PREFERRED = "EDGE_PREFERRED"
    EQUIVALENT = "EQUIVALENT"
    BOTH_INADEQUATE = "BOTH_INADEQUATE"
    UNCERTAIN = "UNCERTAIN"


class ParseStatus(str, Enum):
    VALID = "valid"
    PENDING = "pending"
    INVALID = "invalid"


class JudgeParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedJudgment:
    verdict: Verdict | None
    reason_codes: tuple[str, ...]
    evidence: tuple[Mapping[str, str], ...]
    insufficient_context: bool
    status: ParseStatus
    error: str | None = None


@dataclass(frozen=True)
class OrientationResult:
    orientation: str
    parsed: ParsedJudgment
    candidate_a_backend: str
    candidate_b_backend: str


@dataclass(frozen=True)
class AggregatedPreference:
    label: AggregateLabel | None
    status: ParseStatus
    detail: str


_REASON_CODES = {
    "SUPPORTED_PROGRESS", "CONTRADICTS_OBSERVATION", "INVALID_TOOL_ARGUMENTS",
    "UNSUPPORTED_ASSUMPTION", "REPEATED_NO_PROGRESS", "VALID_ALTERNATIVES",
    "BOTH_DEFICIENT", "INSUFFICIENT_CONTEXT", "NO_CLEAR_ADVANTAGE",
}
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "JUDGE_SYSTEM_PROMPT.txt"


def _system_prompt() -> str:
    # Keep the request contract synchronized with the campaign-owned prompt.
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _visible_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only visible response text and executable actions."""
    return {
        "text": candidate.get("text", ""),
        "tool_actions": candidate.get("tool_actions", []),
    }


# A REAL leak, confirmed 2026-09-18 against actual campaign judge requests
# (not a detector false positive): Claude Code injects a literal
# "You are powered by the model <X>." banner into the session's captured
# history, where <X> is the exact --model value used to launch that
# session. For edge-only-v1 trajectories this is the deliberately generic
# alias "local" (uninformative), but for cloud-only-v1 trajectories it is
# the literal native backend id "deepseek-v4-flash" -- a real, specific
# identifier that tells the judge which of the shared history's two
# candidates continues a session already known to be cloud-backed. This
# must be redacted from the judge-facing view; the underlying raw Dataset
# A/C artifacts are never touched (redaction happens only here, at
# judge-request construction time).
_MODEL_BANNER_RE = re.compile(r"you are powered by the model [^.\n]*\.", re.IGNORECASE)


def redact_model_identity_from_history(history: Sequence[Mapping[str, Any]] | str) -> Sequence[Mapping[str, Any]] | str:
    if isinstance(history, str):
        return _MODEL_BANNER_RE.sub("You are powered by the model [redacted].", history)
    text = json.dumps(history, ensure_ascii=False)
    if not _MODEL_BANNER_RE.search(text):
        return history
    redacted = _MODEL_BANNER_RE.sub("You are powered by the model [redacted].", text)
    return json.loads(redacted)


def build_judge_request(
    issue: str,
    history: Sequence[Mapping[str, Any]] | str,
    candidate_a: Mapping[str, Any],
    candidate_b: Mapping[str, Any],
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the provider-neutral JSON body; candidate identity is absent."""
    payload = {
        "issue": issue,
        "history": redact_model_identity_from_history(history),
        "tool_schemas": list(tool_schemas or []),
        "candidate_A": _visible_candidate(candidate_a),
        "candidate_B": _visible_candidate(candidate_b),
    }
    return {
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 512,
    }


def parse_judge_response(response: str | Mapping[str, Any]) -> ParsedJudgment:
    """Strictly parse a completed judge object; placeholders and extra keys fail."""
    try:
        obj = json.loads(response) if isinstance(response, str) else dict(response)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise JudgeParseError("response is not JSON") from exc
    if not isinstance(obj, dict) or set(obj) != {"verdict", "reason_codes", "evidence", "insufficient_context"}:
        raise JudgeParseError("response keys do not match the strict contract")
    try:
        verdict = Verdict(obj["verdict"])
    except (KeyError, ValueError, TypeError) as exc:
        raise JudgeParseError("invalid verdict") from exc
    codes = obj["reason_codes"]
    if not isinstance(codes, list) or not all(isinstance(c, str) and c in _REASON_CODES for c in codes):
        raise JudgeParseError("invalid reason_codes")
    evidence = obj["evidence"]
    if not isinstance(evidence, list) or len(evidence) > 3:
        raise JudgeParseError("invalid evidence")
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"message_or_candidate_ref", "explanation"} or not all(isinstance(v, str) for v in item.values()):
            raise JudgeParseError("invalid evidence item")
    if not isinstance(obj["insufficient_context"], bool):
        raise JudgeParseError("insufficient_context must be boolean")
    if obj["insufficient_context"] and verdict is not Verdict.UNCERTAIN:
        raise JudgeParseError("insufficient context requires UNCERTAIN")
    return ParsedJudgment(verdict, tuple(codes), tuple(evidence), obj["insufficient_context"], ParseStatus.VALID)


def invalid_judgment(error: str, *, pending: bool = False) -> ParsedJudgment:
    return ParsedJudgment(None, (), (), False, ParseStatus.PENDING if pending else ParseStatus.INVALID, error)


def primary_mapping(pair_id: str, *, version: str = "w5-order-swap-v1") -> dict[str, str]:
    digest = hashlib.sha256(f"{version}\0{pair_id}".encode()).digest()
    first = "edge" if digest[0] % 2 == 0 else "cloud"
    return {"A": first, "B": "cloud" if first == "edge" else "edge"}


def orientation_mappings(pair_id: str, *, version: str = "w5-order-swap-v1") -> tuple[dict[str, str], dict[str, str]]:
    primary = primary_mapping(pair_id, version=version)
    return primary, {"A": primary["B"], "B": primary["A"]}


def verdict_backend(verdict: Verdict, mapping: Mapping[str, str]) -> str | None:
    if verdict is Verdict.A_BETTER:
        return mapping["A"]
    if verdict is Verdict.B_BETTER:
        return mapping["B"]
    return None


def aggregate_orientations(primary: OrientationResult, reverse: OrientationResult) -> AggregatedPreference:
    p, r = primary.parsed, reverse.parsed
    if p.status is not ParseStatus.VALID or r.status is not ParseStatus.VALID:
        return AggregatedPreference(None, p.status if p.status is not ParseStatus.VALID else r.status, "operationally incomplete: missing, truncated, or unparsable judgment")
    assert p.verdict is not None and r.verdict is not None
    if p.verdict is Verdict.EQUIVALENT and r.verdict is Verdict.EQUIVALENT:
        return AggregatedPreference(AggregateLabel.EQUIVALENT, ParseStatus.VALID, "both orientations equivalent")
    if p.verdict is Verdict.BOTH_INADEQUATE and r.verdict is Verdict.BOTH_INADEQUATE:
        return AggregatedPreference(AggregateLabel.BOTH_INADEQUATE, ParseStatus.VALID, "both orientations inadequate")
    pw, rw = verdict_backend(p.verdict, {"A": primary.candidate_a_backend, "B": primary.candidate_b_backend}), verdict_backend(r.verdict, {"A": reverse.candidate_a_backend, "B": reverse.candidate_b_backend})
    if pw == rw == "cloud":
        return AggregatedPreference(AggregateLabel.CLOUD_PREFERRED, ParseStatus.VALID, "cloud wins both orientations")
    if pw == rw == "edge":
        return AggregatedPreference(AggregateLabel.EDGE_PREFERRED, ParseStatus.VALID, "edge wins both orientations")
    return AggregatedPreference(AggregateLabel.UNCERTAIN, ParseStatus.VALID, f"disagreement: primary={p.verdict.value}, reverse={r.verdict.value}")


# Harness self-reference vocabulary: verified 2026-09-18 against real
# captured smoke-campaign judge requests (two different tasks, both edge-run
# and cloud-run sessions) to be Claude Code's OWN product/tool/skill/file
# vocabulary -- "Claude Code", "CLAUDE.md", ".claude/settings.json",
# "claude-api" (a skill name), "claude.ai" (a product surface), bare
# "sonnet"/"opus"/"haiku" (listed as trigger keywords inside a skill
# description), "anthropic.com" (the attribution email domain) -- appearing
# byte-for-byte identically regardless of which backend (Qwen/DeepSeek)
# generated the turn, and identically between candidate A and candidate B's
# shared history/tool_schemas in every one of 20 real pairs checked. Content
# that is provably backend-invariant carries zero signal a judge could use
# to infer which candidate came from which backend, so it is excluded from
# the *shared-context* check only. The candidate responses themselves are
# still checked against the strict, unrestricted pattern below -- if either
# candidate's own text or tool call actually claims a model identity, that
# is a real, unshared signal and must still be caught.
_HARNESS_SELF_REFERENCE_RE = re.compile(
    r"claude\s*code|claude\.md|\.claude[/\\]|claude-api|claude\.ai|claude/agents|"
    r"claude\s+(?:session|desktop|reads?|stops?|exits?)|anthropic\.com|"
    r"\bclaude:|`claude`|\bclaude,\s*anthropic\b",
    re.IGNORECASE,
)
_STRICT_IDENTITY_RE = re.compile(
    r"\b(?:claude|sonnet|opus|haiku|fable|deepseek|v4[- ]?flash|qwen|anthropic|openai|vllm)[-_a-z0-9]*\b",
    re.IGNORECASE,
)
# Loosened for shared context only: drops bare claude/sonnet/opus/haiku/
# anthropic/openai, which in every real instance checked were harness
# self-reference (including a skill-trigger description listing "OpenAI/
# GPT/Gemini/Llama/Mistral/Cohere/Ollama" as an unrelated-provider
# exclusion list), and keeps only the identifiers that would actually
# reveal *this campaign's* backend identity (Qwen vs. DeepSeek -- OpenAI
# is not even one of the two backends in play) if they leaked.
_SHARED_CONTEXT_IDENTITY_RE = re.compile(
    r"\b(?:deepseek|v4[- ]?flash|qwen|inferact|nvfp4)[-_a-z0-9]*\b",
    re.IGNORECASE,
)


def _mask_harness_self_reference(text: str) -> str:
    return _HARNESS_SELF_REFERENCE_RE.sub(" ", text)


def anonymization_violations(request: Mapping[str, Any]) -> list[str]:
    """Return leakage findings; an empty list is the only clean result."""
    # The system rubric necessarily names the forbidden categories. Inspect
    # only quoted evaluation data, never the rubric that describes the check.
    messages = request.get("messages", [])
    user_texts = [m.get("content", "") for m in messages if isinstance(m, Mapping) and m.get("role") != "system"]

    # Split shared context (history/tool_schemas/issue) from the two
    # candidates: they get different identity-pattern strictness (see
    # docstring above). The payload is the JSON-encoded string inside the
    # single user message content, per build_judge_request's shape.
    candidate_text = ""
    shared_text_parts: list[str] = []
    for content in user_texts:
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (TypeError, ValueError, json.JSONDecodeError):
            shared_text_parts.append(json.dumps(content, ensure_ascii=False))
            continue
        if isinstance(payload, Mapping) and "candidate_A" in payload and "candidate_B" in payload:
            candidate_text += json.dumps(payload.get("candidate_A"), ensure_ascii=False)
            candidate_text += json.dumps(payload.get("candidate_B"), ensure_ascii=False)
            shared = {k: v for k, v in payload.items() if k not in ("candidate_A", "candidate_B")}
            shared_text_parts.append(json.dumps(shared, ensure_ascii=False))
        else:
            shared_text_parts.append(json.dumps(payload, ensure_ascii=False))
    shared_text = _mask_harness_self_reference(" ".join(shared_text_parts))

    violations: list[str] = []
    if _STRICT_IDENTITY_RE.search(candidate_text) or _SHARED_CONTEXT_IDENTITY_RE.search(shared_text):
        violations.append("provider_or_model_identity")

    patterns = {
        # Narrowed to actual serving-telemetry field/metric shapes. Bare
        # "cost"/"price" were dropped: verified false positives (generic
        # English "that costs nothing", "how much it would roughly cost" in
        # unrelated tool-schema descriptions) with zero true positives found
        # in this campaign's real judge requests.
        "latency_or_cost": r"(?:\blatency\b|\bttft\b|tokens?[/_-]sec\b|cost[_-]usd|cost_basis|\$[0-9])",
        # Narrowed away from the bare substrings "route"/"routing", which
        # matched AWS Route53 (task source-code git log) and generic English
        # ("Route blocked work back to your user") with zero true positives.
        # Kept the specific field/policy names this campaign actually uses.
        "routing_decision": r"(?:selected_backend|destination_backend|routing[_ ]decision|placement_decision|\bedge-only\b|\bcloud-only\b)",
        "gold_or_final_grade": r"(?:gold[_ -]?(?:code|patch)|final[_ -]?grade|official[_ -]?grade)",
        "future_tool_output": r"(?:future[_ -]?tool[_ -]?outputs?|post[_ -]?dispatch[_ -]?tool|upcoming[_ -]?tool)",
    }
    text = candidate_text + " " + shared_text
    for label, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(label)
    return violations
