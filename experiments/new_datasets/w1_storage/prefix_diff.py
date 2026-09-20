"""Token-level prefix-divergence instrumentation between consecutive
Claude Code LLM calls in the same trajectory (real tokenizer -- not the
chars/4 heuristic behind features.py's input_token_estimate).

For call i vs call i-1 in the same session, computes:
  - prompt_tokens: real token count of the current request (section-wise
    tokenization -- an approximation of the model's actual chat-template
    tokenization, not a byte-exact replay of it; good enough for finding
    a divergence point, not a claim about the exact wire-level count).
  - common_prefix_tokens / prefix_fraction: how many leading tokens are
    identical between the two consecutive requests.
  - first_difference: {section, tool} -- which logical part of the
    request (system / tool_schema:<name> / message[i]:role:block_type)
    the first differing token falls in.

Does NOT report actual_cached_tokens/actual_cache_rate as real numbers:
confirmed by inspecting a captured response body that the edgeproxy's
Anthropic-format translation drops vLLM's own
prompt_tokens_details.cached_tokens field (usage only has
input_tokens/output_tokens), so no real per-call cache ground truth is
currently captured. Left explicitly None with a missing_reason rather
than approximated from the coarser server-wide cumulative
prefix_cache_hits_total/queries_total counters, which are not scoped to
a single call.
"""
from __future__ import annotations

import json
from typing import Any

TOKENIZER_MODEL_ID = "Inferact/Qwen3.8-27B-NVFP4"  # the actual served local model; public on HF Hub
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)
    return _tokenizer


def _flatten_sections(request: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """Ordered (label, text, tool_name) sections matching the request's
    own field order: system, tools, then messages/content blocks."""
    sections: list[tuple[str, str, str | None]] = []
    system = request.get("system")
    if system:
        text = system if isinstance(system, str) else json.dumps(system, sort_keys=True)
        sections.append(("system", text, None))
    for tool in request.get("tools") or []:
        name = tool.get("name") if isinstance(tool, dict) else None
        sections.append((f"tool_schema:{name}", json.dumps(tool, sort_keys=True), name))
    for i, msg in enumerate(request.get("messages") or []):
        role = msg.get("role", "unknown")
        content = msg.get("content")
        if isinstance(content, str):
            sections.append((f"message[{i}]:{role}:text", content, None))
            continue
        for j, block in enumerate(content or []):
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "unknown")
            tool_name = block.get("name") if btype == "tool_use" else None
            sections.append((f"message[{i}]:{role}:{btype}[{j}]", json.dumps(block, sort_keys=True), tool_name))
    return sections


def _tokenize_sections(sections: list[tuple[str, str, str | None]]) -> tuple[list[int], list[tuple[int, int, str, str | None]]]:
    tok = _get_tokenizer()
    all_tokens: list[int] = []
    bounds: list[tuple[int, int, str, str | None]] = []
    for label, text, tool_name in sections:
        ids = tok.encode(text, add_special_tokens=False)
        start = len(all_tokens)
        all_tokens.extend(ids)
        bounds.append((start, len(all_tokens), label, tool_name))
    return all_tokens, bounds


def _locate_section(index: int, bounds: list[tuple[int, int, str, str | None]]) -> tuple[str | None, str | None]:
    for start, end, label, tool_name in bounds:
        if start <= index < end:
            return label, tool_name
    return None, None


def session_cache_progress(baseline: dict[str, Any] | None, current: dict[str, Any] | None) -> dict[str, Any]:
    """Real per-task (per-trajectory) prefix-cache hit rate accumulated so
    far in THIS session, from vLLM's own cumulative
    prefix_cache_hits_total/queries_total counters (local_resources.vllm
    in a captured serving_calls-style record).

    These counters are server-wide, not per-call, so a single snapshot's
    prefix_cache_hit_fraction_lifetime mixes in every other session that
    ever hit the server. Taking the delta between the session's own first
    call and its current call isolates (assuming no other concurrent
    traffic touched the server during that window -- true for this
    sequential single-task-at-a-time capture driver) the hit rate for
    just this session's own calls, which is what a routing decision
    actually cares about. The raw lifetime figure is also passed through
    unchanged (`lifetime_hit_rate_at_dispatch`) so the global number stays
    available alongside the per-task one.
    """
    if not current:
        return {"session_hits_delta": None, "session_queries_delta": None,
                "session_hit_rate_so_far": None, "lifetime_hit_rate_at_dispatch": None}
    lifetime = current.get("prefix_cache_hit_fraction_lifetime")
    if not baseline:
        return {"session_hits_delta": 0, "session_queries_delta": 0,
                "session_hit_rate_so_far": None, "lifetime_hit_rate_at_dispatch": lifetime}
    bh, bq = baseline.get("prefix_cache_hits_total"), baseline.get("prefix_cache_queries_total")
    ch, cq = current.get("prefix_cache_hits_total"), current.get("prefix_cache_queries_total")
    if None in (bh, bq, ch, cq):
        return {"session_hits_delta": None, "session_queries_delta": None,
                "session_hit_rate_so_far": None, "lifetime_hit_rate_at_dispatch": lifetime}
    dq, dh = cq - bq, ch - bh
    return {
        "session_hits_delta": dh,
        "session_queries_delta": dq,
        "session_hit_rate_so_far": (dh / dq) if dq > 0 else None,
        "lifetime_hit_rate_at_dispatch": lifetime,
    }


def compute_prefix_diff(prev_request: dict[str, Any] | None, curr_request: dict[str, Any]) -> dict[str, Any]:
    """Real-tokenizer prefix-divergence features for one call relative to
    the immediately preceding call in the same trajectory. None fields
    when there's no prior call to diff against."""
    curr_sections = _flatten_sections(curr_request)
    curr_tokens, curr_bounds = _tokenize_sections(curr_sections)
    prompt_tokens = len(curr_tokens)

    if prev_request is None:
        return {
            "prompt_tokens": prompt_tokens,
            "common_prefix_tokens": None,
            "prefix_fraction": None,
            "cacheable_prefix_tokens": None,
            "uncached_prefill_tokens": prompt_tokens,
            "first_difference": None,
            "actual_cached_tokens": None,
            "actual_cache_rate": None,
            "actual_cache_rate_missing_reason": "no_prior_call_in_trajectory",
        }

    prev_sections = _flatten_sections(prev_request)
    prev_tokens, _ = _tokenize_sections(prev_sections)

    common = 0
    limit = min(len(prev_tokens), len(curr_tokens))
    while common < limit and prev_tokens[common] == curr_tokens[common]:
        common += 1

    section_label, tool_name = (None, None)
    if common < prompt_tokens:
        section_label, tool_name = _locate_section(common, curr_bounds)

    return {
        "prompt_tokens": prompt_tokens,
        "common_prefix_tokens": common,
        "prefix_fraction": (common / prompt_tokens) if prompt_tokens else None,
        "cacheable_prefix_tokens": common,
        "uncached_prefill_tokens": prompt_tokens - common,
        "first_difference": {"section": section_label, "tool": tool_name} if section_label else None,
        "actual_cached_tokens": None,
        "actual_cache_rate": None,
        "actual_cache_rate_missing_reason": (
            "edgeproxy does not forward vLLM's prompt_tokens_details.cached_tokens "
            "into the captured Anthropic-format response usage block"
        ),
    }
