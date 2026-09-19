"""Blind LLM-judge pass over already-replayed paired calls.

Judge model is DeepSeek-V4-Flash (`CLOUD_DEEPSEEK`, the only cloud backend
this project uses -- see [[claude-models-banned-use-deepseek]]), the same
backend used everywhere else in this experiment. The judge prompt is
deliberately blind: candidate responses are labeled "Response A" /
"Response B" with per-example randomized order, and nothing in the prompt
names a backend, vendor, or model -- the judge sees only the task context
and the two candidate next-steps, exactly what a real quality judgment
should be based on.

This is NOT the real per-call causal ground truth (see
`experiments/agentic_router/README.md` and the project wiki finding on
downstream re-simulation cost) -- it's the cheap, immediate proxy: does an
independent model think these two candidate actions are equivalent given
the same context.

2026-09-16 additions:
- "Full-state" context: several recent conversational turns instead of
  just the literal last user message, still blind.
- Explicit UNCERTAIN verdict, distinct from PARSE_ERROR: the judge saying
  "I genuinely can't tell" is a real, useful signal, not a parsing failure.
- Resumable, incremental per-call output (same pattern as the replay
  batch scripts) -- a killed judge run loses at most one in-flight call.
- A reversed-order consistency runner: judges each example TWICE, once at
  its original randomized order and once with A/B exactly swapped, and
  reports whether the verdict (mapped back to arms) agrees -- this is the
  judge's own self-consistency rate, separate from and complementary to
  the never-yet-remeasured cloud self-consistency baseline noted in the
  project's audit report.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from experiments.capability_router import executor as cr_executor
from experiments.capability_router.config import CLOUD_DEEPSEEK, BackendConfig

Verdict = Literal["A_BETTER", "B_BETTER", "EQUIVALENT", "BOTH_INADEQUATE", "UNCERTAIN", "PARSE_ERROR"]
_VALID_VERDICTS = ("A_BETTER", "B_BETTER", "EQUIVALENT", "BOTH_INADEQUATE", "UNCERTAIN")

# DeepSeek-V4-Flash's real context window is NOT independently confirmed by
# this project's own infra: the gateway's /v1/models listing (checked
# 2026-09-16) returns no context-length metadata for any model, and
# DeepSeek is not even a distinct listed entry there (reached via request
# passthrough, not a registered pooled model). This constant is a
# deliberately CONSERVATIVE assumed floor (DeepSeek-V3-class models are
# publicly documented at 64K tokens minimum, larger in some variants),
# used only as a safety-valve trigger, never presented as a verified hard
# limit.
ASSUMED_CLOUD_CONTEXT_TOKENS_FLOOR = 64_000
CHARS_PER_TOKEN_ESTIMATE = 4  # same rough convention edgeproxy/router.py uses

# v3: full unbounded message history + complete tool schema, replacing v2's
# max_turns=16 window and v1's max_turns=3 window (both preserved as
# superseded pilot data, never silently discarded -- see
# stage1_judge_consistency_PILOT_*_superseded.jsonl). Quantified 2026-09-16
# against all 140 real replayed calls with FULL serialization (every real
# message + complete tools array, no cap): combined prompt p50=23,557 /
# p95=27,860 / max=30,210 estimated tokens -- 0/140 within even 90% of the
# assumed 64K floor above. The safety-valve truncation below exists for
# defensiveness only; no real call collected so far has triggered it.
TRUNCATION_POLICY_VERSION = "v4-system-messages-tools-2026-09-16"
TRUNCATION_TRIGGER_TOKENS = int(0.85 * ASSUMED_CLOUD_CONTEXT_TOKENS_FLOOR)
PRIMARY_ORDER_POLICY_VERSION = "v5-sha256-call-id-2026-09-17"


def primary_order_for_call(call_id: str, seed: int = 20260916) -> tuple[str, str]:
    """Stable 50/50 assignment independent of singleton/batch invocation order.

    This is hash randomization, not guaranteed exact balance within a small
    task. Every pair is still judged in both orientations. The call-ID salt
    and seed are recorded on result rows for audit/resume verification.
    """
    if not call_id:
        raise ValueError("call_id is required for deterministic primary order")
    digest = hashlib.sha256(f"{PRIMARY_ORDER_POLICY_VERSION}|{seed}|{call_id}".encode()).digest()
    return ("local", "cloud") if digest[0] & 1 else ("cloud", "local")


def render_response(response: dict[str, Any] | None) -> str:
    """Render one response's content blocks as plain text for the judge.
    Deliberately drops the `model` field and anything else that could
    identify which backend produced it."""
    if not response:
        return "(no response)"
    blocks = response.get("content") or []
    parts: list[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text":
            parts.append(b.get("text") or "")
        elif btype == "tool_use":
            parts.append(f"[calls tool `{b.get('name')}` with input: {json.dumps(b.get('input'), sort_keys=True)}]")
        # "thinking" blocks deliberately excluded -- reasoning style/length
        # could itself be an identifying signal, and the judge should score
        # the ACTION taken, not the internal reasoning trace.
    return "\n".join(p for p in parts if p) or "(empty response)"


def _render_message(msg: dict[str, Any]) -> str | None:
    role = msg.get("role")
    content = msg.get("content")
    if role not in ("user", "assistant"):
        return None
    texts: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                texts.append(block.get("text") or "")
            elif btype == "tool_use":
                texts.append(f"[calls tool `{block.get('name')}` with input: {json.dumps(block.get('input'), sort_keys=True)}]")
            elif btype == "tool_result":
                bc = block.get("content")
                if isinstance(bc, str):
                    texts.append(bc)
                elif isinstance(bc, list):
                    texts.extend(str(x.get("text", "")) for x in bc if isinstance(x, dict))
            # "thinking" excluded, same rationale as render_response.
    text = "\n".join(t for t in texts if t)
    if not text:
        return None
    return f"[{role}]\n{text}"


def _render_tools(request: dict[str, Any]) -> tuple[str, bool]:
    """Complete tool schema (name/description/input_schema for every tool
    offered on this request), serialized in full -- not a summary. Returns
    (rendered_text, tool_schema_complete). No model/backend identity is
    present in a tool schema (it's just the tool definitions Claude Code
    sent), so this is safe to include verbatim under the blind protocol.
    """
    tools = request.get("tools") or []
    if not tools:
        return "(no tools offered on this request)", True
    try:
        return json.dumps(tools, sort_keys=True, indent=None), True
    except (TypeError, ValueError):
        return "(tool schema present but not serializable)", False


def _render_system(request: dict[str, Any]) -> tuple[str, bool]:
    """Preserve the complete request-time system prompt, including blocks.

    The earlier judge prompt omitted this top-level Anthropic field, even
    though it can contain task and tool-use instructions.  JSON serialization
    retains text and non-text block metadata without inventing a summary.
    """
    system = request.get("system")
    if system is None or system == [] or system == "":
        return "(no system prompt on this request)", True
    try:
        return json.dumps(system, sort_keys=True, ensure_ascii=False), True
    except (TypeError, ValueError):
        return "(system prompt present but not serializable)", False


@dataclass(frozen=True)
class FullStateBundle:
    system_text: str
    context_text: str
    tools_text: str
    original_message_count: int  # total real (renderable) user/assistant turns found
    included_message_count: int  # how many of those made it into context_text
    omitted_message_count: int
    omitted_message_range: tuple[int, int] | None  # inclusive (start,end) positions within the renderable-turn sequence, or None
    task_anchor_present: bool  # is the first real turn included in context_text
    tool_schema_complete: bool
    system_complete: bool
    full_state_used: bool  # True unless the safety-valve truncation below fired
    truncation_policy_version: str


def build_full_state(request: dict[str, Any]) -> FullStateBundle:
    """ALL real conversational turns (both user and assistant sides), not a
    windowed subset -- v1 (max_turns=3) and v2 (max_turns=16) both silently
    dropped real history; both are superseded, never silently discarded
    (see stage1_judge_consistency_PILOT_*_superseded.jsonl). Also serializes
    the complete tool schema (see `_render_tools`), which no prior version
    of this function included at all despite candidate responses often
    being tool_use calls the judge cannot fully assess without it.

    A defensive safety-valve truncation exists for the (currently
    unobserved) case where a request's full history + tool schema would
    push the combined prompt past `TRUNCATION_TRIGGER_TOKENS` -- it trims
    from the MIDDLE of the renderable-turn sequence (task anchor first
    turn + most recent turns are always kept), and every field on
    `FullStateBundle` above is populated honestly to reflect exactly what
    happened, never silently.
    """
    messages = request.get("messages") or []

    renderable: list[tuple[int, str]] = []  # (original message index, rendered text)
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        piece = _render_message(msg)
        if piece is not None:
            renderable.append((i, piece))

    tools_text, tool_schema_complete = _render_tools(request)
    system_text, system_complete = _render_system(request)
    original_count = len(renderable)

    # First pass: include everything, unbounded.
    included = renderable
    omitted_count = 0
    omitted_range: tuple[int, int] | None = None
    full_state_used = True

    context_text = "\n\n".join(p for _, p in included) if included else "(no prior conversational context found)"
    prompt_chars_estimate = len(system_text) + len(context_text) + len(tools_text) + len(_JUDGE_INSTRUCTIONS) + 2000  # +2000 headroom for both response texts, sized generously
    estimated_tokens = prompt_chars_estimate / CHARS_PER_TOKEN_ESTIMATE

    if estimated_tokens > TRUNCATION_TRIGGER_TOKENS and len(renderable) > 2:
        # Keep the first real turn (task anchor) and as many of the most
        # recent turns as fit; drop a contiguous middle range. This has not
        # fired on any real call collected so far (see module docstring),
        # but must behave correctly and report honestly if it ever does.
        first = renderable[0]
        # Binary-search-free simple approach: keep growing the recent tail
        # until the budget is exceeded, then back off by one.
        recent_budget_chars = max(0, TRUNCATION_TRIGGER_TOKENS * CHARS_PER_TOKEN_ESTIMATE
                                   - len(system_text) - len(tools_text) - len(_JUDGE_INSTRUCTIONS) - 2000 - len(first[1]))
        kept_recent: list[tuple[int, str]] = []
        running = 0
        for item in reversed(renderable[1:]):
            if running + len(item[1]) > recent_budget_chars:
                break
            kept_recent.append(item)
            running += len(item[1])
        kept_recent.reverse()
        included = [first] + kept_recent
        omitted_count = original_count - len(included)
        if omitted_count > 0:
            kept_indices = {idx for idx, _ in included}
            omitted_positions = [pos for pos, (idx, _) in enumerate(renderable) if idx not in kept_indices]
            omitted_range = (min(omitted_positions), max(omitted_positions))
        full_state_used = False
        context_text = "\n\n".join(p for _, p in included)

    task_anchor_present = bool(included) and included[0][0] == renderable[0][0] if renderable else False

    return FullStateBundle(
        system_text=system_text,
        context_text=context_text,
        tools_text=tools_text,
        original_message_count=original_count,
        included_message_count=len(included),
        omitted_message_count=omitted_count,
        omitted_message_range=omitted_range,
        task_anchor_present=task_anchor_present,
        tool_schema_complete=tool_schema_complete,
        system_complete=system_complete,
        full_state_used=full_state_used,
        truncation_policy_version=TRUNCATION_POLICY_VERSION,
    )


# Retained for backward compatibility with any existing callers/tests.
def last_user_context(request: dict[str, Any], max_chars: int = 4000) -> str:
    """The most recent user-visible message content only. Superseded by
    `build_full_state` for new judge runs; kept as the single-turn
    building block it always was."""
    messages = request.get("messages") or []
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    texts.append(block.get("text") or "")
                elif block.get("type") == "tool_result":
                    bc = block.get("content")
                    if isinstance(bc, str):
                        texts.append(bc)
                    elif isinstance(bc, list):
                        texts.extend(str(x.get("text", "")) for x in bc if isinstance(x, dict))
            text = "\n".join(t for t in texts if t)
        else:
            continue
        if text:
            return text[:max_chars]
    return "(no prior user-visible context found)"


_JUDGE_INSTRUCTIONS = """You are comparing two candidate next-steps an AI coding assistant could take, given the same context. You are NOT told which system produced which candidate -- judge only the content.

Complete system prompt sent to the assistant on this request:
{system}

Complete tool schema available to the assistant on this request:
{tools}

Full conversation state so far (every real turn, oldest first):
{context}

Response A:
{a}

Response B:
{b}

Judge whether A and B are functionally equivalent as next steps for this task (same intent, same or compatible tool action, similar correctness) or whether one is clearly better. If the context is too thin or ambiguous to judge confidently either way, say so honestly rather than guessing. Reply with EXACTLY one line of JSON, no other text:
{{"verdict": "A_BETTER" | "B_BETTER" | "EQUIVALENT" | "BOTH_INADEQUATE" | "UNCERTAIN", "reason": "<one short sentence>"}}"""


@dataclass(frozen=True)
class JudgeResult:
    call_id: str
    order: tuple[str, str]  # ("local","cloud") or ("cloud","local") -- which arm was A, which was B
    verdict: Verdict
    reason: str
    raw_judge_text: str
    pass_label: str = "primary"  # stable hashed order or its exact A/B swap
    prompt_chars: int = 0
    estimated_prompt_tokens: float = 0.0
    # True only if estimated_prompt_tokens exceeds 90% of the ASSUMED floor
    # above -- a real judgment call would be needed at that point about
    # whether to actually truncate; no real prompt has hit this yet.
    near_assumed_limit: bool = False
    # Full provenance from FullStateBundle, carried onto every record so a
    # later analysis can split/filter by exactly what context each
    # judgment actually saw -- required, not optional, per-record.
    full_state_used: bool = True
    original_message_count: int = 0
    included_message_count: int = 0
    omitted_message_count: int = 0
    omitted_message_range: tuple[int, int] | None = None
    task_anchor_present: bool = False
    tool_schema_complete: bool = False
    system_complete: bool = False
    truncation_policy_version: str = TRUNCATION_POLICY_VERSION
    # Auditable normalization trail -- NEVER silent. verdict_normalization_applied
    # is True only when a known-typo alias fired (see _KNOWN_VERDICT_TYPOS);
    # original_parsed_token preserves exactly what the model actually wrote
    # before correction, so a later audit can always recover the raw token
    # without re-parsing raw_judge_text.
    verdict_normalization_applied: bool = False
    original_parsed_token: str | None = None
    primary_order_policy_version: str = PRIMARY_ORDER_POLICY_VERSION
    primary_order_seed: int = 20260916


# Known real judge-model typo, found 2026-09-16 auditing PARSE_ERROR rows
# (4/64 in the first real batch, all identical): "EQUIVALIENT" for
# "EQUIVALENT". Fixed by explicit, evidence-grounded alias -- NOT a
# fuzzy/edit-distance normalizer, which could silently misread a genuinely
# different verdict as something it isn't. Any other malformed string still
# correctly falls through to PARSE_ERROR. Every application of this alias
# is recorded on the resulting JudgeResult (verdict_normalization_applied +
# original_parsed_token), never applied silently.
_KNOWN_VERDICT_TYPOS = {"EQUIVALIENT": "EQUIVALENT"}


def _judge_one(bundle: FullStateBundle, a_text: str, b_text: str, backend: BackendConfig) -> tuple[Verdict, str, str, int, float, bool, str | None]:
    prompt = _JUDGE_INSTRUCTIONS.format(system=bundle.system_text, tools=bundle.tools_text, context=bundle.context_text, a=a_text, b=b_text)
    prompt_chars = len(prompt)
    estimated_tokens = prompt_chars / CHARS_PER_TOKEN_ESTIMATE
    payload = {
        "model": backend.request_model,
        # 2026-09-16: raised 220 -> 600 after a real PARSE_ERROR was root-caused
        # to genuine truncation, not malformed JSON -- the judge reasoned at
        # length about a hard comparison and never reached the required
        # {"verdict": ...} line within 220 tokens. Since this call uses
        # temperature=0, a bare retry at the same budget would deterministically
        # reproduce the identical truncation; the budget itself had to change.
        "max_tokens": 600,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    raw = cr_executor._stream_post(backend, payload)
    outcome = cr_executor._build_outcome(backend, raw)
    if outcome.status != "OK" or not outcome.response:
        return "PARSE_ERROR", "", f"replay status={outcome.status} detail={outcome.detail}", prompt_chars, estimated_tokens, False, None
    blocks = outcome.response.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
    m = re.search(r'\{.*"verdict"\s*:\s*"([A-Z_]+)".*\}', text, re.DOTALL)
    if not m:
        return "PARSE_ERROR", "", text, prompt_chars, estimated_tokens, False, None
    raw_token = m.group(1)
    verdict_str = _KNOWN_VERDICT_TYPOS.get(raw_token, raw_token)
    normalization_applied = verdict_str != raw_token
    original_token = raw_token if normalization_applied else None
    if verdict_str not in _VALID_VERDICTS:
        return "PARSE_ERROR", "", text, prompt_chars, estimated_tokens, False, None
    reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    reason = reason_m.group(1) if reason_m else ""
    return verdict_str, reason, text, prompt_chars, estimated_tokens, normalization_applied, original_token  # type: ignore[return-value]


def _make_result(call_id: str, order: tuple[str, str], pass_label: str,
                  verdict: Verdict, reason: str, raw_text: str,
                  prompt_chars: int, estimated_tokens: float,
                  bundle: FullStateBundle,
                  normalization_applied: bool = False,
                  original_token: str | None = None,
                  primary_order_seed: int = 20260916) -> "JudgeResult":
    return JudgeResult(
        call_id=call_id, order=order, verdict=verdict, reason=reason,
        raw_judge_text=raw_text, pass_label=pass_label,
        prompt_chars=prompt_chars, estimated_prompt_tokens=estimated_tokens,
        full_state_used=bundle.full_state_used,
        original_message_count=bundle.original_message_count,
        included_message_count=bundle.included_message_count,
        omitted_message_count=bundle.omitted_message_count,
        omitted_message_range=bundle.omitted_message_range,
        task_anchor_present=bundle.task_anchor_present,
        tool_schema_complete=bundle.tool_schema_complete,
        system_complete=bundle.system_complete,
        truncation_policy_version=bundle.truncation_policy_version,
        near_assumed_limit=estimated_tokens >= 0.9 * ASSUMED_CLOUD_CONTEXT_TOKENS_FLOOR,
        verdict_normalization_applied=normalization_applied,
        original_parsed_token=original_token,
        primary_order_seed=primary_order_seed,
    )


def _example_dict(ex: Any) -> dict:
    return ex if isinstance(ex, dict) else asdict(ex)


def _already_judged(out_path: Path, seed: int) -> set[tuple[str, str]]:
    """(call_id, pass_label) pairs already present in an existing output
    file -- lets a killed run resume without re-spending on completed
    judgments."""
    if not out_path.exists():
        return set()
    seen = set()
    with out_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("primary_order_policy_version") != PRIMARY_ORDER_POLICY_VERSION or row.get("primary_order_seed") != seed:
                raise ValueError("judge output uses a different order policy/seed; use a new output namespace")
            expected = primary_order_for_call(row["call_id"], seed)
            if row.get("pass_label", "primary") == "reversed":
                expected = (expected[1], expected[0])
            if tuple(row.get("order", ())) != expected:
                raise ValueError("existing judge row order differs from deterministic assignment")
            seen.add((row["call_id"], row.get("pass_label", "primary")))
    return seen


def judge_examples(
    examples: list[Any],
    out_path: Path,
    seed: int = 20260916,
    backend: BackendConfig = CLOUD_DEEPSEEK,
) -> list[JudgeResult]:
    """Judges every example once (primary pass), writing each result
    incrementally to `out_path` and skipping call_ids already present
    there -- resumable across a kill/restart. Always uses the full,
    unbounded conversation state + complete tool schema (`build_full_state`)
    -- there is no longer a windowed/partial-context option; both earlier
    windowed versions are superseded, not selectable."""
    cr_executor.load_env()
    already = _already_judged(out_path, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[JudgeResult] = []
    with out_path.open("a") as fh:
        for ex in examples:
            d = _example_dict(ex)
            call_id = d["call"]["call_id"]
            if (call_id, "primary") in already:
                continue
            bundle = build_full_state(d["call"]["request"])
            local_text = render_response((d["local_outcome"] or {}).get("response"))
            cloud_text = render_response((d["cloud_outcome"] or {}).get("response"))
            order = primary_order_for_call(call_id, seed)
            a_text = local_text if order[0] == "local" else cloud_text
            b_text = cloud_text if order[1] == "cloud" else local_text
            verdict, reason, raw_text, prompt_chars, est_tokens, norm_applied, orig_token = _judge_one(bundle, a_text, b_text, backend)
            r = _make_result(call_id, order, "primary", verdict, reason, raw_text, prompt_chars, est_tokens, bundle, norm_applied, orig_token, seed)
            fh.write(json.dumps(asdict(r)) + "\n")
            fh.flush()
            results.append(r)
    return results


def judge_examples_with_consistency(
    examples: list[Any],
    out_path: Path,
    seed: int = 20260916,
    backend: BackendConfig = CLOUD_DEEPSEEK,
) -> list[JudgeResult]:
    """Judges every example TWICE: once at its normal randomized order
    (pass_label="primary"), once with A and B exactly swapped
    (pass_label="reversed"). Both passes write incrementally to the same
    resumable `out_path`. Use `consistency_rate` on the result to measure
    how often the judge agrees with itself once the swap is undone. Always
    uses full, unbounded conversation state + complete tool schema.
    """
    cr_executor.load_env()
    already = _already_judged(out_path, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[JudgeResult] = []
    with out_path.open("a") as fh:
        for ex in examples:
            d = _example_dict(ex)
            call_id = d["call"]["call_id"]
            bundle = build_full_state(d["call"]["request"])
            local_text = render_response((d["local_outcome"] or {}).get("response"))
            cloud_text = render_response((d["cloud_outcome"] or {}).get("response"))
            primary_order = primary_order_for_call(call_id, seed)
            texts_by_arm = {"local": local_text, "cloud": cloud_text}

            if (call_id, "primary") not in already:
                a_text, b_text = texts_by_arm[primary_order[0]], texts_by_arm[primary_order[1]]
                verdict, reason, raw_text, prompt_chars, est_tokens, norm_applied, orig_token = _judge_one(bundle, a_text, b_text, backend)
                r = _make_result(call_id, primary_order, "primary", verdict, reason, raw_text, prompt_chars, est_tokens, bundle, norm_applied, orig_token, seed)
                fh.write(json.dumps(asdict(r)) + "\n")
                fh.flush()
                results.append(r)

            if (call_id, "reversed") not in already:
                reversed_order = (primary_order[1], primary_order[0])  # exact swap
                a_text, b_text = texts_by_arm[reversed_order[0]], texts_by_arm[reversed_order[1]]
                verdict, reason, raw_text, prompt_chars, est_tokens, norm_applied, orig_token = _judge_one(bundle, a_text, b_text, backend)
                r = _make_result(call_id, reversed_order, "reversed", verdict, reason, raw_text, prompt_chars, est_tokens, bundle, norm_applied, orig_token, seed)
                fh.write(json.dumps(asdict(r)) + "\n")
                fh.flush()
                results.append(r)
    return results


def _verdict_to_arm(verdict: str, order: tuple[str, str]) -> str | None:
    """Map a blind A/B verdict back to which arm (local/cloud) won, or
    None for EQUIVALENT/BOTH_INADEQUATE/UNCERTAIN/PARSE_ERROR (no winner)."""
    a_arm, b_arm = order
    if verdict == "A_BETTER":
        return a_arm
    if verdict == "B_BETTER":
        return b_arm
    return None


def consistency_rate(results: list[JudgeResult]) -> dict[str, Any]:
    """Pairs up primary/reversed judgments per call_id and reports how
    often the judge's WINNER (once the swap is undone) agrees between the
    two passes. A call where both passes say the same arm won -- or both
    say EQUIVALENT/BOTH_INADEQUATE/UNCERTAIN -- counts as consistent."""
    by_call: dict[str, dict[str, JudgeResult]] = {}
    for r in results:
        by_call.setdefault(r.call_id, {})[r.pass_label] = r

    n_pairs = agree = 0
    disagreements = []
    for call_id, passes in by_call.items():
        if "primary" not in passes or "reversed" not in passes:
            continue
        n_pairs += 1
        p, rev = passes["primary"], passes["reversed"]
        p_winner = _verdict_to_arm(p.verdict, p.order)
        rev_winner = _verdict_to_arm(rev.verdict, rev.order)
        # Both non-committal (EQUIVALENT/BOTH_INADEQUATE/UNCERTAIN/PARSE_ERROR
        # on both passes) counts as consistent -- the judge didn't flip a
        # decisive call just because of presentation order.
        if p_winner == rev_winner:
            agree += 1
        else:
            disagreements.append({
                "call_id": call_id,
                "primary_verdict": p.verdict, "primary_winner": p_winner,
                "reversed_verdict": rev.verdict, "reversed_winner": rev_winner,
            })
    return {
        "n_pairs": n_pairs,
        "agree": agree,
        "consistency_rate": agree / n_pairs if n_pairs else None,
        "disagreements": disagreements,
    }


def summarize(results: list[JudgeResult], pass_label: str = "primary") -> dict[str, Any]:
    """Translate blind A/B verdicts back into local/cloud terms now that the
    mapping is known (the judge itself never saw it). Filters to one pass
    (default "primary") so a consistency run's two passes aren't double
    counted in the headline distribution."""
    filtered = [r for r in results if r.pass_label == pass_label]
    counts = {"local_better": 0, "cloud_better": 0, "equivalent": 0,
              "both_inadequate": 0, "uncertain": 0, "parse_error": 0}
    for r in filtered:
        a_arm, b_arm = r.order
        if r.verdict == "A_BETTER":
            counts[f"{a_arm}_better"] += 1
        elif r.verdict == "B_BETTER":
            counts[f"{b_arm}_better"] += 1
        elif r.verdict == "EQUIVALENT":
            counts["equivalent"] += 1
        elif r.verdict == "BOTH_INADEQUATE":
            counts["both_inadequate"] += 1
        elif r.verdict == "UNCERTAIN":
            counts["uncertain"] += 1
        else:
            counts["parse_error"] += 1
    return {"n": len(filtered), "counts": counts}


def save_results(results: list[JudgeResult], out_path: Path) -> None:
    """Kept for any non-incremental callers; judge_examples[_with_consistency]
    already write incrementally and should be preferred."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")
