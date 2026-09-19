"""Request-time signal for the benchmark-only cohort parent override."""

from __future__ import annotations

import re
from typing import Any, Mapping


SIGNAL_NAME = "root-token-budget-reminder-v1"

_TOKEN_BUDGET_REMINDER = re.compile(
    r"<system-reminder>\s*"
    r"<total_tokens>\d+\s+tokens left</total_tokens>\s*"
    r"</system-reminder>"
)


def _last_text_block(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or not content:
        return None
    block = content[-1]
    if not isinstance(block, Mapping) or block.get("type") != "text":
        return None
    text = block.get("text")
    return text if isinstance(text, str) else None


def is_cohort_parent_candidate(
    request: Any,
    headers: Mapping[str, str],
) -> bool:
    """Identify a likely root call about to emit ``Agent`` delegations.

    This intentionally recognizes one measured Claude Code/SWE-bench request
    shape. It is not a general fan-out predictor and remains protected by the
    separate ``cohort_parent_placement`` experiment flag.
    """
    if not isinstance(request, Mapping):
        return False
    if headers.get("x-claude-code-agent-id"):
        return False
    if not any(
        isinstance(tool, Mapping) and tool.get("name") == "Agent"
        for tool in (request.get("tools") or [])
    ):
        return False

    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    last = messages[-1]
    if not isinstance(last, Mapping) or last.get("role") != "system":
        return False
    text = _last_text_block(last.get("content"))
    return bool(text and _TOKEN_BUDGET_REMINDER.fullmatch(text.strip()))
