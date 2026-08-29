"""Assemble every parent Claude Code assistant turn from stream-JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, TextIO

REPORT_START = "<!-- FANOUT_REPORT_START -->"
REPORT_COMPLETE = "<!-- FANOUT_REPORT_COMPLETE -->"


def _parent_message(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict) or event.get("type") != "assistant":
        return None
    message = event.get("message")
    if not isinstance(message, dict):
        return None
    parent_id = event.get("parent_tool_use_id", message.get("parent_tool_use_id"))
    return message if parent_id is None else None


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def assemble_parent_report(events: Iterable[Any]) -> str:
    """Concatenate complete parent turns, de-duplicated by session/message ID.

    Claude Code may emit the same message repeatedly when partial-message mode
    is enabled. Ordered replacement keeps the last (most complete) value for a
    turn without duplicating its prefix. Subagent messages are excluded by
    ``parent_tool_use_id``.
    """
    turns: OrderedDict[tuple[str, str], str] = OrderedDict()
    fallback_result = ""
    anonymous = 0
    for event in events:
        if isinstance(event, dict) and event.get("type") == "result":
            result = event.get("result")
            if isinstance(result, str):
                fallback_result = result
        message = _parent_message(event)
        if message is None:
            continue
        text = _message_text(message)
        if not text:
            continue
        session_id = str(event.get("session_id") or message.get("session_id") or "")
        message_id = message.get("id")
        if not message_id:
            anonymous += 1
            message_id = f"anonymous-{anonymous}"
        turns[(session_id, str(message_id))] = text

    assembled = "\n\n".join(text.strip() for text in turns.values() if text.strip())
    if not assembled:
        assembled = fallback_result.strip()
    start = assembled.find(REPORT_START)
    end = assembled.find(REPORT_COMPLETE, max(start, 0))
    if start < 0 or end < 0:
        return assembled.strip() + ("\n" if assembled.strip() else "")
    end += len(REPORT_COMPLETE)
    return assembled[start:end].strip() + "\n"


def read_events(lines: Iterable[str]) -> list[Any]:
    events: list[Any] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid stream JSON on line {line_number}: {exc}") from exc
    return events


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="assemble parent assistant turns from Claude Code stream JSON"
    )
    parser.add_argument("path", nargs="?", help="input JSONL; defaults to stdin")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, stdin: TextIO = sys.stdin) -> int:
    args = parse_args(argv)
    try:
        if args.path:
            with open(args.path, encoding="utf-8") as stream:
                report = assemble_parent_report(read_events(stream))
        else:
            report = assemble_parent_report(read_events(stdin))
    except (OSError, ValueError) as exc:
        print(f"report capture failed: {exc}", file=sys.stderr)
        return 1
    if not report:
        print("report capture failed: no parent assistant text found", file=sys.stderr)
        return 1
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
