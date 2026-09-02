#!/usr/bin/env python3
"""Checker for fanout-architecture-map: report shape plus predeclared facts about the codebase."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402

REQUIRED_HEADINGS = [
    "## Executive Summary",
    "## Findings from Each Agent",
    "## Architecture Notes",
    "## Open Questions",
]


def check(sandbox: Path, fixture: Path, report_text: str) -> dict:
    if not report_text.strip():
        return cc.result(False, 0.0, "no report captured")

    missing_headings = cc.has_required_headings(report_text, REQUIRED_HEADINGS)
    has_markers = cc.has_report_markers(report_text)
    answer = cc.extract_answer_json(report_text)
    answer_key = cc.load_json(Path(__file__).parent / "answer_key.json")

    facts_ok = False
    if answer is not None:
        total_ok = answer.get("total_public_functions") == answer_key["total_public_functions"]
        caller = answer.get("can_edit_issue_caller")
        caller_ok = isinstance(caller, str) and cc.normalize_qualname(caller) == cc.normalize_qualname(
            answer_key["can_edit_issue_caller"]
        )
        raise_ok = answer.get("modules_with_raise_statements") == answer_key["modules_with_raise_statements"]
        facts_ok = total_ok and caller_ok and raise_ok

    passed = has_markers and not missing_headings and facts_ok
    score = 0.3 * has_markers + 0.3 * (not missing_headings) + 0.4 * facts_ok
    reason = (
        "report shape and facts correct"
        if passed
        else (
            f"markers present: {has_markers}; missing headings: {missing_headings}; "
            f"facts correct: {facts_ok} (parsed answer: {answer})"
        )
    )
    return cc.result(passed, score, reason, got=answer, want=answer_key)


if __name__ == "__main__":
    raise SystemExit(cc.run_checker_main(check))
