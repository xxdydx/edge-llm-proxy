#!/usr/bin/env python3
"""Checker for fanout-repo-audit: report shape plus predeclared facts about the codebase."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402

REQUIRED_HEADINGS = [
    "## Executive Summary",
    "## Findings from Each Agent",
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
        got_total = answer.get("total_public_functions")
        got_missing = answer.get("packages_without_tests")
        total_ok = got_total == answer_key["total_public_functions"]
        missing_ok = isinstance(got_missing, list) and sorted(
            str(x) for x in got_missing
        ) == sorted(answer_key["packages_without_tests"])
        facts_ok = total_ok and missing_ok

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
