#!/usr/bin/env python3
"""Checker for fanout-parallel-bugfix: three independent fixes, report shape, tests pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402

REQUIRED_HEADINGS = [
    "## Executive Summary",
    "## Fixes Applied",
    "## Open Questions",
]
PROTECTED_FILES = ["test_mod_a.py", "test_mod_b.py", "test_mod_c.py"]
EXPECTED_TEST_COUNT = 11  # 5 (mod_a) + 4 (mod_b) + 2 (mod_c)


def check(sandbox: Path, fixture: Path, report_text: str) -> dict:
    if not report_text.strip():
        return cc.result(False, 0.0, "no report captured")

    changed = cc.files_unchanged(sandbox, fixture, PROTECTED_FILES)
    if changed:
        return cc.result(
            False, 0.0, f"protected test file(s) modified: {changed}", changed=changed
        )

    pytest_result = cc.run_pytest(sandbox)
    tests_ok = cc.pytest_all_passed(pytest_result)
    summary = pytest_result.get("summary") or {}
    right_count = summary.get("tests") == EXPECTED_TEST_COUNT

    missing_headings = cc.has_required_headings(report_text, REQUIRED_HEADINGS)
    has_markers = cc.has_report_markers(report_text)

    passed = tests_ok and right_count and has_markers and not missing_headings
    score = 0.5 * (tests_ok and right_count) + 0.25 * has_markers + 0.25 * (not missing_headings)
    reason = (
        "all three fixes correct and report well-formed"
        if passed
        else (
            f"tests_ok={tests_ok} (collected {summary.get('tests')}, expected {EXPECTED_TEST_COUNT}); "
            f"markers present: {has_markers}; missing headings: {missing_headings}"
        )
    )
    return cc.result(passed, score, reason, pytest=pytest_result)


if __name__ == "__main__":
    raise SystemExit(cc.run_checker_main(check))
