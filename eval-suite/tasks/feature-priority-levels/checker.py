#!/usr/bin/env python3
"""Checker for feature-priority-levels: full test suite passes, test files untouched."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402

PROTECTED_FILES = [
    "test_models.py",
    "test_storage.py",
    "test_validation.py",
    "test_workflow.py",
    "test_permissions.py",
    "test_search.py",
    "test_notifications.py",
    "test_reporting.py",
    "test_api.py",
    "test_priority_feature.py",
]
EXPECTED_TEST_COUNT = 39  # 28 existing + 11 in test_priority_feature.py


def check(sandbox: Path, fixture: Path, report_text: str) -> dict:
    base = cc.edit_task_check(sandbox, fixture, PROTECTED_FILES)
    summary = (base["details"].get("pytest") or {}).get("summary") or {}
    right_count = summary.get("tests") == EXPECTED_TEST_COUNT
    passed = base["passed"] and right_count
    score = base["score"] * (0.5 + 0.5 * right_count)
    reason = base["reason"] if base["passed"] else base["reason"]
    if base["passed"] and not right_count:
        reason = f"all collected tests pass, but collected {summary.get('tests')} not {EXPECTED_TEST_COUNT}"
    return cc.result(passed, score, reason, pytest=base["details"].get("pytest"))


if __name__ == "__main__":
    raise SystemExit(cc.run_checker_main(check))
