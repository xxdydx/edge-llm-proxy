#!/usr/bin/env python3
"""Checker for refactor-rename-for-clarity: full rename, tests still pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402

PROTECTED_FILES = ["test_stats.py"]


def check(sandbox: Path, fixture: Path, report_text: str) -> dict:
    base = cc.edit_task_check(sandbox, fixture, PROTECTED_FILES)
    if not base["passed"]:
        return base

    old_name_count = cc.grep_count(sandbox, r"\bcalc\b", glob="*.py")
    rename_complete = old_name_count == 0
    has_new_function = cc.has_top_level_function(sandbox / "stats.py", "weighted_average")

    passed = rename_complete and has_new_function
    score = 0.5 + 0.25 * rename_complete + 0.25 * has_new_function
    reason = (
        "rename complete and tests pass"
        if passed
        else (
            f"remaining references to 'calc': {old_name_count}; "
            f"stats.weighted_average defined: {has_new_function}"
        )
    )
    return cc.result(passed, score, reason, pytest=base["details"].get("pytest"))


if __name__ == "__main__":
    raise SystemExit(cc.run_checker_main(check))
