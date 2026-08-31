#!/usr/bin/env python3
"""Checker for fix-discount-tier-off-by-one: tests pass, test file untouched."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402

PROTECTED_FILES = ["test_discounts.py"]


def check(sandbox: Path, fixture: Path, report_text: str) -> dict:
    return cc.edit_task_check(sandbox, fixture, PROTECTED_FILES)


if __name__ == "__main__":
    raise SystemExit(cc.run_checker_main(check))
