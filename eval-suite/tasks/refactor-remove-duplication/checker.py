#!/usr/bin/env python3
"""Checker for refactor-remove-duplication: shared impl, no literal duplication, tests pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402

PROTECTED_FILES = ["test_shipping.py"]
DUPLICATE_MARKER = "distance_km / 800.0"
CANDIDATE_FILES = ["shipping_us.py", "shipping_eu.py", "shipping_common.py"]


def check(sandbox: Path, fixture: Path, report_text: str) -> dict:
    base = cc.edit_task_check(sandbox, fixture, PROTECTED_FILES)
    if not base["passed"]:
        return base

    common_path = sandbox / "shipping_common.py"
    has_common_function = cc.has_top_level_function(common_path, "estimate_shipping_days")

    files_with_marker = [
        name
        for name in CANDIDATE_FILES
        if (sandbox / name).is_file()
        and DUPLICATE_MARKER in (sandbox / name).read_text(encoding="utf-8")
    ]
    duplication_removed = len(files_with_marker) <= 1

    still_importable = False
    error = None
    try:
        us_mod = cc.import_module_from_sandbox(sandbox, "shipping_us")
        eu_mod = cc.import_module_from_sandbox(sandbox, "shipping_eu")
        still_importable = (
            hasattr(us_mod, "estimate_shipping_days")
            and hasattr(eu_mod, "estimate_shipping_days")
            and us_mod.estimate_shipping_days(1600) == eu_mod.estimate_shipping_days(1600)
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    passed = has_common_function and duplication_removed and still_importable
    score = 0.25 + 0.25 * has_common_function + 0.25 * duplication_removed + 0.25 * still_importable
    reason = (
        "shared implementation extracted correctly and tests pass"
        if passed
        else (
            f"shipping_common.estimate_shipping_days defined: {has_common_function}; "
            f"files still containing the duplicated body: {files_with_marker}; "
            f"both modules still importable/consistent: {still_importable}"
            + (f"; error: {error}" if error else "")
        )
    )
    return cc.result(passed, score, reason, pytest=base["details"].get("pytest"))


if __name__ == "__main__":
    raise SystemExit(cc.run_checker_main(check))
