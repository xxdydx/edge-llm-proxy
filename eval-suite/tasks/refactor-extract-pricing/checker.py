#!/usr/bin/env python3
"""Checker for refactor-extract-pricing: extraction present, tests still pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402

PROTECTED_FILES = ["test_orders.py"]


def check(sandbox: Path, fixture: Path, report_text: str) -> dict:
    base = cc.edit_task_check(sandbox, fixture, PROTECTED_FILES)
    if not base["passed"]:
        return base

    orders_py = sandbox / "orders.py"
    has_function = cc.has_top_level_function(orders_py, "compute_total")
    calls_it = cc.function_calls_name(orders_py, "process_order", "compute_total")

    correct_value = False
    error = None
    if has_function:
        try:
            module = cc.import_module_from_sandbox(sandbox, "orders")
            value = module.compute_total([{"price": 10.0, "qty": 2}], 0.1)
            correct_value = isinstance(value, (int, float)) and abs(value - 18.0) < 1e-6
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

    passed = has_function and calls_it and correct_value
    score = 0.25 + 0.25 * has_function + 0.25 * calls_it + 0.25 * correct_value
    reason = (
        "extraction correct and tests pass"
        if passed
        else (
            f"compute_total defined: {has_function}; process_order calls it: {calls_it}; "
            f"compute_total([price=10.0,qty=2], 0.1) == 18.0: {correct_value}"
            + (f"; error: {error}" if error else "")
        )
    )
    return cc.result(passed, score, reason, pytest=base["details"].get("pytest"))


if __name__ == "__main__":
    raise SystemExit(cc.run_checker_main(check))
