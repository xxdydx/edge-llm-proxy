#!/usr/bin/env python3
"""Checker for explore-pricing-api: exact match against a predeclared answer key."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402


def check(sandbox: Path, fixture: Path, report_text: str) -> dict:
    answer_key = cc.load_json(Path(__file__).parent / "answer_key.json")
    answer = cc.extract_answer_json(report_text)
    if answer is None:
        return cc.result(False, 0.0, "no parseable JSON answer block found in report")

    got_functions = answer.get("discounts_public_functions")
    got_mutator = answer.get("mutator")
    want_functions = answer_key["discounts_public_functions"]
    want_mutator = answer_key["mutator"]

    functions_ok = isinstance(got_functions, list) and [str(x) for x in got_functions] == want_functions
    mutator_ok = isinstance(got_mutator, str) and got_mutator.strip() == want_mutator

    score = 0.5 * functions_ok + 0.5 * mutator_ok
    passed = functions_ok and mutator_ok
    reason = (
        "both answers correct"
        if passed
        else (
            f"discounts_public_functions {'ok' if functions_ok else f'wrong: got {got_functions!r}'}; "
            f"mutator {'ok' if mutator_ok else f'wrong: got {got_mutator!r}'}"
        )
    )
    return cc.result(passed, score, reason, got=answer, want=answer_key)


if __name__ == "__main__":
    raise SystemExit(cc.run_checker_main(check))
