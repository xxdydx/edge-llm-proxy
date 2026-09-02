#!/usr/bin/env python3
"""Checker for explore-config-defaults: exact match against a predeclared answer key."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402


def check(sandbox: Path, fixture: Path, report_text: str) -> dict:
    answer_key = cc.load_json(Path(__file__).parent / "answer_key.json")
    answer = cc.extract_answer_json(report_text)
    if answer is None:
        return cc.result(False, 0.0, "no parseable JSON answer block found in report")

    got_count = answer.get("distinct_exception_types_caught")
    got_raiser = answer.get("unknown_key_raiser")
    want_count = answer_key["distinct_exception_types_caught"]
    want_raiser = answer_key["unknown_key_raiser"]

    count_ok = got_count == want_count
    raiser_ok = (
        isinstance(got_raiser, str)
        and cc.normalize_qualname(got_raiser) == cc.normalize_qualname(want_raiser)
    )

    score = 0.5 * count_ok + 0.5 * raiser_ok
    passed = count_ok and raiser_ok
    reason = (
        "both answers correct"
        if passed
        else (
            f"distinct_exception_types_caught {'ok' if count_ok else f'wrong: got {got_count!r}'}; "
            f"unknown_key_raiser {'ok' if raiser_ok else f'wrong: got {got_raiser!r}'}"
        )
    )
    return cc.result(passed, score, reason, got=answer, want=answer_key)


if __name__ == "__main__":
    raise SystemExit(cc.run_checker_main(check))
