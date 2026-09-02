#!/usr/bin/env python3
"""Checker for explore-call-graph: exact match against a predeclared answer key."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import checker_common as cc  # noqa: E402


def check(sandbox: Path, fixture: Path, report_text: str) -> dict:
    answer_key = cc.load_json(Path(__file__).parent / "answer_key.json")
    answer = cc.extract_answer_json(report_text)
    if answer is None:
        return cc.result(False, 0.0, "no parseable JSON answer block found in report")

    got_callers = answer.get("callers")
    got_count = answer.get("call_site_count")
    want_callers = answer_key["callers"]
    want_count = answer_key["call_site_count"]

    callers_ok = isinstance(got_callers, list) and [
        cc.normalize_qualname(str(x)) for x in got_callers
    ] == [cc.normalize_qualname(w) for w in want_callers]
    count_ok = got_count == want_count

    score = 0.5 * callers_ok + 0.5 * count_ok
    passed = callers_ok and count_ok
    reason = (
        "both answers correct"
        if passed
        else (
            f"callers {'ok' if callers_ok else f'wrong: got {got_callers!r}'}; "
            f"call_site_count {'ok' if count_ok else f'wrong: got {got_count!r}'}"
        )
    )
    return cc.result(passed, score, reason, got=answer, want=answer_key)


if __name__ == "__main__":
    raise SystemExit(cc.run_checker_main(check))
