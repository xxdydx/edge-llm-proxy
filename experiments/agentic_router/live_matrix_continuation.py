"""Explicit one-time continuation after Sympy/local's recorded timeout.

Never retries an attempted cell. Keeps the v1 controller and its protocol
fingerprint untouched, and writes an honest terminal status on every exit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from . import live_matrix as matrix

SKIPPED = ("swebench-sympy-66abe976e0", "local")
ORDER = tuple((slug, condition) for slug in matrix.TASKS for condition in matrix.CONDITIONS)
SKIP_INDEX = ORDER.index(SKIPPED)


def _status(state: str, **extra: object) -> dict:
    result = {
        "state": state,
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "continuation": "after_sympy_local_timeout_v1",
        "skipped_invalid_cell": {"slug": SKIPPED[0], "condition": SKIPPED[1],
                                 "reason": "claude_per_job_hard_deadline_no_retry"},
        **extra,
    }
    matrix._atomic(matrix.STATUS, result)
    return result


def _validate_existing(protocol: str) -> str | None:
    """Check exact frozen prefix and refuse any already-attempted future cell."""
    for i, (slug, condition) in enumerate(ORDER):
        prior = matrix._prior_checkpoint(slug, condition)
        if i < SKIP_INDEX:
            if prior is None or prior.get("state") != "graded_valid":
                return f"expected_valid_prefix:{slug}:{condition}"
        elif i == SKIP_INDEX:
            if (prior is None or prior.get("state") != "infrastructure_invalid"
                    or prior.get("controller_timed_out") is not False
                    or "claude timed out at configured per-job hard deadline"
                    not in str(prior.get("verdict_detail", ""))):
                return "sympy_local_timeout_checkpoint_mismatch"
        elif prior is not None and prior.get("state") != "graded_valid":
            return f"future_cell_already_attempted:{slug}:{condition}"
        if prior is not None and prior.get("protocol_fingerprint") != protocol:
            return f"protocol_fingerprint_mismatch:{slug}:{condition}"
    return None


def run() -> dict:
    if not matrix.TASK_SELECTION_FROZEN or not matrix._selection_is_v3():
        return _status("selection_manifest_mismatch")
    protocol = matrix._protocol_fingerprint()
    with ExitStack() as stack:
        try:
            for path in (matrix.LIVE_LOCK, matrix.REPLAY_LOCK):
                lock = stack.enter_context(path.open("a+"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _status("pilot_lock_unavailable")
        if not matrix._pilot_finished():
            return _status("pilot_collector_not_finished")
        issue = _validate_existing(protocol)
        if issue:
            return _status("manual_review_required", reason=issue)
        if matrix.PAUSE.exists() or (matrix.pilot_campaign.DRAIN - datetime.now(matrix.pilot_campaign.SGT)).total_seconds() <= 90:
            return _status("paused_or_drained")
        try:
            matrix.pilot_campaign._bounded_flight(matrix.pilot_campaign.LOCAL_27B.name)
            matrix.pilot_campaign._bounded_flight(matrix.pilot_campaign.CLOUD_DEEPSEEK.name)
        except Exception as exc:
            return _status("preflight_failed", error_type=type(exc).__name__)
        attempted = 0
        for slug, condition in ORDER[SKIP_INDEX + 1:]:
            if matrix.PAUSE.exists() or (matrix.pilot_campaign.DRAIN - datetime.now(matrix.pilot_campaign.SGT)).total_seconds() <= 90:
                return _status("paused_or_drained", attempted_new_cells=attempted)
            prior = matrix._prior_checkpoint(slug, condition)
            if prior is not None:
                if prior.get("state") == "graded_valid" and prior.get("protocol_fingerprint") == protocol:
                    continue
                return _status("manual_review_required", reason=f"future_cell_already_attempted:{slug}:{condition}")
            _status("running_remaining_cells", active_cell={"slug": slug, "condition": condition},
                    attempted_new_cells=attempted)
            result = matrix._run_cell(slug, condition)
            attempted += 1
            if result["state"] != "graded_valid":
                return _status("halted_on_invalid_cell", latest={"slug": slug, "condition": condition,
                                                                "state": result["state"]},
                               attempted_new_cells=attempted)
            _status("running_remaining_cells", latest={"slug": slug, "condition": condition,
                                                       "state": result["state"]},
                    attempted_new_cells=attempted)
        return _status("complete_with_recorded_invalid_cell", attempted_new_cells=attempted,
                       valid_cells=sum(matrix._completed(s, c) for s, c in ORDER))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("pass --run after verifying the recorded Sympy/local timeout")
    result = run()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "complete_with_recorded_invalid_cell" else 1


if __name__ == "__main__":
    raise SystemExit(main())
