import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.agentic_router import live_matrix_continuation_v2 as continuation


def _prefix(directory: Path, protocol: str) -> None:
    for slug, condition in continuation.ORDER[:continuation.LAST_ATTEMPTED + 1]:
        timeout = (slug, condition) in continuation.SKIPPED
        row = {"state": "infrastructure_invalid" if timeout else "graded_valid",
               "protocol_fingerprint": protocol}
        if timeout:
            row.update({"controller_timed_out": False,
                        "verdict_detail": "job did not complete | error: claude timed out at configured per-job hard deadline"})
        (directory / f"{continuation.matrix._cell_key(slug, condition)}.json").write_text(json.dumps(row))


def test_only_four_never_attempted_cells_remain():
    assert continuation.ORDER[continuation.LAST_ATTEMPTED + 1:] == (
        ("swebench-pytest-f8692a712a", "routing-learned-agentic-heuristic"),
        ("swebench-pylint-1715969d0b", "cloud"),
        ("swebench-pylint-1715969d0b", "local"),
        ("swebench-pylint-1715969d0b", "routing-learned-agentic-heuristic"),
    )
    assert not set(s for s, _ in continuation.ORDER) & set(continuation.matrix.pilot_campaign.HOLDOUT_TASKS)


def test_two_timeout_prefix_and_future_attempt_fail_closed():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _prefix(root, "frozen")
        with patch.object(continuation.matrix, "CHECKPOINTS", root):
            assert continuation._validate_existing("frozen") is None
            assert continuation._validate_existing("changed").startswith("protocol_fingerprint_mismatch:")
            slug, condition = continuation.ORDER[continuation.LAST_ATTEMPTED + 1]
            (root / f"{continuation.matrix._cell_key(slug, condition)}.json").write_text(
                json.dumps({"state": "running", "protocol_fingerprint": "frozen"}))
            assert continuation._validate_existing("frozen") == (
                f"future_cell_already_attempted:{slug}:{condition}")
