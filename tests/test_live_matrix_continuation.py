import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.agentic_router import live_matrix_continuation as continuation


def _frozen_prefix(directory: Path, protocol: str) -> None:
    for i, (slug, condition) in enumerate(continuation.ORDER[:continuation.SKIP_INDEX + 1]):
        row = {"state": "graded_valid" if i < continuation.SKIP_INDEX else "infrastructure_invalid",
               "protocol_fingerprint": protocol}
        if i == continuation.SKIP_INDEX:
            row.update({"controller_timed_out": False,
                        "verdict_detail": "job did not complete | error: claude timed out at configured per-job hard deadline"})
        (directory / f"{continuation.matrix._cell_key(slug, condition)}.json").write_text(json.dumps(row))


def test_continuation_exactly_seven_never_attempted_cells():
    assert len(continuation.ORDER) == 12
    assert continuation.ORDER[continuation.SKIP_INDEX] == ("swebench-sympy-66abe976e0", "local")
    assert len(continuation.ORDER[continuation.SKIP_INDEX + 1:]) == 7
    assert not set(slug for slug, _ in continuation.ORDER) & set(continuation.matrix.pilot_campaign.HOLDOUT_TASKS)


def test_continuation_refuses_changed_prefix_or_attempted_future():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _frozen_prefix(root, "frozen")
        with patch.object(continuation.matrix, "CHECKPOINTS", root):
            assert continuation._validate_existing("frozen") is None
            assert continuation._validate_existing("changed") == (
                "protocol_fingerprint_mismatch:swebench-flask-70ca03af28:cloud")
            slug, condition = continuation.ORDER[continuation.SKIP_INDEX + 1]
            (root / f"{continuation.matrix._cell_key(slug, condition)}.json").write_text(
                json.dumps({"state": "running", "protocol_fingerprint": "frozen"}))
            assert continuation._validate_existing("frozen") == (
                f"future_cell_already_attempted:{slug}:{condition}")
