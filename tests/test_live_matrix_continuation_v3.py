import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.agentic_router import live_matrix_continuation_v3 as continuation


def test_only_pylint_cells_can_run():
    assert continuation.ORDER[continuation.LAST_ATTEMPTED + 1:] == (
        ("swebench-pylint-1715969d0b", "cloud"),
        ("swebench-pylint-1715969d0b", "local"),
        ("swebench-pylint-1715969d0b", "routing-learned-agentic-heuristic"),
    )


def test_current_running_pytest_heuristic_is_not_mistaken_for_timeout():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for slug, condition in continuation.ORDER[:continuation.LAST_ATTEMPTED + 1]:
            timeout = (slug, condition) in continuation.SKIPPED
            row = {"state": "infrastructure_invalid" if timeout else "graded_valid",
                   "protocol_fingerprint": "frozen"}
            if timeout:
                row.update({"controller_timed_out": False,
                            "verdict_detail": "claude timed out at configured per-job hard deadline"})
            (root / f"{continuation.matrix._cell_key(slug, condition)}.json").write_text(json.dumps(row))
        with patch.object(continuation.matrix, "CHECKPOINTS", root):
            assert continuation._validate_existing("frozen") is None
            slug, condition = continuation.ORDER[continuation.LAST_ATTEMPTED]
            (root / f"{continuation.matrix._cell_key(slug, condition)}.json").write_text(
                json.dumps({"state": "running", "protocol_fingerprint": "frozen"}))
            assert continuation._validate_existing("frozen") == f"timeout_checkpoint_mismatch:{slug}:{condition}"
