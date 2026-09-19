import json
import tempfile
import types
from pathlib import Path
from unittest.mock import patch
import sys

import pytest

from experiments.agentic_router import live_matrix


def test_frozen_matrix_scope_and_deadline():
    assert live_matrix.TASK_SELECTION_FROZEN
    assert len(live_matrix.TASKS) == 4
    assert "swebench-sympy-66abe976e0" in live_matrix.TASKS
    assert not any("django" in slug or "astropy" in slug for slug in live_matrix.TASKS)
    assert live_matrix._selection_is_v3()
    assert len(live_matrix.CONDITIONS) == 3
    assert live_matrix.SEED == 1
    assert live_matrix.AGENT_CAP_S == 1200
    assert live_matrix.CELL_WALL_CAP_S <= 1800
    assert not set(live_matrix.TASKS) & set(live_matrix.pilot_campaign.HOLDOUT_TASKS)
    assert live_matrix.pilot_campaign.DRAIN.astimezone(live_matrix.timezone.utc) < live_matrix.GPU_EXPIRY


def test_task_selection_gate_prevents_any_work_until_replacement_approved():
    with patch.object(live_matrix, "TASK_SELECTION_FROZEN", False):
        assert live_matrix.run()["state"] == "task_selection_pending_replacement"


def test_sanity_reuse_requires_current_image_and_eval_fingerprint():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sanity = root / "grader_sanity"
        sanity.mkdir()
        slug = live_matrix.TASKS[0]
        saved = sanity / f"{slug}.sanity.json"
        saved.write_text(json.dumps({"sanity_pass": True, "instance_id": "case-1",
                                     "fingerprint": {"docker_image_id": "old-image"}}))
        fake_runner = types.SimpleNamespace(load_instances=lambda _slug: iter([{"instance_id": "case-1"}]))
        fake_verifier = types.SimpleNamespace(fingerprint=lambda _instance: {"docker_image_id": "new-image"})
        with (patch.object(live_matrix, "LIVE_LOCK", root / "pilot_live.lock"),
              patch.dict(sys.modules, {"run_swebench": fake_runner,
                                       "verify_official_grader": fake_verifier})):
            assert not live_matrix._sanity_valid(slug)
            saved.write_text(json.dumps({"sanity_pass": True, "instance_id": "case-1",
                                         "fingerprint": {"docker_image_id": "new-image"}}))
            assert live_matrix._sanity_valid(slug)


def test_existing_running_checkpoint_blocks_repeat_spend():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        a, b = root / "live.lock", root / "replay.lock"
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
        pilot_status = root / "pilot_live_status.json"
        pilot_status.write_text(json.dumps({"state": "finished_development"}))
        slug = live_matrix.TASKS[0]
        condition = live_matrix.CONDITIONS[0]
        (checkpoint_dir / f"{live_matrix._cell_key(slug, condition)}.json").write_text(
            json.dumps({"state": "running"})
        )
        with (
            patch.object(live_matrix, "LIVE_LOCK", a),
            patch.object(live_matrix, "REPLAY_LOCK", b),
            patch.object(live_matrix, "CHECKPOINTS", checkpoint_dir),
            patch.object(live_matrix, "STATUS", root / "status.json"),
            patch.object(live_matrix, "PAUSE", root / "no-pause"),
            patch.object(live_matrix, "PILOT_LIVE_STATUS", pilot_status),
            patch.object(live_matrix, "TASK_SELECTION_FROZEN", True),
            patch.object(live_matrix.pilot_campaign, "_bounded_flight") as flight,
            patch.object(live_matrix, "_run_cell") as run_cell,
        ):
            result = live_matrix.run()
        assert result["state"] == "manual_review_required"
        assert run_cell.call_count == 0
        flight.assert_not_called()


def test_old_valid_checkpoint_does_not_silently_resume_new_protocol():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
        slug = live_matrix.TASKS[0]
        condition = live_matrix.CONDITIONS[0]
        (checkpoint_dir / f"{live_matrix._cell_key(slug, condition)}.json").write_text(
            json.dumps({"state": "graded_valid", "protocol_fingerprint": "prior-workload"})
        )
        (root / "pilot_live_status.json").write_text(json.dumps({"state": "finished_development"}))
        with (
            patch.object(live_matrix, "LIVE_LOCK", root / "live.lock"),
            patch.object(live_matrix, "REPLAY_LOCK", root / "replay.lock"),
            patch.object(live_matrix, "CHECKPOINTS", checkpoint_dir),
            patch.object(live_matrix, "STATUS", root / "status.json"),
            patch.object(live_matrix, "PAUSE", root / "no-pause"),
            patch.object(live_matrix, "PILOT_LIVE_STATUS", root / "pilot_live_status.json"),
            patch.object(live_matrix, "TASK_SELECTION_FROZEN", True),
            patch.object(live_matrix.pilot_campaign, "_bounded_flight") as flight,
            patch.object(live_matrix, "_run_cell") as run_cell,
        ):
            result = live_matrix.run()
        assert result["state"] == "manual_review_required"
        assert result["reason"] == "checkpoint_protocol_mismatch"
        flight.assert_not_called()
        run_cell.assert_not_called()


def test_pause_prevents_preflight_and_cells():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pause = root / "PAUSE_NEW_CELLS"
        pause.touch()
        with (
            patch.object(live_matrix, "LIVE_LOCK", root / "live.lock"),
            patch.object(live_matrix, "REPLAY_LOCK", root / "replay.lock"),
            patch.object(live_matrix, "STATUS", root / "status.json"),
            patch.object(live_matrix, "PAUSE", pause),
            patch.object(live_matrix, "TASK_SELECTION_FROZEN", True),
            patch.object(live_matrix.pilot_campaign, "_bounded_flight") as flight,
            patch.object(live_matrix, "_run_cell") as run_cell,
        ):
            result = live_matrix.run()
        assert result["state"] == "paused_or_drained"
        flight.assert_not_called()
        run_cell.assert_not_called()


def test_cli_without_run_never_launches():
    with patch("sys.argv", ["live_matrix"]):
        with pytest.raises(SystemExit) as exc:
            live_matrix.main()
    assert exc.value.code == 2
