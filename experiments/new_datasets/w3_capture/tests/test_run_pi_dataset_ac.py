"""Offline unit and integration tests for run_pi_dataset_ac.py.

No Docker daemon, network access, or GPU inference required.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

W3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W3))

import run_pi_dataset_ac as runner


def _mock_cell(instance_id: str = "conan-io__conan-13788", backend: str = "edge") -> runner.PlannedCell:
    return runner.PlannedCell(
        instance={
            "instance_id": instance_id,
            "repo": "conan-io/conan",
            "problem_statement": "test problem statement",
        },
        source_instances_file=runner.SOURCE_FILES[0],
        backend=backend,
        image=f"sweb.eval.arm64.{instance_id}:latest",
        prompt="prompt content",
        execution_order={"seed": 42, "backends_in_order": [backend, "cloud"], "position": 1},
    )


class RunPiDatasetAcTests(unittest.TestCase):
    def test_plan_deduplicates_twenty_one_tasks_into_forty_two_cells(self):
        plan, protocol, fingerprint = runner.build_plan()
        self.assertEqual(protocol["protocol_version"], "pi-dataset-ac-v1")
        self.assertEqual(protocol["task_count"], 21)
        self.assertEqual(protocol["cell_count"], 42)
        self.assertEqual(len(plan), 42)
        self.assertEqual(len(protocol["tasks"]), 21)
        self.assertEqual(protocol["max_active_seconds"], 1800)
        self.assertIsNone(protocol["max_turns"])
        self.assertEqual(protocol["identical_action_repeat_limit"], 4)

        # Unique cell IDs
        cell_ids = [cell.cell_id for cell in plan]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

        # Backends
        self.assertEqual(set(runner.BACKENDS.keys()), {"edge", "cloud"})
        self.assertEqual(runner.BACKENDS["edge"]["base_url"], "http://host.docker.internal:18012")
        self.assertEqual(runner.BACKENDS["edge"]["model"], "local")
        self.assertEqual(runner.BACKENDS["cloud"]["base_url"], "http://host.docker.internal:18011")
        self.assertEqual(runner.BACKENDS["cloud"]["model"], "deepseek-v4-flash")

    def test_backend_order_is_randomized_and_deterministic_per_task(self):
        plan1, _, _ = runner.build_plan()
        plan2, _, _ = runner.build_plan()
        self.assertEqual([c.cell_id for c in plan1], [c.cell_id for c in plan2])

        # Verify each task has both positions 1 and 2
        for task_id in {c.instance_id for c in plan1}:
            task_cells = [c for c in plan1 if c.instance_id == task_id]
            self.assertEqual(len(task_cells), 2)
            positions = {c.execution_order["position"] for c in task_cells}
            self.assertEqual(positions, {1, 2})
            backends = {c.backend for c in task_cells}
            self.assertEqual(backends, {"edge", "cloud"})

    def test_protocol_checkpoint_is_atomic_and_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            runner.ensure_output_protocol(root, {"version": "v1"}, "fingerprint-123")
            protocol_path = root / "protocol.json"
            self.assertTrue(protocol_path.is_file())

            # Idempotent same fingerprint
            runner.ensure_output_protocol(root, {"version": "v1"}, "fingerprint-123")

            # Mismatched fingerprint raises PiDatasetProtocolError
            with self.assertRaises(runner.PiDatasetProtocolError):
                runner.ensure_output_protocol(root, {"version": "v1"}, "wrong-fingerprint")

    def test_ensure_cell_meta_creates_correct_initial_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            cell = _mock_cell()
            meta_path = runner.ensure_cell_meta(root, cell, "test-fingerprint")
            self.assertTrue(meta_path.is_file())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "pending")
            self.assertEqual(meta["classification"], "pending")
            self.assertEqual(meta["max_active_seconds"], 1800)
            self.assertIsNone(meta["max_turns"])
            self.assertEqual(meta["official_grade_status"], "not_run")
            self.assertIsNone(meta["official_pass"])

    def test_classification_mapping(self):
        self.assertEqual(runner._classify_result("repeated_action", 0), ("repeated_action", "repeat"))
        self.assertEqual(runner._classify_result("trajectory_deadline", 0), ("trajectory_deadline", "timeout"))
        self.assertEqual(runner._classify_result("process_error", 1), ("process_error", "process"))
        self.assertEqual(runner._classify_result("completed", 0), ("completed", "completed"))

    def test_execute_cell_success_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            cell = _mock_cell()
            fingerprint = "test-fp"

            mock_api = MagicMock()
            mock_api.start_container.return_value = None
            mock_api.extract_patch.return_value = "diff --git a/test.py b/test.py\n+fixed\n"
            mock_api.sh.return_value = SimpleNamespace(returncode=0)

            fake_run_result = SimpleNamespace(
                stdout='{"type":"agent_end"}\n',
                returncode=0,
                termination_reason="completed",
                timed_out=False,
                action_count=5,
                model_turn_count=6,
                repeated_action_count=1,
                repeated_action_signature=None,
                output_tokens=150,
            )
            mock_harness = MagicMock()
            mock_harness.setup_pi_container.return_value = None
            mock_harness.run_pi.return_value = fake_run_result

            with patch.object(runner, "_capture_api", return_value=mock_api):
                with patch.object(runner, "_pi_harness", return_value=mock_harness):
                    meta = runner.execute_cell(root, cell, fingerprint)

            self.assertEqual(meta["status"], "completed")
            self.assertEqual(meta["classification"], "completed")
            self.assertEqual(meta["termination_reason"], "completed")
            self.assertFalse(meta["timed_out"])
            self.assertEqual(meta["action_count"], 5)
            self.assertEqual(meta["model_turn_count"], 6)
            self.assertEqual(meta["output_tokens"], 150)
            self.assertTrue(meta["patch_nonempty"])
            self.assertEqual(mock_api.sh.call_count, 1)  # docker rm -f

            # Resume: re-calling execute_cell returns cached meta without re-running
            mock_api.reset_mock()
            resumed = runner.execute_cell(root, cell, fingerprint)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(mock_api.start_container.call_count, 0)

    def test_execute_cell_repeat_breaker_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            cell = _mock_cell()
            fingerprint = "test-fp"

            mock_api = MagicMock()
            mock_api.extract_patch.return_value = ""
            fake_run_result = SimpleNamespace(
                stdout="",
                returncode=0,
                termination_reason="repeated_action",
                timed_out=False,
                action_count=4,
                model_turn_count=4,
                repeated_action_count=4,
                repeated_action_signature="bash:pwd",
                output_tokens=80,
            )
            mock_harness = MagicMock()
            mock_harness.run_pi.return_value = fake_run_result

            with patch.object(runner, "_capture_api", return_value=mock_api):
                with patch.object(runner, "_pi_harness", return_value=mock_harness):
                    meta = runner.execute_cell(root, cell, fingerprint)

            self.assertEqual(meta["status"], "completed")
            self.assertEqual(meta["classification"], "repeat")
            self.assertEqual(meta["termination_reason"], "repeated_action")
            self.assertFalse(meta["timed_out"])

    def test_execute_cell_timeout_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            cell = _mock_cell()
            fingerprint = "test-fp"

            mock_api = MagicMock()
            mock_api.extract_patch.return_value = ""
            fake_run_result = SimpleNamespace(
                stdout="",
                returncode=None,
                termination_reason="trajectory_deadline",
                timed_out=True,
                action_count=12,
                model_turn_count=15,
                repeated_action_count=1,
                repeated_action_signature=None,
                output_tokens=5000,
            )
            mock_harness = MagicMock()
            mock_harness.run_pi.return_value = fake_run_result

            with patch.object(runner, "_capture_api", return_value=mock_api):
                with patch.object(runner, "_pi_harness", return_value=mock_harness):
                    meta = runner.execute_cell(root, cell, fingerprint)

            self.assertEqual(meta["status"], "completed")
            self.assertEqual(meta["classification"], "timeout")
            self.assertEqual(meta["termination_reason"], "trajectory_deadline")
            self.assertTrue(meta["timed_out"])

    def test_execute_cell_setup_error_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            cell = _mock_cell()
            fingerprint = "test-fp"

            mock_api = MagicMock()
            mock_api.start_container.side_effect = RuntimeError("Docker daemon unavailable")
            mock_harness = MagicMock()

            with patch.object(runner, "_capture_api", return_value=mock_api):
                with patch.object(runner, "_pi_harness", return_value=mock_harness):
                    meta = runner.execute_cell(root, cell, fingerprint)

            self.assertEqual(meta["status"], "failed")
            self.assertEqual(meta["classification"], "setup")
            self.assertEqual(meta["termination_reason"], "setup_error")
            self.assertTrue(meta["infrastructure_failure"])

    def test_execute_cell_process_error_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            cell = _mock_cell()
            fingerprint = "test-fp"

            mock_api = MagicMock()
            fake_run_result = SimpleNamespace(
                stdout="",
                returncode=137,
                termination_reason="process_error",
                timed_out=False,
                action_count=1,
                model_turn_count=1,
                repeated_action_count=0,
                repeated_action_signature=None,
                output_tokens=None,
                stderr="Killed",
            )
            mock_harness = MagicMock()
            mock_harness.run_pi.return_value = fake_run_result

            with patch.object(runner, "_capture_api", return_value=mock_api):
                with patch.object(runner, "_pi_harness", return_value=mock_harness):
                    meta = runner.execute_cell(root, cell, fingerprint)

            self.assertEqual(meta["status"], "failed")
            self.assertEqual(meta["classification"], "process")
            self.assertEqual(meta["termination_reason"], "process_error")

    def test_record_official_grade_updates_metadata_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            _, protocol, fingerprint = runner.build_plan()
            runner.ensure_output_protocol(root, protocol, fingerprint)
            cell = _mock_cell()
            runner.ensure_cell_meta(root, cell, fingerprint)

            # Mark cell as completed
            meta_path = root / cell.cell_id / "run_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "completed"
            runner._atomic_write_json(meta_path, meta)

            runner.record_official_grade(
                root,
                cell.cell_id,
                official_pass=True,
                grader_reference="reports/grade-1.json",
                grade_details={"grade_seconds": 12.5},
            )

            updated_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(updated_meta["official_grade_status"], "graded")
            self.assertIs(updated_meta["official_pass"], True)
            self.assertEqual(updated_meta["official_grader_reference"], "reports/grade-1.json")

            # Cannot record twice
            with self.assertRaises(runner.PiDatasetProtocolError):
                runner.record_official_grade(
                    root,
                    cell.cell_id,
                    official_pass=False,
                    grader_reference="reports/grade-2.json",
                )


if __name__ == "__main__":
    unittest.main()
