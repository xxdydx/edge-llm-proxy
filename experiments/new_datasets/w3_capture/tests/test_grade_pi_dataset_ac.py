"""Offline unit and integration tests for grade_pi_dataset_ac.py.

No Docker daemon, network access, or GPU inference required.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

W3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W3))

import grade_pi_dataset_ac as grader
import run_pi_dataset_ac as runner


def _setup_capture_fixture(
    root: Path,
    *,
    tasks: list[tuple[str, str, str]],  # (iid, source_filename, backend)
    completed_iids: set[str],
) -> tuple[str, list[dict], dict[str, dict]]:
    task_docs = []
    instances_by_source: dict[Path, dict[str, dict]] = {src: {} for src in runner.SOURCE_FILES}

    for iid, source_name, _ in tasks:
        source_file = runner.ND / "w2_preflight" / source_name
        task_row = {
            "instance_id": iid,
            "repo": f"repo/{iid.split('__')[0]}",
            "problem_statement": f"problem for {iid}",
            "base_commit": "abcdef123456",
            "FAIL_TO_PASS": ["tests/test.py::test_pass"],
            "PASS_TO_PASS": [],
        }
        instances_by_source[source_file][iid] = task_row
        task_docs.append({
            "instance_id": iid,
            "source_instances_file": source_file.resolve().relative_to(runner.ND.resolve()).as_posix(),
            "source_sha256": grader._sha256_file(source_file),
            "image": f"sweb.eval.arm64.{iid}:latest",
            "prompt_sha256": "mock-sha",
        })

    protocol = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "task_count": len(tasks),
        "cell_count": len(tasks),
        "tasks": task_docs,
        "backends": runner.BACKENDS,
        "max_active_seconds": runner.MAX_ACTIVE_SECONDS,
        "max_turns": None,
        "identical_action_repeat_limit": runner.MAX_IDENTICAL_ACTION_REPEATS,
        "container_platform": "linux/arm64/v8",
        "official_grading": "separate_post_capture_phase",
    }
    fingerprint = runner._sha256_bytes(runner._canonical_json(protocol))
    runner.ensure_output_protocol(root, protocol, fingerprint)

    cells = []
    for iid, source_name, backend in tasks:
        cell_id = f"{iid}__{backend}"
        cell_dir = root / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = cell_dir / "attempt-001"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        patch_path = attempt_dir / "final_patch.diff"
        patch_path.write_text(f"patch diff for {cell_id}\n", encoding="utf-8")
        prompt_path = attempt_dir / "prompt.txt"
        prompt_path.write_text(f"prompt for {iid}", encoding="utf-8")

        is_completed = iid in completed_iids
        source_file = runner.ND / "w2_preflight" / source_name
        cell = runner.PlannedCell(
            instance=instances_by_source[source_file][iid],
            source_instances_file=source_file,
            backend=backend,
            image=f"sweb.eval.arm64.{iid}:latest",
            prompt=f"prompt for {iid}",
            execution_order={"seed": 1, "backends_in_order": [backend, "cloud"], "position": 1},
        )
        identity = runner._cell_identity(cell, fingerprint)
        rel_patch = f"{cell_id}/attempt-001/final_patch.diff"
        attempt = {
            "attempt_number": 1,
            "attempt_dir": "attempt-001",
            "session_id": f"sess-{cell_id}",
            "status": "completed" if is_completed else "pending",
            "patch_path": rel_patch,
        }
        meta = {
            **identity,
            "session_id": attempt["session_id"],
            "status": "completed" if is_completed else "pending",
            "classification": "completed" if is_completed else "pending",
            "attempts": [attempt],
            "patch_path": rel_patch,
            "patch_nonempty": is_completed,
            "patch_bytes": len(patch_path.read_bytes()),
            "official_pass": None,
            "official_grade_status": "not_run",
            "infrastructure_failure": False,
            "protocol_failure": False,
        }
        (cell_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        cells.append(meta)

    return fingerprint, cells, {p: rows for p, rows in instances_by_source.items()}


class GradePiDatasetAcTests(unittest.TestCase):
    def test_correctly_selects_source_rows_across_all_three_input_files(self):
        # 3 tasks: 1 from smoke, 1 from train50, 1 from gym_batch2
        task_specs = [
            ("getmoto__moto-5752", "smoke_instances.json", "edge"),
            ("conan-io__conan-14177", "train50_5tasks_instances.json", "cloud"),
            ("facebookresearch__hydra-1551", "gym_batch2_14tasks_instances.json", "edge"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            _, cells, instances = _setup_capture_fixture(
                root, tasks=task_specs, completed_iids={t[0] for t in task_specs}
            )

            calls = []

            def fake_run_instance(spec, pred, **kwargs):
                calls.append((spec, pred, kwargs))
                iid = pred["instance_id"]
                # Resolve true for first 2, false for 3rd
                resolved = iid != "facebookresearch__hydra-1551"
                return ({}, {iid: {"resolved": resolved, "FAIL_TO_PASS": 1, "PASS_TO_PASS": 1}})

            mock_api = SimpleNamespace(
                make_test_spec=lambda instance: f"spec:{instance['instance_id']}",
                run_instance=fake_run_instance,
                docker_from_env=lambda: object(),
            )

            with patch.object(grader, "_read_instances", side_effect=lambda p: instances[p]):
                result = grader.grade_all_completed(root, api=mock_api, docker_client=object())

            self.assertEqual(result["total_completed_cells_found"], 3)
            self.assertEqual(result["graded_cells"], 3)
            self.assertEqual(len(calls), 3)

            # Verify specs received correct task instance ids from respective files
            called_specs = [call[0] for call in calls]
            self.assertEqual(
                called_specs,
                [
                    "spec:conan-io__conan-14177",
                    "spec:facebookresearch__hydra-1551",
                    "spec:getmoto__moto-5752",
                ],
            )

            # Verify predictions carried correct patches
            for call in calls:
                pred = call[1]
                self.assertIn(f"patch diff for {pred['instance_id']}", pred["model_patch"])

            # Verify official passes were recorded in cell metadata
            meta_conan = json.loads((root / "conan-io__conan-14177__cloud" / "run_meta.json").read_text())
            self.assertEqual(meta_conan["official_grade_status"], "graded")
            self.assertIs(meta_conan["official_pass"], True)

            meta_hydra = json.loads((root / "facebookresearch__hydra-1551__edge" / "run_meta.json").read_text())
            self.assertEqual(meta_hydra["official_grade_status"], "graded")
            self.assertIs(meta_hydra["official_pass"], False)

            # Resuming does not re-grade
            with patch.object(grader, "_read_instances", side_effect=lambda p: instances[p]):
                resumed = grader.grade_all_completed(root, api=mock_api, docker_client=None)
            self.assertEqual(len(calls), 3)  # No new calls made
            self.assertEqual(resumed["already_graded_cells"], 3)

    def test_grades_only_completed_cells_skips_pending(self):
        task_specs = [
            ("getmoto__moto-5752", "smoke_instances.json", "edge"),
            ("getmoto__moto-6178", "smoke_instances.json", "cloud"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            # Only moto-5752 is completed, moto-6178 is pending
            _, _, instances = _setup_capture_fixture(
                root, tasks=task_specs, completed_iids={"getmoto__moto-5752"}
            )

            calls = []

            def fake_run_instance(spec, pred, **kwargs):
                calls.append((spec, pred, kwargs))
                return ({}, {pred["instance_id"]: {"resolved": True}})

            mock_api = SimpleNamespace(
                make_test_spec=lambda instance: f"spec:{instance['instance_id']}",
                run_instance=fake_run_instance,
                docker_from_env=lambda: object(),
            )

            with patch.object(grader, "_read_instances", side_effect=lambda p: instances[p]):
                result = grader.grade_all_completed(root, api=mock_api, docker_client=object())

            self.assertEqual(result["total_completed_cells_found"], 1)
            self.assertEqual(result["graded_cells"], 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["instance_id"], "getmoto__moto-5752")

            # Check that pending cell was recorded in pending_or_skipped_cells
            self.assertIn("getmoto__moto-6178__cloud", result["pending_or_skipped_cells"])

    def test_recovers_interrupted_grade_with_existing_report(self):
        task_specs = [("getmoto__moto-5752", "smoke_instances.json", "edge")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pi_dataset_ac_v1"
            fingerprint, cells, instances = _setup_capture_fixture(
                root, tasks=task_specs, completed_iids={"getmoto__moto-5752"}
            )
            cell_id = "getmoto__moto-5752__edge"
            attempt_dir = root / cell_id / "attempt-001"
            report_path = attempt_dir / "official_grade_abc123.json"
            checkpoint_path = attempt_dir / "official_grade_checkpoint.json"

            run_id = f"pi-ac-v1-grade-{cell_id}-a1-abc123456789"
            report_payload = {
                "protocol_version": runner.PROTOCOL_VERSION,
                "protocol_fingerprint": fingerprint,
                "cell_id": cell_id,
                "instance_id": "getmoto__moto-5752",
                "backend": "edge",
                "capture_attempt": 1,
                "grading_run_id": run_id,
                "status": "complete",
                "resolved": True,
                "report": {"resolved": True},
                "grade_seconds": 15.2,
                "patch_sha256": "fake-sha",
            }
            report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

            checkpoint_payload = {
                "protocol_version": runner.PROTOCOL_VERSION,
                "protocol_fingerprint": fingerprint,
                "cell_id": cell_id,
                "capture_attempt": 1,
                "status": "complete",
                "run_id": run_id,
                "report_path": f"{cell_id}/attempt-001/{report_path.name}",
            }
            checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")

            # api.run_instance should NOT be called because report is recovered
            calls = []
            mock_api = SimpleNamespace(
                make_test_spec=lambda inst: "spec",
                run_instance=lambda *a, **kw: calls.append(a),
                docker_from_env=lambda: object(),
            )

            with patch.object(grader, "_read_instances", side_effect=lambda p: instances[p]):
                result = grader.grade_all_completed(root, api=mock_api, docker_client=None)

            self.assertEqual(len(calls), 0)
            self.assertEqual(result["graded_cells"], 1)

            meta = json.loads((root / cell_id / "run_meta.json").read_text())
            self.assertEqual(meta["official_grade_status"], "graded")
            self.assertIs(meta["official_pass"], True)

    def test_root_shim_delegates_to_grader(self):
        import grade_pi_dataset_ac as root_grader

        self.assertTrue(hasattr(root_grader, "grade_all_completed"))
        self.assertTrue(hasattr(root_grader, "GradingProtocolError"))


if __name__ == "__main__":
    unittest.main()
