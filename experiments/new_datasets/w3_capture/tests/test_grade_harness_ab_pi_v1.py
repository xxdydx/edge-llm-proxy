from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grade_harness_ab_pi_v1 as grader
import run_harness_ab_pi_v1 as runner


def _fixture(root: Path) -> tuple[str, list[dict], dict[Path, dict[str, dict]]]:
    tasks = []
    instances: dict[Path, dict[str, dict]] = {}
    for source_file, instance_id in runner.TARGETS:
        task_row = {
            "instance_id": instance_id,
            "repo": "example/repo",
            "problem_statement": f"problem for {instance_id}",
            "base_commit": "0123456789abcdef",
            "FAIL_TO_PASS": ["tests/test_issue.py::test_issue"],
            "PASS_TO_PASS": [],
        }
        instances[source_file] = {instance_id: task_row}
        tasks.append({
            "instance_id": instance_id,
            "source_instances_file": source_file.resolve().relative_to(runner.ND.resolve()).as_posix(),
            "source_sha256": grader._sha256_file(source_file),
            "image": f"sweb.eval.arm64.{instance_id}:latest",
            "prompt_sha256": "prompt-hash",
        })
    protocol = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "tasks": tasks,
        "backends": runner.BACKENDS,
        "harnesses": list(runner.HARNESSES),
        "max_active_seconds": runner.MAX_ACTIVE_SECONDS,
        "identical_action_repeat_limit": runner.MAX_IDENTICAL_ACTION_REPEATS,
        "container_platform": "linux/arm64/v8",
        "official_grading": "separate_post_capture_phase",
    }
    fingerprint = runner._sha256_bytes(runner._canonical_json(protocol))
    protocol_doc = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "protocol": protocol,
        "created_at_utc": "2026-09-21T00:00:00+00:00",
    }
    (root / "protocol.json").write_text(json.dumps(protocol_doc))

    cells = []
    for source_file, instance_id in runner.TARGETS:
        for backend in runner.BACKENDS:
            for harness in runner.HARNESSES:
                cell_id = f"{instance_id}__{backend}__{harness}"
                cell_dir = root / cell_id
                attempt_dir = cell_dir / "attempt-001"
                attempt_dir.mkdir(parents=True)
                patch_path = attempt_dir / "final_patch.diff"
                patch_path.write_text(f"patch for {cell_id}\n")
                prompt = f"prompt:{instance_id}"
                cell = runner.PlannedCell(
                    instance=instances[source_file][instance_id],
                    source_instances_file=source_file,
                    backend=backend,
                    harness=harness,
                    image=f"sweb.eval.arm64.{instance_id}:latest",
                    prompt=prompt,
                    execution_order={"seed": 1, "harnesses_in_order": list(runner.HARNESSES), "position": 1},
                )
                identity = runner._cell_identity(cell, fingerprint)
                relative_patch = f"{cell_id}/attempt-001/final_patch.diff"
                attempt = {
                    "attempt_number": 1,
                    "attempt_dir": "attempt-001",
                    "session_id": f"session-{cell_id}",
                    "status": "completed",
                    "patch_path": relative_patch,
                }
                meta = {
                    **identity,
                    "session_id": attempt["session_id"],
                    "status": "completed",
                    "attempts": [attempt],
                    "patch_path": relative_patch,
                    "patch_nonempty": True,
                    "patch_bytes": patch_path.stat().st_size,
                    "official_pass": None,
                    "official_grade_status": "not_run",
                    "infrastructure_failure": False,
                    "protocol_failure": False,
                }
                (cell_dir / "run_meta.json").write_text(json.dumps(meta))
                cells.append(meta)
    return fingerprint, cells, instances


class GradeHarnessAbPiTests(unittest.TestCase):
    def test_grades_all_eight_cells_once_with_unique_run_ids_and_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir()
            _, cells, instances = _fixture(root)
            calls = []

            def fake_run_instance(spec, prediction, **kwargs):
                calls.append((spec, prediction, kwargs))
                return ({}, {spec: {"resolved": True, "FAIL_TO_PASS": 1, "PASS_TO_PASS": 1}})

            api = SimpleNamespace(
                make_test_spec=lambda instance: instance["instance_id"],
                run_instance=fake_run_instance,
                docker_from_env=lambda: object(),
            )
            with patch.object(grader, "_read_instances", side_effect=lambda path: instances[path]):
                result = grader.grade_all(root, api=api, docker_client=object())

            self.assertEqual(result["graded_cells"], 8)
            self.assertEqual(len(calls), 8)
            run_ids = [call[2]["run_id"] for call in calls]
            self.assertEqual(len(run_ids), len(set(run_ids)))
            self.assertTrue(all(run_id.startswith(grader.RUN_ID_PREFIX) for run_id in run_ids))
            for cell in cells:
                cell_id = cell["cell_id"]
                meta = json.loads((root / cell_id / "run_meta.json").read_text())
                self.assertIs(meta["official_pass"], True)
                self.assertEqual(meta["official_grade_status"], "graded")
                attempt_dir = root / cell_id / "attempt-001"
                reports = [
                    path for path in attempt_dir.glob("official_grade_*.json")
                    if path.name != "official_grade_checkpoint.json"
                ]
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text())
                self.assertEqual(report["status"], "complete")
                self.assertTrue(report["resolved"])
                checkpoint = json.loads((attempt_dir / "official_grade_checkpoint.json").read_text())
                self.assertEqual(checkpoint["status"], "complete")

            with patch.object(grader, "_read_instances", side_effect=lambda path: instances[path]):
                second = grader.grade_all(root, api=api, docker_client=None)
            self.assertEqual(len(calls), 8)
            self.assertTrue(all(item["status"] == "already_graded" for item in second["cells"]))

    def test_uses_latest_capture_attempt_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir()
            _, cells, instances = _fixture(root)
            meta = cells[0]
            cell_dir = root / meta["cell_id"]
            latest_dir = cell_dir / "attempt-002"
            latest_dir.mkdir()
            latest_patch = latest_dir / "final_patch.diff"
            latest_patch.write_text("latest attempt patch\n")
            old_patch = cell_dir / "attempt-001" / "final_patch.diff"
            relative_patch = f"{meta['cell_id']}/attempt-002/final_patch.diff"
            meta["attempts"].append({
                "attempt_number": 2,
                "attempt_dir": "attempt-002",
                "session_id": "latest-session",
                "status": "completed",
                "patch_path": relative_patch,
            })
            meta["session_id"] = "latest-session"
            meta["patch_path"] = relative_patch
            (cell_dir / "run_meta.json").write_text(json.dumps(meta))
            seen_patches = []

            def fake_run_instance(spec, prediction, **kwargs):
                seen_patches.append(prediction["model_patch"])
                return ({}, {spec: {"resolved": False}})

            api = SimpleNamespace(
                make_test_spec=lambda instance: instance["instance_id"],
                run_instance=fake_run_instance,
                docker_from_env=lambda: object(),
            )
            with patch.object(grader, "_read_instances", side_effect=lambda path: instances[path]):
                grader.grade_all(root, api=api, docker_client=object())
            self.assertTrue(any(patch.startswith("latest attempt patch") for patch in seen_patches))
            self.assertNotIn(old_patch.read_text(), seen_patches)

    def test_rejects_noncompleted_cells_before_grading_anything(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir()
            _, cells, _ = _fixture(root)
            bad = root / cells[0]["cell_id"] / "run_meta.json"
            meta = json.loads(bad.read_text())
            meta["status"] = "running"
            bad.write_text(json.dumps(meta))
            calls = []
            api = SimpleNamespace(run_instance=lambda *args, **kwargs: calls.append(args))
            with self.assertRaises(grader.GradingProtocolError):
                grader.grade_all(root, api=api, docker_client=object())
            self.assertEqual(calls, [])

    def test_rejects_protocol_fingerprint_mismatch_before_grading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir()
            _, cells, _ = _fixture(root)
            bad = root / cells[0]["cell_id"] / "run_meta.json"
            meta = json.loads(bad.read_text())
            meta["protocol_fingerprint"] = "wrong"
            bad.write_text(json.dumps(meta))
            api = SimpleNamespace(run_instance=lambda *args, **kwargs: self.fail("must not grade"))
            with self.assertRaises(grader.GradingProtocolError):
                grader.grade_all(root, api=api, docker_client=object())

    def test_recovers_saved_report_without_repeating_official_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir()
            fingerprint, cells, instances = _fixture(root)
            meta = cells[0]
            cell_id = meta["cell_id"]
            attempt_dir = root / cell_id / "attempt-001"
            run_id = "existing-unique-grade-run"
            report_path = attempt_dir / "official_grade_recovered.json"
            report_path.write_text(json.dumps({
                "protocol_version": runner.PROTOCOL_VERSION,
                "protocol_fingerprint": fingerprint,
                "cell_id": cell_id,
                "grading_run_id": run_id,
                "status": "complete",
                "resolved": True,
                "grade_seconds": 23.0,
                "patch_sha256": "patch-hash",
            }))
            (attempt_dir / "official_grade_checkpoint.json").write_text(json.dumps({
                "protocol_version": runner.PROTOCOL_VERSION,
                "protocol_fingerprint": fingerprint,
                "cell_id": cell_id,
                "status": "running",
                "run_id": run_id,
                "report_path": f"{cell_id}/attempt-001/official_grade_recovered.json",
            }))
            (root / cell_id / "run_meta.json").write_text(json.dumps(meta))
            calls = []
            api = SimpleNamespace(
                make_test_spec=lambda instance: instance["instance_id"],
                run_instance=lambda *args, **kwargs: calls.append(kwargs),
                docker_from_env=lambda: object(),
            )
            with patch.object(grader, "_read_instances", side_effect=lambda path: instances[path]):
                result = grader.grade_all(root, api=api, docker_client=object())
            self.assertEqual(len(calls), 7)
            self.assertNotIn(run_id, [call["run_id"] for call in calls])
            recovered = next(item for item in result["cells"] if item["cell_id"] == cell_id)
            self.assertEqual(recovered["status"], "recovered_grade")
            saved = json.loads((root / cell_id / "run_meta.json").read_text())
            self.assertIs(saved["official_pass"], True)

    def test_failed_grader_call_is_checkpointed_and_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir()
            _, _, instances = _fixture(root)
            calls = []

            def fail_once(spec, prediction, **kwargs):
                calls.append(kwargs["run_id"])
                if len(calls) == 1:
                    raise RuntimeError("mock grader failure")
                return ({}, {spec: {"resolved": True}})

            api = SimpleNamespace(
                make_test_spec=lambda instance: instance["instance_id"],
                run_instance=fail_once,
                docker_from_env=lambda: object(),
            )
            with patch.object(grader, "_read_instances", side_effect=lambda path: instances[path]):
                first = grader.grade_all(root, api=api, docker_client=object())
                call_count = len(calls)
                second = grader.grade_all(root, api=api, docker_client=object())
            self.assertEqual(call_count, 8)
            self.assertEqual(len(calls), 8)
            self.assertEqual(first["cells"][0]["status"], "failed")
            self.assertEqual(second["cells"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
