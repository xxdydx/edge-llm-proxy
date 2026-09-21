from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_harness_ab_pi_v1 as runner


def _cell(harness: str = "claude-code") -> runner.PlannedCell:
    return runner.PlannedCell(
        instance={"instance_id": "conan-io__conan-13788"},
        source_instances_file=runner.GYM_BATCH2,
        backend="edge",
        harness=harness,
        image="sweb.eval.arm64.conan-io__conan-13788:latest",
        prompt="Fix the issue",
        execution_order={"seed": 7, "harnesses_in_order": ["claude-code", "pi"], "position": 1},
    )


class HarnessAbRunnerTests(unittest.TestCase):
    def test_plan_is_exactly_eight_cells_with_deterministic_paired_order(self):
        instances = [
            (runner.GYM_BATCH2, {"instance_id": "conan-io__conan-13788", "repo": "conan-io/conan", "problem_statement": "one"}),
            (runner.TRAIN50_BATCH1, {"instance_id": "iterative__dvc-5336", "repo": "iterative/dvc", "problem_statement": "two"}),
        ]
        api = SimpleNamespace(
            load_swebench_dataset=lambda _path: [],
            render_prompt=lambda row: f"prompt:{row['instance_id']}",
            make_test_spec=lambda _row: SimpleNamespace(arch="arm64"),
        )
        with (
            patch.object(runner, "_load_selected_instances", return_value=instances),
            patch.object(runner, "_capture_api", return_value=api),
            patch.object(runner, "_sha256_file", return_value="source-hash"),
        ):
            first_plan, protocol, fingerprint = runner.build_plan()
            second_plan, _, second_fingerprint = runner.build_plan()

        self.assertEqual(len(first_plan), 8)
        self.assertEqual(len({item.cell_id for item in first_plan}), 8)
        self.assertEqual(fingerprint, second_fingerprint)
        self.assertEqual([item.cell_id for item in first_plan], [item.cell_id for item in second_plan])
        self.assertEqual(protocol["protocol_version"], runner.PROTOCOL_VERSION)
        for instance_id in {cell.instance_id for cell in first_plan}:
            for backend in runner.BACKENDS:
                pair = [cell for cell in first_plan if cell.instance_id == instance_id and cell.backend == backend]
                self.assertEqual([cell.harness for cell in pair], pair[0].execution_order["harnesses_in_order"])
                self.assertEqual([cell.execution_order["position"] for cell in pair], [1, 2])

    def test_claude_stream_summary_counts_turns_actions_and_tokens(self):
        stdout = "\n".join([
            json.dumps({"type": "assistant", "message": {
                "usage": {"output_tokens": 12},
                "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}],
            }}),
            json.dumps({"type": "assistant", "message": {
                "usage": {"output_tokens": 9},
                "content": [{"type": "text", "text": "done"}],
            }}),
            "not json",
        ])
        self.assertEqual(runner._count_claude_events(stdout), (1, 2, 21))

    def test_pi_result_contract_requires_comparable_telemetry(self):
        good = SimpleNamespace(
            stdout="", returncode=0, termination_reason="completed", action_count=3,
            model_turn_count=4, repeated_action_count=1, repeated_action_signature="bash:pwd",
            output_tokens=None,
        )
        normalized = runner._normalized_result("pi", good)
        self.assertEqual(normalized["action_count"], 3)
        self.assertEqual(normalized["model_turn_count"], 4)
        self.assertIsNone(normalized["output_tokens"])
        with self.assertRaises(runner.HarnessProtocolError):
            runner._normalized_result("pi", SimpleNamespace(stdout="", returncode=0, termination_reason="completed"))

    def test_protocol_root_and_cells_refuse_mismatch_or_unowned_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            runner.ensure_output_protocol(root, {"v": 1}, "fingerprint-1")
            runner.ensure_output_protocol(root, {"v": 1}, "fingerprint-1")
            with self.assertRaises(runner.HarnessProtocolError):
                runner.ensure_output_protocol(root, {"v": 2}, "fingerprint-2")

            cell = _cell()
            meta_path = runner.ensure_cell_meta(root, cell, "fingerprint-1")
            original = meta_path.read_bytes()
            with self.assertRaises(runner.HarnessProtocolError):
                runner.ensure_cell_meta(root, cell, "different-fingerprint")
            self.assertEqual(meta_path.read_bytes(), original)

            pi_cell = _cell("pi")
            orphan = root / pi_cell.cell_id
            orphan.mkdir()
            (orphan / "partial.diff").write_text("preserve me")
            with self.assertRaises(runner.HarnessProtocolError):
                runner.ensure_cell_meta(root, pi_cell, "fingerprint-1")
            self.assertEqual((orphan / "partial.diff").read_text(), "preserve me")

    def test_resume_appends_attempt_and_preserves_interrupted_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            runner.ensure_output_protocol(root, {"v": 1}, "fingerprint")
            cell = _cell()
            meta_path = runner.ensure_cell_meta(root, cell, "fingerprint")
            cell_dir = meta_path.parent
            first_attempt = cell_dir / "attempt-001"
            first_attempt.mkdir()
            preserved = first_attempt / "claude_stream.jsonl"
            preserved.write_text("partial stream\n")
            meta = json.loads(meta_path.read_text())
            meta["status"] = "running"
            meta["termination_reason"] = "infrastructure_error"
            meta["action_count"] = 11
            meta["attempts"] = [{
                "attempt_number": 1,
                "attempt_dir": "attempt-001",
                "session_id": "old-session",
                "status": "running",
            }]
            meta_path.write_text(json.dumps(meta))

            resumed_meta, second_attempt = runner._new_attempt(cell, cell_dir, meta_path)

            self.assertEqual(second_attempt.name, "attempt-002")
            self.assertEqual(resumed_meta["attempts"][0]["status"], "interrupted")
            self.assertEqual(resumed_meta["session_id"], resumed_meta["attempts"][-1]["session_id"])
            self.assertIsNone(resumed_meta["termination_reason"])
            self.assertIsNone(resumed_meta["action_count"])
            self.assertEqual(preserved.read_text(), "partial stream\n")
            self.assertTrue((second_attempt / "prompt.txt").exists() is False)

    def test_capture_writes_comparable_metadata_and_skips_completed_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            cell = _cell()
            runner.ensure_output_protocol(root, {"v": 1}, "fingerprint")
            runner.ensure_cell_meta(root, cell, "fingerprint")
            stdout = json.dumps({"type": "assistant", "message": {
                "usage": {"output_tokens": 17},
                "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "pwd"}}],
            }}) + "\n"
            raw = SimpleNamespace(
                stdout=stdout, returncode=0, termination_reason="completed",
                repeated_action_count=1, repeated_action_signature="bash:pwd",
            )
            starts = []

            def fake_run(_name, _prompt, _url, _model, _session, _timeout, out_path):
                Path(out_path).write_text(stdout)
                return raw

            api = SimpleNamespace(
                start_container=lambda *args: starts.append(args),
                setup_container=lambda *_args: None,
                run_claude=fake_run,
                extract_patch=lambda *_args: "diff --git a/x b/x\n",
                sh=lambda *_args: None,
            )

            with patch.object(runner, "_capture_api", return_value=api):
                meta = runner.execute_cell(root, cell, "fingerprint")
                skipped = runner.execute_cell(root, cell, "fingerprint")

            self.assertEqual(len(starts), 1)
            self.assertEqual(meta["status"], "completed")
            self.assertEqual(meta["backend"], "edge")
            self.assertEqual(meta["harness"], "claude-code")
            self.assertEqual(meta["action_count"], 1)
            self.assertEqual(meta["model_turn_count"], 1)
            self.assertEqual(meta["output_tokens"], 17)
            self.assertEqual(meta["official_grade_status"], "not_run")
            self.assertIsNone(meta["official_pass"])
            self.assertEqual(skipped["session_id"], meta["session_id"])

    def test_promotion_gate_waits_for_all_eight_official_grades(self):
        complete = [{
            "status": "completed", "official_grade_status": "graded", "official_pass": True,
        } for _ in range(8)]
        self.assertTrue(runner.all_official_grades_recorded(complete))
        self.assertFalse(runner.all_official_grades_recorded(complete[:7]))
        incomplete = [*complete[:-1], {**complete[-1], "official_pass": None}]
        self.assertFalse(runner.all_official_grades_recorded(incomplete))

    def test_official_grade_is_recorded_once_and_manifest_waits_for_all_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            runner.ensure_output_protocol(root, {"v": 1}, "fingerprint")
            cell = _cell()
            cell_dir = root / cell.cell_id
            cell_dir.mkdir()
            meta = {
                **runner._cell_identity(cell, "fingerprint"),
                "status": "completed", "attempts": [{"status": "completed"}],
                "official_pass": None, "official_grade_status": "not_run",
                "infrastructure_failure": False, "protocol_failure": False,
            }
            (cell_dir / "run_meta.json").write_text(json.dumps(meta))
            runner.record_official_grade(root, cell.cell_id, True, "grade.json")
            stored = json.loads((cell_dir / "run_meta.json").read_text())
            self.assertIs(stored["official_pass"], True)
            self.assertEqual(stored["official_grade_status"], "graded")
            manifest = json.loads((root / "capture_manifest.json").read_text())
            self.assertEqual(manifest["official_grade_status"], "pending")
            manifest_cell = next(row for row in manifest["cells"] if row["cell_id"] == cell.cell_id)
            self.assertIs(manifest_cell["official_pass"], True)
            with self.assertRaises(runner.HarnessProtocolError):
                runner.record_official_grade(root, cell.cell_id, False, "replacement.json")


if __name__ == "__main__":
    unittest.main()
