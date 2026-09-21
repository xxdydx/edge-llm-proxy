from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_harness_ab_pi_v1 as analysis
import monitor_harness_ab_pi_v1 as monitor


def trial_rows(
    *,
    claude_failures: set[tuple[str, str]] | None = None,
    pi_failures: set[tuple[str, str]] | None = None,
    claude_pass: bool = True,
    pi_pass: bool = True,
) -> list[dict]:
    claude_failures = claude_failures or set()
    pi_failures = pi_failures or set()
    rows = []
    for task in analysis.TASK_IDS:
        for backend in analysis.BACKENDS:
            for harness in analysis.HARNESSES:
                failure_set = claude_failures if harness == "claude-code" else pi_failures
                reason = "trajectory_deadline" if (task, backend) in failure_set else "completed"
                rows.append({
                    "instance_id": task,
                    "backend": backend,
                    "harness": harness,
                    "protocol_version": analysis.PROTOCOL_VERSION,
                    "protocol_fingerprint": "frozen-test-fingerprint",
                    "status": "completed",
                    "termination_reason": reason,
                    "official_pass": claude_pass if harness == "claude-code" else pi_pass,
                    "wall_seconds": 100 if harness == "claude-code" else 80,
                    "action_count": 5 if harness == "claude-code" else 4,
                    "model_turn_count": 6 if harness == "claude-code" else 5,
                    "output_tokens": 1000 if harness == "claude-code" else 700,
                })
    return rows


class HarnessAbPiAnalysisTests(unittest.TestCase):
    def test_operational_gate_recommends_pi_only_after_all_frozen_conditions(self):
        rows = trial_rows(claude_failures={(analysis.TASK_IDS[0], "edge"), (analysis.TASK_IDS[1], "cloud")})
        report = analysis.analyze(rows, capture_root=Path("/tmp"))
        self.assertEqual(report["recommendation"], "recommend_pi_migration")
        self.assertEqual(report["censored_failure_reduction_pi_vs_claude"], 2)
        self.assertEqual(report["paired_official_passes"]["graded_pairs"], 4)
        self.assertEqual(report["descriptive_by_harness"]["pi"]["actions"]["mean"], 4.0)

    def test_in_progress_cells_are_reported_without_premature_gate(self):
        rows = trial_rows(claude_failures={(analysis.TASK_IDS[0], "edge"), (analysis.TASK_IDS[1], "cloud")})
        rows.pop()
        report = analysis.analyze(rows, capture_root=Path("/tmp"))
        self.assertEqual(report["recommendation"], "inconclusive_incomplete_trial")
        self.assertEqual(report["terminal_cells"], 7)
        self.assertEqual(len(report["missing_cells"]), 1)

    def test_missing_grades_waits_even_when_failure_gate_passes(self):
        rows = trial_rows(claude_failures={(analysis.TASK_IDS[0], "edge"), (analysis.TASK_IDS[1], "cloud")})
        for row in rows:
            row.pop("official_pass")
        report = analysis.analyze(rows, capture_root=Path("/tmp"))
        self.assertEqual(report["recommendation"], "inconclusive_waiting_for_paired_official_grades")

    def test_pi_infrastructure_failure_blocks_migration(self):
        rows = trial_rows(claude_failures={(analysis.TASK_IDS[0], "edge"), (analysis.TASK_IDS[1], "cloud")})
        pi_row = next(row for row in rows if row["harness"] == "pi")
        pi_row["infrastructure_failure"] = True
        report = analysis.analyze(rows, capture_root=Path("/tmp"))
        self.assertEqual(report["recommendation"], "do_not_migrate_pi_infrastructure_or_protocol_failure")

    def test_process_and_protocol_errors_are_censored_and_blocking(self):
        rows = trial_rows()
        pi_rows = [row for row in rows if row["harness"] == "pi"]
        pi_rows[0]["termination_reason"] = "process_error"
        pi_rows[1]["termination_reason"] = "protocol_error"

        report = analysis.analyze(rows, capture_root=Path("/tmp"))

        self.assertEqual(report["descriptive_by_harness"]["pi"]["censored_harness_failures"], 2)
        self.assertEqual(report["pi_infrastructure_or_protocol_failures"], 2)
        self.assertEqual(report["recommendation"], "do_not_migrate_pi_infrastructure_or_protocol_failure")
        self.assertTrue(analysis.INFRA_PROTOCOL_FAILURES.issubset(analysis.CENSORED_FAILURES))

    def test_pi_official_pass_regression_blocks_migration(self):
        rows = trial_rows(claude_failures={(analysis.TASK_IDS[0], "edge"), (analysis.TASK_IDS[1], "cloud")}, pi_pass=False)
        report = analysis.analyze(rows, capture_root=Path("/tmp"))
        self.assertEqual(report["recommendation"], "do_not_migrate_pi_quality_regression")

    def test_grading_report_path_supports_swebench_resolved_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = root / "grading.json"
            report_path.write_text(json.dumps([{"resolved": True, "report": {"resolved": True}}]))
            row = {"grading_report_path": "grading.json"}
            self.assertIs(analysis.official_pass(row, root), True)


class HarnessAbPiMonitorTests(unittest.TestCase):
    def test_per_cell_checkpoint_overrides_stale_root_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            cell = analysis.expected_cells()[0]
            key = (cell["instance_id"], cell["backend"], cell["harness"])
            cell_dir = capture / "__".join(key)
            cell_dir.mkdir(parents=True)
            (capture / "capture_manifest.json").write_text(json.dumps({
                "cells": [{**cell, "status": "pending"}],
            }))
            (cell_dir / "run_meta.json").write_text(json.dumps({
                **cell,
                "status": "running",
                "termination_reason": None,
            }))

            rows = analysis.load_rows(capture)
            current = next(row for row in rows if analysis.cell_key(row) == key)

            self.assertEqual(current["status"], "running")
            self.assertEqual(sum(analysis.cell_key(row) == key for row in rows), 1)

    def test_snapshot_tracks_expected_cells_health_and_active_stream_freshness_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            cell = analysis.expected_cells()[0]
            key = (cell["instance_id"], cell["backend"], cell["harness"])
            attempt = capture / "__".join(key) / "attempt-001"
            attempt.mkdir(parents=True)
            stream = attempt / "claude_stream.jsonl"
            stream.write_text("{}\n")
            os.utime(stream, (990, 990))
            manifest = capture / "capture_manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps({"cells": [{**cell, "status": "running"}]}))
            before = manifest.read_bytes()

            state = monitor.build_snapshot(
                capture,
                now=1000,
                tmux_alive=lambda session: session == monitor.DRIVER_SESSION,
                edge_health=lambda: (True, {"status": "ok"}),
            )

            self.assertEqual(state["expected_cells"], 8)
            self.assertEqual(state["completed_cells"], 0)
            self.assertTrue(state["driver_tmux_alive"])
            self.assertFalse(state["relay_tmux_alive"])
            self.assertTrue(state["edge_endpoint_ok"])
            active = next(item for item in state["cells"] if item["instance_id"] == key[0] and item["harness"] == key[2])
            self.assertEqual(active["active_stream"]["age_seconds"], 10)
            self.assertTrue(active["active_stream"]["fresh"])
            self.assertEqual(manifest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
