"""Unit tests for experiments/new_datasets/monitor_pi_collection.py.

Tests:
- Loading and count of expected preflight tasks and cells (21 tasks x 2 backends = 42 cells).
- Cell inspection across lifecycle: pending, running, completed, censored.
- Censored failure classifications by reason (trajectory_deadline, repeated_action, adapter_error).
- Grading classification: distinguishing pending, censored, and ungraded from task FAIL.
- Calculation of official failure rate strictly among graded cells.
- Calculation of capture censored failure rates and breakdown by reason.
- Health checks for relay, driver tmux liveness, and edge endpoint.
- Concise hourly summary generation, structure, and atomic persistence.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
ND = REPO_ROOT / "experiments/new_datasets"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ND) not in sys.path:
    sys.path.insert(0, str(ND))

import monitor_pi_collection as monitor
from monitor_pi_collection import (
    build_hourly_summary,
    build_snapshot,
    calculate_metrics,
    expected_cells,
    format_concise_summary_message,
    inspect_cell,
    load_expected_tasks,
    run_monitor_loop,
    write_hourly_summary,
    write_snapshot_and_history,
)


class ExpectedCellsAndLoadingTests(unittest.TestCase):
    def test_load_expected_tasks_from_real_preflight_files(self):
        tasks = load_expected_tasks()
        # Preflight tasks: 2 smoke + 5 train50 + 14 gym_batch2 = 21 unique tasks
        self.assertEqual(len(tasks), 21)
        self.assertEqual(len(set(tasks)), 21)
        self.assertIn("getmoto__moto-5752", tasks)
        self.assertIn("conan-io__conan-13788", tasks)
        self.assertIn("iterative__dvc-5336", tasks)

    def test_expected_cells_count(self):
        cells = expected_cells()
        # 21 tasks * 2 backends (edge, cloud) = 42 cells
        self.assertEqual(len(cells), 42)
        edge_cells = [c for c in cells if c["backend"] == "edge"]
        cloud_cells = [c for c in cells if c["backend"] == "cloud"]
        self.assertEqual(len(edge_cells), 21)
        self.assertEqual(len(cloud_cells), 21)


class CellInspectionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.capture_root = Path(self.temp_dir.name)
        self.now = time.time()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cell_pending_when_dir_missing(self):
        res = inspect_cell(self.capture_root, "getmoto__moto-5752", "edge", self.now)
        self.assertEqual(res["capture_status"], "pending")
        self.assertEqual(res["grading_status"], "pending")
        self.assertIsNone(res["resolved"])
        self.assertIsNone(res["censored_reason"])

    def test_cell_running_with_fresh_stream(self):
        cell_dir = self.capture_root / "getmoto__moto-5752__edge"
        cell_dir.mkdir(parents=True, exist_ok=True)
        stream_file = cell_dir / "pi_stream.jsonl"
        stream_file.write_text('{"event": "start"}\n', encoding="utf-8")

        res = inspect_cell(self.capture_root, "getmoto__moto-5752", "edge", self.now)
        self.assertEqual(res["capture_status"], "running")
        self.assertEqual(res["grading_status"], "running")
        self.assertIsNotNone(res["active_stream"])
        self.assertTrue(res["active_stream"]["fresh"])

    def test_cell_completed_ungraded(self):
        cell_dir = self.capture_root / "getmoto__moto-5752__edge"
        cell_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "instance_id": "getmoto__moto-5752",
            "backend": "edge",
            "termination_reason": "completed",
            "status": "completed",
        }
        (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        res = inspect_cell(self.capture_root, "getmoto__moto-5752", "edge", self.now)
        self.assertEqual(res["capture_status"], "completed")
        self.assertEqual(res["grading_status"], "ungraded")
        self.assertIsNone(res["resolved"])

    def test_cell_censored_by_trajectory_deadline(self):
        cell_dir = self.capture_root / "conan-io__conan-13788__cloud"
        cell_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "instance_id": "conan-io__conan-13788",
            "backend": "cloud",
            "termination_reason": "trajectory_deadline",
            "timeout": True,
        }
        (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        res = inspect_cell(self.capture_root, "conan-io__conan-13788", "cloud", self.now)
        self.assertEqual(res["capture_status"], "censored")
        self.assertEqual(res["censored_reason"], "trajectory_deadline")
        self.assertEqual(res["grading_status"], "censored")
        self.assertIsNone(res["resolved"])

    def test_cell_censored_by_repeated_action(self):
        cell_dir = self.capture_root / "iterative__dvc-5336__edge"
        cell_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "instance_id": "iterative__dvc-5336",
            "backend": "edge",
            "termination_reason": "repeated_action",
            "identical_action_count": 4,
        }
        (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        res = inspect_cell(self.capture_root, "iterative__dvc-5336", "edge", self.now)
        self.assertEqual(res["capture_status"], "censored")
        self.assertEqual(res["censored_reason"], "repeated_action")
        self.assertEqual(res["grading_status"], "censored")

    def test_cell_graded_pass(self):
        cell_dir = self.capture_root / "getmoto__moto-6178__cloud"
        cell_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "instance_id": "getmoto__moto-6178",
            "backend": "cloud",
            "termination_reason": "completed",
        }
        (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        grade = {"resolved": True, "report": {"getmoto__moto-6178": {"resolved": True}}}
        (cell_dir / "grading_result.json").write_text(json.dumps(grade), encoding="utf-8")

        res = inspect_cell(self.capture_root, "getmoto__moto-6178", "cloud", self.now)
        self.assertEqual(res["capture_status"], "completed")
        self.assertEqual(res["grading_status"], "graded_pass")
        self.assertIs(res["resolved"], True)

    def test_cell_graded_fail(self):
        cell_dir = self.capture_root / "getmoto__moto-5752__cloud"
        cell_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "instance_id": "getmoto__moto-5752",
            "backend": "cloud",
            "termination_reason": "completed",
        }
        (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        grade = {"resolved": False, "report": {"getmoto__moto-5752": {"resolved": False}}}
        (cell_dir / "grading_result.json").write_text(json.dumps(grade), encoding="utf-8")

        res = inspect_cell(self.capture_root, "getmoto__moto-5752", "cloud", self.now)
        self.assertEqual(res["capture_status"], "completed")
        self.assertEqual(res["grading_status"], "graded_fail")
        self.assertIs(res["resolved"], False)


class FailureRateAndStrictClassificationTests(unittest.TestCase):
    """Test the core requirement:

    'distinguish pending/censored/ungraded from task FAIL'
    'capture censored failure rate by reason and, after grading, official failure rate among graded cells'
    """

    def test_strict_separation_and_failure_rates(self):
        # Create a synthetic distribution of cells:
        inspections = [
            # 5 Pending
            *([{"capture_status": "pending", "censored_reason": None, "grading_status": "pending"}] * 5),
            # 2 Running
            *([{"capture_status": "running", "censored_reason": None, "grading_status": "running"}] * 2),
            # 3 Censored: 2 trajectory_deadline, 1 repeated_action
            {"capture_status": "censored", "censored_reason": "trajectory_deadline", "grading_status": "censored"},
            {"capture_status": "censored", "censored_reason": "trajectory_deadline", "grading_status": "censored"},
            {"capture_status": "censored", "censored_reason": "repeated_action", "grading_status": "censored"},
            # 2 Ungraded completed runs
            *([{"capture_status": "completed", "censored_reason": None, "grading_status": "ungraded"}] * 2),
            # 3 Graded PASS
            *([{"capture_status": "completed", "censored_reason": None, "grading_status": "graded_pass"}] * 3),
            # 1 Graded FAIL (this is the ONLY task FAIL!)
            {"capture_status": "completed", "censored_reason": None, "grading_status": "graded_fail"},
        ]
        # Total = 5 + 2 + 3 + 2 + 3 + 1 = 16 cells

        metrics = calculate_metrics(inspections)
        counts = metrics["counts"]
        rates = metrics["failure_rates"]

        # Verify exact counts
        self.assertEqual(counts["pending"], 5)
        self.assertEqual(counts["running"], 2)
        self.assertEqual(counts["censored_capture"], 3)
        self.assertEqual(counts["completed_capture"], 6)  # 2 ungraded + 3 pass + 1 fail
        self.assertEqual(counts["attempted_capture"], 9)  # 6 completed + 3 censored
        self.assertEqual(counts["ungraded"], 2)
        self.assertEqual(counts["graded_pass"], 3)
        self.assertEqual(counts["graded_fail"], 1)
        self.assertEqual(counts["total_graded"], 4)

        # STRICT VERIFICATION:
        # Task FAIL must be exactly 1 (not 1 + 3 censored + 5 pending + 2 ungraded = 11)
        self.assertEqual(counts["graded_fail"], 1)

        # Official failure rate among graded cells: 1 fail / 4 total graded = 0.25 (25.0%)
        self.assertAlmostEqual(rates["official_failure_rate_among_graded"], 0.25)
        self.assertAlmostEqual(rates["official_pass_rate_among_graded"], 0.75)

        # Capture censored rate: 3 censored / 9 attempted = 0.3333 (33.33%)
        self.assertAlmostEqual(rates["capture_censored_failure_rate"], 3 / 9, places=4)

        # Censored breakdown by reason
        self.assertEqual(rates["capture_censored_by_reason"]["trajectory_deadline"], 2)
        self.assertEqual(rates["capture_censored_by_reason"]["repeated_action"], 1)
        self.assertAlmostEqual(rates["capture_censored_rates_by_reason"]["trajectory_deadline"], 2 / 9, places=4)
        self.assertAlmostEqual(rates["capture_censored_rates_by_reason"]["repeated_action"], 1 / 9, places=4)

    def test_official_failure_rate_is_none_when_no_cells_graded(self):
        inspections = [
            {"capture_status": "completed", "censored_reason": None, "grading_status": "ungraded"},
            {"capture_status": "censored", "censored_reason": "trajectory_deadline", "grading_status": "censored"},
            {"capture_status": "pending", "censored_reason": None, "grading_status": "pending"},
        ]
        metrics = calculate_metrics(inspections)
        self.assertEqual(metrics["counts"]["total_graded"], 0)
        self.assertIsNone(metrics["failure_rates"]["official_failure_rate_among_graded"])


class HealthAndLivenessTests(unittest.TestCase):
    def test_snapshot_includes_health_and_summary_text(self):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)

        def mock_tmux(session):
            return session == "test-relay"

        def mock_edge():
            return True, {"status": "ok", "latency_ms": 12}

        snap = build_snapshot(
            capture_root=root,
            task_ids=["getmoto__moto-5752"],
            tmux_checker=mock_tmux,
            edge_health_checker=mock_edge,
            driver_session="test-driver",
            relay_session="test-relay",
        )

        health = snap["health"]
        self.assertFalse(health["driver_tmux_alive"])
        self.assertTrue(health["relay_tmux_alive"])
        self.assertTrue(health["edge_health_ok"])

        summary = snap["summary"]
        self.assertIn("Pi permanent collection:", summary)
        self.assertIn("driver: DEAD", summary)
        self.assertIn("relay: ALIVE", summary)
        self.assertIn("edge: OK", summary)

        temp_dir.cleanup()


class HourlySummaryPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.capture_root = self.root / "capture"
        self.monitor_root = self.root / "monitor"
        self.capture_root.mkdir(parents=True, exist_ok=True)
        self.monitor_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hourly_summary_generation_and_write(self):
        snap = build_snapshot(
            capture_root=self.capture_root,
            task_ids=["getmoto__moto-5752"],
            tmux_checker=lambda s: True,
            edge_health_checker=lambda: (True, {"status": "ok"}),
        )
        hourly = build_hourly_summary(snap, "2026-09-21T18:00:00Z")

        self.assertEqual(hourly["hour_timestamp_utc"], "2026-09-21T18:00:00Z")
        self.assertIn("counts", hourly)
        self.assertIn("failure_rates", hourly)
        self.assertIn("health", hourly)
        self.assertIn("concise_message", hourly)

        write_hourly_summary(hourly, self.monitor_root)

        # Check atomic latest file
        latest_file = self.monitor_root / "hourly_summary.json"
        self.assertTrue(latest_file.is_file())
        latest_data = json.loads(latest_file.read_text(encoding="utf-8"))
        self.assertEqual(latest_data["hour_timestamp_utc"], "2026-09-21T18:00:00Z")

        # Check append-only log file
        log_file = self.monitor_root / "hourly_summaries.jsonl"
        self.assertTrue(log_file.is_file())
        lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)

    def test_monitor_once_mode(self):
        run_monitor_loop(
            capture_root=self.capture_root,
            monitor_root=self.monitor_root,
            poll_seconds=1,
            once=True,
            task_ids=["getmoto__moto-5752"],
        )

        self.assertTrue((self.monitor_root / "monitor.json").is_file())
        self.assertTrue((self.monitor_root / "monitor.jsonl").is_file())
        self.assertTrue((self.monitor_root / "hourly_summary.json").is_file())
        self.assertTrue((self.monitor_root / "hourly_summaries.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
