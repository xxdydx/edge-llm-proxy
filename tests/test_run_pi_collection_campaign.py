"""Unit tests for experiments/new_datasets/run_pi_collection_campaign.py.

Tests:
- Atomic checkpoint creation, phase transitions, and state serialization.
- Resumability across campaign phases.
- Strict isolation from legacy Claude data (protocols, harnesses, run_dirs, store roots).
- Strict rejection of turn-count / turn-limit fields anywhere (budgets, records, traces).
- 1800-second trajectory budget inheritance from W3 metadata.
- Validation of exactly 21 unique preflight tasks (42 cells) requiring terminal capture classifications.
- Strict rejection of missing trace joins (fail closed).
- Strict rejection of duplicate session IDs (fail closed).
- Strict rejection of missing or mismatched official grades (fail closed).
- Evidence-faithful ingestion of real traces and grade artifacts into JobStore.
- Remote trace synchronization command structure and credential scrubbing.
- Dataset B certification gate: blocking when certificate fails vs permitting when certified.
- Full campaign orchestrator flow.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
ND = REPO_ROOT / "experiments/new_datasets"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ND) not in sys.path:
    sys.path.insert(0, str(ND))

import run_pi_collection_campaign as campaign  # noqa: E402
from run_pi_collection_campaign import (  # noqa: E402
    CampaignConfig,
    CampaignIsolationError,
    CampaignPaths,
    CampaignPhase,
    DuplicateSessionError,
    GradeMismatchError,
    IncompleteCaptureError,
    MissingGradeError,
    PhaseStatus,
    PreflightTaskCountError,
    TraceJoinError,
    TurnBudgetForbiddenError,
    checkpoint_phase_transition,
    create_initial_state,
    evaluate_b_certification_gate,
    get_expected_cell_ids,
    get_expected_tasks,
    load_or_init_state,
    run_campaign,
    run_capture_ac_phase,
    run_certify_b_phase,
    run_collect_b_phase,
    run_grade_ac_phase,
    run_ingest_export_ac_phase,
    run_sync_traces_phase,
    scrub_secrets,
    sync_remote_traces,
    validate_grading_status,
    validate_terminal_capture_classifications,
    verify_no_turn_limit_field,
    verify_not_legacy_claude_data,
    verify_store_root_isolated,
)


class CampaignStateAndResumptionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.campaign_root = self.root / "campaign"
        self.campaign_root.mkdir(parents=True, exist_ok=True)
        self.paths = CampaignPaths(
            campaign_root=self.campaign_root,
            w3_capture_root=self.root / "w3_capture/pi_dataset_ac_v1",
            pi_store_root=self.campaign_root / "dataset_ac",
            w4_b_root=self.campaign_root / "dataset_b",
            checkpoint_file=self.campaign_root / "campaign_checkpoint.json",
            w3_capture_script=self.root / "run_pi_dataset_ac.py",
            w3_grade_script=self.root / "grade_pi_dataset_ac.py",
            w4_b_script=self.root / "run_pi_dataset_b.py",
            w4_cert_file=self.root / "checkpoint_certificate.json",
            campaign_trace_root=self.campaign_root / "traces",
        )
        self.config = CampaignConfig(
            campaign_id="test-pi-v1",
            protocol_version_ac="pi-dataset-ac-v1",
            protocol_version_b="pi-dataset-b-v1",
            cohort="test-pi",
            paths=self.paths,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_initial_state_and_checkpoint_atomic(self):
        state = load_or_init_state(self.config)
        self.assertEqual(state["campaign_id"], "test-pi-v1")
        self.assertEqual(state["current_phase"], CampaignPhase.INITIAL.value)
        self.assertEqual(state["current_status"], PhaseStatus.PENDING.value)
        self.assertTrue(self.paths.checkpoint_file.is_file())

        saved = json.loads(self.paths.checkpoint_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["current_phase"], CampaignPhase.INITIAL.value)

    def test_checkpoint_phase_transitions_atomic(self):
        state = load_or_init_state(self.config)

        # Transition to CAPTURE_AC
        checkpoint_phase_transition(
            self.config, state, CampaignPhase.CAPTURE_AC, PhaseStatus.RUNNING, {"step": 1}
        )
        self.assertEqual(state["current_phase"], CampaignPhase.CAPTURE_AC.value)
        self.assertEqual(state["current_status"], PhaseStatus.RUNNING.value)
        self.assertEqual(len(state["phase_history"]), 1)

        # Verify disk matches state
        disk_state = json.loads(self.paths.checkpoint_file.read_text(encoding="utf-8"))
        self.assertEqual(disk_state["current_phase"], CampaignPhase.CAPTURE_AC.value)
        self.assertEqual(disk_state["phase_history"][0]["from_phase"], CampaignPhase.INITIAL.value)
        self.assertEqual(disk_state["phase_history"][0]["to_phase"], CampaignPhase.CAPTURE_AC.value)

    def test_resumption_from_intermediate_phase(self):
        state = load_or_init_state(self.config)
        # Mark CAPTURE_AC as completed and transition to GRADE_AC
        checkpoint_phase_transition(
            self.config, state, CampaignPhase.CAPTURE_AC, PhaseStatus.COMPLETED
        )
        checkpoint_phase_transition(
            self.config, state, CampaignPhase.GRADE_AC, PhaseStatus.PENDING
        )

        # Load fresh state from checkpoint file
        resumed_state = load_or_init_state(self.config)
        self.assertEqual(resumed_state["current_phase"], CampaignPhase.GRADE_AC.value)

        # Execute campaign with mock components
        grader_called = []
        sync_called = []
        ingest_called = []

        def mock_grader(cmd):
            grader_called.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="Graded 42 cells\n")

        def mock_sync(cfg):
            sync_called.append(cfg)
            return {"synced": True}

        def mock_ingester(cfg):
            ingest_called.append(cfg)
            return {"total_a_ingested": 42, "total_c_ingested": 42, "exported_files": []}

        # Mock terminal capture validation for the 42 cells so grade phase permits execution
        expected_cells = get_expected_cell_ids()
        for cell_id in expected_cells:
            cell_dir = self.paths.w3_capture_root / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "cell_id": cell_id,
                "protocol_version": "pi-dataset-ac-v1",
                "status": "completed",
                "classification": "completed",
                "session_id": f"sid-{cell_id}",
                "official_grade_status": "graded",
                "official_pass": True,
                "max_active_seconds": 1800,
            }
            (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (cell_dir / "grading_result.json").write_text(
                json.dumps({"resolved": True, "report": {"tests": []}}), encoding="utf-8"
            )

        # Run up to INGEST_EXPORT_AC
        final_state = run_campaign(
            self.config,
            target_phase=CampaignPhase.INGEST_EXPORT_AC,
            grader_fn=mock_grader,
            sync_fn=mock_sync,
            ingest_fn=mock_ingester,
        )

        self.assertEqual(len(grader_called), 1)
        self.assertEqual(len(sync_called), 1)
        self.assertEqual(len(ingest_called), 1)
        self.assertEqual(final_state["current_phase"], CampaignPhase.INGEST_EXPORT_AC.value)
        self.assertEqual(final_state["current_status"], PhaseStatus.COMPLETED.value)


class CampaignLegacyIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_verify_not_legacy_claude_data_rejects_legacy_protocols(self):
        for proto in ["smoke-v1", "train50-batch1-v1", "gym-batch2-v1", "timeout-recovery-v2", "harness-ab-pi-v1"]:
            bad_record = {"protocol_version": proto, "harness": "pi"}
            with self.assertRaises(CampaignIsolationError) as ctx:
                verify_not_legacy_claude_data(bad_record, "test_record")
            self.assertIn("legacy Claude protocol", str(ctx.exception))

    def test_verify_not_legacy_claude_data_rejects_claude_harness(self):
        bad_record = {"protocol_version": "pi-dataset-ac-v1", "harness": "claude-code"}
        with self.assertRaises(CampaignIsolationError) as ctx:
            verify_not_legacy_claude_data(bad_record, "test_record")
        self.assertIn("legacy Claude", str(ctx.exception))

    def test_verify_not_legacy_claude_data_rejects_claude_run_dirs(self):
        bad_record = {
            "protocol_version": "pi-dataset-ac-v1",
            "run_dir": "conan-io__conan-13788__claude-code-v1",
        }
        with self.assertRaises(CampaignIsolationError) as ctx:
            verify_not_legacy_claude_data(bad_record, "test_record")
        self.assertIn("legacy Claude run_dir", str(ctx.exception))

    def test_verify_not_legacy_claude_data_accepts_valid_pi_record(self):
        valid_record = {
            "protocol_version": "pi-dataset-ac-v1",
            "harness": "pi",
            "run_dir": "conan-io__conan-13788__pi_edge",
        }
        # Should not raise
        verify_not_legacy_claude_data(valid_record, "valid_record")

    def test_verify_store_root_isolated_rejects_legacy_root(self):
        legacy_root = self.root / "experiments/new_datasets"
        legacy_root.mkdir(parents=True, exist_ok=True)

        with self.assertRaises(CampaignIsolationError):
            verify_store_root_isolated(legacy_root, legacy_root)

        with self.assertRaises(CampaignIsolationError):
            verify_store_root_isolated(legacy_root / "datasets", legacy_root)

    def test_verify_store_root_isolated_accepts_distinct_pi_root(self):
        legacy_root = self.root / "experiments/new_datasets"
        pi_root = self.root / "experiments/new_datasets/campaigns/pi_permanent_v1/dataset_ac"
        verify_store_root_isolated(pi_root, legacy_root)


class TurnLimitRejectionTests(unittest.TestCase):
    """Test strict rejection of turn-count / turn-limit fields anywhere."""

    def test_rejection_of_turn_limit_in_budget_dict(self):
        forbidden_examples = [
            {"max_main_logical_calls": 40},
            {"calls_used_so_far": 5},
            {"max_turns": 100},
            {"turn_limit": 50},
            {"turn_count_budget": 30},
            {"max_turn_count": 25},
            {"turns_remaining": 12},
            {"turn_budget": 10},
        ]
        for bad_budget in forbidden_examples:
            with self.assertRaises(TurnBudgetForbiddenError) as ctx:
                verify_no_turn_limit_field(bad_budget, "remaining_budget")
            self.assertIn("turn budget field", str(ctx.exception).lower())

    def test_rejection_of_turn_limit_nested_in_record(self):
        nested_record = {
            "task_id": "gym:astropy__astropy-14182",
            "trajectory_id": "gym:astropy__astropy-14182:edge:sid1",
            "remaining_budget_at_prefix": {
                "max_main_logical_calls": 40,
                "max_active_seconds": 1800,
            },
        }
        with self.assertRaises(TurnBudgetForbiddenError) as ctx:
            verify_no_turn_limit_field(nested_record, "nested_record")
        self.assertIn("max_main_logical_calls", str(ctx.exception))

    def test_acceptance_of_1800s_w3_metadata_budget_without_turn_limit(self):
        valid_budget = {
            "trajectory_budget_seconds": 1800.0,
            "max_active_seconds": 1800.0,
        }
        # Must not raise
        verify_no_turn_limit_field(valid_budget, "valid_budget")


class PreflightCellsAndTerminalClassificationTests(unittest.TestCase):
    """Test validation of 21 preflight tasks (42 cells) requiring terminal capture."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.w3_capture_root = self.root / "w3_capture/pi_dataset_ac_v1"
        self.w3_capture_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_expected_task_and_cell_counts(self):
        tasks = get_expected_tasks()
        self.assertEqual(len(tasks), 21)
        self.assertEqual(len(set(tasks)), 21)

        cell_ids = get_expected_cell_ids()
        self.assertEqual(len(cell_ids), 42)
        # Ensure all tasks appear in both edge and cloud cells
        for task in tasks:
            self.assertIn(f"{task}__edge", cell_ids)
            self.assertIn(f"{task}__cloud", cell_ids)

    def test_rejection_when_cell_count_incomplete(self):
        expected_cells = get_expected_cell_ids()
        # Stage only 41 cells (missing 1 cell)
        cells_to_stage = sorted(expected_cells)[:-1]
        for cell_id in cells_to_stage:
            cell_dir = self.w3_capture_root / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "cell_id": cell_id,
                "protocol_version": "pi-dataset-ac-v1",
                "status": "completed",
                "classification": "completed",
                "session_id": f"sid-{cell_id}",
            }
            (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        with self.assertRaises(IncompleteCaptureError) as ctx:
            validate_terminal_capture_classifications(self.w3_capture_root, expected_cells)
        self.assertIn("lack terminal classification", str(ctx.exception))

    def test_rejection_when_cell_has_pending_or_running_status(self):
        expected_cells = get_expected_cell_ids()
        for i, cell_id in enumerate(sorted(expected_cells)):
            cell_dir = self.w3_capture_root / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            # Make one cell running/pending
            if i == 0:
                meta = {
                    "cell_id": cell_id,
                    "protocol_version": "pi-dataset-ac-v1",
                    "status": "running",
                    "classification": "running",
                    "session_id": f"sid-{cell_id}",
                }
            else:
                meta = {
                    "cell_id": cell_id,
                    "protocol_version": "pi-dataset-ac-v1",
                    "status": "completed",
                    "classification": "completed",
                    "session_id": f"sid-{cell_id}",
                }
            (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        with self.assertRaises(IncompleteCaptureError) as ctx:
            validate_terminal_capture_classifications(self.w3_capture_root, expected_cells)
        self.assertIn("lack terminal classification", str(ctx.exception))

    def test_acceptance_when_all_42_cells_have_terminal_classifications(self):
        expected_cells = get_expected_cell_ids()
        terminal_classes = ["completed", "timeout", "repeat", "process"]
        for i, cell_id in enumerate(sorted(expected_cells)):
            cell_dir = self.w3_capture_root / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            cls = terminal_classes[i % len(terminal_classes)]
            meta = {
                "cell_id": cell_id,
                "protocol_version": "pi-dataset-ac-v1",
                "status": "completed" if cls == "completed" else "failed",
                "classification": cls,
                "session_id": f"sid-{cell_id}",
                "max_active_seconds": 1800,
            }
            (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        summary = validate_terminal_capture_classifications(self.w3_capture_root, expected_cells)
        self.assertEqual(summary["total_cells"], 42)
        self.assertEqual(summary["terminal_cells"], 42)


class TraceJoinAndGradeRejectionTests(unittest.TestCase):
    """Test strict rejection of missing trace joins, duplicate sessions, and missing grades."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.campaign_root = self.root / "campaign"
        self.w3_capture_root = self.root / "w3_capture/pi_dataset_ac_v1"
        self.campaign_trace_root = self.campaign_root / "traces"
        self.pi_store_root = self.campaign_root / "dataset_ac"

        self.w3_capture_root.mkdir(parents=True, exist_ok=True)
        self.campaign_trace_root.mkdir(parents=True, exist_ok=True)
        self.pi_store_root.mkdir(parents=True, exist_ok=True)

        self.paths = CampaignPaths(
            campaign_root=self.campaign_root,
            w3_capture_root=self.w3_capture_root,
            pi_store_root=self.pi_store_root,
            w4_b_root=self.campaign_root / "dataset_b",
            checkpoint_file=self.campaign_root / "campaign_checkpoint.json",
            w3_capture_script=self.root / "run_pi_dataset_ac.py",
            w3_grade_script=self.root / "grade_pi_dataset_ac.py",
            w4_b_script=self.root / "run_pi_dataset_b.py",
            w4_cert_file=self.root / "checkpoint_certificate.json",
            campaign_trace_root=self.campaign_trace_root,
        )
        self.config = CampaignConfig(
            campaign_id="test-join-v1",
            paths=self.paths,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rejection_of_missing_trace_joins(self):
        # Stage a completed cell with session_id sid-missing
        cell_dir = self.w3_capture_root / "astropy__astropy-14182__edge"
        cell_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "cell_id": "astropy__astropy-14182__edge",
            "instance_id": "astropy__astropy-14182",
            "backend": "edge",
            "protocol_version": "pi-dataset-ac-v1",
            "status": "completed",
            "classification": "completed",
            "session_id": "sid-missing",
            "official_grade_status": "graded",
            "official_pass": True,
            "max_active_seconds": 1800,
        }
        (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (cell_dir / "grading_result.json").write_text(
            json.dumps({"resolved": True, "report": {}}), encoding="utf-8"
        )

        # Stage trace file containing only unrelated session
        trace_file = self.campaign_trace_root / "edge" / "2026-09-21.jsonl"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": 1758450000,
            "path": "/v1/messages",
            "call": {"session_id": "sid-unrelated", "call_id": "call-1", "tokens": {}},
            "request": {"messages": [{"role": "user", "content": "hello"}]},
            "response": {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"},
            "status": 200,
        }
        trace_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

        # Ingestion must fail closed on missing trace join
        with self.assertRaises(TraceJoinError) as ctx:
            campaign._default_ingest_and_export(self.config)
        self.assertIn("Missing trace join", str(ctx.exception))
        self.assertIn("sid-missing", str(ctx.exception))

    def test_rejection_of_duplicate_sessions(self):
        # Stage two cells sharing the same session_id
        for cname in ["astropy__astropy-14182__edge", "astropy__astropy-14182__cloud"]:
            cell_dir = self.w3_capture_root / cname
            cell_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "cell_id": cname,
                "instance_id": "astropy__astropy-14182",
                "backend": "edge" if "edge" in cname else "cloud",
                "protocol_version": "pi-dataset-ac-v1",
                "status": "completed",
                "classification": "completed",
                "session_id": "sid-duplicate",  # Shared session_id
                "official_grade_status": "graded",
                "official_pass": True,
                "max_active_seconds": 1800,
            }
            (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (cell_dir / "grading_result.json").write_text(
                json.dumps({"resolved": True, "report": {}}), encoding="utf-8"
            )

        trace_file = self.campaign_trace_root / "2026-09-21.jsonl"
        rec = {
            "ts": 1758450000,
            "path": "/v1/messages",
            "call": {"session_id": "sid-duplicate", "call_id": "call-1", "tokens": {}},
            "request": {"messages": []},
            "response": {"content": []},
            "status": 200,
        }
        trace_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

        with self.assertRaises(DuplicateSessionError) as ctx:
            campaign._default_ingest_and_export(self.config)
        self.assertIn("Duplicate session_id", str(ctx.exception))
        self.assertIn("sid-duplicate", str(ctx.exception))

    def test_rejection_of_missing_grades(self):
        cell_dir = self.w3_capture_root / "astropy__astropy-14182__edge"
        cell_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "cell_id": "astropy__astropy-14182__edge",
            "instance_id": "astropy__astropy-14182",
            "backend": "edge",
            "protocol_version": "pi-dataset-ac-v1",
            "status": "completed",
            "classification": "completed",
            "session_id": "sid-nograde",
            "official_grade_status": "not_run",  # Not graded
            "official_pass": None,
            "max_active_seconds": 1800,
        }
        (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        trace_file = self.campaign_trace_root / "2026-09-21.jsonl"
        rec = {
            "ts": 1758450000,
            "path": "/v1/messages",
            "call": {"session_id": "sid-nograde", "call_id": "call-1", "tokens": {}},
            "request": {"messages": [{"role": "user", "content": "prompt"}]},
            "response": {"content": [{"type": "text", "text": "result"}]},
            "status": 200,
        }
        trace_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

        with self.assertRaises(MissingGradeError) as ctx:
            campaign._default_ingest_and_export(self.config)
        self.assertIn("missing official grade", str(ctx.exception).lower())

    def test_rejection_of_mismatched_grades(self):
        cell_dir = self.w3_capture_root / "astropy__astropy-14182__edge"
        cell_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "cell_id": "astropy__astropy-14182__edge",
            "instance_id": "astropy__astropy-14182",
            "backend": "edge",
            "protocol_version": "pi-dataset-ac-v1",
            "status": "completed",
            "classification": "completed",
            "session_id": "sid-mismatch",
            "official_grade_status": "graded",
            "official_pass": True,  # Meta claims PASS
            "max_active_seconds": 1800,
        }
        (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        # Grade report explicitly says FAIL (resolved=False)
        (cell_dir / "grading_result.json").write_text(
            json.dumps({"resolved": False, "report": {"tests": ["FAILED"]}}), encoding="utf-8"
        )

        trace_file = self.campaign_trace_root / "2026-09-21.jsonl"
        rec = {
            "ts": 1758450000,
            "path": "/v1/messages",
            "call": {"session_id": "sid-mismatch", "call_id": "call-1", "tokens": {}},
            "request": {"messages": []},
            "response": {"content": []},
            "status": 200,
        }
        trace_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

        with self.assertRaises(GradeMismatchError) as ctx:
            campaign._default_ingest_and_export(self.config)
        self.assertIn("Grade mismatch", str(ctx.exception))

    def test_evidence_faithful_ingest_joins_real_traces_and_grades(self):
        # Stage valid completed cell with real session
        cell_dir = self.w3_capture_root / "astropy__astropy-14182__edge"
        cell_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "cell_id": "astropy__astropy-14182__edge",
            "instance_id": "astropy__astropy-14182",
            "backend": "edge",
            "protocol_version": "pi-dataset-ac-v1",
            "status": "completed",
            "classification": "completed",
            "session_id": "sid-real-001",
            "official_grade_status": "graded",
            "official_pass": True,
            "max_active_seconds": 1800,
        }
        (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (cell_dir / "grading_result.json").write_text(
            json.dumps({"resolved": True, "report": {"passed": True}}), encoding="utf-8"
        )

        # Stage real trace file with matching session
        trace_file = self.campaign_trace_root / "edge" / "2026-09-21.jsonl"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": 1758451000.0,
            "path": "/v1/messages",
            "call": {
                "session_id": "sid-real-001",
                "call_id": "msg-call-123",
                "tokens": {"input_tokens": 120, "output_tokens": 45},
                "usage_integrity": "verified",
            },
            "request": {
                "messages": [{"role": "user", "content": "fix the bug"}],
                "system": "you are pi",
            },
            "response": {
                "content": [{"type": "text", "text": "fixed"}],
                "stop_reason": "end_turn",
            },
            "status": 200,
            "timing": {"duration_ms": 500.0},
            "usage": {"input_tokens": 120, "output_tokens": 45},
        }
        trace_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

        summary = campaign._default_ingest_and_export(self.config)
        self.assertEqual(summary["total_a_ingested"], 1)
        self.assertEqual(summary["total_c_ingested"], 1)
        self.assertTrue(len(summary["exported_files"]) > 0)

        # Validate exported Dataset A file
        a_file = self.pi_store_root / "datasets/prefix_outcomes.jsonl"
        self.assertTrue(a_file.is_file())
        a_row = json.loads(a_file.read_text(encoding="utf-8").strip())
        self.assertEqual(a_row["session_id"] if "session_id" in a_row else a_row["provenance"]["session_id"], "sid-real-001")
        self.assertTrue(a_row["label_valid"])
        self.assertTrue(a_row["resolved"])

        # Validate 1800-second trajectory budget from W3 metadata and NO turn limit field
        budget = a_row["remaining_budget_at_prefix"]
        self.assertEqual(budget.get("max_active_seconds"), 1800.0)
        self.assertEqual(budget.get("trajectory_budget_seconds"), 1800.0)
        verify_no_turn_limit_field(a_row, "exported A row")

        # Validate exported Dataset C file
        c_file = self.pi_store_root / "datasets/serving_calls.jsonl"
        self.assertTrue(c_file.is_file())
        c_row = json.loads(c_file.read_text(encoding="utf-8").strip())
        self.assertEqual(c_row["invocation_id"], "msg-call-123")
        self.assertEqual(c_row["status"], "200")
        self.assertEqual(c_row["measured_timings"].get("duration_ms"), 500.0)
        verify_no_turn_limit_field(c_row, "exported C row")


class TraceSyncTests(unittest.TestCase):
    """Test remote trace synchronization command construction and secret scrubbing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.campaign_root = self.root / "campaign"
        self.paths = CampaignPaths.from_root(self.campaign_root)
        self.config = CampaignConfig(paths=self.paths)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sync_remote_traces_command_structure(self):
        executed_cmds = []

        def mock_sync_runner(cmd):
            executed_cmds.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="synced")

        summary = sync_remote_traces(self.config, sync_runner=mock_sync_runner)
        self.assertEqual(len(executed_cmds), 2)
        cloud_cmd, edge_cmd = executed_cmds

        # Verify rsync and ssh port 31226 and gw@lum.id
        self.assertEqual(cloud_cmd[0], "rsync")
        self.assertIn("ssh -p 31226", " ".join(cloud_cmd))
        self.assertIn("gw@lum.id:/workspace/flowmesh/traces/pi_dataset_ac_v1/cloud/", cloud_cmd[-2])

        self.assertEqual(edge_cmd[0], "rsync")
        self.assertIn("ssh -p 31226", " ".join(edge_cmd))
        self.assertIn("gw@lum.id:/workspace/flowmesh/traces/pi_dataset_ac_v1/edge/", edge_cmd[-2])

        self.assertEqual(summary["local_trace_root"], str(self.paths.campaign_trace_root))

    def test_sync_remote_traces_scrubs_secrets(self):
        os.environ["LUMID_TOKEN"] = "secret-lumid-key-xyz"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-secret-12345"

        sample_err = "Error connecting to gw@lum.id with Bearer my-secret-jwt-token and sk-ant-test-secret-12345: secret-lumid-key-xyz"
        scrubbed = scrub_secrets(sample_err)
        self.assertNotIn("secret-lumid-key-xyz", scrubbed)
        self.assertNotIn("sk-ant-test-secret-12345", scrubbed)
        self.assertNotIn("my-secret-jwt-token", scrubbed)
        self.assertIn("[REDACTED]", scrubbed)


class DatasetBCertificationGateTests(unittest.TestCase):
    def test_certification_gate_blocked_on_status_blocked(self):
        cert = {
            "certificate_version": "w4-resume-cert-v1",
            "status": "BLOCKED",
            "blockers": [
                "Fresh container has no retained session state; direct resume failed.",
                "Original filesystem/session snapshot not retained.",
            ],
        }
        permitted, blockers = evaluate_b_certification_gate(cert)
        self.assertFalse(permitted)
        self.assertEqual(len(blockers), 2)
        self.assertIn("Fresh container", blockers[0])

    def test_certification_gate_blocked_on_missing_certificate(self):
        permitted, blockers = evaluate_b_certification_gate(None, cert_path=Path("/non/existent/path"))
        self.assertFalse(permitted)
        self.assertTrue(len(blockers) > 0)
        self.assertIn("No checkpoint certificate", blockers[0])

    def test_certification_gate_blocked_on_unknown_status_without_explicit_blockers(self):
        cert = {
            "certificate_version": "w4-resume-cert-v1",
            "status": "INCONCLUSIVE",
            "blockers": [],
        }
        permitted, blockers = evaluate_b_certification_gate(cert)
        self.assertFalse(permitted)
        self.assertTrue(len(blockers) > 0)
        self.assertIn("INCONCLUSIVE", blockers[0])

    def test_certification_gate_blocked_on_quarantine_flag(self):
        cert = {
            "certificate_version": "pi-resume-cert-v1",
            "status": "PASS",
            "quarantine": True,
            "blocker_reason": "pre-call tool observation mismatch",
        }
        permitted, blockers = evaluate_b_certification_gate(cert)
        self.assertFalse(permitted)
        self.assertTrue(any("quarantine" in b.lower() for b in blockers))
        self.assertTrue(any("pre-call tool" in b for b in blockers))

    def test_certification_gate_permitted_when_certified_and_clean(self):
        cert = {
            "certificate_version": "w4-resume-cert-v1",
            "status": "CERTIFIED",
            "blockers": [],
            "tool_state_checks": {"remaining_budget": True, "cwd": True},
        }
        permitted, blockers = evaluate_b_certification_gate(cert)
        self.assertTrue(permitted)
        self.assertEqual(len(blockers), 0)

    def test_certification_gate_permitted_with_w4_pi_checkpoint_certificate(self):
        cert = {
            "certificate_version": "pi-resume-cert-v1",
            "status": "PASS",
            "quarantine": False,
            "blocker_reason": None,
            "checks": {"filesystem_match": True, "request_match": True},
        }
        permitted, blockers = evaluate_b_certification_gate(cert)
        self.assertTrue(permitted)
        self.assertEqual(len(blockers), 0)


class FullCampaignOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.campaign_root = self.root / "campaign"
        self.paths = CampaignPaths(
            campaign_root=self.campaign_root,
            w3_capture_root=self.root / "w3_capture/pi_dataset_ac_v1",
            pi_store_root=self.campaign_root / "dataset_ac",
            w4_b_root=self.campaign_root / "dataset_b",
            checkpoint_file=self.campaign_root / "campaign_checkpoint.json",
            w3_capture_script=self.root / "run_pi_dataset_ac.py",
            w3_grade_script=self.root / "grade_pi_dataset_ac.py",
            w4_b_script=self.root / "run_pi_dataset_b.py",
            w4_cert_file=self.campaign_root / "checkpoint_certificate.json",
            campaign_trace_root=self.campaign_root / "traces",
        )
        self.config = CampaignConfig(
            campaign_id="test-full-pi-v1",
            paths=self.paths,
        )

        # Stage mock 42 terminal cells for capture/grade validation
        expected_cells = get_expected_cell_ids()
        for cell_id in expected_cells:
            cell_dir = self.paths.w3_capture_root / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "cell_id": cell_id,
                "protocol_version": "pi-dataset-ac-v1",
                "status": "completed",
                "classification": "completed",
                "session_id": f"sid-{cell_id}",
                "official_grade_status": "graded",
                "official_pass": True,
                "max_active_seconds": 1800,
            }
            (cell_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (cell_dir / "grading_result.json").write_text(
                json.dumps({"resolved": True, "report": {}}), encoding="utf-8"
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_campaign_halts_at_certification_gate_when_blocked(self):
        collector_invoked = []

        def mock_capture(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="Captured 42 cells")

        def mock_grader(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="Graded 42 cells")

        def mock_sync(cfg):
            return {"synced": True}

        def mock_ingester(cfg):
            return {"total_a_ingested": 42, "total_c_ingested": 42}

        def mock_certifier():
            return False, ["Pi container restore cannot certify executor state."]

        def mock_collector(cmd):
            collector_invoked.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="B collected")

        state = run_campaign(
            self.config,
            runner_fn=mock_capture,
            grader_fn=mock_grader,
            sync_fn=mock_sync,
            ingest_fn=mock_ingester,
            certifier_fn=mock_certifier,
            collector_fn=mock_collector,
        )

        # Collector must NOT have been called!
        self.assertEqual(len(collector_invoked), 0)
        self.assertEqual(state["current_phase"], CampaignPhase.B_CERTIFICATION_BLOCKED.value)
        self.assertEqual(state["current_status"], PhaseStatus.BLOCKED.value)

    def test_campaign_proceeds_to_complete_when_certification_permits(self):
        collector_invoked = []

        def mock_capture(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="Captured 42 cells")

        def mock_grader(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="Graded 42 cells")

        def mock_sync(cfg):
            return {"synced": True}

        def mock_ingester(cfg):
            return {"total_a_ingested": 42, "total_c_ingested": 42}

        def mock_certifier():
            return True, []

        def mock_collector(cmd):
            collector_invoked.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="B collected")

        state = run_campaign(
            self.config,
            runner_fn=mock_capture,
            grader_fn=mock_grader,
            sync_fn=mock_sync,
            ingest_fn=mock_ingester,
            certifier_fn=mock_certifier,
            collector_fn=mock_collector,
        )

        # Collector MUST have been called
        self.assertEqual(len(collector_invoked), 1)
        self.assertEqual(state["current_phase"], CampaignPhase.COMPLETE.value)
        self.assertEqual(state["current_status"], PhaseStatus.COMPLETED.value)


class PhaseDirectExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.campaign_root = self.root / "campaign"
        self.paths = CampaignPaths(
            campaign_root=self.campaign_root,
            w3_capture_root=self.root / "w3_capture/pi_dataset_ac_v1",
            pi_store_root=self.campaign_root / "dataset_ac",
            w4_b_root=self.campaign_root / "dataset_b",
            checkpoint_file=self.campaign_root / "campaign_checkpoint.json",
            w3_capture_script=self.root / "run_pi_dataset_ac.py",
            w3_grade_script=self.root / "grade_pi_dataset_ac.py",
            w4_b_script=self.root / "run_pi_dataset_b.py",
            w4_cert_file=self.campaign_root / "checkpoint_certificate.json",
            campaign_trace_root=self.campaign_root / "traces",
        )
        self.config = CampaignConfig(paths=self.paths)
        self.state = create_initial_state(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_initial_state_structure(self):
        init_state = create_initial_state(self.config)
        self.assertEqual(init_state["schema_version"], 1)
        self.assertEqual(init_state["campaign_id"], self.config.campaign_id)
        self.assertEqual(init_state["protocol_version_ac"], "pi-dataset-ac-v1")
        self.assertIn("sync_traces", init_state["checkpoints"])

    def test_run_capture_ac_phase_direct(self):
        mock_cells = {f"mock_task_{i}__{b}" for i in range(21) for b in ("edge", "cloud")}

        def mock_runner(cmd):
            for cid in mock_cells:
                cdir = self.paths.w3_capture_root / cid
                cdir.mkdir(parents=True, exist_ok=True)
                (cdir / "run_meta.json").write_text(
                    json.dumps({
                        "cell_id": cid,
                        "protocol_version": "pi-dataset-ac-v1",
                        "status": "completed",
                        "classification": "completed",
                        "session_id": f"s-{cid}",
                    }),
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="captured")

        summary = run_capture_ac_phase(
            self.config, self.state, runner_fn=mock_runner, expected_cell_ids=mock_cells
        )
        self.assertEqual(summary["total_cells"], 42)
        self.assertEqual(self.state["current_phase"], CampaignPhase.CAPTURE_AC.value)
        self.assertEqual(self.state["current_status"], PhaseStatus.COMPLETED.value)

    def test_run_grade_ac_phase_direct(self):
        mock_cells = {f"mock_task_{i}__{b}" for i in range(21) for b in ("edge", "cloud")}
        for cid in mock_cells:
            cdir = self.paths.w3_capture_root / cid
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / "run_meta.json").write_text(
                json.dumps({
                    "cell_id": cid,
                    "protocol_version": "pi-dataset-ac-v1",
                    "status": "completed",
                    "classification": "completed",
                    "session_id": f"s-{cid}",
                    "official_grade_status": "graded",
                    "official_pass": True,
                }),
                encoding="utf-8",
            )
            (cdir / "grading_result.json").write_text(
                json.dumps({"resolved": True, "report": {}}), encoding="utf-8"
            )

        def mock_grader(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="graded")

        summary = run_grade_ac_phase(
            self.config, self.state, grader_fn=mock_grader, expected_cell_ids=mock_cells
        )
        self.assertEqual(summary["completed_and_graded_count"], 42)
        self.assertEqual(self.state["current_phase"], CampaignPhase.GRADE_AC.value)
        self.assertEqual(self.state["current_status"], PhaseStatus.COMPLETED.value)

    def test_run_sync_traces_phase_direct(self):
        def mock_sync(cfg):
            return {"cloud_files_count": 2, "edge_files_count": 2}

        summary = run_sync_traces_phase(self.config, self.state, sync_fn=mock_sync)
        self.assertEqual(summary["cloud_files_count"], 2)
        self.assertEqual(self.state["current_phase"], CampaignPhase.SYNC_TRACES.value)
        self.assertEqual(self.state["current_status"], PhaseStatus.COMPLETED.value)

    def test_run_ingest_export_ac_phase_direct(self):
        def mock_ingest(cfg):
            return {"total_a_ingested": 10, "total_c_ingested": 10}

        summary = run_ingest_export_ac_phase(
            self.config, self.state, custom_ingester=mock_ingest
        )
        self.assertEqual(summary["total_a_ingested"], 10)
        self.assertEqual(self.state["current_phase"], CampaignPhase.INGEST_EXPORT_AC.value)
        self.assertEqual(self.state["current_status"], PhaseStatus.COMPLETED.value)

    def test_run_certify_b_phase_direct(self):
        permitted, blockers = run_certify_b_phase(
            self.config, self.state, certifier_fn=lambda: (True, [])
        )
        self.assertTrue(permitted)
        self.assertEqual(self.state["current_phase"], CampaignPhase.CERTIFY_B.value)
        self.assertEqual(self.state["current_status"], PhaseStatus.COMPLETED.value)

    def test_run_collect_b_phase_direct(self):
        def mock_collector(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="b done")

        summary = run_collect_b_phase(self.config, self.state, collector_fn=mock_collector)
        self.assertIsNotNone(summary)
        self.assertEqual(self.state["current_phase"], CampaignPhase.COMPLETE.value)
        self.assertEqual(self.state["current_status"], PhaseStatus.COMPLETED.value)

    def test_validate_grading_status_detects_missing_report(self):
        mock_cells = {"task1__edge"}
        cdir = self.paths.w3_capture_root / "task1__edge"
        cdir.mkdir(parents=True, exist_ok=True)
        # Marked graded, but report file is missing
        (cdir / "run_meta.json").write_text(
            json.dumps({
                "cell_id": "task1__edge",
                "status": "completed",
                "classification": "completed",
                "official_grade_status": "graded",
                "official_pass": True,
            }),
            encoding="utf-8",
        )
        with self.assertRaises(MissingGradeError):
            validate_grading_status(self.paths.w3_capture_root, expected_cell_ids=mock_cells)

    def test_preflight_task_count_error_on_truncated_tasks(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as tf:
            tf.write(json.dumps([{"instance_id": "only-one-task"}]))
            tf.flush()
            with self.assertRaises(PreflightTaskCountError):
                get_expected_tasks(source_files=(Path(tf.name),))


if __name__ == "__main__":
    unittest.main()
