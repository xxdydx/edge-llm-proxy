"""Unit and integration tests for Pi Dataset B collector.

Protocol: pi-dataset-b-v1
Source: pi-dataset-ac-v1

Asserts:
1. Production path invokes real Docker/Pi/edgeproxy/grader actions (with mocked subprocess boundaries).
2. Exact restoration failure causes fail-closed quarantine with explicit certification blockers.
3. Production CLI does not accept mock flags.
4. Algorithm R seed repeatability, uniform distribution, and pre-call eligibility.
5. Branch isolation, Turn 1 intervention, and frozen edge-only-v1 continuation.
6. Complete 5-outcome classification and secret scrubbing.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
ND = REPO_ROOT / "experiments" / "new_datasets"
W4 = ND / "w4_branch"
for p in (str(REPO_ROOT), str(ND), str(W4)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_pi_dataset_b as pb
from w1_storage.schemas import Branch, BranchOutcome, ValidationError
from w1_storage.store import JobStore


class TestPiReservoirSampling(unittest.TestCase):
    """Test seeded prospective Algorithm R reservoir sampling."""

    def test_seed_determinism(self):
        """Same task/trajectory seed yields identical sequence of selections."""
        seed1 = pb.compute_reservoir_seed("gym:conan-13788", "traj-001", base_seed=123)
        seed2 = pb.compute_reservoir_seed("gym:conan-13788", "traj-001", base_seed=123)
        self.assertEqual(seed1, seed2)

        sampler1 = pb.PiReservoirSampler(seed1)
        sampler2 = pb.PiReservoirSampler(seed2)

        boundaries = [
            pb.PiPreCallBoundary(
                boundary_index=i,
                call_index=i,
                turn_index=i,
                session_id="s1",
                request_body={"turn": i},
                request_sha256=f"hash{i}",
                cwd="/testbed",
                remaining_wall_seconds=1800 - i * 10,
            )
            for i in range(1, 11)
        ]

        wins1 = [sampler1.observe(b, True, {"all_pass": True}) for b in boundaries]
        wins2 = [sampler2.observe(b, True, {"all_pass": True}) for b in boundaries]
        self.assertEqual(wins1, wins2)
        self.assertEqual(sampler1.selected_boundary, sampler2.selected_boundary)
        self.assertIsNotNone(sampler1.selected_boundary)

    def test_uniform_selection_distribution(self):
        """Over many trials, each eligible boundary has ~1/N selection probability."""
        n_boundaries = 4
        n_trials = 4000
        counts = {i: 0 for i in range(1, n_boundaries + 1)}

        boundaries = [
            pb.PiPreCallBoundary(
                boundary_index=i,
                call_index=i,
                turn_index=i,
                session_id="s1",
                request_body={"turn": i},
                request_sha256=f"hash{i}",
                cwd="/testbed",
                remaining_wall_seconds=1800 - i * 10,
            )
            for i in range(1, n_boundaries + 1)
        ]

        for trial in range(n_trials):
            sampler = pb.PiReservoirSampler(seed=trial)
            for b in boundaries:
                sampler.observe(b, True, {})
            self.assertIsNotNone(sampler.selected_boundary)
            counts[sampler.selected_boundary.boundary_index] += 1

        for i in range(1, n_boundaries + 1):
            fraction = counts[i] / n_trials
            self.assertGreater(fraction, 0.18, f"Boundary {i} selected too rarely: {fraction}")
            self.assertLess(fraction, 0.32, f"Boundary {i} selected too frequently: {fraction}")

    def test_pre_call_eligibility_enforcement(self):
        """Only pre-call information determines eligibility; post-call queries raise error."""
        cand_ok = {"is_main_agent_boundary": True, "pending_tools_count": 0, "backends_supported": True}
        is_elig, reason, checks = pb.is_eligible_pre_call(cand_ok, remaining_budget_seconds=1200)
        self.assertTrue(is_elig)
        self.assertTrue(all(checks.values()))

        cand_pending = {"is_main_agent_boundary": True, "pending_tools_count": 2, "backends_supported": True}
        is_elig, reason, checks = pb.is_eligible_pre_call(cand_pending, remaining_budget_seconds=1200)
        self.assertFalse(is_elig)
        self.assertFalse(checks["no_pending_tool_actions"])

        is_elig, reason, checks = pb.is_eligible_pre_call(cand_ok, remaining_budget_seconds=0)
        self.assertFalse(is_elig)
        self.assertFalse(checks["nonzero_remaining_budget"])

        cand_aux = {"is_main_agent_boundary": False, "pending_tools_count": 0}
        is_elig, reason, checks = pb.is_eligible_pre_call(cand_aux, remaining_budget_seconds=1200)
        self.assertFalse(is_elig)
        self.assertFalse(checks["main_agent_boundary"])

        cand_leak = {"is_main_agent_boundary": True, "resolved": True}
        with self.assertRaises(pb.EligibilityViolationError):
            pb.is_eligible_pre_call(cand_leak, remaining_budget_seconds=1200)

        cand_grade_leak = {"is_main_agent_boundary": True, "official_pass": False}
        with self.assertRaises(pb.EligibilityViolationError):
            pb.is_eligible_pre_call(cand_grade_leak, remaining_budget_seconds=1200)

    def test_ineligible_boundaries_skipped_by_reservoir(self):
        """Ineligible boundaries are recorded as ineligible and do not advance count."""
        sampler = pb.PiReservoirSampler(seed=42)
        b1 = pb.PiPreCallBoundary(1, 1, 1, "s1", {}, "h1", "/testbed", 1800)
        b2 = pb.PiPreCallBoundary(2, 2, 2, "s1", {}, "h2", "/testbed", 1800)

        sampler.observe(b1, True, {"all_pass": True})
        self.assertEqual(sampler.n_eligible, 1)
        self.assertEqual(sampler.selected_boundary, b1)

        sampler.observe(b2, False, {"pending_tools": False})
        self.assertEqual(sampler.n_eligible, 1)
        self.assertEqual(sampler.selected_boundary, b1)
        self.assertEqual(len(sampler.eligibility_log), 2)
        self.assertFalse(sampler.eligibility_log[1]["eligible"])


class TestCertificationAndFailClosed(unittest.TestCase):
    """Test checkpoint certification, precise blockers, and fail-closed quarantine."""

    def setUp(self):
        self.boundary = pb.PiPreCallBoundary(
            boundary_index=1,
            call_index=1,
            turn_index=1,
            session_id="session-001",
            request_body={"body": {"messages": [{"role": "user", "content": "fix"}]}, "headers": {}},
            request_sha256="req_hash_1",
            cwd="/testbed",
            remaining_wall_seconds=1800,
        )

    def test_certification_passes_on_exact_match(self):
        """Exact restore of session, filesystem, and request passes certification."""
        cp = pb.PiCheckpoint(
            checkpoint_id="cp-pass",
            task_id="gym:conan-13788",
            source_trajectory_id="traj-001",
            source_policy="edge-only-v1",
            split="train",
            selected_boundary=self.boundary,
            filesystem_state={"files_sha256": {"file1.py": "hash1"}},
            pi_session_state={"session_id": "session-001", "session_file_content": "{}"},
            remaining_budget={"max_active_seconds": 1800},
            provenance={},
        )
        probe = {
            "session_restorable": True,
            "files_sha256": {"file1.py": "hash1"},
            "next_request_body": {"body": {"messages": [{"role": "user", "content": "fix"}]}, "headers": {}},
            "available_tools": ["read", "write", "edit", "bash"],
        }
        cert = pb.certify_pi_checkpoint(cp, probe)
        self.assertEqual(cert.status, "PASS")
        self.assertFalse(cert.quarantine)
        self.assertIsNone(cert.blocker_reason)
        self.assertEqual(len(cert.diagnostic_flags), 0)

    def test_certification_quarantines_ephemeral_session(self):
        """If source trajectory ran with --no-session and no session file exists, quarantine."""
        cp = pb.PiCheckpoint(
            checkpoint_id="cp-no-session",
            task_id="gym:conan-13788",
            source_trajectory_id="traj-001",
            source_policy="edge-only-v1",
            split="train",
            selected_boundary=self.boundary,
            filesystem_state={"files_sha256": {"file1.py": "hash1"}},
            pi_session_state={"no_session_flag": True, "session_file_content": None},
            remaining_budget={"max_active_seconds": 1800},
            provenance={},
        )
        probe = {"files_sha256": {"file1.py": "hash1"}}
        cert = pb.certify_pi_checkpoint(cp, probe)

        self.assertEqual(cert.status, "QUARANTINED")
        self.assertTrue(cert.quarantine)
        self.assertIn("CERTIFICATION_BLOCKER_EPHEMERAL_SESSION", cert.blocker_reason)
        self.assertIn("no_session_ephemeral_state", cert.diagnostic_flags)

    def test_certification_quarantines_filesystem_drift(self):
        """Filesystem drift between checkpoint and restore triggers quarantine."""
        cp = pb.PiCheckpoint(
            checkpoint_id="cp-fs-drift",
            task_id="gym:conan-13788",
            source_trajectory_id="traj-001",
            source_policy="edge-only-v1",
            split="train",
            selected_boundary=self.boundary,
            filesystem_state={"files_sha256": {"file1.py": "hash1", "file2.py": "hash2"}},
            pi_session_state={"session_id": "session-001", "session_file_content": "{}"},
            remaining_budget={"max_active_seconds": 1800},
            provenance={},
        )
        probe = {
            "session_restorable": True,
            "files_sha256": {"file1.py": "altered_hash"},
            "next_request_body": self.boundary.request_body,
            "available_tools": ["read", "write", "edit", "bash"],
        }
        cert = pb.certify_pi_checkpoint(cp, probe)

        self.assertEqual(cert.status, "QUARANTINED")
        self.assertTrue(cert.quarantine)
        self.assertIn("CERTIFICATION_BLOCKER_FILESYSTEM_DRIFT", cert.blocker_reason)
        self.assertIn("filesystem_drift", cert.diagnostic_flags)

    def test_certification_quarantines_request_mismatch(self):
        """Next model request mismatch triggers quarantine."""
        cp = pb.PiCheckpoint(
            checkpoint_id="cp-req-mismatch",
            task_id="gym:conan-13788",
            source_trajectory_id="traj-001",
            source_policy="edge-only-v1",
            split="train",
            selected_boundary=self.boundary,
            filesystem_state={"files_sha256": {"file1.py": "hash1"}},
            pi_session_state={"session_id": "session-001", "session_file_content": "{}"},
            remaining_budget={"max_active_seconds": 1800},
            provenance={},
        )
        probe = {
            "session_restorable": True,
            "files_sha256": {"file1.py": "hash1"},
            "next_request_body": {"body": {"messages": [{"role": "user", "content": "DIFFERENT CONTENT"}]}},
            "available_tools": ["read", "write", "edit", "bash"],
        }
        cert = pb.certify_pi_checkpoint(cp, probe)

        self.assertEqual(cert.status, "QUARANTINED")
        self.assertTrue(cert.quarantine)
        self.assertIn("CERTIFICATION_BLOCKER_REQUEST_MISMATCH", cert.blocker_reason)
        self.assertIn("next_request_mismatch", cert.diagnostic_flags)

    def test_certification_quarantines_missing_tools(self):
        """Missing executor tools triggers quarantine."""
        cp = pb.PiCheckpoint(
            checkpoint_id="cp-tools",
            task_id="gym:conan-13788",
            source_trajectory_id="traj-001",
            source_policy="edge-only-v1",
            split="train",
            selected_boundary=self.boundary,
            filesystem_state={"files_sha256": {"file1.py": "hash1"}},
            pi_session_state={"session_id": "session-001", "session_file_content": "{}"},
            remaining_budget={"max_active_seconds": 1800},
            provenance={},
        )
        probe = {
            "session_restorable": True,
            "files_sha256": {"file1.py": "hash1"},
            "next_request_body": self.boundary.request_body,
            "available_tools": ["read", "bash"],  # edit and write are missing!
        }
        cert = pb.certify_pi_checkpoint(cp, probe)

        self.assertEqual(cert.status, "QUARANTINED")
        self.assertTrue(cert.quarantine)
        self.assertIn("CERTIFICATION_BLOCKER_TOOLS_MISSING", cert.blocker_reason)
        self.assertIn("missing_tools", cert.diagnostic_flags)

    def test_certification_quarantines_credential_leak(self):
        """Credentials detected in checkpoint or request triggers quarantine."""
        cp = pb.PiCheckpoint(
            checkpoint_id="cp-leak",
            task_id="gym:conan-13788",
            source_trajectory_id="traj-001",
            source_policy="edge-only-v1",
            split="train",
            selected_boundary=self.boundary,
            filesystem_state={"files_sha256": {"file1.py": "hash1"}},
            pi_session_state={"session_id": "session-001", "session_file_content": "{}", "ANTHROPIC_AUTH_TOKEN": "secret"},
            remaining_budget={"max_active_seconds": 1800},
            provenance={},
        )
        probe = {
            "session_restorable": True,
            "files_sha256": {"file1.py": "hash1"},
            "next_request_body": self.boundary.request_body,
            "available_tools": ["read", "write", "edit", "bash"],
        }
        cert = pb.certify_pi_checkpoint(cp, probe)

        self.assertEqual(cert.status, "QUARANTINED")
        self.assertTrue(cert.quarantine)
        self.assertIn("CERTIFICATION_BLOCKER_CREDENTIAL_CONTAMINATION", cert.blocker_reason)


class TestBranchIsolationAndPolicies(unittest.TestCase):
    """Test branch isolation, turn-1 intervention, frozen edge continuation, and scrubbing."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.b1 = pb.PiPreCallBoundary(1, 1, 1, "s1", {"prompt": "solve"}, "req_sha", "/testbed", 1800)
        self.cp = pb.PiCheckpoint(
            checkpoint_id="cp-test-123456",
            task_id="gym:conan-13788",
            source_trajectory_id="traj-001",
            source_policy="edge-only-v1",
            split="train",
            selected_boundary=self.b1,
            filesystem_state={"git_commit": "abc", "files_sha256": {}},
            pi_session_state={"session_id": "s1", "image_tag": "test-img"},
            remaining_budget={"max_active_seconds": 1800, "remaining_active_seconds": 1800, "turn_cap": None},
            provenance={"source": "test"},
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("run_pi_dataset_b.grade_branch")
    @patch("run_pi_dataset_b.run_pi_container_process")
    @patch("run_pi_dataset_b.exec_in_container")
    @patch("run_pi_dataset_b.setup_pi_container")
    @patch("run_pi_dataset_b.start_container")
    @patch("run_pi_dataset_b.stop_container")
    def test_branch_workspaces_are_strictly_isolated(
        self, mock_stop, mock_start, mock_setup, mock_exec, mock_run_pi, mock_grade
    ):
        """Edge and cloud branches run in completely disjoint workspaces."""
        edge_ws = self.temp_dir / "edge_workspace"
        cloud_ws = self.temp_dir / "cloud_workspace"

        mock_proc = MagicMock()
        mock_proc.stdout = ["{}\n"]
        mock_proc.stderr = None
        mock_proc.wait.return_value = 0
        mock_run_pi.return_value = mock_proc

        mock_exec_diff = MagicMock()
        mock_exec_diff.stdout = "diff --git a/f b/f\n"
        mock_exec.return_value = mock_exec_diff

        mock_grade.return_value = ({"conan-13788": {"resolved": True}}, Path("/tmp/report.json"))

        edge_res = pb.execute_real_branch(self.cp, "edge", edge_ws)
        cloud_res = pb.execute_real_branch(self.cp, "cloud", cloud_ws)

        self.assertNotEqual(edge_res.workspace_dir, cloud_res.workspace_dir)
        self.assertNotEqual(edge_res.container_name, cloud_res.container_name)
        self.assertTrue((edge_ws / "branch_edge.marker").exists())
        self.assertFalse((edge_ws / "branch_cloud.marker").exists())
        self.assertTrue((cloud_ws / "branch_cloud.marker").exists())
        self.assertFalse((cloud_ws / "branch_edge.marker").exists())

    @patch("run_pi_dataset_b.grade_branch")
    @patch("run_pi_dataset_b.run_pi_container_process")
    @patch("run_pi_dataset_b.exec_in_container")
    @patch("run_pi_dataset_b.setup_pi_container")
    @patch("run_pi_dataset_b.start_container")
    @patch("run_pi_dataset_b.stop_container")
    def test_turn1_intervention_and_frozen_continuation(
        self, mock_stop, mock_start, mock_setup, mock_exec, mock_run_pi, mock_grade
    ):
        """Turn 1 differs by backend (edge vs cloud), turn 2+ uses frozen edge-only-v1."""
        mock_proc = MagicMock()
        mock_proc.stdout = ["{}\n"]
        mock_proc.stderr = None
        mock_proc.wait.return_value = 0
        mock_run_pi.return_value = mock_proc

        mock_exec.return_value = MagicMock(stdout="diff")
        mock_grade.return_value = ({"conan-13788": {"resolved": True}}, Path("/tmp/report.json"))

        edge_res = pb.execute_real_branch(self.cp, "edge", self.temp_dir / "e")
        cloud_res = pb.execute_real_branch(self.cp, "cloud", self.temp_dir / "c")

        self.assertEqual(edge_res.initial_backend, "local")
        self.assertEqual(cloud_res.initial_backend, "deepseek-v4-flash")
        self.assertNotEqual(edge_res.initial_backend, cloud_res.initial_backend)

        self.assertEqual(edge_res.continuation_policy, "edge-only-v1")
        self.assertEqual(cloud_res.continuation_policy, "edge-only-v1")

    @patch("run_pi_dataset_b.grade_branch")
    @patch("run_pi_dataset_b.run_pi_container_process")
    @patch("run_pi_dataset_b.exec_in_container")
    @patch("run_pi_dataset_b.setup_pi_container")
    @patch("run_pi_dataset_b.start_container")
    @patch("run_pi_dataset_b.stop_container")
    def test_budget_deduction_and_no_turn_cap(
        self, mock_stop, mock_start, mock_setup, mock_exec, mock_run_pi, mock_grade
    ):
        """Remaining active wall budget is properly deducted from 1800s; no turn cap."""
        mock_proc = MagicMock()
        mock_proc.stdout = ["{}\n"]
        mock_proc.stderr = None
        mock_proc.wait.return_value = 0
        mock_run_pi.return_value = mock_proc
        mock_exec.return_value = MagicMock(stdout="diff")
        mock_grade.return_value = ({"conan-13788": {"resolved": True}}, Path("/tmp/report.json"))

        edge_res = pb.execute_real_branch(self.cp, "edge", self.temp_dir / "e")
        self.assertGreaterEqual(edge_res.wall_seconds_spent, 0.0)
        self.assertLessEqual(edge_res.remaining_wall_seconds, 1800.0)
        self.assertIsNone(self.cp.remaining_budget["turn_cap"])

    def test_credential_scrubbing(self):
        """All secrets and tokens are scrubbed from artifacts and JSON exports."""
        tainted = {
            "ANTHROPIC_AUTH_TOKEN": "sk-ant-secret12345",
            "FLOWMESH_PI_API_KEY": "secret_key_abc",
            "headers": {"authorization": "Bearer token_secret_999", "content-type": "application/json"},
            "nested": [{"user_password": "super_secret_pw", "data": "clean"}],
        }
        scrubbed = pb.scrub_credentials(tainted)
        self.assertEqual(scrubbed["ANTHROPIC_AUTH_TOKEN"], "[REDACTED]")
        self.assertEqual(scrubbed["FLOWMESH_PI_API_KEY"], "[REDACTED]")
        self.assertEqual(scrubbed["headers"]["authorization"], "[REDACTED]")
        self.assertEqual(scrubbed["nested"][0]["user_password"], "[REDACTED]")
        self.assertEqual(scrubbed["nested"][0]["data"], "clean")


class TestPairClassification(unittest.TestCase):
    """Test classification into BOTH_PASS, EDGE_ONLY_PASS, CLOUD_ONLY_PASS, BOTH_FAIL, and INCOMPLETE."""

    def _make_branch(self, label: str, resolved: bool | None, label_valid: bool) -> pb.BranchResult:
        return pb.BranchResult(
            branch_label=label,  # type: ignore[arg-type]
            branch_id=f"pi-b-{label}-test",
            workspace_dir=Path(f"/tmp/{label}"),
            container_name=f"container-{label}",
            initial_backend="local" if label == "edge" else "deepseek-v4-flash",
            continuation_policy="edge-only-v1",
            final_patch="diff",
            final_patch_sha256="hash",
            returncode=0 if label_valid else 1,
            termination_reason="completed" if label_valid else "process_error",
            wall_seconds_spent=30.0,
            remaining_wall_seconds=1770.0,
            resolved=resolved,
            label_valid=label_valid,
            report={},
        )

    def test_both_pass(self):
        e = self._make_branch("edge", resolved=True, label_valid=True)
        c = self._make_branch("cloud", resolved=True, label_valid=True)
        valid, pair_class = pb.classify_pair_outcome(e, c)
        self.assertTrue(valid)
        self.assertEqual(pair_class, "BOTH_PASS")

    def test_edge_only_pass(self):
        e = self._make_branch("edge", resolved=True, label_valid=True)
        c = self._make_branch("cloud", resolved=False, label_valid=True)
        valid, pair_class = pb.classify_pair_outcome(e, c)
        self.assertTrue(valid)
        self.assertEqual(pair_class, "EDGE_ONLY_PASS")

    def test_cloud_only_pass(self):
        e = self._make_branch("edge", resolved=False, label_valid=True)
        c = self._make_branch("cloud", resolved=True, label_valid=True)
        valid, pair_class = pb.classify_pair_outcome(e, c)
        self.assertTrue(valid)
        self.assertEqual(pair_class, "CLOUD_ONLY_PASS")

    def test_both_fail(self):
        e = self._make_branch("edge", resolved=False, label_valid=True)
        c = self._make_branch("cloud", resolved=False, label_valid=True)
        valid, pair_class = pb.classify_pair_outcome(e, c)
        self.assertTrue(valid)
        self.assertEqual(pair_class, "BOTH_FAIL")

    def test_incomplete_when_branch_invalid(self):
        e_bad = self._make_branch("edge", resolved=None, label_valid=False)
        c_ok = self._make_branch("cloud", resolved=True, label_valid=True)
        valid, pair_class = pb.classify_pair_outcome(e_bad, c_ok)
        self.assertFalse(valid)
        self.assertEqual(pair_class, "INCOMPLETE")

        e_ok = self._make_branch("edge", resolved=True, label_valid=True)
        c_bad = self._make_branch("cloud", resolved=None, label_valid=False)
        valid, pair_class = pb.classify_pair_outcome(e_ok, c_bad)
        self.assertFalse(valid)
        self.assertEqual(pair_class, "INCOMPLETE")


class TestResumeAndAtomicOperations(unittest.TestCase):
    """Test atomic write-and-replace, resume handling, and protocol enforcement."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_atomic_write_json(self):
        """atomic_write_json replaces file atomically without leaving .tmp files."""
        target = self.temp_dir / "test_atomic.json"
        data = {"count": 42, "status": "ok"}
        pb.atomic_write_json(target, data)

        self.assertTrue(target.exists())
        loaded = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(loaded, data)

        tmp_files = list(self.temp_dir.glob(".*.tmp"))
        self.assertEqual(len(tmp_files), 0)

    def test_protocol_mismatch_rejection(self):
        """Consuming non-pi-dataset-ac-v1 trajectories raises ProtocolMismatchError."""
        bad_dir = self.temp_dir / "bad_cell"
        bad_dir.mkdir()

        meta_wrong_proto = {
            "protocol_version": "harness-ab-pi-v1",
            "harness": "pi",
            "cell_id": "c1",
        }
        (bad_dir / "run_meta.json").write_text(json.dumps(meta_wrong_proto))
        with self.assertRaises(pb.ProtocolMismatchError):
            pb.load_pi_dataset_ac_trajectory(bad_dir)

        meta_wrong_harness = {
            "protocol_version": "pi-dataset-ac-v1",
            "harness": "claude-code",
            "cell_id": "c2",
        }
        (bad_dir / "run_meta.json").write_text(json.dumps(meta_wrong_harness))
        with self.assertRaises(pb.ProtocolMismatchError):
            pb.load_pi_dataset_ac_trajectory(bad_dir)

    def test_valid_pi_dataset_ac_trajectory_loaded(self):
        """pi-dataset-ac-v1 trajectory with harness 'pi' is successfully loaded."""
        traj_dir = self.temp_dir / "good_cell"
        traj_dir.mkdir()
        meta = {
            "protocol_version": "pi-dataset-ac-v1",
            "harness": "pi",
            "cell_id": "good-cell-001",
            "backend": "edge",
        }
        (traj_dir / "run_meta.json").write_text(json.dumps(meta))
        events = [
            {"type": "session", "version": 3, "id": "sess-xyz", "cwd": "/testbed"},
            {"type": "turn_start"},
            {"type": "message_start", "message": {"role": "user", "content": "solve bug"}},
            {"type": "message_end", "message": {"role": "user"}},
            {"type": "message_start", "message": {"role": "assistant", "model": "local"}},
            {"type": "message_end", "message": {"role": "assistant"}},
        ]
        stream_path = traj_dir / "pi_stream.jsonl"
        with stream_path.open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        loaded = pb.load_pi_dataset_ac_trajectory(traj_dir)
        self.assertEqual(loaded["meta"]["protocol_version"], "pi-dataset-ac-v1")
        self.assertEqual(loaded["meta"]["harness"], "pi")
        self.assertEqual(len(loaded["events"]), 6)

        boundaries = pb.extract_pi_pre_call_boundaries(loaded["events"])
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].session_id, "sess-xyz")
        self.assertEqual(boundaries[0].call_index, 1)


class TestProductionCollectorExecutionAndFailClosed(unittest.TestCase):
    """Test full production collector execution path, assert Docker/Pi/Grader boundaries, and fail-closed quarantine."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.traj_dir = self.temp_dir / "traj_ac"
        self.traj_dir.mkdir()
        meta = {
            "protocol_version": "pi-dataset-ac-v1",
            "harness": "pi",
            "cell_id": "conan-13788__edge__pi",
            "backend": "edge",
        }
        (self.traj_dir / "run_meta.json").write_text(json.dumps(meta))
        events = [
            {"type": "session", "version": 3, "id": "s-conan", "cwd": "/testbed"},
            {"type": "turn_start"},
            {"type": "message_start", "message": {"role": "user", "content": "fix conan"}},
            {"type": "message_end", "message": {"role": "user"}},
            {"type": "message_start", "message": {"role": "assistant", "model": "local"}},
            {"type": "message_end", "message": {"role": "assistant"}},
        ]
        with (self.traj_dir / "pi_stream.jsonl").open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("run_pi_dataset_b.execute_real_branch")
    @patch("run_pi_dataset_b.restore_and_certify_pi_checkpoint")
    def test_uncertified_flow_fails_closed_and_quarantines(
        self, mock_restore_cert, mock_execute_branch
    ):
        """When restore is unsupported or uncertified, collector fails closed, quarantines, and executes NO branches."""
        out_dir = self.temp_dir / "out_uncertified"

        # Mock restoration failing certification (e.g. ephemeral session or restore unsupported)
        mock_restore_cert.return_value = pb.CheckpointCertificate(
            certificate_version="pi-resume-cert-v1",
            status="QUARANTINED",
            quarantine=True,
            blocker_reason="CERTIFICATION_BLOCKER_EPHEMERAL_SESSION: Pi ran with --no-session and cannot restore exact session",
            checks={"session_restorable": False},
            diagnostic_flags=["no_session_ephemeral_state"],
        )

        result = pb.run_prospective_pi_dataset_b_collector(
            task_id="gym:conan-io__conan-13788",
            output_dir=out_dir,
            source_trajectory_dir=self.traj_dir,
        )

        self.assertEqual(result["status"], "QUARANTINED")
        self.assertFalse(result["pair_valid"])
        self.assertEqual(result["observed_pair_class"], "INCOMPLETE")
        self.assertIn("CERTIFICATION_BLOCKER_EPHEMERAL_SESSION", result["blocker_reason"])
        self.assertTrue(Path(result["certificate_path"]).exists())

        # Assert production path did NOT execute branches
        mock_execute_branch.assert_not_called()
        self.assertFalse((out_dir / "branch_outcome.json").exists())

    @patch("run_pi_dataset_b.execute_real_branch")
    @patch("run_pi_dataset_b.restore_and_certify_pi_checkpoint")
    def test_certified_flow_executes_real_branches_and_produces_dataset_b(
        self, mock_restore_cert, mock_execute_branch
    ):
        """When restore is certified, collector forks two isolated branches, grades them, and writes Dataset B."""
        out_dir = self.temp_dir / "out_certified"

        mock_restore_cert.return_value = pb.CheckpointCertificate(
            certificate_version="pi-resume-cert-v1",
            status="PASS",
            quarantine=False,
            blocker_reason=None,
            checks={"session_restorable": True, "filesystem_match": True, "next_request_match": True, "tools_available": True, "credentials_clean": True},
        )

        def make_branch_res(label: str) -> pb.BranchResult:
            return pb.BranchResult(
                branch_label=label,  # type: ignore[arg-type]
                branch_id=f"pi-b-{label}-test",
                workspace_dir=out_dir / "branches" / label,
                container_name=f"container-{label}",
                initial_backend="local" if label == "edge" else "deepseek-v4-flash",
                continuation_policy="edge-only-v1",
                final_patch="diff --git a/fix b/fix\n",
                final_patch_sha256="abc123hash",
                returncode=0,
                termination_reason="completed",
                wall_seconds_spent=35.0,
                remaining_wall_seconds=1765.0,
                resolved=(label == "edge"),  # edge passes, cloud fails -> EDGE_ONLY_PASS
                label_valid=True,
                report={"conan-io__conan-13788": {"resolved": (label == "edge")}},
                invocation_id=f"pi-inv-{label}-test",
            )

        mock_execute_branch.side_effect = [make_branch_res("edge"), make_branch_res("cloud")]

        result = pb.run_prospective_pi_dataset_b_collector(
            task_id="gym:conan-io__conan-13788",
            output_dir=out_dir,
            source_trajectory_dir=self.traj_dir,
        )

        self.assertEqual(result["status"], "COMPLETED")
        self.assertTrue(result["pair_valid"])
        self.assertEqual(result["observed_pair_class"], "EDGE_ONLY_PASS")
        self.assertTrue(Path(result["certificate_path"]).exists())
        self.assertTrue(Path(result["outcome_path"]).exists())

        self.assertEqual(mock_execute_branch.call_count, 2)
        edge_call, cloud_call = mock_execute_branch.call_args_list
        self.assertEqual(edge_call[0][1], "edge")
        self.assertEqual(cloud_call[0][1], "cloud")

        outcome = json.loads(Path(result["outcome_path"]).read_text())
        self.assertEqual(outcome["protocol_version"], "pi-dataset-b-v1")
        self.assertEqual(outcome["intervention"], "next_main_call_only")
        self.assertEqual(outcome["continuation_policy"], "edge-only-v1")
        self.assertEqual(outcome["observed_pair_class"], "EDGE_ONLY_PASS")
        self.assertTrue(outcome["pair_valid"])

    @patch("run_pi_dataset_b.run_pi_container_process")
    @patch("run_pi_dataset_b.exec_in_container")
    @patch("run_pi_dataset_b.setup_pi_container")
    @patch("run_pi_dataset_b.start_container")
    @patch("run_pi_dataset_b.stop_container")
    def test_execute_real_branch_invokes_docker_and_pi(
        self, mock_stop, mock_start, mock_setup, mock_exec, mock_run_pi
    ):
        """execute_real_branch invokes Docker container creation, Pi setup, and Pi process."""
        ws = self.temp_dir / "test_exec_branch_ws"
        cp = pb.PiCheckpoint(
            checkpoint_id="cp-real-001",
            task_id="gym:conan-io__conan-13788",
            source_trajectory_id="traj-001",
            source_policy="edge-only-v1",
            split="train",
            selected_boundary=pb.PiPreCallBoundary(1, 1, 1, "s1", {"prompt": "p"}, "h", "/testbed", 1800),
            filesystem_state={"git_commit": "HEAD", "files_sha256": {}},
            pi_session_state={"session_id": "s1", "image_tag": "sweb.eval.arm64.conan-io__conan-13788:latest"},
            remaining_budget={"remaining_active_seconds": 1800},
            provenance={},
        )

        mock_proc = MagicMock()
        mock_proc.stdout = ["{\"type\":\"turn_start\"}\n"]
        mock_proc.stderr = None
        mock_proc.wait.return_value = 0
        mock_run_pi.return_value = mock_proc

        mock_exec.return_value = MagicMock(stdout="diff content\n")

        with patch("run_pi_dataset_b.grade_branch") as mock_grade:
            mock_grade.return_value = ({"conan-io__conan-13788": {"resolved": True}}, Path("/tmp/r.json"))
            res = pb.execute_real_branch(cp, "edge", ws)

        mock_start.assert_called_once()
        mock_setup.assert_called_once()
        mock_run_pi.assert_called_once()
        mock_stop.assert_called_once()

        self.assertEqual(res.branch_label, "edge")
        self.assertEqual(res.initial_backend, "local")
        self.assertEqual(res.continuation_policy, "edge-only-v1")
        self.assertEqual(res.final_patch, "diff content\n")
        self.assertTrue(res.label_valid)


class TestProductionCLIRejectsMockFlags(unittest.TestCase):
    """Test that production CLI rejects mock flags."""

    def test_cli_rejects_mock_flag(self):
        """Passing --mock or --mock-run to production CLI raises SystemExit."""
        with patch.object(sys, "argv", ["run_pi_dataset_b.py", "--mock"]):
            with self.assertRaises(SystemExit) as cm:
                pb.main()
            self.assertEqual(cm.exception.code, 2)

        with patch.object(sys, "argv", ["run_pi_dataset_b.py", "--mock-run"]):
            with self.assertRaises(SystemExit) as cm:
                pb.main()
            self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
