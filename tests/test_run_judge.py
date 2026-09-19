"""Tests for experiments/agentic_router/run_judge.py: batch discovery and
snapshot integrity audit, on synthetic fixtures -- no network, no real data.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.agentic_router import judge, run_judge


class DiscoverReplayBatchesTests(unittest.TestCase):
    def test_finds_real_batches_only(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "stage1_replay_examples.jsonl").write_text("{}\n")
            (d / "stage1_replay_examples_batch2.jsonl").write_text("{}\n")
            (d / "stage1_replay_examples_batch3.jsonl").write_text("{}\n")
            (d / "stage1_replay_examples_pilot_v1.jsonl").write_text("{}\n")
            (d / "stage1_replay_examples_batch2_PILOT_superseded.jsonl").write_text("{}\n")
            (d / "stage1_replay_examples_PILOT_3turn_superseded.jsonl").write_text("{}\n")
            found = run_judge.discover_replay_batches(d)
            names = {p.name for p in found}
            self.assertEqual(names, {
                "stage1_replay_examples.jsonl",
                "stage1_replay_examples_batch2.jsonl",
                "stage1_replay_examples_batch3.jsonl",
            })

    def test_new_future_batch_is_picked_up_automatically(self):
        # Regression guard for the real bug: a hardcoded file list silently
        # capped a real run at batch1+batch2, missing batch3 entirely.
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "stage1_replay_examples.jsonl").write_text("{}\n")
            (d / "stage1_replay_examples_batch99.jsonl").write_text("{}\n")
            found = run_judge.discover_replay_batches(d)
            self.assertEqual(len(found), 2)


class LoadPairedExamplesFailClosedTests(unittest.TestCase):
    """A corrupted trailing line (kill mid-write can only truncate the
    LAST line, since writes are append-only + flushed per example) must
    not be silently skipped: this file is the dedup source of truth for
    "already replayed", so an understated read risks a duplicate,
    spend-generating replay of a call whose real result sits right next to
    the corrupted line."""

    def test_corrupted_trailing_line_raises_not_skips(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            good = json.dumps({"call": {"call_id": "c1", "trajectory_id": "t1"}})
            (d / "stage1_replay_examples.jsonl").write_text(good + "\n" + '{"call": {"call_id": "c2"' )
            with self.assertRaises(run_judge.CorruptedBatchError):
                run_judge.load_paired_examples(d)

    def test_clean_file_loads_normally(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            good = json.dumps({"call": {"call_id": "c1", "trajectory_id": "t1"}})
            (d / "stage1_replay_examples.jsonl").write_text(good + "\n")
            rows = run_judge.load_paired_examples(d)
            self.assertEqual(len(rows), 1)


def _row(call_id, pass_label, verdict="EQUIVALENT", norm_applied=False, orig_token=None, **overrides):
    base = {
        "call_id": call_id,
        "order": ["local", "cloud"],
        "verdict": verdict,
        "reason": "x",
        "raw_judge_text": "{}",
        "pass_label": pass_label,
        "prompt_chars": 1000,
        "estimated_prompt_tokens": 250.0,
        "near_assumed_limit": False,
        "full_state_used": True,
        "original_message_count": 5,
        "included_message_count": 5,
        "omitted_message_count": 0,
        "omitted_message_range": None,
        "task_anchor_present": True,
        "tool_schema_complete": True,
        "system_complete": True,
        "truncation_policy_version": judge.TRUNCATION_POLICY_VERSION,
        "verdict_normalization_applied": norm_applied,
        "original_parsed_token": orig_token,
    }
    base.update(overrides)
    return base


class AuditSnapshotTests(unittest.TestCase):
    def test_clean_snapshot_passes_everything(self):
        rows = [_row("c1", "primary"), _row("c1", "reversed"), _row("c2", "primary"), _row("c2", "reversed")]
        audit = run_judge.audit_snapshot(rows, expected_call_count=2)
        self.assertTrue(audit["snapshot_clean"])
        self.assertEqual(audit["n_unique_call_ids"], 2)
        self.assertEqual(audit["n_duplicate_pairs"], 0)
        self.assertTrue(audit["pairs_complete"])
        self.assertEqual(audit["n_parse_errors"], 0)
        self.assertTrue(audit["provenance_clean"])
        self.assertTrue(audit["normalization_audit_clean"])

    def test_detects_duplicate_pair(self):
        rows = [_row("c1", "primary"), _row("c1", "primary"), _row("c1", "reversed")]
        audit = run_judge.audit_snapshot(rows)
        self.assertEqual(audit["n_duplicate_pairs"], 1)
        self.assertFalse(audit["snapshot_clean"])

    def test_detects_missing_reversed_pass(self):
        rows = [_row("c1", "primary")]
        audit = run_judge.audit_snapshot(rows)
        self.assertEqual(audit["n_missing_reversed"], 1)
        self.assertFalse(audit["pairs_complete"])
        self.assertFalse(audit["snapshot_clean"])

    def test_detects_parse_error(self):
        rows = [_row("c1", "primary", verdict="PARSE_ERROR"), _row("c1", "reversed")]
        audit = run_judge.audit_snapshot(rows)
        self.assertEqual(audit["n_parse_errors"], 1)
        self.assertFalse(audit["snapshot_clean"])

    def test_detects_provenance_violation(self):
        rows = [_row("c1", "primary", full_state_used=False), _row("c1", "reversed")]
        audit = run_judge.audit_snapshot(rows)
        self.assertEqual(audit["n_provenance_violations"], 1)
        self.assertFalse(audit["provenance_clean"])

    def test_detects_missing_system_prompt_provenance(self):
        rows = [_row("c1", "primary", system_complete=False), _row("c1", "reversed")]
        audit = run_judge.audit_snapshot(rows)
        self.assertFalse(audit["provenance_clean"])
        self.assertIn("system_incomplete", audit["provenance_violations"][0]["issues"])

    def test_detects_normalization_missing_token(self):
        rows = [_row("c1", "primary", norm_applied=True, orig_token=None), _row("c1", "reversed")]
        audit = run_judge.audit_snapshot(rows)
        self.assertEqual(audit["n_normalization_missing_token"], 1)
        self.assertFalse(audit["normalization_audit_clean"])

    def test_detects_token_without_flag_inconsistency(self):
        rows = [_row("c1", "primary", norm_applied=False, orig_token="EQUIVALIENT"), _row("c1", "reversed")]
        audit = run_judge.audit_snapshot(rows)
        self.assertEqual(audit["n_inconsistent_token_without_flag"], 1)
        self.assertFalse(audit["normalization_audit_clean"])

    def test_expected_call_count_mismatch_fails_clean(self):
        rows = [_row("c1", "primary"), _row("c1", "reversed")]
        audit = run_judge.audit_snapshot(rows, expected_call_count=5)
        self.assertFalse(audit["call_count_matches_expected"])
        self.assertFalse(audit["snapshot_clean"])


if __name__ == "__main__":
    unittest.main()
