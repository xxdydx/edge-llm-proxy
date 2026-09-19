"""Safety checks for the paired-label model evaluation contract."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from experiments.agentic_router import analysis, model_comparison as mc


class ModelComparisonTests(unittest.TestCase):
    def test_load_events_excludes_capacity_and_quarantined_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"call": {"call_id": "capacity"}, "local_outcome": {"status": "INVALID_CAPACITY"},
                 "cloud_outcome": {"status": "OK"}},
                {"call": {"call_id": "crash"}, "local_outcome": {"status": "OK"},
                 "cloud_outcome": {"status": "OK"}},
            ]
            (root / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (root / "judge.jsonl").write_text("")
            (root / "quarantine_manifest.json").write_text(json.dumps({
                "paired_call_ids_retained_not_for_quality_training": ["crash"]}))
            events, meta = mc.load_events(root, ("pairs.jsonl",), "judge.jsonl")
            self.assertEqual(events, [])
            self.assertEqual(meta["n_replay_raw"], 2)
            self.assertEqual(meta["n_excluded_arm_status"], 1)
            self.assertEqual(meta["n_excluded_quarantine"], 1)

    def test_load_events_rejects_judge_for_capacity_invalid_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pairs.jsonl").write_text(json.dumps({
                "call": {"call_id": "capacity"},
                "local_outcome": {"status": "INVALID_CAPACITY"},
                "cloud_outcome": {"status": "OK"}}) + "\n")
            (root / "judge.jsonl").write_text(json.dumps({
                "call_id": "capacity", "pass_label": "primary"}) + "\n")
            with self.assertRaisesRegex(ValueError, "nonlabelable replay"):
                mc.load_events(root, ("pairs.jsonl",), "judge.jsonl")

    def test_reserved_tasks_are_not_used_in_development_folds(self):
        # A later held-out label must not alter any development AUC.  This
        # catches accidental tuning on the predeclared two-task holdout.
        names = list(analysis.FEATURE_NAMES)
        def row(group: str, label: int, value: float):
            return analysis.CallRow({k: value for k in names}, label, group)

        dev = [row("swebench-ansible-a", i % 2, float(i)) for i in range(8)]
        dev += [row("swebench-openlibrary-b", i % 2, float(i + 10)) for i in range(8)]
        dev += [row("swebench-qutebrowser-c", i % 2, float(i + 20)) for i in range(8)]
        meta = {"feature_names": names}
        before = mc.compare(dev, meta)
        heldout = [row(mc.RESERVED_PILOT_HOLDOUT_TASKS[0], i % 2, float(i + 30)) for i in range(8)]
        after = mc.compare(dev + heldout, meta)
        self.assertEqual(before["folds"], after["folds"])
        self.assertEqual(after["holdout_reservation"]["n_labeled_holdout_rows"], 8)
        self.assertNotIn("final_reserved_holdout", after)

    def test_shadow_artifact_disables_live_local_proposals(self):
        names = list(analysis.FEATURE_NAMES)
        rows = [analysis.CallRow({k: float(i) for k in names}, i % 2, f"swebench-repo-{i % 3}")
                for i in range(12)]
        meta = {"replay_files": ["base.jsonl"], "judge_file": "judge.jsonl",
                "replay_sha256": {"base.jsonl": "abc"}, "judge_sha256": "def",
                "source_trace_sha256": {"trace.jsonl": "ghi"},
                "feature_source": "synthetic chronology", "feature_derivation_version": "test-v1",
                "judge_protocol_versions": ["test-judge-v1"]}
        artifact = mc.shadow_logreg_artifact(rows, meta)
        self.assertEqual(artifact["positive_class"], "HARM_LOCAL")
        self.assertEqual(artifact["local_risk_threshold"], 0.0)
        self.assertEqual(artifact["status"], "shadow_only_no_live_activation")
        self.assertEqual(artifact["feature_names"], names)
        self.assertEqual(len(artifact["coefficients"]), len(names))


if __name__ == "__main__":
    unittest.main()
