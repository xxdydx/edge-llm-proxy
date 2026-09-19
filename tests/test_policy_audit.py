"""Tests for policy_audit.py -- the leakage audit, per-group fold table,
and baseline policies, on synthetic fixtures.
"""

from __future__ import annotations

import unittest

from experiments.agentic_router import analysis, policy_audit as pa
from experiments.agentic_router.quality_pipeline import LinUCBArm


def _row(label, task_group="g1", **feature_overrides):
    feats = {name: 0.0 for name in analysis.FEATURE_NAMES}
    feats.update(feature_overrides)
    return analysis.CallRow(feats, label, task_group)


class ChronologicalOrderAuditTests(unittest.TestCase):
    def test_valid_order_passes(self):
        rows = [_row(1), _row(0), _row(1)]
        ok, n = pa.audit_chronological_order(rows, [1.0, 2.0, 3.0])
        self.assertTrue(ok)
        self.assertEqual(n, 0)

    def test_out_of_order_timestamps_detected(self):
        rows = [_row(1), _row(0), _row(1)]
        ok, n = pa.audit_chronological_order(rows, [1.0, 5.0, 3.0])  # 5.0 -> 3.0 is a violation
        self.assertFalse(ok)
        self.assertEqual(n, 1)

    def test_ties_are_not_violations(self):
        rows = [_row(1), _row(0)]
        ok, n = pa.audit_chronological_order(rows, [1.0, 1.0])
        self.assertTrue(ok)


class LeakageAuditReconstructionTests(unittest.TestCase):
    """The core technique leakage_audit relies on: from-scratch
    reconstruction using only rows[0:i] must match live incremental state
    at step i. This proves the TECHNIQUE is sensitive to leaks, by
    directly constructing a case where an arm sees an extra illegitimate
    update and confirming its signature then diverges from a clean one."""

    def test_clean_incremental_state_matches_fresh_reconstruction(self):
        rows = [_row(1, x1=1.0), _row(0, x1=-1.0), _row(1, x1=0.5)]
        d = len(analysis.FEATURE_NAMES)
        # Build "live" via the same predict-then-update discipline leakage_audit uses.
        live_local, live_cloud = LinUCBArm(d), LinUCBArm(d)
        for row in rows:
            x = [row.features[f] for f in analysis.FEATURE_NAMES]
            chose_local = live_local.ucb(x) >= live_cloud.ucb(x)
            r = 1.0 if chose_local == (row.label == 1) else 0.0
            (live_local if chose_local else live_cloud).update(x, r)
        # Fresh reconstruction using the identical rule.
        fresh_local, fresh_cloud = LinUCBArm(d), LinUCBArm(d)
        for row in rows:
            x = [row.features[f] for f in analysis.FEATURE_NAMES]
            chose_local = fresh_local.ucb(x) >= fresh_cloud.ucb(x)
            r = 1.0 if chose_local == (row.label == 1) else 0.0
            (fresh_local if chose_local else fresh_cloud).update(x, r)
        self.assertEqual(pa._arm_signature(live_local), pa._arm_signature(fresh_local))
        self.assertEqual(pa._arm_signature(live_cloud), pa._arm_signature(fresh_cloud))

    def test_leaked_state_diverges_from_clean_state(self):
        """Directly demonstrates the comparison leakage_audit relies on IS
        sensitive to a real leak: one arm receives an extra update using a
        row it should not have seen yet ("future" data), the other doesn't
        -- their signatures must differ."""
        d = len(analysis.FEATURE_NAMES)
        clean = LinUCBArm(d)
        leaked = LinUCBArm(d)
        x_future = [1.0] + [0.0] * (d - 1)
        leaked.update(x_future, 1.0)  # illegitimate extra update the clean copy never sees
        self.assertNotEqual(pa._arm_signature(clean), pa._arm_signature(leaked))

    def test_real_leakage_audit_passes_on_the_actual_simulation(self):
        rows = [_row(1, x1=float(i % 3)) for i in range(12)]
        ts = list(range(12))
        result = pa.leakage_audit(rows, ts)
        self.assertTrue(result.passed)
        self.assertEqual(result.n_state_mismatches, 0)
        self.assertTrue(result.chronological_order_valid)


class PerGroupFoldTableTests(unittest.TestCase):
    def test_reports_one_fold_per_group(self):
        rows = (
            [_row(1, task_group="g1", x1=1.0) for _ in range(3)]
            + [_row(0, task_group="g1", x1=-1.0) for _ in range(3)]
            + [_row(1, task_group="g2", x1=1.0) for _ in range(3)]
            + [_row(0, task_group="g2", x1=-1.0) for _ in range(3)]
        )
        table = pa.per_group_fold_table(rows)
        self.assertEqual(table["n_groups"], 2)
        self.assertEqual(table["verdict"], "INDETERMINATE")  # < 8 groups
        self.assertEqual(len(table["folds"]), 2)
        held_outs = {f["held_out_group"] for f in table["folds"]}
        self.assertEqual(held_outs, {"g1", "g2"})
        for f in table["folds"]:
            self.assertEqual(f["n_test"], 6)
            self.assertEqual(f["n_train"], 6)


class BaselinePolicyTests(unittest.TestCase):
    def test_oracle_always_scores_full_reward(self):
        rows = [_row(1), _row(0), _row(1), _row(0)]
        result = pa._run_deterministic_policy(rows, lambda i, row: row.label == 1, "oracle")
        self.assertEqual(result.mean_reward, 1.0)
        self.assertEqual(result.regret_vs_oracle, 0.0)

    def test_always_local_reward_equals_safe_share(self):
        rows = [_row(1), _row(1), _row(0), _row(0)]  # 50% SAFE
        result = pa._run_deterministic_policy(rows, lambda i, row: True, "always_local")
        self.assertEqual(result.mean_reward, 0.5)
        self.assertEqual(result.local_route_rate, 1.0)
        self.assertEqual(result.harm_rate, 0.5)  # half of locally-routed calls were HARM

    def test_always_cloud_reward_equals_harm_share(self):
        rows = [_row(1), _row(1), _row(0)]  # 1/3 HARM
        result = pa._run_deterministic_policy(rows, lambda i, row: False, "always_cloud")
        self.assertAlmostEqual(result.mean_reward, 1 / 3)
        self.assertEqual(result.local_route_rate, 0.0)
        self.assertEqual(result.harm_rate, 0.0)  # never routes local, so no locally-routed harm


if __name__ == "__main__":
    unittest.main()
