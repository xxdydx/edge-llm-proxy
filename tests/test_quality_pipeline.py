"""Tests for the paired-order aggregation policy -- the load-bearing piece
every downstream step (quality model, windows, bandit sim) depends on.
"""

from __future__ import annotations

import unittest

from experiments.agentic_router import quality_pipeline as qp


def _pass(verdict, order):
    return {"verdict": verdict, "order": order}


class AggregateCallLabelTests(unittest.TestCase):
    def test_both_agree_local_wins_is_safe(self):
        # primary: A=local wins -> local. reversed: A=cloud, B=local, B wins -> local.
        primary = _pass("A_BETTER", ("local", "cloud"))
        reversed_ = _pass("B_BETTER", ("cloud", "local"))
        self.assertEqual(qp.aggregate_call_label(primary, reversed_), "SAFE")

    def test_both_agree_cloud_wins_is_harm(self):
        primary = _pass("A_BETTER", ("cloud", "local"))
        reversed_ = _pass("B_BETTER", ("local", "cloud"))
        self.assertEqual(qp.aggregate_call_label(primary, reversed_), "HARM")

    def test_both_equivalent_is_safe(self):
        primary = _pass("EQUIVALENT", ("local", "cloud"))
        reversed_ = _pass("EQUIVALENT", ("cloud", "local"))
        self.assertEqual(qp.aggregate_call_label(primary, reversed_), "SAFE")

    def test_order_sensitive_disagreement_is_unknown(self):
        # primary says local wins, reversed ALSO says local wins by verdict
        # string but that verdict string maps to a DIFFERENT arm due to
        # order -- simulate a genuine flip: primary local wins, reversed
        # cloud wins.
        primary = _pass("A_BETTER", ("local", "cloud"))  # -> local
        reversed_ = _pass("A_BETTER", ("cloud", "local"))  # -> cloud
        self.assertEqual(qp.aggregate_call_label(primary, reversed_), "UNKNOWN")

    def test_both_inadequate_is_unknown(self):
        primary = _pass("BOTH_INADEQUATE", ("local", "cloud"))
        reversed_ = _pass("EQUIVALENT", ("cloud", "local"))
        self.assertEqual(qp.aggregate_call_label(primary, reversed_), "UNKNOWN")

    def test_uncertain_is_unknown(self):
        primary = _pass("UNCERTAIN", ("local", "cloud"))
        reversed_ = _pass("A_BETTER", ("cloud", "local"))
        self.assertEqual(qp.aggregate_call_label(primary, reversed_), "UNKNOWN")

    def test_parse_error_is_unknown(self):
        primary = _pass("PARSE_ERROR", ("local", "cloud"))
        reversed_ = _pass("EQUIVALENT", ("cloud", "local"))
        self.assertEqual(qp.aggregate_call_label(primary, reversed_), "UNKNOWN")

    def test_one_equivalent_one_decisive_is_unknown(self):
        # partial agreement, not a clean match -> conservative UNKNOWN
        primary = _pass("EQUIVALENT", ("local", "cloud"))
        reversed_ = _pass("A_BETTER", ("cloud", "local"))  # -> cloud
        self.assertEqual(qp.aggregate_call_label(primary, reversed_), "UNKNOWN")


class BuildCallLabelsTests(unittest.TestCase):
    def test_skips_calls_missing_a_pass(self):
        rows = [
            {"call_id": "c1", "pass_label": "primary", "verdict": "EQUIVALENT", "order": ["local", "cloud"]},
            # c1 has no reversed pass -> should be skipped
            {"call_id": "c2", "pass_label": "primary", "verdict": "EQUIVALENT", "order": ["local", "cloud"]},
            {"call_id": "c2", "pass_label": "reversed", "verdict": "EQUIVALENT", "order": ["cloud", "local"]},
        ]
        labels = qp.build_call_labels(rows)
        self.assertEqual({lc.call_id for lc in labels}, {"c2"})


if __name__ == "__main__":
    unittest.main()
