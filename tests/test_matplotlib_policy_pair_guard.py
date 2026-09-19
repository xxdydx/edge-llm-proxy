"""Local-only safety tests for the one-task policy-pair launcher."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parent.parent / "scripts/run_matplotlib_policy_pair.py"
spec = importlib.util.spec_from_file_location("matplotlib_policy_pair_guard", SCRIPT)
assert spec and spec.loader
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class MatplotlibPolicyPairGuardTests(unittest.TestCase):
    def test_frozen_task_and_deadline(self) -> None:
        self.assertEqual(guard._sha(guard.TASK), guard.TASK_SHA256)
        self.assertEqual(guard.ORDERS, ("cloud", "routing"))
        self.assertEqual(guard.AGENT_CAP_S, 1200)
        self.assertLess(guard.LATEST_PAIR_START, guard.DRAIN)

    def test_drain_gate_prevents_child_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(guard, "RESULTS", Path(tmp)), \
                patch.object(guard, "_remaining", return_value=0):
            with self.assertRaisesRegex(guard.GateFailure, "insufficient_drain_budget"):
                guard._run([sys.executable, "-c", "raise RuntimeError('must not launch')"],
                           10, "never.log")
            self.assertFalse((Path(tmp) / "never.log").exists())

    def test_timeout_terminates_scoped_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(guard, "RESULTS", Path(tmp)):
            with self.assertRaisesRegex(guard.GateFailure, "timeout_or_drain"):
                guard._run([sys.executable, "-c", "import time; time.sleep(5)"],
                           0.1, "bounded.log")
            self.assertTrue((Path(tmp) / "bounded.log").is_file())
            self.assertEqual(
                __import__("json").loads((Path(tmp) / "controller_status.json").read_text())["stage"],
                "bounded.log",
            )


if __name__ == "__main__":
    unittest.main()
