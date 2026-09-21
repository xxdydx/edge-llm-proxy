#!/usr/bin/env python3
"""Official, resumable SWE-Bench-Fork grader for pi-dataset-ac-v1.

Root shim that delegates to experiments/new_datasets/w3_capture/grade_pi_dataset_ac.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

W3 = Path(__file__).resolve().parent / "experiments" / "new_datasets" / "w3_capture"
if str(W3) not in sys.path:
    sys.path.insert(0, str(W3))

from grade_pi_dataset_ac import (  # noqa: E402, F401
    GradingProtocolError,
    grade_all_completed,
    main,
)

if __name__ == "__main__":
    main()
