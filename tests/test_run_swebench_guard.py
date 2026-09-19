"""Offline guards for bounded live SWE-bench pilot execution."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


RUNNER = Path(__file__).resolve().parent.parent / "eval-suite" / "swebench" / "runner" / "run_swebench.py"
spec = importlib.util.spec_from_file_location("run_swebench", RUNNER)
assert spec and spec.loader
run_swebench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run_swebench
spec.loader.exec_module(run_swebench)


def test_three_consecutive_502s_trip_guard():
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / "trace.jsonl"
        rows = [{"path": "/api/hello", "status": 502},
                {"path": "/v1/messages", "status": 200},
                {"path": "/v1/messages", "status": 502},
                {"path": "/v1/messages", "status": 502},
                {"path": "/v1/messages", "status": 502}]
        p.write_text("".join(json.dumps(row) + "\n" for row in rows))
        assert run_swebench._consecutive_upstream_502(Path(tmp)) == 3
        with p.open("a") as fh:
            fh.write(json.dumps({"path": "/v1/messages", "status": 200}) + "\n")
        assert run_swebench._consecutive_upstream_502(Path(tmp)) == 0


def test_pilot_timeout_and_shadow_flags_parse():
    args = run_swebench.parse_args([
        "--instances", "swebench-flask-70ca03af28",
        "--job-timeout-s", "1200",
        "--agentic-shadow-artifact", "/tmp/model.json",
    ])
    assert args.job_timeout_s == 1200
    assert args.agentic_shadow_artifact == Path("/tmp/model.json")
