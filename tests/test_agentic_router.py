"""Tests for experiments/agentic_router/: collect -> features on a small
synthetic trace fixture. No real trace/verdict files, no network, no GPU --
per the project plan's verification section, these validate the join and
derivation logic before it's ever pointed at real campaign data.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.agentic_router import collect, features
from experiments.agentic_router.schema import AgentCallRecord, AgentTrajectory


def _fake_request(messages=None, tools=None):
    return {
        "model": "claude-sonnet-5",
        "system": [],
        "tools": tools or [],
        "messages": messages or [{"role": "user", "content": "do the task"}],
        "max_tokens": 512,
    }


def _fake_raw_record(ts, placement, request, response=None, features_dict=None, policy="static", reason="fits"):
    return {
        "path": "/v1/messages",
        "ts": ts,
        "id": f"call-{ts}",
        "experiment_id": "unit-test-campaign",
        "placement": placement,
        "policy": policy,
        "reason": reason,
        "request": request,
        "response": response or {"content": [], "stop_reason": "end_turn", "model": placement},
        "features": features_dict or {},
        "token_accounting": {"input_tokens": 100, "cache_details_available": False},
        "timing": {},
        "headers": {},
    }


class CollectTests(unittest.TestCase):
    def test_discover_verdicts_parses_task_condition_seed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            vdir = root / "campaignA" / "verdicts"
            vdir.mkdir(parents=True)
            (vdir / "swebench-foo-bar__cloud__seed1.json").write_text(
                json.dumps({"passed": True, "reason": "ok"})
            )
            entries = collect.discover_verdicts(results_root=root)
            self.assertEqual(len(entries), 1)
            e = entries[0]
            self.assertEqual(e["task_group"], "swebench-foo-bar")
            self.assertEqual(e["condition"], "cloud")
            self.assertEqual(e["seed"], 1)

    def test_discover_verdicts_ignores_malformed_names(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            vdir = root / "campaignA" / "verdicts"
            vdir.mkdir(parents=True)
            (vdir / "not-a-verdict-name.json").write_text("{}")
            self.assertEqual(collect.discover_verdicts(results_root=root), [])

    def test_build_trajectory_orders_calls_by_timestamp_and_joins_verdict(self):
        r1 = _fake_raw_record(2.0, "cloud", _fake_request())
        r2 = _fake_raw_record(1.0, "local", _fake_request())  # out of order on purpose
        traj = collect.build_trajectory(
            [r1, r2], task_group="swebench-foo-bar", campaign="campaignA",
            condition="routing", seed=1, trace_path=Path("dummy.jsonl"),
            task_passed=True, verdict_detail="ok", verdict_path=Path("v.json"),
        )
        self.assertEqual(len(traj.calls), 2)
        self.assertEqual([c.placement for c in traj.calls], ["local", "cloud"])
        self.assertEqual(traj.task_passed, True)
        self.assertEqual(traj.trajectory_id, "campaignA:swebench-foo-bar:routing:seed1")

    def test_build_trajectory_skips_non_message_records(self):
        non_call = {"path": "/health", "ts": 0.5}
        r1 = _fake_raw_record(1.0, "cloud", _fake_request())
        traj = collect.build_trajectory(
            [non_call, r1], task_group="g", campaign="c", condition="cloud", seed=1,
            trace_path=Path("d.jsonl"), task_passed=None, verdict_detail=None,
            verdict_path=Path("v.json"),
        )
        self.assertEqual(len(traj.calls), 1)

    def test_load_verdict_treats_grading_infra_errors_as_unreadable(self):
        """A real bug (2026-09-16): NodeBB (a JS repo) was graded with the
        pytest-only harness and produced 'no junit xml produced ... No
        module named pytest' with passed=False -- that False is NOT a real
        task outcome, and must not be usable as ground truth. Same for a
        disk-full docker pull failure."""
        with TemporaryDirectory() as tmp:
            for reason in (
                "no junit xml produced (pytest exit 1): /usr/bin/python3: No module named pytest",
                "docker run failed: ... no space left on device",
                "docker run failed: Unable to find image 'x' locally",
            ):
                p = Path(tmp) / "v.json"
                p.write_text(json.dumps({"passed": False, "reason": reason}))
                passed, detail = collect.load_verdict(p)
                self.assertIsNone(passed, msg=f"should be unreadable for: {reason}")
                self.assertEqual(detail, reason)

    def test_load_verdict_checks_error_field_not_just_reason(self):
        """Real bug (2026-09-16): a container-setup crash (docker pull
        failure) leaves reason as the generic runner placeholder 'job did
        not complete', with the actual cause only in the separate `error`
        field. Checking `reason` alone missed this class of failure."""
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.json"
            p.write_text(json.dumps({
                "passed": False,
                "reason": "job did not complete",
                "error": "RuntimeError: docker run failed: ...",
            }))
            passed, detail = collect.load_verdict(p)
            self.assertIsNone(passed)

    def test_load_verdict_flags_pytest_exit_4_no_tests_collected(self):
        """pytest exit code 4 is pytest's own documented 'no tests were
        collected' -- a harness/environment problem (e.g. wrong paths,
        missing fixtures), not a genuine task outcome, even when Claude
        Code did real work on the trajectory."""
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.json"
            p.write_text(json.dumps({
                "passed": False,
                "reason": "fail_to_pass 0/3, pass_to_pass 0/55 (pytest exit 4)",
            }))
            passed, detail = collect.load_verdict(p)
            self.assertIsNone(passed)

    def test_load_verdict_real_task_failure_is_not_flagged(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.json"
            p.write_text(json.dumps({"passed": False, "reason": "fail_to_pass 0/17, pass_to_pass 39/39 (pytest exit 1)"}))
            passed, detail = collect.load_verdict(p)
            self.assertEqual(passed, False)

    def test_load_verdict_quarantines_agent_induced_official_collection_crash(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.json"
            p.write_text(json.dumps({"passed": False, "n_fail_to_pass_passed": 0,
                                     "n_pass_to_pass_passed": 0,
                                     "reason": "fail_to_pass 0/1, pass_to_pass 0/8 (official swebench grading, patch_applied=False)",
                                     "error": None}))
            p.with_suffix(".eval.log").write_text(
                "Applied patch tests/test_pylint_runners.py cleanly.\n"
                "collected 9 items\nINTERNALERROR> TypeError from agent-edited code\n")
            passed, detail = collect.load_verdict(p)
            self.assertIsNone(passed)
            self.assertIn("internal error", detail)

    def test_load_verdict_quarantines_no_tests_but_not_regular_all_fail(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.json"
            p.write_text(json.dumps({"passed": False, "n_fail_to_pass_passed": 0,
                                     "n_pass_to_pass_passed": 0,
                                     "reason": "fail_to_pass 0/1, pass_to_pass 0/8 (official swebench grading)"}))
            p.with_suffix(".eval.log").write_text("collected 0 items\nno tests ran in 0.01s\n")
            self.assertIsNone(collect.load_verdict(p)[0])
            p.with_suffix(".eval.log").write_text("collected 9 items\n9 failed in 0.11s\n")
            self.assertIs(collect.load_verdict(p)[0], False)

    def test_locate_trace_file_prefers_most_recently_modified_on_ambiguity(self):
        """A real bug (2026-09-16): when a job is killed and relaunched
        under the same campaign/instance/condition/seed, two trace dirs can
        match the same glob pattern. Ascending alphabetical sort (run-stamps
        are chronological strings) picked the OLDEST/stale one -- this
        caused a real false alarm (a 1-record disk-full-attempt trace read
        as an auth failure, when the real 55-record relaunch was sitting
        right next to it). Must pick the newest by mtime instead."""
        import time as time_mod
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir = root / "suite-run-cloud-20260101T000000Z-swebench-foo-seed1"
            new_dir = root / "suite-run-cloud-20260102T000000Z-swebench-foo-seed1"
            old_dir.mkdir()
            new_dir.mkdir()
            old_line = json.dumps({"experiment_id": "campaignA", "path": "/api/hello"}) + "\n"
            new_line = json.dumps({"experiment_id": "campaignA", "path": "/v1/messages"}) + "\n"
            (old_dir / "t.jsonl").write_text(old_line)
            (new_dir / "t.jsonl").write_text(new_line)
            # Force old_dir's file to have an earlier mtime explicitly (dir
            # creation order alone isn't a reliable mtime guarantee).
            import os
            old_ts = time_mod.time() - 100
            os.utime(old_dir / "t.jsonl", (old_ts, old_ts))

            found = collect.locate_trace_file("campaignA", "swebench-foo", "cloud", 1, traces_root=root)
            self.assertEqual(found, new_dir / "t.jsonl")


class FeaturesTests(unittest.TestCase):
    def _traj(self, calls: list[AgentCallRecord]) -> AgentTrajectory:
        return AgentTrajectory(
            trajectory_id="t", task_group="g", campaign="c", condition="cloud",
            seed=1, verdict_path="v.json", task_passed=True, verdict_detail=None,
            calls=calls,
        )

    def _call(self, idx, placement, tool_names=None, stop_reason="end_turn",
              errored_density=0.0, request=None, input_tokens=100,
              local_prompt_tokens=None, local_token_budget=None, produced_tools=None):
        return AgentCallRecord(
            call_id=f"c{idx}", trajectory_id="t", task_group="g",
            source_campaign="c", source_trace_path="p", record_index=idx,
            turn_index=idx, timestamp_unix_s=float(idx), placement=placement,
            policy="static", reason="fits", request=request or _fake_request(),
            response=None, stop_reason=stop_reason, tool_names=tool_names or [],
            router_features={
                "errored_tool_result_density": errored_density,
                "local_prompt_tokens": local_prompt_tokens,
                "local_token_budget": local_token_budget,
            },
            tokens={"input_tokens": input_tokens},
            tool_use_blocks=[{"tool_name": name, "schema_valid": True} for name in (produced_tools or [])],
        )

    def test_previous_backend_and_consecutive_count(self):
        traj = self._traj([
            self._call(0, "cloud"),
            self._call(1, "cloud"),
            self._call(2, "local"),
        ])
        out = features.derive(traj)
        self.assertIsNone(out.calls[0].derived["previous_backend"])
        self.assertEqual(out.calls[0].derived["consecutive_same_backend_turns"], 0)
        self.assertEqual(out.calls[1].derived["previous_backend"], "cloud")
        self.assertEqual(out.calls[1].derived["consecutive_same_backend_turns"], 1)
        self.assertEqual(out.calls[2].derived["previous_backend"], "cloud")
        self.assertEqual(out.calls[2].derived["consecutive_same_backend_turns"], 2)

    def test_recent_tool_error_count_windowed(self):
        traj = self._traj([
            self._call(0, "local", errored_density=0.5),
            self._call(1, "local", errored_density=0.0),
            self._call(2, "local", errored_density=0.0),
            self._call(3, "local", errored_density=0.0),
        ])
        out = features.derive(traj)
        # call 3's window is calls[0:3] -> only call 0 has nonzero density
        self.assertEqual(out.calls[3].derived["recent_tool_error_count"], 1)

    def test_prior_response_truncated_flag(self):
        traj = self._traj([
            self._call(0, "local", stop_reason="max_tokens"),
            self._call(1, "local", stop_reason="end_turn"),
        ])
        out = features.derive(traj)
        self.assertTrue(out.calls[1].derived["prior_response_truncated_or_invalid"])

    def test_repair_loop_flag_on_repeated_tool_signature(self):
        sig = ["Bash"]
        traj = self._traj([
            self._call(0, "local", tool_names=["Bash", "Read"], produced_tools=sig),
            self._call(1, "local", tool_names=["Bash", "Read"], produced_tools=sig),
            self._call(2, "local", tool_names=["Bash", "Read"], produced_tools=sig),
            self._call(3, "local", tool_names=["Bash", "Read"], produced_tools=[]),
        ])
        out = features.derive(traj)
        self.assertFalse(out.calls[2].derived["repair_loop_flag"])
        self.assertTrue(out.calls[3].derived["repair_loop_flag"])

    def test_current_placement_and_response_do_not_change_model_features(self):
        from dataclasses import replace
        from experiments.agentic_router import analysis

        first = self._call(0, "cloud", tool_names=["Bash"], input_tokens=123,
                           local_prompt_tokens=900, local_token_budget=90_000)
        current = self._call(1, "cloud", tool_names=["Read"], input_tokens=200,
                             local_prompt_tokens=1000, local_token_budget=90_000)
        alternative = replace(current, placement="local", tool_names=["Write", "Bash"],
                              tokens={"input_tokens": 9000}, stop_reason="max_tokens")
        a = features.derive(self._traj([first, current])).calls[1]
        b = features.derive(self._traj([first, alternative])).calls[1]
        self.assertEqual(analysis.call_features(a), analysis.call_features(b))
        self.assertEqual(set(analysis.call_features(a)), set(analysis.FEATURE_NAMES))

    def test_context_utilization_ratio(self):
        traj = self._traj([
            self._call(0, "local", local_prompt_tokens=45_000, local_token_budget=90_000),
        ])
        out = features.derive(traj)
        self.assertAlmostEqual(out.calls[0].derived["context_utilization_ratio"], 0.5)

    def test_estimated_recompute_cost_zero_when_no_switch(self):
        traj = self._traj([
            self._call(0, "local"),
            self._call(1, "local"),
        ])
        out = features.derive(traj)
        self.assertEqual(out.calls[1].derived["estimated_recompute_cost_if_switched"], 0.0)

    def test_estimated_recompute_cost_positive_on_switch(self):
        traj = self._traj([
            self._call(0, "local"),
            self._call(1, "cloud", input_tokens=2000),
        ])
        out = features.derive(traj)
        self.assertGreater(out.calls[1].derived["estimated_recompute_cost_if_switched"], 0.0)

    def test_recent_test_failure_count_detects_marker(self):
        failing_request = _fake_request(messages=[
            {"role": "user", "content": "do the task"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "1 FAILED, 0 passed"}
            ]},
        ])
        traj = self._traj([
            self._call(0, "local", request=failing_request),
            self._call(1, "local"),
        ])
        out = features.derive(traj)
        self.assertEqual(out.calls[1].derived["recent_test_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
