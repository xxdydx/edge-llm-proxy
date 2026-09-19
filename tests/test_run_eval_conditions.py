"""Regression test for the silent condition-filtering bugs in
eval-suite/runner/run_eval.py: a hardcoded `== "routing"` (job dispatch)
and a hardcoded `("cloud", "routing")` tuple (summary table) both silently
dropped any other local-backed condition (e.g. "routing-warm") with no
error — exactly the shape of the already-fixed bug in
eval-suite/swebench/runner/run_swebench.py's run_campaign(). Caught live
2026-09-04 running the first Rung 1 validation job: "jobs: 1" at startup,
"0/0 jobs passed" at the end, no error either time.
"""
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

EVAL_RUNNER_DIR = Path(__file__).resolve().parent.parent / "eval-suite" / "runner"
sys.path.insert(0, str(EVAL_RUNNER_DIR))

spec = importlib.util.spec_from_file_location("run_eval", EVAL_RUNNER_DIR / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
sys.modules["run_eval"] = run_eval
spec.loader.exec_module(run_eval)

SWEBENCH_RUNNER = Path(__file__).resolve().parent.parent / "eval-suite" / "swebench" / "runner" / "run_swebench.py"
swebench_spec = importlib.util.spec_from_file_location("run_swebench", SWEBENCH_RUNNER)
run_swebench = importlib.util.module_from_spec(swebench_spec)
sys.modules["run_swebench"] = run_swebench
swebench_spec.loader.exec_module(run_swebench)


def _job(condition: str, seed: int = 1) -> "run_eval.Job":
    return run_eval.Job(task=SimpleNamespace(id="fix-null-handling"), condition=condition, seed=seed)


def _result(condition: str, passed: bool = True) -> "run_eval.JobResult":
    return run_eval.JobResult(
        job_id=f"fix-null-handling__{condition}__seed1",
        task_id="fix-null-handling",
        category="fix",
        condition=condition,
        seed=1,
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="ok",
        wall_time_s=1.0,
        claude_returncode=0,
        claude_timed_out=False,
        proxy_ok=True,
        error=None,
        sandbox_dir="/tmp/sandbox",
        trace_path=None,
        graph_path=None,
        verdict_path=None,
    )


class RunCampaignConditionRoutingTests(unittest.TestCase):
    def test_runner_fallback_capacity_matches_7b_bootstrap_window(self):
        # run_eval.py's no-profile fallback must not tell a 60K local vLLM
        # that it has a 100K context window: LocalOnly would then forward
        # Claude's 64K completion request unchanged and vLLM would reject it
        # before work. This still governs run_eval.py's own default.
        original = os.environ.pop("EDGEPROXY_MAX_LOCAL_TOKENS", None)
        try:
            self.assertEqual(run_eval.parse_args([]).max_local_tokens, 60_000)
        finally:
            if original is not None:
                os.environ["EDGEPROXY_MAX_LOCAL_TOKENS"] = original

    def test_run_swebench_default_capacity_matches_27b_active_setup(self):
        # 2026-09-16: run_swebench.py's default changed 60K -> 100K. The
        # 27B/100K setup is this project's active local model; a future 7B
        # run must pass --max-local-tokens 60000 explicitly rather than
        # relying on this default.
        original = os.environ.pop("EDGEPROXY_MAX_LOCAL_TOKENS", None)
        try:
            self.assertEqual(run_swebench.parse_args([]).max_local_tokens, 100_000)
        finally:
            if original is not None:
                os.environ["EDGEPROXY_MAX_LOCAL_TOKENS"] = original

    def test_routing_warm_jobs_are_dispatched_not_silently_dropped(self):
        jobs = [_job("cloud"), _job("routing"), _job("routing-warm")]
        seen_conditions = []

        def fake_run_job(ctx, job):
            seen_conditions.append(job.condition)
            return _result(job.condition)

        original = run_eval.run_job
        run_eval.run_job = fake_run_job
        try:
            results = run_eval.run_campaign(ctx=None, jobs=jobs, cloud_parallelism=1, local_parallelism=1)
        finally:
            run_eval.run_job = original

        self.assertEqual(len(results), 3)
        self.assertEqual(sorted(seen_conditions), ["cloud", "routing", "routing-warm"])

    def test_only_cloud_goes_to_the_cloud_pool(self):
        # A future non-"cloud" condition must default to the local pool, not
        # be silently excluded from both.
        jobs = [_job("cloud"), _job("some-future-local-condition")]

        def fake_run_job(ctx, job):
            return _result(job.condition)

        original = run_eval.run_job
        run_eval.run_job = fake_run_job
        try:
            results = run_eval.run_campaign(ctx=None, jobs=jobs, cloud_parallelism=1, local_parallelism=1)
        finally:
            run_eval.run_job = original

        self.assertEqual({r.condition for r in results}, {"cloud", "some-future-local-condition"})

    def test_only_routing_cohort_enables_parent_placement(self):
        ctx = SimpleNamespace(
            python_bin="python",
            upstream="https://cloud.test",
            vllm_url="http://local.test",
            experiment_id="experiment-1",
            max_local_tokens=90_000,
            local_token_margin=0.9,
        )
        common = (ctx, 8000, Path("/tmp/trace"), "episode-1")

        for condition in (
            "cloud",
            "routing",
            "routing-warm",
            "routing-drift",
            "routing-plan",
        ):
            command = run_eval.build_proxy_cmd(
                common[0], condition, common[1], common[2], common[3]
            )
            self.assertNotIn("--cohort-parent-placement", command)

        command = run_eval.build_proxy_cmd(
            common[0], "routing-cohort", common[1], common[2], common[3]
        )
        self.assertIn("--cohort-parent-placement", command)
        self.assertEqual(
            command[command.index("--policy") + 1],
            "static",
        )

    def test_routing_plan_selects_planning_escalation_policy(self):
        self.assertEqual(
            run_eval.CONDITION_POLICY["routing-plan"], "planning-escalation"
        )

    def test_routing_learned_agentic_selects_learned_policy(self):
        self.assertEqual(
            run_eval.CONDITION_POLICY["routing-learned-agentic"],
            "learned-agentic",
        )

    def test_routing_learned_agentic_heuristic_selects_wired_policy(self):
        self.assertEqual(
            run_eval.CONDITION_POLICY["routing-learned-agentic-heuristic"],
            "learned-agentic-heuristic",
        )

    def test_local_condition_selects_local_only_baseline(self):
        # The spec's success standard (edge-llm-client.md) asks for a
        # three-way cloud-only/local-only/routed comparison; "local" is the
        # local-only leg, added alongside the pre-existing "cloud" leg.
        self.assertEqual(run_eval.CONDITION_POLICY["local"], "local-only")

    def test_local_jobs_go_to_the_local_pool_not_dropped(self):
        jobs = [_job("cloud"), _job("local"), _job("routing")]
        seen_conditions = []

        def fake_run_job(ctx, job):
            seen_conditions.append(job.condition)
            return _result(job.condition)

        original = run_eval.run_job
        run_eval.run_job = fake_run_job
        try:
            results = run_eval.run_campaign(ctx=None, jobs=jobs, cloud_parallelism=1, local_parallelism=1)
        finally:
            run_eval.run_job = original

        self.assertEqual(sorted(seen_conditions), ["cloud", "local", "routing"])

    def test_swebench_can_serialize_whole_conditions(self):
        jobs = [
            SimpleNamespace(condition=condition, instance={"slug": slug}, seed=1)
            for condition in ("cloud", "local", "routing-drift")
            for slug in ("a", "b")
        ]
        seen = []

        def fake_run_job(ctx, job):
            seen.append(job.condition)
            return SimpleNamespace(
                passed=True,
                job_id=f"{job.instance['slug']}-{job.condition}",
                wall_time_s=0.0,
                reason="ok",
            )

        original = run_swebench.run_job
        run_swebench.run_job = fake_run_job
        try:
            results = run_swebench.run_campaign(
                None, jobs, 1, 1, sequential_conditions=True
            )
        finally:
            run_swebench.run_job = original

        self.assertEqual(len(results), 6)
        self.assertEqual(seen, ["cloud", "cloud", "local", "local", "routing-drift", "routing-drift"])

    def test_swebench_container_names_are_unique_across_model_campaigns(self):
        job_id = "swebench-openlibrary-c05ccf2c__routing-all-cohort__seed1"
        name_7b = run_swebench.container_name_for("matched-7b-20260909", job_id)
        name_27b = run_swebench.container_name_for("matched-27b-20260909", job_id)

        self.assertNotEqual(name_7b, name_27b)
        self.assertLessEqual(len(name_7b), 63)
        self.assertRegex(name_7b, r"^[a-zA-Z0-9][a-zA-Z0-9_.-]+$")


class WriteSummaryConditionTableTests(unittest.TestCase):
    def test_summary_table_includes_routing_warm_rows(self, tmp_path=None):
        import tempfile

        results = [_result("cloud"), _result("routing"), _result("routing-warm")]
        rows = run_eval.aggregate(results)
        conditions_in_rows = {row["condition"] for row in rows}
        self.assertEqual(conditions_in_rows, {"cloud", "routing", "routing-warm"})

        with tempfile.TemporaryDirectory() as d:
            ctx = SimpleNamespace(
                results_dir=Path(d),
                experiment_id="test-exp",
                run_stamp="20260101T000000Z",
            )
            run_eval.write_summary(ctx, results, rows)
            summary_md = (Path(d) / "summary.md").read_text()

        self.assertIn("routing-warm", summary_md)
        self.assertIn("| routing |", summary_md)
        self.assertIn("| cloud |", summary_md)


if __name__ == "__main__":
    unittest.main()
