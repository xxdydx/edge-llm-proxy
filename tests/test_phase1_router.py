"""Tests for the Phase 1 continuously-learned-routing pilot harness.

The grader tests actually invoke ``docker run`` against the
``phase1-grader:evalplus-0.3.1`` image and are skipped if Docker or the image
is unavailable. They validate the differential grader against official
EvalPlus semantics: gold-vs-gold, known-wrong, exceptions, float atol,
mutation-sensitive inputs, and candidate-stdout isolation.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from dataclasses import replace
from unittest.mock import patch

from experiments.phase1_router import analysis, cost, dataset, features, policy, serde, switch_cost
from experiments.phase1_router.config import EXPERIMENT
from experiments.phase1_router.prompting import extract_code
from experiments.phase1_router.schema import GenerationOutcome, GradeResult, PairedExample, Problem


def _docker_grader_ready() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", EXPERIMENT.grader_image],
            capture_output=True, timeout=20,
        )
        return r.returncode == 0
    except Exception:
        return False


DOCKER_OK = _docker_grader_ready()


# --- serde ---------------------------------------------------------------

class SerdeTests(unittest.TestCase):
    def test_roundtrip_tuple_set_dict_complex(self):
        for obj in [
            (1, 2, 3),
            [(1, 2), (3, 4)],
            {"a": (1, 2), "b": [{1, 2, 3}]},
            {frozenset([1, 2]): "x"},
            complex(1, -2),
            [1, "two", 3.0, True, None],
        ]:
            self.assertEqual(serde.decode(serde.encode(obj)), obj)

    def test_json_safe(self):
        import json
        enc = serde.encode({"k": (1, {2, 3})})
        self.assertEqual(serde.decode(json.loads(json.dumps(enc))), {"k": (1, {2, 3})})


# --- prompting ---------------------------------------------------------------

class ExtractCodeTests(unittest.TestCase):
    def test_prefers_last_fenced_block_defining_entry_point(self):
        text = "```python\ndef foo(): return 1\n```\nthen\n```python\ndef bar(x):\n    return x+1\n```"
        self.assertIn("def bar", extract_code(text, "bar"))
        self.assertIn("def foo", extract_code(text, "foo"))

    def test_raw_text_fallback(self):
        self.assertIn("def baz", extract_code("def baz(x):\n    return x", "baz"))

    def test_none_when_absent(self):
        self.assertIsNone(extract_code("no code here", "qux"))
        self.assertIsNone(extract_code(None, "qux"))


# --- features ----------------------------------------------------------------

class FeatureTests(unittest.TestCase):
    def _p(self, **kw):
        base = dict(
            task_id="Mbpp/1", entry_point="f", prompt='"""\nWrite a function to add.\nassert f(1, 2) == 3\n"""\n',
            canonical_solution="def f(a,b):\n return a+b\n", base_input=[], plus_input=[],
            expected=[], ref_time=[], atol=0.0, contract="", n_base=0, n_plus=0,
            task_group=0, window=0, order=0,
        )
        base.update(kw)
        return Problem(**base)

    def test_all_feature_names_present_and_numeric(self):
        f = features.extract_features(self._p())
        self.assertEqual(set(f), set(features.FEATURE_NAMES))
        self.assertTrue(all(isinstance(v, float) for v in f.values()))

    def test_arity_from_example_assert(self):
        f = features.extract_features(self._p())
        self.assertEqual(f["entry_point_arity"], 2.0)


# --- generate setup (no network) ---------------------------------------------

class GenerateSetupTests(unittest.TestCase):
    def _p(self):
        return Problem(
            task_id="Mbpp/2", entry_point="f",
            prompt='"""\nWrite a function to add.\nassert f(1, 2) == 3\n"""\n',
            canonical_solution="def f(a,b):\n return a+b\n",
            base_input=[], plus_input=[], expected=[], ref_time=[], atol=0.0,
            contract="", n_base=0, n_plus=0, task_group=1, window=0, order=0,
        )

    def test_as_teacher_call_and_request_build_do_not_touch_missing_attrs(self):
        from experiments.phase1_router import generate

        tc = generate._as_teacher_call(self._p(), EXPERIMENT)  # must not raise AttributeError
        self.assertEqual(tc.call_id, "Mbpp_2")
        self.assertEqual(tc.source_trace_path, str(EXPERIMENT.dataset_gz))
        req = tc.request
        self.assertEqual(req["temperature"], EXPERIMENT.temperature)
        self.assertEqual(req["max_tokens"], EXPERIMENT.max_output_tokens)
        self.assertTrue(req["stream"])
        self.assertEqual(req["messages"][0]["role"], "user")
        self.assertIn("add", req["messages"][0]["content"])

    def test_config_has_no_dataset_file_attr(self):
        # guards against reintroducing the removed name
        self.assertFalse(hasattr(EXPERIMENT, "dataset_file"))
        self.assertTrue(hasattr(EXPERIMENT, "dataset_gz"))

    def test_request_disables_qwen_thinking(self):
        # non-trivial MBPP problems otherwise burn the whole budget on <think>
        from experiments.phase1_router import generate

        req = generate._request_for(self._p(), EXPERIMENT)
        self.assertEqual(req.get("chat_template_kwargs"), {"enable_thinking": False})


# --- dataset -------------------------------------------------------------

class DatasetContractTests(unittest.TestCase):
    def test_contract_indented_asserts_are_dedented(self):
        contract = '\n  assert isinstance(x, tuple), "bad" # $_CONTRACT_$\n'
        self.assertTrue(dataset._run_contract("f", contract, [(1, 2)], ["x"]))
        self.assertFalse(dataset._run_contract("f", contract, [[1, 2]], ["x"]))

    def test_empty_contract_accepts(self):
        self.assertTrue(dataset._run_contract("f", "", [1], ["x"]))

    def test_group_hash_is_stable_and_bounded(self):
        g = [dataset._group_of(f"Mbpp/{i}", 15) for i in range(50)]
        self.assertTrue(all(0 <= x < 15 for x in g))
        self.assertEqual(g, [dataset._group_of(f"Mbpp/{i}", 15) for i in range(50)])


# --- switch cost + normalized cost ----------------------------------------

def _mk_example(order, local_ok, cloud_ok, prompt_tokens=500.0,
                local_out=100, cloud_out=200, local_lat=10.0, cloud_lat=20.0):
    p = Problem(task_id=f"Mbpp/{order}", entry_point="f", prompt="p",
                canonical_solution="def f():\n pass\n", base_input=[], plus_input=[],
                expected=[], ref_time=[], atol=0.0, contract="", n_base=1, n_plus=1,
                task_group=order % 3, window=order % 2, order=order)
    lg = GenerationOutcome("local_27b", "OK", "code", "code",
                           {"input_tokens": 10, "cache_read_input_tokens": 100,
                            "cache_creation_input_tokens": 200, "output_tokens": local_out},
                           final_attempt_latency_s=local_lat)
    cg = GenerationOutcome("cloud_deepseek", "OK", "code", "code",
                           {"input_tokens": 0, "output_tokens": cloud_out},
                           final_attempt_latency_s=cloud_lat)
    return PairedExample(
        problem=p, features={"prompt_est_tokens": prompt_tokens},
        local_gen=lg, cloud_gen=cg,
        local_grade=GradeResult("PASS" if local_ok else "FAIL", local_ok, local_ok, 1, 1, 1, 1),
        cloud_grade=GradeResult("PASS" if cloud_ok else "FAIL", cloud_ok, cloud_ok, 1, 1, 1, 1),
        local_ok=local_ok, cloud_ok=cloud_ok,
    )


class SwitchCostTests(unittest.TestCase):
    def test_no_penalty_at_zero_multiplier(self):
        exs = [_mk_example(i, True, True) for i in range(4)]
        route = [True, False, True, False]  # 3 switches
        c0 = switch_cost.sequence_cost(exs, route, 0.0)
        c10 = switch_cost.sequence_cost(exs, route, 10.0)
        self.assertEqual(c0["switch_count"], 3)
        self.assertEqual(c0["cold_token_penalty_kunits"], 0.0)
        self.assertGreater(c10["cold_token_penalty_kunits"], 0.0)
        self.assertGreater(c10["total_cost"], c0["total_cost"])

    def test_all_same_backend_no_switches(self):
        exs = [_mk_example(i, True, True) for i in range(5)]
        self.assertEqual(switch_cost.sequence_cost(exs, [True] * 5, 10.0)["switch_count"], 0)

    def test_local_share_is_not_cost(self):
        exs = [_mk_example(i, True, True) for i in range(6)]
        # two routes with the SAME local share but different switch pattern
        r_clustered = [True, True, True, False, False, False]  # 1 switch
        r_alternating = [True, False, True, False, True, False]  # 5 switches
        c_a = cost.normalized_total_cost(exs, r_clustered, 10.0)
        c_b = cost.normalized_total_cost(exs, r_alternating, 10.0)
        self.assertNotAlmostEqual(c_a, c_b)  # cost depends on switching, not just share

    def test_provider_token_load_cloud_input_is_none_not_zero(self):
        exs = [_mk_example(i, True, True) for i in range(6)]
        tl = cost.token_load(exs, [True, True, True, False, False, False])
        self.assertIsNone(tl["selected_cloud_input_tokens"])
        self.assertTrue(tl["cloud_input_unavailable"])
        self.assertIsNone(tl["local_token_load_pct_incl_cloud_input"])
        self.assertIsNone(tl["local_input_token_share"])
        # output-only variant is computable (both sides measured)
        self.assertIsNotNone(tl["local_token_load_pct_output_only"])
        self.assertAlmostEqual(
            tl["local_token_load_pct_output_only"] + tl["cloud_token_load_pct_output_only"], 1.0
        )
        # raw counts preserved
        self.assertEqual(tl["selected_local_output_tokens"], 300)
        self.assertEqual(tl["selected_cloud_output_tokens"], 600)

    def test_comparable_token_load_is_complete_and_distinct_from_request_share(self):
        exs = [_mk_example(i, True, True, prompt_tokens=(100.0 if i < 3 else 3000.0),
                           local_out=50, cloud_out=100) for i in range(6)]
        ref = {ex.problem.task_id: {"task_id": ex.problem.task_id,
                                    "ref_input_tokens": int(ex.features["prompt_est_tokens"]),
                                    "method": "vllm_tokenizer"} for ex in exs}
        # route the 3 SHORT-prompt problems local -> 50% request share, but a
        # much smaller token-load share
        route = [True, True, True, False, False, False]
        ctl = cost.comparable_token_load(exs, route, ref)
        self.assertIsNotNone(ctl["local_token_load_pct"])  # complete, not missing
        self.assertLess(ctl["local_token_load_pct"], 0.5)  # < request share
        self.assertAlmostEqual(ctl["local_token_load_pct"] + ctl["cloud_token_load_pct"], 1.0)
        self.assertEqual(ctl["reference_input_method_coverage"]["vllm_tokenizer"], 6)
        self.assertEqual(ctl["reference_input_method_coverage"]["coverage_exact_pct"], 1.0)
        # raw counts preserved on both sides
        self.assertEqual(ctl["selected_local_ref_input_tokens"], 3 * 100)
        self.assertEqual(ctl["selected_cloud_ref_input_tokens"], 3 * 3000)

    def test_comparable_token_load_falls_back_to_estimate_and_flags_coverage(self):
        exs = [_mk_example(i, True, True) for i in range(4)]
        ref = {}  # nothing from the tokenizer -> all estimated
        ctl = cost.comparable_token_load(exs, [True, False, True, False], ref)
        self.assertEqual(ctl["reference_input_method_coverage"]["estimated"], 4)
        self.assertEqual(ctl["reference_input_method_coverage"]["coverage_exact_pct"], 0.0)
        self.assertEqual(ctl["reference_input_method_coverage"]["n_tasks_missing_ref_entry"], 4)
        self.assertIsNotNone(ctl["local_token_load_pct"])  # still complete

    def test_raw_components_reported_separately(self):
        exs = [_mk_example(i, True, True) for i in range(4)]
        rc = cost.raw_components(exs, [True, True, False, False])
        self.assertEqual(rc["n_local"], 2)
        self.assertEqual(rc["n_cloud"], 2)
        self.assertIsNone(rc["cloud_input_tokens"])  # gateway 0 -> None
        self.assertEqual(rc["cloud_output_tokens"], 400)
        self.assertEqual(rc["local_output_tokens"], 200)
        self.assertEqual(rc["local_input_tokens"], 2 * (10 + 100 + 200))
        self.assertAlmostEqual(rc["local_latency_s_total"], 20.0)


# --- policies / oracle -------------------------------------------------------

class PolicyTests(unittest.TestCase):
    def test_oracle_routes_local_iff_local_passes(self):
        exs = [_mk_example(0, True, True), _mk_example(1, True, False),
               _mk_example(2, False, True), _mk_example(3, False, False)]
        self.assertEqual(policy.oracle(exs), [True, True, False, False])

    def test_static_heuristic_uses_prompt_length(self):
        exs = [_mk_example(0, True, True, prompt_tokens=100.0),
               _mk_example(1, True, True, prompt_tokens=5000.0)]
        self.assertEqual(policy.static_heuristic(exs, est_token_cutoff=900.0), [True, False])

    def test_no_harm_label_targets_only_cloud_only_quadrant(self):
        exs = [_mk_example(0, True, True), _mk_example(1, True, False),
               _mk_example(2, False, True), _mk_example(3, False, False)]
        self.assertEqual(policy._no_harm_label(exs).tolist(), [1, 1, 0, 1])


# --- capacity gate / joint quality+capacity controller (step 3) ------------

class CapacityGateTests(unittest.TestCase):
    def setUp(self):
        from experiments.phase1_router import capacity
        from experiments.phase1_router.config import EXPERIMENT
        self.capacity = capacity
        # small budget + window so a modest offered load makes capacity bind
        self.cfg = replace(
            EXPERIMENT,
            capacity_budget_gpu_sec_per_sec=1.0,
            capacity_window_s=10.0,
            offered_load_grid=(0.5, 2.0, 8.0),
        )
        # constant 0.5 s/request stub estimator (service_seconds itself is a
        # still injects a constant svc_fn for deterministic capacity-math assertions)
        self.svc = lambda ex, cfg: 0.5

    def _stream(self, n=40):
        # every request is high-quality-eligible (local passes, cloud fails) so
        # the quality gate keeps them all local -> capacity is the only limiter
        return [_mk_example(i, local_ok=True, cloud_ok=False) for i in range(n)]

    def test_no_shed_when_capacity_has_slack(self):
        exs = self._stream(20)
        elig = [True] * len(exs)
        p = [0.9] * len(exs)
        route, diag = self.capacity.joint_route(
            exs, p, elig, lam=0.5, cfg=self.cfg, svc_fn=self.svc)
        self.assertEqual(diag.n_shed_by_capacity, 0)
        self.assertEqual(route, [True] * len(exs))
        self.assertLessEqual(diag.peak_window_util, 1.0 + 1e-9)

    def test_capacity_sheds_when_over_budget(self):
        exs = self._stream(40)
        elig = [True] * len(exs)
        p = [0.9] * len(exs)
        # budget 1.0 GPU-s/s * 10 s window = 10 GPU-s; at 8 req/s and 0.5 s each
        # the window offers 8*10*0.5 = 40 GPU-s of demand -> most must shed
        route, diag = self.capacity.joint_route(
            exs, p, elig, lam=8.0, cfg=self.cfg, svc_fn=self.svc)
        self.assertGreater(diag.n_shed_by_capacity, 0)
        self.assertLess(diag.local_request_share, 1.0)
        self.assertLessEqual(diag.peak_window_util, 1.0 + 1e-9)  # joint never overruns

    def test_quality_only_flags_violation_without_shedding(self):
        exs = self._stream(40)
        elig = [True] * len(exs)
        p = [0.9] * len(exs)
        route, diag = self.capacity.joint_route(
            exs, p, elig, lam=8.0, cfg=self.cfg,
            enforce_capacity=False, svc_fn=self.svc)
        self.assertEqual(diag.n_shed_by_capacity, 0)
        self.assertEqual(sum(route), len(exs))          # nothing shed
        self.assertGreater(diag.peak_window_util, 1.0)  # but the box is overrun

    def test_est_output_tokens_uses_usage_then_default(self):
        ex = _mk_example(0, True, True, local_out=137)
        self.assertEqual(self.capacity.est_output_tokens(ex, self.cfg), 137)
        ex0 = _mk_example(1, True, True, local_out=0)
        self.assertEqual(
            self.capacity.est_output_tokens(ex0, self.cfg),
            self.cfg.capacity_default_output_tokens,
        )

    def test_sweep_row_shape_and_monotonicity(self):
        exs = self._stream(60)
        p = [0.9] * len(exs)
        rows = self.capacity.joint_capacity_sweep(
            exs, p, threshold=0.5, cfg=self.cfg, q_cloud_test=0.0, svc_fn=self.svc)
        self.assertEqual([r["offered_load_req_s"] for r in rows], [0.5, 2.0, 8.0])
        for r in rows:
            for v in ("joint", "quality_only", "capacity_only", "oracle_cap"):
                self.assertIn("quality", r[v])
                self.assertIn("within_epsilon", r[v])
        # joint local share is non-increasing as offered load rises
        shares = [r["joint"]["local_request_share"] for r in rows]
        self.assertGreaterEqual(shares[0] + 1e-9, shares[-1])

    def test_service_seconds_affine_and_anchored(self):
        from experiments.phase1_router.config import EXPERIMENT
        cfg = EXPERIMENT
        # a reference-length generation totals the measured envelope value
        ref = _mk_example(0, True, True, local_out=cfg.capacity_env_ref_output_tokens)
        self.assertAlmostEqual(
            self.capacity.service_seconds(ref, cfg),
            cfg.capacity_env_svc_sec_per_req, places=6)
        # monotonic in output length, and a near-zero-length gen still pays >= floor
        short = _mk_example(1, True, True, local_out=1)
        long = _mk_example(2, True, True, local_out=800)
        self.assertLess(self.capacity.service_seconds(short, cfg),
                        self.capacity.service_seconds(long, cfg))
        self.assertGreaterEqual(self.capacity.service_seconds(short, cfg),
                                cfg.capacity_svc_floor_s - 1e-9)

    def test_report_runs_end_to_end_with_real_estimator(self):
        exs = [_mk_example(i, local_ok=(i % 2 == 0), cloud_ok=(i % 3 == 0))
               for i in range(40)]
        pol = policy.fit_logreg(exs)
        rep = self.capacity.joint_capacity_report(
            exs, {"logreg": pol}, {"logreg": 0.5}, self.cfg, 0.5)
        self.assertNotIn("status", rep)  # no pending marker; estimator is implemented
        self.assertIn("sweep", rep["logreg"])
        self.assertEqual(len(rep["logreg"]["sweep"]), len(self.cfg.offered_load_grid))


# --- epsilon as an operator-tunable knob -----------------------------------

class EpsilonSensitivityTests(unittest.TestCase):
    def test_wider_epsilon_never_yields_less_local_share(self):
        # 3 groups so train/val/test each get real examples
        exs = [_mk_example(i, local_ok=(i % 3 != 0), cloud_ok=True) for i in range(30)]
        for i, ex in enumerate(exs):
            import dataclasses as dc
            exs[i] = dc.replace(ex, problem=dc.replace(ex.problem, task_group=i % 3))
        train = [e for e in exs if e.problem.task_group == 0]
        val = [e for e in exs if e.problem.task_group == 1]
        test = [e for e in exs if e.problem.task_group == 2]
        from experiments.phase1_router.config import EXPERIMENT
        cfg = replace(EXPERIMENT, threshold_grid=tuple(round(i * 0.05, 2) for i in range(21)))
        learned = {"logreg": policy.fit_logreg(train)}
        out = analysis.epsilon_sensitivity(train, val, test, cfg, learned, epsilons=(0.0, 0.05, 0.2))
        rows = out["logreg"]
        shares = [r["local_share"] for r in rows]
        # a looser epsilon can only admit at least as much local share as a
        # tighter one (the threshold search relaxes its constraint monotonically)
        self.assertEqual(shares, sorted(shares))

    def test_reports_one_row_per_epsilon_per_policy(self):
        exs = [_mk_example(i, True, True) for i in range(6)]
        from experiments.phase1_router.config import EXPERIMENT
        learned = {"logreg": policy.fit_logreg(exs)}
        out = analysis.epsilon_sensitivity(exs, exs, exs, EXPERIMENT, learned, epsilons=(0.0, 0.1))
        self.assertEqual(len(out["logreg"]), 2)
        self.assertEqual({r["epsilon"] for r in out["logreg"]}, {0.0, 0.1})


class RunAnalysisWiringTests(unittest.TestCase):
    def test_report_includes_window_strategy_comparison_and_recommendation(self):
        from experiments.phase1_router.analysis import run_analysis
        from experiments.phase1_router.config import EXPERIMENT
        import dataclasses as dc

        cfg = replace(EXPERIMENT, threshold_grid=tuple(round(i * 0.1, 1) for i in range(11)))
        exs = []
        order = 0
        for w in range(5):
            for k in range(10):
                ex = _mk_example(order, local_ok=(k % 2 == 0), cloud_ok=True)
                ex = dc.replace(ex, problem=dc.replace(ex.problem, window=w, task_group=order))
                exs.append(ex)
                order += 1
        report = run_analysis(exs, exs, exs, exs, cfg)
        self.assertIn("rolling_window_strategies", report)
        rws = report["rolling_window_strategies"]
        self.assertEqual(rws["recommended"], "change_point_reset")
        self.assertEqual(
            set(rws["strategies"]),
            {"expanding", "sliding", "recency_decay", "change_point_reset"},
        )
        # existing key preserved for backward compatibility
        self.assertIn("rolling_window", report)


# --- out-of-fold robustness audit (Codex review follow-up) -----------------

class RobustnessAuditTests(unittest.TestCase):
    def setUp(self):
        from experiments.phase1_router import robustness
        from experiments.phase1_router.config import EXPERIMENT
        self.robustness = robustness
        self.cfg = replace(
            EXPERIMENT,
            capacity_window_s=10.0,
            offered_load_grid=(0.5, 2.0, 8.0),
        )

    def _examples(self, n_groups=8, per_group=6):
        # deterministic mix of local-passing and cloud-only examples across
        # n_groups distinct task groups, so regroup_split has enough groups to
        # form a real train/val/test split. _mk_example sets task_group =
        # order % 3, which is too coarse here, so override it per example.
        import dataclasses as dc
        fixed = []
        order = 0
        for g in range(n_groups):
            local_ok = (g % 2 == 0)
            for _ in range(per_group):
                ex = _mk_example(order, local_ok=local_ok, cloud_ok=True)
                fixed.append(dc.replace(ex, problem=dc.replace(ex.problem, task_group=g)))
                order += 1
        return fixed

    def test_poisson_arrivals_are_nondecreasing_and_seed_stable(self):
        t1 = self.robustness.poisson_arrival_times(50, 4.0, seed=7)
        t2 = self.robustness.poisson_arrival_times(50, 4.0, seed=7)
        self.assertEqual(len(t1), 50)
        self.assertEqual(t1, t2)  # deterministic given a seed
        self.assertEqual(t1, sorted(t1))  # a Poisson process is non-decreasing

    def test_poisson_differs_from_periodic_arrival_pattern(self):
        t = self.robustness.poisson_arrival_times(30, 4.0, seed=1)
        periodic = [i / 4.0 for i in range(30)]
        self.assertNotEqual(t, periodic)  # genuinely random gaps, not i/lambda

    def test_regroup_split_is_group_disjoint_and_seed_varies_membership(self):
        exs = self._examples()
        train, val, test = self.robustness.regroup_split(exs, self.cfg, seed=1)
        tr_g = {e.problem.task_group for e in train}
        va_g = {e.problem.task_group for e in val}
        te_g = {e.problem.task_group for e in test}
        self.assertEqual(tr_g & va_g, set())
        self.assertEqual(tr_g & te_g, set())
        self.assertEqual(va_g & te_g, set())
        train2, val2, test2 = self.robustness.regroup_split(exs, self.cfg, seed=99)
        self.assertNotEqual(
            {e.problem.task_group for e in test},
            {e.problem.task_group for e in test2},
        )

    def test_joint_route_poisson_sheds_under_load_like_periodic_version(self):
        exs = self._examples(n_groups=1, per_group=40)
        elig = [True] * len(exs)
        p = [0.9] * len(exs)
        cfg = replace(self.cfg, capacity_max_local_req_per_s=1.0, capacity_window_s=10.0)
        route, shed = self.robustness.joint_route_poisson(
            exs, p, elig, lam=8.0, cfg=cfg, svc_fn=lambda ex, cfg: 0.5, seed=3
        )
        self.assertGreater(shed, 0)
        # every eligible example is either admitted or shed, never both/neither
        self.assertEqual(sum(route) + shed, len(exs))

    def test_repeated_split_audit_runs_out_of_fold_and_summarizes(self):
        exs = self._examples(n_groups=9, per_group=8)
        rows = self.robustness.repeated_split_capacity_audit(
            exs, self.cfg,
            seeds=(1, 2, 3),
            lambdas=(1.0, 4.0),
            windows=(10.0,),
            replicate_for_window={10.0: 1},
        )
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertIn("policy", r)
            self.assertIn("joint_local_share", r)
            self.assertIn("joint_within_epsilon", r)
            self.assertIn("quality_only_quality", r)
            self.assertIn("quality_only_within_epsilon", r)
            self.assertGreaterEqual(r["test_groups"], 1)
        summary = self.robustness.summarize_by_window_lambda(rows)
        self.assertTrue(all("joint_local_share_mean" in s for s in summary))
        # every (window, lambda) pair present in rows appears once in the summary
        self.assertEqual(
            {(r["window_s"], r["lambda"]) for r in rows},
            {(s["window_s"], s["lambda"]) for s in summary},
        )


# --- backend filter (safe resumable single-arm generation) ------------------

class WindowPolicyTests(unittest.TestCase):
    def setUp(self):
        from experiments.phase1_router import window_policies
        from experiments.phase1_router.config import EXPERIMENT
        self.wp = window_policies
        self.cfg = replace(EXPERIMENT, threshold_grid=tuple(round(i * 0.1, 1) for i in range(11)))

    def _ordered(self, n_windows=5, per_window=20):
        # deterministic mix, distinct task groups per window so fitting has
        # signal; local_ok varies within each window
        exs = []
        order = 0
        for w in range(n_windows):
            for k in range(per_window):
                local_ok = (k % 3 != 0)
                ex = _mk_example(order, local_ok=local_ok, cloud_ok=True)
                import dataclasses as dc
                ex = dc.replace(ex, problem=dc.replace(ex.problem, window=w, task_group=order))
                exs.append(ex)
                order += 1
        return exs

    def test_all_four_strategies_run_and_report_every_window(self):
        ordered = self._ordered()
        out = self.wp.compare_window_strategies(ordered, self.cfg)
        self.assertEqual(set(out), {"expanding", "sliding", "recency_decay", "change_point_reset"})
        for strategy, rows in out.items():
            self.assertEqual([r["window"] for r in rows], [0, 1, 2, 3, 4])
            self.assertTrue(all("within_epsilon" in r for r in rows))

    def test_sliding_window_trains_on_fewer_examples_than_expanding_late(self):
        ordered = self._ordered()
        exp_rows = self.wp.run_window_strategy(ordered, self.cfg, "expanding")
        sld_rows = self.wp.run_window_strategy(ordered, self.cfg, "sliding", sliding_k=1)
        # by the last window, expanding has seen 4 windows of history, sliding(k=1) only 1
        self.assertIn("n_train=80", exp_rows[-1]["policy"])  # 4 windows * 20
        self.assertIn("n_train=20", sld_rows[-1]["policy"])  # 1 window * 20

    def test_recency_decay_runs_without_error_and_uses_full_history(self):
        ordered = self._ordered()
        rows = self.wp.run_window_strategy(ordered, self.cfg, "recency_decay", decay=0.3)
        self.assertEqual(len(rows), 5)

    def test_change_point_reset_fires_on_injected_regime_change(self):
        ordered = self._ordered(n_windows=6, per_window=20)
        # cloud degrades sharply from window 3 onward
        drifted = self.wp.inject_synthetic_regime_change(ordered, from_window=3, flip_rate=0.9, seed=1)
        rows = self.wp.run_window_strategy(drifted, self.cfg, "change_point_reset")
        total_resets = rows[-1]["resets_so_far"]
        self.assertGreaterEqual(total_resets, 1)
        # q_always_cloud should visibly drop starting window 3
        self.assertLess(rows[3]["q_always_cloud"], rows[2]["q_always_cloud"])

    def test_no_regime_change_never_triggers_a_reset(self):
        ordered = self._ordered(n_windows=5, per_window=20)  # stationary, no drift injected
        rows = self.wp.run_window_strategy(ordered, self.cfg, "change_point_reset")
        self.assertEqual(rows[-1]["resets_so_far"], 0)


class BackendFilterTests(unittest.TestCase):
    def _problems(self, n=5):
        return [Problem(
            task_id=f"Mbpp/{i}", entry_point="f", prompt="p",
            canonical_solution="def f():\n pass\n", base_input=[], plus_input=[],
            expected=[], ref_time=[], atol=0.0, contract="", n_base=1, n_plus=1,
            task_group=i, window=0, order=i,
        ) for i in range(n)]

    def test_unknown_backend_rejected(self):
        from experiments.phase1_router import orchestrator as o
        with self.assertRaises(SystemExit):
            o.main(["--backends", "cloud_deepseek,bogus", "--full-batch",
                    "--resume-run-dir", "/nonexistent"])

    def test_generate_all_only_runs_listed_backends(self):
        from experiments.phase1_router import orchestrator as o
        import tempfile
        from pathlib import Path
        from experiments.phase1_router.schema import GenerationOutcome

        calls = []

        def fake_generate_one(p, backend, cfg):
            calls.append((p.task_id, backend))
            return GenerationOutcome(backend, "OK", "code", "code",
                                     {"output_tokens": 3}, latency_s=0.1, retry_attempts=1)

        with tempfile.TemporaryDirectory() as d:
            rd = Path(d)
            with patch.object(o, "generate_one", side_effect=fake_generate_one):
                gens = o._generate_all(self._problems(4), rd, __import__(
                    "experiments.phase1_router.config", fromlist=["EXPERIMENT"]).EXPERIMENT,
                    2, ["cloud_deepseek"])
            self.assertTrue(all(b == "cloud_deepseek" for _, b in calls))
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(gens), 4)
            self.assertTrue((rd / "generations.jsonl").exists())
            # resuming with the SAME arm is a no-op (append-only, keyed dedupe)
            calls.clear()
            with patch.object(o, "generate_one", side_effect=fake_generate_one):
                o._generate_all(self._problems(4), rd,
                    __import__("experiments.phase1_router.config", fromlist=["EXPERIMENT"]).EXPERIMENT,
                    2, ["cloud_deepseek"])
            self.assertEqual(calls, [])

    def test_grade_all_regrades_stale_no_code_after_transport_recovery(self):
        from experiments.phase1_router import orchestrator as o
        import tempfile
        from pathlib import Path

        cfg = __import__("experiments.phase1_router.config", fromlist=["EXPERIMENT"]).EXPERIMENT
        with tempfile.TemporaryDirectory() as d:
            rd = Path(d)
            key = o._gen_key("Mbpp/0", "local_27b")
            # a prior transport-error grade sits in the checkpoint as NO_CODE
            o.executor.append_checkpoint(rd / "grades.jsonl", {
                "call_id": "Mbpp/0", "backend": "local_27b", "status": "NO_CODE",
                "base_pass": False, "plus_pass": False, "n_base": 1, "n_base_ok": 0,
                "n_plus": 1, "n_plus_ok": 0, "detail": "no code extracted", "grader_wall_s": 0.0,
            })
            # the resume produced a real completion for it
            gens = {key: {"call_id": "Mbpp/0", "backend": "local_27b", "status": "OK",
                          "completion": "def f(): return 1"}}
            calls = []

            def fake_grade(p, code, cfg):
                calls.append((p.task_id, code))
                from experiments.phase1_router.schema import GradeResult
                return GradeResult("PASS", True, True, 1, 1, 1, 1)

            with patch.object(o, "grade_completion", side_effect=fake_grade):
                out = o._grade_all(self._problems(1), gens, rd, cfg, 1, ["local_27b"])
            self.assertEqual(len(calls), 1)  # the stale NO_CODE row WAS re-graded
            self.assertEqual(out[key]["status"], "PASS")  # last-writer-wins
            # and a second resume is now a no-op
            calls.clear()
            with patch.object(o, "grade_completion", side_effect=fake_grade):
                o._grade_all(self._problems(1), gens, rd, cfg, 1, ["local_27b"])
            self.assertEqual(calls, [])

    def test_analyze_skips_cleanly_when_one_arm_missing(self):
        from experiments.phase1_router import orchestrator as o
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            rd = Path(d)
            gens = {o._gen_key("Mbpp/0", "cloud_deepseek"): {"call_id": "Mbpp/0", "backend": "cloud_deepseek", "status": "OK"}}
            grades = {o._gen_key("Mbpp/0", "cloud_deepseek"): {"call_id": "Mbpp/0", "backend": "cloud_deepseek", "status": "PASS"}}
            # must not raise, must not write analysis_report.json
            o._analyze(self._problems(1), gens, grades, rd,
                       __import__("experiments.phase1_router.config", fromlist=["EXPERIMENT"]).EXPERIMENT)
            self.assertFalse((rd / "analysis_report.json").exists())


# --- differential grader vs official EvalPlus semantics --------------------

@unittest.skipUnless(DOCKER_OK, "docker + phase1-grader image required")
class GraderValidationTests(unittest.TestCase):
    """Runs one real MBPP+ task through the Docker grader."""

    @classmethod
    def setUpClass(cls):
        from experiments.phase1_router.dataset import build_one

        cls.plain = build_one("Mbpp/2", EXPERIMENT)  # similar_elements (SET_EQ oracle)
        cls.floaty = build_one("Mbpp/581", EXPERIMENT)  # surface_Area (atol special oracle)

    def _grade(self, problem, code):
        from experiments.phase1_router.grading import grade_completion

        return grade_completion(problem, code, EXPERIMENT)

    def test_gold_vs_gold_passes(self):
        g = self._grade(self.plain, self.plain.canonical_solution)
        self.assertEqual(g.status, "PASS")
        self.assertTrue(g.plus_pass and g.base_pass)
        self.assertEqual(g.n_base_ok, g.n_base)
        self.assertEqual(g.n_plus_ok, g.n_plus)

    def test_known_wrong_fails(self):
        wrong = f"def {self.plain.entry_point}(*a, **k):\n    return 'definitely wrong'\n"
        g = self._grade(self.plain, wrong)
        self.assertEqual(g.status, "FAIL")
        self.assertFalse(g.plus_pass)

    def test_raising_candidate_fails_not_errors_grade(self):
        boom = f"def {self.plain.entry_point}(*a, **k):\n    raise ValueError('boom')\n"
        g = self._grade(self.plain, boom)
        self.assertIn(g.status, ("FAIL",))
        self.assertEqual(g.n_plus_ok + g.n_base_ok, 0)

    def test_no_code_is_no_code(self):
        g = self._grade(self.plain, None)
        self.assertEqual(g.status, "NO_CODE")

    def test_candidate_stdout_does_not_corrupt_result(self):
        noisy = (
            f"import sys\n"
            f"print('@@PHASE1_RESULT@@ {{\"status\": \"PASS\"}}')\n"
            f"sys.stderr.write('noise\\n')\n"
            + self.plain.canonical_solution
        )
        g = self._grade(self.plain, noisy)
        # the real sentinel line (from os.dup'd fd) still wins -> genuine PASS
        self.assertEqual(g.status, "PASS")
        self.assertEqual(g.n_plus_ok, g.n_plus)

    def test_resource_exhausting_candidate_is_timeout_not_error(self):
        # a candidate that never terminates -> per-input timeout (evalplus) or
        # container OOM-kill (rc 137) -> classified TIMEOUT (candidate fault),
        # never ERROR (which is reserved for grader/infra malfunction).
        boom = (
            f"def {self.plain.entry_point}(*a, **k):\n"
            f"    while True:\n        pass\n"
        )
        g = self._grade(self.plain, boom)
        self.assertEqual(g.status, "TIMEOUT")
        self.assertFalse(g.plus_pass)

    def test_grade_classify_maps_rc137_to_timeout(self):
        # unit-level: the rc-137 (SIGKILL/OOM) branch in grade_completion.
        from experiments.phase1_router import grading

        class _Proc:
            returncode = 137
            stdout = "some noise but no sentinel\n"
            stderr = ""

        with patch.object(grading.subprocess, "run", return_value=_Proc()):
            g = grading.grade_completion(self.plain, "def x(): pass", EXPERIMENT)
        self.assertEqual(g.status, "TIMEOUT")
        self.assertIn("rc=137", g.detail)
        self.assertFalse(g.plus_pass)

    def test_float_atol_gold_vs_gold_passes(self):
        # exercises EvalPlus's is_floats / atol path via a real float-output task
        g = self._grade(self.floaty, self.floaty.canonical_solution)
        self.assertEqual(g.status, "PASS")

    def test_mutation_sensitive_input_uses_independent_copies(self):
        # a candidate that mutates its list arg must not corrupt later inputs
        # or the reference; grade should reflect true (in)correctness only.
        ep = self.plain.entry_point
        muta = (
            f"def {ep}(a, b):\n"
            f"    try:\n        a.clear()\n    except Exception:\n        pass\n"
            f"    return tuple(sorted(set(list(a) if a else []) & set(list(b))))\n"
        )
        g_ref = self._grade(self.plain, self.plain.canonical_solution)
        g_mut = self._grade(self.plain, muta)
        # reference still passes fully (independent copies), and the mutating
        # candidate is judged on its own merits, deterministically.
        self.assertEqual(g_ref.status, "PASS")
        self.assertIn(g_mut.status, ("PASS", "FAIL"))
        g_mut2 = self._grade(self.plain, muta)
        self.assertEqual(  # deterministic (ignore grader_wall_s jitter)
            (g_mut.status, g_mut.n_base_ok, g_mut.n_plus_ok),
            (g_mut2.status, g_mut2.n_base_ok, g_mut2.n_plus_ok),
        )


if __name__ == "__main__":
    unittest.main()
