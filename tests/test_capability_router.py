import contextlib
import threading
import unittest
from unittest.mock import patch

from experiments.capability_router import (
    analysis,
    dataset,
    executor,
    labels,
    models,
    splits,
)
from experiments.capability_router.config import BackendConfig
from experiments.capability_router.schema import LabeledExample, ReplayOutcome, TeacherCall

EMPTY_REQUEST = {"tools": []}


@contextlib.contextmanager
def tempfile_path():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "checkpoint.jsonl"


def _response(*, tool_name=None, tool_input=None, text="ok", model="deepseek-v4-flash"):
    content = [{"type": "text", "text": text}]
    if tool_name is not None:
        content.append({"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": tool_input or {}})
    return {"content": content, "model": model, "stop_reason": "end_turn"}


def _read_schema_request():
    return {
        "tools": [
            {
                "name": "Read",
                "input_schema": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {"file_path": {"type": "string"}},
                },
            }
        ]
    }


def _read_edit_schema_request():
    return {
        "tools": [
            *_read_schema_request()["tools"],
            {
                "name": "Edit",
                "input_schema": {
                    "type": "object",
                    "required": ["file_path", "old_string", "new_string"],
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                },
            },
        ]
    }


def _multi_tool_response(*tools, model="deepseek-v4-flash"):
    return {
        "content": [
            {"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": tool_input}
            for i, (name, tool_input) in enumerate(tools, start=1)
        ],
        "model": model,
        "stop_reason": "tool_use",
    }


class ActionEquivalenceTests(unittest.TestCase):
    def test_teacher_free_text_is_unknown(self):
        teacher = _response()  # no tool call
        outcome = ReplayOutcome("local_27b", "OK", _response(tool_name="Read", tool_input={"file_path": "a.py"}), 1.0)
        label, detail, components = labels.action_equivalence(EMPTY_REQUEST, teacher, outcome)
        self.assertEqual(label, "UNKNOWN")
        self.assertIn("free text", detail)
        self.assertEqual(components.teacher_kind, "text")

    def test_matching_tool_and_input_is_equivalent(self):
        teacher = _response(tool_name="Read", tool_input={"file_path": "a.py"})
        outcome = ReplayOutcome("local_27b", "OK", _response(tool_name="Read", tool_input={"file_path": "a.py"}), 1.0)
        label, _, components = labels.action_equivalence(_read_schema_request(), teacher, outcome)
        self.assertEqual(label, "EQUIVALENT")
        self.assertTrue(components.exact_input_match)

    def test_dict_key_order_does_not_break_equivalence(self):
        teacher = _response(tool_name="Edit", tool_input={"file_path": "a.py", "old_string": "x"})
        outcome = ReplayOutcome(
            "local_27b", "OK",
            _response(tool_name="Edit", tool_input={"old_string": "x", "file_path": "a.py"}),
            1.0,
        )
        label, _, _ = labels.action_equivalence(EMPTY_REQUEST, teacher, outcome)
        self.assertEqual(label, "EQUIVALENT")

    def test_different_tool_name_is_not_equivalent(self):
        teacher = _response(tool_name="Read", tool_input={"file_path": "a.py"})
        outcome = ReplayOutcome("local_27b", "OK", _response(tool_name="Grep", tool_input={"pattern": "x"}), 1.0)
        label, detail, _ = labels.action_equivalence(EMPTY_REQUEST, teacher, outcome)
        self.assertEqual(label, "NOT_EQUIVALENT")
        self.assertIn("tool differs", detail)

    def test_same_tool_different_input_but_schema_valid_is_unknown_not_not_equivalent(self):
        """Correction 2026-09-09: a different-but-schema-valid argument set
        for the same tool is state-dependent (a different old_string could
        be an equally valid fix) and must not be auto-scored a failure."""
        teacher = _response(tool_name="Read", tool_input={"file_path": "a.py"})
        outcome = ReplayOutcome("local_27b", "OK", _response(tool_name="Read", tool_input={"file_path": "b.py"}), 1.0)
        label, detail, components = labels.action_equivalence(_read_schema_request(), teacher, outcome)
        self.assertEqual(label, "UNKNOWN")
        self.assertIn("repository state", detail)
        self.assertFalse(components.exact_input_match)
        self.assertTrue(components.replay_schema_valid)

    def test_same_tool_different_input_and_schema_invalid_is_not_equivalent(self):
        """The one deterministic, state-independent proof of failure: the
        replay's own tool_use fails its requested schema."""
        teacher = _response(tool_name="Read", tool_input={"file_path": "a.py"})
        outcome = ReplayOutcome("local_27b", "OK", _response(tool_name="Read", tool_input={"wrong_key": "b.py"}), 1.0)
        label, detail, components = labels.action_equivalence(_read_schema_request(), teacher, outcome)
        self.assertEqual(label, "NOT_EQUIVALENT")
        self.assertIn("schema validation", detail)
        self.assertFalse(components.replay_schema_valid)

    def test_later_invalid_tool_does_not_change_first_action_label(self):
        """The experiment compares only the first next action, so a later
        parallel tool call cannot supply schema evidence about that action."""
        teacher = _response(tool_name="Read", tool_input={"file_path": "a.py"})
        response = _multi_tool_response(
            ("Read", {"file_path": "b.py"}),
            ("Edit", {"file_path": "a.py"}),  # missing required strings
        )
        outcome = ReplayOutcome("local_27b", "OK", response, 1.0)
        label, _, components = labels.action_equivalence(
            _read_edit_schema_request(), teacher, outcome
        )
        self.assertEqual(label, "UNKNOWN")
        self.assertTrue(components.replay_schema_valid)

    def test_no_tool_call_when_teacher_used_one_is_not_equivalent(self):
        teacher = _response(tool_name="Read", tool_input={"file_path": "a.py"})
        outcome = ReplayOutcome("local_27b", "OK", _response(), 1.0)
        label, detail, _ = labels.action_equivalence(EMPTY_REQUEST, teacher, outcome)
        self.assertEqual(label, "NOT_EQUIVALENT")
        self.assertIn("no tool call", detail)

    def test_execution_error_replay_is_execution_error_not_unknown(self):
        teacher = _response(tool_name="Read", tool_input={"file_path": "a.py"})
        outcome = ReplayOutcome("local_27b", "TRANSPORT_ERROR", None, None, detail="boom")
        label, detail, _ = labels.action_equivalence(EMPTY_REQUEST, teacher, outcome)
        self.assertEqual(label, "EXECUTION_ERROR")
        self.assertIn("boom", detail)


class SchemaValidityTests(unittest.TestCase):
    def test_valid_tool_input_scores_true(self):
        outcome = ReplayOutcome("local_27b", "OK", _response(tool_name="Read", tool_input={"file_path": "a.py"}), 1.0)
        self.assertTrue(labels.schema_validity(_read_schema_request(), outcome))

    def test_invalid_tool_input_scores_false(self):
        outcome = ReplayOutcome("local_27b", "OK", _response(tool_name="Read", tool_input={}), 1.0)
        self.assertFalse(labels.schema_validity(_read_schema_request(), outcome))

    def test_no_tool_calls_is_none_not_true(self):
        outcome = ReplayOutcome("local_27b", "OK", _response(), 1.0)
        self.assertIsNone(labels.schema_validity(EMPTY_REQUEST, outcome))

    def test_execution_error_is_none(self):
        outcome = ReplayOutcome("local_27b", "TRANSPORT_ERROR", None, None)
        self.assertIsNone(labels.schema_validity(EMPTY_REQUEST, outcome))

    def test_only_first_tool_call_is_in_scope(self):
        response = _multi_tool_response(
            ("Read", {"file_path": "a.py"}),
            ("Edit", {"file_path": "a.py"}),  # invalid, but not first action
        )
        outcome = ReplayOutcome("local_27b", "OK", response, 1.0)
        self.assertTrue(labels.schema_validity(_read_edit_schema_request(), outcome))


class DatasetTests(unittest.TestCase):
    def test_fingerprint_stable_for_identical_requests(self):
        req = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}], "tools": []}
        self.assertEqual(dataset.request_fingerprint(req), dataset.request_fingerprint(dict(req)))

    def test_fingerprint_differs_for_different_messages(self):
        req1 = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}], "tools": []}
        req2 = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "bye"}], "tools": []}
        self.assertNotEqual(dataset.request_fingerprint(req1), dataset.request_fingerprint(req2))

    def _call(self, fp_suffix, group="g1"):
        req = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": fp_suffix}], "tools": []}
        return TeacherCall(
            call_id=f"{dataset.request_fingerprint(req)}:0",
            task_group=group,
            source_campaign="c",
            source_trace_path="p",
            record_index=0,
            request=req,
            teacher_response={"content": []},
            teacher_placement="cloud",
        )

    def test_deduplicate_drops_repeated_fingerprint(self):
        c1 = self._call("same")
        c2 = self._call("same")  # identical request -> identical fingerprint
        c3 = self._call("different")
        kept, dropped = dataset.deduplicate([c1, c2, c3])
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 1)

    def test_stratified_sample_covers_every_group_before_repeating(self):
        from experiments.capability_router.config import ExperimentConfig

        grouped = {
            "g1": [self._call(f"g1-{i}", "g1") for i in range(5)],
            "g2": [self._call(f"g2-{i}", "g2") for i in range(1)],
            "g3": [self._call(f"g3-{i}", "g3") for i in range(5)],
        }
        cfg = ExperimentConfig(seed=1, target_examples=3, max_calls_per_task_group=5)
        selected = dataset.stratified_sample(grouped, cfg)
        groups_hit = {c.task_group for c in selected}
        self.assertEqual(len(selected), 3)
        self.assertEqual(groups_hit, {"g1", "g2", "g3"})

    def test_stratified_sample_is_deterministic_for_a_fixed_seed(self):
        from experiments.capability_router.config import ExperimentConfig

        grouped = {"g1": [self._call(f"g1-{i}", "g1") for i in range(10)]}
        cfg = ExperimentConfig(seed=42, target_examples=4, max_calls_per_task_group=10)
        first = [c.call_id for c in dataset.stratified_sample(grouped, cfg)]
        second = [c.call_id for c in dataset.stratified_sample(grouped, cfg)]
        self.assertEqual(first, second)

    def test_excluded_result_dirs_is_empty_by_default(self):
        """Correction 2026-09-09: campaign name alone is not a valid
        exclusion criterion for an already-completed, passing cloud-only
        verdict."""
        from experiments.capability_router.config import EXCLUDED_RESULT_DIRS

        self.assertEqual(EXCLUDED_RESULT_DIRS, frozenset())


class SplitsTests(unittest.TestCase):
    def test_groups_are_disjoint_across_partitions(self):
        groups = [f"g{i}" for i in range(10)]
        split = splits.make_split(groups, seed=1, train_frac=0.6, val_frac=0.2)
        all_assigned = set(split.train_groups) | set(split.val_groups) | set(split.test_groups)
        self.assertEqual(all_assigned, set(groups))
        self.assertEqual(set(split.train_groups) & set(split.val_groups), set())
        self.assertEqual(set(split.train_groups) & set(split.test_groups), set())
        self.assertEqual(set(split.val_groups) & set(split.test_groups), set())

    def test_split_is_deterministic_for_a_fixed_seed(self):
        groups = [f"g{i}" for i in range(10)]
        a = splits.make_split(groups, seed=7, train_frac=0.6, val_frac=0.2)
        b = splits.make_split(groups, seed=7, train_frac=0.6, val_frac=0.2)
        self.assertEqual(a, b)

    def test_save_and_load_round_trip(self):
        import tempfile
        from pathlib import Path

        split = splits.make_split(["a", "b", "c"], seed=1, train_frac=0.34, val_frac=0.33)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "split.json"
            splits.save_split(split, path)
            loaded = splits.load_split(path)
        self.assertEqual(split, loaded)

    def test_repo_of_strips_instance_suffix(self):
        self.assertEqual(splits.repo_of("swebench-openlibrary-c05ccf2c"), "openlibrary")
        self.assertEqual(splits.repo_of("swebench-ansible-39bd8b99"), "ansible")


class ModelFeatureTests(unittest.TestCase):
    def test_missing_feature_uses_sentinel_not_zero(self):
        self.assertEqual(models.feature_value({}, "n_tools"), -1.0)

    def test_bool_feature_converts_to_float(self):
        self.assertEqual(models.feature_value({"is_tool_continuation": True}, "is_tool_continuation"), 1.0)
        self.assertEqual(models.feature_value({"is_tool_continuation": False}, "is_tool_continuation"), 0.0)

    def test_build_feature_matrix_excludes_unknown_and_execution_error(self):
        def ex(label):
            return LabeledExample(
                call=None, features={"n_tools": 1}, local_outcome=None, cloud_outcome=None,
                local_label=label, cloud_label="EQUIVALENT",
                local_schema_valid=True, cloud_schema_valid=True,
                local_label_detail="", cloud_label_detail="",
            )

        examples = [ex("EQUIVALENT"), ex("NOT_EQUIVALENT"), ex("UNKNOWN"), ex("EXECUTION_ERROR")]
        X, y, kept = models.build_feature_matrix(examples, "local_label")
        self.assertEqual(len(kept), 2)
        self.assertEqual(list(y), [1, 0])


class AnalysisScoringTests(unittest.TestCase):
    def _ex(self, local_label, cloud_label="EQUIVALENT", group="g"):
        return LabeledExample(
            call=type("C", (), {"task_group": group})(),
            features={}, local_outcome=None, cloud_outcome=None,
            local_label=local_label, cloud_label=cloud_label,
            local_schema_valid=None, cloud_schema_valid=None,
            local_label_detail="", cloud_label_detail="",
        )

    def test_score_arm_local_share_and_quality_loss(self):
        examples = [self._ex("EQUIVALENT"), self._ex("NOT_EQUIVALENT"), self._ex("UNKNOWN")]
        result = analysis.score_arm("test", examples, [True, True, True], None)
        self.assertEqual(result.total_count, 3)
        self.assertEqual(result.local_count, 3)
        self.assertAlmostEqual(result.local_share, 1.0)
        self.assertEqual(result.known_local_count, 2)
        self.assertEqual(result.violation_count, 1)
        self.assertAlmostEqual(result.quality_loss, 0.5)

    def test_score_arm_no_known_local_calls_has_none_quality_loss(self):
        examples = [self._ex("UNKNOWN")]
        result = analysis.score_arm("test", examples, [True], None)
        self.assertIsNone(result.quality_loss)

    def _ex_usage(self, local_usage=None, cloud_usage=None):
        lo = ReplayOutcome("local", "OK", {"_usage": local_usage} if local_usage is not None else None, 1.0)
        co = ReplayOutcome("cloud", "OK", {"_usage": cloud_usage} if cloud_usage is not None else None, 1.0)
        return LabeledExample(
            call=type("C", (), {"task_group": "g"})(), features={},
            local_outcome=lo, cloud_outcome=co,
            local_label="EQUIVALENT", cloud_label="EQUIVALENT",
            local_schema_valid=None, cloud_schema_valid=None,
            local_label_detail="", cloud_label_detail="",
        )

    def test_token_totals_local_input_sums_all_three_vllm_buckets(self):
        # vLLM reports rendered prompt as uncached + cache-read + cache-creation.
        ex = self._ex_usage(local_usage={
            "input_tokens": 1436, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 72128, "output_tokens": 322,
        })
        t = analysis._token_totals([ex], [True])
        self.assertEqual(t["local_input_tokens"], 1436 + 0 + 72128)
        self.assertEqual(t["local_input_uncached_tokens"], 1436)
        self.assertEqual(t["local_input_cache_creation_tokens"], 72128)
        self.assertEqual(t["local_output_tokens"], 322)

    def test_token_totals_cloud_zero_input_no_cache_is_none_not_zero(self):
        # DeepSeek/Lumid returns input_tokens:0 and no cache buckets -> unavailable.
        ex = self._ex_usage(cloud_usage={"input_tokens": 0, "output_tokens": 175})
        t = analysis._token_totals([ex], [False])
        self.assertIsNone(t["cloud_input_tokens"])
        self.assertIsNotNone(t["cloud_input_tokens_note"])
        self.assertEqual(t["cloud_output_tokens"], 175)

    def test_token_totals_cloud_input_reported_when_present(self):
        ex = self._ex_usage(cloud_usage={"input_tokens": 500, "output_tokens": 20})
        t = analysis._token_totals([ex], [False])
        self.assertEqual(t["cloud_input_tokens"], 500)
        self.assertIsNone(t["cloud_input_tokens_note"])

    def test_epsilon_constrained_threshold_picks_max_share_within_q_router_budget(self):
        # Fake model: predict_proba returns the feature's "n_tools" value
        # directly (0..1-ish sentinel doubling as a fake probability) so the
        # threshold sweep is fully deterministic without a real classifier.
        def ex(local_label, cloud_label, proba_value):
            return LabeledExample(
                call=None, features={"n_tools": proba_value}, local_outcome=None, cloud_outcome=None,
                local_label=local_label, cloud_label=cloud_label,
                local_schema_valid=None, cloud_schema_valid=None,
                local_label_detail="", cloud_label_detail="",
            )

        val = [
            ex("EQUIVALENT", "EQUIVALENT", 0.9),
            ex("EQUIVALENT", "EQUIVALENT", 0.8),
            ex("NOT_EQUIVALENT", "EQUIVALENT", 0.1),
        ]

        class FakeModel:
            name = "fake"

            def predict_proba(self, X):
                import numpy as np
                idx = models.FEATURE_NAMES.index("n_tools")
                return X[:, idx]

        t = analysis.epsilon_constrained_threshold(
            FakeModel(), val, (0.0, 0.15, 0.85, 0.95), q_cloud=1.0, epsilon=0.34
        )
        # t=0.0: proba>=0.0 routes all 3 local -> Q_router=2/3 (idx2 fails), 1.0-0.667=0.333<=0.34 -> within budget, share=1.0
        # t=0.15: routes idx0,idx1 only (0.1 < 0.15) -> Q_router=1.0, share=2/3
        # t=0.85: routes idx0 only (0.8 < 0.85) -> Q_router=1.0, share=1/3
        # t=0.95: routes nothing -> Q_router undefined (None), excluded
        # Largest share within budget is t=0.0's share=1.0.
        self.assertEqual(t, 0.0)

    def test_opportunity_2x2_counts_four_quadrants(self):
        examples = [
            self._ex("EQUIVALENT", "EQUIVALENT"),   # both succeed
            self._ex("EQUIVALENT", "NOT_EQUIVALENT"),  # local only
            self._ex("NOT_EQUIVALENT", "EQUIVALENT"),  # cloud only
            self._ex("NOT_EQUIVALENT", "NOT_EQUIVALENT"),  # neither
            self._ex("UNKNOWN", "EQUIVALENT"),  # excluded: local unknown
        ]
        result = analysis.opportunity_2x2(examples)
        self.assertEqual(result["both"], 1)
        self.assertEqual(result["local_only"], 1)
        self.assertEqual(result["cloud_only"], 1)
        self.assertEqual(result["neither"], 1)
        self.assertEqual(result["excluded_unknown"], 1)

    def test_conditional_probability_local_given_cloud_success(self):
        examples = [
            self._ex("EQUIVALENT", "EQUIVALENT"),
            self._ex("NOT_EQUIVALENT", "EQUIVALENT"),
            self._ex("EQUIVALENT", "NOT_EQUIVALENT"),
        ]
        p = analysis.p_local_success_given_cloud_success(examples)
        self.assertAlmostEqual(p, 0.5)  # 1 of 2 cloud-success rows also had local success

    def test_q_router_uses_chosen_backend_label(self):
        examples = [
            self._ex("EQUIVALENT", "NOT_EQUIVALENT"),  # routed local -> local_label used -> success
            self._ex("NOT_EQUIVALENT", "EQUIVALENT"),  # routed cloud -> cloud_label used -> success
        ]
        route_local = [True, False]
        q = analysis.q_router(examples, route_local)
        self.assertAlmostEqual(q.success_rate, 1.0)
        self.assertEqual(q.known_count, 2)

    def test_router_capture_matches_definition(self):
        # both-success cases: idx0 (both eq), idx1 (both eq); router routes idx0 local (captured), idx1 cloud (not captured)
        # oracle local opportunities = count of local_label == EQUIVALENT = idx0, idx1, idx2 = 3
        examples = [
            self._ex("EQUIVALENT", "EQUIVALENT"),
            self._ex("EQUIVALENT", "EQUIVALENT"),
            self._ex("EQUIVALENT", "NOT_EQUIVALENT"),
        ]
        route_local = [True, False, False]
        capture = analysis.router_capture(examples, route_local)
        self.assertAlmostEqual(capture, 1 / 3)


class ExecutorIdentityTests(unittest.TestCase):
    """No live network: `_stream_post` is monkeypatched with canned raw
    results, so these test only the strict-identity and outcome-building
    logic itself."""

    def _backend(self, **overrides):
        base = dict(
            name="local_27b", base_url="http://127.0.0.1:9", request_model="local",
            expected_model_exact="local", auth_header=None, auth_env_var=None,
        )
        base.update(overrides)
        return BackendConfig(**base)

    def _raw(self, *, status_code=200, model="local", transport_err=None, ttft_s=0.1, output_tokens=5):
        events = [
            {"type": "message_start", "message": {"model": model, "usage": {"input_tokens": 10, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": output_tokens}},
        ]
        return {
            "status_code": status_code, "transport_err": transport_err, "raw_error_body": None,
            "events": events if status_code == 200 and transport_err is None else [],
            "t0": 0.0, "t_end": 1.0, "ttft_s": ttft_s, "last_delta_t": 0.5,
        }

    def test_identity_ok_for_exact_match(self):
        ok, _ = executor._identity_ok(self._backend(), {"model": "local"})
        self.assertTrue(ok)

    def test_identity_fails_for_substring_only_match(self):
        """Exact allowlist, not substring: "local-ish" must fail even
        though it contains "local"."""
        ok, detail = executor._identity_ok(self._backend(), {"model": "local-ish"})
        self.assertFalse(ok)

    def test_identity_fails_for_missing_model_field(self):
        ok, detail = executor._identity_ok(self._backend(), {})
        self.assertFalse(ok)
        self.assertIn("no model field", detail)

    def test_cloud_contamination_rejected_by_exact_allowlist(self):
        cloud = self._backend(name="cloud_deepseek", expected_model_exact="deepseek-v4-flash")
        ok, _ = executor._identity_ok(cloud, {"model": "local"})
        self.assertFalse(ok)

    def test_cloud_accepts_genuine_deepseek_identity(self):
        cloud = self._backend(name="cloud_deepseek", expected_model_exact="deepseek-v4-flash")
        ok, _ = executor._identity_ok(cloud, {"model": "deepseek-v4-flash"})
        self.assertTrue(ok)

    def test_preflight_raises_on_identity_mismatch(self):
        backend = self._backend()
        with patch.object(executor, "_stream_post", return_value=self._raw(model="wrong-backend")):
            with self.assertRaises(executor.BackendIdentityError):
                executor.preflight(backend)

    def test_preflight_raises_on_transport_error(self):
        backend = self._backend()
        with patch.object(executor, "_stream_post", return_value=self._raw(status_code=None, transport_err="refused")):
            with self.assertRaises(executor.BackendIdentityError):
                executor.preflight(backend)

    def test_preflight_passes_on_genuine_identity(self):
        backend = self._backend()
        with patch.object(executor, "_stream_post", return_value=self._raw(model="local")):
            evidence = executor.preflight(backend)
        self.assertEqual(evidence["status"], "OK")

    def test_replay_call_marks_identity_mismatch_not_ok(self):
        backend = self._backend()
        call = TeacherCall(
            call_id="x:0", task_group="g", source_campaign="c", source_trace_path="p",
            record_index=0, request={"model": "local", "messages": []},
            teacher_response={"content": []}, teacher_placement="cloud",
        )
        with patch.object(executor, "_stream_post", return_value=self._raw(model="some-other-model")):
            outcome, _ = executor.replay_call(call, backend)
        self.assertEqual(outcome.status, "IDENTITY_MISMATCH")

    def test_replay_call_transport_error_is_recorded_not_raised(self):
        backend = self._backend()
        call = TeacherCall(
            call_id="x:0", task_group="g", source_campaign="c", source_trace_path="p",
            record_index=0, request={"model": "local", "messages": []},
            teacher_response={"content": []}, teacher_placement="cloud",
        )
        with patch.object(executor, "_stream_post", return_value=self._raw(status_code=None, transport_err="timeout")):
            outcome, _ = executor.replay_call(call, backend)
        self.assertEqual(outcome.status, "TRANSPORT_ERROR")
        self.assertIsNone(outcome.response)

    def test_replay_call_records_ttft_and_usage(self):
        backend = self._backend()
        call = TeacherCall(
            call_id="x:0", task_group="g", source_campaign="c", source_trace_path="p",
            record_index=0, request={"model": "local", "messages": []},
            teacher_response={"content": []}, teacher_placement="cloud",
        )
        with patch.object(executor, "_stream_post", return_value=self._raw(ttft_s=0.25, output_tokens=8)):
            outcome, _ = executor.replay_call(call, backend)
        self.assertEqual(outcome.status, "OK")
        self.assertEqual(outcome.response["_timing"]["ttft_s"], 0.25)
        self.assertEqual(outcome.response["_usage"]["output_tokens"], 8)

    def test_replay_call_applies_local_generation_controls_only_for_local(self):
        backend = self._backend(apply_local_generation_controls=True)
        call = TeacherCall(
            call_id="x:0", task_group="g", source_campaign="c", source_trace_path="p",
            record_index=0,
            request={"model": "local", "messages": [], "output_config": {"effort": "high"}},
            teacher_response={"content": []}, teacher_placement="cloud",
        )
        captured_payload = {}

        def fake_stream_post(be, payload):
            captured_payload.update(payload)
            return self._raw()

        with patch.object(executor, "_stream_post", side_effect=fake_stream_post):
            executor.replay_call(call, backend)
        self.assertEqual(captured_payload["output_config"]["effort"], "medium")


class CheckpointTests(unittest.TestCase):
    def test_append_then_load_round_trips(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.jsonl"
            executor.append_checkpoint(path, {"call_id": "a", "backend": "local_27b", "status": "OK"})
            executor.append_checkpoint(path, {"call_id": "a", "backend": "cloud_deepseek", "status": "OK"})
            loaded = executor.load_checkpoint(path)
        self.assertIn(executor.checkpoint_key("a", "local_27b"), loaded)
        self.assertIn(executor.checkpoint_key("a", "cloud_deepseek"), loaded)

    def test_load_checkpoint_missing_file_is_empty(self):
        from pathlib import Path

        self.assertEqual(executor.load_checkpoint(Path("/nonexistent/path.jsonl")), {})

    def test_load_checkpoint_skips_malformed_trailing_line(self):
        """Append safety: a process killed mid-write can leave the final
        line truncated -- that must not raise and must not lose the rows
        written before it."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.jsonl"
            executor.append_checkpoint(path, {"call_id": "a", "backend": "local_27b", "status": "OK"})
            executor.append_checkpoint(path, {"call_id": "b", "backend": "local_27b", "status": "OK"})
            with path.open("a") as fh:
                fh.write('{"call_id": "c", "backend": "local_27b", "stat')  # truncated, no trailing newline
            loaded = executor.load_checkpoint(path)
        self.assertIn(executor.checkpoint_key("a", "local_27b"), loaded)
        self.assertIn(executor.checkpoint_key("b", "local_27b"), loaded)
        self.assertNotIn(executor.checkpoint_key("c", "local_27b"), loaded)
        self.assertEqual(len(loaded), 2)

    def test_load_checkpoint_skips_row_missing_required_keys(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.jsonl"
            with path.open("w") as fh:
                fh.write('{"status": "OK"}\n')  # missing call_id/backend
            loaded = executor.load_checkpoint(path)
        self.assertEqual(loaded, {})

    def test_append_checkpoint_preserves_earlier_rows_across_many_appends(self):
        """A run that appends many rows over time must never lose or
        reorder an earlier one -- simulates the real full-batch loop's
        append pattern at small scale."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.jsonl"
            for i in range(50):
                executor.append_checkpoint(path, {"call_id": f"c{i}", "backend": "local_27b", "status": "OK", "seq": i})
            loaded = executor.load_checkpoint(path)
        self.assertEqual(len(loaded), 50)
        for i in range(50):
            self.assertEqual(loaded[executor.checkpoint_key(f"c{i}", "local_27b")]["seq"], i)


class ValidateResumeCheckpointTests(unittest.TestCase):
    def _call(self, call_id, group="g"):
        return TeacherCall(
            call_id=call_id, task_group=group, source_campaign="c", source_trace_path="p",
            record_index=0, request={"model": "claude-sonnet-5", "messages": []},
            teacher_response={"content": []}, teacher_placement="cloud",
        )

    def test_empty_checkpoint_always_valid(self):
        executor.validate_resume_checkpoint([self._call("a")], {})  # must not raise

    def test_matching_call_ids_pass(self):
        calls = [self._call("a"), self._call("b")]
        checkpoint = {
            executor.checkpoint_key("a", "cloud_deepseek"): {"call_id": "a", "backend": "cloud_deepseek"},
        }
        executor.validate_resume_checkpoint(calls, checkpoint)  # must not raise

    def test_unknown_call_id_in_checkpoint_is_rejected(self):
        """Resume/fingerprint mismatch rejection: a checkpoint built from a
        different dataset must not be silently trusted."""
        calls = [self._call("a")]
        checkpoint = {
            executor.checkpoint_key("stale-fingerprint-from-old-dataset", "cloud_deepseek"): {
                "call_id": "stale-fingerprint-from-old-dataset", "backend": "cloud_deepseek",
            },
        }
        with self.assertRaises(executor.BackendIdentityError):
            executor.validate_resume_checkpoint(calls, checkpoint)


class RetryableCheckpointTests(unittest.TestCase):
    def test_ok_rows_are_not_retryable(self):
        cp = {"k": {"call_id": "a", "backend": "cloud_deepseek", "status": "OK"}}
        self.assertEqual(executor.retryable_checkpoint(cp), cp)

    def test_invalid_capacity_rows_are_not_retryable(self):
        cp = {"k": {"call_id": "a", "backend": "local_7b", "status": "INVALID_CAPACITY"}}
        self.assertEqual(executor.retryable_checkpoint(cp), cp)

    def test_http_error_rows_are_not_retryable(self):
        cp = {"k": {"call_id": "a", "backend": "local_27b", "status": "HTTP_ERROR"}}
        self.assertEqual(executor.retryable_checkpoint(cp), cp)

    def test_transport_error_rows_are_excluded_so_they_get_retried(self):
        cp = {"k": {"call_id": "a", "backend": "local_27b", "status": "TRANSPORT_ERROR"}}
        self.assertEqual(executor.retryable_checkpoint(cp), {})

    def test_mixed_checkpoint_keeps_only_terminal_rows(self):
        cp = {
            "ok": {"call_id": "a", "backend": "cloud_deepseek", "status": "OK"},
            "err": {"call_id": "b", "backend": "local_27b", "status": "TRANSPORT_ERROR"},
            "cap": {"call_id": "c", "backend": "local_7b", "status": "INVALID_CAPACITY"},
        }
        result = executor.retryable_checkpoint(cp)
        self.assertEqual(set(result), {"ok", "cap"})


class CallMajorConcurrencyTests(unittest.TestCase):
    """Proves the actual production `run_full_replay` path is call-major
    (cloud + every local backend for ONE call submitted together) and
    genuinely concurrent (they overlap on the network, not run one after
    another) -- not just that the checkpoint prevents duplicate work."""

    def _call(self, call_id="a"):
        return TeacherCall(
            call_id=call_id, task_group="g", source_campaign="c", source_trace_path="p",
            record_index=0, request={"model": "claude-sonnet-5", "messages": []},
            teacher_response={"content": []}, teacher_placement="cloud",
        )

    def test_three_backends_for_one_call_run_concurrently_not_sequentially(self):
        """Timing-overlap proof: three fake backends each "take" 0.3s. If
        `_replay_call_all_backends` actually dispatches them concurrently,
        wall time is close to 0.3s. A sequential implementation (the bug
        just found in production) would take close to 0.9s. This directly
        exercises `run_full_replay`'s real per-call dispatcher, not a
        reimplementation of it."""
        import time

        from experiments.capability_router.config import CLOUD_DEEPSEEK, LOCAL_7B, LOCAL_27B
        from experiments.capability_router.orchestrator import _replay_call_all_backends

        call = self._call()
        SLEEP_S = 0.3
        concurrent_peak = {"count": 0, "max": 0}
        peak_lock = threading.Lock()

        def fake_replay_call(c, backend):
            with peak_lock:
                concurrent_peak["count"] += 1
                concurrent_peak["max"] = max(concurrent_peak["max"], concurrent_peak["count"])
            time.sleep(SLEEP_S)
            with peak_lock:
                concurrent_peak["count"] -= 1
            outcome = ReplayOutcome(backend=backend.name, status="OK", response={"content": [], "model": "x"}, latency_s=SLEEP_S)
            return outcome, {}

        checkpoint: dict = {}
        checkpoint_lock = threading.Lock()
        with tempfile_path() as checkpoint_path:
            with patch("experiments.capability_router.orchestrator.executor.replay_call", side_effect=fake_replay_call):
                t0 = time.monotonic()
                examples = _replay_call_all_backends(
                    call, [LOCAL_27B, LOCAL_7B], checkpoint_path, checkpoint, checkpoint_lock
                )
                elapsed = time.monotonic() - t0

        # 3 backends (cloud, local_27b, local_7b) all "in flight" at once.
        self.assertEqual(concurrent_peak["max"], 3, "all 3 backends must have been in flight at the same time")
        # Generous bound: concurrent -> ~SLEEP_S; sequential would be ~3x that.
        self.assertLess(elapsed, SLEEP_S * 2, f"took {elapsed:.2f}s -- looks sequential, not concurrent")
        self.assertEqual(set(examples.keys()), {LOCAL_27B.name, LOCAL_7B.name})

    def test_second_local_backend_does_not_replay_cloud_again(self):
        """Same call, two _replay_call_all_backends invocations (simulating
        two separate calls to the dispatcher sharing one checkpoint) --
        cloud must only ever be replayed once."""
        from experiments.capability_router.config import CLOUD_DEEPSEEK, LOCAL_7B, LOCAL_27B
        from experiments.capability_router.orchestrator import _replay_call_all_backends

        call = self._call()
        replay_calls = []
        lock = threading.Lock()

        def fake_replay_call(c, backend):
            with lock:
                replay_calls.append(backend.name)
            outcome = ReplayOutcome(backend=backend.name, status="OK", response={"content": [], "model": "x"}, latency_s=0.01)
            return outcome, {}

        checkpoint: dict = {}
        checkpoint_lock = threading.Lock()
        with tempfile_path() as checkpoint_path:
            with patch("experiments.capability_router.orchestrator.executor.replay_call", side_effect=fake_replay_call):
                _replay_call_all_backends(call, [LOCAL_27B], checkpoint_path, checkpoint, checkpoint_lock)
                _replay_call_all_backends(call, [LOCAL_7B], checkpoint_path, checkpoint, checkpoint_lock)

            cloud_calls = [b for b in replay_calls if b == CLOUD_DEEPSEEK.name]
            self.assertEqual(len(cloud_calls), 1, "cloud must only be replayed once across both dispatches")

            loaded = executor.load_checkpoint(checkpoint_path)
            cloud_rows_on_disk = [
                r for r in loaded.values() if r["backend"] == CLOUD_DEEPSEEK.name and r["call_id"] == call.call_id
            ]
            self.assertEqual(len(cloud_rows_on_disk), 1, "checkpoint file must contain exactly one cloud row for this call")

    def test_resuming_from_an_on_disk_checkpoint_skips_already_done_pairs(self):
        """A fresh process loading a checkpoint file written by an earlier
        (killed) process must not re-replay pairs already recorded there."""
        from experiments.capability_router.config import CLOUD_DEEPSEEK, LOCAL_27B
        from experiments.capability_router.orchestrator import _replay_call_all_backends

        call = self._call()
        with tempfile_path() as checkpoint_path:
            # Simulate a prior process having already completed the cloud call.
            executor.append_checkpoint(checkpoint_path, {
                "call_id": call.call_id, "backend": CLOUD_DEEPSEEK.name, "status": "OK",
                "response": {"content": [], "model": "deepseek-v4-flash"},
            })

            replay_calls = []

            def fake_replay_call(c, backend):
                replay_calls.append(backend.name)
                outcome = ReplayOutcome(backend=backend.name, status="OK", response={"content": [], "model": "x"}, latency_s=0.01)
                return outcome, {}

            checkpoint = executor.load_checkpoint(checkpoint_path)
            checkpoint_lock = threading.Lock()
            with patch("experiments.capability_router.orchestrator.executor.replay_call", side_effect=fake_replay_call):
                _replay_call_all_backends(call, [LOCAL_27B], checkpoint_path, checkpoint, checkpoint_lock)

        self.assertNotIn(CLOUD_DEEPSEEK.name, replay_calls, "cloud was already checkpointed and must not be replayed")
        self.assertIn(LOCAL_27B.name, replay_calls)


if __name__ == "__main__":
    unittest.main()
