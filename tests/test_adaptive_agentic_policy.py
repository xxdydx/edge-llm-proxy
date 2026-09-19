import unittest
from pathlib import Path

from edgeproxy.router import AdaptiveAgenticPolicy, ArtifactHarmScorer, CallFeatures, Decision, HarmEstimate
from experiments.agentic_router.adaptive_policy import Observation, compare, compare_bandits


def features(**overrides):
    values = dict(
        model="test", has_tools=False, n_tools=0, has_server_tools=False,
        n_messages=1, est_system_tokens=0, max_tokens=64, stream=False,
        is_tool_continuation=False, local_prompt_tokens=100,
    )
    values.update(overrides)
    return CallFeatures(**values)


def scorer(risk):
    return lambda _: HarmEstimate(risk, True, "test-v1")


class AdaptivePolicyTests(unittest.TestCase):
    def test_default_is_shadow_and_preserves_static_baseline(self):
        decision = AdaptiveAgenticPolicy().decide(features())
        self.assertEqual((decision.placement, decision.reason), ("local", "adaptive-shadow"))
        self.assertIn("proposal_reason=adaptive-model-unavailable", decision.detail)

    def test_shadow_preserves_injected_baseline(self):
        class Baseline:
            name = "prior"

            def decide(self, f):
                return Decision("cloud", "prior-heuristic")

        decision = AdaptiveAgenticPolicy(
            harm_scorer=scorer(0.01), baseline_policy=Baseline()
        ).decide(features())
        self.assertEqual(decision.placement, "cloud")
        self.assertIn("baseline=prior-heuristic", decision.detail)
        self.assertIn("proposed=local", decision.detail)

    def test_hard_gate_precedes_scorer_even_in_shadow(self):
        calls = []
        policy = AdaptiveAgenticPolicy(harm_scorer=lambda f: calls.append(f))
        decision = policy.decide(features(has_server_tools=True))
        self.assertEqual((decision.placement, decision.reason), ("cloud", "adaptive-shadow"))
        self.assertIn("proposal_reason=server-side-tool", decision.detail)
        self.assertEqual(calls, [])

    def test_immediate_escalation_and_lower_return_threshold(self):
        for previous, risk, expected in [
            ("local", 0.26, "adaptive-quality-escalation"),
            ("cloud", 0.20, "adaptive-return-hysteresis"),
            ("local", 0.20, "adaptive-quality-eligible"),
            ("cloud", 0.10, "adaptive-quality-eligible"),
        ]:
            with self.subTest(previous=previous, risk=risk):
                decision = AdaptiveAgenticPolicy(harm_scorer=scorer(risk), shadow=False).decide(
                    features(previous_backend=previous)
                )
                self.assertEqual(decision.reason, expected)

    def test_independent_trajectory_has_no_retained_backend_state(self):
        policy = AdaptiveAgenticPolicy(harm_scorer=scorer(0.20), shadow=False)
        first = policy.decide(features(previous_backend="local"))
        independent = policy.decide(features(previous_backend=None))
        self.assertEqual(first.placement, "local")
        self.assertEqual(independent.placement, "cloud")

    def test_missing_invalid_or_unsupported_model_fails_closed_when_active(self):
        cases = [
            (None, "adaptive-unsupported"),
            (HarmEstimate(0.0, False, "v1"), "adaptive-unsupported"),
            (HarmEstimate(float("nan"), True, "v1"), "adaptive-invalid-score"),
            (HarmEstimate("not-a-number", True, "v1"), "adaptive-invalid-score"),
        ]
        for estimate, reason in cases:
            with self.subTest(reason=reason):
                result = AdaptiveAgenticPolicy(
                    harm_scorer=lambda _: estimate, shadow=False
                ).decide(features())
                self.assertEqual((result.placement, result.reason), ("cloud", reason))

    def test_measured_serving_gate_cannot_override_quality_escalation(self):
        call = features(expected_local_service_ms=1.0, expected_cloud_service_ms=100.0)
        result = AdaptiveAgenticPolicy(
            harm_scorer=scorer(0.9), shadow=False, max_local_latency_ratio=2.0
        ).decide(call)
        self.assertEqual(result.reason, "adaptive-quality-escalation")
        slow = AdaptiveAgenticPolicy(
            harm_scorer=scorer(0.01), shadow=False, max_local_latency_ratio=2.0
        ).decide(features(expected_local_service_ms=300.0, expected_cloud_service_ms=100.0))
        self.assertEqual(slow.reason, "adaptive-serving-gate")

    def test_current_json_artifact_scores_but_cannot_activate(self):
        path = Path(__file__).resolve().parents[1] / "experiments/agentic_router/results/quality_model_baseline205_v5_shadow_logreg.json"
        import json
        artifact = json.loads(path.read_text())
        names = artifact["feature_names"]
        scorer = ArtifactHarmScorer.from_json(path, lambda _: {name: 0.0 for name in names})
        estimate = scorer(features())
        self.assertGreaterEqual(estimate.probability, 0.0)
        self.assertLessEqual(estimate.probability, 1.0)
        self.assertFalse(estimate.supported)
        active = AdaptiveAgenticPolicy(harm_scorer=scorer, shadow=False).decide(features())
        self.assertEqual((active.placement, active.reason), ("cloud", "adaptive-unsupported"))
        mismatch = ArtifactHarmScorer(artifact, lambda _: {"wrong": 1.0})(features())
        self.assertFalse(mismatch.supported)
        self.assertIn("feature-name-mismatch", mismatch.detail)
        stale = {**artifact, "training_provenance": {"feature_derivation_version": "old"}}
        with self.assertRaisesRegex(ValueError, "derivation version mismatch"):
            ArtifactHarmScorer(stale, lambda _: {})


class OfflineAdaptationTests(unittest.TestCase):
    @staticmethod
    def synthetic_groups(*, drift: bool):
        rows = []
        for group in range(8):
            for index in range(20):
                x = 1.0 if index % 2 else -1.0
                safe = (x < 0) if not drift or group < 4 else (x > 0)
                n = group * 20 + index
                rows.append(Observation(
                    call_id=str(n), trajectory_id=str(group), task_group=str(group),
                    timestamp=float(n), features=(x,),
                    label="SAFE" if safe else "HARM",
                ))
        return rows

    def test_same_future_groups_and_unknown_preserved(self):
        rows = []
        for group in range(4):
            for index, label in enumerate(("SAFE", "HARM", "UNKNOWN")):
                rows.append(Observation(
                    call_id=f"g{group}-{index}", trajectory_id=f"t{group}",
                    task_group=f"g{group}", timestamp=group * 10 + index,
                    features=(float(index), float(group)), label=label,
                ))
        result = compare(rows, initial_groups=2)
        self.assertEqual(result["status"], "diagnostic_proxy_only")
        self.assertEqual(result["future_groups"], ["g2", "g3"])
        for policy in result["summary"].values():
            self.assertEqual(policy["n_calls"], 6)
            self.assertEqual(policy["n_unknown"], 2)
            self.assertEqual(policy["n_known"], 4)
            self.assertEqual(policy["harm_fraction_all_known"], 0.5)

    def test_overlapping_groups_refuse_leaky_comparison(self):
        rows = [
            Observation("a1", "a", "a", 0.0, (0.0,), "SAFE"),
            Observation("b1", "b", "b", 1.0, (0.0,), "SAFE"),
            Observation("a2", "a", "a", 2.0, (0.0,), "HARM"),
        ]
        with self.assertRaisesRegex(ValueError, "overlapping task groups"):
            compare(rows, initial_groups=1)

    def test_synthetic_stationarity_keeps_same_label_rule(self):
        result = compare(self.synthetic_groups(drift=False), initial_groups=2, sliding_groups=2)
        for group in result["future_groups"]:
            for strategy in ("frozen", "sliding", "recency"):
                self.assertEqual(result["per_group"][group][strategy]["harm_fraction_local_known"], 0.0)

    def test_synthetic_drift_window_relearns_only_after_feedback(self):
        result = compare(self.synthetic_groups(drift=True), initial_groups=2, sliding_groups=2)
        # The first drift group is predicted with pre-drift data by *every*
        # strategy. After two fully observed drift groups, the sliding model
        # can reverse; frozen cannot. This is a mechanism test, not efficacy.
        self.assertEqual(result["per_group"]["4"]["sliding"]["harm_fraction_local_known"], 1.0)
        self.assertEqual(result["per_group"]["6"]["sliding"]["harm_fraction_local_known"], 0.0)
        self.assertEqual(result["per_group"]["6"]["frozen"]["harm_fraction_local_known"], 1.0)

    def test_bandit_uses_delayed_chosen_arm_proxy_feedback(self):
        rows = self.synthetic_groups(drift=False)[:8]
        result = compare_bandits(rows, delay_calls=3, thompson_seeds=(0, 1))
        self.assertEqual(result["status"], "offline_selected_arm_proxy_simulation")
        self.assertEqual(result["policies"]["linucb"]["n_updates_released_before_decision"], 5)
        self.assertEqual(result["policies"]["linucb"]["n_updates_pending_after_last_decision"], 3)
        self.assertEqual(result["policies"]["always_cloud"]["mean_proxy_utility_known"], 1.0)
        self.assertEqual(len(result["policies"]["thompson"]), 2)


if __name__ == "__main__":
    unittest.main()
