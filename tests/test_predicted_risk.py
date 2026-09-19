import unittest

from edgeproxy import router


class PredictedLocalRiskTests(unittest.TestCase):
    def _features(self, **overrides):
        values = {
            "model": "claude-sonnet-5",
            "has_tools": True,
            "n_tools": 22,
            "has_server_tools": False,
            "n_messages": 10,
            "est_system_tokens": 0,
            "max_tokens": 64_000,
            "stream": True,
            "is_tool_continuation": True,
            "local_prompt_tokens": 20_000,
        }
        values.update(overrides)
        return router.CallFeatures(**values)

    def test_frozen_risk_score_for_known_feature_combinations(self):
        low = router.predict_local_risk_score(self._features())
        high = router.predict_local_risk_score(
            self._features(
                local_prompt_tokens=85_000,
                n_messages=100,
                errored_tool_result_density=0.5,
            )
        )

        self.assertAlmostEqual(low, 0.0016074677236446167)
        self.assertAlmostEqual(high, 0.7494675103211559)

    def test_out_of_domain_call_has_no_score(self):
        self.assertIsNone(
            router.predict_local_risk_score(
                self._features(has_tools=False, n_tools=0)
            )
        )
        self.assertIsNone(
            router.predict_local_risk_score(
                self._features(local_prompt_tokens=None)
            )
        )

    def test_extracts_explicit_tool_error_density(self):
        features = router.extract_features(
            {
                "tools": [{"name": "Read", "input_schema": {}}],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "1", "is_error": True},
                            {"type": "tool_result", "tool_use_id": "2"},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "3", "is_error": False}
                        ],
                    },
                ],
            }
        )

        self.assertAlmostEqual(features.errored_tool_result_density, 1 / 3)

    def test_legacy_score_is_stable_when_budget_is_recorded(self):
        without_budget = router.predict_local_risk_score(self._features())
        with_budget = router.predict_local_risk_score(
            self._features(local_token_budget=54_000)
        )

        self.assertEqual(with_budget, without_budget)

    def test_policy_escalates_at_configured_threshold(self):
        policy = router.PredictedRiskPolicy(risk_threshold=0.4)

        below = policy.decide(self._features(predicted_local_risk_score=0.399))
        at = policy.decide(self._features(predicted_local_risk_score=0.4))

        self.assertEqual((below.placement, below.reason), ("local", "fits"))
        self.assertEqual((at.placement, at.reason), ("cloud", "predicted-local-risk"))

    def test_policy_threshold_is_parameterized_and_validated(self):
        self.assertIsInstance(router.build("predicted-risk"), router.PredictedRiskPolicy)
        with self.assertRaises(ValueError):
            router.PredictedRiskPolicy(risk_threshold=1.01)


if __name__ == "__main__":
    unittest.main()
