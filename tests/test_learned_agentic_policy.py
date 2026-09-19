import unittest

from edgeproxy import router


class LearnedAgenticPolicyTests(unittest.TestCase):
    def _features(self, **overrides):
        values = {
            "model": "claude-sonnet-5",
            "has_tools": False,
            "n_tools": 0,
            "has_agent_tool": False,
            "has_server_tools": False,
            "n_messages": 1,
            "est_system_tokens": 0,
            "max_tokens": 64,
            "stream": False,
            "is_tool_continuation": True,
            "local_prompt_tokens": 100,
        }
        values.update(overrides)
        return router.CallFeatures(**values)

    def test_registered_policy_falls_back_safely_without_scorer(self):
        policy = router.build("learned-agentic")

        self.assertIsInstance(policy, router.LearnedAgenticPolicy)
        decision = policy.decide(self._features())
        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "learned-scorer-unavailable")

    def test_scorer_exception_falls_back_safely(self):
        def failing_scorer(features):
            raise RuntimeError("model unavailable")

        decision = router.LearnedAgenticPolicy(scorer=failing_scorer).decide(
            self._features()
        )

        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "learned-scorer-error")

    def test_static_hard_gates_run_before_scorer(self):
        calls = []

        def scorer(features):
            calls.append(features)
            return 1.0

        policy = router.LearnedAgenticPolicy(
            scorer=scorer,
            local_can_tool_call=False,
            max_local_tokens=100_000,
            margin=0.9,
        )
        cases = [
            ("security-monitor-cloud", {"is_security_monitor": True}),
            ("server-side-tool", {"has_server_tools": True}),
            (
                "local-probe-unavailable",
                {"local_cache_prediction_confidence": "unavailable"},
            ),
            ("local-token-count-unavailable", {"local_prompt_tokens": None}),
            ("too-large", {"local_prompt_tokens": 90_000}),
            ("tools-unsupported", {"has_tools": True, "n_tools": 1}),
            ("reliability-circuit-open", {"local_reliability_blocked": True}),
        ]

        for reason, overrides in cases:
            with self.subTest(reason=reason):
                decision = policy.decide(self._features(**overrides))
                self.assertEqual(decision.placement, "cloud")
                self.assertEqual(decision.reason, reason)

        self.assertEqual(calls, [])

    def test_score_at_or_above_threshold_selects_local(self):
        features = self._features(n_messages=7)
        seen = []

        def scorer(received):
            seen.append(received)
            return 0.8

        decision = router.LearnedAgenticPolicy(
            scorer=scorer, threshold=0.8
        ).decide(features)

        self.assertEqual(seen, [features])
        self.assertEqual(decision.placement, "local")
        self.assertEqual(decision.reason, "learned-score-local")

    def test_score_below_threshold_selects_cloud(self):
        decision = router.LearnedAgenticPolicy(
            scorer=lambda features: 0.79, threshold=0.8
        ).decide(self._features())

        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "learned-score-cloud")

    def test_default_heuristic_scores_clean_feasible_call_higher(self):
        clean = self._features(
            branch_turn_ordinal=6,
            errored_tool_result_density=0.0,
            local_prompt_tokens=10_000,
            local_token_budget=90_000,
            predicted_local_risk_score=0.05,
        )
        error_heavy = self._features(
            branch_turn_ordinal=6,
            errored_tool_result_density=0.6,
            local_prompt_tokens=10_000,
            local_token_budget=90_000,
            predicted_local_risk_score=0.05,
        )

        clean_score = router.default_agentic_scorer(clean)
        error_score = router.default_agentic_scorer(error_heavy)

        self.assertGreater(clean_score, error_score)
        self.assertGreaterEqual(
            clean_score, router.DEFAULT_AGENTIC_HEURISTIC_THRESHOLD
        )
        self.assertLess(error_score, router.DEFAULT_AGENTIC_HEURISTIC_THRESHOLD)

    def test_default_heuristic_penalizes_near_budget_headroom(self):
        roomy = self._features(
            branch_turn_ordinal=6,
            local_prompt_tokens=10_000,
            local_token_budget=90_000,
            predicted_local_risk_score=0.05,
        )
        near_budget = self._features(
            branch_turn_ordinal=6,
            local_prompt_tokens=85_000,
            local_token_budget=90_000,
            predicted_local_risk_score=0.05,
        )

        roomy_score = router.default_agentic_scorer(roomy)
        near_budget_score = router.default_agentic_scorer(near_budget)

        self.assertGreater(roomy_score, near_budget_score)
        self.assertLess(
            near_budget_score, router.DEFAULT_AGENTIC_HEURISTIC_THRESHOLD
        )

    def test_registered_heuristic_variant_routes_representative_calls(self):
        policy = router.build(
            "learned-agentic-heuristic",
            max_local_tokens=100_000,
            margin=0.9,
        )

        clean = policy.decide(
            self._features(
                branch_turn_ordinal=6,
                local_prompt_tokens=10_000,
                local_token_budget=90_000,
                predicted_local_risk_score=0.05,
            )
        )
        near_budget = policy.decide(
            self._features(
                branch_turn_ordinal=6,
                local_prompt_tokens=85_000,
                local_token_budget=90_000,
                predicted_local_risk_score=0.05,
            )
        )

        self.assertIsInstance(policy, router.HeuristicAgenticPolicy)
        self.assertEqual(clean.placement, "local")
        self.assertEqual(near_budget.placement, "cloud")


if __name__ == "__main__":
    unittest.main()
