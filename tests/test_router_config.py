import unittest

from edgeproxy import router
from edgeproxy.config import parse_args


class RouterConfigurationTests(unittest.TestCase):
    def _features(self, **overrides):
        values = {
            "model": "claude-sonnet-5",
            "has_tools": False,
            "n_tools": 0,
            "has_server_tools": False,
            "n_messages": 1,
            "est_system_tokens": 0,
            "max_tokens": 64,
            "stream": False,
            "is_tool_continuation": False,
            "local_prompt_tokens": 100,
        }
        values.update(overrides)
        return router.CallFeatures(**values)

    def test_static_policy_uses_setup_specific_context_limit(self):
        policy = router.build("static", max_local_tokens=100_000, margin=0.9)
        self.assertIsInstance(policy, router.StaticPolicy)
        self.assertEqual(policy.max_local_tokens, 100_000)
        self.assertEqual(policy.budget(), 90_000)

    def test_dynamic_output_cap_uses_exact_prompt_and_full_headroom(self):
        policy = router.StaticPolicy(
            max_local_tokens=100_000,
            margin=0.9,
            output_reserve_tokens=0,
        )
        features = self._features(local_prompt_tokens=70_000, max_tokens=64_000)

        self.assertEqual(policy.effective_max_tokens(features), 20_000)
        self.assertEqual(policy.decide(features).placement, "local")

    def test_dynamic_output_cap_subtracts_explicit_reserve(self):
        policy = router.StaticPolicy(
            max_local_tokens=100_000,
            margin=0.9,
            output_reserve_tokens=512,
        )

        self.assertEqual(
            policy.effective_max_tokens(
                self._features(local_prompt_tokens=70_000, max_tokens=64_000)
            ),
            19_488,
        )

    def test_no_output_headroom_routes_cloud(self):
        policy = router.StaticPolicy(max_local_tokens=100_000, margin=0.9)

        decision = policy.decide(
            self._features(local_prompt_tokens=90_000, max_tokens=64_000)
        )

        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "too-large")

    def test_non_static_policy_does_not_require_capacity_configuration(self):
        policy = router.build("cloud-only", max_local_tokens=100_000, margin=0.9)
        self.assertIsInstance(policy, router.CloudOnly)

    def test_proxy_cli_accepts_setup_specific_capacity(self):
        config = parse_args(
            [
                "--max-local-tokens",
                "100000",
                "--local-token-margin",
                "0.9",
                "--local-output-reserve-tokens",
                "0",
                "--experiment-id",
                "fanout-1",
                "--episode-id",
                "fanout-1-routing",
                "--cohort-tracking",
                "observe",
                "--cohort-window-ms",
                "200",
            ]
        )
        self.assertEqual(config.max_local_tokens, 100_000)
        self.assertEqual(config.local_token_margin, 0.9)
        self.assertEqual(config.local_output_reserve_tokens, 0)
        self.assertEqual(config.experiment_id, "fanout-1")
        self.assertEqual(config.episode_id, "fanout-1-routing")
        self.assertEqual(config.cohort_tracking, "observe")
        self.assertEqual(config.cohort_window_ms, 200)

    def test_proxy_cli_rejects_invalid_margin(self):
        with self.assertRaises(SystemExit):
            parse_args(["--local-token-margin", "1.1"])

    def test_proxy_cli_rejects_negative_output_reserve(self):
        with self.assertRaises(SystemExit):
            parse_args(["--local-output-reserve-tokens", "-1"])

    def test_proxy_cli_rejects_negative_cohort_window(self):
        with self.assertRaises(SystemExit):
            parse_args(["--cohort-window-ms", "-1"])

    def test_security_monitor_feature_is_detected_from_system_blocks(self):
        features = router.extract_features(
            {
                "model": "claude-sonnet-5",
                "max_tokens": 64,
                "system": [
                    {
                        "type": "text",
                        "text": "You are a security monitor for autonomous agents.",
                    }
                ],
            }
        )
        self.assertTrue(features.is_security_monitor)

    def test_security_monitor_routes_cloud_before_local_feasibility(self):
        decision = router.StaticPolicy().decide(
            self._features(is_security_monitor=True, local_prompt_tokens=None)
        )
        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "security-monitor-cloud")

    def test_other_feasible_calls_remain_local(self):
        decision = router.StaticPolicy().decide(self._features())
        self.assertEqual(decision.placement, "local")
        self.assertEqual(decision.reason, "fits")


if __name__ == "__main__":
    unittest.main()
