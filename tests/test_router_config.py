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

    def test_non_static_policy_does_not_require_capacity_configuration(self):
        policy = router.build("cloud-only", max_local_tokens=100_000, margin=0.9)
        self.assertIsInstance(policy, router.CloudOnly)

    def test_proxy_cli_accepts_setup_specific_capacity(self):
        config = parse_args(
            ["--max-local-tokens", "100000", "--local-token-margin", "0.9"]
        )
        self.assertEqual(config.max_local_tokens, 100_000)
        self.assertEqual(config.local_token_margin, 0.9)

    def test_proxy_cli_rejects_invalid_margin(self):
        with self.assertRaises(SystemExit):
            parse_args(["--local-token-margin", "1.1"])

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
