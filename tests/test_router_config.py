import unittest

from edgeproxy import router
from edgeproxy.config import parse_args


class RouterConfigurationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
