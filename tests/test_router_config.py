import unittest

from edgeproxy import router
from edgeproxy.config import parse_args


class RouterConfigurationTests(unittest.TestCase):
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

    def test_explicit_local_output_cap_bounds_context_headroom(self):
        policy = router.StaticPolicy(
            max_local_tokens=100_000,
            margin=0.9,
            max_output_tokens=8_192,
        )

        features = self._features(local_prompt_tokens=20_000, max_tokens=32_000)
        self.assertEqual(policy.effective_max_tokens(features), 8_192)
        self.assertEqual(policy.decide(features).placement, "local")

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

    def test_local_only_forces_local_and_clamps_max_tokens_to_window(self):
        # Regression: a 7B setup (60K window). Claude Code sends max_tokens
        # 64000; unclamped, vLLM 400s ("max_completion_tokens cannot be
        # greater than max_model_len") and every call in the LocalOnly
        # baseline session fails. LocalOnly must force local AND clamp.
        policy = router.build("local-only", max_local_tokens=60_000, margin=0.9)
        self.assertIsInstance(policy, router.LocalOnly)
        self.assertEqual(policy.budget(), 54_000)

        f = self._features(local_prompt_tokens=1_200, max_tokens=64_000)
        self.assertEqual(policy.decide(f).placement, "local")
        # 54000 - 1200 = 52800, well under the 60000 window.
        self.assertEqual(policy.effective_max_tokens(f), 52_800)

        # Probe unavailable: still must stay strictly below the window.
        g = self._features(local_prompt_tokens=None, max_tokens=64_000)
        self.assertEqual(policy.decide(g).placement, "local")
        want = policy.effective_max_tokens(g)
        self.assertLess(want, 60_000)
        self.assertGreaterEqual(want, 1)

        # A modest request is left alone.
        h = self._features(local_prompt_tokens=1_000, max_tokens=4_096)
        self.assertEqual(policy.effective_max_tokens(h), 4_096)

    def test_proxy_cli_accepts_setup_specific_capacity(self):
        config = parse_args(
            [
                "--max-local-tokens",
                "100000",
                "--local-token-margin",
                "0.9",
                "--local-output-reserve-tokens",
                "0",
                "--local-max-output-tokens",
                "8192",
                "--local-thinking",
                "disabled",
                "--local-concurrency-limit",
                "6",
                "--experiment-id",
                "fanout-1",
                "--episode-id",
                "fanout-1-routing",
                "--cohort-tracking",
                "observe",
                "--cohort-window-ms",
                "300",
            ]
        )
        self.assertEqual(config.max_local_tokens, 100_000)
        self.assertEqual(config.local_token_margin, 0.9)
        self.assertEqual(config.local_output_reserve_tokens, 0)
        self.assertEqual(config.local_max_output_tokens, 8_192)
        self.assertEqual(config.local_thinking, "disabled")
        self.assertEqual(config.local_concurrency_limit, 6)
        self.assertEqual(config.experiment_id, "fanout-1")
        self.assertEqual(config.episode_id, "fanout-1-routing")
        self.assertEqual(config.cohort_tracking, "observe")
        self.assertEqual(config.cohort_window_ms, 300)

    def test_proxy_cli_defaults_to_300ms_cohort_window(self):
        self.assertEqual(parse_args([]).cohort_window_ms, 300)

    def test_local_generation_safety_defaults_preserve_historical_behavior(self):
        config = parse_args([])
        self.assertEqual(config.local_max_output_tokens, 0)
        self.assertEqual(config.local_thinking, "preserve")

    def test_proxy_cli_rejects_negative_local_output_cap(self):
        with self.assertRaises(SystemExit):
            parse_args(["--local-max-output-tokens", "-1"])

    def test_proxy_cli_defaults_to_eight_local_concurrent_requests(self):
        self.assertEqual(parse_args([]).local_concurrency_limit, 8)

    def test_proxy_cli_rejects_non_positive_local_concurrency_limit(self):
        with self.assertRaises(SystemExit):
            parse_args(["--local-concurrency-limit", "0"])

    def test_cohort_parent_placement_defaults_off_and_is_explicitly_enabled(self):
        self.assertFalse(parse_args([]).cohort_parent_placement)
        self.assertTrue(
            parse_args(["--cohort-parent-placement"]).cohort_parent_placement
        )

    def test_local_cache_salt_scope_defaults_off_and_accepts_ablation_scopes(self):
        self.assertEqual(parse_args([]).local_cache_salt_scope, "off")
        self.assertEqual(
            parse_args(["--local-cache-salt-scope", "condition"]).local_cache_salt_scope,
            "condition",
        )
        self.assertEqual(
            parse_args(["--local-cache-salt-scope", "request"]).local_cache_salt_scope,
            "request",
        )

    def test_agent_tool_availability_is_extracted_by_exact_name(self):
        features = router.extract_features(
            {
                "tools": [
                    {"name": "Read", "input_schema": {}},
                    {"name": "Agent", "input_schema": {}},
                ]
            }
        )
        self.assertTrue(features.has_agent_tool)

    def test_branch_turn_ordinal_counts_tool_result_messages_not_blocks(self):
        features = router.extract_features(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "task"}]},
                    {"role": "assistant", "content": [{"type": "tool_use"}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "one"},
                            {"type": "tool_result", "tool_use_id": "two"},
                        ],
                    },
                    {"role": "system", "content": [{"type": "text", "text": "reminder"}]},
                    {"role": "assistant", "content": [{"type": "tool_use"}]},
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "three"}],
                    },
                    {"role": "system", "content": [{"type": "text", "text": "reminder"}]},
                ]
            }
        )
        self.assertEqual(features.branch_turn_ordinal, 3)
        self.assertTrue(features.is_tool_continuation)

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

    def test_reliability_circuit_open_routes_cloud(self):
        decision = router.StaticPolicy().decide(
            self._features(local_reliability_blocked=True)
        )
        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "reliability-circuit-open")

    def test_reliability_circuit_closed_by_default(self):
        # Default False; ordinary feasible calls are unaffected by this gate
        # existing at all.
        decision = router.StaticPolicy().decide(self._features())
        self.assertEqual(decision.placement, "local")

    def test_reliability_gate_is_checked_after_hard_feasibility_gates(self):
        # A server-tool call must still go cloud for the original reason,
        # not have it masked by the reliability gate.
        decision = router.StaticPolicy().decide(
            self._features(has_server_tools=True, local_reliability_blocked=True)
        )
        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "server-side-tool")

    def test_warm_local_policy_respects_reliability_circuit(self):
        # WarmLocalPolicy delegates to StaticPolicy.decide() first and must
        # not override an already-cloud decision back to local.
        decision = router.WarmLocalPolicy().decide(
            self._features(
                is_tool_continuation=True, local_reliability_blocked=True
            )
        )
        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "reliability-circuit-open")


class BranchDriftPolicyTests(unittest.TestCase):
    """Rung 3: cold-branch gate (inherited from WarmLocalPolicy) plus a
    data-tuned headroom-pressure escalation. Threshold 0.65 is the
    point-estimate boundary found by sweeping 387 real SWE-bench Pro local
    calls, though a cluster-bootstrap follow-up found it is not
    statistically distinguishable from 0.60-0.70 (only 4 of 27 trace
    directories contain any truncation) -- see BranchDriftPolicy's
    docstring and claude-memory/wiki/decisions/Tune Rung 3 branch drift
    policy against real trace data.md for both the sweep and the
    correction. These tests just verify the mechanism is wired correctly
    at the frozen default, not that 0.65 is proven optimal.
    """

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
            "is_tool_continuation": True,  # past the cold-branch gate by default
            "local_prompt_tokens": 100,
        }
        values.update(overrides)
        return router.CallFeatures(**values)

    def test_registered_under_branch_drift(self):
        policy = router.build("branch-drift", max_local_tokens=100_000, margin=0.9)
        self.assertIsInstance(policy, router.BranchDriftPolicy)
        self.assertEqual(policy.budget(), 90_000)
        self.assertEqual(policy.headroom_threshold, 0.65)

    def test_inherits_cold_branch_gate(self):
        # Same as WarmLocalPolicy: a fresh branch's first turn escalates,
        # independent of headroom.
        policy = router.BranchDriftPolicy(max_local_tokens=100_000, margin=0.9)
        decision = policy.decide(
            self._features(is_tool_continuation=False, local_prompt_tokens=100)
        )
        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "cold-branch-first-turn")

    def test_below_threshold_stays_local(self):
        policy = router.BranchDriftPolicy(max_local_tokens=100_000, margin=0.9)
        # 58,000 / 90,000 = 0.644 < 0.65
        decision = policy.decide(self._features(local_prompt_tokens=58_000))
        self.assertEqual(decision.placement, "local")

    def test_at_or_above_threshold_escalates(self):
        policy = router.BranchDriftPolicy(max_local_tokens=100_000, margin=0.9)
        # 58,500 / 90,000 = 0.65 exactly
        decision = policy.decide(self._features(local_prompt_tokens=58_500))
        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "branch-headroom-pressure")

    def test_threshold_is_configurable(self):
        policy = router.BranchDriftPolicy(
            max_local_tokens=100_000, margin=0.9, headroom_threshold=0.9
        )
        # Would have escalated at the default 0.65 threshold, but not at 0.9.
        decision = policy.decide(self._features(local_prompt_tokens=58_500))
        self.assertEqual(decision.placement, "local")

    def test_missing_prompt_tokens_does_not_crash(self):
        # StaticPolicy's own hard feasibility gate already routes cloud
        # when local_prompt_tokens is unavailable, before BranchDriftPolicy's
        # headroom check would even run -- this just confirms that gate
        # composes cleanly through the subclass without raising.
        policy = router.BranchDriftPolicy(max_local_tokens=100_000, margin=0.9)
        decision = policy.decide(self._features(local_prompt_tokens=None))
        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "local-token-count-unavailable")

    def test_respects_reliability_circuit_before_headroom_check(self):
        policy = router.BranchDriftPolicy(max_local_tokens=100_000, margin=0.9)
        decision = policy.decide(
            self._features(local_prompt_tokens=100, local_reliability_blocked=True)
        )
        self.assertEqual(decision.placement, "cloud")
        self.assertEqual(decision.reason, "reliability-circuit-open")


class PlanningEscalationPolicyTests(unittest.TestCase):
    def _features(self, ordinal, **overrides):
        values = {
            "model": "claude-sonnet-5",
            "has_tools": True,
            "n_tools": 1,
            "has_server_tools": False,
            "n_messages": 1,
            "est_system_tokens": 0,
            "max_tokens": 64,
            "stream": True,
            "is_tool_continuation": ordinal is not None and ordinal > 1,
            "branch_turn_ordinal": ordinal,
            "local_prompt_tokens": 100,
        }
        values.update(overrides)
        return router.CallFeatures(**values)

    def test_registered_with_trace_grounded_default(self):
        policy = router.build(
            "planning-escalation", max_local_tokens=100_000, margin=0.9
        )
        self.assertIsInstance(policy, router.PlanningEscalationPolicy)
        self.assertEqual(policy.planning_turns, 3)

    def test_escalates_exactly_first_configured_turns(self):
        policy = router.PlanningEscalationPolicy(planning_turns=3)
        decisions = [policy.decide(self._features(i)) for i in range(1, 7)]

        self.assertEqual(
            [decision.placement for decision in decisions],
            ["cloud", "cloud", "cloud", "local", "local", "local"],
        )
        self.assertEqual(decisions[0].reason, "cold-branch-first-turn")
        self.assertEqual(
            [decision.reason for decision in decisions[1:3]],
            ["early-planning-turn", "early-planning-turn"],
        )

    def test_cutoff_is_configurable(self):
        policy = router.PlanningEscalationPolicy(planning_turns=2)
        self.assertEqual(policy.decide(self._features(2)).placement, "cloud")
        self.assertEqual(policy.decide(self._features(3)).placement, "local")

    def test_beyond_cutoff_is_identical_to_warm_local(self):
        warm = router.WarmLocalPolicy()
        planning = router.PlanningEscalationPolicy(planning_turns=3)
        cases = [
            self._features(4),
            self._features(4, is_tool_continuation=False),
            self._features(4, has_server_tools=True),
            self._features(4, local_reliability_blocked=True),
            self._features(None, is_tool_continuation=True),
        ]

        for features in cases:
            with self.subTest(features=features):
                self.assertEqual(planning.decide(features), warm.decide(features))


if __name__ == "__main__":
    unittest.main()
