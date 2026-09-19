import unittest

from edgeproxy import router


class CombinedPolicyTests(unittest.TestCase):
    """Policy 6 (``all-improvements``): composes BranchDriftPolicy's
    headroom gate, PlanningEscalationPolicy's early-turn gate, and
    PredictedRiskPolicy's learned risk score on top of WarmLocalPolicy's
    cold-branch gate and StaticPolicy's hard feasibility/reliability
    gates. See claude-memory/wiki/decisions/Compose Policy 6 as the
    all-improvements combined policy.md.
    """

    def features(self, **overrides) -> router.CallFeatures:
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
            "branch_turn_ordinal": 5,
        }
        values.update(overrides)
        return router.CallFeatures(**values)

    def test_registered_and_buildable_by_name(self):
        policy = router.build("all-improvements")
        self.assertIsInstance(policy, router.CombinedPolicy)
        self.assertEqual(policy.name, "all-improvements")

    def test_static_hard_gates_still_take_precedence(self):
        # A base StaticPolicy gate (server-side tool) must win over every
        # escalation gate this policy adds, exactly as it does for every
        # other WarmLocalPolicy subclass.
        policy = router.CombinedPolicy()
        decision = policy.decide(self.features(has_server_tools=True))
        self.assertEqual((decision.placement, decision.reason), ("cloud", "server-side-tool"))

    def test_cold_branch_gate_still_takes_precedence(self):
        policy = router.CombinedPolicy()
        decision = policy.decide(self.features(is_tool_continuation=False))
        self.assertEqual((decision.placement, decision.reason), ("cloud", "cold-branch-first-turn"))

    def test_branch_drift_escalates_alone(self):
        policy = router.CombinedPolicy(headroom_threshold=0.65)
        # budget() = 60_000 * 0.9 = 54_000; 40_000/54_000 ≈ 0.741 >= 0.65
        decision = policy.decide(self.features(local_prompt_tokens=40_000, branch_turn_ordinal=10))
        self.assertEqual((decision.placement, decision.reason), ("cloud", "branch-headroom-pressure"))

    def test_planning_turn_escalates_alone(self):
        policy = router.CombinedPolicy(planning_turns=3)
        decision = policy.decide(self.features(branch_turn_ordinal=2, local_prompt_tokens=1_000))
        self.assertEqual((decision.placement, decision.reason), ("cloud", "early-planning-turn"))

    def test_predicted_risk_escalates_alone(self):
        policy = router.CombinedPolicy(risk_threshold=0.4)
        decision = policy.decide(
            self.features(
                predicted_local_risk_score=0.5,
                branch_turn_ordinal=10,
                local_prompt_tokens=1_000,
            )
        )
        self.assertEqual((decision.placement, decision.reason), ("cloud", "predicted-local-risk"))

    def test_no_gate_fires_stays_local(self):
        policy = router.CombinedPolicy()
        decision = policy.decide(
            self.features(
                branch_turn_ordinal=10,
                local_prompt_tokens=1_000,
                predicted_local_risk_score=0.01,
            )
        )
        self.assertEqual((decision.placement, decision.reason), ("local", "fits"))

    def test_precedence_planning_turn_beats_branch_drift_and_risk(self):
        # All three would independently fire; planning-turn must win per
        # the documented precedence.
        policy = router.CombinedPolicy(headroom_threshold=0.65, planning_turns=3, risk_threshold=0.4)
        decision = policy.decide(
            self.features(
                branch_turn_ordinal=1,  # planning-turn fires
                local_prompt_tokens=40_000,  # branch-drift would also fire
                predicted_local_risk_score=0.9,  # predicted-risk would also fire
            )
        )
        self.assertEqual(decision.reason, "early-planning-turn")

    def test_precedence_branch_drift_beats_risk_when_planning_turn_does_not_fire(self):
        policy = router.CombinedPolicy(headroom_threshold=0.65, planning_turns=3, risk_threshold=0.4)
        decision = policy.decide(
            self.features(
                branch_turn_ordinal=10,  # planning-turn does not fire
                local_prompt_tokens=40_000,  # branch-drift fires
                predicted_local_risk_score=0.9,  # predicted-risk would also fire
            )
        )
        self.assertEqual(decision.reason, "branch-headroom-pressure")

    def test_parameters_validated(self):
        with self.assertRaises(ValueError):
            router.CombinedPolicy(risk_threshold=1.5)
        with self.assertRaises(ValueError):
            router.CombinedPolicy(planning_turns=0)


class SharedEscalationHelperTests(unittest.TestCase):
    """The three escalation checks are shared functions, used by both the
    single-signal policies (BranchDriftPolicy etc.) and CombinedPolicy —
    confirms the refactor is behavior-identical for the single-signal
    policies too."""

    def features(self, **overrides) -> router.CallFeatures:
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

    def test_branch_drift_policy_unchanged_after_refactor(self):
        policy = router.BranchDriftPolicy(headroom_threshold=0.65)
        below = policy.decide(self.features(local_prompt_tokens=10_000))
        at = policy.decide(self.features(local_prompt_tokens=40_000))
        self.assertEqual((below.placement, below.reason), ("local", "fits"))
        self.assertEqual((at.placement, at.reason), ("cloud", "branch-headroom-pressure"))

    def test_planning_escalation_policy_unchanged_after_refactor(self):
        policy = router.PlanningEscalationPolicy(planning_turns=3)
        early = policy.decide(self.features(branch_turn_ordinal=1))
        late = policy.decide(self.features(branch_turn_ordinal=5))
        self.assertEqual((early.placement, early.reason), ("cloud", "early-planning-turn"))
        self.assertEqual((late.placement, late.reason), ("local", "fits"))


if __name__ == "__main__":
    unittest.main()
