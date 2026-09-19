import unittest
from concurrent.futures import ThreadPoolExecutor

from edgeproxy.cohort import (
    CohortTracker,
    exhaustive_backend_plan,
    request_contains_prompt,
)


class CohortTrackerTests(unittest.TestCase):
    def test_exact_content_match_records_window_cost_without_waiting(self):
        tracker = CohortTracker(window_ms=300)
        parent = tracker.observe_parent(
            call_id="parent-call",
            session_id="session-1",
            backend="cloud",
            completed_at_unix_s=10.0,
            response={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "agent-a",
                        "name": "Agent",
                        "input": {"prompt": "inspect routing"},
                    },
                    {
                        "type": "tool_use",
                        "id": "agent-b",
                        "name": "Agent",
                        "input": {"prompt": "inspect caching"},
                    },
                ]
            },
        )

        first = tracker.match_child(
            session_id="session-1",
            request={"messages": [{"content": "inspect routing carefully"}]},
            arrived_at_unix_s=10.020,
        )
        second = tracker.match_child(
            session_id="session-1",
            request={"messages": [{"content": [{"text": "inspect caching"}]}]},
            arrived_at_unix_s=10.257,
        )

        self.assertEqual(parent["expected_width"], 2)
        self.assertEqual(parent["parent_backend"], "cloud")
        self.assertEqual(first["detection_method"], "content_exact")
        self.assertEqual(first["ready_width_at_arrival"], 1)
        self.assertEqual(second["ready_width_at_arrival"], 2)
        # 237 ms distinguishes the new 300 ms window from the former default
        # instead of merely asserting the configured value.
        self.assertAlmostEqual(second["arrival_offset_ms"], 237.0)
        self.assertTrue(second["within_configured_window"])
        self.assertEqual(second["actual_wait_ms"], 0.0)
        self.assertTrue(second["dispatch_irrevocable"])

    def test_repeated_prompt_uses_causal_time_and_lowers_confidence(self):
        tracker = CohortTracker(window_ms=300)
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "old-tool",
                    "name": "Agent",
                    "input": {"prompt": "same work"},
                }
            ]
        }
        tracker.observe_parent(
            call_id="old-parent",
            session_id="s",
            backend="local",
            response=response,
            completed_at_unix_s=1.0,
        )
        response["content"][0]["id"] = "new-tool"
        tracker.observe_parent(
            call_id="new-parent",
            session_id="s",
            backend="cloud",
            response=response,
            completed_at_unix_s=2.0,
        )

        match = tracker.match_child(
            session_id="s",
            request={"messages": ["same work"]},
            arrived_at_unix_s=2.1,
        )

        self.assertEqual(match["parent_call_id"], "new-parent")
        self.assertEqual(match["parent_tool_use_id"], "new-tool")
        self.assertEqual(match["detection_method"], "content_and_causal_time")
        self.assertEqual(match["detection_confidence"], "medium")
        self.assertEqual(match["candidate_count"], 2)

    def test_malformed_agent_input_is_not_registered(self):
        tracker = CohortTracker(window_ms=300)

        observed = tracker.observe_parent(
            call_id="bad",
            session_id="s",
            backend="local",
            completed_at_unix_s=1.0,
            response={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "bad-tool",
                        "name": "Agent",
                        "input": {"_unparsed": "{"},
                    }
                ]
            },
        )

        self.assertIsNone(observed)
        self.assertIsNone(
            tracker.match_child(
                session_id="s",
                request={"messages": ["anything"]},
                arrived_at_unix_s=2.0,
            )
        )

    def test_request_match_is_recursive_and_session_scoped(self):
        self.assertTrue(
            request_contains_prompt(
                {"messages": [{"content": [{"text": "prefix exact prompt suffix"}]}]},
                "exact prompt",
            )
        )
        self.assertFalse(request_contains_prompt({"messages": []}, "missing"))

    def test_incremental_delegations_match_before_parent_completion(self):
        tracker = CohortTracker(window_ms=300)

        first_parent = tracker.register_delegation(
            call_id="parent-call",
            session_id="session-1",
            backend="cloud",
            tool_use_id="agent-a",
            prompt="inspect routing",
            observed_at_unix_s=10.0,
        )
        first_child = tracker.match_child(
            session_id="session-1",
            request={"messages": ["inspect routing now"]},
            arrived_at_unix_s=10.05,
        )
        second_parent = tracker.register_delegation(
            call_id="parent-call",
            session_id="session-1",
            backend="cloud",
            tool_use_id="agent-b",
            prompt="inspect caching",
            observed_at_unix_s=14.5,
        )
        second_child = tracker.match_child(
            session_id="session-1",
            request={"messages": ["inspect caching now"]},
            arrived_at_unix_s=14.6,
        )

        self.assertEqual(first_parent["expected_width"], 1)
        self.assertEqual(first_child["parent_tool_use_id"], "agent-a")
        self.assertEqual(second_parent["expected_width"], 2)
        self.assertEqual(second_child["parent_tool_use_id"], "agent-b")

    def test_incremental_registration_is_idempotent_by_tool_use_id(self):
        tracker = CohortTracker(window_ms=300)
        kwargs = {
            "call_id": "parent-call",
            "session_id": "session-1",
            "backend": "cloud",
            "tool_use_id": "agent-a",
            "prompt": "inspect routing",
            "observed_at_unix_s": 10.0,
        }

        first = tracker.register_delegation(**kwargs)
        duplicate = tracker.register_delegation(**kwargs)
        reconciled = tracker.observe_parent(
            call_id="parent-call",
            session_id="session-1",
            backend="cloud",
            response={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "agent-a",
                        "name": "Agent",
                        "input": {"prompt": "inspect routing"},
                    }
                ]
            },
            completed_at_unix_s=20.0,
        )
        child = tracker.match_child(
            session_id="session-1",
            request={"messages": ["inspect routing"]},
            arrived_at_unix_s=10.1,
        )

        self.assertEqual(first["cohort_id"], duplicate["cohort_id"])
        self.assertEqual(duplicate["expected_width"], 1)
        self.assertEqual(reconciled["expected_width"], 1)
        self.assertEqual(child["candidate_count"], 1)

    def test_concurrent_incremental_registration_accumulates_one_cohort(self):
        tracker = CohortTracker(window_ms=300)

        def register(index):
            return tracker.register_delegation(
                call_id="parent-call",
                session_id="session-1",
                backend="cloud",
                tool_use_id=f"agent-{index}",
                prompt=f"inspect area {index}",
                observed_at_unix_s=10.0 + index,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            detections = list(pool.map(register, range(20)))

        self.assertEqual(
            {item["cohort_id"] for item in detections},
            {detections[0]["cohort_id"]},
        )
        final = tracker.match_child(
            session_id="session-1",
            request={"messages": ["inspect area 19"]},
            arrived_at_unix_s=30.0,
        )
        self.assertEqual(final["expected_width"], 20)
        self.assertEqual(final["parent_tool_use_id"], "agent-19")

    def test_exhaustive_plan_checks_every_vector(self):
        seen = []

        def score(vector):
            seen.append(tuple(vector.values()))
            return sum(value == "cloud" for value in vector.values())

        vector, value = exhaustive_backend_plan(["a", "b", "c"], score)

        self.assertEqual(len(seen), 8)
        self.assertEqual(vector, {"a": "local", "b": "local", "c": "local"})
        self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
