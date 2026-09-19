import unittest

from edgeproxy.agentic_history import AgenticHistory


def finish(history, lane, seq, *, placement="cloud", request=None, response=None):
    history.complete(
        lane, seq, request=request or {"messages": []},
        errored_tool_result_density=0.0, placement=placement,
        response=response or {"stop_reason": "end_turn", "content": []},
        tool_use_blocks=[],
    )


class AgenticHistoryTests(unittest.TestCase):
    def test_first_turn_and_independent_lanes(self):
        h = AgenticHistory()
        a0 = h.begin("episode-a|main")
        b0 = h.begin("episode-b|main")
        self.assertEqual(h.snapshot("episode-a|main", a0)[0]["turn_index"], 0.0)
        finish(h, "episode-a|main", a0, placement="local")
        self.assertEqual(h.snapshot("episode-b|main", b0)[0]["consecutive_same_backend_turns"], 0.0)

    def test_prior_in_flight_makes_later_snapshot_unsupported(self):
        h = AgenticHistory()
        first = h.begin("one")
        second = h.begin("one")
        self.assertEqual(h.snapshot("one", second)[1], "earlier-call-in-flight")
        finish(h, "one", first)
        features, reason = h.snapshot("one", second)
        self.assertIsNone(reason)
        self.assertEqual(features["consecutive_same_backend_turns"], 1.0)

    def test_out_of_order_completions_sort_by_original_sequence(self):
        h = AgenticHistory()
        first, second = h.begin("one"), h.begin("one")
        finish(h, "one", second, placement="cloud")
        finish(h, "one", first, placement="local")
        third = h.begin("one")
        features, reason = h.snapshot("one", third)
        self.assertIsNone(reason)
        self.assertEqual(features["consecutive_same_backend_turns"], 1.0)

    def test_prior_produced_tools_not_offered_tools_define_repair(self):
        h = AgenticHistory()
        for _ in range(3):
            seq = h.begin("one")
            finish(h, "one", seq, response={"stop_reason": "tool_use", "content": [{"type": "tool_use", "name": "Edit"}]})
        next_seq = h.begin("one")
        self.assertEqual(h.snapshot("one", next_seq)[0]["repair_loop_flag"], 1.0)

    def test_bounded_history_fails_closed_after_eviction(self):
        h = AgenticHistory(max_calls_per_lane=4)
        for _ in range(5):
            seq = h.begin("one")
            finish(h, "one", seq)
        next_seq = h.begin("one")
        self.assertEqual(h.snapshot("one", next_seq)[1], "history-truncated")

    def test_abort_discards_candidate_and_fails_closed(self):
        h = AgenticHistory()
        first = h.begin("one")
        h.abort("one", first)
        h.abort("one", first)  # idempotent cleanup
        self.assertEqual(list(h.lanes["one"].calls), [])
        self.assertEqual(h.lanes["one"].pending, set())
        second = h.begin("one")
        self.assertEqual(h.snapshot("one", second)[1], "history-truncated")


if __name__ == "__main__":
    unittest.main()
