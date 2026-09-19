import unittest

from experiments.agentic_router.hard_turn_audit import _result_ids


def request_with_results(*items):
    return {"request": {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": identifier, "is_error": error}
        for identifier, error in items
    ]}]}}


class HardTurnAuditTests(unittest.TestCase):
    def test_old_error_stays_in_prompt_but_is_not_new(self):
        previous = request_with_results(("old", True))
        current = request_with_results(("old", True), ("new-success", False))
        previous_ids, _, _ = _result_ids(previous)
        _, current_errors, _ = _result_ids(current)
        self.assertEqual(current_errors - previous_ids, set())

    def test_new_explicit_error_is_counted_once(self):
        previous = request_with_results(("old", True))
        current = request_with_results(("old", True), ("new-error", True))
        previous_ids, _, _ = _result_ids(previous)
        _, current_errors, _ = _result_ids(current)
        self.assertEqual(current_errors - previous_ids, {"new-error"})


if __name__ == "__main__":
    unittest.main()
