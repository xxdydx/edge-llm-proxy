from __future__ import annotations

import unittest

from experiments.agentic_router import analysis, model_comparison, text_model
from experiments.agentic_router.model_comparison import LabeledEvent


def event(cid: str, group: str, label: str, content: str) -> LabeledEvent:
    return LabeledEvent(cid, 1.0, cid, group, label,
                        {name: 0.0 for name in analysis.FEATURE_NAMES}, content)


class TextModelTests(unittest.TestCase):
    def test_request_text_excludes_prior_assistant_content(self):
        request = {"messages": [
            {"role": "user", "content": "Fix flaky tests"},
            {"role": "assistant", "content": "SECRET_CANDIDATE_RESPONSE"},
            {"role": "user", "content": [{"type": "tool_result", "content": "AssertionError in test_x"}]},
        ]}
        text = model_comparison.request_text_for_model(request)
        self.assertIn("Fix flaky tests", text)
        self.assertIn("AssertionError", text)
        self.assertNotIn("SECRET_CANDIDATE_RESPONSE", text)

    def test_vectorizer_fits_only_train_vocabulary(self):
        train = [event("a", "swebench-one-a", "SAFE", "alpha beta"),
                 event("b", "swebench-two-b", "HARM", "beta gamma")]
        test = [event("c", "swebench-three-c", "SAFE", "heldoutunique")]
        Xtr, Xte, vectorizer = text_model._text_matrix(train, test, "word")
        self.assertNotIn("heldoutunique", vectorizer.vocabulary_)
        self.assertEqual(Xtr.shape[1], Xte.shape[1])

    def test_reserved_holdout_excluded_from_development(self):
        events = [event("a", "swebench-one-a", "SAFE", "one"),
                  event("b", "swebench-one-a", "HARM", "two"),
                  event("c", "swebench-two-b", "SAFE", "three"),
                  event("d", "swebench-two-b", "HARM", "four"),
                  event("heldout", model_comparison.RESERVED_PILOT_HOLDOUT_TASKS[0], "HARM", "private")]
        result = text_model.compare(events, {"source": "synthetic"})
        self.assertEqual(result["n_development"], 4)
        self.assertNotIn(model_comparison.RESERVED_PILOT_HOLDOUT_TASKS[0],
                         result["folds"]["leave_task_out"]["word_only"]["folds"])


if __name__ == "__main__":
    unittest.main()
