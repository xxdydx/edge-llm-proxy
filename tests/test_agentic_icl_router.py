from __future__ import annotations

import unittest

from experiments.agentic_router import analysis, icl_router
from experiments.agentic_router.model_comparison import LabeledEvent, RESERVED_PILOT_HOLDOUT_TASKS


def event(cid: str, group: str, label: str, value: float) -> LabeledEvent:
    return LabeledEvent(cid, None, cid, group, label,
                        {name: value for name in analysis.FEATURE_NAMES})


class ICLRouterTests(unittest.TestCase):
    def test_excludes_same_task_reserved_tasks_and_unknown_exemplars(self):
        target = event("target", "swebench-ansible-a", "UNKNOWN", 2)
        pool = [event("same", target.task_group, "HARM", 2),
                event("heldout", RESERVED_PILOT_HOLDOUT_TASKS[0], "HARM", 2),
                event("unknown", "swebench-repo-x", "UNKNOWN", 2),
                event("harm", "swebench-repo-y", "HARM", 3),
                event("safe", "swebench-repo-z", "SAFE", 4)]
        selected = icl_router.choose_exemplars(target, pool)
        self.assertEqual({e.call_id for e in selected}, {"harm", "safe"})
        prompt = icl_router.build_prompt(target, selected)
        self.assertNotIn("swebench-", prompt)  # task identity is omitted
        self.assertIn("harm_probability", prompt)

    def test_holdout_cannot_enter_probe(self):
        target = event("test", RESERVED_PILOT_HOLDOUT_TASKS[0], "UNKNOWN", 0)
        with self.assertRaises(ValueError):
            icl_router.build_prompt(target, [])

    def test_invalid_response_abstains(self):
        self.assertEqual(icl_router.parse_response('{"verdict":"HARM","harm_probability":true}'),
                         ("UNKNOWN", None))
        self.assertEqual(icl_router.parse_response('not json'), ("UNKNOWN", None))
        self.assertEqual(icl_router.parse_response('{"verdict":"SAFE","harm_probability":0.2}'),
                         ("SAFE", 0.2))
