import unittest

from edgeproxy.trace.graph import build_trace_graph, render_mermaid, render_tree


class TraceGraphTests(unittest.TestCase):
    def test_stream_parent_builds_exact_agent_tree(self):
        records = [
            {
                "id": "call-main",
                "ts": 1.0,
                "headers": {"x-claude-code-session-id": "session-1"},
                "request": {"messages": []},
                "response": {
                    "id": "message-main",
                    "content": [
                        {"type": "tool_use", "id": "tool_A", "name": "Agent"}
                    ],
                },
            },
            {
                "id": "call-child",
                "ts": 2.0,
                "headers": {
                    "x-claude-code-session-id": "session-1",
                    "x-claude-code-agent-id": "child_1",
                },
                "request": {"messages": []},
                "response": {
                    "id": "message-child",
                    "content": [
                        {"type": "tool_use", "id": "tool_B", "name": "Read"},
                        {"type": "tool_use", "id": "tool_C", "name": "Grep"},
                    ],
                },
            },
        ]
        stream = [
            {
                "parent_tool_use_id": "tool_A",
                "message": {"id": "message-child"},
            }
        ]

        graph = build_trace_graph(records, stream_events=stream)

        spawn = next(edge for edge in graph["edges"] if edge["type"] == "spawned_agent")
        self.assertEqual(spawn["source"], "tool:tool_A")
        self.assertEqual(spawn["target"], "agent:session-1:child_1")
        self.assertEqual(spawn["detection_method"], "ground_truth_stream")
        self.assertEqual(spawn["detection_confidence"], "exact")
        self.assertEqual(graph["unresolved"], [])
        self.assertEqual(
            render_tree(graph),
            "Main agent\n"
            "└── Agent [tool_A]\n"
            "    └── Child agent [child_1]\n"
            "        ├── Read [tool_B]\n"
            "        └── Grep [tool_C]\n",
        )

    def test_missing_stream_parent_is_reported_not_guessed(self):
        graph = build_trace_graph(
            [
                {
                    "id": "child-call",
                    "ts": 1.0,
                    "headers": {
                        "x-claude-code-session-id": "session-1",
                        "x-claude-code-agent-id": "child-unknown",
                    },
                    "request": {"messages": []},
                    "response": {"id": "child-message", "content": []},
                }
            ]
        )

        self.assertEqual(
            graph["unresolved"][0]["reason"], "parent_tool_use_id_unavailable"
        )
        self.assertIn("Unresolved child agents", render_tree(graph))

    def test_graph_is_deterministic_when_trace_order_changes(self):
        first = {
            "id": "a",
            "ts": 1.0,
            "headers": {"x-claude-code-session-id": "s"},
            "request": {"messages": []},
            "response": {
                "id": "m-a",
                "content": [{"type": "tool_use", "id": "t-a", "name": "Read"}],
            },
        }
        second = {
            "id": "b",
            "ts": 2.0,
            "headers": {"x-claude-code-session-id": "s"},
            "request": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "t-a"}
                        ],
                    }
                ]
            },
            "response": {"id": "m-b", "content": []},
        }

        self.assertEqual(
            build_trace_graph([first, second]), build_trace_graph([second, first])
        )

    def test_content_linking_builds_cohort_and_cache_edges_without_ground_truth(self):
        prompt = "Inspect the router and report its placement invariants."
        parent = {
            "id": "parent",
            "ts": 1.0,
            "placement": "cloud",
            "headers": {"x-claude-code-session-id": "session-1"},
            "request": {"system": "shared", "messages": [{"content": "root"}]},
            "response": {
                "id": "parent-message",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "agent-tool",
                        "name": "Agent",
                        "input": {"prompt": prompt},
                    }
                ],
            },
        }
        child = {
            "id": "child",
            "ts": 1.037,
            "placement": "local",
            "status": 200,
            "headers": {
                "x-claude-code-session-id": "session-1",
                "x-claude-code-agent-id": "child-1",
            },
            "request": {
                "system": "shared",
                "messages": [{"content": f"Delegation: {prompt}"}],
            },
            "response": {"id": "child-message", "content": []},
        }

        graph = build_trace_graph([parent, child])

        spawn = next(edge for edge in graph["edges"] if edge["type"] == "spawned_agent")
        self.assertEqual(spawn["detection_method"], "content_exact")
        self.assertEqual(spawn["detection_confidence"], "high")
        self.assertIn("structural_shared_prefix_chars", spawn["cache_relationship"])
        self.assertEqual(graph["cohorts"][0]["expected_width"], 1)
        self.assertEqual(graph["cohorts"][0]["arrival_span_ms"], 0.0)
        self.assertTrue(graph["cohorts"][0]["primary_case"])

    def test_content_linker_is_scored_against_stream_ground_truth(self):
        prompt = "Inspect one file."
        records = [
            {
                "id": "parent",
                "ts": 1.0,
                "headers": {"x-claude-code-session-id": "s"},
                "request": {},
                "response": {
                    "id": "m-parent",
                    "content": [{
                        "type": "tool_use", "id": "t", "name": "Agent",
                        "input": {"prompt": prompt},
                    }],
                },
            },
            {
                "id": "child",
                "ts": 2.0,
                "headers": {
                    "x-claude-code-session-id": "s",
                    "x-claude-code-agent-id": "a",
                },
                "request": {"messages": [{"content": prompt}]},
                "response": {"id": "m-child", "content": []},
            },
        ]
        graph = build_trace_graph(
            records,
            stream_events=[{"parent_tool_use_id": "t", "message": {"id": "m-child"}}],
        )

        self.assertEqual(graph["linker_validation"]["accuracy"], 1.0)
        self.assertEqual(graph["linker_validation"]["false_cohort_rate"], 0.0)

    def test_all_failed_children_are_excluded_from_arrival_distribution(self):
        graph = build_trace_graph([{
            "id": "failed", "ts": 1.0, "status": 502,
            "headers": {
                "x-claude-code-session-id": "s",
                "x-claude-code-agent-id": "a",
            },
            "request": {}, "response": {},
        }])

        self.assertFalse(graph["analysis_eligibility"]["arrival_distribution_eligible"])
        self.assertEqual(
            graph["analysis_eligibility"]["exclusion_reason"], "all_child_calls_failed"
        )

    def test_mermaid_is_deterministic_and_contains_edge_cache_metrics(self):
        prompt = "Inspect."
        records = [
            {
                "id": "p", "ts": 1.0, "placement": "cloud",
                "headers": {"x-claude-code-session-id": "s"},
                "request": {"system": "shared"},
                "response": {"content": [{
                    "type": "tool_use", "id": "t", "name": "Agent",
                    "input": {"prompt": prompt},
                }]},
            },
            {
                "id": "c", "ts": 1.01, "placement": "local", "status": 200,
                "headers": {
                    "x-claude-code-session-id": "s",
                    "x-claude-code-agent-id": "a",
                },
                "request": {"system": "shared", "messages": [{"content": prompt}]},
                "response": {"content": []},
            },
        ]
        graph = build_trace_graph(records)

        rendered = render_mermaid(graph)
        self.assertEqual(rendered, render_mermaid(graph))
        self.assertTrue(rendered.startswith("flowchart LR\n"))
        self.assertIn("classDef agentNode", rendered)
        self.assertIn("classDef callNode", rendered)
        self.assertIn("classDef toolNode", rendered)
        self.assertNotIn("classDef call fill", rendered)
        self.assertIn("structural ~", rendered)
        self.assertIn("link high", rendered)


if __name__ == "__main__":
    unittest.main()
