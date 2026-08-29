import unittest

from edgeproxy.trace.graph import build_trace_graph, render_tree


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

        self.assertIn(
            {
                "source": "tool:tool_A",
                "target": "agent:session-1:child_1",
                "type": "spawned_agent",
            },
            graph["edges"],
        )
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


if __name__ == "__main__":
    unittest.main()
