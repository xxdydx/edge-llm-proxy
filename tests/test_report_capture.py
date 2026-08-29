import unittest

from edgeproxy.report_capture import (
    REPORT_COMPLETE,
    REPORT_START,
    assemble_parent_report,
    parse_args,
)


class ReportCaptureTests(unittest.TestCase):
    def test_cli_defaults_to_stdin(self):
        self.assertIsNone(parse_args([]).path)

    def test_assembles_all_parent_turns_and_excludes_subagents(self):
        events = [
            {
                "type": "assistant",
                "session_id": "session",
                "parent_tool_use_id": "tool-parent",
                "message": {
                    "id": "subagent",
                    "content": [{"type": "text", "text": "exclude me"}],
                },
            },
            {
                "type": "assistant",
                "session_id": "session",
                "parent_tool_use_id": None,
                "message": {
                    "id": "turn-1",
                    "content": [
                        {
                            "type": "text",
                            "text": f"progress\n{REPORT_START}\n## Executive Summary\nfirst half",
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "session_id": "session",
                "parent_tool_use_id": None,
                "message": {
                    "id": "turn-2",
                    "content": [
                        {
                            "type": "text",
                            "text": f"second half\n{REPORT_COMPLETE}\nignored tail",
                        }
                    ],
                },
            },
        ]

        report = assemble_parent_report(events)

        self.assertTrue(report.startswith(REPORT_START))
        self.assertIn("first half\n\nsecond half", report)
        self.assertTrue(report.rstrip().endswith(REPORT_COMPLETE))
        self.assertNotIn("progress", report)
        self.assertNotIn("exclude me", report)
        self.assertNotIn("ignored tail", report)

    def test_repeated_message_id_keeps_latest_complete_value(self):
        def event(text):
            return {
                "type": "assistant",
                "session_id": "session",
                "message": {
                    "id": "same-turn",
                    "content": [{"type": "text", "text": text}],
                },
            }

        report = assemble_parent_report(
            [
                event(f"{REPORT_START}\npartial"),
                event(f"{REPORT_START}\ncomplete\n{REPORT_COMPLETE}"),
            ]
        )

        self.assertIn("complete", report)
        self.assertNotIn("partial", report)


if __name__ == "__main__":
    unittest.main()
