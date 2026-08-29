import json
import tempfile
import unittest
from pathlib import Path

from edgeproxy.episode import (
    build_user_event,
    capture_episode_metadata,
    rewind_command,
)
from edgeproxy.episode import main as episode_main


class EpisodeTests(unittest.TestCase):
    def test_stream_input_contains_prompt_but_metadata_does_not(self):
        prompt = "private evaluation prompt"
        event = build_user_event(prompt)

        metadata = capture_episode_metadata(
            [
                {
                    "type": "user",
                    "uuid": "checkpoint-start",
                    "session_id": "session-1",
                    "parent_tool_use_id": None,
                    "message": event["message"],
                    "isReplay": True,
                }
            ],
            experiment_id="experiment-1",
            episode_id="experiment-1-routing",
            condition="routing",
            expected_session_id="session-1",
            working_directory="/repo/edgeproxy",
        )

        self.assertEqual(
            event["message"]["content"][0]["text"], "private evaluation prompt"
        )
        self.assertNotIn(prompt, str(metadata))
        self.assertEqual(metadata["experiment_id"], "experiment-1")
        self.assertEqual(metadata["episode_id"], "experiment-1-routing")
        self.assertEqual(
            metadata["checkpointing"]["initial_checkpoint_id"],
            "checkpoint-start",
        )

    def test_capture_deduplicates_replays_and_ignores_child_messages(self):
        events = [
            {
                "type": "user",
                "uuid": "start",
                "session_id": "session-1",
                "parent_tool_use_id": None,
            },
            {
                "type": "user",
                "uuid": "start",
                "session_id": "session-1",
                "parent_tool_use_id": None,
            },
            {
                "type": "user",
                "uuid": "child",
                "session_id": "session-1",
                "parent_tool_use_id": "agent-tool",
            },
            {
                "type": "user",
                "uuid": "green",
                "session_id": "session-1",
                "parent_tool_use_id": None,
            },
        ]

        metadata = capture_episode_metadata(
            events,
            experiment_id="experiment-1",
            episode_id="episode-1",
            condition="routing",
            expected_session_id="session-1",
        )

        self.assertEqual(
            metadata["checkpointing"]["checkpoints"],
            [
                {"checkpoint_id": "start", "kind": "episode_start"},
                {"checkpoint_id": "green", "kind": "user_turn"},
            ],
        )
        self.assertEqual(metadata["checkpointing"]["latest_checkpoint_id"], "green")

    def test_capture_rejects_a_different_session(self):
        with self.assertRaisesRegex(ValueError, "did not match"):
            capture_episode_metadata(
                [{"type": "result", "session_id": "wrong-session"}],
                experiment_id="experiment-1",
                episode_id="episode-1",
                condition="routing",
                expected_session_id="expected-session",
            )

    def test_rewind_command_requires_known_checkpoint(self):
        metadata = {
            "claude_session_id": "session-1",
            "checkpointing": {
                "initial_checkpoint_id": "start",
                "latest_checkpoint_id": "green",
                "checkpoints": [
                    {"checkpoint_id": "start"},
                    {"checkpoint_id": "green"},
                ],
                "green_checkpoint_ids": ["green"],
            },
        }

        self.assertEqual(
            rewind_command(metadata, "latest", "/bin/claude"),
            [
                "/bin/claude",
                "-p",
                "--resume",
                "session-1",
                "--rewind-files",
                "green",
            ],
        )
        with self.assertRaisesRegex(ValueError, "unknown checkpoint"):
            rewind_command(metadata, "missing", "/bin/claude")

    def test_mark_green_updates_metadata_and_green_rewind_alias(self):
        metadata = {
            "claude_session_id": "session-1",
            "checkpointing": {
                "initial_checkpoint_id": "start",
                "latest_checkpoint_id": "after-tests",
                "checkpoints": [
                    {"checkpoint_id": "start"},
                    {"checkpoint_id": "after-tests"},
                ],
                "green_checkpoint_ids": [],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")

            self.assertEqual(episode_main(["mark-green", str(path)]), 0)
            updated = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            updated["checkpointing"]["green_checkpoint_ids"], ["after-tests"]
        )
        self.assertEqual(
            rewind_command(updated, "green", "claude")[-1], "after-tests"
        )


if __name__ == "__main__":
    unittest.main()
