"""Create Claude stream input and persist checkpoint metadata per episode."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO


def build_user_event(prompt: str) -> dict[str, Any]:
    """Build one Claude Code streaming-input user turn without logging it."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
        "parent_tool_use_id": None,
    }


def read_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid stream JSON on line {line_number}: {exc}") from exc
        if isinstance(event, dict):
            events.append(event)
    return events


def capture_episode_metadata(
    events: Iterable[dict[str, Any]],
    *,
    experiment_id: str,
    episode_id: str,
    condition: str,
    expected_session_id: str | None = None,
    working_directory: str | None = None,
) -> dict[str, Any]:
    """Extract main-session checkpoint UUIDs without copying prompt content."""
    session_ids: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    seen_checkpoints: set[str] = set()
    lineage_links: list[dict[str, str]] = []
    seen_lineage_links: set[tuple[str, str]] = set()
    for event in events:
        session_id = event.get("session_id")
        if session_id and str(session_id) not in session_ids:
            session_ids.append(str(session_id))
        message = event.get("message")
        message_id = message.get("id") if isinstance(message, dict) else None
        parent_tool_use_id = event.get("parent_tool_use_id")
        if parent_tool_use_id is None and isinstance(message, dict):
            parent_tool_use_id = message.get("parent_tool_use_id")
        if message_id and parent_tool_use_id:
            key = (str(message_id), str(parent_tool_use_id))
            if key not in seen_lineage_links:
                seen_lineage_links.add(key)
                lineage_links.append(
                    {
                        "child_message_id": key[0],
                        "parent_tool_use_id": key[1],
                    }
                )
        if (
            event.get("type") != "user"
            or event.get("parent_tool_use_id") is not None
            or not event.get("uuid")
        ):
            continue
        checkpoint_id = str(event["uuid"])
        if checkpoint_id in seen_checkpoints:
            continue
        seen_checkpoints.add(checkpoint_id)
        checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "kind": "episode_start" if not checkpoints else "user_turn",
            }
        )

    if expected_session_id:
        unexpected = [value for value in session_ids if value != expected_session_id]
        if unexpected:
            raise ValueError(
                "Claude stream session did not match the harness session: "
                + ", ".join(unexpected)
            )
        claude_session_id = expected_session_id
    else:
        claude_session_id = session_ids[0] if len(session_ids) == 1 else None

    return {
        "schema_version": "edgeproxy.episode.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "episode_id": episode_id,
        "condition": condition,
        "claude_session_id": claude_session_id,
        "working_directory": working_directory,
        "lineage_ground_truth": {
            "source": "claude_stream_validation_only",
            "available": bool(lineage_links),
            "routing_input": False,
            "links": sorted(
                lineage_links,
                key=lambda link: (
                    link["parent_tool_use_id"], link["child_message_id"]
                ),
            ),
        },
        "checkpointing": {
            "provider": "claude-code",
            "tracks": ["Write", "Edit", "NotebookEdit"],
            "does_not_track": ["Bash", "external_changes"],
            "initial_checkpoint_id": (
                checkpoints[0]["checkpoint_id"] if checkpoints else None
            ),
            "latest_checkpoint_id": (
                checkpoints[-1]["checkpoint_id"] if checkpoints else None
            ),
            "checkpoints": checkpoints,
            "green_checkpoint_ids": [],
            "rewind_requires_explicit_apply": True,
        },
    }


def rewind_command(
    metadata: dict[str, Any], checkpoint: str, claude_bin: str
) -> list[str]:
    checkpointing = metadata.get("checkpointing") or {}
    known = [
        str(item.get("checkpoint_id"))
        for item in checkpointing.get("checkpoints") or []
        if isinstance(item, dict) and item.get("checkpoint_id")
    ]
    if checkpoint in {"start", "initial"}:
        checkpoint_id = checkpointing.get("initial_checkpoint_id")
    elif checkpoint == "latest":
        checkpoint_id = checkpointing.get("latest_checkpoint_id")
    elif checkpoint == "green":
        green = checkpointing.get("green_checkpoint_ids") or []
        checkpoint_id = green[-1] if green else None
    else:
        checkpoint_id = checkpoint
    if not checkpoint_id or str(checkpoint_id) not in known:
        raise ValueError(f"unknown checkpoint: {checkpoint}")
    session_id = metadata.get("claude_session_id")
    if not session_id:
        raise ValueError("episode metadata has no Claude session ID")
    return [
        claude_bin,
        "-p",
        "--resume",
        str(session_id),
        "--rewind-files",
        str(checkpoint_id),
    ]


def _input_command(args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt:
        print("episode input failed: prompt is empty", file=sys.stderr)
        return 2
    print(json.dumps(build_user_event(prompt), ensure_ascii=False))
    return 0


def _capture_command(args: argparse.Namespace) -> int:
    try:
        with args.stream.open(encoding="utf-8") as fh:
            metadata = capture_episode_metadata(
                read_events(fh),
                experiment_id=args.experiment_id,
                episode_id=args.episode_id,
                condition=args.condition,
                expected_session_id=args.session_id,
                working_directory=args.working_directory,
            )
        if (
            args.require_checkpoint
            and not metadata["checkpointing"]["initial_checkpoint_id"]
        ):
            raise ValueError(
                "Claude stream has no replayed main-user UUID; checkpoint unavailable"
            )
        args.output.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"episode capture failed: {exc}", file=sys.stderr)
        return 2
    return 0


def _rewind_command(args: argparse.Namespace) -> int:
    try:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        command = rewind_command(metadata, args.checkpoint, args.claude_bin)
        working_directory = metadata.get("working_directory")
        if not args.apply:
            print(shlex.join(command))
            return 0
        if not working_directory:
            raise ValueError("episode metadata has no working directory")
        return subprocess.run(command, cwd=working_directory, check=False).returncode
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"episode rewind failed: {exc}", file=sys.stderr)
        return 2


def _mark_green_command(args: argparse.Namespace) -> int:
    try:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        checkpointing = metadata.get("checkpointing") or {}
        known = [
            str(item.get("checkpoint_id"))
            for item in checkpointing.get("checkpoints") or []
            if isinstance(item, dict) and item.get("checkpoint_id")
        ]
        checkpoint_id = (
            checkpointing.get("latest_checkpoint_id")
            if args.checkpoint == "latest"
            else args.checkpoint
        )
        if not checkpoint_id or str(checkpoint_id) not in known:
            raise ValueError(f"unknown checkpoint: {args.checkpoint}")
        green = checkpointing.setdefault("green_checkpoint_ids", [])
        if str(checkpoint_id) not in green:
            green.append(str(checkpoint_id))
        args.metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"green checkpoint failed: {exc}", file=sys.stderr)
        return 2
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    input_parser = subparsers.add_parser("input", help="emit one stream-JSON prompt")
    input_parser.add_argument("--prompt", help="prompt text; defaults to stdin")
    input_parser.set_defaults(run=_input_command)

    capture_parser = subparsers.add_parser(
        "capture", help="write prompt-free episode/checkpoint metadata"
    )
    capture_parser.add_argument("stream", type=Path)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--experiment-id", required=True)
    capture_parser.add_argument("--episode-id", required=True)
    capture_parser.add_argument("--condition", required=True)
    capture_parser.add_argument("--session-id")
    capture_parser.add_argument("--working-directory")
    capture_parser.add_argument("--require-checkpoint", action="store_true")
    capture_parser.set_defaults(run=_capture_command)

    green_parser = subparsers.add_parser(
        "mark-green",
        help="mark an exposed Claude checkpoint after the caller's tests pass",
    )
    green_parser.add_argument("metadata", type=Path)
    green_parser.add_argument("--checkpoint", default="latest")
    green_parser.set_defaults(run=_mark_green_command)

    rewind_parser = subparsers.add_parser(
        "rewind", help="show or explicitly apply a Claude file rewind"
    )
    rewind_parser.add_argument("metadata", type=Path)
    rewind_parser.add_argument("--checkpoint", default="latest")
    rewind_parser.add_argument("--claude-bin", default="claude")
    rewind_parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the rewind; without this flag only print the command",
    )
    rewind_parser.set_defaults(run=_rewind_command)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
