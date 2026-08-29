"""Build a deterministic causal graph and readable tree from EdgeProxy traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .record import consumed_tool_result_ids


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                value["_trace_line_number"] = line_number
                records.append(value)
    return records


def _normalise(record: dict[str, Any], stream_parents: dict[str, str]) -> dict[str, Any]:
    call = record.get("call") or {}
    causality = call.get("causality") or {}
    headers = record.get("headers") or {}
    response = record.get("response") or {}
    request = record.get("request") or {}
    tool_blocks = call.get("tool_use_blocks")
    if not isinstance(tool_blocks, list):
        tool_blocks = [
            {
                "content_block_index": index,
                "tool_use_id": block.get("id"),
                "tool_name": block.get("name"),
                "schema_valid": None,
                "validation_error": None,
            }
            for index, block in enumerate(response.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
    response_id = response.get("id")
    explicit_parent = causality.get("agent_parent_tool_use_id")
    return {
        "call_id": str(call.get("call_id") or record.get("id")),
        "experiment_id": call.get("experiment_id", record.get("experiment_id")),
        "episode_id": call.get("episode_id", record.get("episode_id")),
        "session_id": str(
            call.get("session_id")
            or headers.get("x-claude-code-session-id")
            or "unknown-session"
        ),
        "agent_id": (
            str(causality.get("agent_id") or headers.get("x-claude-code-agent-id"))
            if causality.get("agent_id") or headers.get("x-claude-code-agent-id")
            else None
        ),
        "agent_parent_tool_use_id": (
            str(explicit_parent)
            if explicit_parent
            else stream_parents.get(str(response_id)) if response_id else None
        ),
        "timestamp_unix_s": call.get("timestamp_unix_s", record.get("ts")),
        "line_number": record.get("_trace_line_number", 0),
        "consumed_tool_result_ids": list(
            causality.get("consumed_tool_result_ids")
            or consumed_tool_result_ids(request)
        ),
        "parent_call_ids": list(causality.get("parent_call_ids") or []),
        "root_call_id": causality.get("root_call_id"),
        "tool_use_blocks": tool_blocks,
    }


def build_trace_graph(
    records: Iterable[dict[str, Any]],
    *,
    stream_events: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Return a stable graph; input order is never used as semantic order."""
    stream_parents: dict[str, str] = {}
    for event in stream_events:
        message = event.get("message") or {}
        message_id = message.get("id") if isinstance(message, dict) else None
        parent_id = event.get("parent_tool_use_id")
        if parent_id is None and isinstance(message, dict):
            parent_id = message.get("parent_tool_use_id")
        if message_id and parent_id:
            prior = stream_parents.get(str(message_id))
            stream_parents[str(message_id)] = min(
                str(parent_id), prior or str(parent_id)
            )

    normalised = [_normalise(dict(record), stream_parents) for record in records]
    normalised.sort(
        key=lambda row: (
            float(row["timestamp_unix_s"] or 0),
            row["call_id"],
            int(row["line_number"] or 0),
        )
    )

    nodes: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str, str]] = set()
    tool_producers: dict[str, str] = {}
    child_parent_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    agent_first_seen: dict[tuple[str, str | None], tuple[float, str]] = {}

    for row in normalised:
        session = row["session_id"]
        agent = row["agent_id"]
        agent_key = f"agent:{session}:{agent or 'main'}"
        first_key = (float(row["timestamp_unix_s"] or 0), row["call_id"])
        agent_first_seen[(session, agent)] = min(
            first_key, agent_first_seen.get((session, agent), first_key)
        )
        nodes.setdefault(
            agent_key,
            {
                "id": agent_key,
                "type": "agent",
                "session_id": session,
                "agent_id": agent,
                "label": "Main agent" if agent is None else "Child agent",
            },
        )
        call_key = f"call:{row['call_id']}"
        nodes[call_key] = {
            "id": call_key,
            "type": "call",
            "call_id": row["call_id"],
            "experiment_id": row["experiment_id"],
            "episode_id": row["episode_id"],
            "session_id": session,
            "agent_id": agent,
            "timestamp_unix_s": row["timestamp_unix_s"],
            "parent_call_ids": row["parent_call_ids"],
            "root_call_id": row["root_call_id"],
        }
        edges.add((agent_key, call_key, "made_call"))
        if agent is not None and row["agent_parent_tool_use_id"]:
            child_parent_candidates[(session, agent)].add(
                str(row["agent_parent_tool_use_id"])
            )

        for block in row["tool_use_blocks"]:
            tool_id = block.get("tool_use_id")
            if not tool_id:
                continue
            tool_id = str(tool_id)
            tool_key = f"tool:{tool_id}"
            nodes.setdefault(
                tool_key,
                {
                    "id": tool_key,
                    "type": "tool_use",
                    "tool_use_id": tool_id,
                    "tool_name": block.get("tool_name") or "Unknown tool",
                    "schema_valid": block.get("schema_valid"),
                    "validation_error": block.get("validation_error"),
                    "session_id": session,
                    "agent_id": agent,
                    "timestamp_unix_s": row["timestamp_unix_s"],
                    "content_block_index": block.get("content_block_index", 0),
                    "producing_call_id": row["call_id"],
                },
            )
            tool_producers.setdefault(tool_id, row["call_id"])
            edges.add((agent_key, tool_key, "produced_tool_use"))
            edges.add((call_key, tool_key, "produced_tool_use"))

        for tool_id in row["consumed_tool_result_ids"]:
            tool_id = str(tool_id)
            edges.add((f"tool:{tool_id}", call_key, "result_consumed_by"))

    call_roots: dict[str, str] = {}
    for row in normalised:
        call_id = row["call_id"]
        derived_parents = [
            tool_producers[str(tool_id)]
            for tool_id in row["consumed_tool_result_ids"]
            if str(tool_id) in tool_producers
        ]
        parents = list(dict.fromkeys([*row["parent_call_ids"], *derived_parents]))
        roots = list(dict.fromkeys(call_roots.get(parent, parent) for parent in parents))
        root = roots[0] if len(roots) == 1 else str(row["root_call_id"] or call_id)
        call_roots[call_id] = root
        nodes[f"call:{call_id}"]["parent_call_ids"] = parents
        nodes[f"call:{call_id}"]["root_call_id"] = root
        for parent in parents:
            edges.add((f"call:{parent}", f"call:{call_id}", "continued_by"))

    unresolved: list[dict[str, Any]] = []
    for (session, agent), candidates in sorted(child_parent_candidates.items()):
        child_key = f"agent:{session}:{agent}"
        valid = sorted(tool_id for tool_id in candidates if tool_id in tool_producers)
        if len(valid) == 1:
            edges.add((f"tool:{valid[0]}", child_key, "spawned_agent"))
        else:
            unresolved.append(
                {
                    "session_id": session,
                    "agent_id": agent,
                    "reason": (
                        "conflicting_parent_tool_use_ids"
                        if valid
                        else "parent_tool_use_not_in_trace"
                    ),
                    "candidate_parent_tool_use_ids": sorted(candidates),
                }
            )

    child_agents = {
        (node["session_id"], node["agent_id"])
        for node in nodes.values()
        if node["type"] == "agent" and node["agent_id"] is not None
    }
    resolved_children = {
        (nodes[target]["session_id"], nodes[target]["agent_id"])
        for source, target, kind in edges
        if kind == "spawned_agent" and target in nodes
    }
    for session, agent in sorted(child_agents - resolved_children):
        if not any(
            item["session_id"] == session and item["agent_id"] == agent
            for item in unresolved
        ):
            unresolved.append(
                {
                    "session_id": session,
                    "agent_id": agent,
                    "reason": "parent_tool_use_id_unavailable",
                    "candidate_parent_tool_use_ids": [],
                }
            )

    return {
        "schema_version": "edgeproxy.trace_graph.v1",
        "experiment_ids": sorted(
            {
                str(row["experiment_id"])
                for row in normalised
                if row["experiment_id"]
            }
        ),
        "episode_ids": sorted(
            {str(row["episode_id"]) for row in normalised if row["episode_id"]}
        ),
        "nodes": sorted(nodes.values(), key=lambda node: node["id"]),
        "edges": [
            {"source": source, "target": target, "type": kind}
            for source, target, kind in sorted(edges, key=lambda edge: (edge[2], edge[0], edge[1]))
        ],
        "unresolved": sorted(
            unresolved, key=lambda item: (item["session_id"], item["agent_id"])
        ),
    }


def _short(value: str | None, width: int = 12) -> str:
    if not value:
        return "unknown"
    if value.startswith("chatcmpl-tool-"):
        value = value.removeprefix("chatcmpl-tool-")
    return value if len(value) <= width else value[:width] + "…"


def render_tree(graph: dict[str, Any]) -> str:
    """Render agents and their tools as a deterministic Unicode tree."""
    nodes = {node["id"]: node for node in graph.get("nodes") or []}
    tools_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    child_by_tool: dict[str, list[str]] = defaultdict(list)
    sessions: set[str] = set()
    for node in nodes.values():
        if node["type"] == "agent":
            sessions.add(node["session_id"])
        elif node["type"] == "tool_use":
            agent_key = f"agent:{node['session_id']}:{node.get('agent_id') or 'main'}"
            tools_by_agent[agent_key].append(node)
    for edge in graph.get("edges") or []:
        if edge["type"] == "spawned_agent":
            child_by_tool[edge["source"]].append(edge["target"])
    for values in tools_by_agent.values():
        values.sort(
            key=lambda node: (
                float(node.get("timestamp_unix_s") or 0),
                int(node.get("content_block_index") or 0),
                node["tool_use_id"],
            )
        )
    for values in child_by_tool.values():
        values.sort()

    unresolved_by_session: dict[str, list[str]] = defaultdict(list)
    for item in graph.get("unresolved") or []:
        unresolved_by_session[item["session_id"]].append(item["agent_id"])

    def agent_lines(
        agent_key: str, prefix: str, connector: str, label: str
    ) -> list[str]:
        lines = [prefix + connector + label]
        child_prefix = prefix + (
            "    " if connector == "└── " else "│   " if connector else ""
        )
        tools = tools_by_agent.get(agent_key, [])
        for index, tool in enumerate(tools):
            is_last = index == len(tools) - 1
            branch = "└── " if is_last else "├── "
            continuation = "    " if is_last else "│   "
            tool_key = tool["id"]
            tool_label = f"{tool['tool_name']} [{_short(tool['tool_use_id'])}]"
            lines.append(child_prefix + branch + tool_label)
            children = child_by_tool.get(tool_key, [])
            for child_index, child_key in enumerate(children):
                child = nodes[child_key]
                child_branch = "└── " if child_index == len(children) - 1 else "├── "
                child_label = f"Child agent [{_short(child.get('agent_id'))}]"
                nested = agent_lines(
                    child_key,
                    child_prefix + continuation,
                    child_branch,
                    child_label,
                )
                lines.extend(nested)
        return lines

    output: list[str] = []
    multi_session = len(sessions) > 1
    for session_index, session in enumerate(sorted(sessions)):
        if session_index:
            output.append("")
        if multi_session:
            output.append(f"Session [{_short(session)}]")
        main_key = f"agent:{session}:main"
        unresolved_agents = sorted(set(unresolved_by_session.get(session, [])))
        if main_key in nodes:
            main_connector = "├── " if multi_session and unresolved_agents else (
                "└── " if multi_session else ""
            )
            output.extend(agent_lines(main_key, "", main_connector, "Main agent"))
        if unresolved_agents:
            group_connector = "└── "
            output.append(group_connector + "Unresolved child agents (parent ID unavailable)")
            group_prefix = "    "
            for index, agent in enumerate(unresolved_agents):
                branch = "└── " if index == len(unresolved_agents) - 1 else "├── "
                child_key = f"agent:{session}:{agent}"
                output.extend(
                    agent_lines(
                        child_key,
                        group_prefix,
                        branch,
                        f"Child agent [{_short(agent)}]",
                    )
                )
    return "\n".join(output).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="EdgeProxy JSONL trace")
    parser.add_argument(
        "--claude-stream",
        type=Path,
        help="Claude stream JSONL for exact subagent-parent edges",
    )
    parser.add_argument("--json-output", type=Path, help="write the machine-readable graph here")
    parser.add_argument("--tree-output", type=Path, help="write the human-readable tree here")
    args = parser.parse_args()

    records = _read_jsonl(args.trace)
    stream_events = _read_jsonl(args.claude_stream) if args.claude_stream else []
    graph = build_trace_graph(records, stream_events=stream_events)
    tree = render_tree(graph)
    if args.json_output:
        args.json_output.write_text(
            json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.tree_output:
        args.tree_output.write_text(tree, encoding="utf-8")
    if not args.json_output and not args.tree_output:
        print(tree, end="")


if __name__ == "__main__":
    main()
