"""Build deterministic causal, cohort, Mermaid, and tree trace views."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from ..cohort import iter_strings
from .record import consumed_tool_result_ids


CONFIDENCE_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "exact": 4}


def _hash_text(namespace: bytes, value: str) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(b"\0")
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _cohort_id(session_id: str, parent_call_id: str) -> str:
    return _hash_text(
        b"edgeproxy-cohort-v1", f"{session_id}\0{parent_call_id}"
    )[:24]


def _response_tool_inputs(response: Any) -> dict[str, dict[str, Any]]:
    content = response.get("content") if isinstance(response, dict) else None
    return {
        str(block["id"]): dict(block.get("input") or {})
        for block in content or []
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and block.get("id")
        and isinstance(block.get("input"), dict)
    }


def _common_prefix_chars(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _cache_relationship(left: str, right: str, child: dict[str, Any]) -> dict[str, Any]:
    """Summarise structural, prospective, and realised reuse without prompt text."""
    shared = _common_prefix_chars(left, right)
    child_length = len(right)
    cache_probe = child.get("cache_probe") or {}
    prediction = cache_probe.get("prediction") if isinstance(cache_probe, dict) else {}
    actual = cache_probe.get("actual") if isinstance(cache_probe, dict) else {}
    tokens = child.get("tokens") or {}
    realised = (
        actual.get("cache_read_input_tokens")
        if isinstance(actual, dict)
        else None
    )
    if realised is None:
        realised = tokens.get("cache_read_tokens")
    return {
        "structural_shared_prefix_chars": shared,
        "structural_shared_prefix_est_tokens": shared // 4,
        "structural_share_of_child": (
            round(shared / child_length, 6) if child_length else None
        ),
        "prospective_cache_state": (
            prediction.get("state") if isinstance(prediction, dict) else None
        ),
        "realized_cache_read_tokens": realised,
    }


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
    tool_inputs = _response_tool_inputs(response)
    enriched_tool_blocks: list[dict[str, Any]] = []
    for block in tool_blocks:
        if not isinstance(block, dict):
            continue
        enriched = dict(block)
        tool_id = enriched.get("tool_use_id")
        tool_input = tool_inputs.get(str(tool_id), {}) if tool_id else {}
        if enriched.get("tool_name") == "Agent":
            prompt = tool_input.get("prompt")
            enriched["_delegation_prompt"] = prompt if isinstance(prompt, str) else None
            enriched["_tool_input_valid"] = bool(
                isinstance(prompt, str) and prompt and "_unparsed" not in tool_input
            )
        enriched_tool_blocks.append(enriched)
    response_id = response.get("id")
    explicit_parent = causality.get("agent_parent_tool_use_id")
    stream_parent = stream_parents.get(str(response_id)) if response_id else None
    timing = call.get("timing") or record.get("timing") or {}
    tokens = call.get("tokens") or {}
    cohort = call.get("cohort") or record.get("cohort_detection")
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
            else str(stream_parent) if stream_parent else None
        ),
        "parent_detection_method": (
            "explicit_request"
            if explicit_parent
            else "ground_truth_stream" if stream_parent else None
        ),
        "parent_detection_confidence": (
            "exact" if explicit_parent or stream_parent else "unknown"
        ),
        "timestamp_unix_s": call.get("timestamp_unix_s", record.get("ts")),
        "completed_at_unix_s": (
            float(call.get("timestamp_unix_s", record.get("ts")) or 0)
            + float(timing.get("total_ms") or 0) / 1000
        ),
        "backend": call.get("backend", record.get("placement")),
        "http_status": call.get("http_status", record.get("status")),
        "timing": timing,
        "tokens": tokens,
        "cache_probe": call.get("cache_probe"),
        "cohort": cohort if isinstance(cohort, dict) else None,
        "line_number": record.get("_trace_line_number", 0),
        "consumed_tool_result_ids": list(
            causality.get("consumed_tool_result_ids")
            or consumed_tool_result_ids(request)
        ),
        "parent_call_ids": list(causality.get("parent_call_ids") or []),
        "root_call_id": causality.get("root_call_id"),
        "tool_use_blocks": enriched_tool_blocks,
        # Used only while linking; never copied into the graph sidecar.
        "request_strings": tuple(iter_strings(request)),
        "request_canonical": json.dumps(
            [
                request.get("model"),
                request.get("system") or [],
                request.get("tools") or [],
                request.get("messages") or [],
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
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
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    tool_producers: dict[str, str] = {}
    exact_parent_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    exact_parent_methods: dict[tuple[str, str, str], str] = {}
    content_parent_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    agent_first_seen: dict[tuple[str, str | None], tuple[float, str]] = {}
    agent_request_strings: dict[tuple[str, str], set[str]] = defaultdict(set)
    tool_prompts: dict[str, str] = {}
    tool_ready_at: dict[str, float] = {}
    call_request_text: dict[str, str] = {}
    first_call_by_agent: dict[tuple[str, str | None], str] = {}

    def add_edge(
        source: str, target: str, kind: str, **properties: Any
    ) -> None:
        key = (source, target, kind)
        edges.setdefault(key, {}).update(
            {name: value for name, value in properties.items() if value is not None}
        )

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
        call_request_text[row["call_id"]] = row["request_canonical"]
        first_call_by_agent.setdefault((session, agent), row["call_id"])
        nodes[call_key] = {
            "id": call_key,
            "type": "call",
            "call_id": row["call_id"],
            "experiment_id": row["experiment_id"],
            "episode_id": row["episode_id"],
            "session_id": session,
            "agent_id": agent,
            "timestamp_unix_s": row["timestamp_unix_s"],
            "completed_at_unix_s": row["completed_at_unix_s"],
            "backend": row["backend"],
            "http_status": row["http_status"],
            "timing": row["timing"],
            "tokens": row["tokens"],
            "cache_probe": row["cache_probe"],
            "cohort": row["cohort"],
            "parent_call_ids": row["parent_call_ids"],
            "root_call_id": row["root_call_id"],
        }
        add_edge(agent_key, call_key, "made_call")
        if agent is not None:
            agent_request_strings[(session, agent)].update(row["request_strings"])
        if agent is not None and row["agent_parent_tool_use_id"]:
            tool_id = str(row["agent_parent_tool_use_id"])
            exact_parent_candidates[(session, agent)].add(tool_id)
            exact_parent_methods[(session, agent, tool_id)] = str(
                row["parent_detection_method"] or "explicit_request"
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
                    "delegation_prompt_hash": (
                        _hash_text(
                            b"edgeproxy-delegation-prompt-v1",
                            str(block.get("_delegation_prompt")),
                        )
                        if block.get("_delegation_prompt")
                        else None
                    ),
                    "tool_input_valid": block.get("_tool_input_valid"),
                },
            )
            tool_producers.setdefault(tool_id, row["call_id"])
            tool_ready_at[tool_id] = float(row["completed_at_unix_s"] or 0)
            if block.get("_delegation_prompt"):
                tool_prompts[tool_id] = str(block["_delegation_prompt"])
            add_edge(agent_key, tool_key, "produced_tool_use")
            add_edge(call_key, tool_key, "produced_tool_use")

        for tool_id in row["consumed_tool_result_ids"]:
            tool_id = str(tool_id)
            add_edge(f"tool:{tool_id}", call_key, "result_consumed_by")

    # Infer lineage solely from proxied content. Ground-truth stream links are
    # kept separate above so this detector can be scored against them.
    for tool_id, prompt in tool_prompts.items():
        tool_node = nodes.get(f"tool:{tool_id}") or {}
        session = tool_node.get("session_id")
        for (child_session, agent), strings in agent_request_strings.items():
            if child_session != session:
                continue
            if any(prompt in text for text in strings):
                content_parent_candidates[(child_session, agent)].add(tool_id)

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
            add_edge(f"call:{parent}", f"call:{call_id}", "continued_by")

    unresolved: list[dict[str, Any]] = []
    selected_content_links: dict[tuple[str, str], str] = {}
    selected_exact_links: dict[tuple[str, str], str] = {}
    all_child_keys = sorted(
        {
            *exact_parent_candidates.keys(),
            *content_parent_candidates.keys(),
            *agent_request_strings.keys(),
        }
    )
    for session, agent in all_child_keys:
        child_key = f"agent:{session}:{agent}"
        exact = sorted(
            tool_id
            for tool_id in exact_parent_candidates.get((session, agent), set())
            if tool_id in tool_producers
        )
        content = sorted(
            tool_id
            for tool_id in content_parent_candidates.get((session, agent), set())
            if tool_id in tool_producers
        )
        inferred_content: str | None = content[0] if len(content) == 1 else None
        if len(content) > 1:
            content_first_seen = agent_first_seen.get((session, agent), (0.0, ""))[0]
            content_preceding = [
                tool_id
                for tool_id in content
                if tool_ready_at.get(tool_id, float("inf")) <= content_first_seen + 0.1
            ]
            if content_preceding:
                content_latest_time = max(
                    tool_ready_at[tool_id] for tool_id in content_preceding
                )
                content_latest = [
                    tool_id
                    for tool_id in content_preceding
                    if tool_ready_at[tool_id] == content_latest_time
                ]
                inferred_content = (
                    content_latest[0] if len(content_latest) == 1 else None
                )
        if inferred_content:
            selected_content_links[(session, agent)] = inferred_content
        if len(exact) == 1:
            selected = exact[0]
            selected_exact_links[(session, agent)] = selected
            method = exact_parent_methods.get(
                (session, agent, selected), "ground_truth_stream"
            )
            confidence = "exact"
        elif len(exact) > 1:
            selected = None
            method = "conflicting_exact_lineage"
            confidence = "unknown"
        elif len(content) == 1:
            selected = content[0]
            method = "content_exact"
            confidence = "high"
        elif len(content) > 1:
            first_seen = agent_first_seen.get((session, agent), (0.0, ""))[0]
            preceding = [
                tool_id
                for tool_id in content
                if tool_ready_at.get(tool_id, float("inf")) <= first_seen + 0.1
            ]
            if preceding:
                latest_time = max(tool_ready_at[tool_id] for tool_id in preceding)
                latest = [
                    tool_id
                    for tool_id in preceding
                    if tool_ready_at[tool_id] == latest_time
                ]
            else:
                latest = []
            selected = latest[0] if len(latest) == 1 else None
            method = "content_and_causal_time" if selected else "ambiguous_content"
            confidence = "medium" if selected else "unknown"
        else:
            selected = None
            method = "unavailable"
            confidence = "unknown"

        nodes[child_key]["parent_link"] = {
            "parent_tool_use_id": selected,
            "detection_method": method,
            "detection_confidence": confidence,
            "exact_candidates": exact,
            "content_candidates": content,
        }
        if selected:
            parent_call_id = tool_producers[selected]
            child_call_id = first_call_by_agent.get((session, agent))
            relationship = (
                _cache_relationship(
                    call_request_text.get(parent_call_id, ""),
                    call_request_text.get(child_call_id, ""),
                    nodes.get(f"call:{child_call_id}", {}),
                )
                if child_call_id
                else {}
            )
            add_edge(
                f"tool:{selected}",
                child_key,
                "spawned_agent",
                detection_method=method,
                detection_confidence=confidence,
                cache_relationship=relationship,
            )
        else:
            unresolved.append(
                {
                    "session_id": session,
                    "agent_id": agent,
                    "reason": (
                        method
                        if exact or content
                        else "parent_tool_use_id_unavailable"
                    ),
                    "candidate_parent_tool_use_ids": sorted({*exact, *content}),
                    "detection_confidence": confidence,
                }
            )

    child_agents = {
        (node["session_id"], node["agent_id"])
        for node in nodes.values()
        if node["type"] == "agent" and node["agent_id"] is not None
    }
    resolved_children = {
        (nodes[target]["session_id"], nodes[target]["agent_id"])
        for (source, target, kind), _properties in edges.items()
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
                    "detection_confidence": "unknown",
                }
            )

    validation_children = sorted(selected_exact_links)
    correct = sum(
        selected_content_links.get(child) == selected_exact_links[child]
        for child in validation_children
    )
    incorrect = sum(
        child in selected_content_links
        and selected_content_links[child] != selected_exact_links[child]
        for child in validation_children
    )
    inferred_with_ground_truth = correct + incorrect
    linker_validation = {
        "ground_truth_children": len(validation_children),
        "correct_content_links": correct,
        "incorrect_content_links": incorrect,
        "unresolved_content_links": len(validation_children) - inferred_with_ground_truth,
        "accuracy": (
            round(correct / len(validation_children), 6)
            if validation_children
            else None
        ),
        "false_cohort_rate": (
            round(incorrect / inferred_with_ground_truth, 6)
            if inferred_with_ground_truth
            else None
        ),
    }

    cohorts_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes.values():
        if (
            node.get("type") == "tool_use"
            and node.get("tool_name") == "Agent"
            and node.get("tool_input_valid") is not False
        ):
            cohorts_by_parent[str(node["producing_call_id"])].append(node)
    spawn_targets_by_tool = {
        source: target
        for (source, target, kind), _properties in edges.items()
        if kind == "spawned_agent"
    }
    cohorts: list[dict[str, Any]] = []
    for parent_call_id, tools in sorted(cohorts_by_parent.items()):
        parent = nodes.get(f"call:{parent_call_id}") or {}
        session = str(parent.get("session_id") or "unknown-session")
        linked_agents = [
            spawn_targets_by_tool[f"tool:{tool['tool_use_id']}"]
            for tool in tools
            if f"tool:{tool['tool_use_id']}" in spawn_targets_by_tool
        ]
        arrival_times = sorted(
            agent_first_seen[
                (nodes[agent_key]["session_id"], nodes[agent_key]["agent_id"])
            ][0]
            for agent_key in linked_agents
        )
        link_confidences = [
            nodes[agent_key].get("parent_link", {}).get(
                "detection_confidence", "unknown"
            )
            for agent_key in linked_agents
        ]
        cohort_confidence = (
            min(link_confidences, key=lambda value: CONFIDENCE_ORDER.get(value, 0))
            if link_confidences
            else "unknown"
        )
        cohorts.append(
            {
                "cohort_id": _cohort_id(session, parent_call_id),
                "session_id": session,
                "parent_call_id": parent_call_id,
                "parent_agent_id": parent.get("agent_id"),
                "parent_backend": parent.get("backend"),
                "primary_case": parent.get("backend") == "cloud",
                "expected_width": len(tools),
                "observed_linked_width": len(linked_agents),
                "linked_agent_ids": [nodes[key]["agent_id"] for key in linked_agents],
                "arrival_span_ms": (
                    round((arrival_times[-1] - arrival_times[0]) * 1000, 3)
                    if len(arrival_times) > 1
                    else 0.0 if arrival_times else None
                ),
                "detection_confidence": cohort_confidence,
            }
        )
        for left_index, left_agent_key in enumerate(linked_agents):
            left_node = nodes[left_agent_key]
            left_call_id = first_call_by_agent.get(
                (left_node["session_id"], left_node["agent_id"])
            )
            for right_agent_key in linked_agents[left_index + 1 :]:
                right_node = nodes[right_agent_key]
                right_call_id = first_call_by_agent.get(
                    (right_node["session_id"], right_node["agent_id"])
                )
                if not left_call_id or not right_call_id:
                    continue
                add_edge(
                    left_agent_key,
                    right_agent_key,
                    "sibling_cache_overlap",
                    cohort_id=_cohort_id(session, parent_call_id),
                    cache_relationship=_cache_relationship(
                        call_request_text.get(left_call_id, ""),
                        call_request_text.get(right_call_id, ""),
                        nodes.get(f"call:{right_call_id}", {}),
                    ),
                )

    child_call_nodes = [
        node
        for node in nodes.values()
        if node.get("type") == "call" and node.get("agent_id") is not None
    ]
    successful_child_calls = sum(
        isinstance(node.get("http_status"), int)
        and 200 <= int(node["http_status"]) < 300
        for node in child_call_nodes
    )
    analysis_eligibility = {
        "child_calls": len(child_call_nodes),
        "successful_child_calls": successful_child_calls,
        "arrival_distribution_eligible": bool(successful_child_calls),
        "exclusion_reason": (
            None
            if successful_child_calls
            else "all_child_calls_failed" if child_call_nodes else "no_child_calls"
        ),
    }

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
            {"source": source, "target": target, "type": kind, **edges[(source, target, kind)]}
            for source, target, kind in sorted(edges, key=lambda edge: (edge[2], edge[0], edge[1]))
        ],
        "cohorts": cohorts,
        "linker_validation": linker_validation,
        "analysis_eligibility": analysis_eligibility,
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


def _mermaid_id(value: str) -> str:
    return "n" + sha256(value.encode("utf-8")).hexdigest()[:12]


def _mermaid_label(value: Any) -> str:
    return html.escape(str(value), quote=True).replace("\n", " ")


def _relationship_label(edge: dict[str, Any]) -> str:
    relationship = edge.get("cache_relationship") or {}
    parts: list[str] = []
    shared_tokens = relationship.get("structural_shared_prefix_est_tokens")
    share = relationship.get("structural_share_of_child")
    if shared_tokens is not None:
        structural = f"structural ~{shared_tokens:,} tok"
        if share is not None:
            structural += f" ({float(share):.0%})"
        parts.append(structural)
    prospective = relationship.get("prospective_cache_state")
    if prospective is not None:
        parts.append(f"prospective {prospective}")
    realised = relationship.get("realized_cache_read_tokens")
    if realised is not None:
        parts.append(f"realized {int(realised):,} tok")
    confidence = edge.get("detection_confidence")
    if confidence:
        parts.append(f"link {confidence}")
    return "<br/>".join(_mermaid_label(part) for part in parts)


def render_mermaid(graph: dict[str, Any]) -> str:
    """Render the full call graph with cohort and cache metrics on its edges."""
    nodes = sorted(graph.get("nodes") or [], key=lambda node: node["id"])
    output = ["flowchart LR"]
    for node in nodes:
        mermaid_id = _mermaid_id(node["id"])
        node_type = node.get("type")
        if node_type == "agent":
            label = "Main agent" if node.get("agent_id") is None else (
                f"Child {_short(str(node.get('agent_id')))}"
            )
            output.append(f'  {mermaid_id}(["{_mermaid_label(label)}"])')
        elif node_type == "call":
            timing = node.get("timing") or {}
            label_parts = [
                f"Call {_short(str(node.get('call_id')))}",
                str(node.get("backend") or "backend unknown"),
            ]
            if timing.get("ttft_ms") is not None:
                label_parts.append(f"TTFT {float(timing['ttft_ms']):.0f} ms")
            if timing.get("total_ms") is not None:
                label_parts.append(f"total {float(timing['total_ms']):.0f} ms")
            label = "<br/>".join(_mermaid_label(part) for part in label_parts)
            output.append(f'  {mermaid_id}["{label}"]')
        else:
            label = f"{node.get('tool_name') or 'Tool'} {_short(str(node.get('tool_use_id')))}"
            output.append(f'  {mermaid_id}{{{{"{_mermaid_label(label)}"}}}}')

    visible_edges = []
    for edge in graph.get("edges") or []:
        if (
            edge.get("type") == "produced_tool_use"
            and str(edge.get("source", "")).startswith("agent:")
        ):
            continue
        visible_edges.append(edge)
    for edge in sorted(
        visible_edges,
        key=lambda item: (item["type"], item["source"], item["target"]),
    ):
        source = _mermaid_id(edge["source"])
        target = _mermaid_id(edge["target"])
        kind = edge["type"]
        if kind in {"spawned_agent", "sibling_cache_overlap"}:
            label = _relationship_label(edge)
        else:
            label = _mermaid_label(kind.replace("_", " "))
        arrow = "-.->" if kind == "sibling_cache_overlap" else "-->"
        label_clause = f'|"{label}"|' if label else ""
        output.append(f"  {source} {arrow}{label_clause} {target}")

    output.extend(
        [
            "  classDef agentNode fill:#e8f1ff,stroke:#3563a9,color:#10233f",
            "  classDef callNode fill:#fff7df,stroke:#9b7326,color:#35260c",
            "  classDef toolNode fill:#e9f8ee,stroke:#39764a,color:#15301d",
        ]
    )
    for node_type, class_name in (
        ("agent", "agentNode"),
        ("call", "callNode"),
        ("tool_use", "toolNode"),
    ):
        ids = [_mermaid_id(node["id"]) for node in nodes if node.get("type") == node_type]
        if ids:
            output.append(f"  class {','.join(ids)} {class_name}")
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
    parser.add_argument("--mermaid-output", type=Path, help="write the Mermaid graph here")
    args = parser.parse_args()

    records = _read_jsonl(args.trace)
    stream_events = _read_jsonl(args.claude_stream) if args.claude_stream else []
    graph = build_trace_graph(records, stream_events=stream_events)
    tree = render_tree(graph)
    mermaid = render_mermaid(graph)
    if args.json_output:
        args.json_output.write_text(
            json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.tree_output:
        args.tree_output.write_text(tree, encoding="utf-8")
    if args.mermaid_output:
        args.mermaid_output.write_text(mermaid, encoding="utf-8")
    if not args.json_output and not args.tree_output and not args.mermaid_output:
        print(mermaid, end="")


if __name__ == "__main__":
    main()
