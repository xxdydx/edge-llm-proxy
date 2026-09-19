#!/usr/bin/env python3
"""Measure repeated tool_result content across sibling agents in trace cohorts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from edgeproxy.trace.graph import build_trace_graph
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "traces"
TOKENIZER_PATH = Path(
    "/Users/arul/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/"
    "snapshots/d149729398750b98c0af14eb82c78cfe92750796"
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def corpus_paths() -> list[Path]:
    candidates = list(TRACES.glob("swebench-pro-15-run-*/*.jsonl"))
    candidates += list(TRACES.glob("eval-suite-*-fanout-*/trace.jsonl"))
    candidates += list(TRACES.glob("fanout/*.jsonl"))
    candidates += list(TRACES.glob("fanout-policy-pair/*.jsonl"))
    # Avoid byte-identical copied traces (notably dated raw files + trace.jsonl).
    unique: dict[str, Path] = {}
    for path in sorted(candidates):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        unique.setdefault(digest, path)
    return list(unique.values())


def agent_id(row: dict) -> str | None:
    call = row.get("call") or {}
    causality = call.get("causality") or {}
    headers = row.get("headers") or {}
    value = causality.get("agent_id") or headers.get("x-claude-code-agent-id")
    return str(value) if value else None


def total_input_tokens(row: dict) -> int | None:
    call = row.get("call") or {}
    tokens = call.get("tokens") or {}
    accounting = row.get("token_accounting") or {}
    usage = row.get("usage") or {}
    for value in (
        tokens.get("total_input_tokens"),
        tokens.get("input_tokens"),
        accounting.get("total_input_tokens"),
        accounting.get("input_tokens"),
        usage.get("input_tokens"),
    ):
        if isinstance(value, int):
            return value
    return None


def request_pieces(request: dict):
    yield json.dumps(request.get("system") or [], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    yield json.dumps(request.get("tools") or [], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for message in request.get("messages") or []:
        yield json.dumps({"role": message.get("role")}, separators=(",", ":"))
        content = message.get("content") if isinstance(message, dict) else None
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            yield json.dumps(block, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def iter_tool_results(request: dict, tokenizer):
    """Yield unique content plus offset in a deterministic tokenized history."""
    cursor = 0
    for piece in request_pieces(request):
        block = json.loads(piece)
        encoded_tokens = tokenizer.encode(piece, add_special_tokens=False)
        if isinstance(block, dict):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                raw = block.get("content")
                canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                yield str(block.get("tool_use_id") or ""), canonical, cursor
        cursor += len(encoded_tokens)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    paths = corpus_paths()
    corpus_tokens = 0
    canonical_corpus_tokens = 0
    rows_with_tokens = 0
    all_rows = 0
    cohort_count = 0
    eligible_cohorts = 0
    sibling_agents: set[tuple[str, str]] = set()
    occurrences: list[dict] = []

    for path in paths:
        rows = read_jsonl(path)
        all_rows += len(rows)
        for row in rows:
            value = total_input_tokens(row)
            if value is not None:
                corpus_tokens += value
                rows_with_tokens += 1
            canonical_corpus_tokens += sum(
                len(tokenizer.encode(piece, add_special_tokens=False))
                for piece in request_pieces(row.get("request") or {})
            )
        graph = build_trace_graph(rows)
        cohorts = graph.get("cohorts") or []
        cohort_count += len(cohorts)
        by_agent: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            aid = agent_id(row)
            if aid:
                by_agent[aid].append(row)
        for cohort in cohorts:
            agents = [str(x) for x in cohort.get("linked_agent_ids") or []]
            if len(agents) < 2:
                continue
            eligible_cohorts += 1
            cohort_id = f"{path}:{cohort['cohort_id']}"
            for aid in agents:
                sibling_agents.add((cohort_id, aid))
                # A trace request is a cumulative history. Count each tool-use result
                # occurrence once per sibling, at its first observed history snapshot.
                seen_ids: set[tuple[str, str]] = set()
                for row in sorted(by_agent.get(aid, []), key=lambda x: (x.get("ts") or 0, x.get("id") or "")):
                    for tool_id, content, token_offset in iter_tool_results(row.get("request") or {}, tokenizer):
                        digest = hashlib.sha256(content.encode()).hexdigest()
                        occurrence_key = (tool_id, digest)
                        if occurrence_key in seen_ids:
                            continue
                        seen_ids.add(occurrence_key)
                        occurrences.append({
                            "cohort": cohort_id,
                            "agent": aid,
                            "call": row.get("id"),
                            "tool_use_id": tool_id,
                            "hash": digest,
                            "content": content,
                            "token_offset": token_offset,
                            "content_tokens": len(tokenizer.encode(content, add_special_tokens=False)),
                        })

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for occurrence in occurrences:
        groups[(occurrence["cohort"], occurrence["hash"])].append(occurrence)
    duplicates = []
    for (_cohort, _digest), group in groups.items():
        if len({x["agent"] for x in group}) > 1 and len({x["token_offset"] for x in group}) > 1:
            duplicates.append(group)

    duplicate_tokens = sum(group[0]["content_tokens"] * (len(group) - 1) for group in duplicates)

    # Never emit raw tool output; traces can contain private source text.
    result = {
        "files": len(paths),
        "trace_rows": all_rows,
        "rows_with_total_input_tokens": rows_with_tokens,
        "corpus_total_input_tokens": corpus_tokens,
        "canonical_qwen25_corpus_tokens": canonical_corpus_tokens,
        "cohorts": cohort_count,
        "eligible_width_ge_2_cohorts": eligible_cohorts,
        "cohort_sibling_agents": len(sibling_agents),
        "unique_tool_result_occurrences": len(occurrences),
        "cross_sibling_duplicate_hash_groups_at_different_token_offsets": len(duplicates),
        "duplicate_occurrences": sum(len(g) - 1 for g in duplicates),
        "duplicate_tokens_beyond_first_occurrence": duplicate_tokens,
        "duplicate_fraction_of_canonical_corpus_tokens": duplicate_tokens / canonical_corpus_tokens,
        "duplicate_groups": [
            {
                "hash": group[0]["hash"],
                "occurrences": len(group),
                "agents": len({x["agent"] for x in group}),
                "token_offsets": sorted({x["token_offset"] for x in group}),
                "content_tokens": group[0]["content_tokens"],
                "utf8_bytes": len(group[0]["content"].encode()),
            }
            for group in duplicates
        ],
        "paths": [str(path.relative_to(ROOT)) for path in paths],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
