"""Repo-maintained judge launcher + snapshot integrity audit.

Moved out of the session scratchpad (2026-09-16) so it's testable and
reusable rather than a throwaway script. Preserves the two properties that
mattered in practice: automatic discovery of every non-superseded replay
batch (a hardcoded file list silently capped an earlier run at 117/140
calls), and full resumability (a killed run loses at most one in-flight
judge call, never previously-written rows).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import judge

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# v3 omitted the top-level system prompt; keep its 410 judgments as historical
# evidence, but never mix them with the corrected v4 full-state judgments.
DEFAULT_OUT_NAME = "stage1_judge_consistency_v4.jsonl"


def discover_replay_batches(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Every real (non-superseded, non-pilot) replay batch file, in a
    stable sorted order. Glob-based on purpose: a hardcoded filename list
    is exactly what silently dropped batch3 from an earlier real run."""
    return sorted(
        p for p in results_dir.glob("stage1_replay_examples*.jsonl")
        if "superseded" not in p.name.casefold() and "pilot" not in p.name.casefold()
    )


class CorruptedBatchError(RuntimeError):
    """Raised when a replay batch file has a line that fails to parse.
    Fails CLOSED on purpose: this file is the dedup source of truth for
    "has this call already been replayed" -- silently skipping a bad line
    would understate what's already done and risk a duplicate,
    spend-generating replay of a call that in fact already has a result
    sitting right next to the corrupted line. The caller must quarantine
    the file and repair/re-verify it before any further replay proceeds."""


def load_paired_examples(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Append-only + per-example flush (see replay.py callers) means only
    the LAST line of a file can ever be malformed, from a kill mid-write.
    Even so: raises rather than skipping, because this file's completeness
    is exactly what downstream dedup trusts."""
    rows: list[dict[str, Any]] = []
    for p in discover_replay_batches(results_dir):
        for lineno, line in enumerate(open(p), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CorruptedBatchError(
                    f"{p}: line {lineno} failed to parse ({exc}). This file is a dedup source of "
                    f"truth -- quarantine it (move aside, do not delete) and repair/re-verify before "
                    f"any further replay proceeds, rather than skip and risk a duplicate spend."
                ) from exc
    return rows


def run(
    results_dir: Path = RESULTS_DIR,
    out_name: str = DEFAULT_OUT_NAME,
) -> dict[str, Any]:
    """Judge every currently-available paired example (primary + reversed),
    resumable. Returns a summary dict; does not print/log -- callers own
    presentation."""
    rows = load_paired_examples(results_dir)
    out_path = results_dir / out_name
    judge.judge_examples_with_consistency(rows, out_path)
    all_rows = [json.loads(line) for line in open(out_path) if line.strip()]
    return audit_snapshot(all_rows, expected_call_count=len(rows))


def audit_snapshot(rows: list[dict[str, Any]], expected_call_count: int | None = None) -> dict[str, Any]:
    """The integrity checks a judge snapshot must pass before any downstream
    analysis trusts it: unique call_ids, exactly one primary + one reversed
    pass per call, zero duplicate (call_id, pass_label) pairs, every v3
    provenance invariant, and a normalization audit (every
    verdict_normalization_applied row carries a real original_parsed_token,
    and no row claims a token without the flag)."""
    n = len(rows)
    call_ids = {r["call_id"] for r in rows}

    pair_counts = Counter((r["call_id"], r["pass_label"]) for r in rows)
    duplicates = {k: v for k, v in pair_counts.items() if v > 1}

    by_call: dict[str, set[str]] = {}
    for r in rows:
        by_call.setdefault(r["call_id"], set()).add(r["pass_label"])
    missing_primary = [c for c, labels in by_call.items() if "primary" not in labels]
    missing_reversed = [c for c, labels in by_call.items() if "reversed" not in labels]
    complete_pairs = sum(1 for labels in by_call.values() if labels == {"primary", "reversed"})

    verdict_counts = Counter(r["verdict"] for r in rows)

    provenance_violations = []
    for r in rows:
        issues = []
        if r.get("truncation_policy_version") != judge.TRUNCATION_POLICY_VERSION:
            issues.append("wrong_policy_version")
        if r.get("full_state_used") is not True:
            issues.append("full_state_used_false")
        if r.get("omitted_message_count", -1) != 0:
            issues.append("nonzero_omitted")
        if r.get("omitted_message_range") is not None:
            issues.append("nonempty_omitted_range")
        if r.get("tool_schema_complete") is not True:
            issues.append("tool_schema_incomplete")
        if r.get("system_complete") is not True:
            issues.append("system_incomplete")
        if r.get("task_anchor_present") is not True:
            issues.append("no_task_anchor")
        if not r.get("prompt_chars"):
            issues.append("missing_prompt_chars")
        if not r.get("estimated_prompt_tokens"):
            issues.append("missing_estimated_tokens")
        if issues:
            provenance_violations.append({"call_id": r["call_id"], "pass_label": r["pass_label"], "issues": issues})

    normalization_applied = [r for r in rows if r.get("verdict_normalization_applied")]
    normalization_missing_token = [r for r in normalization_applied if not r.get("original_parsed_token")]
    inconsistent_token_without_flag = [
        r for r in rows if not r.get("verdict_normalization_applied") and r.get("original_parsed_token")
    ]

    return {
        "n_rows": n,
        "n_unique_call_ids": len(call_ids),
        "expected_call_count": expected_call_count,
        "call_count_matches_expected": (expected_call_count is None or len(call_ids) == expected_call_count),
        "n_duplicate_pairs": len(duplicates),
        "duplicate_pairs": list(duplicates.keys())[:10],
        "n_missing_primary": len(missing_primary),
        "n_missing_reversed": len(missing_reversed),
        "n_complete_pairs": complete_pairs,
        "n_calls": len(by_call),
        "pairs_complete": complete_pairs == len(by_call),
        "verdict_counts": dict(verdict_counts),
        "n_parse_errors": verdict_counts.get("PARSE_ERROR", 0),
        "n_provenance_violations": len(provenance_violations),
        "provenance_violations": provenance_violations[:10],
        "provenance_clean": len(provenance_violations) == 0,
        "n_normalization_applied": len(normalization_applied),
        "n_normalization_missing_token": len(normalization_missing_token),
        "n_inconsistent_token_without_flag": len(inconsistent_token_without_flag),
        "normalization_audit_clean": (
            len(normalization_missing_token) == 0 and len(inconsistent_token_without_flag) == 0
        ),
        "snapshot_clean": (
            len(duplicates) == 0
            and complete_pairs == len(by_call)
            and verdict_counts.get("PARSE_ERROR", 0) == 0
            and len(provenance_violations) == 0
            and len(normalization_missing_token) == 0
            and len(inconsistent_token_without_flag) == 0
            and (expected_call_count is None or len(call_ids) == expected_call_count)
        ),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
