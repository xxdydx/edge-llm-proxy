"""Offline disclosure of what the blind v4 judge did and did not see.

The v4 judge prompt includes the system prompt, all renderable text/tool
history and complete tool schemas, but deliberately omits prior thinking
blocks (model-style leakage).  This audit adds omission counts and prompt
hashes *without editing or relabeling* any persisted judge result.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import judge, pilot_campaign

OUT = pilot_campaign.RESULTS / "pilot_judge_context_audit_v4.json"


def _block_counts(request: dict) -> Counter[str]:
    counts: Counter[str] = Counter()
    for msg in request.get("messages") or []:
        if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict):
                counts["non_dict"] += 1
                continue
            kind = block.get("type") or "missing_type"
            counts[str(kind)] += 1
            if kind == "tool_result" and isinstance(block.get("content"), list):
                for inner in block["content"]:
                    if isinstance(inner, dict):
                        counts[f"tool_result_inner:{inner.get('type') or 'missing_type'}"] += 1
                    else:
                        counts["tool_result_inner:non_dict"] += 1
    return counts


def audit() -> dict:
    pairs = {row["call"]["call_id"]: row for row in pilot_campaign._load_pilot_pairs()}
    judge_rows = [json.loads(line) for line in pilot_campaign.JUDGE_PATH.open() if line.strip()]
    per_call: dict[str, dict] = {}
    hashes: list[dict] = []
    total: Counter[str] = Counter()
    for call_id, pair in pairs.items():
        request = pair["call"]["request"]
        counts = _block_counts(request)
        total.update(counts)
        per_call[call_id] = {
            "message_block_types": dict(counts),
            "thinking_blocks_intentionally_omitted": counts["thinking"],
            "other_unrendered_block_types": {
                kind: n for kind, n in counts.items()
                if kind not in ("text", "tool_use", "tool_result", "thinking", "tool_result_inner:text")
            },
        }
    for row in judge_rows:
        pair = pairs[row["call_id"]]
        bundle = judge.build_full_state(pair["call"]["request"])
        by_arm = {
            "local": judge.render_response(pair["local_outcome"].get("response")),
            "cloud": judge.render_response(pair["cloud_outcome"].get("response")),
        }
        a, b = row["order"]
        prompt = judge._JUDGE_INSTRUCTIONS.format(
            system=bundle.system_text, tools=bundle.tools_text,
            context=bundle.context_text, a=by_arm[a], b=by_arm[b],
        )
        hashes.append({"call_id": row["call_id"], "pass_label": row["pass_label"],
                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "prompt_chars_match_record": len(prompt) == row["prompt_chars"]})
    return {"version": "v4-retrospective-context-disclosure", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_pairs": len(pairs), "n_judge_rows": len(judge_rows),
            "all_prompt_lengths_match_record": all(row["prompt_chars_match_record"] for row in hashes),
            "total_message_block_types": dict(total), "per_call": per_call,
            "prompt_hashes": hashes,
            "interpretation": "full visible system/task/tool history without truncation; prior thinking omitted to preserve blind comparison; not a byte-for-byte copy of the original request"}


def main() -> int:
    result = audit()
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(OUT)
    print(json.dumps({k: result[k] for k in
                      ("n_pairs", "n_judge_rows", "all_prompt_lengths_match_record", "total_message_block_types")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
