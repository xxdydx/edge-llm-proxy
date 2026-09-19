"""Synthetic-only tests; no source prompts or cloud calls."""

import json

from experiments.agentic_router import run_local_variant_judge as run


def _raw(text):
    return {"transport_err": None, "status_code": 200,
            "events": [
                {"type": "message_start", "message": {"model": "deepseek-v4-flash", "role": "assistant"}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                {"type": "message_stop"},
            ]}


def test_strict_json_and_identity():
    assert run._parse_judge(_raw('{"verdict":"A_BETTER","reason":"valid"}'))[:2] == ("A_BETTER", "ok")
    assert run._parse_judge(_raw('```json\n{"verdict":"A_BETTER","reason":"valid"}\n```'))[1] == "not_one_line"
    bad = _raw('{"verdict":"A_BETTER","reason":"valid"}')
    bad["events"][0]["message"]["model"] = "other"
    assert run._parse_judge(bad)[1] == "model_identity_mismatch"


def test_reverse_mapping_and_unknown():
    primary = {"verdict": "A_BETTER", "order": ["no_thinking", "thinking_on"]}
    reversed_ = {"verdict": "B_BETTER", "order": ["thinking_on", "no_thinking"]}
    assert run._label(primary, reversed_) == "SAFE_NO_THINK_PROXY"
    assert run._label({"verdict": "EQUIVALENT", "order": primary["order"]},
                      {"verdict": "EQUIVALENT", "order": reversed_["order"]}) == "SAFE_NO_THINK_PROXY"
    assert run._label(primary, {"verdict": "EQUIVALENT", "order": reversed_["order"]}) == "UNKNOWN"
    assert run._label(primary, {"verdict": "A_BETTER", "order": reversed_["order"]}) == "UNKNOWN"
    assert run._label({"verdict": "UNCERTAIN", "order": primary["order"]}, reversed_) == "UNKNOWN"


def test_candidate_render_drops_only_thinking():
    response = {"content": [{"type": "thinking", "thinking": "private reasoning"},
                            {"type": "text", "text": "I will read the file."},
                            {"type": "tool_use", "name": "Read", "input": {"file_path": "example.py"}}]}
    rendered = run._render_candidate(response)
    assert "private reasoning" not in rendered
    assert "I will read the file." in rendered
    assert json.dumps({"file_path": "example.py"}, sort_keys=True) in rendered
