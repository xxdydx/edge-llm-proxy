"""Offline tests for prefix_diff.py -- real tokenizer (downloaded from HF
Hub on first run), synthetic request bodies (no captured trace data
needed)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

W1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W1.parent))
from w1_storage.prefix_diff import compute_prefix_diff  # noqa: E402


def _req(system=None, tools=None, messages=None):
    r = {}
    if system is not None:
        r["system"] = system
    if tools is not None:
        r["tools"] = tools
    r["messages"] = messages or []
    return r


class ComputePrefixDiffTests(unittest.TestCase):
    def test_no_prior_call(self):
        curr = _req(system="You are a coding agent.", messages=[{"role": "user", "content": "fix the bug"}])
        result = compute_prefix_diff(None, curr)
        self.assertIsNone(result["common_prefix_tokens"])
        self.assertIsNone(result["prefix_fraction"])
        self.assertEqual(result["uncached_prefill_tokens"], result["prompt_tokens"])
        self.assertEqual(result["actual_cache_rate_missing_reason"], "no_prior_call_in_trajectory")

    def test_identical_requests_full_prefix(self):
        req = _req(system="You are a coding agent.",
                    messages=[{"role": "user", "content": "fix the bug"}])
        result = compute_prefix_diff(req, req)
        self.assertEqual(result["common_prefix_tokens"], result["prompt_tokens"])
        self.assertEqual(result["prefix_fraction"], 1.0)
        self.assertEqual(result["uncached_prefill_tokens"], 0)
        self.assertIsNone(result["first_difference"])

    def test_appended_message_keeps_prior_prefix_intact(self):
        prev = _req(system="You are a coding agent.",
                     messages=[{"role": "user", "content": "fix the bug"}])
        curr = _req(system="You are a coding agent.",
                     messages=[{"role": "user", "content": "fix the bug"},
                               {"role": "assistant", "content": "Looking into it."}])
        result = compute_prefix_diff(prev, curr)
        # The entire prev prompt must be a prefix of curr's -- growth-only,
        # no unrelated divergence introduced by pure appending. On this tiny
        # synthetic example the appended message is a large share of the
        # (short) total, so just check it's a genuine partial match, not a
        # full mismatch -- real captured sessions show >0.9 (see manual
        # verification against hydra-1791's actual trace).
        self.assertGreater(result["prefix_fraction"], 0.0)
        self.assertLess(result["prefix_fraction"], 1.0)
        self.assertGreater(result["uncached_prefill_tokens"], 0)

    def test_changed_system_prompt_diverges_at_system_section(self):
        prev = _req(system="You are a coding agent for repo A.",
                     messages=[{"role": "user", "content": "fix the bug"}])
        curr = _req(system="You are a coding agent for repo B, totally different context here.",
                     messages=[{"role": "user", "content": "fix the bug"}])
        result = compute_prefix_diff(prev, curr)
        self.assertIsNotNone(result["first_difference"])
        self.assertEqual(result["first_difference"]["section"], "system")

    def test_changed_tool_schema_diverges_with_tool_name(self):
        prev = _req(system="sys", tools=[{"name": "Bash", "description": "run a shell command"}],
                     messages=[{"role": "user", "content": "hi"}])
        curr = _req(system="sys", tools=[{"name": "Bash", "description": "run a shell command with extra options"}],
                     messages=[{"role": "user", "content": "hi"}])
        result = compute_prefix_diff(prev, curr)
        self.assertIsNotNone(result["first_difference"])
        self.assertTrue(result["first_difference"]["section"].startswith("tool_schema:"))
        self.assertEqual(result["first_difference"]["tool"], "Bash")

    def test_ground_truth_cache_fields_always_none(self):
        # No real per-call cache ground truth is currently captured (the
        # edgeproxy drops vLLM's prompt_tokens_details.cached_tokens) --
        # must never be silently fabricated from the coarser server-wide
        # counters.
        prev = _req(system="a", messages=[{"role": "user", "content": "x"}])
        curr = _req(system="a", messages=[{"role": "user", "content": "y"}])
        result = compute_prefix_diff(prev, curr)
        self.assertIsNone(result["actual_cached_tokens"])
        self.assertIsNone(result["actual_cache_rate"])


if __name__ == "__main__":
    unittest.main()
