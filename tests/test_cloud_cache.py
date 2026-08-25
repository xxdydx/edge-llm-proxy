import unittest

from edgeproxy.cloud_cache import (
    CloudCacheTracker,
    cache_scope,
    cloud_cache_trace,
    prefix_chain,
    static_lineage_key,
)
from edgeproxy.trace.record import SSEDecoder


def request_with_system(
    text: str,
    *,
    ttl: str = "5m",
    model: str = "claude-sonnet-5",
) -> dict:
    control = {"type": "ephemeral"}
    if ttl != "5m":
        control["ttl"] = ttl
    return {
        "model": model,
        "max_tokens": 64,
        "stream": True,
        "system": [{"type": "text", "text": text, "cache_control": control}],
        "messages": [{"role": "user", "content": "reply briefly"}],
    }


class PrefixIdentityTests(unittest.TestCase):
    def test_sibling_messages_share_static_lineage(self):
        first = request_with_system("S" * 5000)
        second = request_with_system("S" * 5000)
        second["messages"] = [{"role": "user", "content": "different suffix"}]
        self.assertEqual(static_lineage_key(first), static_lineage_key(second))

    def test_dictionary_order_does_not_change_static_lineage(self):
        first = request_with_system("S" * 5000)
        first["tools"] = [
            {"name": "read", "description": "read", "input_schema": {"type": "object"}}
        ]
        second = request_with_system("S" * 5000)
        second["tools"] = [
            {"input_schema": {"type": "object"}, "description": "read", "name": "read"}
        ]
        self.assertEqual(static_lineage_key(first), static_lineage_key(second))

    def test_meaningful_prompt_change_changes_exact_key(self):
        first = prefix_chain(request_with_system("A" * 5000), "scope")
        second = prefix_chain(request_with_system("B" * 5000), "scope")
        self.assertNotEqual(first.points[-1].key, second.points[-1].key)

    def test_scope_is_opaque_and_credential_specific(self):
        first = cache_scope("https://example.test/claude", {"authorization": "secret-one"})
        second = cache_scope("https://example.test/claude", {"authorization": "secret-two"})
        self.assertNotEqual(first, second)
        self.assertNotIn("secret", first)

    def test_top_level_automatic_cache_targets_last_block(self):
        request = request_with_system("S" * 5000)
        request["system"][0].pop("cache_control")
        request["cache_control"] = {"type": "ephemeral"}
        chain = prefix_chain(request, "scope")
        self.assertTrue(chain.valid)
        self.assertEqual(len(chain.breakpoints), 1)
        self.assertTrue(chain.breakpoints[0].automatic)
        self.assertEqual(chain.breakpoints[0].point_index, len(chain.points) - 1)

    def test_conflicting_automatic_ttl_disables_chain(self):
        request = request_with_system("S" * 5000)
        request["messages"][0]["content"] = [
            {
                "type": "text",
                "text": "suffix",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]
        request["cache_control"] = {"type": "ephemeral"}
        chain = prefix_chain(request, "scope")
        self.assertFalse(chain.valid)
        self.assertEqual(chain.disabled_reason, "conflicting-automatic-breakpoint-ttl")

    def test_ttl_metadata_does_not_change_content_key(self):
        five = prefix_chain(request_with_system("S" * 5000, ttl="5m"), "scope")
        hour = prefix_chain(request_with_system("S" * 5000, ttl="1h"), "scope")
        self.assertEqual(five.points[-1].key, hour.points[-1].key)
        self.assertNotEqual(five.breakpoints[0].ttl_s, hour.breakpoints[0].ttl_s)

    def test_more_than_four_breakpoints_disables_chain(self):
        request = request_with_system("unused")
        request["system"] = [
            {
                "type": "text",
                "text": f"block-{index}",
                "cache_control": {"type": "ephemeral"},
            }
            for index in range(5)
        ]
        chain = prefix_chain(request, "scope")
        self.assertFalse(chain.valid)
        self.assertEqual(chain.disabled_reason, "too-many-breakpoints")

    def test_longer_ttl_must_precede_shorter_ttl(self):
        request = request_with_system("unused")
        request["system"] = [
            system_block
            for system_block in (
                {
                    "type": "text",
                    "text": "short first",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "long second",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
            )
        ]
        chain = prefix_chain(request, "scope")
        self.assertFalse(chain.valid)
        self.assertEqual(chain.disabled_reason, "invalid-ttl-order")


class TrackerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.request = request_with_system("stable " * 1000)
        self.chain = prefix_chain(self.request, "scope")
        self.tracker = CloudCacheTracker()

    def create(self, at: float = 10.0):
        prediction = self.tracker.probe(self.chain, at - 1)
        return self.tracker.observe_cloud_usage(
            self.chain,
            prediction,
            request_started_at=at - 1,
            response_started_at=at,
            status=200,
            usage={
                "input_tokens": 10,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 1800,
            },
        )

    def test_creation_admits_entry_at_response_start(self):
        self.assertEqual(self.tracker.probe(self.chain, 1).state, "unknown")
        observation = self.create(10)
        self.assertEqual(observation.entries_created, 1)
        warm = self.tracker.probe(self.chain, 11)
        self.assertEqual(warm.state, "warm")
        entry = self.tracker.entries[warm.matched_prefix_hash]
        self.assertEqual(entry.created_at, 10)
        self.assertEqual(entry.expires_at, 310)

    def test_probe_does_not_eagerly_refresh(self):
        self.create(10)
        warm = self.tracker.probe(self.chain, 100)
        expiry = self.tracker.entries[warm.matched_prefix_hash].expires_at
        self.tracker.probe(self.chain, 200)
        self.assertEqual(self.tracker.entries[warm.matched_prefix_hash].expires_at, expiry)

    def test_confirmed_read_refreshes_from_request_start(self):
        self.create(10)
        prediction = self.tracker.probe(self.chain, 250)
        self.tracker.observe_cloud_usage(
            self.chain,
            prediction,
            request_started_at=250,
            response_started_at=252,
            status=200,
            usage={
                "input_tokens": 10,
                "cache_read_input_tokens": prediction.estimated_read_tokens,
                "cache_creation_input_tokens": 0,
            },
        )
        entry = self.tracker.entries[prediction.matched_prefix_hash]
        self.assertEqual(entry.expires_at, 550)
        self.assertEqual(self.tracker.probe(self.chain, 311).state, "warm")

    def test_expired_entry_becomes_cold(self):
        self.create(10)
        prediction = self.tracker.probe(self.chain, 311)
        self.assertEqual(prediction.state, "cold")
        self.assertEqual(prediction.reason, "expired-entry")

    def test_false_warm_is_invalidated(self):
        self.create(10)
        prediction = self.tracker.probe(self.chain, 20)
        observation = self.tracker.observe_cloud_usage(
            self.chain,
            prediction,
            request_started_at=20,
            response_started_at=21,
            status=200,
            usage={
                "input_tokens": 1800,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        )
        self.assertEqual(observation.invalidated_prefix_hash, prediction.matched_prefix_hash)
        self.assertNotIn(prediction.matched_prefix_hash, self.tracker.entries)

    def test_failure_and_missing_usage_do_not_mutate(self):
        prediction = self.tracker.probe(self.chain, 1)
        for status, usage in ((500, {}), (200, {"input_tokens": 10})):
            result = self.tracker.observe_cloud_usage(
                self.chain,
                prediction,
                request_started_at=1,
                response_started_at=2,
                status=status,
                usage=usage,
            )
            self.assertFalse(result.applied)
        self.assertFalse(self.tracker.entries)

    def test_seen_lineage_does_not_cross_cache_scope(self):
        self.create(10)
        other_scope = prefix_chain(self.request, "different-scope")
        self.assertEqual(self.tracker.probe(other_scope, 11).state, "unknown")

    def test_below_minimum_prompt_does_not_admit_creation(self):
        request = request_with_system("short")
        chain = prefix_chain(request, "scope")
        prediction = self.tracker.probe(chain, 1)
        result = self.tracker.observe_cloud_usage(
            chain,
            prediction,
            request_started_at=1,
            response_started_at=2,
            status=200,
            usage={
                "input_tokens": 10,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        )
        self.assertEqual(result.entries_created, 0)

    def test_trace_compares_prediction_to_actual(self):
        self.create(10)
        prediction = self.tracker.probe(self.chain, 20)
        trace = cloud_cache_trace(
            prediction,
            {
                "input_tokens": 10,
                "cache_read_input_tokens": prediction.estimated_read_tokens,
                "cache_creation_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0},
            },
        )
        self.assertTrue(trace["agreement"]["warm_prediction_correct"])
        self.assertEqual(trace["agreement"]["cached_token_error"], 0)

    def test_trace_does_not_call_plain_input_uncached_without_cache_detail(self):
        prediction = self.tracker.probe(self.chain, 1)
        trace = cloud_cache_trace(prediction, {"input_tokens": 1800})
        self.assertIsNone(trace["actual"]["uncached_input_tokens"])

    def test_expired_deep_entry_falls_back_to_hour_shallow_entry(self):
        request = {
            "model": "claude-sonnet-5",
            "system": [
                {
                    "type": "text",
                    "text": "stable " * 1000,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
                {
                    "type": "text",
                    "text": "less stable " * 1000,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [{"role": "user", "content": "reply"}],
        }
        chain = prefix_chain(request, "scope")
        prediction = self.tracker.probe(chain, 1)
        self.tracker.observe_cloud_usage(
            chain,
            prediction,
            request_started_at=1,
            response_started_at=2,
            status=200,
            usage={
                "input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 3000,
            },
        )
        fallback = self.tracker.probe(chain, 400)
        self.assertEqual(fallback.state, "warm")
        self.assertEqual(fallback.matched_point_index, 0)
        self.assertEqual(fallback.ttl_s, 3600)


class LookbackTests(unittest.TestCase):
    def setUp(self):
        self.base = {"type": "text", "text": "base " * 1200}
        initial = {
            "model": "claude-sonnet-5",
            "system": [
                {**self.base, "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [],
        }
        self.tracker = CloudCacheTracker()
        chain = prefix_chain(initial, "scope")
        prediction = self.tracker.probe(chain, 1)
        self.tracker.observe_cloud_usage(
            chain,
            prediction,
            request_started_at=1,
            response_started_at=2,
            status=200,
            usage={
                "input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 1500,
            },
        )

    def extended(self, added: int):
        blocks = [dict(self.base)]
        blocks.extend({"type": "text", "text": f"suffix-{index}"} for index in range(added))
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return prefix_chain(
            {"model": "claude-sonnet-5", "system": blocks, "messages": []}, "scope"
        )

    def test_old_breakpoint_is_found_nineteen_blocks_back(self):
        self.assertEqual(self.tracker.probe(self.extended(19), 3).state, "warm")

    def test_old_breakpoint_is_not_found_twenty_blocks_back(self):
        self.assertNotEqual(self.tracker.probe(self.extended(20), 3).state, "warm")


class SSEDecoderTests(unittest.TestCase):
    def test_event_marker_can_be_split_across_chunks(self):
        decoder = SSEDecoder()
        self.assertEqual(decoder.feed(b"event: x\ndata: {\"type\":\"content_"), [])
        events = decoder.feed(b"block_delta\",\"index\":0}\n\n")
        self.assertEqual(events[0]["type"], "content_block_delta")


if __name__ == "__main__":
    unittest.main()
