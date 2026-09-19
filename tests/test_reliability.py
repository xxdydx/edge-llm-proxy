import random
import unittest

from edgeproxy.reliability import ReliabilityCircuitBreaker


class ReliabilityCircuitBreakerTests(unittest.TestCase):
    def test_unknown_class_never_blocks(self):
        breaker = ReliabilityCircuitBreaker()
        self.assertIsNone(breaker.failure_rate("never-seen"))
        blocked, note = breaker.should_block("never-seen")
        self.assertFalse(blocked)
        self.assertEqual(note, "reliability-ok")

    def test_below_min_samples_never_blocks_even_at_100_percent_failure(self):
        breaker = ReliabilityCircuitBreaker(min_samples=5, threshold=0.25)
        for _ in range(4):
            breaker.record("class-a", success=False)
        self.assertIsNone(breaker.failure_rate("class-a"))
        blocked, note = breaker.should_block("class-a")
        self.assertFalse(blocked)
        self.assertEqual(note, "reliability-ok")

    def test_trips_once_threshold_crossed_with_enough_samples(self):
        breaker = ReliabilityCircuitBreaker(
            min_samples=5, threshold=0.25, probe_fraction=0.0
        )
        # 2/5 = 0.4 failure rate, above the 0.25 threshold.
        for success in (True, False, True, False, True):
            breaker.record("class-a", success=success)
        self.assertAlmostEqual(breaker.failure_rate("class-a"), 0.4)
        blocked, note = breaker.should_block("class-a")
        self.assertTrue(blocked)
        self.assertEqual(note, "reliability-circuit-open")

    def test_exactly_at_threshold_does_not_trip(self):
        breaker = ReliabilityCircuitBreaker(min_samples=4, threshold=0.25)
        # 1/4 = 0.25, exactly at threshold — should not block (<=, not <).
        for success in (True, True, True, False):
            breaker.record("class-a", success=success)
        self.assertAlmostEqual(breaker.failure_rate("class-a"), 0.25)
        blocked, _ = breaker.should_block("class-a")
        self.assertFalse(blocked)

    def test_probe_fraction_lets_some_calls_through_when_open(self):
        # rng seeded so the sequence of should_block() rolls is deterministic.
        breaker = ReliabilityCircuitBreaker(
            min_samples=5,
            threshold=0.25,
            probe_fraction=0.5,
            rng=random.Random(0),
        )
        for _ in range(10):
            breaker.record("class-a", success=False)
        results = [breaker.should_block("class-a") for _ in range(200)]
        notes = {note for _, note in results}
        # With probe_fraction=0.5 over 200 draws, both outcomes must appear.
        self.assertIn("reliability-probe", notes)
        self.assertIn("reliability-circuit-open", notes)
        probe_blocked = [blocked for blocked, note in results if note == "reliability-probe"]
        self.assertTrue(all(b is False for b in probe_blocked))
        open_blocked = [blocked for blocked, note in results if note == "reliability-circuit-open"]
        self.assertTrue(all(b is True for b in open_blocked))

    def test_recovery_after_open_class_starts_succeeding_again(self):
        breaker = ReliabilityCircuitBreaker(
            window=5, min_samples=5, threshold=0.25, probe_fraction=0.0
        )
        for _ in range(5):
            breaker.record("class-a", success=False)
        self.assertTrue(breaker.should_block("class-a")[0])
        # window=5, so 5 more successes fully evict the failing history.
        for _ in range(5):
            breaker.record("class-a", success=True)
        self.assertEqual(breaker.failure_rate("class-a"), 0.0)
        blocked, note = breaker.should_block("class-a")
        self.assertFalse(blocked)
        self.assertEqual(note, "reliability-ok")

    def test_classes_are_independent(self):
        # This test verifies class isolation, not the randomized recovery
        # stream. Disable probes so an open circuit cannot legitimately let
        # the one asserted call through and make the test flaky.
        breaker = ReliabilityCircuitBreaker(
            min_samples=5, threshold=0.25, probe_fraction=0.0
        )
        for _ in range(5):
            breaker.record("failing-class", success=False)
            breaker.record("healthy-class", success=True)
        self.assertTrue(breaker.should_block("failing-class")[0])
        self.assertFalse(breaker.should_block("healthy-class")[0])

    def test_record_ignores_none_class_key(self):
        breaker = ReliabilityCircuitBreaker()
        breaker.record(None, success=False)  # must not raise
        self.assertIsNone(breaker.failure_rate(None))
        self.assertFalse(breaker.should_block(None)[0])


if __name__ == "__main__":
    unittest.main()
