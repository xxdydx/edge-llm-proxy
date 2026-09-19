import unittest
from dataclasses import replace

from edgeproxy.completion import (
    CompletionEstimator,
    predict_local_completion,
    predict_local_ttft_ms,
)
from edgeproxy.router import extract_features
from edgeproxy.telemetry import LocalBackendState


class CompletionEstimatorTests(unittest.TestCase):
    def test_frozen_ttft_formula_known_pairs(self):
        # Hand-evaluated with Decimal from the four frozen coefficients:
        # 19.9430278667 + .00394159703426*1024 + .130965859853*1024
        # + .0000020506*(1024**2) = 160.23847366485424.
        self.assertAlmostEqual(
            predict_local_ttft_ms(1024, 0), 160.23847366485424, places=10
        )
        self.assertAlmostEqual(
            predict_local_ttft_ms(2048, 1024), 168.57508891913648, places=10
        )
        self.assertAlmostEqual(
            predict_local_ttft_ms(24576, 24560), 120.51930284392176, places=10
        )

    def test_ewma_is_empty_then_updates_real_outcomes(self):
        estimator = CompletionEstimator(alpha=0.25)
        self.assertIsNone(estimator.history("tools-a", concurrency=1))

        estimator.record(
            "tools-a", output_tokens=100, tpot_ms=30.0, concurrency=1
        )
        estimator.record(
            "tools-a", output_tokens=200, tpot_ms=50.0, concurrency=1
        )
        history = estimator.history("tools-a", concurrency=1)

        self.assertIsNotNone(history)
        assert history is not None
        self.assertEqual(history.ewma_output_tokens, 125.0)
        self.assertEqual(history.ewma_tpot_ms, 35.0)
        self.assertEqual(history.output_samples, 2)
        self.assertEqual(history.tpot_samples, 2)
        self.assertEqual(history.requested_tpot_bucket, "1")
        self.assertEqual(history.selected_tpot_bucket, "1")
        self.assertIsNone(estimator.history("tools-b", concurrency=1))

    def test_prediction_none_until_both_class_measurements_exist(self):
        estimator = CompletionEstimator()
        features = replace(
            extract_features(
                {
                    "model": "model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "work"}],
                }
            ),
            local_prompt_tokens=1024,
            estimated_local_cached_tokens=0,
        )
        state = LocalBackendState(concurrency_limit=2)

        self.assertIsNone(
            predict_local_completion(features, "tools-a", state, estimator)
        )
        estimator.record(
            "tools-a", output_tokens=100, tpot_ms=None, concurrency=1
        )
        self.assertIsNone(
            predict_local_completion(features, "tools-a", state, estimator)
        )

        estimator.record(
            "tools-a", output_tokens=None, tpot_ms=10.0, concurrency=1
        )
        prediction = predict_local_completion(
            features, "tools-a", state, estimator
        )
        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertAlmostEqual(prediction.decode_ms, 1000.0)
        self.assertEqual(prediction.queue_wait_ms, 0.0)

    def test_queue_wait_charges_one_service_time_per_full_wave(self):
        estimator = CompletionEstimator()
        estimator.record(
            "tools-a", output_tokens=100, tpot_ms=10.0, concurrency=1
        )
        features = replace(
            extract_features(
                {
                    "model": "model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "work"}],
                }
            ),
            local_prompt_tokens=1024,
            estimated_local_cached_tokens=0,
        )
        state = LocalBackendState(concurrency_limit=2)
        leases = [state.begin_request() for _ in range(2)]
        try:
            prediction = predict_local_completion(
                features, "tools-a", state, estimator
            )
        finally:
            for lease in leases:
                lease.release()

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.queue_waves_ahead, 1)
        self.assertAlmostEqual(
            prediction.queue_wait_ms, prediction.ttft_ms + prediction.decode_ms
        )
        self.assertAlmostEqual(
            prediction.total_ms,
            2 * (prediction.ttft_ms + prediction.decode_ms),
        )

    def test_prediction_uses_higher_tpot_at_higher_execution_concurrency(self):
        estimator = CompletionEstimator(alpha=1.0)
        estimator.record(
            "tools-a", output_tokens=100, tpot_ms=30.0, concurrency=1
        )
        estimator.record(
            "tools-a", output_tokens=None, tpot_ms=90.0, concurrency=4
        )
        features = replace(
            extract_features(
                {
                    "model": "model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "work"}],
                }
            ),
            local_prompt_tokens=1024,
            estimated_local_cached_tokens=0,
        )
        low_state = LocalBackendState(concurrency_limit=8)
        high_state = LocalBackendState(concurrency_limit=8)
        leases = [high_state.begin_request() for _ in range(3)]
        try:
            low = predict_local_completion(
                features, "tools-a", low_state, estimator
            )
            high = predict_local_completion(
                features, "tools-a", high_state, estimator
            )
        finally:
            for lease in leases:
                lease.release()

        assert low is not None and high is not None
        self.assertEqual(low.execution_concurrency, 1)
        self.assertEqual(high.execution_concurrency, 4)
        self.assertEqual(low.selected_tpot_bucket, "1")
        self.assertEqual(high.selected_tpot_bucket, "4+")
        self.assertGreater(high.total_ms, low.total_ms)

    def test_missing_bucket_falls_back_to_nearest_lower_observed_bucket(self):
        estimator = CompletionEstimator(alpha=1.0)
        estimator.record(
            "tools-a", output_tokens=100, tpot_ms=30.0, concurrency=1
        )
        estimator.record(
            "tools-a", output_tokens=None, tpot_ms=50.0, concurrency=2
        )

        features = replace(
            extract_features(
                {
                    "model": "model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "work"}],
                }
            ),
            local_prompt_tokens=1024,
            estimated_local_cached_tokens=0,
        )
        state = LocalBackendState(concurrency_limit=8)
        leases = [state.begin_request() for _ in range(4)]
        try:
            prediction = predict_local_completion(
                features, "tools-a", state, estimator
            )
        finally:
            for lease in leases:
                lease.release()

        assert prediction is not None
        self.assertEqual(prediction.expected_tpot_ms, 50.0)
        self.assertEqual(prediction.requested_tpot_bucket, "4+")
        self.assertEqual(prediction.selected_tpot_bucket, "2-3")
        self.assertTrue(prediction.tpot_bucket_fallback)
        self.assertEqual(prediction.tpot_samples, 1)


if __name__ == "__main__":
    unittest.main()
