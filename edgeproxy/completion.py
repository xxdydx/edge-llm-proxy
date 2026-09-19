"""Observe-only local completion-time prediction for Rung 4.

The TTFT coefficients below are frozen from the 420-call fit documented in
``claude-memory/wiki/findings/Prefix cache TTFT curve over 420 calls.md``.
That study was validated only on Qwen2.5-7B-Instruct-AWQ on an RTX 5080.
We use it as the initial estimator for both current deployments, but its
accuracy for Qwen3.8-27B-NVFP4 is unverified and must not be claimed to
transfer until it is measured there.

Queueing is intentionally a crude fixed-capacity wave approximation, not a
queueing-theory model.  TPOT is conditioned on observed concurrency using
three trace-grounded buckets: 1, 2-3, and 4+ requests.  The retrospective
corpus had 340 / 138 / 31 samples in those buckets (the last combines only 17
at concurrency 4 and 14 at 5), so exact high-concurrency buckets would be too
sparse for separate online EWMAs.  This module is observational only and never
changes a placement decision.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

from .router import CallFeatures
from .telemetry import LocalBackendState

DEFAULT_EWMA_ALPHA = 0.2

CONCURRENCY_BUCKETS = ("1", "2-3", "4+")

TTFT_INTERCEPT_MS = 19.9430278667
TTFT_TOTAL_TOKENS_COEFFICIENT = 0.00394159703426
TTFT_UNCACHED_TOKENS_COEFFICIENT = 0.130965859853
TTFT_TOKEN_SQUARES_COEFFICIENT = 0.0000020506


@dataclass(frozen=True)
class CompletionHistory:
    ewma_output_tokens: float | None
    ewma_tpot_ms: float | None
    output_samples: int
    tpot_samples: int
    requested_tpot_bucket: str
    selected_tpot_bucket: str | None


@dataclass
class _MutableCompletionHistory:
    ewma_output_tokens: float | None = None
    output_samples: int = 0

    tpot_by_bucket: dict[str, _MutableTPOTHistory] = field(default_factory=dict)


@dataclass
class _MutableTPOTHistory:
    ewma_tpot_ms: float | None = None
    tpot_samples: int = 0


@dataclass(frozen=True)
class LocalCompletionPrediction:
    total_ms: float
    ttft_ms: float
    queue_wait_ms: float
    decode_ms: float
    expected_output_tokens: float
    expected_tpot_ms: float
    requests_in_flight: int
    execution_concurrency: int
    concurrency_limit: int
    queue_waves_ahead: int
    output_samples: int
    tpot_samples: int
    requested_tpot_bucket: str
    selected_tpot_bucket: str
    tpot_bucket_fallback: bool

    def as_trace(self) -> dict[str, float | int | str | bool]:
        return {
            "total_ms": round(self.total_ms, 3),
            "ttft_ms": round(self.ttft_ms, 3),
            "queue_wait_ms": round(self.queue_wait_ms, 3),
            "decode_ms": round(self.decode_ms, 3),
            "expected_output_tokens": round(self.expected_output_tokens, 3),
            "expected_tpot_ms": round(self.expected_tpot_ms, 3),
            "requests_in_flight": self.requests_in_flight,
            "execution_concurrency": self.execution_concurrency,
            "concurrency_limit": self.concurrency_limit,
            "queue_waves_ahead": self.queue_waves_ahead,
            "output_samples": self.output_samples,
            "tpot_samples": self.tpot_samples,
            "requested_tpot_bucket": self.requested_tpot_bucket,
            "selected_tpot_bucket": self.selected_tpot_bucket,
            "tpot_bucket_fallback": self.tpot_bucket_fallback,
        }


class CompletionEstimator:
    """Thread-safe class output EWMA and concurrency-conditioned TPOT EWMAs."""

    def __init__(self, *, alpha: float = DEFAULT_EWMA_ALPHA) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._lock = threading.Lock()
        self._classes: dict[str, _MutableCompletionHistory] = {}

    def record(
        self,
        class_key: str | None,
        *,
        output_tokens: int | None,
        tpot_ms: float | None,
        concurrency: int | None,
    ) -> None:
        """Record the real measurements available from one completed local call.

        Non-streaming calls currently supply output tokens but no measured TPOT,
        so the two EWMAs and their sample counts advance independently. A TPOT
        sample without a positive integer dispatch concurrency is ignored: it
        must not contaminate an invented global bucket. Invalid or absent values
        are ignored rather than converted into zeroes.
        """
        if class_key is None:
            return
        valid_output = (
            float(output_tokens)
            if isinstance(output_tokens, int) and output_tokens >= 0
            else None
        )
        valid_tpot = (
            float(tpot_ms)
            if isinstance(tpot_ms, (int, float))
            and math.isfinite(tpot_ms)
            and tpot_ms > 0
            else None
        )
        bucket = concurrency_bucket(concurrency)
        if bucket is None:
            valid_tpot = None
        if valid_output is None and valid_tpot is None:
            return
        with self._lock:
            history = self._classes.setdefault(class_key, _MutableCompletionHistory())
            if valid_output is not None:
                history.ewma_output_tokens = self._update(
                    history.ewma_output_tokens, valid_output
                )
                history.output_samples += 1
            if valid_tpot is not None:
                tpot_history = history.tpot_by_bucket.setdefault(
                    bucket, _MutableTPOTHistory()
                )
                tpot_history.ewma_tpot_ms = self._update(
                    tpot_history.ewma_tpot_ms, valid_tpot
                )
                tpot_history.tpot_samples += 1

    def history(
        self, class_key: str | None, *, concurrency: int
    ) -> CompletionHistory | None:
        """Return class history resolved for one execution concurrency.

        Exact-bucket data wins. If that bucket is empty, use the nearest lower
        bucket with data; if none exists, use the nearest higher bucket. This
        is an explicit reuse of observed data, never numeric interpolation.
        """
        requested_bucket = concurrency_bucket(concurrency)
        if class_key is None or requested_bucket is None:
            return None
        with self._lock:
            history = self._classes.get(class_key)
            if history is None:
                return None
            selected_bucket = self._select_tpot_bucket(history, requested_bucket)
            tpot_history = (
                history.tpot_by_bucket[selected_bucket]
                if selected_bucket is not None
                else None
            )
            return CompletionHistory(
                ewma_output_tokens=history.ewma_output_tokens,
                ewma_tpot_ms=(
                    tpot_history.ewma_tpot_ms if tpot_history is not None else None
                ),
                output_samples=history.output_samples,
                tpot_samples=(
                    tpot_history.tpot_samples if tpot_history is not None else 0
                ),
                requested_tpot_bucket=requested_bucket,
                selected_tpot_bucket=selected_bucket,
            )

    @staticmethod
    def _select_tpot_bucket(
        history: _MutableCompletionHistory, requested_bucket: str
    ) -> str | None:
        requested_index = CONCURRENCY_BUCKETS.index(requested_bucket)
        candidates = (
            [requested_bucket]
            + list(reversed(CONCURRENCY_BUCKETS[:requested_index]))
            + list(CONCURRENCY_BUCKETS[requested_index + 1 :])
        )
        for candidate in candidates:
            tpot_history = history.tpot_by_bucket.get(candidate)
            if tpot_history is not None and tpot_history.ewma_tpot_ms is not None:
                return candidate
        return None

    def _update(self, previous: float | None, value: float) -> float:
        if previous is None:
            return value
        return self._alpha * value + (1.0 - self._alpha) * previous


def concurrency_bucket(concurrency: int | None) -> str | None:
    """Map positive execution concurrency to the trace-grounded TPOT buckets."""
    if not isinstance(concurrency, int) or isinstance(concurrency, bool):
        return None
    if concurrency == 1:
        return "1"
    if 2 <= concurrency <= 3:
        return "2-3"
    if concurrency >= 4:
        return "4+"
    return None


def predict_local_ttft_ms(total_prompt_tokens: int, resident_tokens: int) -> float:
    """Frozen no-queue TTFT(N, R) fit from the 7B/RTX-5080 study."""
    n = float(total_prompt_tokens)
    r = float(resident_tokens)
    return (
        TTFT_INTERCEPT_MS
        + TTFT_TOTAL_TOKENS_COEFFICIENT * n
        + TTFT_UNCACHED_TOKENS_COEFFICIENT * (n - r)
        + TTFT_TOKEN_SQUARES_COEFFICIENT * (n**2 - r**2)
    )


def predict_local_completion(
    features: CallFeatures,
    tool_suite_hash: str | None,
    local_backend_state: LocalBackendState,
    completion_estimator: CompletionEstimator,
) -> LocalCompletionPrediction | None:
    """Predict local completion, or None until both class EWMAs exist.

    Queue wait is ``floor(in_flight / concurrency_limit) * service_ms``.
    Each full fixed-capacity wave ahead is conservatively charged one copy of
    this call's own TTFT-plus-decode service estimate.  This deliberately uses
    only the exact proxy counter and configured limit; it is a simple starting
    approximation, not a calibrated queue model.
    """
    if (
        features.local_prompt_tokens is None
        or features.estimated_local_cached_tokens is None
        or local_backend_state.concurrency_limit <= 0
    ):
        return None
    in_flight = local_backend_state.requests_in_flight
    execution_concurrency = in_flight + 1
    history = completion_estimator.history(
        tool_suite_hash, concurrency=execution_concurrency
    )
    if (
        history is None
        or history.ewma_output_tokens is None
        or history.ewma_tpot_ms is None
        or history.selected_tpot_bucket is None
    ):
        return None

    ttft_ms = predict_local_ttft_ms(
        features.local_prompt_tokens, features.estimated_local_cached_tokens
    )
    decode_ms = history.ewma_output_tokens * history.ewma_tpot_ms
    service_ms = ttft_ms + decode_ms
    waves_ahead = in_flight // local_backend_state.concurrency_limit
    queue_wait_ms = waves_ahead * service_ms
    return LocalCompletionPrediction(
        total_ms=service_ms + queue_wait_ms,
        ttft_ms=ttft_ms,
        queue_wait_ms=queue_wait_ms,
        decode_ms=decode_ms,
        expected_output_tokens=history.ewma_output_tokens,
        expected_tpot_ms=history.ewma_tpot_ms,
        requests_in_flight=in_flight,
        execution_concurrency=execution_concurrency,
        concurrency_limit=local_backend_state.concurrency_limit,
        queue_waves_ahead=waves_ahead,
        output_samples=history.output_samples,
        tpot_samples=history.tpot_samples,
        requested_tpot_bucket=history.requested_tpot_bucket,
        selected_tpot_bucket=history.selected_tpot_bucket,
        tpot_bucket_fallback=(
            history.requested_tpot_bucket != history.selected_tpot_bucket
        ),
    )
