"""Turn per-seed pass/fail counts into a rate with a confidence interval.

This is the piece that converts the campaign's raw per-seed verdicts into the
"rates with intervals" the paired campaign exists to produce (see summary.md's
routing-inputs discussion and the project brief this suite was built from).
With only 3-5 seeds per (task, condition) cell, the interval method matters:
a naive normal approximation is known to misbehave at small n and can even
produce bounds outside [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist


@dataclass
class RateEstimate:
    passes: int
    n: int
    rate: float
    ci_low: float
    ci_high: float


def binomial_rate_ci(passes: int, n: int, confidence: float = 0.95) -> RateEstimate:
    """Estimate a pass rate and its confidence interval from `passes` out of `n` seeds.

    Uses the Wilson score interval rather than a normal (Wald) approximation:
    Wald collapses to zero width at passes in {0, n} and can leave [0, 1]
    entirely, both of which are live risks at the n=3-5 seed counts one
    campaign cell actually has. Wilson stays inside [0, 1] by construction and
    degrades gracefully at small n and extreme proportions.
    """
    if n == 0:
        return RateEstimate(passes=0, n=0, rate=0.0, ci_low=0.0, ci_high=1.0)

    rate = passes / n
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    z2 = z * z

    denominator = 1 + z2 / n
    center = rate + z2 / (2 * n)
    adjustment = z * ((rate * (1 - rate) / n) + (z2 / (4 * n * n))) ** 0.5

    ci_low = max(0.0, (center - adjustment) / denominator)
    ci_high = min(1.0, (center + adjustment) / denominator)

    return RateEstimate(passes=passes, n=n, rate=rate, ci_low=ci_low, ci_high=ci_high)
