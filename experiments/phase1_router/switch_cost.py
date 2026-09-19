"""Cache-aware switching cost.

When two consecutive requests are routed to different backends, the second
one cannot reuse the first's warm prefix / KV state: it pays a cold prefill.
We charge that as ``multiplier * prompt_est_tokens`` extra "cold token" units
on every backend flip, and report the frontier at multiplier 0x (no penalty)
and 10x so the cache term's effect on the routing decision is visible.

The base efficiency unit is prompt-normalised: 1.0 per served request plus
the switch penalty. This is a declared proxy, not provider billing.
"""

from __future__ import annotations

from .schema import PairedExample


def sequence_cost(
    examples: list[PairedExample],
    route_local: list[bool],
    multiplier: float,
) -> dict[str, float]:
    """Cost of serving ``examples`` in order under ``route_local``.

    Returns total cost, per-request mean, the raw switch count, and the
    cold-token penalty total. ``examples`` must already be in the intended
    serving order (selection / arrival order)."""
    assert len(examples) == len(route_local)
    switches = 0
    penalty = 0.0
    base = float(len(examples))  # 1 unit per served request
    prev: bool | None = None
    for ex, local in zip(examples, route_local):
        if prev is not None and local != prev:
            switches += 1
            penalty += multiplier * float(ex.features.get("prompt_est_tokens", 0.0)) / 1000.0
        prev = local
    total = base + penalty
    return {
        "multiplier": multiplier,
        "total_cost": total,
        "mean_cost_per_request": total / base if base else 0.0,
        "switch_count": switches,
        "switch_rate": switches / (base - 1) if base > 1 else 0.0,
        "cold_token_penalty_kunits": penalty,
    }


def cost_curve_over_local_share(
    examples: list[PairedExample],
    order_by_local_pref_desc: list[int],
    multiplier: float,
    steps: int = 21,
) -> list[dict[str, float]]:
    """Sweep local share from 0% to 100% by admitting problems in the given
    priority order, and record (local_share, quality, cost) at each step.
    Quality is plus_pass accuracy of whichever arm served each request."""
    n = len(examples)
    curve = []
    for s in range(steps):
        k = round(n * s / (steps - 1))
        local_idx = set(order_by_local_pref_desc[:k])
        route = [i in local_idx for i in range(n)]
        correct = sum(
            (ex.local_ok if local else ex.cloud_ok)
            for i, (ex, local) in enumerate(zip(examples, route))
        )
        c = sequence_cost(examples, route, multiplier)
        curve.append({
            "local_share": k / n if n else 0.0,
            "quality": correct / n if n else 0.0,
            "mean_cost_per_request": c["mean_cost_per_request"],
            "switch_rate": c["switch_rate"],
        })
    return curve
