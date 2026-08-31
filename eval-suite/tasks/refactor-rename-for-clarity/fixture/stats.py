"""Statistics helpers."""


def calc(values, weights):
    """Compute a weighted average of `values` using `weights`."""
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight
