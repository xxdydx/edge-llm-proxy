"""EU shipping estimates."""


def estimate_shipping_days(distance_km, express=False):
    """Estimate delivery days from distance, with an express multiplier."""
    base_days = 1 + distance_km / 800.0
    if express:
        base_days *= 0.5
    return max(1, round(base_days))
