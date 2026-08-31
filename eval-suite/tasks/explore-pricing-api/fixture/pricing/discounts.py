"""Discount calculations for the pricing package."""


def bulk_discount_rate(quantity):
    """Return the discount rate earned for buying `quantity` units at once."""
    if quantity >= 100:
        return 0.20
    if quantity >= 50:
        return 0.10
    if quantity >= 10:
        return 0.05
    return 0.0


def loyalty_discount_rate(years):
    """Return the discount rate earned for `years` of loyalty membership."""
    if years >= 5:
        return 0.15
    if years >= 1:
        return 0.05
    return 0.0


def _round_currency(amount):
    """Round to the nearest cent. Private helper, not part of the public API."""
    return round(amount, 2)


def apply_discounts(price, quantity, years):
    """Apply the larger of the bulk or loyalty discount and return the new price."""
    rate = max(bulk_discount_rate(quantity), loyalty_discount_rate(years))
    return _round_currency(price * (1 - rate))
