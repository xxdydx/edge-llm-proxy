"""Bulk discount pricing."""


def bulk_discount_rate(quantity):
    """Return the discount rate earned for buying `quantity` units at once."""
    if quantity >= 100:
        return 0.20
    if quantity > 50:
        return 0.10
    if quantity >= 10:
        return 0.05
    return 0.0
