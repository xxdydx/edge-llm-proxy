"""Sales tax calculations."""

_RATES = {"CA": 0.0725, "NY": 0.04, "OR": 0.0}


def sales_tax_rate(region):
    """Return the sales tax rate for a two-letter region code, defaulting to 0."""
    return _RATES.get(region, 0.0)


def apply_tax(amount, region):
    """Return a new amount with sales tax added."""
    return round(amount * (1 + sales_tax_rate(region)), 2)
