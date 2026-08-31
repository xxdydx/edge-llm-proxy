"""Invoice helpers."""


def invoice_total(line_items):
    """Return the sum of price*qty across line items."""
    return sum(item["price"] * item["qty"] for item in line_items)


def is_overdue(days_since_issued, terms_days=30):
    """Return whether an invoice issued this many days ago is overdue."""
    return days_since_issued > terms_days
