"""Order processing."""


def process_order(items, discount_pct=0.0):
    """Validate items, compute the discounted total, and format it as a price string."""
    if not items:
        raise ValueError("items must not be empty")
    for item in items:
        if item["price"] < 0 or item["qty"] < 0:
            raise ValueError("negative price or quantity")
    subtotal = sum(item["price"] * item["qty"] for item in items)
    total = subtotal * (1 - discount_pct)
    return f"${total:.2f}"
