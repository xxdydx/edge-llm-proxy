"""Shopping cart helpers."""


class Cart:
    def __init__(self):
        self.items = []

    def add_item(self, name, price, quantity=1):
        """Add one line item to the cart and return the updated item list."""
        self.items.append({"name": name, "price": price, "quantity": quantity})
        return list(self.items)

    def total(self):
        """Return the cart subtotal."""
        return sum(item["price"] * item["quantity"] for item in self.items)


def merge_carts(primary, other):
    """Combine two carts' line items so `primary` holds everything from both."""
    primary.items.extend(other.items)
