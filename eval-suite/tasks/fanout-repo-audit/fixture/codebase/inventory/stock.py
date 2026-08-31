"""Stock-level helpers."""


def reorder_needed(on_hand, reorder_point):
    """Return whether stock should be reordered."""
    return on_hand <= reorder_point


def apply_restock(on_hand, incoming):
    """Return the new on-hand quantity after restocking."""
    return on_hand + incoming


def days_of_supply(on_hand, daily_usage):
    """Return how many days the current stock will last, or None if usage is zero."""
    if daily_usage == 0:
        return None
    return on_hand / daily_usage
