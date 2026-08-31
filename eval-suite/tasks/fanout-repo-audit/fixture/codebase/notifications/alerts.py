"""Alerting helpers - no test coverage yet."""


def format_low_stock_alert(item_name, on_hand):
    return f"LOW STOCK: {item_name} has only {on_hand} left"


def format_overdue_alert(invoice_id, days_late):
    return f"OVERDUE: invoice {invoice_id} is {days_late} day(s) late"


def severity_for_days_late(days_late):
    if days_late > 30:
        return "critical"
    if days_late > 7:
        return "warning"
    return "info"


def dedupe_alerts(alerts):
    """Remove exact-duplicate alert strings while preserving order."""
    seen = set()
    result = []
    for alert in alerts:
        if alert not in seen:
            seen.add(alert)
            result.append(alert)
    return result
