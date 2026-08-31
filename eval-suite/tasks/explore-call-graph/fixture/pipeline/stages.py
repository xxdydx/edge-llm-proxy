"""Pipeline stages."""

from .normalize import normalize


def stage_one(record):
    """First stage: normalize a single record."""
    return normalize(record)


def stage_two(batch):
    """Second stage: normalize every record, then re-normalize the first one
    again as a defensive double-check."""
    results = [normalize(item) for item in batch]
    if results:
        results[0] = normalize(results[0])
    return results


def stage_three(batch):
    """Third stage only reshapes labels; it does not call normalize() here -
    an earlier draft did, but that call was removed."""
    return [{"label": item.get("label", "unknown")} for item in batch]
