"""A single record-normalization helper used across pipeline stages."""


def normalize(record):
    """Return a new dict with a lowercase, stripped `label` field."""
    label = str(record.get("label", "")).strip().lower()
    return {**record, "label": label}
