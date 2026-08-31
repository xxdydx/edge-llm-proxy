"""Application settings with layered defaults."""

DEFAULT_TIMEOUT_S = 30
DEFAULT_RETRIES = 3
DEFAULT_REGION = "us-east"

_ALLOWED_KEYS = {"timeout_s", "retries", "region"}


def load_settings(overrides=None):
    """Merge `overrides` over the defaults, rejecting unknown keys."""
    settings = {
        "timeout_s": DEFAULT_TIMEOUT_S,
        "retries": DEFAULT_RETRIES,
        "region": DEFAULT_REGION,
    }
    if not overrides:
        return settings
    for key in overrides:
        if key not in _ALLOWED_KEYS:
            raise ValueError(f"unknown setting: {key}")
    settings.update(overrides)
    return settings
