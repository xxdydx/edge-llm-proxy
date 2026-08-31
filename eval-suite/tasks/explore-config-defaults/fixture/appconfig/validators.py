"""Validation helpers with several distinct failure modes."""

from .settings import load_settings


def validate_timeout(raw):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"timeout must be an integer, got {raw!r}")
    if value <= 0:
        raise ValueError("timeout must be positive")
    return value


def validate_retries(raw):
    # note: unlike validate_timeout, this does not reject a missing KeyError
    # up front - callers are expected to pass a plain value, not a mapping.
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"retries must be an integer, got {raw!r}")
    return value


def safe_load(overrides):
    """Load settings, translating any failure into a single result dict."""
    try:
        return {"ok": True, "settings": load_settings(overrides)}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except KeyError as exc:
        return {"ok": False, "error": f"missing key: {exc}"}
    except TypeError as exc:
        return {"ok": False, "error": f"type error: {exc}"}
