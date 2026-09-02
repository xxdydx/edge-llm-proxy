"""Input validation helpers."""

from models import VALID_STATUSES


def validate_title(title):
    if not title or not title.strip():
        raise ValueError("title must not be empty")
    if len(title) > 200:
        raise ValueError("title too long")
    return title.strip()


def validate_status(status):
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    return status


def validate_role(role):
    if role not in ("member", "admin"):
        raise ValueError(f"invalid role: {role}")
    return role
