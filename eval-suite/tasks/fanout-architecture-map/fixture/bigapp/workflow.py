"""Status transition rules for issues."""

ALLOWED_TRANSITIONS = {
    "open": {"in_progress"},
    "in_progress": {"open", "closed"},
    "closed": set(),
}


def can_transition(current_status, new_status):
    return new_status in ALLOWED_TRANSITIONS.get(current_status, set())


def apply_transition(issue, new_status):
    """Move `issue` to `new_status`, raising ValueError if the transition isn't allowed."""
    if not can_transition(issue.status, new_status):
        raise ValueError(f"cannot transition from {issue.status} to {new_status}")
    issue.status = new_status
    return issue
