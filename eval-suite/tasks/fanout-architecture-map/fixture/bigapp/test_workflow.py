import pytest
from models import Issue
from workflow import can_transition, apply_transition


def test_open_to_in_progress_allowed():
    assert can_transition("open", "in_progress") is True


def test_open_to_closed_not_allowed_directly():
    assert can_transition("open", "closed") is False


def test_apply_transition_updates_status():
    issue = Issue(id=1, title="Bug", status="open")
    apply_transition(issue, "in_progress")
    assert issue.status == "in_progress"


def test_apply_transition_rejects_invalid():
    issue = Issue(id=1, title="Bug", status="closed")
    with pytest.raises(ValueError):
        apply_transition(issue, "open")
