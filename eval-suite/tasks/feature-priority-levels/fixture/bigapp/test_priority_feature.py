import pytest
from models import Issue, User
from validation import validate_priority
from workflow import apply_transition
from search import filter_by_priority
from reporting import priority_breakdown
import api


@pytest.fixture(autouse=True)
def reset_store():
    api.issue_store = api.issue_store.__class__()


def test_issue_defaults_to_medium_priority():
    issue = Issue(id=1, title="Bug")
    assert issue.priority == "medium"


def test_issue_accepts_explicit_priority():
    issue = Issue(id=1, title="Bug", priority="high")
    assert issue.priority == "high"


def test_validate_priority_accepts_known_values():
    for p in ("low", "medium", "high", "critical"):
        assert validate_priority(p) == p


def test_validate_priority_rejects_unknown():
    with pytest.raises(ValueError):
        validate_priority("urgent")


def test_create_issue_accepts_priority_kwarg():
    issue = api.create_issue("Bug", priority="critical")
    assert issue.priority == "critical"


def test_create_issue_defaults_priority_to_medium():
    issue = api.create_issue("Bug")
    assert issue.priority == "medium"


def test_critical_issue_requires_resolution_comment_to_close():
    issue = Issue(id=1, title="Bug", status="in_progress", priority="critical")
    with pytest.raises(ValueError):
        apply_transition(issue, "closed")
    apply_transition(issue, "closed", resolution_comment="fixed the root cause")
    assert issue.status == "closed"


def test_non_critical_issue_closes_without_comment():
    issue = Issue(id=1, title="Bug", status="in_progress", priority="medium")
    apply_transition(issue, "closed")
    assert issue.status == "closed"


def test_filter_by_priority():
    issues = [
        Issue(id=1, title="a", priority="high"),
        Issue(id=2, title="b", priority="low"),
        Issue(id=3, title="c", priority="high"),
    ]
    result = filter_by_priority(issues, "high")
    assert [i.id for i in result] == [1, 3]


def test_priority_breakdown():
    issues = [
        Issue(id=1, title="a", priority="high"),
        Issue(id=2, title="b", priority="high"),
        Issue(id=3, title="c", priority="low"),
    ]
    result = priority_breakdown(issues)
    assert result == {"high": 2, "low": 1}


def test_change_status_via_api_enforces_critical_resolution_comment():
    owner = User(id=1, name="Owner", role="admin")
    issue = api.create_issue("Critical bug", priority="critical")
    api.change_status(issue.id, "in_progress", owner)
    with pytest.raises(ValueError):
        api.change_status(issue.id, "closed", owner)
    api.change_status(issue.id, "closed", owner, resolution_comment="patched")
    assert issue.status == "closed"
