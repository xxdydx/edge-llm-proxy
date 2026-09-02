import pytest
from models import User
import api


@pytest.fixture(autouse=True)
def reset_store():
    api.issue_store = api.issue_store.__class__()


def test_create_and_find_issue():
    api.create_issue("Login broken")
    found = api.find_issues("login")
    assert len(found) == 1


def test_change_status_requires_permission():
    issue = api.create_issue("Bug", assignee_id=1)
    owner = User(id=1, name="Owner", role="member")
    stranger = User(id=2, name="Stranger", role="member")
    api.change_status(issue.id, "in_progress", owner)
    assert issue.status == "in_progress"
    with pytest.raises(PermissionError):
        api.change_status(issue.id, "closed", stranger)


def test_summary_report():
    api.create_issue("a")
    api.create_issue("b")
    report = api.summary_report()
    assert report == {"open": 2}
