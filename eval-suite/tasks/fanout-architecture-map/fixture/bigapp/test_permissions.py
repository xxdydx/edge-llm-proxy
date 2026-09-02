from models import Issue, User
from permissions import can_edit_issue, can_delete_issue


def test_admin_can_edit_any_issue():
    admin = User(id=1, name="Admin", role="admin")
    issue = Issue(id=1, title="Bug", assignee_id=99)
    assert can_edit_issue(admin, issue) is True


def test_member_can_only_edit_own_issue():
    member = User(id=2, name="Bob", role="member")
    own_issue = Issue(id=1, title="Bug", assignee_id=2)
    other_issue = Issue(id=2, title="Bug 2", assignee_id=3)
    assert can_edit_issue(member, own_issue) is True
    assert can_edit_issue(member, other_issue) is False


def test_only_admin_can_delete():
    admin = User(id=1, name="Admin", role="admin")
    member = User(id=2, name="Bob", role="member")
    issue = Issue(id=1, title="Bug")
    assert can_delete_issue(admin, issue) is True
    assert can_delete_issue(member, issue) is False
