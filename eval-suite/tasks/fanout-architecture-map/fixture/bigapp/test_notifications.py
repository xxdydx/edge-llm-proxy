from models import Issue, Comment
from notifications import format_status_change, format_new_comment, format_assignment


def test_format_status_change():
    issue = Issue(id=5, title="Bug", status="in_progress")
    msg = format_status_change(issue, "open")
    assert msg == "Issue #5 moved from open to in_progress"


def test_format_new_comment():
    issue = Issue(id=5, title="Bug")
    comment = Comment(author_id=2, text="fixed")
    msg = format_new_comment(issue, comment)
    assert "fixed" in msg and "#5" in msg


def test_format_assignment():
    issue = Issue(id=5, title="Bug")
    msg = format_assignment(issue, 3)
    assert msg == "Issue #5 assigned to user 3"
