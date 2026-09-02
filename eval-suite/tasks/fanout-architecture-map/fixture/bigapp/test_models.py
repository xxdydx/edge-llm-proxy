from models import Issue, User


def test_issue_defaults_to_open():
    issue = Issue(id=1, title="Bug")
    assert issue.status == "open"
    assert issue.comments == []


def test_add_comment():
    issue = Issue(id=1, title="Bug")
    issue.add_comment(author_id=5, text="looking into it")
    assert len(issue.comments) == 1
    assert issue.comments[0].text == "looking into it"


def test_user_fields():
    user = User(id=1, name="Alice", role="admin")
    assert user.role == "admin"
