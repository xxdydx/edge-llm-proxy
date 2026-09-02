from models import Issue
from storage import IssueStore


def test_issue_store_add_assigns_id():
    store = IssueStore()
    issue = store.add(Issue(id=None, title="Bug"))
    assert issue.id == 1
    second = store.add(Issue(id=None, title="Bug 2"))
    assert second.id == 2


def test_issue_store_get_and_all():
    store = IssueStore()
    store.add(Issue(id=None, title="Bug"))
    assert len(store.all()) == 1
    assert store.get(1).title == "Bug"


def test_issue_store_delete():
    store = IssueStore()
    store.add(Issue(id=None, title="Bug"))
    store.delete(1)
    assert store.get(1) is None
