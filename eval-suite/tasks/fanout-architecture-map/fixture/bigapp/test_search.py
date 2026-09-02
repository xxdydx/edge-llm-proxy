from models import Issue
from search import filter_by_status, filter_by_assignee, search_by_title


def _issues():
    return [
        Issue(id=1, title="Login bug", status="open", assignee_id=1),
        Issue(id=2, title="Signup bug", status="closed", assignee_id=2),
        Issue(id=3, title="Logout issue", status="open", assignee_id=1),
    ]


def test_filter_by_status():
    result = filter_by_status(_issues(), "open")
    assert [i.id for i in result] == [1, 3]


def test_filter_by_assignee():
    result = filter_by_assignee(_issues(), 1)
    assert [i.id for i in result] == [1, 3]


def test_search_by_title_case_insensitive():
    result = search_by_title(_issues(), "BUG")
    assert [i.id for i in result] == [1, 2]
