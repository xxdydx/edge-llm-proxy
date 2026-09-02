from models import Issue
from reporting import status_breakdown, assignee_load


def _issues():
    return [
        Issue(id=1, title="a", status="open", assignee_id=1),
        Issue(id=2, title="b", status="open", assignee_id=1),
        Issue(id=3, title="c", status="closed", assignee_id=1),
        Issue(id=4, title="d", status="in_progress", assignee_id=2),
    ]


def test_status_breakdown():
    result = status_breakdown(_issues())
    assert result == {"open": 2, "closed": 1, "in_progress": 1}


def test_assignee_load_excludes_closed():
    result = assignee_load(_issues())
    assert result == {1: 2, 2: 1}
