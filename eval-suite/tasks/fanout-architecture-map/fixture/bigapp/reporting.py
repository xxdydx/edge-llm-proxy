"""Summary statistics over a set of issues."""

from collections import Counter


def status_breakdown(issues):
    """Return a dict of status -> count."""
    return dict(Counter(i.status for i in issues))


def assignee_load(issues):
    """Return a dict of assignee_id -> number of open/in_progress issues assigned."""
    counts = Counter()
    for issue in issues:
        if issue.status != "closed" and issue.assignee_id is not None:
            counts[issue.assignee_id] += 1
    return dict(counts)
