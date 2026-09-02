"""Notification message formatting."""


def format_status_change(issue, old_status):
    return f"Issue #{issue.id} moved from {old_status} to {issue.status}"


def format_new_comment(issue, comment):
    return f"New comment on #{issue.id} by user {comment.author_id}: {comment.text}"


def format_assignment(issue, assignee_id):
    return f"Issue #{issue.id} assigned to user {assignee_id}"
