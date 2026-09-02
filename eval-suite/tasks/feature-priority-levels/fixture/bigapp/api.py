"""Top-level orchestration layer - the service's public interface."""

from models import Issue
from storage import IssueStore, UserStore
from validation import validate_title
from workflow import apply_transition
from permissions import can_edit_issue
from notifications import format_status_change, format_assignment
from search import filter_by_status, search_by_title
from reporting import status_breakdown

issue_store = IssueStore()
user_store = UserStore()


def create_issue(title, assignee_id=None):
    title = validate_title(title)
    issue = Issue(id=None, title=title, assignee_id=assignee_id)
    return issue_store.add(issue)


def change_status(issue_id, new_status, user):
    issue = issue_store.get(issue_id)
    if issue is None:
        raise ValueError(f"no such issue: {issue_id}")
    if not can_edit_issue(user, issue):
        raise PermissionError(f"user {user.id} cannot edit issue {issue_id}")
    old_status = issue.status
    apply_transition(issue, new_status)
    return format_status_change(issue, old_status)


def assign_issue(issue_id, assignee_id):
    issue = issue_store.get(issue_id)
    if issue is None:
        raise ValueError(f"no such issue: {issue_id}")
    issue.assignee_id = assignee_id
    return format_assignment(issue, assignee_id)


def list_open_issues():
    return filter_by_status(issue_store.all(), "open")


def find_issues(query):
    return search_by_title(issue_store.all(), query)


def summary_report():
    return status_breakdown(issue_store.all())
