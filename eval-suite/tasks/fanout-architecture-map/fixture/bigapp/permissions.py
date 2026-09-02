"""Role-based access checks."""


def can_edit_issue(user, issue):
    """Admins can edit any issue; members can only edit issues assigned to them."""
    if user.role == "admin":
        return True
    return issue.assignee_id == user.id


def can_delete_issue(user, issue):
    """Only admins can delete issues."""
    return user.role == "admin"
