"""Filtering and searching over issues."""


def filter_by_status(issues, status):
    return [i for i in issues if i.status == status]


def filter_by_assignee(issues, assignee_id):
    return [i for i in issues if i.assignee_id == assignee_id]


def search_by_title(issues, query):
    query = query.lower()
    return [i for i in issues if query in i.title.lower()]
