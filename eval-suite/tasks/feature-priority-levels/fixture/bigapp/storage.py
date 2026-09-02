"""In-memory repositories for users, issues, and projects."""


class IssueStore:
    def __init__(self):
        self._issues = {}
        self._next_id = 1

    def add(self, issue):
        issue.id = self._next_id
        self._issues[issue.id] = issue
        self._next_id += 1
        return issue

    def get(self, issue_id):
        return self._issues.get(issue_id)

    def all(self):
        return list(self._issues.values())

    def delete(self, issue_id):
        self._issues.pop(issue_id, None)


class UserStore:
    def __init__(self):
        self._users = {}
        self._next_id = 1

    def add(self, user):
        user.id = self._next_id
        self._users[user.id] = user
        self._next_id += 1
        return user

    def get(self, user_id):
        return self._users.get(user_id)

    def all(self):
        return list(self._users.values())


class ProjectStore:
    def __init__(self):
        self._projects = {}
        self._next_id = 1

    def add(self, project):
        project.id = self._next_id
        self._projects[project.id] = project
        self._next_id += 1
        return project

    def get(self, project_id):
        return self._projects.get(project_id)
