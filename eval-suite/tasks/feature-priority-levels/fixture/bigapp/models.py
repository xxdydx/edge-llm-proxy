"""Core data models."""

from dataclasses import dataclass, field

VALID_STATUSES = ("open", "in_progress", "closed")


@dataclass
class User:
    id: int
    name: str
    role: str  # "member" or "admin"


@dataclass
class Comment:
    author_id: int
    text: str


@dataclass
class Issue:
    id: int
    title: str
    status: str = "open"
    assignee_id: int | None = None
    comments: list = field(default_factory=list)

    def add_comment(self, author_id, text):
        self.comments.append(Comment(author_id=author_id, text=text))


@dataclass
class Project:
    id: int
    name: str
    issue_ids: list = field(default_factory=list)
