Your working directory contains `bigapp/`, a small issue-tracking service
spread across several modules: `models.py`, `storage.py`, `validation.py`,
`workflow.py`, `permissions.py`, `search.py`, `notifications.py`,
`reporting.py`, and `api.py` (the top-level public interface). Every existing
test file currently passes. `test_priority_feature.py` also exists but
currently fails, because the priority feature it exercises is not
implemented yet.

Implement issue priority levels end-to-end:

1. `Issue` (in `models.py`) gets a new `priority` field. Valid values are
   `"low"`, `"medium"`, `"high"`, `"critical"`; if not given, it defaults to
   `"medium"`.
2. Add `validate_priority(priority)` to `validation.py`, following the same
   pattern as the existing `validate_status`/`validate_role`: return the
   value if valid, raise `ValueError` for anything else.
3. `api.create_issue` gets a new optional `priority` keyword argument
   (default `"medium"`), validated through `validate_priority`.
4. Workflow rule: closing a `"critical"`-priority issue requires a
   non-empty resolution comment. Extend `workflow.apply_transition` with an
   optional `resolution_comment=None` parameter; raise `ValueError` if the
   issue being closed is `"critical"` priority and no resolution comment is
   given. Issues of any other priority close exactly as before - do not
   change that existing behavior.
5. Add `filter_by_priority(issues, priority)` to `search.py`, matching the
   style of the existing `filter_by_status`/`filter_by_assignee`.
6. Add `priority_breakdown(issues)` to `reporting.py`, matching the style
   and return shape of the existing `status_breakdown`.
7. `api.change_status` needs to accept and pass through an optional
   `resolution_comment=None` argument to `workflow.apply_transition`.

Make `test_priority_feature.py` pass without modifying it, and without
breaking any of the other existing tests. Do not modify any `test_*.py`
file. Do not run `pip install` or access the network. Run the full test
suite yourself (`python3 -m pytest -q`) before you finish to confirm
everything passes.
