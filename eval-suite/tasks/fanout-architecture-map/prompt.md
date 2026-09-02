Your working directory is `bigapp/`, a small multi-module issue-tracking
service. This is a read-only architecture audit: do not create, edit, or
delete any files, run `pip install`, or access the network.

Launch exactly three subagents concurrently, one per slice:
1. `models.py`, `storage.py`, `validation.py`
2. `workflow.py`, `permissions.py`, `search.py`
3. `notifications.py`, `reporting.py`, `api.py`

Use foreground Agent tool calls: emit the three independent Agent calls
together so they run in parallel, do not set `run_in_background`, and do not
return while any agent is pending.

Each subagent should read only its own three files (ignore every
`test_*.py` file) and report, for each of its files: the module-level
function names defined directly in it (count only free functions defined
with `def` at module level - do not count methods defined inside a class),
and whether that file's own code contains at least one `raise` statement
written directly in it (a function elsewhere raising when it is *called*
does not count).

After every subagent result has arrived, consolidate their findings and
return one Markdown report. Start the report with this exact line:
<!-- FANOUT_REPORT_START -->
Then use these exact level-two headings in this order: Executive Summary;
Findings from Each Agent; Architecture Notes; Open Questions.

At the very end of the report, include one fenced JSON code block with
exactly these keys:

```json
{
  "total_public_functions": 0,
  "can_edit_issue_caller": "module.function_name",
  "modules_with_raise_statements": 0
}
```

Where `total_public_functions` is the sum of module-level function counts
across all nine files, `can_edit_issue_caller` is the one function anywhere
in the codebase (excluding test files) that calls
`permissions.can_edit_issue`, given as `"module.function_name"`, and
`modules_with_raise_statements` is how many of the nine source files contain
at least one `raise` statement directly in their own code.

Do not return a launch/progress/waiting message. End the completed report
with this exact line: <!-- FANOUT_REPORT_COMPLETE -->
