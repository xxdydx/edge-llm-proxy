Your working directory is `codebase/`. This is a read-only audit: do not
create, edit, or delete any files, run `pip install`, or access the network.

Launch exactly three subagents concurrently, one per subdirectory: (1)
`billing`, (2) `inventory`, (3) `notifications`. Use foreground Agent tool
calls: emit the three independent Agent calls together so they run in
parallel, do not set `run_in_background`, and do not return while any agent
is pending. Each subagent should report, for its one subdirectory only: the
public (non-underscore) top-level function names defined there, and whether
a test file is present in that subdirectory.

After every subagent result has arrived, return one consolidated Markdown
report. Start the report with this exact line:
<!-- FANOUT_REPORT_START -->
Then use these exact level-two headings in this order: Executive Summary;
Findings from Each Agent; Open Questions.

At the very end of the report, include one fenced JSON code block with
exactly these keys, filled in from what the subagents found:

```json
{
  "total_public_functions": 0,
  "packages_without_tests": ["..."]
}
```

Do not return a launch/progress/waiting message. End the completed report
with this exact line: <!-- FANOUT_REPORT_COMPLETE -->
