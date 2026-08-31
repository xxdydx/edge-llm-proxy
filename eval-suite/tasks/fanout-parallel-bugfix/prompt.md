Your working directory contains three independent Python modules, each with
one failing test: `mod_a.py` / `test_mod_a.py`, `mod_b.py` / `test_mod_b.py`,
and `mod_c.py` / `test_mod_c.py`. Each module has exactly one bug, and the
three bugs are unrelated to each other.

Launch exactly three subagents concurrently, one per module: (1) fix the bug
in `mod_a.py` so `test_mod_a.py` passes, (2) fix the bug in `mod_b.py` so
`test_mod_b.py` passes, (3) fix the bug in `mod_c.py` so `test_mod_c.py`
passes. Use foreground Agent tool calls: emit the three independent Agent
calls together so they run in parallel, do not set `run_in_background`, and
do not return while any agent is pending. Each subagent must only edit its
one assigned module file, and must not edit any test file. Do not run
`pip install` or access the network.

After every subagent result has arrived, return one consolidated Markdown
report. Start the report with this exact line:
<!-- FANOUT_REPORT_START -->
Then use these exact level-two headings in this order: Executive Summary;
Fixes Applied; Open Questions.

Do not return a launch/progress/waiting message. End the completed report
with this exact line: <!-- FANOUT_REPORT_COMPLETE -->
