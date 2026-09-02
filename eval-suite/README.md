# eval-suite

A rung-3 task suite (see `PLAN.md` §5 and `claude-memory/wiki/topics/Quality
Measurement.md`): deterministic agentic tasks with predeclared programmatic
checkers, run through the full Claude Code loop under both the cloud-only and
static-routing policies. This is the piece that turns single-trajectory
observations ("one local run failed its task") into pass rates with
confidence intervals across seeds - the binding constraint called out in the
project's next-priority note.

It generalizes `scripts/run_fanout_policy_pair.sh` (one fixed fan-out prompt,
one cloud/routing pair) into N tasks x 2 conditions x M seeds, run against
synthetic fixture repos instead of the live `edgeproxy/` package so concurrent
jobs never race on the same files.

## Layout

```
eval-suite/
  tasks/<task-id>/
    task.json       # metadata: category, timeout, fanout_required, expects_report
    prompt.md        # the exact prompt given to Claude Code
    fixture/         # pristine starting files, copied fresh into every job's sandbox
    checker.py       # programmatic verdict: {"passed", "score", "reason", "details"}
    answer_key.json  # exact expected facts, for exploration/fanout tasks
  lib/
    checker_common.py  # shared, stdlib-only helpers every checker.py imports
  runner/
    task_spec.py     # task discovery/loading
    stats.py         # per-cell pass-rate confidence interval
    run_eval.py       # the campaign runner
  results/<experiment-id>/   # gitignored: sandboxes, proxy logs, claude stream, verdicts, summary

flowmesh/traces/<suite-name>-run-<condition>-<run-stamp>-<task-id>-seed<seed>/
  # gitignored, same as every other edgeproxy trace: trace.jsonl,
  # trace.graph.json, trace.mmd - one folder per job, outside eval-suite/
```

## The tasks

### Original 12 (small, single-fixture, mostly &lt;100 lines)

| id | category | what it tests |
| --- | --- | --- |
| `explore-pricing-api` | exploration | read-only comprehension, exact answer key |
| `explore-config-defaults` | exploration | exception-handling comprehension, decoy-resistant |
| `explore-call-graph` | exploration | call-site tracing, resistant to naive `grep -c` |
| `fix-discount-tier-off-by-one` | edit | boundary-condition bugfix, unit-tested |
| `fix-null-handling` | edit | missing-key crash fix, unit-tested |
| `fix-sorting-ties` | edit | tie-break rule fix, unit-tested |
| `fix-lru-cache-eviction` | edit | stateful class bugfix, unit-tested |
| `refactor-extract-pricing` | refactor | extract-function, behavior + structure checked |
| `refactor-remove-duplication` | refactor | cross-file dedup, literal-duplication check |
| `refactor-rename-for-clarity` | refactor | consistent rename across call sites |
| `fanout-repo-audit` | fanout | forced 3-way concurrent subagents, report + answer key |
| `fanout-parallel-bugfix` | fanout | forced 3-way concurrent subagents, non-overlapping edits |

These are deliberately small and mechanical (median job wall time 57s across
a 120-job campaign) - they validate that the harness, checkers, and both
policies work correctly end-to-end, but they are too easy to reveal where
local serving actually breaks down. `bigapp/` below exists to close that gap.

### Hard tier: `bigapp/`, a ~370-line, 9-module issue-tracking service

Both tasks below share the same fixture shape (`models.py`, `storage.py`,
`validation.py`, `workflow.py`, `permissions.py`, `search.py`,
`notifications.py`, `reporting.py`, `api.py`, plus one test file per
module) - roughly 5x the line count of anything in the original 12, and
still well inside the local backend's 100K-token budget, but large enough
to require real cross-file reading instead of one-file comprehension.

| id | category | what it tests |
| --- | --- | --- |
| `feature-priority-levels` | feature | a real feature (issue priority levels) requiring coordinated changes across 6 of the 9 modules, graded by a held-out 11-test file exercised through the public `api.py` interface |
| `fanout-architecture-map` | fanout | exactly 3 concurrent subagents, each reading a 3-file slice of the whole codebase, consolidating into an architecture report whose facts (e.g. total function count, the one cross-module call site into `permissions.can_edit_issue`) require genuine synthesis across agents, not just concatenation |

`feature-priority-levels` reuses the same `edit_task_check` pattern as the
original edit/refactor tasks (protected test files, full pytest suite must
pass) - the difference is scope, not mechanism: touching 1 file vs. 6.
`fanout-architecture-map` reuses the fan-out report-marker/heading/answer-key
pattern from `fanout-repo-audit`, scaled to a 9-file codebase split three
ways instead of a 3-file codebase split three ways.

Exploration and fanout tasks are read-only or delegate-only and end with a
fenced JSON answer block (between `<!-- ANSWER_START -->` / `<!-- ANSWER_END
-->` markers, or `<!-- FANOUT_REPORT_START/COMPLETE -->` for fan-out tasks,
reusing `edgeproxy.report_capture`'s existing convention) so grading never
depends on an LLM judge - only exact/structural checks, per PLAN.md §5's
"programmatic checkers, not LLM judges." Edit/refactor tasks are graded by
running the fixture's own `pytest` suite plus, for refactors, an AST/text
check that the requested structural change actually happened (not just that
behavior didn't regress).

Every fixture is a small stdlib-only Python package (no third-party
dependencies), so no `pip install` step or network access is ever required
inside the sandbox.

## Running a campaign

```bash
# from the repo root, on the GPU box, after bootstrap has started vLLM:
python3 eval-suite/runner/run_eval.py \
  --tasks all \
  --conditions cloud,routing \
  --seeds 5 \
  --cloud-parallelism 4 \
  --local-parallelism 1
```

- `--conditions cloud,routing` maps to `edgeproxy --policy cloud-only` and
  `--policy static`, exactly like the reference script.
- `--local-parallelism` defaults to 1: routing jobs share one real vLLM engine
  on one GPU, so running them concurrently confounds pass/fail and latency
  with queueing. Raise it only if you are deliberately studying that.
- Each job gets an isolated sandbox (`results/<exp>/sandboxes/<job-id>/`), its
  own edgeproxy instance/port, a captured Claude stream transcript, and a
  checker verdict.
- Raw traces are **not** buried inside `eval-suite/results/` - they go under
  the project's canonical top-level `traces/` directory (same as every other
  edgeproxy trace, and gitignored the same way), one folder per job:
  `traces/<suite-name>-run-<condition>-<run-stamp>-<task-id>-seed<seed>/`
  (e.g. `traces/eval-suite-1-run-cloud-20260830T120000Z-explore-pricing-api-seed1/`).
  Each folder holds `trace.jsonl` (the raw edgeproxy trace), `trace.graph.json`
  and `trace.mmd` (via `edgeproxy.trace.graph`, same as the reference script).
  `--suite-name` (default `eval-suite-1`) distinguishes this suite from any
  future one without changing the rest of the naming; `--traces-root`
  overrides the destination if you don't want `<repo-dir>/traces`.
- `--tasks explore-pricing-api,fix-null-handling` runs a subset; useful for a
  quick smoke run before committing to the full 14 x 2 x N job matrix (N x 5
  = 140 jobs). The two hard-tier tasks have much longer timeouts (600s and
  900s vs. 240-600s for the original 12) and take noticeably longer per job -
  budget extra wall time when including them.

Output: `results/<experiment-id>/summary.json` (every job plus per-cell
pass rate, CI, mean score, mean wall time) and `summary.md` (the same as a
table).

### Prerequisites

- `claude` CLI on `PATH` (or `--claude-bin`), cloud credentials configured.
- For the routing condition: vLLM already running and reachable at
  `--vllm-url` (see `scripts/run_fanout_policy_pair.sh` for how bootstrap
  starts it).
- **pytest available on the interpreter that runs checkers** (`--python-bin`,
  default `sys.executable`). This repo's `.venv` does not currently have
  `pytest` installed even though `tests/` depends on it - point `--python-bin`
  at an interpreter that has it, or `pip install pytest` into `.venv` first.
  `eval-suite/lib/checker_common.run_pytest` will otherwise report every
  edit/refactor task as failed with "No module named pytest", which looks
  identical to a real task failure unless you check the checker's raw stderr.

## Validating a checker without spending on a live run

Every checker is a subprocess with a fixed CLI contract, so it can be
exercised directly against a hand-written "gold" fix instead of a live Claude
Code transcript:

```bash
python3 eval-suite/tasks/fix-null-handling/checker.py \
  --sandbox /path/to/a/fixed/copy \
  --fixture eval-suite/tasks/fix-null-handling/fixture \
  --report /dev/null \
  --out /tmp/verdict.json
```

All 14 checkers were validated this way before being trusted: each one fails
against its untouched buggy/unrefactored fixture and passes against a
hand-written correct solution (and, for exploration/fanout tasks, fails
against a plausible wrong answer too). `feature-priority-levels` was
additionally validated against a full hand-written gold implementation of
the feature spec (all 39 tests), not just a partial patch.

## Adding a task

1. `mkdir -p eval-suite/tasks/<id>/fixture` and write the starting files.
2. Write `prompt.md`. Read-only/fanout tasks should end with the exact
   marker/fence convention shown in existing tasks so the checker can parse a
   structured answer without an LLM judge.
3. Write `checker.py` using `lib/checker_common`: `edit_task_check(...)` for a
   straightforward "tests pass, protected files untouched" task, or compose
   the lower-level helpers (`run_pytest`, `has_top_level_function`,
   `function_calls_name`, `grep_count`, `extract_answer_json`,
   `has_required_headings`, `has_report_markers`) for anything more specific.
   Always end with `cc.run_checker_main(check)`.
4. Write `task.json` (see any existing task for the schema) and, for
   exploration/fanout tasks, `answer_key.json`.
5. Validate offline the same way as step "Validating a checker" above before
   trusting it in a live campaign.

## What this does and doesn't answer

The first full paired campaign ran on `qwen38-27b` (RTX 6000 Ada) on
2026-09-01, experiment `eval-suite-20260901T104145Z`: all 12 original tasks x
2 conditions x 5 seeds = 120 jobs, **119/120 passed (99.2%)**. The one gap
(`explore-call-graph`/cloud, 4/5) is a genuine model error (wrong function
name), not a checker bug - see `results/eval-suite-20260901T104145Z/`.
Within the routing condition, 402/462 calls (87.0%) were actually served
locally; the rest went cloud, concentrated in a specific per-session call
type rather than spread evenly - worth characterizing further.

That result is real but limited: every original-12 fixture is 30-82 lines,
median job wall time was 57s, and 87% local placement mostly reflects that
these small fixtures trivially satisfy the static policy's feasibility
gates rather than the router making an interesting choice. It doesn't show
whether local holds up on large-context, multi-file, or long-horizon work -
the two hard-tier tasks (`feature-priority-levels`,
`fanout-architecture-map`) exist to test exactly that gap and have not yet
been run as part of a full campaign.

The confidence-interval calculation in `runner/stats.py` (Wilson score
interval) is implemented and live-validated on the 120-job campaign above.
