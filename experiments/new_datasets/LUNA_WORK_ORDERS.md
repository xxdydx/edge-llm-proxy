# Mandatory task-harvest override for all Luna workers

Use `TASK_SOURCE_LOCK.json`, `TASK_HARVEST_SPEC.md`, and `harvest_task_manifest.py` from this packet. These replace any older open-ended source-selection wording below. `SWE-bench/SWE-smith-py` is the only synthetic source for this campaign.

W2 must run the included selector, not write a different selector or invent IDs. It must return the complete literal training/validation lists, duplicate/exposure exclusions, exact revision hashes, a per-repository supply report, preflight certificates and frozen-stage manifests. W6 must independently check the tests, hashes, split ownership, missing-value semantics and real preflight reports. Preparation-time synthetic tests do not certify source downloads or benchmark runtimes.

No experimental generation from provisional ID lists. No replacement based on experimental pass/fail. A branch slot maps to its final frozen task through `slot_id`; keep its split when a valid preflight replacement occurs. All worker outputs must distinguish requested quota from verified eligible count.

---

# Luna work-order templates

Sonnet should instantiate these after auditing actual repository paths. All workers inherit SONNET_MASTER_PROMPT.md and the frozen shared contracts. Use the existing user-configured Luna launcher, not an invented native model override.

## Common assignment wrapper

You are an engineering worker, not an experimental coding agent. Implement only the assigned pipeline component. Do not solve benchmark issues, alter Qwen/DeepSeek candidate patches, launch uncontrolled paid jobs, change the target models, train routers, or modify files owned by another worker.

First read the shared schemas/protocol and inspect actual interfaces. Propose any contract change to Sonnet before implementing it. Synthetic fixtures must be tagged and kept out of real exports. Preserve existing tests. Return exact changes, runnable test commands with actual outcomes, integration notes, and blockers. Never report work or tests you did not perform.

## W1 — Contracts, persistence and legacy import

Own: dataset schema/storage/importer modules and their unit tests.

Implement validated records for tasks, trajectories, prefixes, invocations, checkpoints, branch pairs, preferences and graders. Maintain stable identity relationships and raw artifact hashes. Implement transactional job leases, immutable attempts, atomic JSONL exports and import-only legacy migration. Distinguish logical calls, transport attempts and streaming events. Preserve nullable missingness and protocol partitions.

Importer requirements: original call.request/call.response, local replay and cloud replay become separate referenced objects; attach terminal labels only from unambiguous official grade joins; historical replay-only pairs never become branch evidence. Do not infer judge aggregation from ambiguous endpoint-specific labels.

Acceptance: round-trip typed fixtures; hash/reference verification; interrupted writer recovery; duplicate job claim tests; legacy records with zero partial usage; branch export empty when no real branch evidence; no mutation of input files.

## W2 — Task adapters and grading

Own: source-task adapters, preflight, split/task manifests, grading wrappers and tests.

Implement SWE-Gym and SWE-smith adapters against pinned official source code. The experimental agent remains Claude Code. Correctly distinguish SWE-smith bug-introducing task patches from generated repair patches. Separate agent-facing task information from grader-only references. Verify buggy and reference states in separate clean environments. Enforce unique grader identity using run, task, patch and evaluator hashes.

Build deterministic task/reserve ordering and grouped splits. Canonicalise upstream repositories and duplicate/mutation groups across sources. Record every task exclusion and its pre-outcome reason. Expose typed preflight and grader results to W3/W4.

Acceptance: actual environment smoke once authorised; reference material inaccessible to experimental tools; distinct patches never reuse the wrong cached grade; interpreter/dependency errors are not successful tests; timeout/infra/protocol states are separate; source-specific patch handling fixture.

## W3 — Claude Code boundary capture and Dataset A

Own: experimental launch adapter, proxy boundary hooks, main-call classification, original-run orchestration and A export integration.

Instrument actual pre-dispatch logical requests. Commit complete request/state references before forwarding, then capture every invocation/response and Claude Code tool observations. Maintain causal IDs, compaction mapping and task-isolated session/HOME state. Respect explicit edge-only and cloud-only policies; no engineering-model fallback. Make worker orchestration invisible to target agents.

Preserve every original prefix, then select up to five eligible prefixes uniformly without replacement for core preference annotation. Derive counters using prior observations only. Join terminal grades from W2 without reading labels during sampling. Implement a pre-call barrier API for W4, with explicit quiescence and remaining-budget state.

Acceptance: stream event vs call tests; requests have no future material; 'No module named pytest' fixture; local/cloud fresh task state; no memory sharing; proxy failure does not produce missing-history valid rows; complete durability before cleanup.

## W4 — Checkpoint certification and Dataset B

Own: checkpoint manager, restore/reconstruction adapter, branch scheduler and B tests.

Implement one seeded reservoir checkpoint per selected source trajectory, captured before response generation. Preserve filesystem plus relevant Claude Code executor/session state. Prove supported restore semantics rather than assuming rewind/fork suffices. Certify restored filesystem, target request, working directory, tool IDs, read-before-edit state and remaining budgets. Explicitly reject unsupported persistent-process states.

From a certified checkpoint execute edge-next/edge-tail and cloud-next/edge-tail using real Claude Code tool execution. Generate fresh initial candidates; retain response hashes. Record random branch order, all four outcome combinations, incomplete pairs and random repeated pairs. Grade through W2. B-specific judge annotations must use the exact executed initial candidates.

Acceptance: Bash edits/untracked/tmp files/cwd/symlinks fixture; no shared writable branch state; no new 'continue' prompt; same next-request verification; no original-future splice; edge tail after each intervention; sampling simulation; capture failures not silently replaced; no B-valid record without certificate and valid grades.

This work package must report a genuine blocker rather than fabricate exact restoration. A/C collection may proceed separately when Sonnet authorises it.

## W5 — Shadow preference labels and serving records

Own: shadow generation scheduler, judge rendering/parsing/aggregation, serving telemetry integration and controlled profiling.

Reuse original candidates only with verified configuration/request provenance; create exactly the missing candidate otherwise under an explicit alternative protocol. Do not execute shadow actions. Version a backend-anonymous judge view and use JUDGE_SYSTEM_PROMPT.txt. Two separate orientations with saved identity mappings; strict aggregation; no semantic retry-until-agreement; no silent context truncation.

Capture serving snapshots per actual invocation, not per historical logical state. Map provider usage using audited semantics, preserve raw values, mark unknowns null. Separate pre-dispatch estimates from post-response measurements. Do not infer per-request cache or decode timings from unsupported aggregate/buffered observations. Profile only on approved dedicated resources, log priming/background costs, and prevent shadow runs from contaminating unlabelled baseline load.

Acceptance: reversed candidate mapping; both-inadequate vs equivalent; missing vs uncertain; judge injection fixture; missing snapshot; zero partial usage; buffered stream; original/replay cache mismatch; no tools executed during profiling/shadow; bounded concurrency/retry.

## W6 — Independent audit and release gate

Own: acceptance test suite, data-integrity reports and findings, not unreviewed rewrites of other workers' modules.

Attempt to falsify the main claims: exact state restoration, correct outcome joins, proper sampling, no hidden model fallback, complete pre-call history, no future-label leakage, and correctly attributed serving measurements. Use malformed/incomplete fixtures and interrupted-job fault injection. Audit effective sandbox mounts/settings and benchmark-reference separation.

Report each finding with a minimal reproduction and affected data scope. Sonnet coordinates fixes. Verify that corrected exports are regenerated under versioned transformations and that invalid originals remain traceable. Independently inspect a few real accepted A/B/C records and their raw evidence after authorised smoke collection.

Acceptance: signed-off checklist with exact tests and data counts, or explicit BLOCKED items. Never approve B because both branches merely have the same task ID or the same final checkout.
