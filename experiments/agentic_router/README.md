# agentic_router — Stage 1 of the post-MBPP routing plan

MBPP-based Phase 1 (`experiments/phase1_router/`) is frozen as the
controlled-baseline result (see `claude-memory/wiki/decisions/`). This
package builds a call-level dataset from real Claude Code / SWE-bench Pro
trajectories, then replays each captured call against both backends for
paired call-level labels — before anything is wired into a live routing
policy (Stage 2, gated on this stage's checkpoint).

## Pipeline

```
collect.build_trajectories()   -> AgentTrajectory list (real trace + verdict, offline join)
features.derive_all(...)       -> same list, .derived populated per call
replay.replay_all(...)         -> PairedCallExample list (real GPU + cloud spend)
```

`collect.py` and `features.py` are pure offline joins/derivations over
already-recorded data — no network, no spend, safe to re-run freely.
`replay.py` sends real requests to both backends and costs real GPU/cloud
time; do not run it without confirming scope first (see the project plan,
`~/.claude/plans/hazy-meandering-turing.md`, and
`gpu-eval-campaign-preferences` in the memory layer).

## What's reused vs. new

- Live harness, trace schema, and the call-level replay mechanism
  (`experiments/capability_router/executor.py`'s `replay_call`) are reused
  as-is — see module docstrings for exact file:line provenance.
- Genuinely new here: joining final task outcome onto every call in a
  trajectory (`collect.py`), and the handful of trajectory-level features
  that don't exist anywhere in `edgeproxy` yet — `previous_backend`,
  `consecutive_same_backend_turns`, `recent_tool_error_count`,
  `recent_test_failure_count`, `prior_response_truncated_or_invalid`,
  `repair_loop_flag`, `context_utilization_ratio`,
  `estimated_recompute_cost_if_switched` (`features.py`).

## Stage 1 checkpoint

Before any Stage 2 (live router-in-the-loop) work: fit the same
logreg/GBM quality-model approach used on MBPP against this call-level
paired dataset, and run the same 80-seed-style robustness audit
(`experiments/phase1_router/robustness.py`'s methodology). Only proceed to
Stage 2 if this produces a materially more reliable signal than MBPP's
47.5% epsilon-compliance baseline.
