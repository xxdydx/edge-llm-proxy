"""Agentic trace dataset: Stage 1 of the post-MBPP routing plan.

MBPP-based Phase 1 (see ``experiments/phase1_router/``) is frozen as the
controlled-baseline result -- see
``claude-memory/wiki/decisions/Freeze MBPP as the Phase 1 controlled
baseline.md``. This package builds a call-level dataset from real Claude
Code / SWE-bench Pro trajectories instead: collect real multi-turn traces
(``collect``), derive the handful of trajectory-level features that don't
already exist in ``edgeproxy``'s recorded traces (``features``), and replay
each captured call against both backends for paired call-level labels
(``replay``), before anything is wired into a live routing policy.
"""
