# Repo navigation — which file belongs to which research direction

The project has had two directions. The **new** one (continuously-learned
quality/cost router) is active; the **old** one (hand-rule policy ladder) is now
a source of baselines, features, and later agent-workflow validation. Nothing
from the old direction is deleted — it is reused.

Full context: `edge-llm-client.md` (the binding spec) and
`claude-memory/wiki/` (`summary.md` → `hot.md` → `index.md`).

---

## NEW direction — continuously-learned edge/cloud router

Goal: for each request, choose local (small model on a GPU box) vs cloud
(DeepSeek-V4-Flash via Lumid), learning from a rolling window, to maximise local
serving load subject to `q_routed ≥ q_cloud − ε` **and** a measured edge
capacity/SLO envelope. Master note:
`claude-memory/wiki/projects/Continuously learned edge-cloud routing project.md`.

| path | what it is |
|---|---|
| `experiments/phase1_router/` | **Phase 1 pilot.** 300 paired MBPP+ tasks, local 27B vs cloud DeepSeek, unit-test graded. `orchestrator.py` runs generate→grade→features→policy→analysis→plots. `analysis.py` = quality gate + ε threshold + curves. `capacity.py` = **step 3** joint quality+capacity controller (offline `λ` sweep, admitted-local-rate cap at the envelope's 2 req/s). `policy.py` = baselines + the request-only `logreg`/`gbm` classifiers. |
| `experiments/capacity_profile/` | **27B capacity envelope.** Closed-loop concurrency ladder + Poisson arrival ramp against a live 27B box; measures the ~2 req/s SLO-safe rate that the capacity gate uses. |
| `experiments/capability_router/` | **Offline capability bootstrap.** 120 historical requests replayed through DeepSeek / 27B / 7B; initialises features/labels/baselines. Predates Phase 1; not the rolling policy itself. |
| `experiments/*/results/` | run artifacts — **gitignored** (phase1_router/results alone is ~1 GB). |

## OLD direction — hand-rule policy ladder (now baselines/features)

Numbered "Rungs" from `claude-memory/wiki/topics/Routing Policy.md` /
`Routing policy ladder OSDI plan`. These run inside the live `edgeproxy` proxy
and are exercised by the SWE-bench Pro campaigns.

| path | what it is | ladder rung |
|---|---|---|
| `edgeproxy/router.py` | `StaticPolicy` (feasibility only), `WarmLocalPolicy` (cold-branch escalation), `BranchDriftPolicy` (headroom pressure), `PlanningEscalationPolicy` (first-3-turns cloud), `PredictedRiskPolicy` (truncation-risk score), `CombinedPolicy` | Rung 0/1/3 + planning arm |
| `edgeproxy/reliability.py` | per-tool-suite circuit breaker (detect-only) | Rung 1 |
| `edgeproxy/cohort.py`, `cohort_parent.py`, `coordinator.py` | cohort tracking, benchmark parent-placement override, leader/follower barrier | Rung 4 |
| `edgeproxy/completion.py` | observe-only local-completion-time predictor | Rung 4 |
| `scripts/analyze_call_risk.py` | fits/validates the truncation-risk model behind `PredictedRiskPolicy`. **Also feeds the new direction** as one candidate input to the quality gate (`P(truncate) > τ` veto) — see the project note. |
| `scripts/analyze_cohort_parent_precision.py`, `analyze_planning_turns.py`, `analyze_duplicate_tool_results.py` | one-off analyses behind the cohort / planning / dedup findings |
| `scripts/run_fanout_policy_pair.sh`, `scripts/matched_matrix_metrics.py` | fan-out A/B campaign driver + metrics |
| `eval-suite/swebench/` | SWE-bench Pro campaigns (Docker-graded `fail_to_pass`/`pass_to_pass`). Phase 3 agent-workflow validation for the new direction. |
| `eval-suite/runner/`, `eval-suite/tasks/` | programmatic-checker eval suite + fixtures |

## SHARED infra (both directions)

| path | what it is |
|---|---|
| `edgeproxy/server.py` | the Anthropic-compatible local/cloud proxy (streaming tee, trace recording, per-call placement) |
| `edgeproxy/config.py`, `trace/`, `telemetry.py`, `timing.py`, `local_cache.py`, `cloud_cache.py`, `cost.py`, `shaping.py`, `episode.py`, `report_capture.py` | proxy support: config, JSONL trace schema, GPU/KV/queue telemetry, timing, cache probes, cost accounting, link shaping, episode/checkpoint metadata |
| `stdio_relay.py` | remote end of the SSH-command-exec bridge (Docker-on-Mac → remote vLLM); `AllowTcpForwarding no` blocks plain `-L` |
| `direct-gpu.sh`, `scripts/direct_gpu_relay.py`, `DIRECT_GPU_WORKFLOW.md` | direct-SSH workflow after a user provisions the pinned image: validate GPU → stream sanitized source → persistent bootstrap → real smoke → stable laptop relay |
| `flowmesh-up.sh`, `bootstrap.sh`, `trace-up.sh`, `ssh-workflow.yaml`, `ssh-workflow-5090.yaml`, `Dockerfile`, `setups/` | legacy CLI-submitted GPU box provisioning + shared box-side vLLM bring-up |
| `scripts/measure_*.py`, `scripts/edge_tpot.py` | standalone measurement scripts (TTFT curve, throughput, link shaping, Anthropic cache) |
| `tests/` | the canonical test suite (`python -m pytest tests/`). Running pytest from the repo root also tries to collect `eval-suite/**/fixture/` files and hits basename clashes — **run `tests/` explicitly.** |

## Docs / planning

| path | what it is |
|---|---|
| `edge-llm-client.md` | the binding spec (Goal / What you build / Plan / Success metrics). Conversation cannot redefine it. |
| `PLAN.md` | research plan (older Tier 0–3 numbering; distinct from the ladder's Rung 0–6) |
| `claude-memory/wiki/` | the knowledge base — read `summary.md` → `hot.md` → `index.md` first; gitignored |
| `README.md`, `setup-instructions.md` | setup |
| `PI-progress-report*.md`, `TRACE_QUALITY_REPORT_2026-08-29.md` | scratch progress reports — regenerated, not source of truth; `PI-progress-report*` is gitignored |
