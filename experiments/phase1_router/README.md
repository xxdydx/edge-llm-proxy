# Phase 1 pilot — continuously-learned edge/cloud routing

Existence-case pilot for [[Continuously learned edge-cloud routing project]],
section "First existence-case pilot (approved 2026-09-10)".

## What it does

- **Workload**: MBPP+ **v0.2.0** (EvalPlus content hash `ee43ecabebf20dee…`),
  300 standalone coding problems selected by a seeded shuffle of the walkable
  order (skips the handful of tasks whose contract rejects every input).
  Loaded through `evalplus.data.get_mbpp_plus` so the official per-task input
  deserialisation applies. Not committed; the local gz snapshot's SHA256 and
  the EvalPlus hash are recorded in each run's `dataset_manifest.json`.
- **Pair**: Qwen3.8-27B-NVFP4 (local vLLM, 100K ctx, relay `:18004`) vs
  DeepSeek-V4-Flash (Lumid). No Claude Code, no tools, no agent loop, no 7B.
- **Paired generation**: identical prompt bytes to both, `temperature=0`,
  reusing `capability_router.executor` (identity allowlist, bounded fail-fast:
  2 attempts / 90 s read-inactivity / 10 s connect / 2 s backoff, checkpoint
  resume).
- **Grading**: **model code only ever runs in Docker.** Each completion is
  graded by `evalplus.eval.untrusted_check` inside
  `phase1-grader:evalplus-0.3.1` (`--network none --read-only --cap-drop ALL
  --pids-limit --memory --cpus --user nobody`, tmpfs `/tmp`). Groundtruth and
  per-input reference times are precomputed on the host from the **trusted
  canonical** solution (`trusted_exec`, independent deepcopy per input). The
  container deepcopies inputs again before the candidate runs. Result is
  emitted on a saved real-stdout fd behind a sentinel so candidate stdout
  cannot corrupt parsing. `plus_pass` (all base + EvalPlus inputs correct) is
  the correctness label.
- **Policies**: `always_cloud`, `always_local`, `static_heuristic` (prompt-len
  cutoff), `oracle` (route local iff local passes), and `logreg` / `gbm` —
  **this experiment's own** request-only classifiers of P(local plus_pass).
  These are NOT routing-ladder Policy 3/4/5/6. Grouped train/val/test split
  (`n_task_groups=15`, group-disjoint); threshold chosen on VAL under
  `epsilon = 0.02`, reported once on TEST.
- **Rolling-window online sim**: 5 chronological windows over the selection
  order; from window k≥1 fit logreg on windows < k, evaluate on window k,
  compare to always-cloud / always-local / oracle.

## Required outputs (`analysis_report.json` + `plots/`)

1. **quality vs local-traffic share** — per learned policy + oracle, at switch
   multiplier 0x and 10x.
2. **quality-cost Pareto across routing thresholds** — cost is the *normalized
   total cost per request* (declared proxy, assumptions in
   `cost.COST_ASSUMPTIONS`), **not** local share; epsilon-feasible points marked.
3. **cache-switch-cost sensitivity** — normalized cost vs the local-output
   token price ratio, at 0x and 10x cold-prefill penalty.
   Raw components (cloud in/out tokens, local in/out tokens, local & cloud
   latency, switch count) are reported **separately, never summed by default**.

## Step 3 — joint quality + capacity controller (`capacity.py`)

Built 2026-09-11. A second gate on top of the quality classifier above:
even a request the quality gate marks safe is only admitted local if the
27B box has *room*. The 300-pair dataset has no arrival timestamps, so
capacity is a **simulated offered load `lambda`** (req/s) streamed over
`ordered_all` (all 300, selection order).

- **Primary gate**: a rolling admitted-local-**rate** cap at
  `capacity_max_local_req_per_s` (2.0, the SLO-safe rate from
  `experiments/capacity_profile/results/run-20260910T135027Z`). The envelope's
  binding constraint was **latency**, not GPU saturation (KV < 8%, no
  preemptions), so this is a rate limit, not a GPU-second budget.
  10 s rolling window (not the envelope's 60 s — 300 requests can't sustain a
  rate over 60 s at high `lambda`).
- **Secondary/diagnostic**: `service_seconds()` (affine, floor + slope,
  anchored to the envelope's 0.51 s/req at 256 output tokens) feeds
  `local_service_share` / `peak_window_gpu_util` but does not gate.
- `joint_capacity_sweep` compares **joint** (both gates) / **quality_only**
  (capacity ignored, violations still flagged) / **capacity_only** /
  **oracle_cap** per `lambda`. Wired into `analysis.run_analysis` ->
  `report["joint_capacity"]`, plotted as `plots/joint_capacity_admission.*`.
- Result: [[Joint quality-capacity controller offline sweep on the Phase 1 pilot]]
  — joint trades local share 39.7% -> 18.3% as offered load rises 0.5 -> 12
  req/s, without breaking the quality constraint; a naive quality-only router
  would run the box at 1.15-2.75x the SLO-safe rate at >= 4 req/s.

## Run

```
python -m experiments.phase1_router.orchestrator --dataset-only          # host only, no model
python -m experiments.phase1_router.orchestrator --smoke-test 3 --resume-run-dir results/run-...
python -m experiments.phase1_router.orchestrator --full-batch --resume-run-dir results/run-...
python -m experiments.phase1_router.orchestrator --analysis-only --resume-run-dir results/run-...
```

Tests: `tests/test_phase1_router.py` (42; 8 are live Docker grader-fidelity
checks — gold-vs-gold, known-wrong, raising, atol, mutation-independence,
stdout isolation; `CapacityGateTests` covers `capacity.py`).
