# Frozen four-task live behavior check — not launched

Scope: four **development** tasks only, one run per condition/seed: Flask 5014 (`swebench-flask-70ca03af28`), Sympy 11618 (`swebench-sympy-66abe976e0`), pytest 10051 (`swebench-pytest-f8692a712a`), pylint 4604 (`swebench-pylint-1715969d0b`). The two reserved holdout tasks remain untouched until a model/threshold is frozen. Django 10554 was removed after its 1,865-test official grader proved too heavy for this preliminary round; Astropy 12907 was rejected after its base/gold sanity exceeded 180 seconds. Sympy's official base/gold sanity passed, with a reported 37.3-second wall time. This task list is frozen against campaign task manifest v3, and the controller will refuse an incompatible selection manifest or prior cell checkpoint. Conditions: always-cloud, always-local, and the existing `routing-learned-agentic-heuristic` baseline. The new artifact only observes in shadow; it never changes placement.

Before any launch, confirm with the campaign owner: no overlapping workload or GPU instability; vLLM `/v1/models` gives the expected root/fingerprint/context; cloud responses reveal the actual backend/model fingerprint (not merely a configured alias); local cache probe renders the same effort-adjusted request as local dispatch; four instance Docker graders are valid. Reuse a previous cloud run only if instance, seed, agent config, cloud backend fingerprint, model parameters, harness revision, and grading are identical. Otherwise run the cloud arm again. A URL or model alias alone does not establish matching fingerprints.

With `EDGEPROXY_VLLM_URL` pointing to the verified local relay, the proposed invocation is:

```bash
uv run python eval-suite/swebench/runner/run_swebench.py \
  --instances swebench-flask-70ca03af28,swebench-sympy-66abe976e0,swebench-pytest-f8692a712a,swebench-pylint-1715969d0b \
  --conditions cloud,local,routing-learned-agentic-heuristic \
  --seeds 1 --cloud-parallelism 1 --local-parallelism 1 \
  --sequential-conditions \
  --vllm-url http://127.0.0.1:18004 \
  --max-local-tokens 100000 --local-token-margin 0.90 \
  --agentic-shadow-artifact experiments/agentic_router/results/quality_model_baseline205_v5_shadow_logreg.json \
  --experiment-id agentic-router-live-pilot-v1
```

This command is a **plan**, not authority to launch before coordination. Set `--upstream` and `--claude-model` explicitly to the verified cloud endpoint/agent setting at launch; do not infer a provider fingerprint from their aliases. The proxy runner should enable `local_cache_tracking=observe` for exact local prompt counts on every condition; if it does not, shadow reports unavailable features, never guesses. Do not start new work after 2026-09-17 16:30 SGT; preserve local checkpoints before 17:00 GPU expiry.

Scheduling constraint: the six-task `pilot_live.py` collector holds `eval-suite/swebench/results/agentic-router-pilot-v1/pilot_live.lock` for its whole run and waits for paired replay/judging before advancing. The four-task matrix must **not** be launched concurrently with it or the replay controller. The campaign owner must gracefully stop the replay `--follow` controller after all pairs are terminally accounted before this matrix starts; the matrix does not kill unidentified processes. The prepared `live_matrix.py` controller acquires both pilot locks non-blockingly, requires `pilot_live_status.json` to say finished/drained, checks a pause file and the fixed deadline before every cell, and runs one cell at a time with a 20-minute agent cap and 30-minute whole-cell cap. Its own watchdog ends each cell with a cleanup reserve **before** 16:30 SGT, rather than allowing a last job to run until 17:00 GPU expiry. It checkpoints metrics without prompt text and refuses to auto-repeat any interrupted or invalid cell. Its explicit command, **only after parent/campaign coordination**, is:

```bash
uv run python -m experiments.agentic_router.live_matrix --run
```

The raw multi-condition runner command above documents the exact workload but does **not** enforce the shared locks; prefer the controller. If insufficient time remains before drain, leave cells unrun rather than mixing overlapping GPU load into latency/cache comparisons.

For each task × condition report: official grade and grading validity, completion time, call count/recovery loops, local request and comparable token share, cloud and local input/output tokens separately, cache-read and missingness, backend fingerprints, and shadow score coverage/reasons. One seed per cell is a behavioral sanity check, not a task-success rate, a two-percentage-point guarantee, or proof the learned router is superior.
