# Capability router: cloud-relative teacher-forcing study

## What this measures

For a fixed set of real historical coding requests drawn from **passing**
cloud-only SWE-bench Pro trajectories, replay the exact same request
independently through:

- **local_27b** — Qwen3.8-27B-NVFP4, via the local relay
- **cloud_deepseek** — DeepSeek-V4-Flash, via the same Lumid gateway the
  original trajectory used

The recorded cloud action from the original passing trajectory is the
**teacher**. Each replay is scored for **action equivalence** to that
teacher action (same tool, same normalized arguments) and, independently,
for **schema validity** (does its tool_use input satisfy the tool's own
JSON schema).

## What this is *not*

**This is cloud-relative teacher forcing, not proof of globally correct
coding.** The teacher label only tells you "did this reply do what the
already-successful cloud trajectory did at this exact turn" — never "was
this action objectively correct." Two specific things are always
unknowable from a trace file alone and are labeled `UNKNOWN` rather than
scored either way:

- **Free text.** If the teacher's own action was prose (no tool call), its
  "correctness" cannot be judged syntactically, so every replay against it
  is `UNKNOWN` regardless of what either backend said.
- **State-dependent equivalence.** A different-but-still-valid tool call
  (e.g. an `Edit` with different `old_string` that also happens to fix the
  bug) cannot be distinguished from a genuinely wrong one without the
  per-turn repository snapshot at that exact point, which this project does
  not record. Only an *exact* structural match is called `EQUIVALENT`; a
  clean, comparable mismatch (different tool, or same tool with different
  input) is `NOT_EQUIVALENT`; everything else stays `UNKNOWN`.

Every learned model and baseline in `analysis.py` is trained and evaluated
**only on known (`EQUIVALENT`/`NOT_EQUIVALENT`) labels** for exactly this
reason — `UNKNOWN` rows are excluded, not treated as a default class.

## Fixed scope

- Qwen3.8-27B-NVFP4 vs DeepSeek-V4-Flash first; endpoints/model ids are
  configurable in `config.py` for a later 7B repeat.
- Request-only features (see `models.FEATURE_NAMES`) — no cache state,
  online adaptation, cohort signals, GPU/queue telemetry, embeddings, or
  bandits.
- ~120 deduplicated requests across at least 12 SWE-bench task groups where
  available (`config.ExperimentConfig.min_task_groups`); the plan
  explicitly allows adjusting the count down rather than forcing
  unscorable calls into the sample just to hit a number.
- Grouped by SWE-bench task instance for both sampling and the train/test
  split, so the same task's calls never appear on both sides of a split.
- No fallback: `executor.py` enforces a preflight identity probe, a
  per-replay identity check on every single call, and a postflight probe —
  any identity mismatch is a recorded outcome (`IDENTITY_MISMATCH`), never
  a silently-accepted guess. The cloud backend is additionally checked
  against a "zero-genuine-local-contamination" gate: a cloud reply whose
  `model` field looks local-flavored is never accepted as real cloud data
  (this project has hit exactly that failure mode twice before, see the
  wiki).

## Pipeline

```
dataset.py      -- select & dedupe calls from real passing cloud trajectories
executor.py     -- strict dual replay against local_27b and cloud_deepseek
labels.py       -- action equivalence + schema validity, per replay
models.py       -- logistic regression + LightGBM on known local_label rows
analysis.py     -- AlwaysCloud / AlwaysLocal / Policy4 / learned / oracle,
                   threshold sweep, epsilon-constrained opportunity frontier
plots.py        -- Pareto plot (falls back to CSV if matplotlib is absent)
orchestrator.py -- runs all of the above end to end
```

```
python -m experiments.capability_router.orchestrator            # full run
python -m experiments.capability_router.orchestrator --dataset-only
python -m experiments.capability_router.orchestrator --fresh-dataset
```

Outputs land in `experiments/capability_router/data/` (the selected dataset
+ manifest) and `experiments/capability_router/results/` (labeled
examples, `analysis_report.json`, `pareto.png`/`.csv`).

## Reuse, not reimplementation

- `edgeproxy.trace.record.request_identity` / `validate_tool_use_blocks` —
  dedup fingerprint and schema-validity checks.
- `edgeproxy.trace.replay.calls` — filters trace records to real
  `/v1/messages` calls.
- `edgeproxy.router.extract_features` / `CallFeatures` / `PlanningEscalationPolicy`
  — the exact same request-only features and the exact deployed Policy 4,
  read only, never modified.
- `eval-suite/runner/run_eval.py.load_dotenv_into_environ` — the project's
  existing safe `.env` loader (declares key names into `os.environ`, never
  logs values).

No file under `edgeproxy/` is edited by this package.
