# Exact task-harvest contract — native Claude Code collection

**This contract replaces open-ended source/task selection in the original master prompt.** Task sources, repository quotas, deterministic ranking, split ownership, initial IDs, and replacement rules are fixed below. Sonnet/Luna must not independently search for easier tasks, draw a convenient first slice, or select tasks because one experimental model succeeds.

The attached `harvest_task_manifest.py` materialises the complete instance-ID lists. It does not use an LLM and does not execute experiments. Source metadata and the two startup IDs were checked on the official public sites. Full parquet download was unavailable in the preparation container, so the complete 600-task ID manifest and per-repository eligible supply have **not** been verified here. The script must produce and validate those manifests on the user's network-enabled machine before collection. No benchmark runtime has been measured by the author of this packet.

## 1. The source decision is fixed

| Key | Exact Hugging Face dataset | Config / split | Revision |
|---|---|---|---|
| gym | `SWE-Gym/SWE-Gym` | default / train | `bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb` |
| lite | `SWE-Gym/SWE-Gym-Lite` | default / train | `f70b1a29ab120eb0a0ee7a1deb029825e735b2b0` |
| smith | `SWE-bench/SWE-smith-py` | default / train | `77cab9055d42ab4a5c25c89a8f937096db13558e` |
| verified | `princeton-nlp/SWE-bench_Verified` | default / test | `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` |

Use only these sources for this campaign. `lite` supplies membership priority for real tasks and the startup tasks; it is **not an additional set of independent tasks** on top of `gym`. `verified` supplies final-evaluation reservations only; do not run those tasks now.

Use `SWE-smith-py`, not the retired monolithic `SWE-bench/SWE-smith` dataset. The upstream project explicitly recommends language-specific releases. This campaign is intentionally Python-focused; do not add JavaScript, web browsing, UI, GPU-training or multi-language tasks to meet a row target. Public OpenHands trajectories, SPROUT, MBPP and generic prompt datasets are not collection sources for this campaign.

Pinned parquet filenames are in `TASK_SOURCE_LOCK.json`. Gym/Lite each have `data/train-00000-of-00001.parquet`; Verified has `data/test-00000-of-00001.parquet`; Smith-py has all 11 files from `data/train-00000-of-00011.parquet` through `data/train-00010-of-00011.parquet`. Fetch the pinned files, not a mutable viewer row order. Record each file's actual SHA-256 and downloaded size.

## 2. Exact real-task quotas: 300 training + 60 validation

All these tasks come from `SWE-Gym/SWE-Gym`. Canonical repository names are used for counting; preserve the original source spelling too.

| Canonical repository | Training | Validation |
|---|---:|---:|
| `getmoto/moto` | 50 | 10 |
| `python/mypy` | 50 | 10 |
| `pydantic/pydantic` | 40 | 8 |
| `facebookresearch/hydra` | 30 | 6 |
| `iterative/dvc` | 40 | 8 |
| `conan-io/conan` | 30 | 6 |
| `pandas-dev/pandas` | 40 | 8 |
| `dask/dask` | 20 | 4 |
| **Total** | **300** | **60** |

This is a design choice spanning APIs, type checking, validation, configuration and data tooling. It is not a claim that every task from these repositories is fast. MONAI, Modin and Bokeh are outside this initial real-task allowlist to keep integration scope bounded; do not use them as automatic replacements.

## 3. Exact synthetic-task quotas: 200 training + 40 validation

All come from `SWE-bench/SWE-smith-py`. Take **10 eligible unique task groups from each of these 20 training repositories**:

```text
pallets/click
pallets/jinja
pallets/markupsafe
marshmallow-code/marshmallow
marshmallow-code/apispec
python-jsonschema/jsonschema
oauthlib/oauthlib
mahmoud/boltons
mewwts/addict
arrow-py/arrow
spulec/freezegun
suor/funcy
gruns/furl
python-hyper/h11
theskumar/python-dotenv
pycqa/flake8
davidhalter/parso
erikrose/parsimonious
r1chardj0n3s/parse
lepture/mistune
```

Take **10 from each of these four validation-only repositories**:

```text
mozilla/bleach
seperman/deepdiff
cloudpipe/cloudpickle
cookiecutter/cookiecutter
```

These four repositories must contribute **zero training tasks in this campaign**. Report this as a repository-held-out synthetic validation slice, separately from the real-task validation slice whose repositories overlap training. Do not claim the repositories were absent from foundation-model pretraining or all historical project experiments.

The official SWE-smith Python profile registry contains these repositories. Profile existence does not prove that 10 tasks survive this campaign's metadata, duplicate and runtime checks. The script reports shortages instead of inventing IDs or relaxing filters.

The primary target is therefore **500 train + 100 validation**, with **28 training repositories and 32 distinct development repositories**, conditional on verified task supply. Maximum repository share in training is 50/500 = 10%.

## 4. Literal startup task IDs

Run these two first, counted inside the `getmoto/moto` training quota rather than extra tasks:

| Order | Instance ID | Prescribed base commit | Issue |
|---|---|---|---|
| 1 | `getmoto__moto-5752` | `b2300f1eae1323e3e8bc45f97e530ce129dff12e` | `describe_parameters` depends on filter order |
| 2 | `getmoto__moto-6178` | `9f91ac0eb96b1868905e1555cb819757c055b652` | DynamoDB query brackets in `KeyConditionExpression` |

These identifiers and metadata were checked against the public Lite release. Use the complete original `problem_statement`, not the shorthand issue text above. Verify membership and base commits in the pinned downloads. They are integration smoke tasks, not certified fast/easy model-success examples.

If prior exposure or a metadata mismatch disqualifies one, report the explicit conflict. Do not silently substitute another and retain the same smoke ID. Runtime-preflight rejection may use the documented same-repository backup process, preserving the replacement record.

## 5. How every remaining exact ID is selected

Use the included implementation; do not reimplement this in a prompt or ask an LLM to rank tasks.

1. Read the pinned source rows, preserving dataset/revision/file/row locators.
2. Canonicalise source repositories. A Smith mirror such as `swesmith/owner__repo.<commit>` counts as upstream `owner/repo`, not an independent repository.
3. Require a nonempty issue with at least 40 characters and nonempty `FAIL_TO_PASS` evidence. Do not turn reference patches into issue descriptions for rows missing an issue.
4. For Smith require an image reference, nonempty `PASS_TO_PASS`, and a mutation confined to 1–3 `.py` files. Exclude mutations in test/build files by the included predicate. This is a declared limited-scope synthetic subset, not the full release.
5. Group exact issue-text/patch aliases within a canonical repository and identical synthetic mutation-location signatures. Select one representative per group. Preserve membership and audit approximate/semantic duplicates separately; the heuristic is not a proof that all near-duplicates are eliminated.
6. Apply the prior-exposure exclusion list to entire detected groups. Identifiers are not sufficient for every historical alias; the source adapter must also audit historical issue/repository mappings.
7. Within real repositories, prefer Lite members. Within each tier rank by SHA-256 of the UTF-8 string `20260917|task-order|<source_key>|<instance_id>`.
8. Fill the fixed per-repository quotas. Force the two startup IDs into training. Assign validation using the separately seeded hash implemented in the script, not experimental labels.
9. Assign all unused eligible unique groups to a split-owned reserve queue before observing model outcomes. A replacement must match source, canonical repository and split; at most three preflight candidates per slot (primary plus two backups).
10. Interleave repositories deterministically for collection order, with the two smoke IDs first. Freeze each stage before its experimental runs.

If supply is insufficient, the tool exits with `BLOCKED_SHORTFALL` and writes the exact repository/required/available counts. Continue diagnostics and engineering, but do not alter the source list, relax a filter, duplicate variants or substitute a validation task without an explicit versioned plan revision.

Reference patch statistics and hidden test metadata used for orchestration must never appear as router features or be supplied as hints to Qwen/DeepSeek. Task selection based on mutation scope narrows the population; disclose that design in the data card.

## 6. Exact acquisition and manifest commands

Keep this packet and all source/grading data **outside the experimental `/testbed`**, even if located within the engineering project. Run from this packet's directory:

```bash
python -m pip install -r requirements-task-manifest.txt
python -m unittest discover -s tests -v
python -m pip freeze > task_manifest_environment.txt

python harvest_task_manifest.py fetch \
  --cache .task-source-cache \
  --out task_catalogue_v1

python harvest_task_manifest.py select \
  --catalogue task_catalogue_v1/catalogue.jsonl \
  --exclude-ids prior_exposed_instance_ids.txt \
  --out task_plan_v1
```

Before `select`, Sonnet must create `prior_exposed_instance_ids.txt` by auditing historical task manifests/traces. One source `instance_id` or `source_key:instance_id` per line; `#` comments are allowed. Create an empty file only if the inventory really found no mappings, and explicitly record unresolved historical IDs. Do not mark the final reserve 'untouched' until that audit is complete.

The script implements `fetch`, `select` and `finalise`; these are real included commands. It does **not** implement Claude Code execution, Docker preflight, checkpointing, grading or the collection runner. Those remain engineering work assigned by the master prompt. The `fetch` command only accesses the public task data; it makes no paid model call.

Its outputs include:

```text
task_catalogue_v1/catalogue.jsonl
task_catalogue_v1/source_downloads.json

task_plan_v1/train.provisional.jsonl
task_plan_v1/train.ids.txt
task_plan_v1/validation.provisional.jsonl
task_plan_v1/validation.ids.txt
task_plan_v1/reserves.jsonl
task_plan_v1/smoke.jsonl
task_plan_v1/branch_task_slots.jsonl
task_plan_v1/final_reserved.ids.jsonl
task_plan_v1/excluded.jsonl
task_plan_v1/manifest_report.json
```

**The `.ids.txt` files are the complete literal task lists Sonnet/Luna must harvest.** No experimental calls start merely because they exist; runtime preflight must pass for the stage. Preserve hashes and the source lock alongside them. The metadata catalogue is not an agent-facing prompt table and is not a terminal-outcome dataset.

## 7. Preflight and bounded replacements

Use approved Linux x86_64 execution infrastructure and the exact upstream environment/evaluator. A Mac can orchestrate it, but do not silently change architecture or rebuild incompatible environments and report them as equivalent.

Official adapter sources:

- SWE-Gym environment/grader: `https://github.com/SWE-Gym/SWE-Bench-Fork`
- SWE-Gym task/images reference: `https://github.com/SWE-Gym/SWE-Gym`
- SWE-smith profiles/registry/evaluator: `https://github.com/SWE-bench/SWE-smith`
- SWE-smith harness semantics: `https://swesmith.com/guides/harnesses/`

Pin the installed code commits and record resolved image digests. Resolve images with the official mappings/registry; do not construct a guessed tag from the task ID. For Gym, initialise the prescribed base checkout. For Smith, initialise the prescribed **buggy state**: the task patch creates the bug, not its gold repair. Follow the source profile to resolve working directory, environment activation and test command; do not unconditionally run generic `python -m pytest` in an unactivated base interpreter.

Check each selected task without invoking an experimental model:

- correct buggy setup and runtime dependencies;
- source-prescribed base/gold or clean/bug validation;
- protected grading material not visible to the agent;
- runtime-network-independent CPU execution;
- warm source-prescribed grading completes within 120 seconds for this fast-subset protocol.

Measure cold startup separately. Do not shorten/remove tests until a task crosses the 120-second threshold. Preserve slow tasks as declared exclusions. Failed assertions in a properly running baseline test are different from missing modules or image failures.

The pipeline must write an explicitly versioned preflight ledger. Example below is **synthetic schema illustration**, not an actual preflight certificate:

```json
{"task_uid":"gym:getmoto__moto-5752","status":"pass","selection_used_model_outcome":false,"warm_grade_seconds":24.5,"certificate_ref":"protected-preflight/task-5752/report.json","image_digest":"sha256:<actual-image-digest>"}
```

Allowed durable rejection reasons are exactly:

```text
task_sanity_failed
runtime_over_limit
unsupported_architecture
network_dependency
unavailable_image
duplicate_exposure
unsupported_grader
invalid_source_metadata
```

Transient network errors, download interruptions and ongoing preflight are `pending`; they do not trigger convenient substitutions. Model failures, judge disagreement and unexpected edge/cloud preference are **never preflight rejection reasons**. The script checks declared fields; W6 must independently inspect the actual certificates. A made-up JSON pass is not evidence.

Freeze progressively:

```bash
python harvest_task_manifest.py finalise \
  --plan task_plan_v1 --preflight preflight.v1.jsonl \
  --stage smoke --out frozen_smoke_v1

# After preflighting the next stage, preserve earlier frozen assignments:
python harvest_task_manifest.py finalise \
  --plan task_plan_v1 --preflight preflight.v2.jsonl \
  --previous-freeze frozen_smoke_v1 \
  --stage train50 --out frozen_train50_v1
```

Continue with `train200`, `train500`, then `all600`, each using the previous frozen stage and a ledger that preserves all earlier records. Output directories must be new/empty. A source lock or previous task/certificate change blocks freezing. Use content-addressed evidence and do not edit accepted certificates after seeing experimental outcomes.

`finalise` writes `pending_preflight.jsonl` and `accepted.preview.jsonl` when blocked. Preflight only the pending replacement IDs it selects, not arbitrary reserves. When all stage slots pass, it writes `train.jsonl`, `validation.jsonl` and `FROZEN.json`. The collector executes only rows in the appropriate frozen stage and tracks finished logical task/policy jobs across cumulative stages—do not rerun the first 50 tasks at each expansion.

## 8. Which tasks produce A, B and C

**A and preference table:** every frozen training/validation task gets one edge-only and one cloud-only original Claude Code attempt. Preserve all pre-call states; sample up to five main-agent call positions per original trajectory for the core preference pairs. Up to 1,200 original trajectories and 6,000 core preference pairs. Short/invalid trajectories produce fewer, never duplicated rows.

**B:** `branch_task_slots.jsonl` designates 80 training task slots (48 real, 32 synthetic) and 20 validation slots (12 real, 8 synthetic), selected deterministically before outcomes. For each slot, retain one random eligible checkpoint from its edge-only original and one from its cloud-only original. This gives up to 200 checkpoint pairs / 400 primary continuations. Join by `slot_id` to the frozen runtime-approved task: a preflight replacement changes the task UID, not the slot assignment. The master prompt governs verified restoration and the edge-only continuation after the experimentally varied next call. Repeat a preselected random 10% of checkpoint pairs once, keeping split ownership.

**C:** log every actual experimental invocation, including retries, shadow generation, branches, judges and profiler calls with distinct purpose tags. Start controlled profiling from ten saved development requests, not final-reserve requests, across warm/cold and low/high load: 40 foreground invocations plus separately counted priming/background activity.

No separate downloaded dataset is needed for B or C. They are measurements from these selected Claude Code tasks. Task sources provide environments; the experimental model pair produces the traces and outcomes.

## 9. Final reservation

The selector reserves 100 Verified task IDs, with upstream repositories not present in this campaign's development allowlist, after the known-exposure exclusion. It interleaves repositories deterministically. This is a proposed evaluation subset, not the complete official benchmark.

Do not create original trajectories, candidate pairs, judge results or serving profiles from these tasks now. The ID-only output omits issue text and patch/test contents, but full source cache still contains protected evaluation data: keep it out of experimental and modelling workspaces. Residual historical or foundation-model exposure is not ruled out by an ID split.

## 10. Evidence and limits

Preparation status: public source pages/revisions and two smoke IDs verified; selector compiled and tested using **synthetic metadata fixtures**; no full task parquet downloads, Docker task preflights or experimental trajectories were run in the preparation container. On the user's machine, Sonnet must replace these pending checks with actual file/row counts, literal manifests, source hashes, resolved image digests and runtime certificates.

This closed plan is the recommended initial scope, not a mathematical ranking of all possible task datasets. If its exact quotas are infeasible after the stated checks, report the concrete shortfall. A dataset plan that refuses to invent missing tasks is preferable to a falsely 'complete' 600-row manifest.
