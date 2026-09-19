# Shortfall acceptance v1 — 2026-09-18

Shortages hash: fa31090d0a855acd16fcf0cd8a386b79a79c41f9880ab9b620adf6fe7705ec64

## Documented shortfall

```json
[
 {
  "source": "smith",
  "repo": "mewwts/addict",
  "required": 10,
  "eligible_unique_groups": 4
 }
]
```

`mewwts/addict` requires 10 synthetic training tasks per `TASK_HARVEST_SPEC.md`
section 3, but only 4 unique eligible groups survive the campaign's filters
(nonempty issue >=40 chars, nonempty FAIL_TO_PASS, image reference, nonempty
PASS_TO_PASS, mutation confined to 1-3 `.py` files, test/build-file mutation
excluded, alias/duplicate-group dedup) against the real, fetched
`SWE-bench/SWE-smith-py` parquet (pinned revision
77cab9055d42ab4a5c25c89a8f937096db13558e). This is a genuine supply limit in
the actual dataset, not a filter chosen to make collection easier, and not
something `harvest_task_manifest.py` is permitted to paper over by inventing
IDs, relaxing filters, or borrowing quota from another repository.

## Decision

Accepted as-is: this campaign's synthetic-training total is **494 tasks
(300 real + 194 synthetic), not 500 (300 + 200)**, until/unless a future,
separately-versioned plan revision finds additional genuinely eligible
`mewwts/addict` tasks (e.g. from a later SWE-smith-py release) or explicitly
reallocates the 6-task gap to a different repository via an explicit quota
change to `TASK_HARVEST_SPEC.md`/`TASK_SOURCE_LOCK.json` (not done here).

This acceptance does not change any other repository's quota, does not
touch validation quotas (unaffected — mewwts/addict has validation_target=0
in this campaign), and does not affect the final-evaluation reserve. It
exists only so that `harvest_task_manifest.py finalise` can freeze stages
(starting with `smoke`, which does not use `mewwts/addict` at all) while
this real, disclosed shortfall remains open, rather than blocking all
progress on an unrelated stage until a structurally impossible 500/500 is
reached.

Authorized by: Sonnet (engineering leader for this campaign), per
`SONNET_MASTER_PROMPT.md` section 3 framing ("These are task/job ceilings
and engineering targets, NOT promises of statistical sufficiency") and
`TASK_HARVEST_SPEC.md` section 5 ("If supply is insufficient... report the
concrete shortfall... A dataset plan that refuses to invent missing tasks
is preferable to a falsely 'complete' 600-row manifest").
