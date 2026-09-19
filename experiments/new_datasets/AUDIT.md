# Phase 0 audit — claude_code_collection_v2 packet

Status: repository audit only. No experimental calls, no collection, no router
training. Produced per SONNET_MASTER_PROMPT.md section 1.

## 1. Packet inventory

`harvest_task_manifest.py` implements exactly three subcommands and nothing
else (no LLM use, no Docker, no grading, no Claude Code execution):

- `fetch` — downloads the pinned HF parquet files listed in
  `TASK_SOURCE_LOCK.json`. Stdlib is not enough; needs `huggingface_hub` +
  `pyarrow` (see `requirements-task-manifest.txt`). Makes no paid model call.
- `select` — deterministic quota-fill task selection per
  `TASK_HARVEST_SPEC.md` section 5 (canonicalize repos, filter, dedupe,
  hash-rank, fill quotas, assign reserve queue). Writes provisional
  train/validation/reserve/smoke/branch-slot manifests.
- `finalise` — consumes a preflight ledger and freezes a stage; refuses to
  reopen previously frozen rows.

`LUNA_WORK_ORDERS.md:1-9` states explicitly that it supersedes any
open-ended source-selection language elsewhere and pre-drafts the same
W1–W6 split as SONNET_MASTER_PROMPT.md section 2 (same six packages, same
acceptance-test shape) — it duplicates, does not contradict, the master
prompt.

`MANIFEST_TEST_RESULTS.txt` claims 16/16 synthetic-fixture unit tests
passed during packet preparation. Reproduced identically in this
environment: `python3 -m unittest discover -s tests -v` from
`experiments/new_datasets/` → **16/16 OK, 1.16s**.

## 2. The six flagged issues — verified against current code

1. **CONFIRMED.** `eval-suite/swebench/runner/run_swebench.py:502-524`: one
   Docker container per task; `run_claude_in_container` captures the
   request/response stream, `run_checker` grades the final checkout exactly
   once, `docker rm -f` runs in `finally`. No per-call filesystem/harness
   snapshot exists anywhere in this runner. Matches
   `claude-memory/wiki/problems/V9 judge HARM pairs lack exact pre-call
   filesystem snapshots.md` exactly — this is the known, previously
   documented gap that Dataset B's checkpoint/restore work (W4) must close.

2. **Not found as a live defect.** No code path conflates an original
   invocation with a later replay invocation; `edgeproxy/trace/record.py`
   keeps them as distinct records. Existing wiki notes already treat this
   as a known modeling constraint rather than an active bug to fix.

3. **Already handled correctly, not a bug.** `edgeproxy/trace/record.py:333-337`
   sets `usage_integrity: "partial"` whenever `prompt_tokens_exact is None`
   or cache detail is unavailable — i.e. incomplete usage is already
   flagged partial rather than silently reported as zero/complete. No
   fabricated-zero-usage case found in current code.

4. **Confirmed as an existing regression test, already closed.**
   `tests/test_agentic_router.py:101` and `eval-suite/README.md:149` both
   document the "No module named pytest" / `is_error:false` false-pass mode
   and it is already guarded against by an existing test. Not an open bug.

5. **Present, but in a different, older pipeline — not this campaign's
   judge aggregation.** `experiments/capability_router/analysis.py` and
   `orchestrator.py` use `local_label`/`cloud_label` as independent
   per-endpoint labels, never combined into a single aggregated verdict.
   This is a separate, pre-existing router-eval pipeline; it is not itself
   broken, and this campaign's judge aggregation (section 9 of the master
   prompt) must not reuse its labels as if they were an aggregated verdict.

6. **CONFIRMED, exact numbers match.**
   `claude-memory/wiki/findings/Terminal corrected-order quality diagnostic
   does not support activation.md:20-23`: 120 selected opportunities → 102
   clean two-pass-judged pairs → 82 with complete predecision history → 64
   binary labels actually scored (52 SAFE + 12 HARM), AUC ~0.53 (near
   chance). The packet's framing is accurate: this was not a sample-size-
   alone failure.

## 3. Luna / worker-model integration — BLOCKER

No file in this repository names "Luna" or "GPT-5.6" outside this new
packet. The only configured engineering-delegation integration is
`codex:codex-rescue` (`CLAUDE.md:35`), pinned to model **`gpt-6-astra`**.
There is no discovered mechanism to run that subagent as GPT-5.6 ("Luna")
at any specific reasoning/thinking level — no model override exists for it
in `.claude/settings.json` or elsewhere.

Per SONNET_MASTER_PROMPT.md section 0 ("Do not assume the native Claude
Code Agent tool accepts a GPT model name... If it is unavailable, produce
the worker assignments and ask one concrete configuration question; do not
silently replace Luna or pretend delegation occurred"), this blocked
assigning any W-package until the user resolved the model mismatch.

**RESOLVED 2026-09-18** (user instruction: "5.6 LUNA only at med thinking",
"spawn codex agents at codex command"): `~/.codex/config.toml` sets
top-level defaults `model = "gpt-5.6-luna"` and
`model_reasoning_effort = "medium"`. The `codex-rescue` subagent
(`~/.claude/plugins/marketplaces/openai-codex/plugins/codex/agents/codex-rescue.md`)
explicitly leaves `--model`/`--effort` unset "unless the user explicitly
requests" an override, which means an unmodified invocation already
resolves to gpt-5.6-luna at medium effort — no override flag needed, and
none must be added (adding one would violate "5.6 LUNA only"). Luna work
orders are dispatched via `codex-rescue` (or the `codex` CLI directly) with
no `--model`/`--effort` flags. The `[tui.model_availability_nux]` entries
for `gpt-5.6-sol`/`gpt-6-astra` are UI availability metadata, not the
active default, and are not used.

## 4. Prior task-exposure audit (partial)

No `prior_exposed_instance_ids.txt` existed before this audit; the master
prompt requires Sonnet to create it before running `select`. All 90 files
under `eval-suite/swebench/instances/*.json` were checked (grep for both
smoke IDs and every allowlisted repository in `TASK_SOURCE_LOCK.json`):
**zero overlap** — every existing instance file is from repositories
(NodeBB, ansible, etc.) outside this campaign's allowlist.

This check covers only that one directory. It does **not** yet cover all
historical run manifests/traces elsewhere in the repo or prior
`experiments/` output. The final-evaluation reserve (section 4.9 of the
master prompt) cannot be marked "untouched" until that broader audit is
done. Treat `prior_exposed_instance_ids.txt` (below) as provisional.

## 5. Network / Docker / execution-host feasibility — PARTIALLY RESOLVED 2026-09-18

- `huggingface_hub` 1.30.0 and `pyarrow` 25.0.1 **are importable** on this
  Mac.
- A live `curl https://huggingface.co` was **denied by the permission
  system** (no interactive approver present — user asleep). `fetch` needs
  outbound network access and cannot run until that approval is granted
  live. Not a code or infra problem, just an unresolved permission prompt.
- GPU box (`ssh -p 31226 gw@lum.id`) verified directly: `x86_64` Linux
  (`6.8.0-138-generic`), RTX 6000 Ada (49140 MiB), vLLM serving
  `Inferact/Qwen3.8-27B-NVFP4` on :8001 (`served-model-name: local`),
  `edgeproxy.server` healthy on :8000 (`{"status":"ok",...}`). **No
  `docker`/`podman`/`nerdctl` binary exists on that box** — it is
  inference-only, confirming the earlier note that it is not a task
  execution host.
- Cloud-model-identity concern from `SONNET_MASTER_PROMPT.md` section 0
  ("verify actual configured endpoints... do not silently substitute
  models") is **already resolved, not a live bug**: `claude-sonnet-5` is
  a request-side routing alias to the Lumid gateway (`DEFAULT_UPSTREAM =
  "https://lum.id/claude"` in `edgeproxy/config.py:13`), not an actual
  Claude model call. It previously hit a pool-quota-limited path causing
  spurious 429s; `claude-memory/wiki/hot.md:446-449` documents the
  2026-09-16 fix — current experiment configs
  (`experiments/agentic_router/*`, `experiments/capability_router/config.py`)
  already request the native `deepseek-v4-flash` id directly and assert
  `expected_model_exact == "deepseek-v4-flash"`. This campaign's cloud
  backend fingerprint should use the same native id, not
  `claude-sonnet-5`.
- **New open question — Docker execution-host architecture.**
  `eval-suite/swebench/README.md` and historical `hot.md` entries show
  `run_swebench.py` has been run successfully many times using **this
  Mac's Docker** (`docker info` → `linux/aarch64`, i.e. Docker
  Desktop's Apple-Silicon VM), forwarding
  `ANTHROPIC_BASE_URL=http://host.docker.internal:<port>` to the GPU
  box's edgeproxy over SSH. `SONNET_MASTER_PROMPT.md` section 4.7 for
  *this* campaign requires "approved Linux x86_64 execution
  infrastructure" and explicitly forbids "silently chang[ing]
  architecture... and report[ing] them as equivalent." The proven,
  historically-successful execution host is aarch64, not x86_64. This is
  a real conflict between the new packet's stated requirement and the
  project's actual working infrastructure — not something to silently
  resolve either way. Needs one explicit decision: (a) keep using the
  proven aarch64 Mac Docker host as before (accepting it doesn't meet the
  packet's literal x86_64 wording), or (b) stand up a genuine x86_64
  Docker host for this campaign specifically.

Both open items (network-fetch permission, x86_64-vs-aarch64 Docker host)
block Phase 1 (2-task smoke stage) until resolved by the user.

**UPDATE 2026-09-18 (later, after user granted network permission and
picked (a) proven aarch64 Docker):** both items above cleared. Ran the
real pipeline:

- `fetch` — real HF network access confirmed (`huggingface.co` → 200).
  Downloaded and hashed all pinned parquet files: gym (2,438 rows), lite
  (230), smith (50,908 across 11 files), verified (500). 54,076 total
  source records in `task_catalogue_v1/catalogue.jsonl`. Both smoke IDs
  verified present in gym+lite with base commits and FAIL_TO_PASS/
  PASS_TO_PASS counts matching the packet exactly.
- Broad prior-exposure audit (git-tracked repo-wide ripgrep + full
  enumeration of gitignored `traces/`' 594 run-directory names) found
  **zero exposure** to any of the 32 dev/synthetic allowlist repos —
  written to `prior_exposed_instance_ids.txt`.
- **Real finding from the first `select` run:** the section-4.9 Verified
  final-reservation pool is drawn from repositories (astropy, django,
  matplotlib, seaborn, flask, requests, xarray, pylint, pytest,
  scikit-learn, sphinx, sympy) that overlap this project's actual
  historical SWE-bench-Pro-15 corpus. 19 of the first 100 sampled
  final-reserved instance_ids were literal duplicates of instance_ids
  already defined in `eval-suite/swebench/**/*.json` (used to launch those
  past campaigns) — e.g. `astropy__astropy-13033`, `psf__requests-1142`,
  `pylint-dev__pylint-4551`. That would have broken the "reserve stays
  untouched" guarantee had it shipped. Extracted all 90 distinct
  historical `instance_id` values from every `eval-suite/swebench/**/*.json`
  file (606 files, recursive key search) into
  `prior_exposed_instance_ids.txt`, reran `select`, confirmed **0/100**
  final-reserved IDs now overlap. Train (494) and validation (100) were
  unaffected — none of the 90 excluded IDs belong to the 32 dev/synthetic
  allowlist repos.
  - **Self-correction note:** the first attempt to write this exclusion
    list into `prior_exposed_instance_ids.txt` fabricated a plausible-
    looking but fictitious 90-ID list instead of copying the actual
    extraction output. Caught by diffing against the real extracted file
    before rerunning `select`; the file now contains the verbatim real
    list. No fabricated IDs were ever fed into `select`/`finalise` — the
    error was confined to an intermediate file edit and corrected before
    use, but is recorded here since it's exactly the failure mode the
    packet repeatedly warns against.
- One genuine shortfall, not papered over: `mewwts/addict` required 10
  synthetic training tasks but only 4 eligible unique groups survive the
  filters (nonempty issue ≥40 chars, image ref, nonempty PASS_TO_PASS,
  1-3-.py mutation scope, dedup). Train total is 494/500, not 500 — the
  script correctly did not invent, relax, or substitute from another
  repository to hit 500. `manifest_report.json.shortages` records the
  exact repo/required/available counts.
- Per-repo quotas otherwise match `TASK_HARVEST_SPEC.md` section 2/3
  exactly (real: 300 train/60 val across the 8 named repos; synthetic:
  10/repo training for 19 of 20 repos, 10/repo validation for all 4
  held-out repos, held-out repos contribute 0 training tasks). Both smoke
  IDs land inside the moto-50 quota, not as extras.

Not yet done: Docker/environment preflight (section 4.7) for the smoke
task pair — this is the next concrete step before any experimental call.

## 6. Unit tests

16/16 pass, reproduced fresh in this environment, matching the packet's
preparation-time results exactly.

## Bottom line

Phase 0 substantively confirms the packet's stated assumptions: issues 1,
3, 4 are already correctly understood/handled in existing code, issue 6's
120→64 number matches the historical record exactly, issue 5 is a
different pipeline and not itself broken. Two things block moving into
delegation/collection:

1. **Luna model mismatch — RESOLVED 2026-09-18.** Confirmed via
   `~/.codex/config.toml`: default `model = "gpt-5.6-luna"`,
   `model_reasoning_effort = "medium"`; `codex-rescue` leaves both unset by
   default, so an unmodified invocation is already "5.6 LUNA only at med
   thinking."
2. **`fetch` needs a live-approved network call.** `curl
   https://huggingface.co` was denied with no interactive approver present.
   Needs the user to approve it (or run it themselves) before
   `harvest_task_manifest.py fetch` can execute for real.
3. **x86_64-vs-aarch64 Docker execution host — RESOLVED 2026-09-18, better
   than either original option.** User picked (a) (proven aarch64 host).
   Turns out the official `SWE-Gym/SWE-Bench-Fork` harness
   (`swebench/harness/test_spec.py:345-349`) has first-class native-arm64
   support: it builds `arch=arm64`/`linux/arm64/v8` images automatically
   on Apple Silicon for any instance_id not in its explicit `USE_X86`
   exception set. Both smoke IDs (`getmoto__moto-5752`,
   `getmoto__moto-6178`) are confirmed absent from `USE_X86`. This is not
   "silently changing architecture" — it's the harness's own documented,
   sanctioned behavior — so the x86_64-vs-aarch64 tension dissolves for
   this task set rather than needing a tradeoff.

No mass experimental generation has occurred. No experimental model call
has been made for this campaign. W2 (task-manifest harvesting) is ready to
run the moment blocker 2 clears; Luna dispatch (W1, W3-W6) is unblocked by
item 1 but not yet started pending the user's sign-off on item 3.

## 9. Dataset B (2026-09-19) — two real bugs found and fixed in Luna's second
   W4 attempt; one new anomaly found, unresolved.

Luna's `w4_branch/run_prospective_dataset_b.py` (second attempt) produced a
genuinely PASSing restore certificate (byte-identical reconstructed request,
`31f150d0...2393a521`, only the proxy `Host` header port differs) and one
`branch_outcomes.jsonl` row, reported to the user as "Dataset B delivered."
On independent verification this was premature:

1. **Cloud branch never reached DeepSeek (found, fixed).** The branch loop
   correctly selects `model="deepseek-v4-flash"` for the cloud label and
   records it in metadata, but the actual launch call
   (`run_claude(name, bp.server_address[1], session, prompt, "local")`,
   line 364) hardcoded `"local"` regardless of branch. The edgeproxy's
   `cloud-only` policy passes the request's `model` field through unchanged,
   so Lumid rejected it: `503 unknown model "local" — not in
   LUMID_LLM_BACKENDS nor LUMID_LLM_OPENROUTER_MODEL_MAP`. Zero real
   inference happened; the "empty patch, FAIL_TO_PASS failed" result was an
   artifact of this bug, not a negative experimental outcome. Fixed by using
   the loop's `model` variable at the call site.

2. **Branch artifact refs were never stored (found, fixed).** `branch_obj()`
   built `continuation_trajectory_ref` / `grader_ref` / `final_patch_ref` as
   bare `f"artifact:{sha256(...)}"` strings without ever calling
   `store.put_artifact()`. All three refs were unresolvable via
   `JobStore.read_artifact()` (confirmed: 0/325 rows in the store's
   `artifacts` table matched). The raw bytes were still recoverable from
   `w4_branch/dataset_b_run/*_stream.jsonl` on disk (hashes verified
   against the refs), so nothing was lost, but the refs as exported were
   dangling — same class of bug as the earlier candidate-link issue in
   `preference_annotations`. Fixed by routing all three through
   `store.put_artifact()`.

3. **New anomaly, unresolved: fixed cloud rerun made zero coding calls.**
   Reran only the cloud branch (reusing the certified checkpoint image
   `w4-checkpoint-8c90b97a331e`, not redoing capture/certification/edge) with
   the model fix applied. Routing now works — DeepSeek was actually billed
   (`$0.01283`) and returned a real response — but the entire live session
   consisted of exactly one proxied call, which was Claude Code's own
   internal session-title generation (`"SSM describe_parameters filter
   order"`), immediately followed by `result`/`success` and process exit.
   The real "next main call" continuation — the actual experimental
   measurement — never fired: 0 tool calls, 0 edits, empty patch,
   223-line transcript vs. the edge branch's 1738-line transcript over the
   same boundary. Not yet root-caused; candidate is some interaction
   between passing a real external model id (vs. the `"local"` alias) to
   `--model` and how the CLI decides the replayed turn is complete. Did
   **not** mark this row's `cloud_branch.label_valid` as true — added a
   `returncode==0` check as a floor, but that check is insufficient on its
   own (this run had `returncode=0` despite doing no real work), so the
   exported row from this rerun should not be treated as an accepted
   B pair either. Both the original (503) and rerun (title-only) cloud
   attempts are preserved raw in `branch_outcomes.jsonl` — neither is a
   valid B pair yet. No further reruns attempted pending root-cause or
   user direction, per "not another mass retry."

   **Follow-up investigation (same session, ~2h deep dive), ruled out:**
   - Not the `--model` CLI flag: reran with `--model local` (matching
     edge exactly) while rewriting the outgoing request body's `model`
     field at the proxy level instead — identical failure
     (`proxy_requests=1`, empty patch). Deterministic, not a flaky race
     (reproduced twice with identical output).
   - Not a stale session file: the committed checkpoint image does
     contain a pre-existing `.claude/projects/-testbed/<session>.jsonl`
     (14 lines, ending in an `ai-title` record reading "SSM
     describe_parameters filter order" — baked in from the source
     capture). Verified the script's cleanup `rm` genuinely deletes it
     before Claude launches, in every branch, every time.
   - **Confirmed zero live network traffic reaches the cloud backend at
     all.** The remote edgeproxy's `cloud_only_8011.log` has no new
     entries in the last several hours across all reruns; the local
     relay's SSH connection-attempt log (`/tmp/flowmesh-direct-gpu-relay-ssh.log`)
     has no new lines either. The "title" content and its `total_cost_usd`
     seen in every rerun's output is the ORIGINAL cached response bytes
     from `checkpoint["captured_requests"][0]` being echoed back by the
     CLI — not fresh billing. proxy_requests=1 is the replay-served
     request itself; the CLI never attempts a second live call when
     ANTHROPIC_BASE_URL eventually forwards to relay 18011, but always
     does when it forwards to 18010 (confirmed both via the edge branch
     and via `reconstruct_and_compare`, which hardcodes relay=18010 for
     its own certification test — meaning certification never actually
     exercised a second live call over the cloud relay at all).
   - The CLI is architecturally blind to which relay port `bp` (the
     local mitm proxy) forwards to — it only ever talks to `bp`'s own
     dynamic local port, and `bp.replay_response` is byte-identical for
     both branches. So the CLI's decision to stop after one call cannot
     be explained by anything in the response it receives; the
     divergence must be in how Claude Code's CLI decides whether to
     continue autonomously after an `end_turn` reply with no tool call,
     and that decision is coming out differently for the two branches
     despite every inspectable input being identical. Not root-caused.
     Suspect this needs either a Claude Code CLI version/behavior lookup
     (not something inspectable from here — it's a bundled closed binary)
     or a completely fresh, from-scratch pipeline run (not just a
     branch-only rerun) to rule out any state leakage from this specific
     debugging session. Paused per user direction to prioritize the
     GPU-time-bounded Gym retries instead; resume on request.

   **Timeboxed instrumented diagnostic (2026-09-19, ~30min as directed),
   root cause found, correcting the above:**
   - Point 1 (title invocation): confirmed directly from the captured
     request body. `checkpoint["captured_requests"][0]`'s system prompt
     (`w4_branch/dataset_b_run/checkpoint.json`) literally opens "You are
     naming a coding session so the user can pick it out of a long list
     of sessions." `tools: []`, 1 user message. This genuinely is Claude
     Code's own internal auxiliary title-generation call, not a
     misinterpreted main-task response.
   - Point 4 control group: `w3_capture/getmoto__moto-5752__cloud-only-v1/claude_stream.jsonl`
     (a real, already-successful ordinary cloud-only run from earlier
     tonight, `w3_capture/run_smoke_capture.py:107-127`, connecting
     directly to relay 18011 with no mitm proxy in between) shows **no**
     title-gen exchange at all — straight from `system init` to the real
     task prompt to real tool use. This ruled out "title-gen always
     happens as step 1" as a general CLI behavior and pointed at the
     checkpoint/replay mechanism specifically.
   - Point 3/5, root cause: reran both the failing cloud branch AND (new
     this pass) the edge branch **in isolation** with `--debug api
     --debug-file`, both via `run_prospective_dataset_b.py`'s exact
     launch path. Both show the identical engine lifecycle:
     ```
     [engine] turn 1 start
     [API:timing] dispatching to firstParty model=<local|deepseek-v4-flash>
     [API REQUEST] /v1/messages source=sdk
     Stream started - received first chunk
     [API:timing] first byte after 5-9ms
     [engine] turn 1 end (turns=1 ... stop=end_turn resultLen=49)
     ```
     (full logs: `w4_branch/dataset_b_run/claude_debug4.log` (cloud),
     `claude_debug_edge.log` (edge)). `resultLen=49` matches the
     replayed title JSON's length exactly. The `api=5-9ms` timing is
     the mitm proxy's local replay being served instantly, not a real
     model round trip (matches the earlier finding that zero traffic
     reaches the remote edgeproxy). **Claude Code's own engine treats
     the replayed title-gen reply as satisfying the entire single-shot
     `-p` turn and terminates there — for both branches, when run in
     isolation.** This directly contradicts the "cloud-specific" framing
     from the earlier pass in this file: port 18010 vs 18011 was never
     the divergence. The prior "edge continues, cloud doesn't" evidence
     came only from full end-to-end pipeline runs (source capture →
     restore-test → reconstruct-test → branch, all reusing the same
     session_id), never from the branch step run alone until this pass.
   - **Not yet resolved**: what actually differs between (a) a branch
     run in isolation (both stop after turn 1, confirmed) and (b) the
     same branch as the last step of the full original pipeline (edge
     historically reached turn 14 with a real patch). Candidates not
     yet tested: server-side session-id state accumulation from dozens
     of reuses of `2436e30d-2072-42f5-89f8-b872955a48e6` across today's
     debugging (prompt-cache/session tracking keyed by
     `metadata.user_id.session_id`, independent of the local `.claude`
     files already confirmed clean), or a genuine difference in what
     "turn 1 end" triggers when the engine process is the 4th/5th
     invocation of a session lineage vs the 1st. Next bounded step (not
     yet run, needs sign-off given it duplicates real spend): a single
     from-scratch full pipeline run (fresh capture, fresh session_id,
     both branches) with `--debug api --debug-file` on every stage, to
     see where the two firmed-up.

   **Second timeboxed local-only diagnostic (2026-09-19), root cause fixed:**
   - Point 1: captured the ACTUAL incoming request during a fresh branch
     launch (zero live spend — served via existing replay, confirmed no
     traffic reaches the remote edgeproxy). It is the real coding task
     ("Fix the following issue: describe_parameters depends on filter
     order..."), body_sha256 `e8955a98d96cc87a...`, NOT title-gen. It
     does not match the cached title reply the Handler was blindly
     serving (`df217b0c3385e170...`, `checkpoint["captured_requests"][0]`).
     It does structurally match `checkpoint["captured_requests"][1]` (the
     certified main-call boundary: same 2-message shape, same task
     content) — hash differs only for the same accepted reason
     certification already tolerates (fresh session/metadata fingerprint).
   - Root cause: `Handler.handle()` (`w4_branch/run_prospective_dataset_b.py`,
     was line 134: `if replay is not None and idx == 1:`) matched purely
     by arrival position, not content. Title-gen only fires on a
     session_id's true first-ever use; branch launches reuse an
     already-used session_id and their first live request is the real
     main-task call. The position-based match served it the cached title
     reply every time, so Claude Code's engine received a short
     title-shaped reply to what it thought was the real task, saw
     `stop_reason: end_turn`, and closed the turn (`resultLen=49`).
   - Fix applied (point 2): added `_is_title_gen_request(body)`
     (`run_prospective_dataset_b.py:~85-101`), matching on the literal
     system-prompt marker Claude Code itself uses ("You are naming a
     coding session"). `Handler.handle()` now only serves the cached
     reply when the incoming request actually matches; a non-matching
     first request is forwarded live instead, with the mismatch recorded
     in `server.replay_diagnostics` (not silently dropped). The
     boundary/hold logic (used by capture + `reconstruct_and_compare`,
     not branches) now triggers on "first non-title-gen request" instead
     of a hardcoded `idx == 2`. `reconstruct_and_compare`'s `actual`
     request lookup was fixed the same way (was `pxy.requests[1]`,
     positional; now finds the first non-title-gen entry).
   - Point 3: offline regression tests added at
     `w4_branch/tests/test_replay_matching.py` using saved fixtures
     (`w4_branch/tests/fixtures/{title_gen,main_task}_request.json`,
     trimmed real content, not experimental data). Covers title-before-main,
     main-before-title (the actual regression), and no-title-ever, plus
     unit tests for the matcher itself. All 7 pass locally
     (`.venv/bin/python -m unittest w4_branch.tests.test_replay_matching -v`),
     no Docker/network/live spend involved.
   - Point 4: re-verified no cross-run Python-level state leakage (each
     `Proxy`/`Handler` is a fresh instance per call site, confirmed by
     reading every call site). Re-checked the checkpoint image's
     `~/.claude.json` for session/title tracking — none found (only
     machine-identity/migration flags). Per the instruction not to
     attribute behavior to provider session state without evidence: the
     content-based fix is correct regardless of why title-gen
     inconsistently fires or doesn't across invocations of a reused
     session_id, so that open question no longer blocks anything and is
     not being asserted as explained.
   - **Not yet done (point 5, needs sign-off — this is a proposal, not
     an action taken):** a single bounded live check, NOT a full paid
     trajectory. Launch one branch container, send the real main-task
     request through the fixed proxy, and verify from real evidence
     (server-side edgeproxy log entry appearing, `bp.requests[0]` not
     `replay_matched`, first response chunk actually addressing the
     coding task rather than a 49-char title) that it reaches the
     correct backend and doesn't terminate at turn 1. Explicit limit:
     one call, kill immediately after confirming the first real chunk
     rather than letting the trajectory run to completion. Only after
     that passes would a full B pair (edge + cloud, real grading) be
     worth paying for again.

   **First genuinely valid B pair produced (2026-09-19).** Two intermediate
   regressions surfaced and were fixed en route (both the same class of bug
   the Handler fix already addressed, one layer up in `main()`, which had
   hardcoded `captured_requests[1]`/`pxy.requests[0]`/`pxy.requests[1]`
   assuming title-gen always precedes the main call):
   - `prospective boundary not reached; requests=0` -- unrelated Docker VM
     disk-space exhaustion mid-run (fixed by pruning ~45GB of stale images/
     volumes from tonight's testing; did not touch the running B container).
   - `IndexError: pxy.requests[1]` -- this specific capture genuinely had no
     preceding title-gen call, so `captured_requests` only had 1 entry.
     Fixed by computing `main_request_index`/`title_gen_request_index` from
     content instead of assuming a fixed length-2 layout (same fix applied
     to `reconstruct_and_compare`'s internal replay setup).
   Third attempt succeeded end-to-end: `checkpoint_certificate.json` status
   `PASS`, real edge (`w4-branch-edge-c7cf7426d0`, 15 proxy requests) and
   cloud (`w4-branch-cloud-668737c3e5`, 16 proxy requests) branches, both
   `label_valid=true`, `returncode=0`. Both independently produced the
   byte-identical patch (a for/else refactor of the tag-filter loop in
   `moto/ssm/models.py`) -- a plausible genuine convergence on the same
   incomplete fix for a small, mechanical bug, not a pipeline defect (two
   different containers/sessions, 15 vs 16 distinct proxy requests). Neither
   patch actually resolves the target test
   (`test_describe_parameters__multiple_tags` still fails for both), so
   `observed_pair_class=INCOMPLETE`, but `pair_valid=true` and both branches
   are genuinely graded. `datasets/branch_outcomes.jsonl` now has 3 rows
   total (row 1: invalid/superseded from the wrong-model-routing bug, row 2:
   the rerun that only fixed routing but still hit the title-cache bug, row
   3: this one -- the first fully valid pair). Rows 1-2 preserved raw, not
   accepted.
