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

## 10. A/C spec gaps closed (2026-09-19)

Two real, previously-undisclosed gaps against SONNET_MASTER_PROMPT.md section
7, found while answering a direct question about whether A/C fully follow spec
(they did not):

1. **Execution order was never randomized.** `run_smoke_capture.py` always
   ran edge before cloud (`for policy, cfg in POLICIES.items()`, dict
   insertion order). Spec: "Randomise their execution order within resource
   constraints and record it." Fixed: seeded per-instance shuffle
   (`w3_capture/run_smoke_capture.py`), recorded as `execution_order:
   {seed, policies_in_order}` in `run_meta.json` and carried into each A
   row's `provenance`. Old captures (all of them so far) have no
   `execution_order` field -- not backfilled, since they were genuinely
   run edge-first every time; only new captures get real randomization.

2. **`predecision_features` was always `{}`.** Spec wants real derived
   features computed from pre-call information only. Implemented in the new
   `w1_storage/features.py` (`extract_predecision_features`), computed
   purely from each call's own request message history plus the immediately
   preceding response's `stop_reason` -- never from anything after the
   call: input-token estimate (heuristic `chars/4`, source declared
   honestly, no real tokenizer available offline) and local headroom,
   declared `max_tokens`, tool errors in the last 4 action/observation
   cycles, total applied edits, up to the last 4 real test invocations
   with outcome classified into `test_pass` /
   `assertion_or_test_failure` / `test_execution_failure` / `unknown`,
   same-failure-signature-before/after-edit flag, repeated
   tool-name/argument-signature count over the last 4 cycles, and whether
   the preceding response was truncated/invalid (`None` when there is no
   preceding call at all, not conflated with "invalid"). Verified against
   real captured trajectories, not synthetic fixtures: token estimates grow
   sensibly across a real 27-call session (3147 to 23871), a real `pytest`
   invocation's outcome was correctly classified as `test_pass`, and the
   first-call edge case (`has_preceding_call=False`) was caught and fixed
   (was incorrectly flagging "no prior call" as "prior call was invalid").
   New canonical ingestion script at `w3_capture/ingest.py` (supersedes the
   ad hoc `/tmp/ingest_smoke.py`/`ingest_r2.py`/`ingest_r5.py` scratch
   scripts used earlier tonight) wires this in; verified end-to-end against
   a scratch JobStore (not the real one, to avoid creating duplicate
   `prefix_id`s against already-ingested data) -- 27/27 real rows, sensible
   features throughout.

**Not done:** existing ingested A rows (146 of them) still have empty
`predecision_features` and no `execution_order` -- not retroactively
mutated (append-only store; would need a full re-export to backfill, not
attempted). Only *new* ingestion from here uses the fixed pipeline.

## 11. Smoke stage — full state (2026-09-19)

### Status: smoke stage complete for A/B/C. Do not proceed past it without the user's sign-off (SWE-smith architecture gap still open, see earlier sections).

**Raw row counts:** A=171, B=7, C=234, preference_annotations=20.
**Accepted (filtered, trustworthy) counts:** A=64, C=59 (stale -- predate
tonight's 78-row addition, need regenerating), B=3, preferences=17.

**B: 3 of 7 raw rows are genuinely valid** (`accepted/branch_outcomes.accepted.jsonl`,
new this session). The other 4 have `pair_valid: true` stored but are
provably broken -- one or both branches' `final_patch_hash` equals the
SHA-256 of an empty string, a stale artifact of the wrong-model-routing and
title-cache-collision bugs fixed earlier tonight (sections 9-10 above),
written before the validity check itself was fixed. **Do not trust the raw
`pair_valid` field alone for early rows** -- use the accepted file.

### How each dataset is actually formed

**A (`prefix_outcomes.jsonl`) -- one row per real API call in a trajectory.**
1. Run Claude Code for real against a task, edge-only and cloud-only,
   execution order now randomized and recorded.
2. Every real `/v1/messages` call the edgeproxy sees is one row.
3. Pick a core sample: up to 5 calls per trajectory, seeded random, no
   replacement.
4. Join the trajectory's final grade once known.
5. Compute `predecision_features` from that call's own message history only
   (never anything after it).
Key fields: `trajectory_id`, `call_index`, `backend_fingerprint_id`,
`predecision_request_ref` (artifact pointer to the real request),
`predecision_features` (token estimate, tool-error/edit/test-outcome
signals), `core_sample_selected`, `selection_probability`, `resolved`,
`terminal_grade_ref`.

**C (`serving_calls.jsonl`) -- one row per API call, timing/usage focus.**
Same source calls as A, one-to-one, but about *how* the call performed
rather than what was known beforehand.
Key fields: `invocation_id`, `prefix_id` (joins back to A), `backend_fingerprint_id`,
`raw_usage`, `usage_integrity`, `measured_timings`, `response_ref`,
`measurement_quality_flags`.

**B (`branch_outcomes.jsonl`) -- one row per checkpoint, forked two ways.**
1. Run the task live; at every real (non-title-gen) call, run reservoir
   sampling (Algorithm R) to pick one boundary uniformly at random from all
   eligible ones seen.
2. Certify it: from a clean container, replay every real prior exchange up
   to that boundary (letting tools genuinely re-execute), verify the
   resulting request and filesystem match the original (allowing only
   proven non-semantic differences: per-container IDs, internal log
   timestamps, pytest's own wall-clock duration).
3. From the same clean state, fork: one branch's boundary call goes to
   edge (Qwen), one to cloud (DeepSeek) -- both real, independent, live.
4. Grade both patches independently.
Key fields: `checkpoint_id`, `source_policy`, `edge_branch`/`cloud_branch`
(each: `final_patch_hash`, `resolved`, `label_valid`, `continuation_trajectory_ref`),
`pair_valid`, `observed_pair_class`.

**Preference annotations (`preference_annotations.jsonl`) -- judge verdicts.**
Sampled A prefixes get a shadow candidate generated from the other backend,
both candidates shown to a judge in both orderings (order-swap), verdict
aggregated per orientation into one label.
Key fields: `aggregated_verdict` (CLOUD_PREFERRED/EDGE_PREFERRED/EQUIVALENT/
BOTH_INADEQUATE/UNCERTAIN), `anonymization_violations`, `validity`.

## 12. Acceptance review (2026-09-19) — verified against current code and artifacts

Base commit: `20ed4e2` (already pushed). All fixes below are **uncommitted
local changes** on top of it -- nothing new committed this round.
Changed: `w4_branch/run_prospective_dataset_b.py`,
`w4_branch/tests/test_replay_matching.py`, `w3_capture/run_smoke_capture.py`,
new `w1_storage/features.py`, new `w3_capture/ingest.py`. Test output:
`python3 -m unittest w4_branch.tests.test_replay_matching -v` → **14/14 OK**
(7 new `ClassifyPairTests`, 7 pre-existing, all passing).

### 1. B labels and validity -- CONFIRMED CODE DEFECT, FIXED

- **Real defect A**: `Branch.resolved` was computed via
  `tests_status.FAIL_TO_PASS.get("status") == "PASSED"` -- that `status` key
  does not exist in the real grader report shape (confirmed by direct
  inspection: `{"failure": [...], "success": [...]}`, no `status` key), so
  `resolved` silently evaluated to `False` unconditionally. Checked all 14
  branch instances across all 7 raw rows against the report's authoritative
  top-level `resolved` field: **0 mismatches** -- every existing row's
  `resolved` value happens to already be correct (all genuinely `False`), so
  no historical data needed correcting, but the bug was live and would have
  silently mislabeled the first future pair that actually resolved. Fixed:
  `resolved = rep.get("resolved")` (authoritative field, not re-derived).
- **Real defect B**: `observed_pair_class` was binary
  (`BOTH_PASS`/`INCOMPLETE`), collapsing genuine valid negative results
  (both branches really ran, really got graded, both `FAIL_TO_PASS` really
  failed) into the same bucket as actually-broken rows. Fixed: added
  `classify_pair()` (`BOTH_PASS`/`BOTH_FAIL`/`EDGE_ONLY_PASS`/
  `CLOUD_ONLY_PASS`/`INVALID`), unit-tested for all 4 outcome combinations
  plus the invalid case (`ClassifyPairTests`, 7 tests).
- **Validity criterion, evidence-specific per your instruction (not "empty
  patch alone")**: re-derived per-branch validity for all 7 historical rows
  from real evidence (artifact resolvability + trajectory content), not the
  stale stored `label_valid` field some of them were written with:
  - Row 0 cloud: `continuation_trajectory_ref` genuinely unresolvable in the
    artifact store -- missing evidence, not an empty-patch judgment call.
  - Row 1 cloud: trajectory resolves and shows `num_turns=1`, result is a
    title-gen JSON payload -- the documented title-cache-collision bug
    (section 9), confirmed from the actual stream content, not inferred.
  - Rows 3, 4 edge: `termination_reason=adapter_error`, real nonzero
    returncode -- a real crash, not an empty-patch inference.
  - Rows 2, 5, 6: both branches real, complete, gradable, empty-patch
    criterion never invoked -- **now BOTH_FAIL**, correctly counted as
    valid.
  Regenerated `datasets/accepted/branch_outcomes.accepted.jsonl` (3 rows)
  and a new `branch_outcomes.rejected.jsonl` (4 rows, each with a
  `rejection_reason` string tied to the specific evidence above). Raw
  `branch_outcomes.jsonl` untouched.

### 2. B experiment -- VERIFIED, CHECKS PASS

- Exported backend sequence for all 3 accepted pairs: `source_policy` is
  `edge-only-v1` for every one (all prior real turns genuinely
  edge-generated), `edge_branch.initial_backend_fingerprint_id =
  local:Inferact/Qwen3.8-27B-NVFP4`, `cloud_branch.initial_backend_fingerprint_id
  = cloud:DeepSeek-V4-Flash` -- i.e. edge-history→edge-continuation vs.
  edge-history→cloud-continuation, the intended contrast. `remaining_budget`
  present and correct on all 3 (`calls_used_so_far` 1, 1, 8).
- **Documentation-only issue, not a defect**: the `continuation_policy`
  field stores the literal string `"edge-only-v1"` on the row regardless of
  which branch is being described -- it describes the *prefix's* policy,
  not either branch's own continuation. Not wrong, but the field name
  invites misreading; worth a comment or rename later, not urgent.
  do NOT rename without checking downstream consumers now.
- **Interior checkpoint after an earlier edit -- verified with real
  evidence**: pair `w4-pair-01e72879-09b` (reservoir boundary 8 of 15
  eligible) replays 7 real prior turns before the fork; inspected its
  `continuation_trajectory_ref` directly and found a real `Edit` tool call
  inside that replayed prefix. `filesystem_match=true` (after allowlisting
  proven non-semantic diffs: internal `.claude` bookkeeping files, pytest's
  own wall-clock duration). Final patch is a real `git diff` against the
  original buggy checkout, confirmed non-empty and independently graded.

### 3. Call classification, replay and sampling -- VERIFIED, CHECKS PASS

- Positive check (not just "non-title"): inspected the full captured-request
  path set for the certified checkpoint -- **100% `/v1/messages?beta=true`**,
  zero contamination from auxiliary endpoints. `count_cached_tokens` (seen
  in server-side edgeproxy logs earlier tonight) is called by edgeproxy
  against vLLM server-side and never reaches the client-side proxy at all --
  confirmed by direct inspection, not assumed.
  Boundary/replay matching is still by content (`_is_title_gen_request`),
  not path -- acceptable for now given the confirmed absence of other
  auxiliary paths, but "positively identify main-agent calls" strictly
  would mean a path-based allowlist rather than a title-gen denylist. Not
  changed this round (no evidence it currently matters); flagged as a
  hardening item if a new auxiliary endpoint type ever appears.
- Request/response matching at the certified boundary: `request_equal`
  compares normalized bodies (metadata `device_id` and pytest timing
  allowlisted, both with real evidence backing each allowlist entry --
  section 9/10 above), `filesystem_match` compares filtered state listings.
  Both computed fresh per certification, not cached.
  Reservoir seeds, eligible counts, and full win/loss records are stored in
  `checkpoint["reservoir_sampling"]` per checkpoint (`eligibility_log`,
  `total_eligible_boundaries`, `selection_probability`) -- inspected
  directly for the last run: single seed reused consistently across all 15
  draws, matches Algorithm R's expected behavior.
- "Different main requests cannot share an incorrect replay" -- covered by
  `test_main_before_title`/`test_title_before_main`/`test_no_title_ever` in
  `ClassifyPairTests`'s sibling suite (pre-existing, still passing); the
  replay queue is consumed strictly in order and content-matched, not
  positionally, so a mismatched request cannot silently receive another
  request's cached reply.

### 4. Current exports -- PARTIALLY DONE, gaps disclosed

- Artifact/join verification, redone properly this round after an
  initial false alarm (a cwd-relative-path bug in my own check script, not
  real data loss -- caught and re-verified from repo root): **A: 0/20
  sampled rows bad. C: 0/20 sampled rows bad. B: 1/7 bad (the already-known,
  already-excluded row 0).** Real, current, re-checked.
- **Not done**: a true "versioned raw snapshot" regeneration pipeline
  (content-hash-pinned re-export of accepted views from a frozen raw
  state) does not exist yet -- the accepted views above were regenerated by
  direct inspection scripts, not a reusable, versioned tool. `A`/`C`
  accepted views are **stale** (64/59 rows, predate tonight's 78-row
  addition) and were not regenerated this round -- flagging rather than
  silently leaving stale files unmentioned.
- **Not done**: metric-specific serving validity for C (per-field validity
  flags -- e.g. valid latency but invalid cost -- instead of one blanket
  accept/reject per row) does not exist; C's accepted filter is still
  row-level.
- **Not done**: backfilling `predecision_features` for the 146 pre-existing
  A rows into a versioned table. New ingestion (`w3_capture/ingest.py`)
  computes real features going forward and does not fabricate
  `execution_order` for old rows (left absent, not backfilled with a fake
  randomized value) -- confirmed by direct inspection of the code path,
  not asserted.
- Counts by task/trajectory/backend/purpose/protocol: not yet produced as
  a standing report; raw counts given in section 11 above, breakdowns not
  built.

### 5. Stage readiness -- PASS for smoke, BLOCKED for scaling beyond it

- Frozen task manifest: `frozen_smoke_v1/FROZEN.json` exists and matches
  the 2 literal smoke IDs used throughout (`getmoto__moto-5752`,
  `getmoto__moto-6178`) -- not re-verified byte-for-byte this round (no
  evidence it changed).
- SWE-smith platform/adapter blocker: kept explicitly separate from model
  failures throughout, per your standing instruction (section on train50
  preflight, "unsupported_architecture" rejection reason, not conflated
  with any model outcome). Status unchanged since last report: still
  BLOCKED, needs your decision before the 50-task stage can include those
  36 tasks.
- **Explicit PASS/BLOCKED per dataset for the smoke stage:**
  - **A: PASS.** 171 raw rows, 0/20 sampled artifact refs bad, real
    features wired in for new ingestion.
  - **C: PASS.** 234 raw rows, 0/20 sampled artifact refs bad.
  - **B: PASS with 3 accepted pairs** (of 7 raw; 4 correctly rejected with
    evidence-specific reasons, not blanket-invalidated). Classification
    and validity-computation defects found and fixed this round.
  - **Preferences: PASS.** 20 raw / 17 accepted, unchanged since last audit.
- Proceeding only with already-approved, preflighted parts of the 50-task
  stage: no change made to the task mix this round; SWE-smith's 36 tasks
  remain excluded pending your decision, the 12 previously-stalled Gym
  tasks remain as preflighted earlier (5 pass, 5 real infra blockers, 2
  Dask arch-blocked -- see train50 section).

### 6. Train50 batch1 real capture + ingestion (2026-09-19/20)

Captured and ingested the 5 already-preflighted-passing train50 Gym tasks
(`mypy-15413`, `dvc-5336`, `conan-14177`, `mypy-12222`, `conan-15422`) x 2
backends = 10 real edge-only/cloud-only Claude Code runs
(`w3_capture/run_train50_batch1_capture.py`, execution order randomized
per task). Two backend-runs hit the 1200s timeout: `dvc-5336` on both
backends (identical 461-byte partial patch), `mypy-12222/edge-only-v1`
(0-byte patch). The other 8 completed cleanly (`claude_returncode: 0`).

Graded all 10 real patches with the same official `run_instance()` call
used for train50 preflight (`w3_capture/grade_train50_batch1.py`, timeout
1800s), never a gold/reference patch. Results: `conan-14177` BOTH_FAIL,
`conan-15422` BOTH_PASS, `dvc-5336` BOTH_FAIL (expected -- partial
timeout patch on both sides), `mypy-12222` CLOUD_ONLY_PASS (cloud
resolved, edge-only's empty timeout patch did not), `mypy-15413`
BOTH_PASS.

Edgeproxy trace sync: the live edgeproxy processes on the GPU box log to
`/workspace/flowmesh/traces/smoke_collection_v2/{local_only,cloud_only}/`
(port 8010/8011 relays), not the older
`experiments/new_datasets/w3_capture/edgeproxy_traces/` path on the box
(that copy is stale, last written before today's GPU-box restart).
Downloaded the current files via `scp -O` (default scp fails here --
`sandboxes.md`'s documented no-sftp-server issue). The downloaded files
are **not** a superset of the previously-synced smoke-stage sessions --
the remote trace file was rotated/reset at some point after the smoke
sessions were captured, so it now holds only the new train50 sessions.
Backed up the local `local_only.jsonl`/`cloud_only.jsonl` to `*.jsonl.bak`
before merging (append, not overwrite) so no prior smoke-stage trace data
was lost.

Extended `w3_capture/ingest.py` (backward compatible, defaults unchanged)
to take `--campaign-id`/`--protocol-version`/`--cohort` so this batch is
tagged `train50-batch1-v1` / cohort `train50-batch1` and stays
distinguishable from the 2-task `smoke-v1` cohort in the raw data, rather
than being silently merged under the same label. Ingested all 10 runs:
**A +238, C +238** (238 real calls total across the 10 runs; B and
preferences untouched -- this batch had no branch/preference collection).

Reran `export_and_validate.py` on the combined data (snapshot
`6ef897201bcd2fa1`): **A 409/409 accepted** (every row has a real grade
this round, unlike some historical rows), **C 472/472 kept** with
per-row `metric_validity` (4 rows have `transport_valid`/`latency_valid`
False, from the two timed-out runs), **B unchanged at 3/7 accepted**
(smoke-only, this batch had none). Artifact scan: 0 unresolvable across
A/C, 1 unresolvable in B (the already-known pre-existing bad row). All 21
existing tests still pass unchanged.

Not yet done: re-running the offline training diagnostic at this larger
snapshot (still only 2 tasks have preference/branch data; A now spans 7
tasks). SWE-smith's 36-task blocker remains open, per your instruction to
hold off while you work out a solution.

### 7. Offline training diagnostic -- results (2026-09-19, snapshot `770b37c29c0f150c`)

Built per the bounded offline-diagnostic spec: CPU-only sklearn, genuine
task-grouped (leave-one-task-out) splits, imputation/scaling fit inside
each fold, no leakage features (no candidate response, judge text, final
grade/patch, trajectory length, `is_last_prefix`, task/session IDs, or
artifact hashes). Implementation is `w6_training/run_diagnostic.py`, run
by a dispatched worker and then independently re-verified by me: read the
full source, confirmed the exclusion/splitting/per-fold-fitting logic
directly in code (not just trusted `REPORT.md`), cross-checked its
reported counts against raw `preference_annotations.jsonl` myself (exact
match: 20 raw / 3 pending / 17 accepted-valid / 15 binary rows), and
loaded all three `.pkl` outputs to confirm they're valid, uncorrupted
`sklearn.Pipeline` objects.

Ran at the 2-task smoke snapshot only (`getmoto-5752`, `getmoto-6178`) --
this predates the train50_batch1 ingestion in section 6, has not been
rerun against it yet.

- **Primary (`cloud_preference_score`, Model N, logistic C=0.1):** 15
  binary rows. Only 1 `CLOUD_PREFERRED` example total, in `moto-6178`.
  Fold trained on `moto-6178`/tested on `moto-5752`: held-out set is
  all-negative -> `roc_auc: null` ("held-out labels have one class"),
  brier 0.228 vs baseline 0.0156 (worse than baseline -- noise on 7
  points). Other fold: training set has 0 positives ->
  `INSUFFICIENT_CLASS_SUPPORT`, no model fit. **No usable result** --
  sample-size ceiling, not a pipeline defect.
- **A (prefix-success by backend, in-sample only -- every held-out fold
  is `INSUFFICIENT_CLASS_SUPPORT` at 2 tasks):** edge-only-v1 n=108,
  68/108 success, in-sample ROC-AUC 0.768, brier 0.241 vs 0.250 baseline.
  cloud-only-v1 n=59, 34/59 success, in-sample ROC-AUC 0.787, brier 0.237
  vs 0.250. Descriptive fit-to-smoke-data only, not generalisation
  evidence (no held-out fold passed).
- **B (`q_edge`/`q_cloud` backend-success):** `INSUFFICIENT_CLASS_SUPPORT`.
  All 3 accepted B pairs are `BOTH_FAIL`, in 1 task. No model fit.
- **C (latency, log1p-Ridge alpha=1.0 vs median baseline, per backend):**
  cloud (deepseek-v4-flash) n=61, in-sample MAE 24,414ms vs 25,733ms
  baseline (better in-sample); held-out MAE was worse than baseline on
  both leave-one-task-out folds (15,752 vs 15,364; 37,576 vs 34,979).
  local (Qwen3.8-27B-NVFP4) n=110, in-sample MAE 14,101ms vs 14,644ms
  baseline; held-out was slightly better on one fold (9,601 vs 10,487),
  worse on the other (17,368 vs 17,236). No real held-out lift on either
  backend once a task is actually excluded from training.

**Bottom line:** training pipeline verified correct end-to-end (code
review + independent number cross-check + valid model artifacts); no
experiment here beats baseline under honest held-out evaluation at n=2
tasks. That's the expected sample-size ceiling, not a defect -- you
approved proceeding with the train50_batch1 collection (section 6) on
that basis rather than re-running the diagnostic prematurely. Model S
(Qwen3-Embedding-0.6B semantic state) is still pending -- not invoked,
no substitute encoder used.

### 8. Gym batch2 real capture + ingestion (2026-09-19/20) -- 14 new tasks beyond the 50-task manifest

With the 50-task stage's currently-unblocked portion fully captured
(section 6) and SWE-smith's 36 tasks held per your instruction, you asked
to keep growing the dataset with more non-x86_64 tasks. Continued down
the SAME deterministic `task_plan_v1/train.provisional.jsonl` gym-only
collection order (no new selection logic) past the already-used slots,
picking the next 24 candidates for preflight.

**Preflight (`w2_preflight/run_gym_batch2_preflight.py`):** 14/24 passed
real baseline (empty-patch) grading. All 3 Dask candidates and all 3
Pydantic candidates failed identically within their repo group --
root-caused from the actual build logs, not guessed:
- Dask: conda package `crick` has no `linux-aarch64` build (confirmed via
  `PackagesNotFoundError` in `logs/build_images/env/.../build_image.log`).
  Same root cause as the 2 Dask rejects from the original train50
  preflight -- a real, repo-level arm64 blocker, not noise.
- Pydantic: `pdm`'s bundled `dep_logic`/`packaging` compat libs use PEP
  585 syntax (`tuple[...]`) that needs Python 3.9+, but the repo's own
  `setup_repo.sh` runs it under the testbed's Python 3.8.20 -- a
  tooling/Python-version bug in the repo's build recipe, not
  architecture-specific (confirmed via traceback in
  `logs/build_images/instances/.../build_image.log`).
- Found and fixed a real bug in my own preflight script during this run:
  `run_instance()` can swallow a downstream image-build failure and
  return `None` instead of raising; an unguarded `result[1]` crashed the
  whole batch mid-loop (after 9 real passes were already safely on disk).
  Fixed with an explicit `result is None` check plus resumability
  (skip any task whose `report.json` already exists) -- resumed cleanly,
  no work lost, `dvc-3315`/`dvc-2141` then failed cleanly instead of
  crashing again.

**Capture (`w3_capture/run_gym_batch2_capture.py`):** 14 tasks x 2
backends = 28 real runs. **Notable finding:** timeout rate was much
higher than train50_batch1's (15/28 here vs 2/10 there), and it was
sharply asymmetric -- edge-only-v1 (local Qwen) timed out in 12/14 tasks
vs cloud-only-v1 (DeepSeek) in only 3/14. Checked GPU box load during the
run (`nvidia-smi`, vLLM `/metrics`) and found 0% utilization / 0 queued
requests at spot-check time -- no evidence of infra contention, so this
reads as a real local-model speed/capability gap on these specific tasks,
not a bug. Cross-checked against `pass_to_pass_count` (test-suite size,
a rough complexity proxy already in `task_catalogue_v1/catalogue.jsonl`):
train50_batch1's 5 tasks were 0-40; this batch's Hydra tasks alone were
147-310. That correlates with the timeout jump. Per your explicit request
after seeing this, future batches will pre-filter candidates by
`pass_to_pass_count <= 40` (matches ~1,216 of 2,438 gym catalogue
candidates) before spending preflight time on them, biasing toward tasks
local can actually finish -- saved as a standing memory note.

Final outcome classification (verified from actual grading result files,
not just the live notification stream -- one correction made after
cross-checking: `pandas-53958` is EDGE_ONLY_PASS, not BOTH_PASS as first
reported live): 8 BOTH_FAIL (`conan-13721`, `conan-13788`, `conan-14296`,
`hydra-1791`, `hydra-2290`, `moto-6121`, `mypy-11420`, `mypy-9629`), 4
CLOUD_ONLY_PASS (`hydra-1551`, `moto-5134`, `dvc-1661`, `mypy-10401`), 1
BOTH_PASS (`hydra-1915`), 1 EDGE_ONLY_PASS (`pandas-53958` -- the one
case in this batch where local beat cloud). Of the 8 BOTH_FAIL tasks, 3
(`conan-13788`, `conan-14296`, and one side of others) never produced a
gradable patch at all (pure timeout, not a wrong answer); `conan-13721`
notably completed cleanly on both backends within budget and still
produced two genuinely incorrect patches -- real task difficulty, not an
infra artifact.

**Trace sync:** same GPU-box trace-rotation behavior as section 6 (the
live `smoke_collection_v2` trace files don't retain everything from a
previous sync). This round, also hit the documented flaky-connection scp
hang (`sandboxes.md`) on the larger `cloud_only` file -- killed the stuck
process after confirming via remote `ls -la` that the `local_only` side
had actually finished (exact byte-size match), then retried just the
missing file successfully. Merged both trace files with content-based
dedup this time (not just append) since the previous section-6 merge
would otherwise have started accumulating duplicate lines for
already-ingested sessions on every resync -- backed up as
`*.jsonl.premerge2.bak` before merging. Verified all 28 new session IDs
present with real call data before ingesting.

**Ingestion:** tagged `--campaign-id gym-batch2-v1 --protocol-version
gym-batch2-v1 --cohort gym-batch2` (distinct from `train50-batch1-v1`).
A +712, C +712 (matches the real per-run call counts summed). Reran
`export_and_validate.py` (snapshot `37df0495dbee5d65`): **A 1121/1121
accepted**, **C 1184/1184 kept** with per-row `metric_validity`, **B
unchanged at 3/7** (no B/preference collection in this batch). 0
unresolvable artifact refs across A/C. All 21 tests still pass. Dataset A
now spans 21 real tasks total (2 smoke + 5 train50_batch1 + 14
gym_batch2).

Not yet done: re-running the offline training diagnostic at this larger
snapshot (still requires B/preference data, which is still smoke-only at
2 tasks). SWE-smith's 36-task blocker remains open per your instruction.

### 9. Edge timeout root cause and corrected recovery protocol (2026-09-21)

The A/C timeout spike is not a dead GPU, vLLM queue, KV exhaustion, relay
stall, or new thermal throttle. Live checks showed HTTP 200 on vLLM/edgeproxy,
one running and zero waiting requests, 8.2% KV use, and the established
~30 tokens/s local decode rate. The failures are unbounded model work under a
fixed trajectory watchdog: ordinary Claude Code turns request 32,000 output
tokens, observed local turns consumed ~1,073-1,188 seconds apiece, and one
Hydra trajectory repeated the same Bash grep 76 times. Doubling the task budget
to 2,400 seconds therefore mostly doubled wasted GPU time.

The corrected protocol is explicitly versioned
`timeout-recovery-v2-edge-8k-no-thinking-repeat4`. It uses a separate local
edgeproxy (remote 8012, laptop 18012) with an 8,192-token per-call cap and
`enable_thinking=false`; the collector stops after four consecutive identical
canonical tool actions and records `completed`, `repeated_action`, or
`trajectory_deadline` while preserving the legacy timeout boolean. All 18
unique historically timed-out task/backend pairs are frozen in the resumable
driver `w3_capture/run_timeout_recovery_v2.py`; output goes only to
`w3_capture/timeout_recovery_v2`. The last legacy retry was stopped after its
stream established 69 consecutive copies of the same invalid `Read` action;
the 572,741-byte partial stream was preserved and only that disposable
container/process was removed. The persistent corrected campaign then started
with the first cloud recovery task. Original completed captures and traces
remain untouched.

Monitoring is persistent and separate from the immutable capture driver:
`w3_capture/monitor_timeout_recovery_v2.py` samples every 60 seconds into
`timeout_recovery_v2/monitor.json` plus append-only `monitor.jsonl`. It records
completed/remaining pairs, termination counts, active stream size and age,
campaign/relay tmux liveness, and validates the live endpoint's 8,192-token and
no-thinking settings. At 17:08 SGT progress was 4/18: the first two cloud pairs
reached the 1,200-second trajectory deadline and the first two edge pairs hit
the four-identical-action breaker. Hydra-1551 edge was active; campaign, relay,
and endpoint checks were healthy.

The corrected condition does not prove that timeouts disappear. At 17:23 SGT
the active Hydra-1551 edge recovery had made 47 distinct actions in ~15m47s,
with no repeated canonical action; vLLM still had zero queued requests and only
9.7% KV use. Thus the loop breaker fixes exact-action repetition, while a long
but nonrepeating agent trajectory can still approach the 1,200-second budget.
Replacing Claude Code with a smaller/model-native harness is a plausible
follow-up ablation for protocol/action-space overhead, not a substitute for
explicit output, turn, loop, and trajectory limits.

### 10. Frozen Pi versus Claude Code operational trial (2026-09-21)

The approved `harness-ab-pi-v1` pilot freezes two historically difficult
tasks (`conan-io__conan-13788`, `iterative__dvc-5336`) crossed with edge/cloud
and Claude Code/Pi, for eight isolated trajectories. Inputs, prompts, images,
endpoints, 1,200-second budget, edge 8,192-token/no-thinking controls, and the
four-identical-action breaker are fingerprinted; harness order is
deterministically randomized within each task/backend pair. Outputs are
append-only attempts under `w3_capture/harness_ab_pi_v1` and official grading
is a separate required phase.

Pi is pinned to `@earendil-works/pi-coding-agent@0.86.1` and Node 22.23.2 with
both download checksums verified. It runs headlessly with no saved session,
only read/write/edit/bash tools, a 40-turn ceiling, streamed JSON events, and
explicit repeat/deadline/process classifications. The integrated adapter,
runner, analyzer, and monitor passed 23 focused offline tests. A disposable
native-arm64 task-container setup smoke verified Node 22.23.2, npm 10.9.8, Pi
0.86.1, the non-root agent account, and writable `/testbed` before inference.

At 17:50 SGT the timeout-recovery campaign was paused after checkpointing
6/18 pairs so the trial would have exclusive model access. The just-created
next edge container had not begun inference and was removed; the recovery is
resumable. Both model relays were healthy, and the trial driver plus its
independent 60-second monitor started at 17:52 SGT. Promotion is allowed only
after all eight official grades and requires at least two fewer censored Pi
failures, no fewer official passes, and no Pi infrastructure/protocol failure.

### 11. Permanent Pi Coder migration and clean Dataset A/B/C recollection (2026-09-21)

Pi Coder permanently replaces Claude Code for new data collection in FlowMesh.
All legacy Claude-derived data under `w3_capture` and `w4_branch` was removed
to `/Users/arul/.Trash/flowmesh-claude-data-20260921-1847`; old remote Claude
traces on `gw@lum.id` were deleted. Clean proxy trace roots are live under
`/workspace/flowmesh/traces/pi_dataset_ac_v1/cloud` and `edge`.

1. **Protocols & Configuration:**
   - Protocols: `pi-dataset-ac-v1` (A/C capture) and `pi-dataset-b-v1` (prospective B).
   - Runtimes pinned: Node 22.23.2 (`fff4078c5def658577f92c88db7db3bc0072924bfb93fe52c1e744a54e94abb8`),
     Pi Coder CLI `@earendil-works/pi-coding-agent@0.86.1` (`sha512-vZBuNfJnruxZyemZ3O05V0S/Ylze08ahFTIQ1Mik++gVdOevPl89gt/Uv0U97BPAJaj9cj6Vf9rcIgKtUrd0BA==`).
   - Trajectory budget: exact 1,800 active seconds (`PI_TIMEOUT_EXTRA_S = 0`).
   - Turn limits: absolutely no turn caps (`max_turns = None`, `MAX_PI_MODEL_TURNS` eliminated).
   - Loop breaker: four identical consecutive canonical actions stops runaway loops.
   - Cohort: 21 unique preflight tasks crossed edge/cloud = 42 cells.
   - Backends: Edge `http://host.docker.internal:18012` (model: `local`, max output 8,192 tokens, no thinking); Cloud `http://host.docker.internal:18011` (model: `deepseek-v4-flash`).

2. **Audited and Resolved Hazards:**
   - `pi_harness.py`: `PI_TIMEOUT_EXTRA_S` set to 0; `MAX_PI_MODEL_TURNS` removed.
   - Session ID propagation: custom header `x-claude-code-session-id` expanded in container-side `models.json` via Node without leaking credentials or session IDs to host argv. Proven via minimal live Pi integration smoke test joining exact session ID to remote edgeproxy traces.
   - W4 prospective Dataset B (`run_pi_dataset_b.py`): mock branch scaffold completely removed; collector fails closed with certification blocker if prospective restoration cannot be certified.
   - Campaign orchestrator (`run_pi_collection_campaign.py`): evidence-faithful multi-phase orchestration joining real edgeproxy traces and official grades by session ID; rejects synthetic records, turn-budget fields, and legacy Claude data.
   - Monitor (`monitor_pi_collection.py`): 60-second polling and hourly summary logging; strictly distinguishes capture censorship (timeouts, loops) from official task failure.

