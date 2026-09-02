# SWE-bench Pro tier

Real repos, real GitHub issues, real professionally-verified tests — the
strongest possible hard tier, replacing the earlier synthetic `megaapp/`
generator attempt (abandoned once this was proposed; see git history / prior
session notes for why: real repos + pre-verified tests eliminate the two
riskiest parts of a hand-built fixture, correctness of the scenario and
correctness of the checker).

Source: [ScaleAI/SWE-bench_Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)
(731 instances, 11 repos, 4 languages, MIT-licensed). Scoped to the 266
Python instances across `qutebrowser/qutebrowser`, `ansible/ansible`, and
`internetarchive/openlibrary` — the other 8 repos are Go/JS/TS and would need
new, unvalidated checker logic per language.

## Why this needs a different runner

Every other task in `eval-suite/` copies a small synthetic fixture into a
plain host directory and runs `claude -p` against it with `cwd=sandbox_dir`.
SWE-bench Pro instances ship as a **prebuilt Docker image**
(`jefzda/sweap-images:<dockerhub_tag>`) with the real repo and its real
dependencies already installed - there is no "just copy some files" version
of this that preserves fidelity (the repo's actual test suite needs the
repo's actual installed dependencies, which live in the image, not in any
directory that could be `docker cp`'d out).

So `swebench/runner/run_swebench.py` is a separate runner from
`eval-suite/runner/run_eval.py`, reusing only what's generic between them
(imported directly, not duplicated): `RunContext`, `build_proxy_cmd`,
`start_proxy`/`stop_proxy`, `free_port`, `wait_for_health`,
`load_dotenv_into_environ`, `CONDITION_POLICY`, and `binomial_rate_ci`.
edgeproxy itself is completely unchanged; what's different is entirely
*where the agent's session runs*.

Per job:
1. `docker run -d --entrypoint bash <image> -c "sleep infinity"` - a fresh
   container per job (never reused across seeds/conditions).
2. Run the instance's own `before_repo_set_cmd` (from the dataset) to check
   out the correct pre-fix source + post-fix test file combination.
3. Bootstrap `curl` if missing, create a non-root `agent` user (`claude
   --dangerously-skip-permissions` refuses to run as root), install the
   `claude` binary.
4. `docker exec -i -u agent` running `claude -p` with
   `ANTHROPIC_BASE_URL=http://host.docker.internal:<port>` pointed at a
   per-job edgeproxy instance on the host - same isolated-proxy-per-job
   pattern as the rest of the suite.
5. Grade by running the exact `fail_to_pass` + `pass_to_pass` pytest node
   IDs from the dataset (not a full suite run) inside the container, parsed
   from JUnit XML, both counts must be 100%.
6. `docker rm -f` the container.

## Quirks discovered and handled generically (not per-instance hacks)

- **Image layout varies.** Some instances set `ENTRYPOINT=/bin/bash`
  (qutebrowser, openlibrary); others set `CMD=["bash"]` with no entrypoint
  (ansible). Always passing `--entrypoint bash` normalizes both.
- **Root can't use `--dangerously-skip-permissions`.** Every container's
  default user is root. A non-root `agent` user is created per container.
- **`host.docker.internal` needs `--add-host=host.docker.internal:host-gateway`**
  to resolve on Linux (it works out of the box on Docker Desktop). Passed
  unconditionally so the same code works on this laptop and the GPU box.
- **qutebrowser's `pytest.ini` hardcodes the pre-7.0 `--strict` flag**, which
  pytest 7.4.2 (installed in that image) escalates to a fatal error during
  argument parsing, before any test runs - every pytest invocation crashed,
  not just some. Fixed with `--override-ini='addopts=...'`, applied only
  where `instances/*.json` sets `pytest_addopts_override` (qutebrowser only;
  ansible's pytest 6.2.4 and openlibrary's pytest 8.3.5 have no such flag in
  their own configs and need no override).
- **qutebrowser's GUI tests need `QT_QPA_PLATFORM=offscreen`** in a headless
  container - the agent discovered this itself during the validation run
  without being told; `instances/*.json`'s `extra_env` sets it unconditionally
  for qutebrowser instances so future runs don't have to rediscover it.
- **Fresh containers have no `curl`** even though the base image needs it to
  fetch the `claude` binary - bootstrapped with a one-time `apt-get install`
  before the download.

## The 15 selected instances

Filtered to Python, then selected across percentiles of patch length (a
difficulty proxy) from `2,130` to `174,888` characters - easy to
very-very-hard, spanning all three repos. See `manifest.json` for the full
list with `n_fail_to_pass`/`n_pass_to_pass` counts, or `instances/*.json` for
full per-instance detail (problem statement, exact test IDs, docker image).

One candidate was swapped out during selection: a qutebrowser instance at a
similar difficulty rank had `pass_to_pass=1821` - correct by the dataset's
own labels, but running 1,821 tests per job would make every seed of that
one instance impractically slow. Replaced with an ansible instance at
comparable patch size and `pass_to_pass=4`.

## Validated

Two full end-to-end runs on `swebench-qutebrowser-e64622cd` (rank 1, the
easiest instance) through the actual runner code (not just manual steps):
both produced a genuinely different, independently-correct fix (not
memorization of the gold patch), graded against the complete official
criteria (all `fail_to_pass` + all `pass_to_pass`, not a partial check).
$0.10-$0.15 and 3-4 minutes per run on this laptop under `amd64` emulation
(these images are `linux/amd64`; this laptop is `arm64`) - expect faster and
cheaper on the GPU box's native `amd64` hardware, and note the GPU box is
also required at all for the routing condition (this laptop has no vLLM).

**Not yet run:** the full 15-instance sweep. Each instance costs real cloud
spend and wall time that scales with difficulty (the hardest instance's
patch is 82x longer than the easiest), so this is intentionally left for an
explicit decision on scope (how many seeds, cloud-only first vs. paired)
rather than launched automatically.

## Running it

```bash
python3 eval-suite/swebench/runner/run_swebench.py \
  --instances all \
  --conditions cloud \
  --seeds 1
```

`--conditions cloud,routing` requires a reachable vLLM (`--vllm-url`), same
as `run_eval.py` - i.e., the GPU box. `--instances swebench-qutebrowser-e64622cd`
(comma-separated slugs) runs a subset; `manifest.json` lists every slug.

Output: `swebench/results/<experiment-id>/summary.md` (per-instance pass
rate with Wilson CI, same statistics module as the rest of the suite) and
`summary.json` (full detail including each job's `claude_result` - cost,
turns, duration, the model's own summary of its fix).
