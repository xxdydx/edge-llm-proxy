# edge-llm-proxy

A local small/quantized model + KV/prefix cache that serves latency-sensitive
calls itself and offloads heavy ones to the cloud, behind an
Anthropic-API-compatible endpoint. Research plan: [PLAN.md](PLAN.md).

Dev runs on disposable FlowMesh GPU boxes. The Qwen2.5-7B baseline uses an RTX
5080; the Qwen3.8-27B NVFP4 comparison uses an RTX 5090. Sessions are wiped on TTL
expiry, so each box is bootstrapped from the selected setup profile.

---

## Setup (once)

```bash
pip install "flowmesh[cli]"
flowmesh init https://lum.id/fm --api-key <flowmesh-api-key>
```

Create `.env` in the repo root (gitignored — the repo is public):

```
ANTHROPIC_AUTH_TOKEN=lm_pat_live_...     # Lumid claude:proxy PAT
```

Only needed if you'll rebuild the container image:

```bash
docker buildx create --name xbuilder --driver docker-container --use
docker run --privileged --rm tonistiigi/binfmt --install amd64
```

## Daily use

```bash
./flowmesh-up.sh --setup qwen25-7b   # RTX 5080 baseline
./flowmesh-up.sh --setup qwen38-27b  # RTX 5090 comparison
```

With no `--setup`, the command keeps the 7B default. Each invocation submits the
profile's workflow, waits for SSH, writes a setup-specific alias to
`~/.ssh/config`, copies `.env`, uploads the current sanitized local source, and
runs `bootstrap.sh --setup <name>`. It ends with a `ready.` banner and task id.
Run the two commands in separate terminals if both boxes are needed at once;
their task IDs, SSH aliases, VS Code tunnel names, results, and traces do not
collide.

It also starts a VS Code tunnel and opens a window on the box's files, so on a
normal day that one command is all of it.

```bash
ssh fmbox-qwen25-7b                                # 7B shell
ssh fmbox-qwen38-27b                               # 27B shell
scp -r fmbox-qwen38-27b:~/edge-llm-proxy-main/results/qwen38-27b-5090 results/
flowmesh task stop <task-id>                       # release (TTL is 8h)
```

Inspect either resolved profile without provisioning a GPU:

```bash
./flowmesh-up.sh --setup qwen38-27b --print-config
./bootstrap.sh --setup qwen38-27b --print-config
```

## Recording traces

On the laptop. Needs no GPU — v0 forwards everything to cloud and records it.

```bash
source ./trace-up.sh     # must be sourced, not executed
claude
```

Starts edgeproxy on `:8765`, loads `.env`, exports `ANTHROPIC_BASE_URL`.
Traces land in `traces/YYYY-MM-DD.jsonl`.

```bash
.venv/bin/python -m edgeproxy.trace.inspect traces/*.jsonl   # summary
kill $(cat logs/proxy.pid)                                   # stop
```

### Observe Anthropic prompt-cache state

The cloud-cache shadow is opt-in and cannot affect routing:

```bash
.venv/bin/python -m edgeproxy.server \
  --port 8765 \
  --policy cloud-only \
  --cloud-cache-tracking observe
```

Each `/v1/messages` trace then includes `cloud_cache.prediction`, provider
cache usage, and prediction agreement. A confirmed cache read refreshes the
shadow TTL; a prediction by itself never does. State is in memory and returns
to `unknown` after a proxy restart.

Run the smallest real-cloud validation from the laptop with:

```bash
.venv/bin/python scripts/measure_anthropic_cache.py \
  --suite smoke \
  --model claude-sonnet-5 \
  --prefix-tokens 2048 \
  --max-input-token-budget 10000 \
  --confirm-live
```

The runner loads the existing `.env`, starts an isolated cloud-only proxy,
uses the Count Tokens endpoint to size the prefix, and sends a cold/warm pair.
`--confirm-live` is mandatory because this incurs real upstream calls. A valid
cache result requires the response to expose Anthropic's
`cache_read_input_tokens` / `cache_creation_input_tokens`; an Anthropic-
compatible backend without those fields is reported as invalid rather than as
a cache miss. Use `--help` for correctness, performance, TTL, cost-cap, and
diagnostics options.

## Running Claude Code against the local model

On the box, once `bootstrap.sh` has vLLM up:

```bash
curl -fsSL https://claude.ai/install.sh | bash     # once per session, no root needed
export PATH="$HOME/.local/bin:$PATH"

export ANTHROPIC_BASE_URL=http://localhost:8001    # vLLM directly
export ANTHROPIC_AUTH_TOKEN=dummy                  # vLLM ignores it
claude
```

vLLM validates the model name, so it must be one of `--served-model-name`.
Claude Code asks for `claude-sonnet-5` and similar, so serve those aliases too:

```bash
SERVED_NAME="local claude-sonnet-5 claude-haiku-4-5-20251001" ./bootstrap.sh serve
```

Expect tool-calling turns to fail — a 4-bit 7B does not reliably emit the
`<tool_call>` format the parser needs. Sidecalls work.

### VS Code

**Use Tunnels, not Remote-SSH.** Remote-SSH cannot work here: the FlowMesh
entrypoint writes `AllowTcpForwarding no` into `/etc/ssh/sshd_config.d/` at
*container start*, so it is reapplied every session no matter what the image
contains. Remote-SSH needs a forwarded channel and fails with
`channel N: open failed: administratively prohibited`.

Tunnels dial **out** to Microsoft's relay instead, so the restriction doesn't
apply. `flowmesh-up.sh` opens the window for you. Each setup has a distinct
tunnel; to connect by hand, use for example:

- <https://vscode.dev/tunnel/flowmesh-qwen25-7b>, or
- desktop VS Code → **Remote Explorer** → switch the dropdown from *SSH* to
  **Tunnels** → `flowmesh-qwen25-7b`

The 27B tunnel is `flowmesh-qwen38-27b`. The similarly named SSH entries are
for terminal/scp access; VS Code must use the **Tunnels** list.

The tunnel login lives in `~/.vscode/cli/token.json` on the box and dies with
the session, so `flowmesh-up.sh` stashes it to `~/.flowmesh/vscode-cli.tar.gz`
and restores it each run. **That file is a real GitHub-issued credential** —
mode 600, outside the repo, treat it like `.env`. If it's missing you'll get a
one-time device-code prompt and the script will say so rather than hang.

Run the tunnel under `tmux` if you start it manually, or it dies with your
terminal:

```bash
ssh -t fmbox-qwen25-7b \
  'tmux new -s tunnel "~/.vscode-server/code-* tunnel --name flowmesh-qwen25-7b"'
```

First connection downloads ~100 MB of VS Code Server onto the box, and does so
**every session** since the disk is wiped. Budget a minute or two.

On the box: vLLM on `:8001`, edgeproxy on `:8000`. Point a harness at it with
`export ANTHROPIC_BASE_URL=http://localhost:8000`.

---

## What lives where

The split is deliberate: **stable things are baked into the image, things still
being iterated on are installed per session.**

| Component | In the image | Installed by `bootstrap.sh` |
| --- | --- | --- |
| system tools | gcc/build-essential, git, curl, tmux, rsync | — |
| python | 3.12 + venv at `/opt/venv` | — |
| proxy deps | fastapi, uvicorn, httpx | — |
| vLLM | — | yes (~5 min) |
| model weights | — | yes (~3 min, ~5 GB) |

vLLM is out of the image on purpose — pinning a version into a multi-GB push
couples the slowest operation to a decision that's still changing.

`build-essential` is **not optional**: Triton JIT-compiles a C extension at
runtime, several minutes into vLLM startup, and fails with an unhelpful
traceback without a compiler.

### Bringing up vLLM + Qwen on the box

`./flowmesh-up.sh` already does this. To do it by hand:

```bash
ssh fmbox-qwen25-7b
cd ~ && rm -rf edge-llm-proxy-main
curl -sL https://github.com/xxdydx/edge-llm-proxy/archive/refs/heads/main.tar.gz | tar xz
cd edge-llm-proxy-main
cp ~/.env .env          # if you copied one over
./bootstrap.sh --setup qwen25-7b
```

The one-command path uploads a sanitized snapshot of the current local working
tree by default, including uncommitted source changes. It excludes `.env`,
traces, results, logs, `claude-memory`, Git metadata, and caches; `.env` is sent
separately. Set `SOURCE_MODE=github` to restore the public-main download path.

Defaults: **`Qwen/Qwen2.5-7B-Instruct-AWQ`**, 4-bit AWQ, 60K serving context
using the model's documented static YaRN scaling, `--gpu-memory-utilization
0.90`, FP16/BF16 KV cache, prefix caching on, and tool calling on with the
`hermes` parser. Serves on `:8001`, edgeproxy on `:8000`.

Cold run is ~15 min: vLLM install ~5, weights ~3 (5 GB), engine startup ~4
(CUDA graph capture dominates).

Check it works:

```bash
curl -s localhost:8001/v1/messages -H 'content-type: application/json' \
  -d '{"model":"local","max_tokens":60,
       "messages":[{"role":"user","content":"Name three Python web frameworks."}]}'
```

### `bootstrap.sh`

Select a setup first, then optionally name a phase. All phases run by default:

```bash
./bootstrap.sh --setup qwen25-7b check
./bootstrap.sh --setup qwen38-27b install
./bootstrap.sh --setup qwen38-27b model
./bootstrap.sh --setup qwen38-27b serve
```

The public files in `setups/` contain model-specific defaults. Explicit
environment variables (and machine-local `.env` values) override a profile.
The 7B setup selects Qwen2.5 AWQ, Hermes, static YaRN, and forced FlashAttention
on the RTX 5080. The 27B setup selects the approved
`Inferact/Qwen3.8-27B-NVFP4` checkpoint, Qwen3 parsers, a 100K context cap, FP8
KV, eager execution, and at most eight sequences on the RTX 5090. The official
full-precision checkpoint is 55.6 GB and does not fit one 5090. Do not reuse
the 7B TTFT or cache calibration for the 27B hybrid architecture.
The proxy receives the selected setup's context limit as well: with the common
0.90 safety margin, the 27B router admits at most 90K input-plus-reserved-output
tokens locally and sends larger calls to cloud.

Overridable via environment:

| Variable | 7B default | 27B setup |
| --- | --- |
| `MODEL` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | `Inferact/Qwen3.8-27B-NVFP4` |
| `SERVED_NAME` | `local` | `local` |
| `MAX_MODEL_LEN` | `60000` | `100000` |
| `NATIVE_MAX_MODEL_LEN` | `32768` | `262144` |
| `YARN_FACTOR` / `YARN_ROPE_THETA` | `4.0` / `1000000` | unused at 100K |
| `VLLM_HF_OVERRIDES` | generated for Qwen2.5 above 32K | empty |
| `GPU_MEM_UTIL` | `0.90` | `0.90` |
| `KV_CACHE_DTYPE` | `auto` | `fp8` |
| `ATTENTION_BACKEND` | forced `FLASH_ATTN` | `auto`, record selected backend |
| `VLLM_SERVER_DEV_MODE` | direct bootstrap `0`; FlowMesh `1` | same |
| `VLLM_FORK_BRANCH` | `vllm-cache-probe-cu130` | same |
| `VLLM_PRECOMPILED_WHEEL_COMMIT` | `4ca856b0b59d87c7b167d1bd8c748421719c9a57` | same |
| `TOOL_CALL_PARSER` / `REASONING_PARSER` | `hermes` / none | `qwen3_coder` / `qwen3` |
| `VLLM_EXTRA_ARGS` | empty | `--enforce-eager --max-num-seqs 8` |
| `EDGEPROXY_MAX_LOCAL_TOKENS` | `60000` | `100000` |
| `EDGEPROXY_LOCAL_TOKEN_MARGIN` | `0.90` | `0.90` |
| `VLLM_PORT` / `PROXY_PORT` | `8001` / `8000` | same |

```bash
MODEL=Qwen/Qwen3-8B-AWQ ./bootstrap.sh
MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.85 ./bootstrap.sh serve
KV_CACHE_DTYPE=fp8 ./bootstrap.sh serve
```

At the default 60K cap, bootstrap passes Qwen2.5's YaRN settings through
`--hf-overrides`; it never bypasses the model-length guard with
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. Static YaRN can reduce short-context quality,
so validate both short prompts and prompts beyond 32K. FlashAttention is an
explicit requirement: bootstrap passes `--attention-backend FLASH_ATTN` and
fails if the startup log does not confirm it. Torch compilation and CUDA graph
capture remain enabled. `VLLM_USE_FLASHINFER_SAMPLER=0` disables only
FlashInfer's JIT-compiled sampler. The selected attention backend and
graph-capture evidence are copied into each
`results/<experiment-namespace>/env-*.txt` file.

`serve` writes `results/<experiment-namespace>/env-<timestamp>.txt` with the GPU
details and cache telemetry. Traces use the same namespace. Pull both directories
off the disposable box before the session dies.

### vLLM speaks Anthropic natively

vLLM 0.26 ships `vllm/entrypoints/anthropic` and registers `/v1/messages` and
`/v1/messages/count_tokens` alongside the OpenAI routes, emitting correct
Anthropic SSE grammar. **So edgeproxy forwards requests unchanged — there is no
Anthropic↔OpenAI translation layer, and none is needed.**

Tool calling is off by default, and vLLM *rejects* any request carrying a
`tools` field without it — which is every Claude Code main-loop call. Hence
`--enable-auto-tool-choice --tool-call-parser hermes` in `bootstrap.sh`. The
parser is model-specific: `hermes` suits Qwen2.5, so change `TOOL_CALL_PARSER`
if you change models.

`bootstrap.sh` also enables `--enable-prompt-tokens-details`. On vLLM 0.27,
the Anthropic response then reports exact post-request prefix-cache usage:

```text
usage.cache_read_input_tokens      tokens reused from the local KV cache
usage.cache_creation_input_tokens  newly cached input tokens
usage.input_tokens                 uncached input-token remainder
```

For these detailed responses, total input is:

```text
input_tokens + cache_read_input_tokens + cache_creation_input_tokens
```

`edgeproxy` preserves these fields in each trace's top-level `usage` object.
The trace inspector reports request-level hit rate and token-weighted reuse
separately for local and cloud placements. Historical records without the
detail fields remain readable, but are excluded from cache-hit-rate
denominators rather than being misclassified as misses.

Every new trace record—local or cloud—also carries the same normalized summary:

```json
"token_accounting": {
  "input_tokens": 1200,
  "output_tokens": 50,
  "tokens_processed": 1250,
  "cache_read_input_tokens": 800,
  "cache_creation_input_tokens": 300,
  "uncached_input_tokens": 100,
  "cache_details_available": true
}
```

Here `input_tokens` is always total input, unlike the provider's raw detailed
`usage.input_tokens`, which is the uncached remainder. `tokens_processed` is
logical token volume (`input_tokens + output_tokens`), not equal-cost GPU work:
cache reads, fresh prefill, and autoregressive output have different costs. If
a backend reports input/output usage but omits cache details, the totals remain
populated while the three cache-breakdown fields are `null`. Failed calls with
no provider usage also use `null`, keeping “unknown” distinct from measured
zero.

Successful `/v1/messages` records also include cache-aware Anthropic list-price
accounting under `cost_savings`. Cloud calls use the provider's actual uncached,
cache-read, 5-minute-write, 1-hour-write, and output counts. Local calls use
the observed local input/output volume plus the pre-dispatch cloud-cache
prediction to estimate the Anthropic bill avoided by routing locally:

```json
"cost_savings": {
  "available": true,
  "requested_model": "claude-sonnet-5",
  "source": "local-usage-plus-cloud-cache-prediction",
  "confidence": "estimated-from-cloud-cache-tracker",
  "cloud_cost_usd": 0.004321,
  "request_saved_usd": 0.004321,
  "running_saved_usd": 0.127654
}
```

`request_saved_usd` is zero for cloud placements. `running_saved_usd` is
updated atomically in JSONL write order, covers the current UTC-daily trace
file, and is recovered if the proxy restarts. Unknown model prices, missing
usage, or unavailable cloud-cache predictions produce explicit `null` costs.
The bundled table is standard global Anthropic list pricing dated 2026-08-26;
it excludes batch/priority/data-residency modifiers, taxes, tool fees, local
electricity, and gateway-specific pricing. A local saving is necessarily an
estimate because local and Anthropic tokenizers and generated output can
differ.

The background `local_resources.vllm` snapshot also includes vLLM's cumulative
prefix-query/hit counters when the installed version exports them. Those
counters are an audit signal only: they are lifetime, process-wide totals and
cannot be assigned to one request under concurrency. The response `usage`
fields are the authoritative per-request observation. Neither source predicts
whether the incoming request will hit before it is routed; that still requires
the planned shadow radix tracker or cache probe.

Quick check against a running box:

```bash
curl -s localhost:8001/v1/messages -H 'content-type: application/json' \
  -d '{"model":"local","max_tokens":60,
       "messages":[{"role":"user","content":"Name three Python web frameworks."}]}'
```

After deploying this version and deliberately restarting vLLM, validate the
cache-detail path through `edgeproxy` with two byte-identical requests:

```bash
nonce="$(date +%s%N)"
prompt="$nonce $(printf 'prefix-cache-validation %.0s' {1..256})"
body="$(jq -nc --arg prompt "$prompt" \
  '{model:"local",max_tokens:8,stream:true,
    messages:[{role:"user",content:$prompt}]}')"

for repetition in 1 2; do
  curl -Ns http://127.0.0.1:8000/v1/messages \
    -H 'content-type: application/json' \
    --data "$body" >/dev/null
done

trace_file="$(ls -t traces/*.jsonl | head -1)"
jq 'select(.path == "/v1/messages" and .placement == "local") |
    {usage, prefix_counters: {
      queries: .local_resources.vllm.prefix_cache_queries_total,
      hits: .local_resources.vllm.prefix_cache_hits_total,
      lifetime_fraction: .local_resources.vllm.prefix_cache_hit_fraction_lifetime
    }}' "$trace_file" | tail -n 36
```

The second response should have positive—and substantially greater—
`usage.cache_read_input_tokens` than the first. A small first-request hit is
possible because the chat template itself may already be resident. The counter
snapshot may lag by one sampling interval and is not expected to equal either
individual request. Confirm the running command contains
`--enable-prompt-tokens-details` if the usage fields are absent; setting a flag
after vLLM has started cannot change that process.

### Edge latency, TPOT, and throughput benchmark

`scripts/edge_tpot.py` drives vLLM directly so proxy and cloud effects do not
contaminate the edge-engine measurement. `prompts.txt` supplies three fixed
prompt seeds. The runner expands them to exact token lengths and sweeps prompt
length, output length, concurrency, and cold/warm prefix state. The older
`measure_local_throughput.py` entry point remains available for compatibility.

Start with a short smoke test:

```bash
/opt/venv/bin/python scripts/edge_tpot.py \
  --tokenizer Qwen/Qwen2.5-7B-Instruct-AWQ \
  --prompt-lengths 1024 \
  --output-lengths 128 \
  --concurrency 1,2 \
  --cache-states cold,warm \
  --repetitions 1 \
  --output results/edge-tpot-smoke.csv
```

Then run the declared full matrix (1K/8K/24K prompts, 128/512/2048 outputs,
concurrency 1/2/4/8, cold/warm, three repetitions):

```bash
/opt/venv/bin/python scripts/edge_tpot.py \
  --tokenizer Qwen/Qwen2.5-7B-Instruct-AWQ \
  --output results/edge-tpot-full.csv
```

The raw CSV has one row per request and labels it `backend=edge`. TPOT is
`decode_ms / (output_tokens - 1)`: the first token belongs to TTFT, leaving one
inter-token interval for every subsequent token. The derived `-summary.csv`
reports p50/p90 TTFT, TPOT, and end-to-end latency, per-request decode speed,
aggregate batch throughput, realized cache fraction, and cache-state validity.
A requested warm condition may be invalid under eviction pressure; the runner
records the actual vLLM counter delta instead of assuming a hit.

The benchmark requires vLLM's cache-reset route. In vLLM 0.27 it is a
development-only endpoint. `flowmesh-up.sh` enables it by default on the
isolated experiment box. When starting the serving phase directly, use:

```bash
VLLM_SERVER_DEV_MODE=1 ./bootstrap.sh serve
```

Setting the variable after vLLM starts does nothing. Development mode exposes
cache-management routes that can disrupt in-flight service, so use it only on
the isolated experiment box, not an externally accessible production server.
For a deliberately production-like FlowMesh run, disable it with
`VLLM_SERVER_DEV_MODE=0 ./flowmesh-up.sh`.

---

## Rebuilding the container image

**Use `xbuilder`.** Docker Desktop's default builder cannot target
`linux/amd64` from an arm64 Mac — it fails to deliver the image while appearing
to succeed, which is exactly how `build-essential` silently went missing for
several sessions.

```bash
docker buildx use xbuilder            # or pass --builder xbuilder
docker login ghcr.io -u xxdydx        # PAT needs write:packages

TAG=$(git rev-parse --short HEAD)
docker buildx build --platform linux/amd64 --builder xbuilder \
  -t ghcr.io/xxdydx/edge-llm-dev:$TAG \
  -t ghcr.io/xxdydx/edge-llm-dev:latest \
  --push .
```

Verify it contains what you think, pulling fresh rather than trusting the local
build cache:

```bash
docker run --rm --platform linux/amd64 --entrypoint sh \
  ghcr.io/xxdydx/edge-llm-dev:$TAG -c "uname -m; which gcc; gcc --version | head -1"
```

Then pin the **digest** (printed by the push as `pushing manifest for ...@sha256:...`)
in [ssh-workflow.yaml](ssh-workflow.yaml):

```yaml
  image: ghcr.io/xxdydx/edge-llm-dev@sha256:<digest>
```

Digest, not `:latest` — workers cache by tag, so a new `:latest` is not
guaranteed to be re-pulled. A digest is the image's content hash, so there is
nothing to get stale.

The [Dockerfile](Dockerfile) must keep building `FROM` the FlowMesh SSH base and
must **not** set `ENTRYPOINT`, `CMD`, or a non-root `USER` — the base's
entrypoint provisions the session user and starts sshd, and needs root at
container start.

---

## Known environment quirks

- No `nvcc` or CUDA toolkit in the image, so anything that JIT-compiles CUDA
  kernels fails. `bootstrap.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0` for this
  reason.
- The session user has no root and no `apt`. Anything missing must go in the
  image.
- No `node`/`npm`, so Claude Code isn't present. It doesn't need to be — traces
  are recorded on the laptop and replayed against the box. If you do want it
  there, `curl -fsSL https://claude.ai/install.sh | bash` needs no root.
- `python3` is not on `PATH`; use `/opt/venv/bin/python`.
- Build is emulated (arm64 Mac → amd64), so image builds are slow.

## Layout

```
edgeproxy/          the proxy — see PLAN.md for design
  server.py         /v1/messages, streaming tee
  trace/            JSONL recording + inspection
bootstrap.sh        box-side setup (runs on the GPU machine)
flowmesh-up.sh      laptop-side driver (submit → ssh → bootstrap)
Dockerfile          session image
ssh-workflow.yaml   task spec, image pinned by digest
results/            per-run environment provenance
```
