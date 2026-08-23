# edge-llm-proxy

A local small/quantized model + KV/prefix cache that serves latency-sensitive
calls itself and offloads heavy ones to the cloud, behind an
Anthropic-API-compatible endpoint. Research plan: [PLAN.md](PLAN.md).

Dev runs on a disposable FlowMesh GPU box (RTX 5080, 16 GB). Sessions are wiped
on TTL expiry, so everything is rebuilt from git each time.

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
./flowmesh-up.sh
```

Submits the task, waits for SSH, writes an `fmbox` alias to `~/.ssh/config`,
copies `.env`, fetches the repo on the box, and runs `bootstrap.sh`. Ends with a
`ready.` banner and the task id.

It also starts a VS Code tunnel and opens a window on the box's files, so on a
normal day that one command is all of it.

```bash
ssh fmbox                                          # plain shell
scp fmbox:~/edge-llm-proxy-main/results/* results/ # pull results before the TTL
flowmesh task stop <task-id>                       # release (TTL is 8h)
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
apply. `flowmesh-up.sh` opens the window for you; to connect by hand:

- <https://vscode.dev/tunnel/fmbox>, or
- desktop VS Code → **Remote Explorer** → switch the dropdown from *SSH* to
  **Tunnels** → `fmbox`

Note both the SSH alias and the tunnel are named `fmbox`, so Remote Explorer
shows two similar entries — the SSH one is the one that doesn't work.

The tunnel login lives in `~/.vscode/cli/token.json` on the box and dies with
the session, so `flowmesh-up.sh` stashes it to `~/.flowmesh/vscode-cli.tar.gz`
and restores it each run. **That file is a real GitHub-issued credential** —
mode 600, outside the repo, treat it like `.env`. If it's missing you'll get a
one-time device-code prompt and the script will say so rather than hang.

Run the tunnel under `tmux` if you start it manually, or it dies with your
terminal:

```bash
ssh -t fmbox 'tmux new -s tunnel "~/.vscode-server/code-* tunnel --name fmbox"'
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
ssh fmbox
cd ~ && rm -rf edge-llm-proxy-main
curl -sL https://github.com/xxdydx/edge-llm-proxy/archive/refs/heads/main.tar.gz | tar xz
cd edge-llm-proxy-main
cp ~/.env .env          # if you copied one over
./bootstrap.sh
```

Defaults: **`Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`**, 4-bit AWQ, 32K context,
`--gpu-memory-utilization 0.90`, prefix caching on, tool calling on with the
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

Four phases, all four by default, or name one:

```bash
./bootstrap.sh check      # GPU, scratch, sm_120 kernel support   ~30s
./bootstrap.sh install    # vLLM into /opt/venv                   ~5min
./bootstrap.sh model      # pull weights                          ~3min
./bootstrap.sh serve      # launch vLLM + edgeproxy               ~2min
```

Overridable via environment:

| Variable | Default |
| --- | --- |
| `MODEL` | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` |
| `SERVED_NAME` | `local` (space-separated list for aliases) |
| `MAX_MODEL_LEN` | `32768` |
| `GPU_MEM_UTIL` | `0.90` |
| `KV_CACHE_DTYPE` | `auto` (`fp8` roughly doubles KV capacity) |
| `VLLM_SERVER_DEV_MODE` | `0`; set `1` only for cache-reset benchmarks |
| `TOOL_CALL_PARSER` | `hermes` (model-specific) |
| `VLLM_EXTRA_ARGS` | empty (`--enforce-eager` skips CUDA graphs) |
| `VLLM_PORT` / `PROXY_PORT` | `8001` / `8000` |

```bash
MODEL=Qwen/Qwen3-8B-AWQ ./bootstrap.sh
MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.85 ./bootstrap.sh serve
KV_CACHE_DTYPE=fp8 ./bootstrap.sh serve
```

`serve` writes `results/env-<timestamp>.txt` with the GPU details and KV cache
size. That number is the denominator for every prefix-cache experiment — pull it
off the box before the session dies.

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

### Local latency and throughput benchmark

`scripts/measure_local_throughput.py` drives vLLM directly so proxy and cloud
effects do not contaminate the local-engine measurement. `prompts.txt` supplies
three fixed prompt seeds. The runner expands them to exact token lengths and
sweeps prompt length, output length, concurrency, and cold/warm prefix state.

Start with a short smoke test:

```bash
/opt/venv/bin/python scripts/measure_local_throughput.py \
  --tokenizer Qwen/Qwen2.5-7B-Instruct-AWQ \
  --prompt-lengths 1024 \
  --output-lengths 128 \
  --concurrency 1,2 \
  --cache-states cold,warm \
  --repetitions 1 \
  --output results/local-throughput-smoke.csv
```

Then run the declared full matrix (1K/8K/24K prompts, 128/512/2048 outputs,
concurrency 1/2/4/8, cold/warm, three repetitions):

```bash
/opt/venv/bin/python scripts/measure_local_throughput.py \
  --tokenizer Qwen/Qwen2.5-7B-Instruct-AWQ \
  --output results/local-throughput-full.csv
```

The raw CSV has one row per request. The derived `-summary.csv` reports p50/p90
TTFT and end-to-end latency, per-request post-first-token decode speed, aggregate
batch throughput, realized cache fraction, and cache-state validity. A requested
warm condition may be invalid under eviction pressure; the runner records the
actual vLLM counter delta instead of assuming a hit.

The benchmark requires vLLM's cache-reset route. In vLLM 0.27 it is a
development-only endpoint, so vLLM must have been started with:

```bash
VLLM_SERVER_DEV_MODE=1 ./bootstrap.sh serve
```

Setting the variable after vLLM starts does nothing. Development mode exposes
cache-management routes that can disrupt in-flight service, so use it only on
the isolated experiment box, not an externally accessible production server.

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
