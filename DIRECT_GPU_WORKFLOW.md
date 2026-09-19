# Direct-SSH GPU workflow

Use this when the GPU box is created manually from:

```text
ghcr.io/xxdydx/edge-llm-dev@sha256:500a0d2f1f3674ec04ca79b51f0118f600aeef0c148389990aeeb0353c713cc1
```

The box's Lumid port changes every time. Everything after box creation is one
command:

```bash
./direct-gpu.sh up --setup qwen38-27b --ssh-port 31226
```

For the existing 7B profile:

```bash
./direct-gpu.sh up --setup qwen25-7b --ssh-port PORT
```

## What `up` guarantees

1. Refuses the wrong GPU or insufficient VRAM before downloading anything.
2. Streams the current working tree over SSH, excluding `.env`, Git metadata,
   traces, results, logs, private memory, virtualenvs and caches. SCP is not
   used because the Lumid gateway closes its SCP subsystem.
3. Runs `bootstrap.sh` in remote tmux, so laptop sleep or a dropped monitoring
   connection does not kill installation, weight download or vLLM startup.
4. Preserves an already-healthy server and refuses to replace an unhealthy
   existing vLLM process automatically.
5. Checks the exact model root and context length, requires recorded KV-cache
   geometry, and runs a real completion that must return HTTP 200 and
   `FLOWMESH_OK`.
6. Starts a stable laptop endpoint using SSH command execution because the box
   disables ordinary SSH forwarding:

   - 27B: `http://127.0.0.1:18004`
   - 7B: `http://127.0.0.1:18001`

7. Leaves experiment checkpoints and results on the laptop. The disposable box
   holds only the model, runtime, source snapshot and diagnostic logs.

Cloud credentials are not copied by default. Add `--copy-env` only when the
remote edgeproxy itself needs cloud access; direct paired experiments keep the
cloud client and secrets on the laptop.

## Operations

```bash
./direct-gpu.sh status --setup qwen38-27b --ssh-port PORT
./direct-gpu.sh logs --setup qwen38-27b --ssh-port PORT
./direct-gpu.sh relay --setup qwen38-27b --ssh-port PORT
./direct-gpu.sh stop-relay --setup qwen38-27b --ssh-port PORT
```

After `up`, run the Phase 1 local arm on the laptop:

```bash
PHASE1_LOCAL_URL=http://127.0.0.1:18004 \
  .venv/bin/python -m experiments.phase1_router.orchestrator \
  --full-batch \
  --resume-run-dir experiments/phase1_router/results/run-20260910T064025Z \
  --backends local_27b \
  --gen-parallelism 4 \
  --grade-parallelism 6
```

The orchestrator's append-only checkpoint means a box expiry or network outage
only retries transport-failed or missing rows. It does not rerun the frozen 300
successful pairs.

## Why the image remains unchanged

The pinned image already carries the stable, expensive system dependencies:
CUDA 13.2 compiler, cuRAND headers, Python/uv, build tools, tmux and Claude
Code. On the 2026-09-15 RTX 6000 Ada cold start, installing patched vLLM and
project dependencies took about four minutes; downloading the 26.4 GB model
dominated setup. Baking vLLM would add roughly 7 GB to the image while coupling
the image to a fast-moving research fork. Runtime installation is therefore
the smaller and more reproducible maintenance burden. Reconsider only if
repeated measurements show dependency installation, rather than model pull,
is the dominant failure or startup cost.

