#!/usr/bin/env bash
#
# bootstrap.sh — bring a fresh FlowMesh SSH box up to a working edge-LLM dev env.
#
# Sessions are disposable: the filesystem is wiped on TTL expiry, so this runs
# from scratch every time. Target is under 10 minutes cold.
#
# The session image has no git and no root, so fetch the repo as a tarball —
# GitHub serves one for any public repo, no auth required:
#
#   curl -L https://github.com/xxdydx/edge-llm-proxy/archive/refs/heads/main.tar.gz | tar xz
#   cd edge-llm-proxy-main && ./bootstrap.sh
#
# Phases (runs all in order by default, or name one to run just it):
#
#   check    env + GPU + sm_120 kernel support     ~30s,  no downloads
#   install  uv, venv, vLLM, project deps          ~5min
#   model    pull model weights                    ~3min, ~5GB
#   serve    launch vLLM + edgeproxy, wait healthy ~2min
#
# Examples:
#   ./bootstrap.sh check          # just the day-one sm_120 risk check
#   ./bootstrap.sh                # everything
#   MODEL=Qwen/Qwen3-8B-AWQ ./bootstrap.sh
#
set -euo pipefail

# ---------------------------------------------------------------- config ----
# Override any of these from the environment.

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}"
SERVED_NAME="${SERVED_NAME:-local}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"   # set fp8 to roughly double KV capacity

# Spliced unquoted into the vLLM launch so it word-splits into separate flags.
# The escape hatch when a backend misbehaves on this GPU, e.g.:
#   VLLM_EXTRA_ARGS=--enforce-eager    skip torch.compile + CUDA graph capture
#   VLLM_EXTRA_ARGS="-O0"              lowest compilation level
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

VLLM_PORT="${VLLM_PORT:-8001}"
PROXY_PORT="${PROXY_PORT:-8000}"

# vLLM on Blackwell consumer (sm_120) needs a CUDA 12.8+ build. If the default
# wheel fails the check phase, this is the first knob to turn.
VLLM_VERSION="${VLLM_VERSION:-}"           # empty = latest
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# .env holds ANTHROPIC_AUTH_TOKEN (the Lumid claude:proxy PAT) and any
# EDGEPROXY_* overrides. It is gitignored — the repo is public — so it does not
# arrive with `git clone` and has to be copied over separately each session.
# Only `serve` needs it; `check` and `install` are fine without.
if [ -f "$REPO_DIR/.env" ]; then
  set -a; . "$REPO_DIR/.env"; set +a
else
  warn_env=1
fi

# ------------------------------------------------------------- utilities ----

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# Pick the biggest writable scratch location for weights and logs. The box
# advertises ~80GB of local scratch but the mount point varies by worker.
detect_scratch() {
  local best="" best_free=0
  for cand in /scratch /workspace /mnt/scratch /data "$HOME"; do
    [ -d "$cand" ] && [ -w "$cand" ] || continue
    local free
    free=$(df -Pk "$cand" 2>/dev/null | awk 'NR==2 {print $4}') || continue
    if [ "${free:-0}" -gt "$best_free" ]; then best="$cand"; best_free="$free"; fi
  done
  [ -n "$best" ] || die "no writable scratch directory found"
  printf '%s' "$best"
}

SCRATCH="${SCRATCH:-$(detect_scratch)}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
LOG_DIR="$SCRATCH/logs"

# Prefer the venv baked into the custom image (see Dockerfile) — it already has
# Python 3.12 and the proxy deps. Fall back to building one on scratch when
# running on the stock image.
if [ -w /opt/venv ]; then VENV="${VENV:-/opt/venv}"; else VENV="${VENV:-$SCRATCH/venv}"; fi

# ----------------------------------------------------------------- check ----

phase_check() {
  log "scratch:  $SCRATCH ($(df -Ph "$SCRATCH" | awk 'NR==2 {print $4}') free)"
  log "HF_HOME:  $HF_HOME"

  command -v nvidia-smi >/dev/null || die "nvidia-smi not found — no GPU on this box?"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
             --format=csv,noheader | sed 's/^/    /'

  log "cpu: $(nproc) cores | mem: $(free -g | awk 'NR==2 {print $2}')Gi"

  # The day-one schedule risk. A wheel can import fine and still have no
  # kernels compiled for this architecture, which only fails at generation
  # time — so check the compiled arch list explicitly, not just cuda.is_available().
  if [ -x "$VENV/bin/python" ]; then
    log "checking torch/vLLM sm_120 support"
    # The custom image ships a venv holding only the proxy deps, so "a venv
    # exists" does not imply "torch is installed". Exit 2 means not-there-yet,
    # which is fine before install; exit 1 means torch is present but has no
    # kernels for this GPU, which is the case actually worth stopping for.
    local rc=0
    "$VENV/bin/python" - <<'PY' || rc=$?
import sys

try:
    import torch
except ModuleNotFoundError:
    sys.exit(2)

cap  = torch.cuda.get_device_capability()
sm   = f"sm_{cap[0]}{cap[1]}"
arch = torch.cuda.get_arch_list()

print(f"    torch     {torch.__version__}  (cuda {torch.version.cuda})")
print(f"    device    {torch.cuda.get_device_name(0)}  {sm}")
print(f"    archs     {' '.join(arch)}")

if sm not in arch and not any(a.startswith("sm_90") and cap[0] >= 9 for a in arch):
    print(f"    !! torch has no kernels for {sm}", file=sys.stderr)
    sys.exit(1)

try:
    import vllm
    print(f"    vllm      {vllm.__version__}")
except ImportError:
    print("    vllm      not installed yet")
PY
    case "$rc" in
      0) ;;
      2) warn "torch not installed yet — sm_120 verdict deferred until after install" ;;
      *) die "torch has no kernels for this GPU — see the sm_120 fallbacks in PLAN.md §2" ;;
    esac
  else
    warn "venv not built yet — run './bootstrap.sh install' then re-check"
  fi
}

# --------------------------------------------------------------- install ----

phase_install() {
  if ! command -v uv >/dev/null; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if [ -x "$VENV/bin/python" ]; then
    log "reusing venv at $VENV"
  else
    log "creating venv at $VENV"
    uv venv "$VENV" --python 3.12
  fi

  log "installing vLLM (this is the slow part)"
  local spec="vllm${VLLM_VERSION:+==$VLLM_VERSION}"
  VIRTUAL_ENV="$VENV" uv pip install "$spec" --torch-backend=auto \
    || VIRTUAL_ENV="$VENV" uv pip install "$spec" --extra-index-url "$TORCH_INDEX" \
    || die "vLLM install failed — see the sm_120 fallback notes in PLAN.md §2"

  if [ -f "$REPO_DIR/pyproject.toml" ]; then
    log "installing project deps"
    VIRTUAL_ENV="$VENV" uv pip install -e "$REPO_DIR"
  else
    warn "no pyproject.toml yet — skipping project install"
  fi
}

# ----------------------------------------------------------------- model ----

phase_model() {
  log "pulling $MODEL into $HF_HOME"
  mkdir -p "$HF_HOME"
  VIRTUAL_ENV="$VENV" uv pip install -q "huggingface_hub[cli]"
  "$VENV/bin/hf" download "$MODEL" \
    || die "model download failed (gated repo? try 'hf auth login')"
}

# ----------------------------------------------------------------- serve ----

# wait_for_http <url> <name> [timeout] [pid] [logfile]
#
# Polls until healthy, but watches the process too: if it has already exited
# there is nothing to wait for, so bail immediately and print the tail of its
# log rather than burning the full timeout on a corpse.
wait_for_http() {
  local url="$1" name="$2" timeout="${3:-300}" pid="${4:-}" logfile="${5:-}" waited=0
  log "waiting for $name at $url"
  until curl -sf "$url" >/dev/null 2>&1; do
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      warn "$name (pid $pid) exited after ${waited}s without becoming healthy"
      if [ -n "$logfile" ] && [ -f "$logfile" ]; then
        warn "last 40 lines of $logfile:"
        tail -40 "$logfile" >&2
      fi
      die "$name failed to start"
    fi
    sleep 3; waited=$((waited + 3))
    [ "$waited" -lt "$timeout" ] || die "$name did not come up in ${timeout}s — check ${logfile:-$LOG_DIR}"
  done
  log "$name is up (${waited}s)"
}

phase_serve() {
  mkdir -p "$LOG_DIR"

  log "starting vLLM on :$VLLM_PORT"
  nohup "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --enable-prefix-caching \
    $VLLM_EXTRA_ARGS \
    > "$LOG_DIR/vllm.log" 2>&1 &
  local vllm_pid=$!
  echo "$vllm_pid" > "$LOG_DIR/vllm.pid"

  wait_for_http "http://localhost:$VLLM_PORT/health" vLLM 600 \
    "$vllm_pid" "$LOG_DIR/vllm.log"

  # Provenance for the results directory — KV capacity in blocks is the number
  # every prefix-cache and cohort experiment gets normalised against.
  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$REPO_DIR/results"
  {
    echo "timestamp    $stamp"
    echo "model        $MODEL"
    echo "max_model_len $MAX_MODEL_LEN  gpu_mem_util $GPU_MEM_UTIL  kv_dtype $KV_CACHE_DTYPE"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
    grep -iE 'gpu blocks|kv cache size|graph captur' "$LOG_DIR/vllm.log" | head -20 || true
  } > "$REPO_DIR/results/env-$stamp.txt"
  log "wrote results/env-$stamp.txt"

  if [ -n "${warn_env:-}" ]; then
    warn "no .env found — edgeproxy has no upstream token to relay."
    warn "copy it over:  scp .env fmbox:$REPO_DIR/.env"
  fi

  if [ -f "$REPO_DIR/edgeproxy/server.py" ]; then
    log "starting edgeproxy on :$PROXY_PORT"
    nohup "$VENV/bin/python" -m edgeproxy.server \
      --port "$PROXY_PORT" \
      --trace-dir "$REPO_DIR/traces" \
      --vllm-url "http://localhost:$VLLM_PORT" \
      > "$LOG_DIR/proxy.log" 2>&1 &
    local proxy_pid=$!
    echo "$proxy_pid" > "$LOG_DIR/proxy.pid"
    wait_for_http "http://localhost:$PROXY_PORT/health" edgeproxy 60 \
      "$proxy_pid" "$LOG_DIR/proxy.log"
  else
    warn "edgeproxy not built yet — vLLM only"
  fi

  cat <<EOF

  ready.

    vLLM     http://localhost:$VLLM_PORT   (logs: $LOG_DIR/vllm.log)
    proxy    http://localhost:$PROXY_PORT  (logs: $LOG_DIR/proxy.log)

    point the harness at it:
      export ANTHROPIC_BASE_URL=http://localhost:$PROXY_PORT

    smoke test:
      curl -s localhost:$VLLM_PORT/v1/completions -H 'content-type: application/json' \\
        -d '{"model":"$SERVED_NAME","prompt":"hello","max_tokens":10}' | head -c 300

    !! commit or rsync results/ before you disconnect — this box is disposable.

EOF
}

# ------------------------------------------------------------------ main ----

case "${1:-all}" in
  check)   phase_check ;;
  install) phase_install; phase_check ;;
  model)   phase_model ;;
  serve)   phase_serve ;;
  # check runs twice on purpose: once up front for GPU/scratch, and again after
  # install, which is the first point at which the sm_120 verdict is knowable.
  all)     phase_check; phase_install; phase_check; phase_model; phase_serve ;;
  *)       die "unknown phase '$1' (check|install|model|serve|all)" ;;
esac
