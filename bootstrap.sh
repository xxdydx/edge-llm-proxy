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
#   ./bootstrap.sh check                         # default 7B setup
#   ./bootstrap.sh --setup qwen38-27b            # complete 27B setup
#   ./bootstrap.sh --setup qwen38-27b check      # inspect its GPU environment
#   ./bootstrap.sh --setup qwen38-27b --print-config
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_NAME="${FLOWMESH_SETUP:-qwen25-7b}"
PHASE="all"
PRINT_CONFIG=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --setup)
      [ "$#" -ge 2 ] || { echo "[x] --setup requires a name" >&2; exit 2; }
      SETUP_NAME="$2"; shift 2 ;;
    --print-config)
      PRINT_CONFIG=1; shift ;;
    check|install|model|harness|serve|all)
      PHASE="$1"; shift ;;
    -h|--help)
      cat <<'EOF'
usage: ./bootstrap.sh [--setup NAME] [--print-config] [check|install|model|harness|serve|all]

setups: qwen25-7b | qwen38-27b
EOF
      exit 0 ;;
    *) echo "[x] unknown argument '$1'" >&2; exit 2 ;;
  esac
done

case "$SETUP_NAME" in
  qwen25-7b|qwen38-27b) ;;
  *) echo "[x] unknown setup '$SETUP_NAME' (qwen25-7b | qwen38-27b)" >&2; exit 2 ;;
esac

# Secrets and machine-local overrides remain in .env. Load them before the
# public setup profile so profile assignments can use ${VAR:-default}: an
# existing environment/.env value wins, then profile, then the script's generic
# fallback. This preserves the script's historical .env override behavior.
if [ -f "$REPO_DIR/.env" ]; then
  set -a; . "$REPO_DIR/.env"; set +a
else
  warn_env=1
fi

SETUP_FILE="$REPO_DIR/setups/$SETUP_NAME.env"
[ -f "$SETUP_FILE" ] || { echo "[x] setup file missing: $SETUP_FILE" >&2; exit 2; }
# shellcheck source=/dev/null
. "$SETUP_FILE"
FLOWMESH_SETUP="$SETUP_NAME"

# =============================================================================
#  MODEL — change these two lines to swap models. Nothing else needs touching.
#
#  The parser is model-specific and must match, or tool calls are emitted as
#  plain text and the harness never executes them.
#
#    MODEL                                    TOOL_CALL_PARSER   notes
#    Qwen/Qwen2.5-7B-Instruct-AWQ             hermes             ~139K KV tokens
#    Qwen/Qwen3-8B-AWQ                        hermes             2.6x KV/token —
#                                                                use KV_CACHE_DTYPE=fp8
#    Qwen/Qwen2.5-Coder-7B-Instruct-AWQ       hermes             codes well, does
#                                                                NOT emit <tool_call>
#    mistralai/Ministral-8B-Instruct-2410     mistral
#    Salesforce/xLAM-2-8b-fc-r                xlam               function-calling
#                                                                specialist
#    meta-llama/Llama-3.1-8B-Instruct         llama
#
#  Hybrid models require architecture-aware cache accounting. The Qwen3.5
#  profile uses vLLM's hybrid cache manager and must be calibrated separately;
#  never reuse the Qwen2.5 TTFT/KV geometry.
# =============================================================================

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
REASONING_PARSER="${REASONING_PARSER:-}"
QUANTIZATION="${QUANTIZATION:-}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
EXPERIMENT_NAMESPACE="${EXPERIMENT_NAMESPACE:-$SETUP_NAME}"

# ---------------------------------------------------------------- config ----
# Override any of these from the environment.

# Unquoted at the call site so multiple names word-split into separate aliases.
# vLLM rejects requests naming a model it does not serve, and Claude Code asks
# for "claude-sonnet-5" et al, so those need to be aliases if you point the
# harness straight at vLLM.
SERVED_NAME="${SERVED_NAME:-local}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-60000}"
NATIVE_MAX_MODEL_LEN="${NATIVE_MAX_MODEL_LEN:-32768}"
YARN_FACTOR="${YARN_FACTOR:-4.0}"
YARN_ROPE_THETA="${YARN_ROPE_THETA:-1000000}"
# Empty means bootstrap generates the documented Qwen2.5 YaRN override when
# MAX_MODEL_LEN exceeds NATIVE_MAX_MODEL_LEN. Non-Qwen models must supply their
# own model-specific JSON rather than inheriting a potentially unsafe recipe.
VLLM_HF_OVERRIDES="${VLLM_HF_OVERRIDES:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Keep FP16/BF16 KV by default. FP8 buys capacity, but needs a separate
# correctness run on sm_120 before it is an acceptable default.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# FlashAttention is an explicit experiment requirement: do not silently fall
# back to another backend. Override only for a deliberate comparison or
# diagnosis (for example ATTENTION_BACKEND=TRITON_ATTN).
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"

# vLLM 0.27 gates destructive benchmark/debug routes such as
# POST /reset_prefix_cache behind this environment variable. Leave it off for
# ordinary serving; cache-controlled experiments must launch with
# VLLM_SERVER_DEV_MODE=1.
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-0}"

# Local agent calls should be deterministic.  The server-side generation
# override supplies the default; edgeproxy also rewrites local requests because
# an explicit client temperature takes precedence over a server default.
VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-0}"

# Constrained tool decoding is available by default in current vLLM, but make
# both halves explicit so a model/server upgrade cannot silently change them.
# Requests using tool_choice=auto still need at least one tool with strict=true;
# edgeproxy adds that field to every client tool on the local path.
STRUCTURED_OUTPUT_BACKEND="${STRUCTURED_OUTPUT_BACKEND:-xgrammar}"
export VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING:-true}"

# Spliced unquoted into the vLLM launch so it word-splits into separate flags.
# The escape hatch when a backend misbehaves on this GPU, e.g.:
#   VLLM_EXTRA_ARGS=--enforce-eager    skip torch.compile + CUDA graph capture
#   VLLM_EXTRA_ARGS="-O0"              lowest compilation level
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# FlashInfer JIT-compiles its *sampling* kernels on first use, which needs ninja
# and nvcc. The session image has neither, so disable that sampler. This does
# not disable FlashInfer attention: attention is selected independently by
# vLLM and is reported from the startup log below.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

VLLM_PORT="${VLLM_PORT:-8001}"
PROXY_PORT="${PROXY_PORT:-8000}"

# vLLM on Blackwell consumer (sm_120) needs a CUDA 12.8+ build. If the default
# wheel fails the check phase, this is the first knob to turn.
VLLM_VERSION="${VLLM_VERSION:-}"           # empty = latest
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

print_config() {
  cat <<EOF
setup=$SETUP_NAME
workflow=${FLOWMESH_WORKFLOW:-}
ssh_alias=${FLOWMESH_SSH_ALIAS:-}
tunnel_name=${FLOWMESH_TUNNEL_NAME:-}
experiment_namespace=$EXPERIMENT_NAMESPACE
model=$MODEL
tool_call_parser=$TOOL_CALL_PARSER
reasoning_parser=${REASONING_PARSER:-none}
quantization=${QUANTIZATION:-none}
language_model_only=$LANGUAGE_MODEL_ONLY
max_model_len=$MAX_MODEL_LEN
native_max_model_len=$NATIVE_MAX_MODEL_LEN
attention_backend=$ATTENTION_BACKEND
kv_cache_dtype=$KV_CACHE_DTYPE
gpu_mem_util=$GPU_MEM_UTIL
vllm_extra_args=${VLLM_EXTRA_ARGS:-none}
edgeproxy_max_local_tokens=${EDGEPROXY_MAX_LOCAL_TOKENS:-60000}
edgeproxy_local_token_margin=${EDGEPROXY_LOCAL_TOKEN_MARGIN:-0.90}
EOF
}

if [ "$PRINT_CONFIG" -eq 1 ]; then
  print_config
  exit 0
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

def parse_sm(a):
    if not a.startswith("sm_"):
        return None
    digits = a[3:]
    return int(digits[:-1]), int(digits[-1])

# CUDA guarantees a cubin built for compute capability X.y runs unmodified on
# any device X.z with z >= y (same major, lower-or-equal minor) — e.g. sm_86
# kernels are binary-compatible with an sm_89 (Ada) device. Check that before
# giving up, instead of requiring an exact arch-string match.
same_family = any(
    p is not None and p[0] == cap[0] and p[1] <= cap[1]
    for p in (parse_sm(a) for a in arch)
)
cross_gen = any(a.startswith("sm_90") and cap[0] >= 9 for a in arch)

if sm not in arch and not same_family and not cross_gen:
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

  log "installing vLLM fork (probe patch, precompiled Python-only build)"
  # This branch is the probe-only Python patch directly on top of the wheel
  # commit below. Keep both defaults aligned: Python-only vLLM installs reuse
  # native binaries from that exact upstream base.
  local vllm_branch="${VLLM_FORK_BRANCH:-vllm-cache-probe-cu130}"
  local vllm_src="$SCRATCH/vllm-src"
  if [ ! -d "$vllm_src" ]; then
    mkdir -p "$vllm_src"
    curl -L "https://github.com/xxdydx/vllm/archive/refs/heads/${vllm_branch}.tar.gz" \
      | tar xz -C "$vllm_src" --strip-components=1
  fi
  export VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.0.0}"
  export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-4ca856b0b59d87c7b167d1bd8c748421719c9a57}"
  VIRTUAL_ENV="$VENV" VLLM_USE_PRECOMPILED=1 uv pip install -e "$vllm_src" --torch-backend=auto --index-strategy unsafe-best-match \
    || VIRTUAL_ENV="$VENV" VLLM_USE_PRECOMPILED=1 uv pip install -e "$vllm_src" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
    || die "vLLM fork install failed — see PLAN.md §2, or set VLLM_PRECOMPILED_WHEEL_COMMIT if the fork's base commit has no prebuilt wheel"

  if [ -f "$REPO_DIR/pyproject.toml" ]; then
    log "installing project deps"
    VIRTUAL_ENV="$VENV" uv pip install -e "$REPO_DIR"
  else
    warn "no pyproject.toml yet — skipping project install"
  fi
}

# ----------------------------------------------------------------- model ----

# Claude Code — the harness. Installed here rather than in the image because
# its installer executes the downloaded amd64 binary to verify itself, which
# segfaults under the QEMU emulation the cross-build uses. Runs fine here, on
# real hardware. No root needed; lands in ~/.local/bin.
phase_harness() {
  if command -v claude >/dev/null; then
    log "claude already installed: $(command -v claude)"
    return 0
  fi
  log "installing Claude Code"
  curl -fsSL https://claude.ai/install.sh | bash \
    || { warn "Claude Code install failed — the box still serves, you just cannot drive it from here"; return 0; }
  export PATH="$HOME/.local/bin:$PATH"
  command -v claude >/dev/null && log "claude: $(command -v claude)"
}

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

  [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] \
    || die "MAX_MODEL_LEN must be a positive integer (got '$MAX_MODEL_LEN')"
  [[ "$NATIVE_MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] \
    || die "NATIVE_MAX_MODEL_LEN must be a positive integer (got '$NATIVE_MAX_MODEL_LEN')"
  case "$LANGUAGE_MODEL_ONLY" in
    0|1) ;;
    *) die "LANGUAGE_MODEL_ONLY must be 0 or 1 (got '$LANGUAGE_MODEL_ONLY')" ;;
  esac

  local hf_override_args=()
  if [ "$MAX_MODEL_LEN" -gt "$NATIVE_MAX_MODEL_LEN" ]; then
    if [ -z "$VLLM_HF_OVERRIDES" ]; then
      case "$MODEL" in
        Qwen/Qwen2.5-*)
          VLLM_HF_OVERRIDES="{\"rope_parameters\":{\"factor\":$YARN_FACTOR,\"original_max_position_embeddings\":$NATIVE_MAX_MODEL_LEN,\"rope_theta\":$YARN_ROPE_THETA,\"rope_type\":\"yarn\"}}"
          ;;
        *)
          die "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds native $NATIVE_MAX_MODEL_LEN for $MODEL; set model-specific VLLM_HF_OVERRIDES"
          ;;
      esac
    fi
    hf_override_args=(--hf-overrides "$VLLM_HF_OVERRIDES")
    log "long context: $MAX_MODEL_LEN tokens with model-specific RoPE override"
    warn "static YaRN can reduce short-context quality; validate both short and >32K prompts"
  elif [ -n "$VLLM_HF_OVERRIDES" ]; then
    hf_override_args=(--hf-overrides "$VLLM_HF_OVERRIDES")
    log "applying explicit Hugging Face config override"
  fi

  local attention_args=()
  if [ "$ATTENTION_BACKEND" != auto ]; then
    attention_args=(--attention-backend "$ATTENTION_BACKEND")
    warn "forcing attention backend: $ATTENTION_BACKEND"
  else
    log "attention backend: auto (vLLM selects the best compatible optimized backend)"
  fi

  local reasoning_args=()
  if [ -n "$REASONING_PARSER" ]; then
    reasoning_args=(--reasoning-parser "$REASONING_PARSER")
  fi

  local quantization_args=()
  if [ -n "$QUANTIZATION" ]; then
    quantization_args=(--quantization "$QUANTIZATION")
  fi

  local language_model_args=()
  if [ "$LANGUAGE_MODEL_ONLY" -eq 1 ]; then
    language_model_args=(--language-model-only)
  fi

  case " $VLLM_EXTRA_ARGS " in
    *" --enforce-eager "*|*" -O0 "*)
      warn "VLLM_EXTRA_ARGS disables or reduces CUDA graphs/torch.compile optimizations"
      ;;
  esac

  log "starting vLLM on :$VLLM_PORT"
  nohup "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name $SERVED_NAME \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${hf_override_args[@]}" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    "${attention_args[@]}" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    "${reasoning_args[@]}" \
    "${quantization_args[@]}" \
    "${language_model_args[@]}" \
    --override-generation-config "{\"temperature\":$VLLM_TEMPERATURE}" \
    --structured-outputs-config.backend "$STRUCTURED_OUTPUT_BACKEND" \
    $VLLM_EXTRA_ARGS \
    > "$LOG_DIR/vllm.log" 2>&1 &
  local vllm_pid=$!
  echo "$vllm_pid" > "$LOG_DIR/vllm.pid"

  wait_for_http "http://localhost:$VLLM_PORT/health" vLLM 600 \
    "$vllm_pid" "$LOG_DIR/vllm.log"

  local attention_line
  # vLLM 0.27.1 writes e.g.:
  #   Using AttentionBackendEnum.FLASH_ATTN backend.
  # Older/newer versions may use "Using FLASH_ATTN attention backend".
  attention_line="$(grep -iE 'using .*attention.*backend|using .*flash_attn.*backend' "$LOG_DIR/vllm.log" | tail -1 || true)"
  if [ -n "$attention_line" ]; then
    log "verified: $attention_line"
    if [ "$ATTENTION_BACKEND" != auto ] \
       && ! grep -Fqi "$ATTENTION_BACKEND" <<<"$attention_line"; then
      die "requested $ATTENTION_BACKEND but vLLM reported: $attention_line"
    fi
  elif [ "$ATTENTION_BACKEND" != auto ]; then
    die "vLLM is healthy but did not explicitly confirm required attention backend $ATTENTION_BACKEND"
  else
    warn "vLLM is healthy, but its log did not expose the selected attention backend"
  fi

  # Convert vLLM's token-level KV occupancy into an estimated GiB figure in
  # proxy traces. This is derived rather than hard-coded so changing MODEL or
  # KV_CACHE_DTYPE keeps the estimate honest. An explicit environment value
  # remains available for architectures that do not expose standard attention
  # fields in their Hugging Face config.
  local kv_bytes_per_token="${EDGEPROXY_KV_BYTES_PER_TOKEN:-}"
  if [ -z "$kv_bytes_per_token" ]; then
    kv_bytes_per_token=$("$VENV/bin/python" - "$MODEL" "$KV_CACHE_DTYPE" <<'PY'
import sys
from transformers import AutoConfig

model, cache_dtype = sys.argv[1:]
config = AutoConfig.from_pretrained(model, local_files_only=True)
config = getattr(config, "text_config", config)

layer_types = getattr(config, "layer_types", None)
if layer_types and any(kind not in ("full_attention", "attention") for kind in layer_types):
    print(
        "hybrid cache has no single KV-bytes/token value; leaving GiB estimate unavailable",
        file=sys.stderr,
    )
    sys.exit(2)

layers = getattr(config, "num_hidden_layers")
kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads"))
head_dim = getattr(
    config,
    "head_dim",
    getattr(config, "hidden_size") // getattr(config, "num_attention_heads"),
)

if cache_dtype.lower().startswith("fp8"):
    dtype_bytes = 1
else:
    model_dtype = str(getattr(config, "torch_dtype", "float16")).lower()
    dtype_bytes = 4 if "float32" in model_dtype else 2

# Key + value tensors, for every layer and KV head.
print(2 * layers * kv_heads * head_dim * dtype_bytes)
PY
    ) || warn "could not derive KV bytes/token; KV GiB trace fields will be null"
  fi
  local proxy_resource_args=()
  if [ -n "$kv_bytes_per_token" ]; then
    proxy_resource_args=(--kv-bytes-per-token "$kv_bytes_per_token")
    log "KV telemetry: $kv_bytes_per_token bytes/token"
  fi

  # Provenance for the results directory — KV capacity in blocks is the number
  # every prefix-cache and cohort experiment gets normalised against.
  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local result_dir="$REPO_DIR/results/$EXPERIMENT_NAMESPACE"
  mkdir -p "$result_dir"
  {
    echo "timestamp    $stamp"
    echo "setup        $SETUP_NAME"
    echo "namespace    $EXPERIMENT_NAMESPACE"
    echo "model        $MODEL"
    echo "tool_parser  $TOOL_CALL_PARSER  reasoning_parser ${REASONING_PARSER:-none}"
    echo "quantization ${QUANTIZATION:-none}  language_model_only $LANGUAGE_MODEL_ONLY"
    echo "max_model_len $MAX_MODEL_LEN  gpu_mem_util $GPU_MEM_UTIL  kv_dtype $KV_CACHE_DTYPE"
    echo "vllm_extra_args ${VLLM_EXTRA_ARGS:-none}"
    echo "router_cap ${EDGEPROXY_MAX_LOCAL_TOKENS:-60000}  router_margin ${EDGEPROXY_LOCAL_TOKEN_MARGIN:-0.90}"
    echo "native_model_len $NATIVE_MAX_MODEL_LEN  attention_backend $ATTENTION_BACKEND"
    if [ -n "$VLLM_HF_OVERRIDES" ]; then echo "hf_overrides $VLLM_HF_OVERRIDES"; fi
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
    grep -iE 'gpu blocks|kv cache size|graph captur|attention backend' "$LOG_DIR/vllm.log" | head -30 || true
  } > "$result_dir/env-$stamp.txt"
  log "wrote results/$EXPERIMENT_NAMESPACE/env-$stamp.txt"

  if [ -n "${warn_env:-}" ]; then
    warn "no .env found — edgeproxy has no upstream token to relay."
    warn "copy it over:  scp .env ${FLOWMESH_SSH_ALIAS:-fmbox}:$REPO_DIR/.env"
  fi

  if [ -f "$REPO_DIR/edgeproxy/server.py" ]; then
    log "starting edgeproxy on :$PROXY_PORT"
    nohup "$VENV/bin/python" -m edgeproxy.server \
      --port "$PROXY_PORT" \
      --trace-dir "${EDGEPROXY_TRACE_DIR:-$REPO_DIR/traces/$EXPERIMENT_NAMESPACE}" \
      --vllm-url "http://localhost:$VLLM_PORT" \
      --local-cache-tracking "${EDGEPROXY_LOCAL_CACHE_TRACKING:-observe}" \
      --cloud-cache-tracking "${EDGEPROXY_CLOUD_CACHE_TRACKING:-observe}" \
      --max-local-tokens "${EDGEPROXY_MAX_LOCAL_TOKENS:-60000}" \
      --local-token-margin "${EDGEPROXY_LOCAL_TOKEN_MARGIN:-0.90}" \
      "${proxy_resource_args[@]}" \
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

    !! copy results/$EXPERIMENT_NAMESPACE and traces/$EXPERIMENT_NAMESPACE
       before you disconnect — this box is disposable.

EOF
}

# ------------------------------------------------------------------ main ----

case "$PHASE" in
  check)   phase_check ;;
  install) phase_install; phase_check ;;
  model)   phase_model ;;
  harness) phase_harness ;;
  serve)   phase_serve ;;
  # check runs twice on purpose: once up front for GPU/scratch, and again after
  # install, which is the first point at which the sm_120 verdict is knowable.
  all)     phase_check; phase_install; phase_check; phase_harness; phase_model; phase_serve ;;
  *)       die "unknown phase '$PHASE' (check|install|harness|model|serve|all)" ;;
esac
