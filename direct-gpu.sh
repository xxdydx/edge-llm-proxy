#!/usr/bin/env bash
# Provision and operate a user-supplied FlowMesh GPU box reached by direct SSH.
#
# The user creates the box from the pinned edge-llm-dev image, then supplies the
# changing gateway port:
#
#   ./direct-gpu.sh up --setup qwen38-27b --ssh-port 31226
#   ./direct-gpu.sh status --setup qwen38-27b --ssh-port 31226
#   ./direct-gpu.sh logs --setup qwen38-27b --ssh-port 31226
#   ./direct-gpu.sh stop-relay --setup qwen38-27b --ssh-port 31226
#
# `up` validates hardware, streams a secrets-free source snapshot (SCP is not
# available on the Lumid gateway), starts bootstrap in remote tmux, validates
# the exact model/context/KV geometry with a real completion, then exposes vLLM
# on a stable laptop port. Experiment drivers run on the laptop so checkpoints
# and results survive the disposable GPU box.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-}"
if [ -n "$ACTION" ]; then shift; fi

SETUP_NAME="qwen38-27b"
SSH_TARGET="gw@lum.id"
SSH_PORT=""
REMOTE_DIR="/workspace/flowmesh"
LOCAL_PORT=""
COPY_ENV=0
BOOTSTRAP_TIMEOUT_S=1800

usage() {
  cat <<'EOF'
usage: ./direct-gpu.sh ACTION --ssh-port PORT [options]

actions:
  up          validate GPU, stage source, bootstrap, verify, start relay
  status      read-only GPU/vLLM/relay status
  logs        show remote bootstrap and vLLM log tails
  relay       start only the laptop-side relay after verifying vLLM
  stop-relay  stop only this setup's laptop-side relay

options:
  --setup NAME          qwen38-27b (default) or qwen25-7b
  --ssh-target USER@HOST  default: gw@lum.id
  --ssh-port PORT       required changing Lumid gateway port
  --remote-dir PATH     default: /workspace/flowmesh
  --local-port PORT     default: 18004 for 27B, 18001 for 7B
  --copy-env            copy .env separately to the box (off by default)
  --bootstrap-timeout S default: 1800

Pinned box image:
  ghcr.io/xxdydx/edge-llm-dev@sha256:500a0d2f1f3674ec04ca79b51f0118f600aeef0c148389990aeeb0353c713cc1
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --setup) SETUP_NAME="${2:?missing setup}"; shift 2 ;;
    --ssh-target) SSH_TARGET="${2:?missing SSH target}"; shift 2 ;;
    --ssh-port) SSH_PORT="${2:?missing SSH port}"; shift 2 ;;
    --remote-dir) REMOTE_DIR="${2:?missing remote directory}"; shift 2 ;;
    --local-port) LOCAL_PORT="${2:?missing local port}"; shift 2 ;;
    --copy-env) COPY_ENV=1; shift ;;
    --bootstrap-timeout) BOOTSTRAP_TIMEOUT_S="${2:?missing timeout}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[x] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$ACTION" in
  up|status|logs|relay|stop-relay) ;;
  *) usage >&2; exit 2 ;;
esac
case "$SETUP_NAME" in
  qwen38-27b)
    EXPECTED_GPU="NVIDIA RTX 6000 Ada Generation"
    MIN_VRAM_MIB=40000
    EXPECTED_MODEL="Inferact/Qwen3.8-27B-NVFP4"
    EXPECTED_CONTEXT=262144
    DEFAULT_LOCAL_PORT=18004
    ;;
  qwen25-7b)
    EXPECTED_GPU="NVIDIA GeForce RTX 5080"
    MIN_VRAM_MIB=14000
    EXPECTED_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
    EXPECTED_CONTEXT=60000
    DEFAULT_LOCAL_PORT=18001
    ;;
  *) echo "[x] unsupported setup: $SETUP_NAME" >&2; exit 2 ;;
esac

[[ "$SSH_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "[x] --ssh-port is required" >&2; exit 2; }
[[ "$SSH_TARGET" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$ ]] \
  || { echo "[x] unsafe --ssh-target: $SSH_TARGET" >&2; exit 2; }
[[ "$REMOTE_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] \
  || { echo "[x] unsafe --remote-dir: $REMOTE_DIR" >&2; exit 2; }
[[ "$BOOTSTRAP_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] \
  || { echo "[x] invalid --bootstrap-timeout" >&2; exit 2; }
LOCAL_PORT="${LOCAL_PORT:-$DEFAULT_LOCAL_PORT}"
[[ "$LOCAL_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "[x] invalid --local-port" >&2; exit 2; }

SSH_OPTS=(-T -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=15 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=12)
REMOTE_BOOTSTRAP_SESSION="flowmesh-bootstrap-$SETUP_NAME"
REMOTE_SMOKE_SESSION="flowmesh-smoke-$SETUP_NAME"
LOCAL_RELAY_SESSION="direct-gpu-relay-$SETUP_NAME"
STATE_DIR="$SCRIPT_DIR/logs/direct-gpu-$SETUP_NAME"
RELAY_PID_FILE="$STATE_DIR/relay.pid"
RELAY_LOG="$STATE_DIR/relay.log"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

ssh_run() {
  local attempt
  for attempt in 1 2 3; do
    if ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"; then return 0; fi
    [ "$attempt" -eq 3 ] || sleep 2
  done
  return 1
}

remote_health() {
  ssh_run "curl -sf --max-time 5 http://127.0.0.1:8001/health >/dev/null"
}

validate_gpu() {
  local row name total
  row="$(ssh_run "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1")" \
    || die "cannot query GPU through ssh -p $SSH_PORT $SSH_TARGET"
  name="${row%,*}"; total="${row##*, }"
  [ "$name" = "$EXPECTED_GPU" ] \
    || die "wrong GPU: expected '$EXPECTED_GPU', got '$name'"
  [[ "$total" =~ ^[0-9]+$ ]] && [ "$total" -ge "$MIN_VRAM_MIB" ] \
    || die "insufficient VRAM: expected at least ${MIN_VRAM_MIB} MiB, got '$total'"
  log "GPU verified: $name, ${total} MiB"
}

validate_model() {
  remote_health || die "vLLM is not healthy on the box"
  ssh_run "/opt/venv/bin/python -c 'import json,urllib.request; d=json.load(urllib.request.urlopen(\"http://127.0.0.1:8001/v1/models\", timeout=5))[\"data\"][0]; assert d[\"root\"] == \"$EXPECTED_MODEL\", d; assert d[\"max_model_len\"] == $EXPECTED_CONTEXT, d; print(d[\"root\"], d[\"max_model_len\"])'" \
    || die "served model identity/context validation failed"
  ssh_run "grep -q 'GPU KV cache size:' /scratch/logs/vllm.log" \
    || die "vLLM health passed but KV capacity was not recorded"
  log "model verified: $EXPECTED_MODEL, context=$EXPECTED_CONTEXT"
}

pack_and_stream_source() {
  [ -f "$SCRIPT_DIR/bootstrap.sh" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ] \
    || die "run from a complete flowmesh working tree"
  log "streaming secrets-free working tree (SCP is unsupported by this gateway)"
  LANG=C LC_ALL=C COPYFILE_DISABLE=1 tar --no-xattrs --no-fflags -C "$SCRIPT_DIR" -czf - \
    --exclude='./.git' --exclude='./.env' --exclude='./.env.*' \
    --exclude='./.venv' --exclude='./venv' --exclude='./claude-memory' \
    --exclude='./traces' --exclude='./results' --exclude='./logs' \
    --exclude='./experiments/*/results' --exclude='./.claude' \
    --exclude='./.pytest_cache' --exclude='./.ruff_cache' \
    --exclude='./.mypy_cache' --exclude='*/__pycache__' \
    --exclude='./experiments/new_datasets/.task-source-cache' \
    --exclude='./experiments/new_datasets/*/_vendor' \
    --exclude='*.pyc' --exclude='*.pid' --exclude='.DS_Store' . \
    | ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
        "mkdir -p '$REMOTE_DIR' && tar -xzf - --no-same-owner -C '$REMOTE_DIR'"
}

copy_env() {
  [ "$COPY_ENV" -eq 1 ] || return 0
  [ -f "$SCRIPT_DIR/.env" ] || die "--copy-env requested but .env is missing"
  log "copying .env through an encrypted SSH stream"
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "umask 077; cat > '$REMOTE_DIR/.env'" \
    < "$SCRIPT_DIR/.env"
}

start_bootstrap() {
  if remote_health; then
    log "healthy vLLM already exists; preserving it"
    return 0
  fi
  if ssh_run "pgrep -f '[v]llm serve' >/dev/null"; then
    die "a non-healthy vLLM process already exists; inspect it before replacing it"
  fi
  ssh_run "if tmux has-session -t '$REMOTE_BOOTSTRAP_SESSION' 2>/dev/null; then
      echo existing
    else
      mkdir -p '$REMOTE_DIR/logs'
      tmux new-session -d -s '$REMOTE_BOOTSTRAP_SESSION' \"cd '$REMOTE_DIR' && VLLM_SERVER_DEV_MODE=1 ./bootstrap.sh --setup '$SETUP_NAME' all >logs/direct-bootstrap.log 2>&1\"
      echo started
    fi"
  log "waiting up to ${BOOTSTRAP_TIMEOUT_S}s for vLLM health"
  local waited=0
  until remote_health; do
    if ! ssh_run "tmux has-session -t '$REMOTE_BOOTSTRAP_SESSION' 2>/dev/null"; then
      ssh_run "tail -120 '$REMOTE_DIR/logs/direct-bootstrap.log'" || true
      die "bootstrap exited before vLLM became healthy"
    fi
    sleep 5; waited=$((waited + 5))
    [ "$waited" -lt "$BOOTSTRAP_TIMEOUT_S" ] || die "bootstrap timed out"
    if [ $((waited % 60)) -eq 0 ]; then
      log "bootstrap still active (${waited}s); remote log: $REMOTE_DIR/logs/direct-bootstrap.log"
    fi
  done
}

smoke_model() {
  local smoke_dir="/scratch/logs"
  local status="$smoke_dir/direct-smoke-status.txt"
  local response="$smoke_dir/direct-smoke-response.json"
  ssh_run "rm -f '$status' '$response'; tmux kill-session -t '$REMOTE_SMOKE_SESSION' 2>/dev/null || true; tmux new-session -d -s '$REMOTE_SMOKE_SESSION' \"curl -sS --max-time 180 -o '$response' -w 'http=%{http_code} total=%{time_total}' http://127.0.0.1:8001/v1/completions -H 'content-type: application/json' -d '{\\\"model\\\":\\\"local\\\",\\\"prompt\\\":\\\"Reply with exactly: FLOWMESH_OK\\\",\\\"max_tokens\\\":16,\\\"temperature\\\":0}' >'$status' 2>&1\""
  local waited=0
  while ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
      "tmux has-session -t '$REMOTE_SMOKE_SESSION' 2>/dev/null" \
      >/dev/null 2>&1; do
    sleep 2; waited=$((waited + 2)); [ "$waited" -lt 190 ] || die "smoke timed out"
  done
  ssh_run "grep -q '^http=200 ' '$status' && grep -q 'FLOWMESH_OK' '$response'" \
    || { ssh_run "cat '$status'; cat '$response'" || true; die "real generation smoke failed"; }
  local result
  result="$(ssh_run "cat '$status'")"
  log "real generation verified: $result"
}

start_relay() {
  mkdir -p "$STATE_DIR"
  command -v tmux >/dev/null || die "tmux is required for a persistent laptop relay"
  if tmux has-session -t "$LOCAL_RELAY_SESSION" 2>/dev/null; then
    log "relay already running: http://127.0.0.1:$LOCAL_PORT"
    return 0
  fi
  if lsof -nP -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    die "local port $LOCAL_PORT is already owned by another process"
  fi
  rm -f "$RELAY_LOG" "$RELAY_PID_FILE"
  tmux new-session -d -s "$LOCAL_RELAY_SESSION" \
    "exec python3 '$SCRIPT_DIR/scripts/direct_gpu_relay.py' \
      --listen-port '$LOCAL_PORT' --ssh-port '$SSH_PORT' \
      --ssh-target '$SSH_TARGET' --remote-relay '$REMOTE_DIR/stdio_relay.py' \
      >>'$RELAY_LOG' 2>&1"
  tmux list-panes -t "$LOCAL_RELAY_SESSION" -F '#{pane_pid}' > "$RELAY_PID_FILE"
  local waited=0
  until curl -sf --max-time 5 "http://127.0.0.1:$LOCAL_PORT/health" >/dev/null; do
    tmux has-session -t "$LOCAL_RELAY_SESSION" 2>/dev/null \
      || { cat "$RELAY_LOG" >&2; die "relay exited"; }
    sleep 1; waited=$((waited + 1)); [ "$waited" -lt 30 ] || die "relay health timed out"
  done
  log "relay verified: http://127.0.0.1:$LOCAL_PORT -> remote vLLM :8001"
}

stop_relay() {
  if tmux has-session -t "$LOCAL_RELAY_SESSION" 2>/dev/null; then
    tmux kill-session -t "$LOCAL_RELAY_SESSION"
  fi
  rm -f "$RELAY_PID_FILE"
  log "relay stopped"
}

show_status() {
  validate_gpu
  ssh_run "printf 'vllm_health='; curl -sS --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health || true; echo; nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader; curl -sS --max-time 5 http://127.0.0.1:8001/v1/models 2>/dev/null || true; echo"
  if tmux has-session -t "$LOCAL_RELAY_SESSION" 2>/dev/null; then
    echo "relay=http://127.0.0.1:$LOCAL_PORT tmux=$LOCAL_RELAY_SESSION"
  else
    echo "relay=stopped"
  fi
}

case "$ACTION" in
  up)
    validate_gpu
    pack_and_stream_source
    copy_env
    start_bootstrap
    validate_model
    smoke_model
    start_relay
    log "ready; set PHASE1_LOCAL_URL=http://127.0.0.1:$LOCAL_PORT"
    ;;
  status) show_status ;;
  logs)
    ssh_run "echo '== bootstrap =='; tail -100 '$REMOTE_DIR/logs/direct-bootstrap.log' 2>/dev/null || true; echo '== vllm =='; tail -100 /scratch/logs/vllm.log 2>/dev/null || true"
    ;;
  relay) validate_gpu; validate_model; start_relay ;;
  stop-relay) stop_relay ;;
esac
