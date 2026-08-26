#!/usr/bin/env bash
#
# flowmesh-up.sh — one command from a cold Mac to a running vLLM + edgeproxy.
#
# Submits ssh-workflow.yaml, waits for the session, uploads a sanitized snapshot
# of the current local source plus .env separately, then runs bootstrap.sh.
# This means uncommitted implementation work reaches each disposable box while
# secrets, traces, results, logs, and the private memory layer do not.
#
#   ./flowmesh-up.sh --setup qwen25-7b
#   ./flowmesh-up.sh --setup qwen38-27b
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_NAME="${FLOWMESH_SETUP:-qwen25-7b}"
TUNNEL_ONLY=0
PRINT_CONFIG=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --setup)
      [ "$#" -ge 2 ] || { echo "[x] --setup requires a name" >&2; exit 2; }
      SETUP_NAME="$2"; shift 2 ;;
    --tunnel-only)
      TUNNEL_ONLY=1; shift ;;
    --print-config)
      PRINT_CONFIG=1; shift ;;
    -h|--help)
      cat <<'EOF'
usage: ./flowmesh-up.sh [--setup NAME] [--print-config] [--tunnel-only]

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
SETUP_FILE="$SCRIPT_DIR/setups/$SETUP_NAME.env"
[ -f "$SETUP_FILE" ] || { echo "[x] setup file missing: $SETUP_FILE" >&2; exit 2; }
# shellcheck source=/dev/null
. "$SETUP_FILE"

WORKFLOW="${WORKFLOW:-$FLOWMESH_WORKFLOW}"
SSH_ALIAS="${FLOWMESH_SSH_ALIAS:-fmbox-$SETUP_NAME}"
TUNNEL_NAME="${FLOWMESH_TUNNEL_NAME:-flowmesh-$SETUP_NAME}"
EXPERIMENT_NAMESPACE="${EXPERIMENT_NAMESPACE:-$SETUP_NAME}"
REPO_URL="${REPO_URL:-https://github.com/xxdydx/edge-llm-proxy}"
REPO_DIR_NAME="edge-llm-proxy-main"
SOURCE_MODE="${SOURCE_MODE:-local}"  # local (default) or github
# FlowMesh boxes are isolated experiment environments, and the controlled
# cold/warm benchmarks require vLLM's cache-reset route. Keep this overridable
# so a deliberately production-like run can disable development endpoints.
VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

print_config() {
  cat <<EOF
setup=$SETUP_NAME
workflow=$WORKFLOW
ssh_alias=$SSH_ALIAS
tunnel_name=$TUNNEL_NAME
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

# ------------------------------------------------------------- vs code ------
# Remote-SSH cannot work here: the FlowMesh entrypoint writes
#   AllowTcpForwarding no
# into /etc/ssh/sshd_config.d/ at *container start*, so it is re-applied every
# session no matter what the image contains. Tunnels dial out instead of needing
# an inbound forwarded port, so they are unaffected.
#
# There is no unattended login. The token cannot be copied between boxes (it is
# encrypted with a machine-bound key), and a classic GitHub PAT satisfies
# `tunnel user login` but the tunnels service rejects it with 401 — only the
# device flow issues a token with the right scopes.
#
# So this runs *before* bootstrap: you authorise in ~20s while the slow part
# (vLLM install, weights, CUDA graph capture — ~15 min) runs unattended after.
vscode_tunnel() {
  log "installing VS Code CLI on the box (if absent)"
  ssh_run 'set -e
    if [ ! -x ~/code ] && ! ls ~/.vscode-server/code-* >/dev/null 2>&1; then
      curl -sL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" \
        -o /tmp/vscode-cli.tgz && tar -xf /tmp/vscode-cli.tgz -C ~ && chmod +x ~/code
    fi' || { warn "could not install the VS Code CLI"; return 1; }

  # No unattended option exists. A classic GitHub PAT satisfies `tunnel user
  # login` but the tunnels service rejects it with 401 — only the device flow
  # issues a token with the right scopes. And the resulting token cannot be
  # copied between boxes: it is stored encrypted with a machine-bound key.
  # So: surface the device code and wait, once per session.

  # --provider github skips the interactive account menu, which has no TTY here
  # and would otherwise spin forever redrawing itself.
  log "starting tunnel"
  ssh_run 'CLI=$(ls ~/code ~/.vscode-server/code-* 2>/dev/null | head -1)
    tmux kill-session -t tunnel 2>/dev/null || true
    rm -f ~/tunnel.log
    if ! "$CLI" tunnel user show 2>&1 | grep -qi "logged in with"; then
      tmux new -d -s tunnel "$CLI tunnel user login --provider github >~/tunnel.log 2>&1"
    fi'
  # If already logged in, the loop below starts the tunnel on its first pass.

  local waited=0
  while [ "$waited" -lt 300 ]; do
    # Ask the CLI whether it is logged in rather than grepping its log for a
    # success string — the login command prints nothing reliable on success,
    # and `tunnel user show` is the authoritative answer.
    if [ -z "${tunnel_started:-}" ] \
       && ssh_run 'CLI=$(ls ~/code ~/.vscode-server/code-* 2>/dev/null | head -1)
                   "$CLI" tunnel user show 2>&1 | grep -qi "logged in with"'; then
      log "authorised — starting tunnel"
      ssh_run 'CLI=$(ls ~/code ~/.vscode-server/code-* 2>/dev/null | head -1)
        tmux kill-session -t tunnel 2>/dev/null || true
        rm -f ~/tunnel.log
        tmux new -d -s tunnel "$CLI tunnel --accept-server-license-terms --name '"$TUNNEL_NAME"' >~/tunnel.log 2>&1"'
      tunnel_started=1
    fi

    if ssh_run 'grep -qi "devtunnels.ms\|Open:" ~/tunnel.log 2>/dev/null'; then
      log "tunnel is up: https://vscode.dev/tunnel/$TUNNEL_NAME"
      return 0
    fi

    if [ -z "${prompted:-}" ] && ssh_run 'grep -q "github.com/login/device" ~/tunnel.log 2>/dev/null'; then
      prompted=1
      printf '\n\033[1;33m  GitHub login needed (once per session):\033[0m\n'
      ssh_run 'grep -o "https://github.com/login/device[^ ]*\|use code [A-Z0-9-]*" ~/tunnel.log | sort -u' \
        | sed 's/^/    /'
      printf '  waiting for you to authorise...\n\n'
    fi

    sleep 3; waited=$((waited + 3))
  done

  warn "tunnel did not come up in ${waited}s — check: ssh $SSH_ALIAS 'cat ~/tunnel.log'"
  return 1
}

# `--tunnel-only`: the devbox from an earlier run of this setup is still up,
# but the 300s login window in vscode_tunnel closed before you got to the
# device-code prompt. This re-runs just the tunnel dance against the setup's
# SSH alias already sitting in ~/.ssh/config — no workflow resubmit, no
# re-running the 15-minute bootstrap.
if [ "$TUNNEL_ONLY" -eq 1 ]; then
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_ALIAS" true 2>/dev/null \
    || die "$SSH_ALIAS isn't reachable — is that setup's devbox still up?"
  ssh_run() { ssh "$SSH_ALIAS" "$@"; }
  vscode_tunnel && log "tunnel ready: https://vscode.dev/tunnel/$TUNNEL_NAME"
  exit $?
fi

command -v flowmesh >/dev/null || die "flowmesh CLI not found — see setup-instructions.md"
[ -f "$WORKFLOW" ] || die "$WORKFLOW not found — run this from the repo root"
[ -f bootstrap.sh ] && [ -f pyproject.toml ] \
  || die "run this from the repository root (bootstrap.sh and pyproject.toml are required)"
[ -f .env ] || warn ".env not found locally — edgeproxy will come up with no upstream token"
case "$SOURCE_MODE" in
  local|github) ;;
  *) die "SOURCE_MODE must be 'local' or 'github' (got '$SOURCE_MODE')" ;;
esac

# Validate and package before submitting a paid GPU task. If the local tree is
# malformed, fail here rather than leaving an unusable box running.
if [ "$SOURCE_MODE" = local ]; then
  LOCAL_SOURCE_ARCHIVE="$(mktemp -t flowmesh-source).tar.gz"
  log "packing current local source (excluding secrets, traces, results, logs, and memory)"
  COPYFILE_DISABLE=1 tar -czf "$LOCAL_SOURCE_ARCHIVE" \
    --exclude='./.git' \
    --exclude='./.env' \
    --exclude='./.env.*' \
    --exclude='./.venv' \
    --exclude='./venv' \
    --exclude='./claude-memory' \
    --exclude='./traces' \
    --exclude='./results' \
    --exclude='./logs' \
    --exclude='./.claude' \
    --exclude='./.pytest_cache' \
    --exclude='./.ruff_cache' \
    --exclude='./.mypy_cache' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pid' \
    --exclude='.DS_Store' \
    .
  tar -tzf "$LOCAL_SOURCE_ARCHIVE" | grep -E '^\./bootstrap\.sh$' >/dev/null \
    || die "source archive validation failed: bootstrap.sh missing"
  tar -tzf "$LOCAL_SOURCE_ARCHIVE" | grep -E '^\./pyproject\.toml$' >/dev/null \
    || die "source archive validation failed: pyproject.toml missing"
fi

# ------------------------------------------------------------------ submit --

log "setup: $SETUP_NAME | workflow: $WORKFLOW | model: $MODEL"
log "submitting $WORKFLOW"
SUBMIT_OUT="$(flowmesh workflow submit "$WORKFLOW")"
echo "$SUBMIT_OUT"
TASK_ID="$(echo "$SUBMIT_OUT" | grep -oE 'tsk-[a-f0-9-]+' | head -1)"
[ -n "$TASK_ID" ] || die "couldn't parse a task ID out of 'flowmesh workflow submit' output above"
log "task: $TASK_ID"

# The script stays in the foreground for the life of the box, and Ctrl+C
# releases the task. Deliberately NOT trapped on EXIT: if bootstrap fails we
# want the box left alive to debug, not torn down. Set NO_AUTOSTOP=1 to keep the
# task running after Ctrl+C.
cleanup() {
  trap - INT TERM HUP
  rm -f "${LOCAL_SOURCE_ARCHIVE:-}"
  printf '\n'
  if [ -n "${NO_AUTOSTOP:-}" ]; then
    log "leaving $TASK_ID running (NO_AUTOSTOP set)"
    exit 0
  fi
  log "stopping task $TASK_ID"
  flowmesh task stop "$TASK_ID" >/dev/null 2>&1 \
    && log "task stopped" \
    || warn "could not stop it — run: flowmesh task stop $TASK_ID"
  exit 0
}
trap cleanup INT TERM HUP

# ------------------------------------------------------- capture ssh info --
# `flowmesh ssh connect` blocks until the session is ready, prints the exact
# ssh(1) invocation it's about to exec into, then hands the terminal off to
# it. We want that ready-wait and that exact command line — but not the
# interactive handoff, since this script drives its own non-interactive ssh
# calls afterward. So: run it in the background with stdin from /dev/null,
# poll the log for the "Connecting:" line, then kill it.
CONNECT_LOG="$(mktemp)"
flowmesh ssh connect "$TASK_ID" </dev/null >"$CONNECT_LOG" 2>&1 &
CONNECT_PID=$!

log "waiting for SSH session to come up (this can take a couple minutes)"
# A task that never gets scheduled ("No worker satisfies the task hardware and
# capability requirements") fails in milliseconds and looks identical, from the
# connect log alone, to one that is still starting. Poll the task's own status
# so a terminal state ends the wait instead of burning the full 10 minutes.
for _ in $(seq 1 120); do
  grep -q '^Connecting:' "$CONNECT_LOG" && break
  # `|| st=""` matters here: under `set -e -o pipefail`, a single transient
  # failure of `flowmesh task info` (network blip, brief rate limit) would
  # otherwise kill the whole script with no message, even though the task
  # itself is fine. Treat a failed fetch as "no status yet" and retry.
  st="$(flowmesh task info "$TASK_ID" 2>/dev/null | grep -m1 '"status"' | cut -d'"' -f4)" || st=""
  case "$st" in
    FAILED|CANCELLED|DONE)
      die "task ended early ($st): $(flowmesh task info "$TASK_ID" | grep -m1 '"error"')" ;;
  esac
  sleep 5
done
kill "$CONNECT_PID" 2>/dev/null || true
wait "$CONNECT_PID" 2>/dev/null || true

SSH_LINE="$(grep '^Connecting:' "$CONNECT_LOG" | sed 's/^Connecting: //')"
[ -n "$SSH_LINE" ] || die "session never became ready — see $CONNECT_LOG"
rm -f "$CONNECT_LOG"

# Shape depends on accessMode. `forward`/`direct` prints a literal `-p PORT`
# plus a reachable host:port. `proxy` prints `-o ProxyCommand=flowmesh ssh
# proxy <task-id> ...` instead: no port, and the ProxyCommand value itself
# contains spaces with no quoting in the printed line, so this can't be
# whitespace-split like a normal argv (that would chop "flowmesh ssh proxy
# <task-id>" into separate bogus tokens). We don't need to: for proxy mode the
# ProxyCommand is always exactly "flowmesh ssh proxy $TASK_ID" and the host is
# always literally $TASK_ID, both of which we already have as variables — so
# detect the mode and build the option list directly instead of parsing it out
# of the text.
if echo "$SSH_LINE" | grep -q 'ProxyCommand='; then
  SSH_MODE=proxy
  SSH_TARGET="$(echo "$SSH_LINE" | grep -oE "[A-Za-z0-9_.-]+@${TASK_ID}\$")"
  [ -n "$SSH_TARGET" ] || die "couldn't parse ssh target from: $SSH_LINE"
  SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o LogLevel=ERROR -o "ProxyCommand=flowmesh ssh proxy $TASK_ID")
  log "ssh target: $SSH_TARGET (proxy)"
else
  SSH_MODE=direct
  SSH_PORT="$(echo "$SSH_LINE" | grep -oE -- '-p [0-9]+' | awk '{print $2}')"
  SSH_TARGET="$(echo "$SSH_LINE" | grep -oE '[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+$')"
  [ -n "$SSH_PORT" ] && [ -n "$SSH_TARGET" ] || die "couldn't parse host/port from: $SSH_LINE"
  # `-o Port=` rather than a bare port flag: ssh spells it -p and scp spells it
  # -P (lowercase -p means "preserve mtimes" to scp, and takes no argument), so a
  # shared array can only work via the config-style option both accept.
  SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o LogLevel=ERROR -o "Port=$SSH_PORT")
  log "ssh target: $SSH_TARGET:$SSH_PORT"
fi

ssh_run() { ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"; }

# ------------------------------------------------------------ ssh config ----
# The session port changes every time, so anything that wants a stable name —
# VS Code Remote-SSH, scp, rsync, and plain SSH need ~/.ssh/config
# rewritten each run. Keep it inside markers so we replace our own block and
# leave the rest of the user's config alone.
update_ssh_config() (
  local cfg="$HOME/.ssh/config"
  local begin="# >>> flowmesh-up:$SETUP_NAME >>>" end="# <<< flowmesh-up:$SETUP_NAME <<<"
  local legacy_begin="# >>> flowmesh-up >>>" legacy_end="# <<< flowmesh-up <<<"
  local lock="$HOME/.ssh/.flowmesh-up-config.lock" tmp="" waited=0
  mkdir -p "$HOME/.ssh"; touch "$cfg"

  # Two setup launchers may reach this function together. mkdir is atomic on
  # macOS, so use it as a small dependency-free lock around the read/replace.
  until mkdir "$lock" 2>/dev/null; do
    sleep 0.1; waited=$((waited + 1))
    [ "$waited" -lt 300 ] || die "timed out locking $cfg"
  done
  trap '[ -n "$tmp" ] && rm -f "$tmp"; rmdir "$lock" 2>/dev/null || true' EXIT
  tmp="$(mktemp "$HOME/.ssh/config.flowmesh.XXXXXX")"

  awk -v b="$begin" -v e="$end" -v lb="$legacy_begin" -v le="$legacy_end" '
    $0 == b || $0 == lb { skip = 1; next }
    skip && ($0 == e || $0 == le) { skip = 0; next }
    !skip { print }
  ' "$cfg" > "$tmp"

  {
    echo "$begin"
    echo "Host $SSH_ALIAS"
    echo "    HostName ${SSH_TARGET#*@}"
    echo "    User ${SSH_TARGET%@*}"
    if [ "$SSH_MODE" = proxy ]; then
      echo "    ProxyCommand flowmesh ssh proxy $TASK_ID"
    else
      echo "    Port $SSH_PORT"
    fi
    cat <<EOF
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ServerAliveInterval 30
$end
EOF
  } >> "$tmp"

  mv "$tmp" "$cfg"; tmp=""; chmod 600 "$cfg"
  log "ssh alias '$SSH_ALIAS' written to $cfg"
)
update_ssh_config

# A freshly-ready session can still be a beat away from accepting SSH.
log "waiting for the SSH port to actually accept connections"
for _ in $(seq 1 20); do
  ssh_run true 2>/dev/null && break
  sleep 3
done

# ---------------------------------------------------------------- .env ------

if [ -f .env ]; then
  log "copying .env"
  scp "${SSH_OPTS[@]}" .env "$SSH_TARGET:~/.env"
fi

# vscode_tunnel() is defined near the top of the script, alongside the
# --tunnel-only entrypoint that reuses it after the fact.
REMOTE_FOLDER="/home/flowmesh/$REPO_DIR_NAME"
CODE_BIN="${CODE_BIN:-/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code}"

if vscode_tunnel; then
  TUNNEL_OK=1
else
  TUNNEL_OK=0
  warn "continuing without a tunnel — bootstrap still runs"
fi

# ------------------------------------------------------------- source -------
# The repository is public, but the working implementation is often ahead of
# GitHub. Package the current local tree by default. Explicit excludes prevent
# credentials, real prompts, measurements, private notes, and bulky caches from
# crossing onto the box. .env is transferred separately above.
REMOTE_SOURCE_ARCHIVE="flowmesh-source.tar.gz"
if [ "$SOURCE_MODE" = local ]; then
  log "uploading current local source"
  scp "${SSH_OPTS[@]}" "$LOCAL_SOURCE_ARCHIVE" "$SSH_TARGET:~/$REMOTE_SOURCE_ARCHIVE"
  rm -f "$LOCAL_SOURCE_ARCHIVE"
fi

# ------------------------------------------------------------- bootstrap ----
# Deliberately after the tunnel: the login above wants ~20s of your attention,
# this wants ~15 minutes of none. ./bootstrap.sh with no args runs
# check -> install -> model -> serve, including its own health polling and
# ready banner, so there is nothing to duplicate here.

log "installing $SOURCE_MODE source and running bootstrap.sh (slow part: vLLM install + weights)"
if [ "$SOURCE_MODE" = local ]; then
  ssh_run bash -s <<REMOTE
set -euo pipefail
cd ~
rm -rf "$REPO_DIR_NAME"
mkdir "$REPO_DIR_NAME"
tar -xzf "$REMOTE_SOURCE_ARCHIVE" -C "$REPO_DIR_NAME"
rm -f "$REMOTE_SOURCE_ARCHIVE"
if [ -f ~/.env ]; then cp ~/.env "$REPO_DIR_NAME/.env"; fi
cd "$REPO_DIR_NAME"
VLLM_SERVER_DEV_MODE="$VLLM_SERVER_DEV_MODE" ./bootstrap.sh --setup "$SETUP_NAME"
REMOTE
else
  ssh_run bash -s <<REMOTE
set -euo pipefail
cd ~
rm -rf "$REPO_DIR_NAME"
curl -sL "$REPO_URL/archive/refs/heads/main.tar.gz" | tar xz
if [ -f ~/.env ]; then cp ~/.env "$REPO_DIR_NAME/.env"; fi
cd "$REPO_DIR_NAME"
VLLM_SERVER_DEV_MODE="$VLLM_SERVER_DEV_MODE" ./bootstrap.sh --setup "$SETUP_NAME"
REMOTE
fi

if [ -x "$CODE_BIN" ]; then
  log "opening VS Code on the box"
  "$CODE_BIN" --folder-uri "vscode-remote://tunnel+$TUNNEL_NAME$REMOTE_FOLDER" || true
else
  warn "open manually: vscode://vscode-remote/tunnel+$TUNNEL_NAME$REMOTE_FOLDER"
fi

if [ "$TUNNEL_OK" -eq 1 ]; then
  VSCODE_HINT="    vs code      https://vscode.dev/tunnel/$TUNNEL_NAME
                 (Remote Explorer -> Tunnels -> $TUNNEL_NAME; NOT the SSH entry)"
else
  VSCODE_HINT="    vs code      not authenticated — run: ./flowmesh-up.sh --setup $SETUP_NAME --tunnel-only
                 (if that hangs too, check: ssh $SSH_ALIAS 'cat ~/tunnel.log')"
fi

cat <<EOF

  done.

    setup      $SETUP_NAME
    task id    $TASK_ID
    reconnect  flowmesh ssh connect $TASK_ID
    stop       flowmesh task stop $TASK_ID

    shell        ssh $SSH_ALIAS
$VSCODE_HINT
    copy results scp -r $SSH_ALIAS:~/$REPO_DIR_NAME/results/$EXPERIMENT_NAMESPACE ./results/
    copy traces  scp -r $SSH_ALIAS:~/$REPO_DIR_NAME/traces/$EXPERIMENT_NAMESPACE ./traces/
    tunnel logs  ssh $SSH_ALIAS 'tmux attach -t tunnel'

  Leave this running. Ctrl+C stops the task and releases the GPU.

EOF

# Hold the foreground so Ctrl+C reaches the trap above. Nothing to poll — the
# box is up and everything else runs on it.
while true; do sleep 3600; done
