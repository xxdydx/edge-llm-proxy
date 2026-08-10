#!/usr/bin/env bash
#
# flowmesh-up.sh — one command from a cold Mac to a running vLLM + edgeproxy.
#
# Submits ssh-workflow.yaml, waits for the session, pushes .env over, then
# runs the box-side bootstrap.sh non-interactively. Run this from the repo
# root on your Mac (not on the box):
#
#   ./flowmesh-up.sh
#
set -euo pipefail

WORKFLOW="${WORKFLOW:-ssh-workflow.yaml}"
REPO_URL="${REPO_URL:-https://github.com/xxdydx/edge-llm-proxy}"
REPO_DIR_NAME="edge-llm-proxy-main"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

command -v flowmesh >/dev/null || die "flowmesh CLI not found — see setup-instructions.md"
[ -f "$WORKFLOW" ] || die "$WORKFLOW not found — run this from the repo root"
[ -f .env ] || warn ".env not found locally — edgeproxy will come up with no upstream token"

# ------------------------------------------------------------------ submit --

log "submitting $WORKFLOW"
SUBMIT_OUT="$(flowmesh workflow submit "$WORKFLOW")"
echo "$SUBMIT_OUT"
TASK_ID="$(echo "$SUBMIT_OUT" | grep -oE 'tsk-[a-f0-9-]+' | head -1)"
[ -n "$TASK_ID" ] || die "couldn't parse a task ID out of 'flowmesh workflow submit' output above"
log "task: $TASK_ID"

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
for _ in $(seq 1 120); do
  grep -q '^Connecting:' "$CONNECT_LOG" && break
  sleep 5
done
kill "$CONNECT_PID" 2>/dev/null || true
wait "$CONNECT_PID" 2>/dev/null || true

SSH_LINE="$(grep '^Connecting:' "$CONNECT_LOG" | sed 's/^Connecting: //')"
[ -n "$SSH_LINE" ] || die "session never became ready — see $CONNECT_LOG"
rm -f "$CONNECT_LOG"

SSH_PORT="$(echo "$SSH_LINE" | grep -oE -- '-p [0-9]+' | awk '{print $2}')"
SSH_TARGET="$(echo "$SSH_LINE" | grep -oE '[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+$')"
[ -n "$SSH_PORT" ] && [ -n "$SSH_TARGET" ] || die "couldn't parse host/port from: $SSH_LINE"
log "ssh target: $SSH_TARGET:$SSH_PORT"

# `-o Port=` rather than a bare port flag: ssh spells it -p and scp spells it
# -P (lowercase -p means "preserve mtimes" to scp, and takes no argument), so a
# shared array can only work via the config-style option both accept.
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
          -o LogLevel=ERROR -o "Port=$SSH_PORT")

ssh_run() { ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"; }

# ------------------------------------------------------------ ssh config ----
# The session port changes every time, so anything that wants a stable name —
# VS Code Remote-SSH, scp, rsync, plain `ssh fmbox` — needs ~/.ssh/config
# rewritten each run. Keep it inside markers so we replace our own block and
# leave the rest of the user's config alone.
update_ssh_config() {
  local cfg="$HOME/.ssh/config"
  local begin="# >>> flowmesh-up >>>" end="# <<< flowmesh-up <<<"
  mkdir -p "$HOME/.ssh"; touch "$cfg"

  awk -v b="$begin" -v e="$end" '
    $0 == b { skip = 1 } !skip { print } $0 == e { skip = 0 }
  ' "$cfg" > "$cfg.tmp"

  cat >> "$cfg.tmp" <<EOF
$begin
Host fmbox
    HostName ${SSH_TARGET#*@}
    User ${SSH_TARGET%@*}
    Port $SSH_PORT
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ServerAliveInterval 30
$end
EOF

  mv "$cfg.tmp" "$cfg"; chmod 600 "$cfg"
  log "ssh alias 'fmbox' written to $cfg"
}
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

# ------------------------------------------------------------- bootstrap ----
# ./bootstrap.sh with no args already runs check -> install -> model -> serve,
# including its own health-check polling and a ready banner — no need to
# duplicate any of that here.

log "fetching repo and running bootstrap.sh (this is the slow part: weights + vLLM install)"
ssh_run bash -s <<REMOTE
set -euo pipefail
cd ~
rm -rf "$REPO_DIR_NAME"
curl -sL "$REPO_URL/archive/refs/heads/main.tar.gz" | tar xz
if [ -f ~/.env ]; then cp ~/.env "$REPO_DIR_NAME/.env"; fi
cd "$REPO_DIR_NAME"
./bootstrap.sh
REMOTE

# ------------------------------------------------------------- vs code ------
# Remote-SSH cannot work here: the FlowMesh entrypoint writes
#   AllowTcpForwarding no
# into /etc/ssh/sshd_config.d/ at *container start*, so it is re-applied every
# session no matter what the image contains. Tunnels dial out instead of needing
# an inbound forwarded port, so they are unaffected.
#
# The tunnel login lives in ~/.vscode/cli/token.json on the box and dies with
# the session. We stash it on the laptop after the first login and restore it on
# every later run — otherwise you would re-authenticate with GitHub every single
# time. Only the two small credential files are stashed; ~/.vscode/cli/servers/
# is a ~100MB download that re-fetches on its own.

VSCODE_STASH="$HOME/.flowmesh/vscode-cli.tar.gz"
REMOTE_FOLDER="/home/flowmesh/$REPO_DIR_NAME"
CODE_BIN="${CODE_BIN:-/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code}"

vscode_tunnel() {
  log "installing VS Code CLI on the box (if absent)"
  ssh_run 'set -e
    if [ ! -x ~/code ] && ! ls ~/.vscode-server/code-* >/dev/null 2>&1; then
      curl -sL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" \
        -o /tmp/vscode-cli.tgz && tar -xf /tmp/vscode-cli.tgz -C ~ && chmod +x ~/code
    fi' || { warn "could not install the VS Code CLI"; return 1; }

  if [ -f "$VSCODE_STASH" ]; then
    log "restoring saved VS Code login"
    ssh_run 'tar xzf - -C ~' < "$VSCODE_STASH" || warn "could not restore login"
  fi

  log "starting tunnel"
  ssh_run 'CLI=$(ls ~/code ~/.vscode-server/code-* 2>/dev/null | head -1)
    tmux kill-session -t tunnel 2>/dev/null || true
    rm -f ~/tunnel.log
    tmux new -d -s tunnel "$CLI tunnel --accept-server-license-terms --name fmbox >~/tunnel.log 2>&1"'

  local waited=0
  while [ "$waited" -lt 90 ]; do
    if ssh_run 'grep -qi "devtunnels.ms\|Open this link" ~/tunnel.log 2>/dev/null'; then
      log "tunnel is up"
      log "stashing login for next session"
      mkdir -p "$(dirname "$VSCODE_STASH")"; chmod 700 "$(dirname "$VSCODE_STASH")"
      ssh_run 'tar czf - -C ~ .vscode/cli/token.json .vscode/cli/code_tunnel.json 2>/dev/null' \
        > "$VSCODE_STASH" && chmod 600 "$VSCODE_STASH" || true

      if [ -x "$CODE_BIN" ]; then
        log "opening VS Code"
        "$CODE_BIN" --folder-uri "vscode-remote://tunnel+fmbox$REMOTE_FOLDER" || true
      else
        warn "VS Code CLI not found at: $CODE_BIN"
        warn "open manually: vscode://vscode-remote/tunnel+fmbox$REMOTE_FOLDER"
      fi
      return 0
    fi

    if ssh_run 'grep -q "github.com/login/device" ~/tunnel.log 2>/dev/null'; then
      warn "one-time GitHub login required — do this, then re-run ./flowmesh-up.sh:"
      ssh_run 'cat ~/tunnel.log' >&2
      warn "(the login is stashed afterwards, so this is only needed once)"
      return 1
    fi

    sleep 3; waited=$((waited + 3))
  done

  warn "tunnel did not come up in ${waited}s — check: ssh fmbox 'cat ~/tunnel.log'"
  return 1
}

vscode_tunnel || true

cat <<EOF

  done.

    task id    $TASK_ID
    reconnect  flowmesh ssh connect $TASK_ID
    stop       flowmesh task stop $TASK_ID

    connect      ssh fmbox
    vs code      Remote-SSH: Connect to Host... -> fmbox   (auto-forwards ports)
    copy files   scp fmbox:~/edge-llm-proxy-main/results/* ./results/

    or forward the proxy port manually:
      ssh -N -L 8000:localhost:8000 fmbox

EOF
