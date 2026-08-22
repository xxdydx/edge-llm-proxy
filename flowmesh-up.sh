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

# The script stays in the foreground for the life of the box, and Ctrl+C
# releases the task. Deliberately NOT trapped on EXIT: if bootstrap fails we
# want the box left alive to debug, not torn down. Set NO_AUTOSTOP=1 to keep the
# task running after Ctrl+C.
cleanup() {
  trap - INT TERM HUP
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

  {
    echo "$begin"
    echo "Host fmbox"
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
  } >> "$cfg.tmp"

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
REMOTE_FOLDER="/home/flowmesh/$REPO_DIR_NAME"
CODE_BIN="${CODE_BIN:-/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code}"

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
        tmux new -d -s tunnel "$CLI tunnel --accept-server-license-terms --name fmbox >~/tunnel.log 2>&1"'
      tunnel_started=1
    fi

    if ssh_run 'grep -qi "devtunnels.ms\|Open:" ~/tunnel.log 2>/dev/null'; then
      log "tunnel is up: https://vscode.dev/tunnel/fmbox"
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

  warn "tunnel did not come up in ${waited}s — check: ssh fmbox 'cat ~/tunnel.log'"
  return 1
}

vscode_tunnel || warn "continuing without a tunnel — bootstrap still runs"

# ------------------------------------------------------------- bootstrap ----
# Deliberately after the tunnel: the login above wants ~20s of your attention,
# this wants ~15 minutes of none. ./bootstrap.sh with no args runs
# check -> install -> model -> serve, including its own health polling and
# ready banner, so there is nothing to duplicate here.

log "fetching repo and running bootstrap.sh (slow part: vLLM install + weights)"
ssh_run bash -s <<REMOTE
set -euo pipefail
cd ~
rm -rf "$REPO_DIR_NAME"
curl -sL "$REPO_URL/archive/refs/heads/main.tar.gz" | tar xz
if [ -f ~/.env ]; then cp ~/.env "$REPO_DIR_NAME/.env"; fi
cd "$REPO_DIR_NAME"
./bootstrap.sh
REMOTE

if [ -x "$CODE_BIN" ]; then
  log "opening VS Code on the box"
  "$CODE_BIN" --folder-uri "vscode-remote://tunnel+fmbox$REMOTE_FOLDER" || true
else
  warn "open manually: vscode://vscode-remote/tunnel+fmbox$REMOTE_FOLDER"
fi

cat <<EOF

  done.

    task id    $TASK_ID
    reconnect  flowmesh ssh connect $TASK_ID
    stop       flowmesh task stop $TASK_ID

    shell        ssh fmbox
    vs code      https://vscode.dev/tunnel/fmbox
                 (Remote Explorer -> Tunnels -> fmbox; NOT the SSH entry)
    copy files   scp fmbox:~/$REPO_DIR_NAME/results/* ./results/
    tunnel logs  ssh fmbox 'tmux attach -t tunnel'

  Leave this running. Ctrl+C stops the task and releases the GPU.

EOF

# Hold the foreground so Ctrl+C reaches the trap above. Nothing to poll — the
# box is up and everything else runs on it.
while true; do sleep 3600; done
