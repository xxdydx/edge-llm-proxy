# trace-up.sh — start the recorder and point this shell at it.
#
#   source ./trace-up.sh        (or:  . ./trace-up.sh)
#   claude
#
# MUST be sourced, not executed: a child process cannot export ANTHROPIC_BASE_URL
# back into your shell. Deliberately no `set -e` for the same reason — an error
# would kill the shell you sourced it from.

_tu_port="${EDGEPROXY_PORT:-8765}"
_tu_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" 2>/dev/null && pwd)"
_tu_ok=1

# --- refuse to run unsourced, since it would silently do nothing useful -------
_tu_sourced=0
if [ -n "${ZSH_VERSION:-}" ]; then
  case "${ZSH_EVAL_CONTEXT:-}" in *:file*) _tu_sourced=1 ;; esac
elif [ -n "${BASH_VERSION:-}" ]; then
  [ "${BASH_SOURCE[0]}" != "$0" ] && _tu_sourced=1
fi
if [ "$_tu_sourced" -ne 1 ]; then
  echo "run it as:  source ./trace-up.sh" >&2
  exit 1
fi

# --- credentials -------------------------------------------------------------
# zsh's `.` searches $PATH when the path has no slash, hence the explicit ./
if [ -f "$_tu_dir/.env" ]; then
  set -a; . "$_tu_dir/.env"; set +a
else
  echo "[!] no .env found in $_tu_dir" >&2
  _tu_ok=0
fi

if [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
  echo "[!] ANTHROPIC_AUTH_TOKEN is not set — claude will get a 401" >&2
  _tu_ok=0
fi

# --- recorder ----------------------------------------------------------------
if curl -sf "http://localhost:$_tu_port/health" >/dev/null 2>&1; then
  echo "==> edgeproxy already running on :$_tu_port"
else
  echo "==> starting edgeproxy on :$_tu_port"
  mkdir -p "$_tu_dir/logs"
  ( cd "$_tu_dir" && nohup .venv/bin/edgeproxy \
      --port "$_tu_port" --trace-dir "$_tu_dir/traces" \
      > "$_tu_dir/logs/proxy.log" 2>&1 & echo $! > "$_tu_dir/logs/proxy.pid" )

  _tu_waited=0
  until curl -sf "http://localhost:$_tu_port/health" >/dev/null 2>&1; do
    sleep 1; _tu_waited=$((_tu_waited + 1))
    if [ "$_tu_waited" -ge 15 ]; then
      echo "[!] edgeproxy did not come up — see $_tu_dir/logs/proxy.log" >&2
      _tu_ok=0
      break
    fi
  done
fi

# --- point this shell at it --------------------------------------------------
export ANTHROPIC_BASE_URL="http://localhost:$_tu_port"

if [ "$_tu_ok" -eq 1 ]; then
  echo "==> ANTHROPIC_BASE_URL=$ANTHROPIC_BASE_URL"
  echo "==> token ${ANTHROPIC_AUTH_TOKEN:0:12}...  traces -> $_tu_dir/traces"
  echo
  echo "    now type:  claude"
  echo "    inspect:   .venv/bin/python -m edgeproxy.trace.inspect traces/*.jsonl"
  echo "    stop:      kill \$(cat logs/proxy.pid)"
else
  echo "[!] setup incomplete — see the errors above" >&2
fi

unset _tu_port _tu_dir _tu_ok _tu_sourced _tu_waited
