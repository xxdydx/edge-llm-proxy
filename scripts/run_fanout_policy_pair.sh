#!/usr/bin/env bash
# Run the same read-only Claude Code fan-out workload through isolated
# cloud-only and static-policy edgeproxy instances.
#
# Run this on the GPU dev box after bootstrap has started vLLM. It starts its
# own proxies and never touches the bootstrap-managed proxy on port 8000.
# The default is concurrent because the cloud-only condition does not use vLLM.
# Pass --mode sequential for a fully isolated timing comparison.

set -euo pipefail

repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# The remote box keeps credentials and optional proxy defaults in this
# gitignored file. It is optional so callers may instead provide variables in
# their shell environment.
if [ -f "$repo_dir/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_dir/.env"
  set +a
fi

python_bin="${PYTHON_BIN:-${VENV:-/opt/venv}/bin/python}"
claude_bin="${CLAUDE_BIN:-claude}"
vllm_url="${EDGEPROXY_VLLM_URL:-http://127.0.0.1:8001}"
upstream="${EDGEPROXY_UPSTREAM:-https://lum.id/claude}"
trace_root="${TRACE_ROOT:-$repo_dir/traces/fanout-policy-pair}"
result_root="${RESULT_ROOT:-$repo_dir/results/fanout-policy-pair}"
cloud_port="${CLOUD_PROXY_PORT:-8010}"
static_port="${STATIC_PROXY_PORT:-8011}"
run_mode="${RUN_MODE:-concurrent}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"

prompt='Your working directory is the edgeproxy/ directory. Explore only this directory and its descendants. Do not access, read, modify, or mention its parent directory or any sibling directory. Launch several parallel read-only subagents, each covering a distinct part of this directory (routing, request handling, cache and trace logic, telemetry/config, and tests). Do not write files, install packages, run network commands, or commit anything. Return one detailed consolidated report of what each agent found, including architecture, data flow, risks, and open questions.'

cloud_pid=""
static_pid=""
cloud_run_pid=""
static_run_pid=""

die() {
  echo "error: $*" >&2
  exit 1
}

stop_pid() {
  local pid="$1"
  [ -n "$pid" ] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  [ -n "$cloud_run_pid" ] && kill "$cloud_run_pid" 2>/dev/null || true
  [ -n "$static_run_pid" ] && kill "$static_run_pid" 2>/dev/null || true
  stop_pid "$cloud_pid"
  stop_pid "$static_pid"
}
trap cleanup EXIT INT TERM

wait_for_health() {
  local port="$1"
  local pid="$2"
  local deadline=$((SECONDS + 30))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    if curl --fail --silent --show-error "http://127.0.0.1:${port}/health" >/dev/null; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

start_proxy() {
  local label="$1"
  local policy="$2"
  local port="$3"
  local trace_dir="$4"
  local log_path="$5"

  mkdir -p "$trace_dir"
  "$python_bin" -m edgeproxy.server \
    --host 127.0.0.1 \
    --port "$port" \
    --upstream "$upstream" \
    --vllm-url "$vllm_url" \
    --trace-dir "$trace_dir" \
    --policy "$policy" \
    --local-cache-tracking observe \
    --cloud-cache-tracking observe \
    --max-local-tokens "${EDGEPROXY_MAX_LOCAL_TOKENS:-100000}" \
    --local-token-margin "${EDGEPROXY_LOCAL_TOKEN_MARGIN:-0.90}" \
    --shaping none \
    >"$log_path" 2>&1 &
  local pid=$!

  if ! wait_for_health "$port" "$pid"; then
    tail -40 "$log_path" >&2 || true
    die "$label edgeproxy did not become healthy"
  fi
  printf '%s' "$pid"
}

run_condition() {
  local label="$1"
  local policy="$2"
  local port="$3"
  local trace_dir="$trace_root/$label-$run_stamp"
  local proxy_log="$result_root/${label}_proxy_${run_stamp}.log"
  local claude_log="$result_root/${label}_claude_${run_stamp}.md"
  local trace_source
  local trace_dest="$trace_root/${label}_${run_stamp}.jsonl"
  local pid

  echo "==> starting $label proxy ($policy) on :$port"
  pid="$(start_proxy "$label" "$policy" "$port" "$trace_dir" "$proxy_log")"
  if [ "$label" = "cloud" ]; then
    cloud_pid="$pid"
  else
    static_pid="$pid"
  fi

  echo "==> running Claude Code through $label proxy"
  (
    cd "$repo_dir/edgeproxy"
    ANTHROPIC_BASE_URL="http://127.0.0.1:$port" \
      "$claude_bin" -p "$prompt" --dangerously-skip-permissions
  ) | tee "$claude_log"

  # Claude Code returns only after its requests have completed, but wait a
  # moment for the proxy's append-only recorder to flush the final record.
  sleep 1
  trace_source="$trace_dir/$(date -u +%F).jsonl"
  [ -s "$trace_source" ] || die "no trace written for $label condition"
  cp "$trace_source" "$trace_dest"
  echo "==> saved $label trace: $trace_dest"

  stop_pid "$pid"
  if [ "$label" = "cloud" ]; then
    cloud_pid=""
  else
    static_pid=""
  fi
}

usage() {
  cat <<'EOF'
Usage: run_fanout_policy_pair.sh [--mode concurrent|sequential]

Runs the cloud-only and static-policy conditions with the same read-only
Claude Code fan-out prompt. Default mode is concurrent. Set RUN_MODE or pass
--mode sequential to avoid overlap between the two Claude Code processes.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      [ "$#" -ge 2 ] || die "--mode requires concurrent or sequential"
      run_mode="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$run_mode" in
  concurrent|sequential) ;;
  *) die "--mode must be concurrent or sequential, got: $run_mode" ;;
esac

[ -d "$repo_dir/edgeproxy" ] || die "expected edgeproxy/ under $repo_dir"
[ -x "$python_bin" ] || die "Python not executable: $python_bin"
command -v "$claude_bin" >/dev/null || die "Claude executable not found: $claude_bin"
command -v curl >/dev/null || die "curl is required"

# `python -m edgeproxy.server` must resolve the checked-out package, regardless
# of where the caller invoked this script from.
cd "$repo_dir"

mkdir -p "$trace_root" "$result_root"

echo "==> repo:    $repo_dir"
echo "==> vLLM:    $vllm_url"
echo "==> results: $result_root"
echo "==> run:     $run_stamp"
echo "==> mode:    $run_mode"

if [ "$run_mode" = "concurrent" ]; then
  run_condition cloud cloud-only "$cloud_port" &
  cloud_run_pid=$!
  run_condition routing static "$static_port" &
  static_run_pid=$!

  set +e
  wait "$cloud_run_pid"
  cloud_status=$?
  wait "$static_run_pid"
  static_status=$?
  set -e
  cloud_run_pid=""
  static_run_pid=""

  [ "$cloud_status" -eq 0 ] || die "cloud condition failed ($cloud_status)"
  [ "$static_status" -eq 0 ] || die "routing condition failed ($static_status)"
else
  run_condition cloud cloud-only "$cloud_port"
  run_condition routing static "$static_port"
fi

echo "==> complete"
