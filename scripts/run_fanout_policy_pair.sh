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
# The FlowMesh role rejects Claude Code's Opus default. Callers can override
# this with CLAUDE_MODEL, but Sonnet is the portable default for this runner.
claude_model="${CLAUDE_MODEL:-sonnet}"
vllm_url="${EDGEPROXY_VLLM_URL:-http://127.0.0.1:8001}"
upstream="${EDGEPROXY_UPSTREAM:-https://lum.id/claude}"
trace_root="${TRACE_ROOT:-$repo_dir/traces/fanout-policy-pair}"
result_root="${RESULT_ROOT:-$repo_dir/results/fanout-policy-pair}"
# Leave ports dynamic by default. Fixed ports may be supplied when required,
# but a previous interrupted run must never be mistaken for this run.
cloud_port="${CLOUD_PROXY_PORT:-}"
static_port="${STATIC_PROXY_PORT:-}"
run_mode="${RUN_MODE:-concurrent}"
run_condition_selection="${RUN_CONDITION:-pair}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"

prompt='Your working directory is the edgeproxy/ directory. Explore only this directory and its descendants. Do not access, read, modify, or mention its parent directory or any sibling directory. Launch five read-only subagents concurrently, covering: (1) routing, (2) request handling, (3) cache and trace logic, (4) telemetry/config/timing/shaping/cost, and (5) tests and testability. Use foreground Agent tool calls: emit the independent Agent calls together so they run in parallel, do not set run_in_background, and do not return while any agent is pending. Do not write files, install packages, run network commands, or commit anything. After every agent result has arrived, return one detailed consolidated Markdown report. Start the report with this exact line: <!-- FANOUT_REPORT_START -->. Then use these exact level-two headings in this order: Executive Summary; Findings from Each Agent; Architecture and Request Data Flow; Risks; Testing Gaps; Disagreements or Overlaps Between Agents; Open Questions. Do not return a launch/progress/waiting message. End the completed report with this exact line: <!-- FANOUT_REPORT_COMPLETE -->'

cloud_run_pid=""
static_run_pid=""
started_proxy_pid=""

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
  [ -n "$cloud_run_pid" ] && wait "$cloud_run_pid" 2>/dev/null || true
  [ -n "$static_run_pid" ] && wait "$static_run_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

free_port() {
  "$python_bin" -c '
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
'
}

wait_for_health() {
  local port="$1"
  local pid="$2"
  local deadline=$((SECONDS + 30))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
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
    --local-output-reserve-tokens "${EDGEPROXY_LOCAL_OUTPUT_RESERVE_TOKENS:-0}" \
    --shaping none \
    >"$log_path" 2>&1 &
  started_proxy_pid=$!

  if ! wait_for_health "$port" "$started_proxy_pid"; then
    tail -40 "$log_path" >&2 || true
    stop_pid "$started_proxy_pid"
    started_proxy_pid=""
    return 1
  fi
}

wait_for_trace() {
  local trace_source="$1"
  local deadline=$((SECONDS + 30))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ -s "$trace_source" ] && return 0
    sleep 0.2
  done
  return 1
}

validate_report() {
  local report_path="$1"
  local min_report_bytes="${MIN_REPORT_BYTES:-2000}"
  local report_bytes

  report_bytes="$(wc -c <"$report_path" | tr -d '[:space:]')"
  if [ "$report_bytes" -lt "$min_report_bytes" ]; then
    echo "error: report is only ${report_bytes} bytes; expected at least ${min_report_bytes}" >&2
    return 1
  fi
  if ! grep -Fxq '<!-- FANOUT_REPORT_COMPLETE -->' "$report_path"; then
    echo "error: report has no completion marker" >&2
    return 1
  fi
  if [ "$(sed -n '1p' "$report_path")" != '<!-- FANOUT_REPORT_START -->' ]; then
    echo "error: report does not start at the beginning marker" >&2
    return 1
  fi
  local heading
  for heading in \
    '## Executive Summary' \
    '## Findings from Each Agent' \
    '## Architecture and Request Data Flow' \
    '## Risks' \
    '## Testing Gaps' \
    '## Disagreements or Overlaps Between Agents' \
    '## Open Questions'
  do
    if ! grep -Fxq "$heading" "$report_path"; then
      echo "error: report is missing required heading: $heading" >&2
      return 1
    fi
  done
}

run_condition() (
  local label="$1"
  local policy="$2"
  local port="$3"
  local trace_dir="$trace_root/$label-$run_stamp"
  local proxy_log="$result_root/${label}_proxy_${run_stamp}.log"
  local claude_log="$result_root/${label}_claude_${run_stamp}.md"
  local claude_partial="$result_root/${label}_claude_${run_stamp}.partial.md"
  local claude_stream="$result_root/${label}_claude_${run_stamp}.stream.jsonl"
  local trace_source
  local trace_dest="$trace_root/${label}_${run_stamp}.jsonl"
  local trace_graph="$trace_root/${label}_${run_stamp}.graph.json"
  local trace_tree="$trace_root/${label}_${run_stamp}.tree.txt"
  local pid=""

  cleanup_condition() {
    stop_pid "$pid"
  }
  trap cleanup_condition EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  echo "==> starting $label proxy ($policy) on :$port"
  if ! start_proxy "$label" "$policy" "$port" "$trace_dir" "$proxy_log"; then
    tail -40 "$proxy_log" >&2 || true
    die "$label edgeproxy did not become healthy"
  fi
  pid="$started_proxy_pid"

  echo "==> running Claude Code through $label proxy"
  if ! (
    cd "$repo_dir/edgeproxy"
    ANTHROPIC_BASE_URL="http://127.0.0.1:$port" \
      "$claude_bin" -p "$prompt" --model "$claude_model" \
        --dangerously-skip-permissions --output-format stream-json --verbose
  ) | tee "$claude_stream" | "$python_bin" -m edgeproxy.report_capture \
    | tee "$claude_partial"; then
    die "$label Claude Code run failed"
  fi

  if ! validate_report "$claude_partial"; then
    die "$label Claude Code returned an incomplete report; preserved at $claude_partial"
  fi
  mv "$claude_partial" "$claude_log"
  echo "==> saved $label report: $claude_log"

  trace_source="$trace_dir/$(date -u +%F).jsonl"
  if ! wait_for_trace "$trace_source"; then
    tail -40 "$proxy_log" >&2 || true
    die "no trace written for $label condition"
  fi
  cp "$trace_source" "$trace_dest"
  echo "==> saved $label trace: $trace_dest"

  "$python_bin" -m edgeproxy.trace.graph "$trace_dest" \
    --claude-stream "$claude_stream" \
    --json-output "$trace_graph" \
    --tree-output "$trace_tree"
  [ -s "$trace_graph" ] || die "$label trace graph was not created"
  [ -s "$trace_tree" ] || die "$label trace tree was not created"
  echo "==> saved $label trace graph: $trace_graph"
  echo "==> saved $label trace tree:  $trace_tree"

  stop_pid "$pid"
  pid=""
)

usage() {
  cat <<'EOF'
Usage: run_fanout_policy_pair.sh [--mode concurrent|sequential]
                                  [--condition pair|cloud|routing]

Runs the cloud-only and static-policy conditions with the same read-only
Claude Code fan-out prompt. Default mode is concurrent. Set RUN_MODE or pass
--mode sequential to avoid overlap between the two Claude Code processes.
Use --condition routing for a single static-policy validation run.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      [ "$#" -ge 2 ] || die "--mode requires concurrent or sequential"
      run_mode="$2"
      shift 2
      ;;
    --condition)
      [ "$#" -ge 2 ] || die "--condition requires pair, cloud, or routing"
      run_condition_selection="$2"
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
case "$run_condition_selection" in
  pair|cloud|routing) ;;
  *) die "--condition must be pair, cloud, or routing, got: $run_condition_selection" ;;
esac

[ -d "$repo_dir/edgeproxy" ] || die "expected edgeproxy/ under $repo_dir"
[ -x "$python_bin" ] || die "Python not executable: $python_bin"
command -v "$claude_bin" >/dev/null || die "Claude executable not found: $claude_bin"
command -v curl >/dev/null || die "curl is required"

# `python -m edgeproxy.server` must resolve the checked-out package, regardless
# of where the caller invoked this script from.
cd "$repo_dir"

[ -n "$cloud_port" ] || cloud_port="$(free_port)"
[ -n "$static_port" ] || static_port="$(free_port)"
[ "$cloud_port" != "$static_port" ] || die "cloud and static proxy ports must differ"

mkdir -p "$trace_root" "$result_root"

echo "==> repo:    $repo_dir"
echo "==> vLLM:    $vllm_url"
echo "==> model:   $claude_model"
echo "==> results: $result_root"
echo "==> run:     $run_stamp"
echo "==> mode:    $run_mode"
echo "==> condition: $run_condition_selection"
echo "==> ports:   cloud=$cloud_port routing=$static_port"

if [ "$run_condition_selection" = "cloud" ]; then
  run_condition cloud cloud-only "$cloud_port"
elif [ "$run_condition_selection" = "routing" ]; then
  run_condition routing static "$static_port"
elif [ "$run_mode" = "concurrent" ]; then
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
