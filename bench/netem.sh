#!/usr/bin/env bash
#
# netem.sh — kernel-level link shaping, for hosts that have NET_ADMIN.
#
# The FlowMesh session image does not: no root, no capability. So the proxy
# shapes in userspace instead (edgeproxy/shaping.py) and this exists to
# cross-check that the two agree where both are possible.
#
#   sudo ./bench/netem.sh apply home-broadband
#   sudo ./bench/netem.sh status
#   sudo ./bench/netem.sh clear
#
# Run edgeproxy with --shaping=netem so it records the condition without also
# applying its own delay — otherwise the two stack and every number is double.
#
set -euo pipefail

DEV="${DEV:-eth0}"

# Must match PRESETS in edgeproxy/shaping.py, or proxy-side and kernel-side
# runs are not comparable.
preset() {
  case "$1" in
    colo)           echo "5ms 1ms 200mbit" ;;
    branch-office)  echo "20ms 5ms 50mbit" ;;
    home-broadband) echo "30ms 10ms 5mbit" ;;
    cellular)       echo "80ms 40ms 5mbit" ;;
    *) echo "unknown preset '$1' (colo|branch-office|home-broadband|cellular)" >&2; exit 1 ;;
  esac
}

case "${1:-status}" in
  apply)
    read -r delay jitter rate <<< "$(preset "${2:?preset required}")"
    tc qdisc del dev "$DEV" root 2>/dev/null || true
    tc qdisc add dev "$DEV" root netem delay "$delay" "$jitter" distribution normal rate "$rate"
    echo "applied ${2} to $DEV: delay $delay ±$jitter, rate $rate"
    echo "start edgeproxy with --shaping=netem --link-preset ${2}"
    ;;
  clear)
    tc qdisc del dev "$DEV" root 2>/dev/null || true
    echo "cleared $DEV"
    ;;
  status)
    tc qdisc show dev "$DEV"
    ;;
  *)
    echo "usage: $0 {apply <preset>|clear|status}" >&2; exit 1 ;;
esac
