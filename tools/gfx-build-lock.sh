#!/usr/bin/env bash
# Runs a build command while holding one of two machine-wide build slots, so parallel agents do not start more
# than two heavy cargo builds at once.
#
# Usage: tools/gfx-build-lock.sh cargo build --release --example gfx_testbed
set -euo pipefail
[[ $# -gt 0 ]] || { echo "usage: $0 <command...>" >&2; exit 2; }

lock_dir="${BEVY_GFX_LOCK_DIR:-${XDG_RUNTIME_DIR:-/tmp}/bevy-gfx}"
mkdir -p "$lock_dir"
while true; do
  for slot in 1 2; do
    exec 9>"$lock_dir/build-$slot.lock"
    if flock -n 9; then
      exec "$@"
    fi
  done
  sleep 2
done
