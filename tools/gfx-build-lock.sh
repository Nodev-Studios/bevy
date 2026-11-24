#!/usr/bin/env bash
# Runs a build command while holding one of two machine-wide build slots, so parallel agents do not start more
# than two heavy cargo builds at once.
#
# The build runs in its own systemd scope with a memory cap and no swap (BEVY_GFX_BUILD_MEMORY, default 10G), so on
# a machine short of memory the out-of-memory killer stops the build alone, not the agent or the desktop.
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
      exec systemd-run --user --scope --quiet -p MemoryMax="${BEVY_GFX_BUILD_MEMORY:-10G}" -p MemorySwapMax=0 "$@"
    fi
  done
  sleep 2
done
