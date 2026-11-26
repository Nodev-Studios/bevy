#!/usr/bin/env bash
# Runs a command inside a virtual KWin session, so its window never shows on the real desktop, and holds the
# machine-wide GPU lock while it runs: one GPU run at a time, however many agents work in parallel.
#
# Usage: tools/gfx-headless.sh [--seconds N] [--size WxH] [--memory 14G] <command...>
#   tools/gfx-headless.sh --seconds 120 --size 2560x1440 cargo run --release --example solari --features bevy_solari
#
# Exit code: the command's own. If the command is still running after --seconds it is stopped and the exit code is 124.
# Pick the GPU with WGPU_ADAPTER_NAME (a substring of the adapter's name, e.g. NVIDIA or AMD).
set -euo pipefail

seconds=300
size=1280x720
memory=14G
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --seconds) seconds="$2"; shift 2 ;;
    --size) size="$2"; shift 2 ;;
    --memory) memory="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "gfx-headless: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ $# -gt 0 ]] || { echo "usage: $0 [--seconds N] [--size WxH] [--memory 14G] <command...>" >&2; exit 2; }
width="${size%x*}"
height="${size#*x}"

lock_dir="${BEVY_GFX_LOCK_DIR:-${XDG_RUNTIME_DIR:-/tmp}/bevy-gfx}"
mkdir -p "$lock_dir"
exec 9>"$lock_dir/gpu.lock"
flock 9

socket="wl-gfx-headless-$$"
inner=$(mktemp)
trap 'rm -f "$inner"' EXIT
{
  echo '#!/usr/bin/env bash'
  echo "cd $(printf %q "$PWD")"
  echo "unset DISPLAY"
  echo "export WAYLAND_DISPLAY=$socket"
  printf 'exec'
  printf ' %q' "$@"
  echo
} > "$inner"
chmod +x "$inner"

# A memory cap with no swap: a runaway run is killed alone instead of the whole desktop session.
# The session gets its own D-Bus bus, and everything it started lives in one named scope that is stopped when the
# run ends: the desktop portal and secret service it activates outlive their bus otherwise (1 500 after two days).
unit="gfx-headless-$$-$RANDOM"
trap 'rm -f "$inner"; systemctl --user stop "$unit.scope" 2>/dev/null || true' EXIT
status=0
systemd-run --user --scope --quiet --unit "$unit" -p MemoryMax="$memory" -p MemorySwapMax=0 \
  dbus-run-session -- timeout "$seconds" kwin_wayland --virtual --no-lockscreen --no-global-shortcuts \
  --socket "$socket" --width "$width" --height "$height" --exit-with-session "$inner" || status=$?
exit "$status"
