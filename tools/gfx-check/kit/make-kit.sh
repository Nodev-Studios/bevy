#!/usr/bin/env bash
# Builds the hardware-test kit for a person with hardware the project lacks (plan section 6): the testbed and
# gfx-check with dlss, nrd and fsr, the vendor libraries they load at run time, the licences, a launcher and the
# guide in Spanish, packed into one archive. The archive is never committed: it holds NVIDIA's and AMD's binaries.
#
# Usage: tools/gfx-check/kit/make-kit.sh linux|windows OUT_DIR
#   eval "$(tools/fetch-sdks.sh dlss nrd fsr)" first: DLSS_SDK, NRD_SDK and VULKAN_SDK build it, DLSS_SDK and FSR_SDK
#   hold the libraries it ships.
#   NO_BUILD=1      pack what target/kit-<os> already holds
#   BUILD_WRAPPER   a command each cargo build runs under, e.g. "tools/gfx-build-lock.sh"
#   JOBS            cargo's -j
#
# Writes OUT_DIR/gfx-prueba-linux.tar.gz or OUT_DIR/gfx-prueba-windows.zip, and beside it the archive's file list.
set -euo pipefail

os="${1:-}"
out="${2:-}"
[[ "$os" == linux || "$os" == windows ]] && [[ -n "$out" ]] || {
  echo "usage: $0 linux|windows OUT_DIR" >&2
  exit 2
}
: "${DLSS_SDK:?eval \"\$(tools/fetch-sdks.sh dlss nrd fsr)\" first}"
: "${NRD_SDK:?eval \"\$(tools/fetch-sdks.sh dlss nrd fsr)\" first}"
[[ "$os" == linux ]] || : "${FSR_SDK:?eval \"\$(tools/fetch-sdks.sh dlss nrd fsr)\" first}"

root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
features=bevy_solari,area_light_luts,bevy_graphics,dlss,nrd,fsr

# Its own target dir: the remapped flags below would otherwise rebuild the shared one from scratch each time.
export CARGO_TARGET_DIR="$root/target/kit-$os"
# Without these, the build machine's home and checkout are compiled into every panic message, and from there
# reach the friend's logs. Later prefixes win, so the specific ones come after the home.
export RUSTFLAGS="--remap-path-prefix=$HOME=/home/builder --remap-path-prefix=$HOME/.cargo=/cargo \
--remap-path-prefix=$HOME/.rustup=/rustup --remap-path-prefix=$root=/bevy"

if [[ "$os" == linux ]]; then
  build=(cargo build)
  bin="$CARGO_TARGET_DIR/release"
  exe=""
else
  build=(cargo xwin build --target x86_64-pc-windows-msvc)
  bin="$CARGO_TARGET_DIR/x86_64-pc-windows-msvc/release"
  exe=".exe"
fi
if [[ -z "${NO_BUILD:-}" ]]; then
  if [[ "$os" == linux ]]; then
    # The build machine's glibc is newer than most people's: link the few libm functions it re-versioned to
    # their oldest versions (glibc-compat.c). Only the testbed's own link changes, so its dependencies stay built.
    compat="$CARGO_TARGET_DIR/glibc-compat.o"
    mkdir -p "$CARGO_TARGET_DIR"
    cc -c -O2 -fPIC tools/gfx-check/kit/glibc-compat.c -o "$compat"
    link=(-C "link-arg=$compat")
    for name in $(sed -n 's/^COMPAT([a-z]*, \([a-z0-9]*\),.*/\1/p' tools/gfx-check/kit/glibc-compat.c); do
      link+=(-C "link-arg=-Wl,--wrap=$name")
    done
    # shellcheck disable=SC2086 # BUILD_WRAPPER is a command line on purpose.
    ${BUILD_WRAPPER:-} cargo rustc ${JOBS:+-j "$JOBS"} --release --example gfx_testbed --features "$features" \
      -- "${link[@]}"
  else
    # shellcheck disable=SC2086
    ${BUILD_WRAPPER:-} "${build[@]}" ${JOBS:+-j "$JOBS"} --release --example gfx_testbed --features "$features"
  fi
  # shellcheck disable=SC2086
  ${BUILD_WRAPPER:-} "${build[@]}" ${JOBS:+-j "$JOBS"} --release -p gfx-check
fi
if [[ "$os" == linux ]]; then
  # glibc 2.35 is Ubuntu 22.04's. A newer symbol means glibc-compat.c needs another line.
  newest="$(objdump -T "$bin/examples/gfx_testbed" "$bin/gfx-check" | grep -v ' w ' | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
  if [[ "$(printf '%s\n' "$newest" GLIBC_2.35 | sort -V | tail -1)" != GLIBC_2.35 ]]; then
    objdump -T "$bin/examples/gfx_testbed" | grep -F "($newest)" >&2
    echo "the testbed needs $newest: add these functions to tools/gfx-check/kit/glibc-compat.c" >&2
    exit 1
  fi
fi

name="gfx-prueba-$os"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
kit="$stage/$name"
mkdir -p "$kit/assets/textures" "$kit/golden" "$kit/licencias"

cp "$bin/examples/gfx_testbed$exe" "$bin/gfx-check$exe" "$kit/"
# The only file the testbed's two scenes load; everything else is built in code or embedded.
cp assets/textures/uv_checker_bw.png "$kit/assets/textures/"
cp tools/gfx-check/budgets.toml "$kit/"
cp tools/gfx-check/golden/* "$kit/golden/"
cp docs/gfx/PRUEBA-HARDWARE.md "$kit/LEEME.txt"

# Libraries loaded at run time, so no import table lists them: NGX loads DLSS's from the program's folder.
if [[ "$os" == linux ]]; then
  runtime=("$DLSS_SDK"/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.* "$DLSS_SDK"/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.*)
else
  runtime=("$DLSS_SDK"/lib/Windows_x86_64/rel/nvngx_dlss.dll "$DLSS_SDK"/lib/Windows_x86_64/rel/nvngx_dlssd.dll
    "$FSR_SDK"/Kits/FidelityFX/signedbin/amd_fidelityfx_loader_vk.dll
    "$FSR_SDK"/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll)
fi
cp "${runtime[@]}" "$kit/"

cp LICENSE-MIT "$kit/licencias/bevy-LICENSE-MIT.txt"
cp LICENSE-APACHE "$kit/licencias/bevy-LICENSE-APACHE.txt"
cp "$DLSS_SDK/LICENSE.txt" "$kit/licencias/NVIDIA-DLSS-SDK-LICENSE.txt"
cp "$NRD_SDK/LICENSE.txt" "$kit/licencias/NVIDIA-NRD-LICENSE.txt"
{
  echo "This kit is a test build of a Bevy fork's graphics, sent to one person to run on their own hardware."
  echo
  echo "It uses NVIDIA DLSS (nvngx_dlss / libnvidia-ngx-dlss: Super Resolution and Ray Reconstruction) and"
  echo "NVIDIA Real-time Denoisers (NRD, linked into gfx_testbed), under the NVIDIA RTX SDKs licence in"
  echo "NVIDIA-DLSS-SDK-LICENSE.txt and NVIDIA-NRD-LICENSE.txt. NVIDIA, RTX and DLSS are trademarks of"
  echo "NVIDIA Corporation."
  if [[ "$os" == windows ]]; then
    echo
    echo "It uses AMD FidelityFX Super Resolution (amd_fidelityfx_loader_vk.dll, amd_fidelityfx_upscaler_vk.dll),"
    echo "redistributed in binary form under the licence in AMD-FidelityFX-LICENSE.md, which also holds"
    echo "AMD's copyright notice and disclaimers. AMD and FidelityFX are trademarks of Advanced Micro Devices, Inc."
  fi
  echo
  echo "Bevy is dual-licensed under MIT or Apache-2.0 (bevy-LICENSE-MIT.txt, bevy-LICENSE-APACHE.txt)."
} > "$kit/licencias/AVISOS.txt"
[[ "$os" == linux ]] || cp "$FSR_SDK/docs/license.md" "$kit/licencias/AMD-FidelityFX-LICENSE.md"

if [[ "$os" == linux ]]; then
  cat > "$kit/probar.sh" <<'EOF'
#!/bin/sh
# Prueba de graficos: ejecuta todas las pruebas y deja resultados-<fecha>.zip en esta carpeta.
cd "$(dirname "$0")" || exit 1
exec ./gfx-check friend "$@"
EOF
  chmod +x "$kit/probar.sh" "$kit/gfx_testbed" "$kit/gfx-check"
else
  # CRLF and ASCII only: cmd.exe reads a .bat in the console's code page.
  printf '%s\r\n' '@echo off' \
    'rem Prueba de graficos: ejecuta todas las pruebas y deja resultados-<fecha>.zip en esta carpeta.' \
    'cd /d "%~dp0"' 'gfx-check.exe friend %*' 'echo.' 'pause' > "$kit/PROBAR.bat"
fi

(cd "$kit" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt)
if [[ "$os" == linux ]]; then
  archive="$out/$name.tar.gz"
  tar -C "$stage" --owner=0 --group=0 --numeric-owner -czf "$archive" "$name"
else
  archive="$out/$name.zip"
  rm -f "$archive"
  (cd "$stage" && zip -qr -X "$archive" "$name")
fi
(cd "$kit" && find . -type f -printf '%10s  %P\n' | sort -k2) > "$out/$name.files.txt"
ls -l "$archive"
