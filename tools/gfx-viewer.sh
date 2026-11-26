#!/usr/bin/env bash
# Opens gfx_viewer, the interactive viewer of the graphics ladder, on this screen: builds it (release) with the
# features the ladder ships with and runs it, passing every argument through.
#
#   tools/gfx-viewer.sh                                  the testbed's interior, preset Auto
#   tools/gfx-viewer.sh --scene sponza --preset rt-full  a scene of tools/fetch-scenes.sh, a preset
#   tools/gfx-viewer.sh --scene path/to/file.glb         any glTF file
#   tools/gfx-viewer.sh --help                           every flag
#
# It exports the SDK paths of tools/fetch-sdks.sh (DLSS finds its libraries through $DLSS_SDK; NRD builds from
# $NRD_SDK). The first run fetches the SDKs (accepting their licences) and builds for a while; later runs start
# in seconds. Press H in the window for the keys. Set GFX_VIEWER_FEATURES to build with other features.
set -euo pipefail

cd "$(dirname "$0")/.."
eval "$(tools/fetch-sdks.sh dlss nrd)"
export BEVY_ASSET_ROOT="$PWD"

# The ladder's full set on Linux: Solari, the light LUTs, the ladder, NRD, DLSS and the baker; plus the viewer's
# free camera and JPEG textures (most glTF scenes use them). `fsr` is left out: FSR runs only on Windows with
# AMD's DLLs (docs/graphics-shipping.md), and `Auto` never picks it.
features="${GFX_VIEWER_FEATURES:-bevy_solari,area_light_luts,bevy_graphics,nrd,dlss,bevy_baked_gi_bake,free_camera,jpeg}"

# A build is heavy: through the machine's build lock when this checkout has it (it waits for a free slot).
build=(cargo build --release --example gfx_viewer --features "$features")
if [[ -x tools/gfx-build-lock.sh ]]; then
  tools/gfx-build-lock.sh "${build[@]}"
else
  "${build[@]}"
fi
exec target/release/examples/gfx_viewer "$@"
