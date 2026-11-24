#!/usr/bin/env bash
# Fetches the vendor SDKs the optional graphics features build against, into a folder OUTSIDE this repository,
# and prints the environment variables cargo needs. No SDK file is ever committed here: their licences do not
# allow it. Running this script means accepting each SDK's licence; read them first.
#
#   dlss   NVIDIA DLSS SDK (feature `dlss`), plus the Khronos Vulkan headers its build needs.
#          Licence: LICENSE.txt in the SDK (NVIDIA RTX SDKs licence).
#   nrd    NVIDIA Real-Time Denoisers (vendor/nrd_wgpu). Source only: nrd_wgpu's build script builds it with
#          CMake, which downloads NRD's own dependencies (ShaderMake, MathLib, a DXC binary) into its build folder.
#          Licence: LICENSE.txt in the SDK (NVIDIA RTX SDKs licence).
#   fsr    AMD FidelityFX SDK (vendor/fsr_wgpu): the FidelityFX API headers and the signed Vulkan DLLs
#          (`amd_fidelityfx_loader_vk.dll`, `amd_fidelityfx_upscaler_vk.dll`) that run FSR 4 / 3.1 on Windows.
#          A sparse, blobless checkout of only those paths and the documentation, plus the Khronos Vulkan
#          headers. Building needs none of it (the DLLs are loaded at run time); the layout test and a
#          Windows bundle do.
#          Licence: Kits/FidelityFX/docs/license.md in the SDK (headers MIT; the DLLs binary-redistributable).
#
# Usage: tools/fetch-sdks.sh [dlss|nrd|fsr ...]  (no argument: every SDK listed above)
# Folder: $BEVY_GFX_SDK_DIR, default ~/.local/share/bevy-graficos
set -euo pipefail

dir="${BEVY_GFX_SDK_DIR:-$HOME/.local/share/bevy-graficos}"
mkdir -p "$dir"
[[ $# -gt 0 ]] || set -- dlss nrd fsr

# The tag dlss_wgpu 6 is written against.
DLSS_TAG=v310.9.1

fetch_vulkan_headers() {
  if [[ ! -d "$dir/vulkan-headers/.git" ]]; then
    git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers "$dir/vulkan-headers"
  fi
  # dlss_wgpu looks for $VULKAN_SDK/include/vulkan and $VULKAN_SDK/include/vk_video.
  mkdir -p "$dir/vulkan-sdk"
  ln -sfn ../vulkan-headers/include "$dir/vulkan-sdk/include"
  echo "export VULKAN_SDK=$dir/vulkan-sdk"
}

fetch_dlss() {
  if [[ ! -d "$dir/dlss/.git" ]]; then
    git clone --depth 1 -b "$DLSS_TAG" https://github.com/NVIDIA/DLSS "$dir/dlss"
  fi
  fetch_vulkan_headers
  echo "export DLSS_SDK=$dir/dlss"
}

# The tag nrd_wgpu's shim and settings are written against.
NRD_TAG=v4.18.0

fetch_nrd() {
  if [[ ! -d "$dir/nrd/.git" ]]; then
    git clone --depth 1 -b "$NRD_TAG" https://github.com/NVIDIA-RTX/NRD "$dir/nrd"
  fi
  echo "export NRD_SDK=$dir/nrd"
}

# The tag fsr_wgpu's bindings and layout tests are written against: FSR SDK 2.4.0 (2026-10-28), FSR 4.1.2 on Vulkan.
FSR_TAG=v2.4.0

fetch_fsr() {
  if [[ ! -d "$dir/fsr/.git" ]]; then
    git clone --depth 1 --filter=blob:none --no-checkout -b "$FSR_TAG" \
      https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK "$dir/fsr"
    git -C "$dir/fsr" sparse-checkout set --no-cone \
      /readme.md /3rdpartynotice.md /docs/license.md \
      /Kits/FidelityFX/readme.md /Kits/FidelityFX/docs/ \
      /Kits/FidelityFX/api/include/ /Kits/FidelityFX/upscalers/include/ \
      /Kits/FidelityFX/signedbin/amd_fidelityfx_loader_vk.dll \
      /Kits/FidelityFX/signedbin/amd_fidelityfx_loader_vk.lib \
      /Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll
    git -C "$dir/fsr" checkout "$FSR_TAG"
  fi
  # Only fsr_wgpu's layout test reads headers; it needs Vulkan's besides the SDK's.
  fetch_vulkan_headers
  echo "export FSR_SDK=$dir/fsr"
}

for sdk in "$@"; do
  case "$sdk" in
    dlss) fetch_dlss ;;
    nrd) fetch_nrd ;;
    fsr) fetch_fsr ;;
    *) echo "fetch-sdks: unknown SDK '$sdk'" >&2; exit 2 ;;
  esac
done
