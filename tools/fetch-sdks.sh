#!/usr/bin/env bash
# Fetches the vendor SDKs the optional graphics features build against, into a folder OUTSIDE this repository,
# and prints the environment variables cargo needs. No SDK file is ever committed here: their licences do not
# allow it. Running this script means accepting each SDK's licence; read them first.
#
#   dlss   NVIDIA DLSS SDK (feature `dlss`), plus the Khronos Vulkan headers its build needs.
#          Licence: LICENSE.txt in the SDK (NVIDIA RTX SDKs licence).
#
# Usage: tools/fetch-sdks.sh [dlss ...]      (no argument: every SDK listed above)
# Folder: $BEVY_GFX_SDK_DIR, default ~/.local/share/bevy-graficos
set -euo pipefail

dir="${BEVY_GFX_SDK_DIR:-$HOME/.local/share/bevy-graficos}"
mkdir -p "$dir"
[[ $# -gt 0 ]] || set -- dlss

# The tag dlss_wgpu 6 is written against.
DLSS_TAG=v310.9.1

fetch_dlss() {
  if [[ ! -d "$dir/dlss/.git" ]]; then
    git clone --depth 1 -b "$DLSS_TAG" https://github.com/NVIDIA/DLSS "$dir/dlss"
  fi
  if [[ ! -d "$dir/vulkan-headers/.git" ]]; then
    git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers "$dir/vulkan-headers"
  fi
  # dlss_wgpu looks for $VULKAN_SDK/include/vulkan and $VULKAN_SDK/include/vk_video.
  mkdir -p "$dir/vulkan-sdk"
  ln -sfn ../vulkan-headers/include "$dir/vulkan-sdk/include"
  echo "export DLSS_SDK=$dir/dlss"
  echo "export VULKAN_SDK=$dir/vulkan-sdk"
}

for sdk in "$@"; do
  case "$sdk" in
    dlss) fetch_dlss ;;
    *) echo "fetch-sdks: unknown SDK '$sdk'" >&2; exit 2 ;;
  esac
done
