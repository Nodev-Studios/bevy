#!/usr/bin/env bash
# Compiles the seven FSR 3.1 upscaler passes in shaders/ to the SPIR-V in spv/, for the one configuration this
# crate runs: HDR colour, reverse-Z depth, render-resolution motion vectors without jitter, 32-bit maths, no
# wave operations, the reference Lanczos reprojection and no sharpening.
#
# The defines are wgpu-ffx's (`wgpu-ffx-shaders-spv/src/fsr3upscaler/perm.toml` at 6fe706406), with each
# permutation fixed to the value above. The SPIR-V is committed so that a build needs no shader compiler; rerun this
# after changing a shader and commit both. Needs glslc (shaderc); the committed files were made with the version
# in spv/GLSLC_VERSION.
set -euo pipefail
cd "$(dirname "$0")"

defines=(
  -DFFX_GPU=1
  -DFFX_GLSL=1
  -DFFX_HALF=0
  -DFFX_WAVE64=0
  # wgpu has no subgroup quad operations behind its SPIR-V passthrough guarantee.
  -DFFX_SPD_NO_WAVE_OPERATIONS=1
  -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
  -DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0
  -DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
  -DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
  -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2
  -DFFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE=0
  -DFFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT=1
  -DFFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS=1
  -DFFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS=0
  -DFFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH=1
  -DFFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING=0
)

mkdir -p spv
for pass in prepare_inputs luma_pyramid shading_change_pyramid shading_change prepare_reactivity \
  luma_instability accumulate; do
  glslc --target-env=vulkan1.2 -Ishaders/include "${defines[@]}" -fshader-stage=comp -O \
    "shaders/src/fsr3upscaler/ffx_fsr3upscaler_${pass}_pass.glsl" -o "spv/${pass}.spv"
done
glslc --version | head -2 > spv/GLSLC_VERSION
