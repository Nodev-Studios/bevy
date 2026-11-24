# wgpu_ffx: AMD FSR 3.1 for TSR

AMD's FidelityFX Super Resolution 3.1 upscaler (FSR 3.1.4) on wgpu 30, Vulkan only. `bevy_anti_alias`'s TSR runs
it as `TsrAlgorithm::Fsr3`.

## Where it comes from, and its licence

- **wgpu-ffx** by the wgpu-ffx developers (<https://github.com/cwfitzgerald/wgpu-ffx>, MIT), branch `trunk`
  at `6fe706406` (2026-09-21), version 0.1.0. The Rust in `src/` is a trimmed copy of its `wgpu-ffx/src/`.
- **AMD FidelityFX SDK v1.1.4** (<https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK>, tag `v1.1.4`, MIT,
  Copyright (C) 2024 Advanced Micro Devices, Inc.). `shaders/` is the part of wgpu-ffx's vendored copy of the SDK's
  Vulkan GLSL that the seven upscaler passes include, with wgpu-ffx's edits (samplers renumbered from bindings
  1000/1001, a format on every storage image, the luma history declared `rgba16f` where AMD declares `rgba8`, and
  no `GL_EXT_shader_image_load_formatted`). Every shader file keeps AMD's MIT notice.
- `LICENSE.MIT` is wgpu-ffx's licence, which carries both copyrights. Keep it with the code.

This directory sits under `vendor/`, outside the workspace, because it carries GLSL and SPIR-V rather than Bevy
code, and so that it can be compared with wgpu-ffx file by file.

## What differs from wgpu-ffx, and why

- **wgpu 30.** `ShaderModuleDescriptorPassthrough` takes `entry_points` instead of `num_workgroups` (the change of
  wgpu-ffx's PR #24, closed unmerged).
- **The accumulate pass binds the dilated motion vectors** when motion vectors are at the render resolution, as
  its GLSL declares (`FSR3UPSCALER_BIND_SRV_DILATED_MOTION_VECTORS` at binding 2). wgpu-ffx 0.1.0 binds the input
  vectors there, which reprojects the history along the wrong vectors (its open PR #28).
- **SPIR-V passthrough only.** wgpu-ffx fell back to naga's SPIR-V front end, which needs wgpu's `spirv` feature in
  every build. This copy needs no wgpu feature at all; FSR runs where `FsrContext::is_supported` says so (Vulkan
  with `PASSTHROUGH_SHADERS`).
- **One configuration.** The SPIR-V is compiled for what Bevy renders: HDR colour, reverse-Z depth with an infinite
  far plane, unjittered motion vectors at the render resolution, 32-bit maths, no wave operations, the reference
  Lanczos reprojection, no sharpening. wgpu-ffx's permutation tables, RCAS, the debug view, the unimplemented
  reactive-mask generation, the exposure and reactive-mask inputs and the context flags are gone.
- **The view owns the dilated depth, the dilated motion vectors and the reconstructed-depth buffer**, which
  wgpu-ffx took from the caller.
- **No blocking in `dispatch`.** wgpu-ffx pushed an error scope and waited on it every frame, and panicked on an
  error. wgpu's own error handler reports errors instead.
- **Clears without a shader.** With reverse-Z both per-frame buffer clears are to zero, so `clear_buffer` replaces
  wgpu-ffx's WGSL clearing pass.

## Rebuilding the SPIR-V

`./compile-shaders.sh` compiles `shaders/` into `spv/` with `glslc`; the version used is in `spv/GLSLC_VERSION`.
The committed files are its output: rerunning it with that version reproduces them byte for byte. The script
lists every define and the value each permutation is fixed to.

## Testing

`cargo test --manifest-path vendor/wgpu_ffx/Cargo.toml` runs two frames on the first Vulkan adapter and fails on
any wgpu validation error. On this project's machine run the test binary through `tools/gfx-headless.sh`.
