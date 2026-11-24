# fsr_wgpu

Runs AMD FSR upscaling for wgpu on Vulkan through the signed DLLs of AMD's FSR SDK 2.4.0 (2026-10-28), with
wgpu's raw Vulkan handles. It is to FSR what `dlss_wgpu` is to DLSS. The DLL runs FSR 4 (4.1.2, machine-learned)
where the GPU supports it and the older FSR 3.1 (3.1.6) elsewhere; `FsrSdk::probe` says which one it chose.

The crate is outside the Bevy workspace (`"vendor"` is in the root manifest's `exclude`), like `nrd_wgpu`.

**Nothing here has run on Windows or on a Radeon with the DLL.** Everything below that is about the DLL's
behaviour comes from AMD's documentation and sample, not from a run.

## What is in this repository, and what is not

Nothing from the SDK is committed: no header, no DLL, no source. What is here:

- `src/ffi.rs`: the few C structs and constants of AMD's API this crate calls, written by hand from the SDK's
  public headers (MIT; their notice is in the file), with one table of every size, offset and value.
- `src/device.rs`, `src/lib.rs`, `examples/device_probe.rs`: ours.

The DLLs are not linked: `amd_fidelityfx_loader_vk.dll` is loaded at run time with `LoadLibrary`, as AMD
recommends, so building needs no SDK at all, on any system. They are binary-redistributable; a Windows bundle
puts them next to the executable (see "Shipping").

## Which GPUs get FSR 4

From the SDK's own documents, which disagree with each other:

- `Kits/FidelityFX/docs/whats-new/index.md` (2.4.0): FSR 4.1 on **Radeon RX 6000 (RDNA 2)** and the Ryzen AI
  Max 300 / 400 ("Strix / Gorgon Halo"), added in this release; a lighter INT8 model on Ryzen AI 9 300 / 400,
  Ryzen Z2 A and Z2 / AI Z2 Extreme.
- `Kits/FidelityFX/docs/techniques/super-resolution-ml.md`: its performance table lists FP8 on RX 9070 XT /
  9060 XT and INT8 on RX 7900 XTX / 7800 XT / 7600 and RX 6900 XT / 6700 XT / 6600 / 6500 XT; its "Limitations"
  section still says "FSR4 requires an AMD 7000 series discrete GPU or AMD 9000 series GPU or later".
- So: RX 9000 (RDNA 4) runs the FP8 model; RX 7000 (RDNA 3) and, from 2.4.0, RX 6000 (RDNA 2) the INT8 model.
  Older Radeons, desktop Ryzen integrated graphics (not on the integrated list) and every other vendor get FSR
  3.1.6 from the same DLL. The driver version needed is not documented for the upscaler.

The integration in `bevy_anti_alias` only offers FSR on AMD adapters, so in practice the DLL is never asked on
another vendor's GPU.

## The device

The SDK documents no Vulkan device requirement. `register_device_extensions`, called while wgpu creates the
device, enables what the DLL names in its binary and what AMD's sample device enables
(`Kits/Cauldron2/framework/render/vk/device_vk.cpp`), where the adapter has it and wgpu does not already chain the
feature struct (a second struct of the same type is invalid Vulkan): `VK_KHR_cooperative_matrix`,
`VK_EXT_shader_float8` (with `shaderFloat8CooperativeMatrix`), `VK_KHR_compute_shader_derivatives`,
`synchronization2`, and the Vulkan 1.2 features the sample turns on wholesale that wgpu never chains (8-bit storage,
scalar block layout, standard uniform buffer layout, subgroup extended types, separate depth/stencil layouts). Bevy
asks wgpu for every adapter feature, so `SHADER_F16` (float16 and 16-bit storage) and, where wgpu reports it,
`EXPERIMENTAL_COOPERATIVE_MATRIX` (which makes wgpu enable `cooperativeMatrix` itself) are on already. What it cannot enable, because wgpu owns those structs and their fields are private, is listed
in `src/device.rs`; the first a Windows run may miss is `computeFullSubgroups`.

`examples/device_probe.rs` creates a device the way Bevy does with that callback, under Vulkan's validation
layer. On 2026-11-24 it passed on the RTX 5070 (NVIDIA 620.73.04, with `--any-vendor`) and on the Ryzen 9600X's
integrated GPU (Mesa RADV 26.3.1): device created, no validation error. With a duplicated feature struct put in on
purpose, the layer reported `VUID-VkDeviceCreateInfo-sType-unique`, so it was active.

## Each frame

`FsrUpscaler::render` follows `dlss_wgpu`: the textures' state changes go on the caller's encoder, FSR records
into a command buffer of its own. It describes each texture as the SDK's `ffxApiGetImageResourceDescriptionVK`
does, passes the inputs in the read state AMD's sample uses, and moves the depth texture from wgpu's
`DEPTH_STENCIL_READ_ONLY_OPTIMAL` to `SHADER_READ_ONLY_OPTIMAL` and back around FSR's dispatch, because FSR's
Vulkan backend reads every input in that layout (as AMD's sample gives it depth).

## The layout check

```sh
eval "$(tools/fetch-sdks.sh fsr)"   # from the root of the fork: FSR SDK v2.4.0 (sparse) and the Vulkan headers
cd vendor/fsr_wgpu
cargo test
```

`ffi::tests::layout_matches_sdk_headers` compiles the table of `src/ffi.rs` as C++ `static_assert`s against the
fetched headers, for Linux with the system compiler and for Windows' MSVC ABI with `clang++` and the MSVC headers
`cargo xwin` caches. The same table is asserted on the Rust side at compile time for whatever target is built.
Without `FSR_SDK` the test says it skipped.

## Shipping

The `.exe` imports nothing of FSR. At run time it needs, next to it:

- `amd_fidelityfx_loader_vk.dll`, loaded by this crate;
- `amd_fidelityfx_upscaler_vk.dll`, loaded by the loader (from the executable's folder, or the loader's).

Both import the MSVC runtime (`MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`): the machine needs the
Visual C++ 2015–2022 redistributable. The upscaler DLL also imports `vulkan-1.dll` (the driver's loader) and
`VERSION.dll` (Windows). Both DLLs come from `Kits/FidelityFX/signedbin/` of the fetched SDK.
