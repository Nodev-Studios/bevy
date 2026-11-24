# nrd_wgpu

Runs NVIDIA's NRD denoisers (ReBLUR diffuse + specular today) as ordinary wgpu compute pipelines: NRD's SPIR-V
goes in through passthrough shader modules, its texture pools are wgpu textures, and its per-frame dispatch list is
replayed into a `wgpu::CommandEncoder`. No raw Vulkan handle is used. It is to NRD what `dlss_wgpu` is to DLSS.

The crate is outside the Bevy workspace (`"vendor"` is in the root manifest's `exclude`), so the workspace builds
without the SDK.

## What is in this repository, and what is not

Nothing from the NRD SDK is committed: no source, header, SPIR-V, or translation of `NRD.hlsli`. What is here:

- `shim/`: a C interface of this crate's own (its structs, names and enum values are chosen here and mirrored in
  `src/ffi.rs`) and a C++ file that implements it by calling NRD's public API. It is compiled against the
  fetched SDK's headers at build time.
- `shaders/`: two compute passes of ours. `pack` turns the input contract (plain demodulated radiance with world
  hit distance, normal and roughness, view depth, motion) into ReBLUR's encodings; `unpack` turns ReBLUR's output
  back. The encodings are NRD's: both files `#include "NRD.hlsli"` from the fetched SDK and call its functions
  by name, and `build.rs` compiles them to SPIR-V with the DXC that NRD's build downloads.
- `src/`, `examples/reblur_plane.rs`, `build.rs`: ours.

A binary built with this crate contains NRD's object code (its library and its compiled shaders). NRD is under
NVIDIA's RTX SDKs licence (`LICENSE.txt` in the SDK); a person must read it before anything built with it ships.

## Building

Needs CMake, Ninja or Make, a C++17 compiler, Git, and network access on the first build (NRD's CMake project
fetches ShaderMake, MathLib and a DXC binary into the build folder).

```sh
eval "$(tools/fetch-sdks.sh nrd)"   # from the root of the fork: clones NRD v4.18.0, prints NRD_SDK
cd vendor/nrd_wgpu
cargo build --release --example reblur_plane
```

`build.rs` builds NRD as a static library with only SPIR-V embedded (`NRD_EMBEDS_SPIRV_SHADERS=ON`, DXIL and
DXBC off) and without quad intrinsics (`NRD_SUPPORTS_QUAD_INTRINSICS=OFF`: they need
`VK_KHR_compute_shader_derivatives`, which wgpu does not enable). NRD's configure step writes
`Shaders/NRDConfig.hlsli` into the SDK folder; everything else stays in `OUT_DIR`.

**Windows, cross-built from Linux** (`cargo xwin build --release --target x86_64-pc-windows-msvc --example
reblur_plane`): NRD's project builds its ShaderMake tool for the host, then calls it by the target's
executable name (`ShaderMake.exe`); its resource script includes `version.h` while the file is `Version.h`. On a
Linux host targeting Windows, `build.rs` gives both their other name with a symlink, and passes `-mssse3` to
clang-cl for NRD's math library. The resulting `.exe` imports only system DLLs and the MSVC runtime
(`MSVCP140.dll`, `VCRUNTIME140*.dll`). It has been linked, never run.

**Windows, native** (not tried): install Visual Studio 2022 with the C++ workload, CMake and Git; clone
`https://github.com/NVIDIA-RTX/NRD` at tag `v4.18.0` and set `NRD_SDK` to it (or run `tools/fetch-sdks.sh nrd`
from Git Bash); then `cargo build --release --example reblur_plane`.

## The check

`examples/reblur_plane.rs` runs full ReBLUR frames (pack, every dispatch NRD asks for, unpack) on a synthetic
noisy floor and exits 0 when the gate passes: per channel and lobe, the denoised mean within 2 % of the truth,
the variance down at least 10 times, no NaN or infinity in any frame, no wgpu error. It needs no window, only a
Vulkan adapter, chosen with `WGPU_ADAPTER_NAME`. See its header for the scenarios and options.

## Known limits, with the measurements behind them

All measured on 2026-11-24 with NRD 4.18.0 on an RTX 5070 (NVIDIA 620.73.04) and on the Ryzen 9600X's integrated
GPU (Mesa RADV 26.3.1), both through Vulkan.

- **Format-less storage images.** Every one of NRD's 83 SPIR-V permutations declares the
  `StorageImageWriteWithoutFormat` capability and 8 declare `StorageImageReadWithoutFormat`; in Vulkan they need
  the device features `shaderStorageImageWriteWithoutFormat` / `shaderStorageImageReadWithoutFormat`, which wgpu
  never enables and has no way to ask for. Both drivers above ran the shaders correctly anyway. A driver is free
  to refuse; AMD's Windows driver has not been tried. Rewriting the formats into NRD's SPIR-V would modify NVIDIA's
  object code and is not done.
- **16-bit float textures darken the result.** Both drivers round toward zero when a shader stores into a 16-bit
  float texture, so a value blended a little closer to its target each frame stops short of it. With the textures
  NRD asks for (`TexturePrecision::Half`) the denoised mean of a uniform floor came out 2.1 % dark after 32
  frames at 1920x1080 and 2560x1440, and up to 4.0 % dark after 96 frames at 640x360. `TexturePrecision::Full`
  creates 32-bit textures in their place (NRD's format-less images accept them) and stays within 0.15 %, at
  646.9 MiB of textures at 2560x1440 instead of 400.8 MiB (40.4 MiB at 640x360).
- **Probabilistic lobe selection with uncorrelated hit distances.** When each pixel samples one lobe (the other
  gets radiance 0 and hit distance 0, reconstructed by ReBLUR) and the hit distances are white noise (0.5 to 2.5
  world units), the denoised mean comes out 2.4 % dark for diffuse and 4.1 % for specular, on both drivers. With
  the same random hit distances and both lobes sampled in every pixel it is within 0.12 %, so the drift comes
  from ReBLUR's handling of reconstructed hit distances next to real ones, not from this crate's passes. NRD's own
  README warns that probabilistic lobe selection can introduce bias. A rendered scene, whose hit distances are
  correlated between neighbours, has not been measured yet.
- **Unexercised shaders.** `REBLUR_SplitScreen` and `REBLUR_Validation` are NRD debug views this crate never asks
  for; every other ReBLUR shader runs in the check.
- **Cost.** `Denoiser::set_timestamps` and `read_timestamps` time every dispatch on the GPU;
  `reblur_plane --timestamps dispatches` prints them. At 2560x1440 on the RTX 5070, `Full` precision, the medians of
  60 frames run one at a time (several runs, 2026-11-24, the same in the Bevy testbed's `interior` scene):

  | Dispatch | ms |
  |---|---|
  | pack (this crate) | 0.27–0.32 |
  | Classify tiles | 0.02 |
  | Hit distance reconstruction | 0.34 |
  | Pre-pass | 0.69–0.72 |
  | Temporal accumulation | 1.12–1.14 |
  | History fix | 0.53–0.54 |
  | Blur | 0.63–0.64 |
  | Post-blur | 0.60–0.61 |
  | Temporal stabilization | 0.57–0.58 |
  | unpack (this crate) | 0.22 |
  | **Total** | **5.03–5.07** |

  `Half` totals 3.96 ms (one run). NVIDIA's README gives 2.55 ms for ReBLUR alone on an RTX 4080 at 1440p, with
  16-bit textures. By public spec sheets that card has about 1.6 times the RTX 5070's FP32 throughput and about the
  same memory bandwidth, so `Half`'s 3.5 ms of ReBLUR dispatches (1.4 times NVIDIA's figure) is in the range that
  card ratio predicts; `Full` adds about 1 ms. Quad intrinsics are off (`build.rs`) and were not measured: turning
  them on needs `VK_KHR_compute_shader_derivatives`, which wgpu cannot enable.
- **One compute pass per dispatch.** The GPU here is shared with a desktop, and with the dispatches recorded in one
  compute pass, a frame took 0.9 to 3.2 ms longer, back to back, than the same dispatches in passes of their own
  (12 of 12 paired rounds in one process). Writing a timestamp after each dispatch inside one pass gave the same
  gain. Why is not known: wgpu inserts the same barriers either way. `denoise` records a pass per dispatch.
- **History confidence.** NRD's README says ReBLUR's anti-lag (`ReblurSettings::antilag`,
  `fast_history_clamping_sigma_scale`) is not enough on its own against lag after a change of light, and that
  history confidence inputs are the better cure. This crate does not take them yet.
- **`mock` feature.** Builds without the SDK: nothing of NRD is compiled or linked and `Denoiser::new` always
  fails. It exists so that a build with every feature on (`cargo doc --all-features`) works without the SDK.
