# Shipping a game built on this fork's graphics

What a game must put beside its executable, and which notices it must reproduce, for each cargo feature of the
graphics work (`bevy_graphics`, `bevy_solari`, `dlss`, `nrd`, `fsr`, `bevy_baked_gi`), on Windows and Linux. The
licence texts are in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) at the root of the repository.

Nothing in this document replaces reading the licences yourself. The NVIDIA RTX SDKs licence in particular has
conditions beyond notices (trademark placement, distribution terms); see the end of this page.

## Summary

| Feature | Windows: files beside the `.exe` | Linux: files beside the executable | Notices to reproduce |
|---|---|---|---|
| none (every build) | the Visual C++ runtime (`VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`): every Rust MSVC binary imports it | nothing | SGSR 2 (BSD-3-Clause) and FidelityFX FSR 3.1 (MIT): TSR is in every build |
| `dlss` | `nvngx_dlss.dll`; `nvngx_dlssd.dll` for Ray Reconstruction | `libnvidia-ngx-dlss.so.310.9.1`; `libnvidia-ngx-dlssd.so.310.9.1` for Ray Reconstruction | NVIDIA DLSS (programming guide section 9.5 and the third-party texts in 9.6), plus the NVIDIA marks in credits / splash / about box |
| `nrd` | nothing (NRD is linked into the `.exe`); the C++ runtime `MSVCP140.dll` (Visual C++ 2015-2022 redistributable) | nothing (`libstdc++.so.6` from the system) | NVIDIA RTX SDKs licence terms (NRD), plus the NVIDIA marks in credits |
| `fsr` | `amd_fidelityfx_loader_vk.dll`, `amd_fidelityfx_upscaler_vk.dll`, and the Visual C++ 2015-2022 redistributable | nothing (FSR 4 never runs on Linux) | AMD FidelityFX SDK (MIT, the DLLs' licence in SDK 2.4.0) |
| `bevy_baked_gi` | each bake folder: `manifest.gi.ron`, one `<region>.ktx2` per region, one `<region>.refl.ktx2` per region with a reflection probe | the same | none |
| `bevy_solari`, `bevy_graphics` | nothing | nothing | none beyond Bevy's own |

Where the files come from: `tools/fetch-sdks.sh dlss nrd fsr` puts each SDK under `~/.local/share/bevy-graficos/`
(`$DLSS_SDK`, `$NRD_SDK`, `$FSR_SDK`). No SDK file is in this repository, and none may be: their licences do not
allow it in a source repository.

## Every build: TSR

`TsrPlugin` is part of `AntiAliasPlugin` with no feature, so every binary built from this fork contains:

- **Qualcomm's Snapdragon Game Super Resolution 2** (SGSR 2), ported to WESL in
  `crates/bevy_anti_alias/src/tsr/{convert,activate,upscale}.wesl`. BSD-3-Clause: "Redistributions in binary form
  must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution."
- **AMD FidelityFX FSR 3.1**'s compute passes, through `vendor/wgpu_ffx` (MIT, `vendor/wgpu_ffx/LICENSE.MIT`).
  MIT asks for the notice "in all copies or substantial portions of the Software".

Put both notices from `THIRD_PARTY_NOTICES.md` in the game's documentation or credits (a licences screen or a
text file shipped with the game both serve).

On Windows, every Rust binary built for `x86_64-pc-windows-msvc` imports `VCRUNTIME140.dll` and
`VCRUNTIME140_1.dll`. Most machines have them; a store or installer usually installs the Visual C++ 2015-2022
redistributable anyway, and `nrd` and `fsr` need it for `MSVCP140.dll` too.

## `dlss`: NVIDIA DLSS Super Resolution and Ray Reconstruction

**Files.** From `$DLSS_SDK/lib/<platform>/rel/` (never `dev/`: that build draws an on-screen debug overlay and
must not ship), into the folder of the executable:

| | Windows (`Windows_x86_64`) | Linux (`Linux_x86_64`) |
|---|---|---|
| Super Resolution (DLSS SR, DLAA) | `nvngx_dlss.dll` | `libnvidia-ngx-dlss.so.310.9.1` |
| Ray Reconstruction (the denoiser of RtFull, and of RtLite on RTX) | `nvngx_dlssd.dll` | `libnvidia-ngx-dlssd.so.310.9.1` |

Keep the file names: NVIDIA's runtime loads them by name, and the DLLs are signed (a renamed or altered DLL does
not load). At run time `dlss_wgpu` 6.0.1 searches `$DLSS_SDK/lib/<platform>/rel` when that variable is set, and
NVIDIA's runtime also searches the application's folder (DLSS programming guide, section 4.2: "Ensure that after
game installation, the file resides in the same folder as the game executable"). A player's machine has no
`$DLSS_SDK`, so beside the executable is the only place that works. The frame-generation libraries
(`nvngx_dlssg.dll`, `libnvidia-ngx-dlssg.so.*`) are not used by this fork; do not ship them.

**Also required:** the app inserts a `DlssProjectId` before `DefaultPlugins`; a build with `dlss` panics without
one.

**What happens when they are missing.** Nothing fails and nothing warns. Bevy's DLSS plugin logs
`DLSS is not supported on this system` at `info` level, the `DlssSuperResolutionSupported` and
`DlssRayReconstructionSupported` resources are never inserted, and the ladder resolves as if the card had no
DLSS: **every NVIDIA player gets NRD + TSR on the traced rungs and TSR (or TAA) on Raster**, with a worse image (more
lag, softer reflections). Only the log shows it (see "Checking a build" below). With only `nvngx_dlss.dll` and no
`nvngx_dlssd.dll`, Super Resolution works and the traced rungs use NRD.
The game also exits cleanly: before 2026-11-25 such a build crashed at exit on NVIDIA (NGX left initialized by
`dlss_wgpu`); the fork now shuts NGX down when DLSS turns out unavailable (`crates/bevy_anti_alias/src/dlss/mod.rs`).

**Notices and other terms.**
- The DLSS programming guide (section 4.2.3) asks for "the full text of copyright and license blurbs that are
  found in section 9.5" in the game's public documentation, because the DLL contains third-party code. They are in
  `THIRD_PARTY_NOTICES.md`.
- The NVIDIA RTX SDKs licence (`$DLSS_SDK/LICENSE.txt`) requires, for applications using DLSS, attributing DLSS
  and including the NVIDIA marks "on splash screens, in the about box of the application (if present), and in
  credits for game applications" (supplement, trademark section (b)), and has its own distribution requirements
  (section 2). Its supplement also requires telling NVIDIA before the commercial release of an application that
  incorporates the DLSS SDK ("Notification for the DLSS SDK …": company, publisher and developer name, SDK used,
  application name, platform, ship date, link; the address is in the licence). Read it before shipping.

## `nrd`: NVIDIA Real-Time Denoisers (ReBLUR)

**Files.** None: `vendor/nrd_wgpu` builds NRD as a static library with its SPIR-V embedded, so NRD is inside the
executable. On Windows the C++ code adds `MSVCP140.dll` to the imports (the Visual C++ 2015-2022
redistributable); on Linux it uses the system's `libstdc++.so.6`.

**What happens when NRD cannot run.** A build without `nrd` has no denoiser for traced lighting except DLSS Ray
Reconstruction, so `Auto` gives Raster to every card without it (AMD, Intel, and NVIDIA without the DLSS DLLs):
log line `graphics: Raster: neither DLSS Ray Reconstruction nor NRD can denoise ray traced lighting here`. If NRD
fails on a device at run time, Bevy's NRD node logs an `error` (`NRD cannot denoise this view …` or `NRD failed;
this view shows the noisy lighting from now on …`).

**Terms.** NRD is under the NVIDIA RTX SDKs licence (`$NRD_SDK/LICENSE.txt`, the same licence family as DLSS's
with NRD-specific wording). For an SDK other than DLSS/NGX it requires attributing the SDK and including the
NVIDIA marks "on the credit screen for applications that have such credit screen, or where a credit screen is not
present prominently in end user documentation" (supplement, trademark section (c)). NRD is the denoiser of every
traced frame on AMD and Intel cards, so a game that ships traced rungs ships NRD.

## `fsr`: AMD FSR 4 / 3.1 through AMD's DLLs (Windows, Radeon)

**Files.** From `$FSR_SDK/Kits/FidelityFX/signedbin/`, beside the `.exe`:

- `amd_fidelityfx_loader_vk.dll` (loaded by `vendor/fsr_wgpu` with `LoadLibrary`),
- `amd_fidelityfx_upscaler_vk.dll` (loaded by the loader, from the executable's folder or its own),
- the Visual C++ 2015-2022 redistributable (both DLLs import `MSVCP140.dll`, `VCRUNTIME140.dll`,
  `VCRUNTIME140_1.dll`).

The `.exe` imports nothing of FSR; the DLLs are signed and must not be renamed or altered. Details:
`vendor/fsr_wgpu/README.md`, section "Shipping". On Linux, and on any GPU that is not a Radeon, FSR does nothing and
needs no file.

**What happens when they are missing.** Bevy's FSR plugin logs `FSR is not available: …` at `warn` level and the
ladder records `graphics: FSR cannot run here: it needs Windows and FSR's DLL` if the game asked for FSR; the
upscaler falls back to TSR. `Auto` never picks FSR today, so a game that does not offer FSR in its options can leave
the feature off and ship no DLL.

**When they load:** `FSR: the DLL runs FSR <version> (provider id …) on this GPU` at `info` level.

**Notice.** AMD FSR SDK 2.4.0's `Kits/FidelityFX/docs/license.md` lists both DLLs among the files under the MIT
licence (the SDK's binary-only licence covers the files it does not list). Reproduce that MIT notice
(`THIRD_PARTY_NOTICES.md`).

## `bevy_baked_gi`: baked indirect light

**Files.** For each bake (one folder per level, written by the baker under the `bevy_baked_gi_bake` feature), ship
the whole folder as game assets, under the asset path the game's `BakedGi` entity names:

- `manifest.gi.ron` (the regions, their light keys and weights, the scene hash),
- `<region>.ktx2` for every region in the manifest (the irradiance volume),
- `<region>.refl.ktx2` for every region whose manifest entry has a `reflection` (its reflection probe).

The file names are the ones in `manifest.gi.ron` (`file:` of each region and of its `reflection`); check them
against the folder before shipping. The game does not need the `bevy_baked_gi_bake` feature, `bevy_solari`, or a
ray-tracing GPU to load a bake; it needs `bevy_baked_gi`, which turns on Bevy's `ktx2` and `zstd` image support.

**What happens when they are missing.**
- No manifest, or a manifest that fails to load: `warn` `baked GI manifest "…" failed to load`, `BakedGiStatus`
  stays `Absent`, and the level keeps the lighting it has without a bake.
- A missing `.ktx2` of a region: Bevy's asset server logs the path it could not load; that region's volume or
  probe has no texture, so it adds no light (read in the code, not run).
- A level that moved since the bake: `warn` `baked GI is stale: …; rebake`, status `Stale`.
- A device that cannot bind arrays of textures: `warn` `baked GI not loaded: …`, status `Degraded`.

## Checking a build: the log

Run the game once on each machine you can (an RTX card, a Radeon, an integrated GPU) with `RUST_LOG=info` (Bevy's
default level already shows `info`) and read these lines:

| What | Line | Logged by |
|---|---|---|
| What the ladder chose | `graphics: <rung>, denoiser <denoiser>, upscaler <upscaler> at <quality> for a WxH output` | `crates/bevy_graphics/src/apply.rs` (`log_resolved`), each time the result changes |
| Why anything differs from what was asked | `graphics: <reason>`, one line per fallback, right after it | the same |
| DLSS libraries not found | `DLSS is not supported on this system` | `crates/bevy_anti_alias/src/dlss/mod.rs` |
| DLSS Ray Reconstruction not usable | `graphics: NRD denoises: DLSS Ray Reconstruction cannot run here`, only when the game asked for Ray Reconstruction; under `Auto` the ladder just picks NRD | `bevy_graphics` |
| DLSS not usable for upscaling | `graphics: DLSS cannot run here` (when DLSS was asked) | `bevy_graphics` |
| FSR's DLLs loaded | `FSR: the DLL runs FSR …` | `crates/bevy_anti_alias/src/fsr/mod.rs` |
| FSR's DLLs missing | `FSR is not available: …` | the same |
| NRD failed on this device | `NRD cannot denoise this view …` / `NRD failed; …` | `crates/bevy_anti_alias/src/nrd/` |
| Bake missing or stale | `baked GI manifest … failed to load` / `baked GI is stale: …` | `crates/bevy_baked_gi/src/runtime.rs` |

**On an RTX card with the DLSS files in place**, a traced rung must say `denoiser DlssRayReconstruction`. If it
says `denoiser Nrd` and, earlier, `DLSS is not supported on this system`, the DLSS files are missing or not beside
the executable. Under `Auto` that is the only sign: no `graphics:` fallback line is logged, because nothing the game
asked for was refused. Measured on 2026-11-25 (RTX 5070, Linux, `DLSS_SDK` unset, no library beside the testbed):
`DLSS is not supported on this system`, then `graphics: RtFull, denoiser Nrd, upscaler Taa at NativeAa for a
640x360 output`.

A game can also read the result in code: the `ResolvedGraphics` resource (rung, denoiser, upscaler, quality, and
`fallbacks`), and `BakedGiStatus`.

## Licence terms beyond notices (read them)

- **NVIDIA RTX SDKs licence** (DLSS and NRD; `LICENSE.txt` in each SDK): the SDK may be distributed only "as
  incorporated in object code format into a software application subject to the distribution requirements",
  the application "must have material additional functionality, beyond the included portions of the SDK", the
  SDK must be distributed "subject to the terms at least as protective as the terms of this license", and the
  trademark placement above, and, for DLSS, the notification to NVIDIA before a commercial release. NRD's copy
  of the licence (v4.18.0) has the same structure, with the notification clause limited to the DLSS and NGX SDKs.
  Questions: the address in its section 19.
- **AMD FidelityFX SDK**: MIT for the files shipped here.
- **SGSR 2**: BSD-3-Clause; also forbids using Qualcomm's or the contributors' names to endorse the game without
  permission.

This page was written from the fork at `dev` `67b6553fd`, `dlss_wgpu` 6.0.1 with DLSS SDK v310.9.1, NRD v4.18.0
and AMD FSR SDK 2.4.0 (2026-11-25). A Windows build of the full feature set has been linked from Linux, never run.
