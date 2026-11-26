//! The whole policy of the graphics ladder: [`resolve`] turns what was asked for and what the machine can do
//! into what will run, and says why the two differ.

use crate::{
    raster::RasterLevel, DenoiserChoice, GpuDeviceType, GpuVendor, GraphicsCapabilities, GraphicsPreset,
    GraphicsSettings, UpscaleQuality, UpscalerChoice,
};
use bevy_ecs::prelude::*;
use bevy_math::UVec2;
use bevy_reflect::{Reflect, ReflectDeserialize, ReflectSerialize};
use core::fmt;
use serde::{Deserialize, Serialize};

const GIB: u64 = 1 << 30;

/// Under this much video memory `Auto` stays on [`GraphicsRung::Raster`].
// to tune: 5.5 and not 6, because a card sold as "6 GB" reports a little under 6 GiB (a "12 GB" RTX 5070
// reports 11.94 GiB), and the RTX 2060 6 GB is meant to reach `RtLite`.
const MIN_TRACED_VIDEO_MEMORY: u64 = 11 * GIB / 2;

/// From this much video memory on, `Auto` may pick [`GraphicsRung::RtFull`].
// to tune: 11 and not 12, for the same reason.
const MIN_RT_FULL_VIDEO_MEMORY: u64 = 11 * GIB;

const PIXELS_1080P: u64 = 1920 * 1080;
const PIXELS_1440P: u64 = 2560 * 1440;
/// Ultrawide 1440p, 4.95 MP: up to here an output is not treated as 4K.
const PIXELS_ULTRAWIDE_1440P: u64 = 3440 * 1440;

/// What will actually run for a [`GraphicsSettings`] on a machine, from [`resolve`].
#[derive(Resource, Reflect, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[reflect(Resource, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedGraphics {
    /// The rung of the ladder that renders.
    pub rung: GraphicsRung,
    /// What anti-aliases the frame and, when [`Self::render_scale`] is under 1, upscales it.
    pub upscaler: ResolvedUpscaler,
    /// What denoises the ray traced lighting.
    pub denoiser: ResolvedDenoiser,
    /// How far below the output's resolution the frame is rendered.
    pub quality: ResolvedQuality,
    /// Which of Raster's two effect tables the camera gets ([`raster`](crate::raster)). [`RasterLevel::Light`]
    /// only under [`GraphicsPreset::Auto`] on a weak adapter ([`Fallback::LightRaster`]); `Full` everywhere
    /// else, the traced rungs included (they read only its bloom and shadow filter, which are the same).
    #[serde(default)]
    pub raster: RasterLevel,
    /// The internal resolution as a share of the output's on each axis: [`ResolvedQuality::scale`] of
    /// [`Self::quality`].
    pub render_scale: f32,
    /// Every place where what runs is not what was asked for, or where `Auto` stopped short of the top, with
    /// its reason. Empty when everything asked for was given.
    pub fallbacks: Vec<Fallback>,
    /// Meshes that could not be given a ray traceable copy, so they are missing from traced shadows,
    /// reflections and bounce light. [`resolve`] returns 0; the apply layer counts.
    pub unfitted_meshes: u32,
    /// Skinned and morphed meshes, which are never ray traced: they are drawn and lit, but cast no traced
    /// shadow. [`resolve`] returns 0; the apply layer counts.
    pub skipped_deforming_meshes: u32,
}

/// A rung of the graphics ladder.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum GraphicsRung {
    /// Rasterized lighting.
    Raster,
    /// Ray traced lighting, cheap parameter set.
    RtLite,
    /// Ray traced lighting, full parameter set.
    RtFull,
}

impl GraphicsRung {
    /// Whether this rung traces rays.
    pub fn is_traced(self) -> bool {
        self != Self::Raster
    }
}

/// What anti-aliases and upscales the frame.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum ResolvedUpscaler {
    /// Temporal anti-aliasing at the output's resolution. No upscaling.
    Taa,
    /// The vendor-neutral temporal upscaler.
    Tsr,
    /// DLSS Super Resolution.
    DlssSuperResolution,
    /// DLSS Ray Reconstruction, which is also the denoiser.
    DlssRayReconstruction,
    /// FSR.
    Fsr,
}

/// What denoises the ray traced lighting.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum ResolvedDenoiser {
    /// Nothing: the rung is [`GraphicsRung::Raster`].
    None,
    /// DLSS Ray Reconstruction.
    DlssRayReconstruction,
    /// NRD.
    Nrd,
}

/// How far below the output's resolution the frame is rendered. [`UpscaleQuality`] without its `Auto`.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum ResolvedQuality {
    /// The output's resolution.
    NativeAa,
    /// Two thirds of the output's resolution on each axis.
    Quality,
    /// 58 % of the output's resolution on each axis.
    Balanced,
    /// Half the output's resolution on each axis.
    Performance,
}

impl ResolvedQuality {
    /// The internal resolution as a share of the output's on each axis. These are DLSS's ratios, used for
    /// every upscaler so a quality name means one resolution.
    pub fn scale(self) -> f32 {
        match self {
            Self::NativeAa => 1.0,
            Self::Quality => 2.0 / 3.0,
            Self::Balanced => 0.58,
            Self::Performance => 0.5,
        }
    }
}

/// One place where what runs is not what was asked for, or where `Auto` stopped short of the top, and why.
///
/// Its [`Display`](fmt::Display) text is a sentence fit for a log or a report.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum Fallback {
    /// Raster: the renderer is not on Vulkan.
    NotVulkan,
    /// Raster, under `Auto` only: the adapter is a CPU rasterizer.
    SoftwareAdapter,
    /// Raster: the app was built without ray traced lighting.
    RayTracingNotCompiled,
    /// Raster: the device lacks a feature ray traced lighting needs.
    NoRayQuery,
    /// Raster, under `Auto` only: the adapter is an integrated or a virtual GPU.
    IntegratedGpu,
    /// Raster: neither DLSS Ray Reconstruction nor NRD can run, and traced lighting cannot be shown without
    /// a denoiser.
    NoDenoiser,
    /// Raster, under `Auto` only: the adapter has too little video memory, or it could not be read.
    LowVideoMemory,
    /// `RtLite` instead of `RtFull`, under `Auto` only: the adapter has too little video memory for `RtFull`.
    VideoMemoryBelowRtFull,
    /// `RtLite` instead of `RtFull`, under `Auto` only: the adapter is not on the list of cards known to be
    /// fast enough for `RtFull`.
    NotOnFastRtList,
    /// DLSS Ray Reconstruction was asked for and cannot run here; NRD denoises.
    DlssRayReconstructionUnavailable,
    /// NRD was asked for and is not in this app; DLSS Ray Reconstruction denoises.
    NrdUnavailable,
    /// TSR was asked for and is not in this app.
    TsrUnavailable,
    /// DLSS was asked for and cannot run here.
    DlssUnavailable,
    /// FSR was asked for and cannot run here: it needs Windows and FSR's DLL.
    FsrUnavailable,
    /// FSR was asked for on an adapter that is not a Radeon.
    FsrNeedsRadeon,
    /// An upscaler other than DLSS was asked for, but DLSS Ray Reconstruction denoises and it upscales too.
    RayReconstructionUpscales,
    /// The frame renders at the output's resolution although a lower one was wanted: no upscaler can run.
    NoUpscaler,
    /// Raster, under `Auto` only: the adapter has ray tracing but is on the list of cards too slow at it for
    /// `RtLite`.
    SlowRayTracing,
    /// Raster's light table and a lower internal resolution, under `Auto` only: the adapter is integrated,
    /// virtual or a CPU rasterizer, or has under 5.5 GiB of video memory.
    LightRaster,
    /// NRD denoises instead of DLSS Ray Reconstruction, under `Auto` only, on `RtLite` on an RTX 20 or 30
    /// card: Ray Reconstruction's cost on those cards' tensor cores is unmeasured.
    NrdOnOlderRtx,
}

impl fmt::Display for Fallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotVulkan => "Raster: ray tracing needs the Vulkan backend",
            Self::SoftwareAdapter => "Raster: the adapter is a CPU rasterizer",
            Self::RayTracingNotCompiled => "Raster: the app was built without ray traced lighting",
            Self::NoRayQuery => "Raster: the device lacks a feature ray traced lighting needs",
            Self::IntegratedGpu => "Raster: the adapter is an integrated or virtual GPU",
            Self::NoDenoiser => {
                "Raster: neither DLSS Ray Reconstruction nor NRD can denoise ray traced lighting here"
            }
            Self::LowVideoMemory => {
                "Raster: the adapter has under 5.5 GiB of video memory, or its size could not be read"
            }
            Self::VideoMemoryBelowRtFull => {
                "RtLite: the adapter has under 11 GiB of video memory"
            }
            Self::NotOnFastRtList => {
                "RtLite: the adapter is not on the list of cards fast enough for RtFull"
            }
            Self::DlssRayReconstructionUnavailable => {
                "NRD denoises: DLSS Ray Reconstruction cannot run here"
            }
            Self::NrdUnavailable => {
                "DLSS Ray Reconstruction denoises: NRD is not in this app"
            }
            Self::TsrUnavailable => "TSR is not in this app",
            Self::DlssUnavailable => "DLSS cannot run here",
            Self::FsrUnavailable => "FSR cannot run here: it needs Windows and FSR's DLL",
            Self::FsrNeedsRadeon => "FSR is offered on Radeon cards only",
            Self::RayReconstructionUpscales => {
                "DLSS Ray Reconstruction upscales as it denoises, so no other upscaler runs"
            }
            Self::NoUpscaler => "rendering at the output's resolution: no upscaler can run",
            Self::SlowRayTracing => {
                "Raster: the adapter is on the list of cards too slow at ray tracing for RtLite"
            }
            Self::LightRaster => {
                "Raster's light table at a lower resolution: the adapter is integrated or has under 5.5 GiB of video memory"
            }
            Self::NrdOnOlderRtx => {
                "NRD denoises: DLSS Ray Reconstruction's cost on RTX 20 and 30 cards is unmeasured"
            }
        })
    }
}

/// Decides what runs: the rung, the denoiser, the upscaler and the internal resolution.
///
/// Pure: it reads its arguments and nothing else, so the same call gives the same answer on any machine.
/// `output` is the size in pixels of what the camera draws to.
///
/// - **Preset.** A rung asked for by name is given whenever it can run at all: it is refused only off
///   Vulkan, without ray traced lighting in the app or on the device, or without a denoiser.
///   [`GraphicsPreset::Auto`] also refuses CPU rasterizers, integrated and virtual GPUs and cards with little
///   video memory, and cards on a list of slow ray tracers, and picks `RtFull` only on a card with enough
///   memory whose name is on a list.
/// - **Raster's table.** Under `Auto`, an integrated, virtual or software adapter, or one under 5.5 GiB of
///   video memory, gets Raster's light table ([`raster`](crate::raster)) and a lower internal resolution.
/// - **Denoiser.** On a traced rung, DLSS Ray Reconstruction where it is supported, otherwise NRD; a choice
///   that cannot run falls back to the other one. Under [`DenoiserChoice::Auto`] on `RtLite`, an RTX 20 or
///   30 card gets NRD when the app has it.
/// - **Upscaler.** DLSS Ray Reconstruction upscales itself. Otherwise DLSS Super Resolution where it is
///   supported, otherwise TSR; at the output's resolution without DLSS, plain TAA. FSR is never picked
///   unless asked for by name. A choice that cannot run falls back.
/// - **Internal resolution.** Under [`UpscaleQuality::Auto`], from the rung and `output`.
///
/// Every difference from what was asked is in [`ResolvedGraphics::fallbacks`].
pub fn resolve(
    settings: &GraphicsSettings,
    capabilities: &GraphicsCapabilities,
    output: UVec2,
) -> ResolvedGraphics {
    let mut fallbacks = Vec::new();

    let rung = resolve_rung(settings.preset, capabilities, &mut fallbacks);
    let raster = if settings.preset == GraphicsPreset::Auto
        && rung == GraphicsRung::Raster
        && is_weak(capabilities)
    {
        fallbacks.push(Fallback::LightRaster);
        RasterLevel::Light
    } else {
        RasterLevel::Full
    };
    let denoiser = resolve_denoiser(rung, settings.denoiser, capabilities, &mut fallbacks);
    let wanted_quality = match settings.upscale_quality {
        UpscaleQuality::Auto => auto_quality(rung, raster, output),
        UpscaleQuality::NativeAa => ResolvedQuality::NativeAa,
        UpscaleQuality::Quality => ResolvedQuality::Quality,
        UpscaleQuality::Balanced => ResolvedQuality::Balanced,
        UpscaleQuality::Performance => ResolvedQuality::Performance,
    };
    let upscaler = resolve_upscaler(
        settings.upscaler,
        denoiser,
        wanted_quality,
        capabilities,
        &mut fallbacks,
    );
    // `Off` reaches here with an upscaler only as DLSS Ray Reconstruction, which then runs at native.
    let quality = if upscaler == ResolvedUpscaler::Taa || settings.upscaler == UpscalerChoice::Off {
        ResolvedQuality::NativeAa
    } else {
        wanted_quality
    };

    ResolvedGraphics {
        rung,
        upscaler,
        denoiser,
        quality,
        raster,
        render_scale: quality.scale(),
        fallbacks,
        unfitted_meshes: 0,
        skipped_deforming_meshes: 0,
    }
}

fn resolve_rung(
    preset: GraphicsPreset,
    capabilities: &GraphicsCapabilities,
    fallbacks: &mut Vec<Fallback>,
) -> GraphicsRung {
    let asked = match preset {
        GraphicsPreset::Auto => None,
        GraphicsPreset::Raster => return GraphicsRung::Raster,
        GraphicsPreset::RtLite => Some(GraphicsRung::RtLite),
        GraphicsPreset::RtFull => Some(GraphicsRung::RtFull),
    };
    let auto = asked.is_none();
    let video_memory = capabilities.video_memory_bytes.unwrap_or(0);

    // The rules that send a machine to Raster, in the order they are tried. A rule that is not `always` is
    // `Auto`'s judgement of speed, which a rung asked for by name overrules. A ray query feature bit alone
    // never selects a traced rung: integrated GPUs and CPU rasterizers advertise it too.
    let rules = [
        (Fallback::NotVulkan, true, !capabilities.vulkan),
        (
            Fallback::SoftwareAdapter,
            false,
            capabilities.device_type == GpuDeviceType::Cpu,
        ),
        (Fallback::RayTracingNotCompiled, true, !capabilities.solari),
        (Fallback::NoRayQuery, true, !capabilities.ray_query),
        (
            Fallback::IntegratedGpu,
            false,
            matches!(
                capabilities.device_type,
                GpuDeviceType::Integrated | GpuDeviceType::Virtual
            ),
        ),
        (
            Fallback::NoDenoiser,
            true,
            !(capabilities.dlss_ray_reconstruction || capabilities.nrd),
        ),
        (
            Fallback::LowVideoMemory,
            false,
            video_memory < MIN_TRACED_VIDEO_MEMORY,
        ),
        (
            Fallback::SlowRayTracing,
            false,
            on_slow_rt_list(&capabilities.adapter_name),
        ),
    ];
    if let Some((reason, ..)) = rules
        .into_iter()
        .find(|&(_, always, fails)| fails && (always || auto))
    {
        fallbacks.push(reason);
        return GraphicsRung::Raster;
    }

    if let Some(rung) = asked {
        return rung;
    }
    if video_memory < MIN_RT_FULL_VIDEO_MEMORY {
        fallbacks.push(Fallback::VideoMemoryBelowRtFull);
        GraphicsRung::RtLite
    } else if !on_fast_rt_list(&capabilities.adapter_name) {
        fallbacks.push(Fallback::NotOnFastRtList);
        GraphicsRung::RtLite
    } else {
        GraphicsRung::RtFull
    }
}

/// Whether `Auto` judges the adapter too weak for Raster's full table at the output's resolution: the same
/// signals that send it to Raster as [`Fallback::IntegratedGpu`], [`Fallback::SoftwareAdapter`] or
/// [`Fallback::LowVideoMemory`], read whatever rule sent it there (a card without ray tracing has those signals
/// too). Unlike `LowVideoMemory`, a size that could not be read does not count: off Vulkan it is never read,
/// and a fast card there would lose its effects for nothing. The device type still counts.
// to tune: a GPU class, not a measurement of speed. A fast iGPU (780M class) or a slow discrete card under
// 5.5 GiB is misjudged either way; the player's overrides and quality setting still reach everything.
fn is_weak(capabilities: &GraphicsCapabilities) -> bool {
    matches!(
        capabilities.device_type,
        GpuDeviceType::Integrated | GpuDeviceType::Virtual | GpuDeviceType::Cpu
    ) || capabilities
        .video_memory_bytes
        .is_some_and(|bytes| bytes < MIN_TRACED_VIDEO_MEMORY)
}

/// The model number after `prefix` in an adapter's name (`"RTX "` → `4070` in `NVIDIA GeForce RTX 4070 Ti`),
/// and the text after it.
fn model_after<'a>(adapter_name: &'a str, prefix: &str) -> Option<(u32, &'a str)> {
    let (_, rest) = adapter_name.split_once(prefix)?;
    let rest = rest.trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    Some((rest[..end].parse().ok()?, rest[end..].trim_start()))
}

/// Whether an adapter's name is on the list of cards assumed fast enough for `RtFull`.
// to tune: a guess from one card (the RTX 5070) and published ratios, until other cards are measured.
fn on_fast_rt_list(adapter_name: &str) -> bool {
    // A laptop part is far slower than the desktop card of the same number (an "RTX 5070 Ti Laptop GPU" runs
    // like a desktop RTX 4060 Ti-4070 in public reviews), so it is one rung lower: never `RtFull` under
    // `Auto`.
    if adapter_name.contains("Laptop") {
        return false;
    }
    if let Some((model, rest)) = model_after(adapter_name, "RTX ") {
        // RTX 3080 Ti, 3090 and 3090 Ti: ray tracing on par with an RTX 4070 or above in published reviews.
        if model == 3090 || (model == 3080 && rest.starts_with("Ti")) {
            return true;
        }
        // GeForce RTX 4070 and up, RTX 5070 and up: an xx70 or higher of the 40 series or a later one.
        return model >= 4000 && model % 1000 >= 70;
    }
    // Radeon RX 9070 class, and RX 7900 XT and up (the XT and the XTX, not the GRE). `GFX1201` is the RX 9070
    // class's chip (Navi 48) as RADV names it when Mesa's `amdgpu.ids` predates the card; `GFX1200` (Navi 44,
    // the RX 9060 class) is left off, as the RX 9060 XT is.
    adapter_name.contains("RX 9070")
        || adapter_name.contains("RX 7900 XT")
        || adapter_name.contains("GFX1201")
}

/// Whether an adapter's name is on the list of cards that have ray tracing but are assumed too slow at it for
/// `RtLite`, which `Auto` sends to Raster.
// to tune: from public specifications, not measured. Each has a small chip (16-28 RT units or Xe cores, 4-8 GB
// on a 64-128 bit bus): Intel Arc A310/A380 (8 Xe cores, 6 GB) and A580 (24, 8 GB); GeForce RTX 2050 and 3050
// (16-20 RT cores, 4-8 GB) and every "Laptop" xx50 (RTX 4050 Laptop: 20 RT cores, 6 GB); Radeon RX 6400/6500
// XT (12/16 ray accelerators, 4 GB), RX 6600 (28, against the 6600 XT's 32 at higher clocks) and RX 7600
// (32 RDNA 3 units on a 128-bit bus, behind an RX 6600 XT in reviews' ray traced tests).
fn on_slow_rt_list(adapter_name: &str) -> bool {
    if adapter_name.contains("Arc") {
        // "Intel(R) Arc(tm) A380 Graphics", "Intel(R) Arc(tm) A570M Graphics".
        return model_after(adapter_name, "Arc(tm) A")
            .or_else(|| model_after(adapter_name, "Arc A"))
            .is_some_and(|(model, _)| (300..600).contains(&model));
    }
    if let Some((model, _)) = model_after(adapter_name, "RTX ") {
        return matches!(model, 2050 | 3050)
            || (model % 100 == 50 && adapter_name.contains("Laptop"));
    }
    if let Some((model, rest)) = model_after(adapter_name, "RX ") {
        let xt = rest.starts_with("XT");
        // The RX 6500 XT is the only desktop 6500, and below the RX 6600; the 6600 XT and 7600 XT stay off.
        return (6400..6600).contains(&model) || (matches!(model, 6600 | 7600) && !xt);
    }
    false
}

/// An RTX 20 or 30 card: `GeForce` RTX 2050-3090 Ti, laptop parts included.
fn is_older_rtx(capabilities: &GraphicsCapabilities) -> bool {
    capabilities.vendor == GpuVendor::Nvidia
        && model_after(&capabilities.adapter_name, "RTX ")
            .is_some_and(|(model, _)| (2000..4000).contains(&model))
}

fn resolve_denoiser(
    rung: GraphicsRung,
    choice: DenoiserChoice,
    capabilities: &GraphicsCapabilities,
    fallbacks: &mut Vec<Fallback>,
) -> ResolvedDenoiser {
    if !rung.is_traced() {
        return ResolvedDenoiser::None;
    }
    // A traced rung is only reached with at least one of the two denoisers usable.
    //
    // Under `Auto` on `RtLite`, an RTX 20 or 30 card gets NRD and DLSS Super Resolution instead of DLSS Ray
    // Reconstruction. On the RTX 5070, RR is 2.2 ms of RtLite's 3.24 ms frame at 1080p, where NRD + TSR do
    // the same job in 1.21 + 0.39 ms (E2, O3). RR is a network on the tensor cores; on RTX 20/30 cards they
    // have a far smaller share of a 5070's throughput than the 0.36 shader ratio RtLite's gate assumes, so RR
    // alone could take half the frame there. Nobody measured it, and nobody now will (the hardware test was
    // dropped), so the measured, cheaper path is the default. The cost: NRD's slightly longer lag after a light
    // switches (1.7 % against RR's 1.1-1.3 %) and softer metal reflections. RTX 40/50 keep RR; `RtFull` keeps
    // RR on every card (its RTX 30 cards are the 3080 Ti and 3090 class); a player who asks for RR gets it.
    if choice == DenoiserChoice::Auto
        && rung == GraphicsRung::RtLite
        && capabilities.nrd
        && capabilities.dlss_ray_reconstruction
        && is_older_rtx(capabilities)
    {
        fallbacks.push(Fallback::NrdOnOlderRtx);
        return ResolvedDenoiser::Nrd;
    }
    match choice {
        DenoiserChoice::Nrd if capabilities.nrd => ResolvedDenoiser::Nrd,
        DenoiserChoice::Nrd => {
            fallbacks.push(Fallback::NrdUnavailable);
            ResolvedDenoiser::DlssRayReconstruction
        }
        DenoiserChoice::Auto | DenoiserChoice::DlssRayReconstruction => {
            if capabilities.dlss_ray_reconstruction {
                ResolvedDenoiser::DlssRayReconstruction
            } else {
                if choice == DenoiserChoice::DlssRayReconstruction {
                    fallbacks.push(Fallback::DlssRayReconstructionUnavailable);
                }
                ResolvedDenoiser::Nrd
            }
        }
    }
}

fn resolve_upscaler(
    choice: UpscalerChoice,
    denoiser: ResolvedDenoiser,
    wanted_quality: ResolvedQuality,
    capabilities: &GraphicsCapabilities,
    fallbacks: &mut Vec<Fallback>,
) -> ResolvedUpscaler {
    if denoiser == ResolvedDenoiser::DlssRayReconstruction {
        // The denoiser is settled first: it changes the picture far more than the upscaler does.
        if matches!(choice, UpscalerChoice::Tsr | UpscalerChoice::Fsr) {
            fallbacks.push(Fallback::RayReconstructionUpscales);
        }
        return ResolvedUpscaler::DlssRayReconstruction;
    }

    let fsr_usable = capabilities.fsr.is_some() && capabilities.vendor == GpuVendor::Amd;
    match choice {
        UpscalerChoice::Auto => {}
        UpscalerChoice::Off => return ResolvedUpscaler::Taa,
        UpscalerChoice::Tsr if capabilities.tsr => return ResolvedUpscaler::Tsr,
        UpscalerChoice::Tsr => fallbacks.push(Fallback::TsrUnavailable),
        UpscalerChoice::Dlss if capabilities.dlss_super_resolution => {
            return ResolvedUpscaler::DlssSuperResolution;
        }
        UpscalerChoice::Dlss => fallbacks.push(Fallback::DlssUnavailable),
        UpscalerChoice::Fsr if fsr_usable => return ResolvedUpscaler::Fsr,
        UpscalerChoice::Fsr => {
            fallbacks.push(if capabilities.fsr.is_none() {
                Fallback::FsrUnavailable
            } else {
                Fallback::FsrNeedsRadeon
            });
            // Whoever asks for FSR gets the other vendor-neutral upscaler when there is one.
            if capabilities.tsr {
                return ResolvedUpscaler::Tsr;
            }
        }
    }

    // `Auto`'s order, and where a choice that cannot run lands. FSR is not in it: it is code that has never
    // run on a machine we could test.
    if capabilities.dlss_super_resolution {
        ResolvedUpscaler::DlssSuperResolution
    } else if wanted_quality == ResolvedQuality::NativeAa {
        ResolvedUpscaler::Taa
    } else if capabilities.tsr {
        ResolvedUpscaler::Tsr
    } else {
        fallbacks.push(Fallback::NoUpscaler);
        ResolvedUpscaler::Taa
    }
}

/// The internal resolution each rung renders at when the player leaves the quality on `Auto`.
///
/// Measured on the testbed on the RTX 5070 (2026-11-24, E2): `RtLite` at Quality instead of Balanced at 1080p
/// output lowers its error against the path tracer (NRD + TSR: 2.55 → 2.46 % on `interior`, 3.54 → 3.38 % on
/// `many-lights`) and its motion error (2.06 → 2.01 %) for 0.35 ms of GPU on that card. It stays at Balanced,
/// the plan's value: those 0.35 ms were the 5070's headroom, and the rung is for the RTX 3060 / RX 6600 XT
/// class, where nothing measured protects them (review M1-4). At 1440p Balanced instead of Performance does the
/// same (3.41 → 3.22 %, motion 2.01 → 1.91 %) for 0.55 ms, and is kept. `RtFull` at native 1440p is no closer to
/// the path tracer than at Quality (DLSS-RR: 1.99 against 1.91 % on `interior`) and costs twice DLSS-RR's time;
/// at 1080p native it moves better than at Quality (motion 1.02 against 1.11 % with DLSS-RR, 1.73 against 2.03 %
/// with NRD) and stays far under its gate. Raster stays native at 1080p: with TAA it errs 1.10 % against a
/// frame supersampled from 3840x2160 on `interior`, TSR at Quality 1.17 %, and its GPU frame there (2.27 ms) is
/// under its 2.8 ms gate. Raster's light table renders lower: see [`raster_light`](crate::raster::raster_light).
///
/// An ultrawide 1440p output (3440x1440, 4.95 MP) is one step lower than 2560x1440, not 4K's Performance:
/// at Performance it would render fewer pixels (1720x720) than 2560x1440 does at Quality (1707x960).
fn auto_quality(rung: GraphicsRung, raster: RasterLevel, output: UVec2) -> ResolvedQuality {
    use ResolvedQuality::{Balanced, NativeAa, Performance, Quality};
    let pixels = u64::from(output.x) * u64::from(output.y);
    // The quality at up to 1080p, 1440p, ultrawide 1440p and above.
    let [p1080, p1440, ultrawide, above] = match (rung, raster) {
        (GraphicsRung::RtLite, _) => [Balanced, Balanced, Balanced, Performance],
        (GraphicsRung::Raster, RasterLevel::Light) => {
            [LIGHT_RASTER_1080P, Performance, Performance, Performance]
        }
        (GraphicsRung::Raster | GraphicsRung::RtFull, _) => {
            [NativeAa, Quality, Balanced, Performance]
        }
    };
    if pixels <= PIXELS_1080P {
        p1080
    } else if pixels <= PIXELS_1440P {
        p1440
    } else if pixels <= PIXELS_ULTRAWIDE_1440P {
        ultrawide
    } else {
        above
    }
}

/// Raster's light table at up to 1080p output. See [`raster_light`](crate::raster::raster_light) for the
/// measurement.
const LIGHT_RASTER_1080P: ResolvedQuality = ResolvedQuality::Balanced;

#[cfg(test)]
mod tests;
