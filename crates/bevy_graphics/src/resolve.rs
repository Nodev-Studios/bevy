//! The whole policy of the graphics ladder: [`resolve`] turns what was asked for and what the machine can do
//! into what will run, and says why the two differ.

use crate::{
    DenoiserChoice, GpuDeviceType, GpuVendor, GraphicsCapabilities, GraphicsPreset,
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
///   video memory, and picks `RtFull` only on a card with enough memory whose name is on a list.
/// - **Denoiser.** On a traced rung, DLSS Ray Reconstruction where it is supported, otherwise NRD; a choice
///   that cannot run falls back to the other one.
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
    let denoiser = resolve_denoiser(rung, settings.denoiser, capabilities, &mut fallbacks);
    let wanted_quality = match settings.upscale_quality {
        UpscaleQuality::Auto => auto_quality(rung, output),
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

/// Whether an adapter's name is on the list of cards assumed fast enough for `RtFull`.
// to tune: a guess from one card (the RTX 5070) and published ratios, until other cards are measured.
fn on_fast_rt_list(adapter_name: &str) -> bool {
    // GeForce RTX 4070 and up, RTX 5070 and up: an xx70 or higher of the 40 series or a later one.
    let geforce_model = adapter_name.split_once("RTX ").and_then(|(_, rest)| {
        let rest = rest.trim_start();
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        rest[..end].parse::<u32>().ok()
    });
    if let Some(model) = geforce_model {
        return model >= 4000 && model % 1000 >= 70;
    }
    // Radeon RX 9070 class, and RX 7900 XT and up (the XT and the XTX, not the GRE).
    adapter_name.contains("RX 9070") || adapter_name.contains("RX 7900 XT")
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
/// Measured on the testbed on the RTX 5070 (2026-11-24, E2): RtLite at Quality instead of Balanced at 1080p
/// output lowers its error against the path tracer (NRD + TSR: 2.55 → 2.46 % on `interior`, 3.54 → 3.38 % on
/// `many-lights`) and its motion error (2.06 → 2.01 %) for 0.35 ms of GPU, with the rung's frame far under its
/// gate; at 1440p Balanced instead of Performance does the same (3.41 → 3.22 %, motion 2.01 → 1.91 %) for
/// 0.55 ms. RtFull at native 1440p is no closer to the path tracer than at Quality (DLSS-RR: 1.99 against
/// 1.91 % on `interior`) and costs twice DLSS-RR's time. Raster stays native at 1080p: with TAA it errs 1.10 %
/// against a frame supersampled from 3840x2160 on `interior`, TSR at Quality 1.17 %, and its gate is met there.
// An ultrawide 1440p output has more pixels than 2560x1440 and is treated as 4K.
fn auto_quality(rung: GraphicsRung, output: UVec2) -> ResolvedQuality {
    let pixels = u64::from(output.x) * u64::from(output.y);
    match rung {
        GraphicsRung::RtLite => {
            if pixels <= PIXELS_1080P {
                ResolvedQuality::Quality
            } else if pixels <= PIXELS_1440P {
                ResolvedQuality::Balanced
            } else {
                ResolvedQuality::Performance
            }
        }
        GraphicsRung::Raster | GraphicsRung::RtFull => {
            if pixels <= PIXELS_1080P {
                ResolvedQuality::NativeAa
            } else if pixels <= PIXELS_1440P {
                ResolvedQuality::Quality
            } else {
                ResolvedQuality::Performance
            }
        }
    }
}

#[cfg(test)]
mod tests;
