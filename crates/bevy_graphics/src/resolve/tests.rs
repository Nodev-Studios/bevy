use super::*;
use crate::{FsrLevel, GraphicsOverrides};

const P1080: UVec2 = UVec2::new(1920, 1080);
const P1440: UVec2 = UVec2::new(2560, 1440);
const P2160: UVec2 = UVec2::new(3840, 2160);

/// A discrete card on Vulkan with ray query, in an app built with ray traced lighting and TSR, and no
/// denoiser.
fn card(name: &str, vendor: GpuVendor, gib: f64) -> GraphicsCapabilities {
    GraphicsCapabilities {
        adapter_name: name.into(),
        vendor,
        device_type: GpuDeviceType::Discrete,
        vulkan: true,
        video_memory_bytes: Some((gib * GIB as f64) as u64),
        solari: true,
        ray_query: true,
        tsr: true,
        ..Default::default()
    }
}

/// A `GeForce` RTX card in an app built with `dlss` and without NRD.
fn rtx(model: &str, gib: f64) -> GraphicsCapabilities {
    GraphicsCapabilities {
        dlss_super_resolution: true,
        dlss_ray_reconstruction: true,
        ..card(
            &format!("NVIDIA GeForce RTX {model}"),
            GpuVendor::Nvidia,
            gib,
        )
    }
}

/// A Radeon under Linux in an app built with NRD.
fn radeon(model: &str, gib: f64) -> GraphicsCapabilities {
    GraphicsCapabilities {
        nrd: true,
        ..card(&format!("AMD Radeon RX {model}"), GpuVendor::Amd, gib)
    }
}

/// The three adapters of the machine this was written on, as `vulkaninfo` reported them on 2026-11-24, in an
/// app built with `dlss` and NRD.
fn rtx_5070() -> GraphicsCapabilities {
    GraphicsCapabilities {
        video_memory_bytes: Some(12_820_938_752),
        nrd: true,
        ..rtx("5070", 0.0)
    }
}

fn ryzen_integrated() -> GraphicsCapabilities {
    GraphicsCapabilities {
        device_type: GpuDeviceType::Integrated,
        video_memory_bytes: Some(11_258_961_920),
        ..radeon("integrated", 0.0)
    }
}

fn llvmpipe() -> GraphicsCapabilities {
    GraphicsCapabilities {
        device_type: GpuDeviceType::Cpu,
        video_memory_bytes: Some(32_703_139_840),
        nrd: true,
        ..card(
            "llvmpipe (LLVM 23.1.5, 256 bits)",
            GpuVendor::Other(0x10005),
            0.0,
        )
    }
}

/// A card without ray tracing hardware.
fn gtx_1060() -> GraphicsCapabilities {
    GraphicsCapabilities {
        ray_query: false,
        ..card("NVIDIA GeForce GTX 1060 6GB", GpuVendor::Nvidia, 5.9)
    }
}

fn auto() -> GraphicsSettings {
    GraphicsSettings::default()
}

fn preset(preset: GraphicsPreset) -> GraphicsSettings {
    GraphicsSettings {
        preset,
        ..Default::default()
    }
}

fn upscaler(preset: GraphicsPreset, upscaler: UpscalerChoice) -> GraphicsSettings {
    GraphicsSettings {
        preset,
        upscaler,
        ..Default::default()
    }
}

fn denoiser(denoiser: DenoiserChoice) -> GraphicsSettings {
    GraphicsSettings {
        denoiser,
        ..Default::default()
    }
}

fn quality(preset: GraphicsPreset, upscale_quality: UpscaleQuality) -> GraphicsSettings {
    GraphicsSettings {
        preset,
        upscale_quality,
        ..Default::default()
    }
}

// Short names so that a row's expected result stays on one line.
const D_RR: ResolvedDenoiser = ResolvedDenoiser::DlssRayReconstruction;
const U_RR: ResolvedUpscaler = ResolvedUpscaler::DlssRayReconstruction;
const U_SR: ResolvedUpscaler = ResolvedUpscaler::DlssSuperResolution;

struct Row {
    name: &'static str,
    settings: GraphicsSettings,
    capabilities: GraphicsCapabilities,
    output: UVec2,
    rung: GraphicsRung,
    denoiser: ResolvedDenoiser,
    upscaler: ResolvedUpscaler,
    quality: ResolvedQuality,
    /// Light exactly when the row's fallbacks say so.
    raster: RasterLevel,
    fallbacks: &'static [Fallback],
}

#[expect(
    clippy::too_many_lines,
    reason = "A table: one row per case, read top to bottom."
)]
fn rows() -> Vec<Row> {
    use Fallback as F;
    use GraphicsPreset as P;
    use GraphicsRung::{Raster, RtFull, RtLite};
    use ResolvedDenoiser as D;
    use ResolvedQuality::{Balanced, NativeAa, Performance, Quality};
    use ResolvedUpscaler as U;

    fn row(
        name: &'static str,
        settings: GraphicsSettings,
        capabilities: GraphicsCapabilities,
        output: UVec2,
        (rung, denoiser, upscaler, quality): (
            GraphicsRung,
            ResolvedDenoiser,
            ResolvedUpscaler,
            ResolvedQuality,
        ),
        fallbacks: &'static [Fallback],
    ) -> Row {
        Row {
            name,
            settings,
            capabilities,
            output,
            rung,
            denoiser,
            upscaler,
            quality,
            raster: if fallbacks.contains(&Fallback::LightRaster) {
                RasterLevel::Light
            } else {
                RasterLevel::Full
            },
            fallbacks,
        }
    }

    vec![
        // `Auto`'s six rules, each on the machine that trips it and nothing before it.
        row(
            "rule 1: not Vulkan",
            auto(),
            GraphicsCapabilities {
                vulkan: false,
                ..radeon("9070 XT", 15.9)
            },
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::NotVulkan],
        ),
        row(
            "rule 1: CPU rasterizer (this machine's llvmpipe)",
            auto(),
            llvmpipe(),
            P1080,
            (Raster, D::None, U::Tsr, Balanced),
            &[F::SoftwareAdapter, F::LightRaster],
        ),
        row(
            "rule 2: built without ray traced lighting",
            auto(),
            GraphicsCapabilities {
                solari: false,
                ray_query: false,
                ..rtx_5070()
            },
            P1080,
            (Raster, D::None, U_SR, NativeAa),
            &[F::RayTracingNotCompiled],
        ),
        row(
            "rule 2: no ray query",
            auto(),
            GraphicsCapabilities {
                nrd: true,
                ..gtx_1060()
            },
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::NoRayQuery],
        ),
        row(
            "rule 3: integrated GPU (this machine's Ryzen)",
            auto(),
            ryzen_integrated(),
            P1080,
            (Raster, D::None, U::Tsr, Balanced),
            &[F::IntegratedGpu, F::LightRaster],
        ),
        row(
            "rule 3: virtual GPU",
            auto(),
            GraphicsCapabilities {
                device_type: GpuDeviceType::Virtual,
                ..rtx_5070()
            },
            P1080,
            (Raster, D::None, U_SR, Balanced),
            &[F::IntegratedGpu, F::LightRaster],
        ),
        row(
            "rule 4: no denoiser (a Radeon before NRD is in the app)",
            auto(),
            GraphicsCapabilities {
                nrd: false,
                ..radeon("9070 XT", 15.9)
            },
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::NoDenoiser],
        ),
        row(
            "rule 5: 5.4 GiB",
            auto(),
            rtx("2060", 5.4),
            P1080,
            (Raster, D::None, U_SR, Balanced),
            &[F::LowVideoMemory, F::LightRaster],
        ),
        row(
            "rule 5: video memory unknown (full table: unknown is not judged weak)",
            auto(),
            GraphicsCapabilities {
                video_memory_bytes: None,
                ..rtx_5070()
            },
            P1080,
            (Raster, D::None, U_SR, NativeAa),
            &[F::LowVideoMemory],
        ),
        row(
            "rule 6: 5.9 GiB, a 6 GB card",
            auto(),
            rtx("2060", 5.9),
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[F::VideoMemoryBelowRtFull],
        ),
        row(
            "rule 6: 7.9 GiB, an 8 GB card",
            auto(),
            rtx("4060", 7.9),
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[F::VideoMemoryBelowRtFull],
        ),
        row(
            "rule 6: 10.9 GiB on the fast list is still under 11",
            auto(),
            rtx("4070", 10.9),
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[F::VideoMemoryBelowRtFull],
        ),
        row(
            "rule 6: 11.9 GiB, a 12 GB card on the fast list",
            auto(),
            rtx("4070 SUPER", 11.9),
            P1080,
            (RtFull, D_RR, U_RR, NativeAa),
            &[],
        ),
        row(
            "rule 6: this machine's RTX 5070, 11.94 GiB",
            auto(),
            rtx_5070(),
            P1080,
            (RtFull, D_RR, U_RR, NativeAa),
            &[],
        ),
        row(
            "rule 6: a 12 GB card off the fast list (RTX 3060)",
            auto(),
            rtx("3060", 11.9),
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[F::NotOnFastRtList],
        ),
        row(
            "rule 6: a 16 GB card off the fast list (RTX 5060 Ti)",
            auto(),
            rtx("5060 Ti", 15.9),
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[F::NotOnFastRtList],
        ),
        row(
            "rule 6: a 16 GB Radeon off the fast list (RX 9060 XT)",
            auto(),
            radeon("9060 XT", 15.9),
            P1080,
            (RtLite, D::Nrd, U::Tsr, Balanced),
            &[F::NotOnFastRtList],
        ),
        row(
            "rule 6: a Radeon on the fast list (RX 9070 XT)",
            auto(),
            radeon("9070 XT", 15.9),
            P1440,
            (RtFull, D::Nrd, U::Tsr, Quality),
            &[],
        ),
        row(
            "rule 6: RX 7900 XTX is on the fast list",
            auto(),
            radeon("7900 XTX", 23.9),
            P1440,
            (RtFull, D::Nrd, U::Tsr, Quality),
            &[],
        ),
        row(
            "rule 6: RX 7900 GRE is not",
            auto(),
            radeon("7900 GRE", 15.9),
            P1440,
            (RtLite, D::Nrd, U::Tsr, Balanced),
            &[F::NotOnFastRtList],
        ),
        // The rules are tried in order: the first that fails gives the reason.
        row(
            "order: rule 1 before 2 (no Vulkan and no ray query)",
            auto(),
            GraphicsCapabilities {
                vulkan: false,
                ..gtx_1060()
            },
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::NotVulkan],
        ),
        row(
            "order: rule 2 before 3 (integrated GPU without ray query)",
            auto(),
            GraphicsCapabilities {
                ray_query: false,
                ..ryzen_integrated()
            },
            P1080,
            (Raster, D::None, U::Tsr, Balanced),
            &[F::NoRayQuery, F::LightRaster],
        ),
        row(
            "order: rule 3 before 4 (integrated GPU without a denoiser)",
            auto(),
            GraphicsCapabilities {
                nrd: false,
                ..ryzen_integrated()
            },
            P1080,
            (Raster, D::None, U::Tsr, Balanced),
            &[F::IntegratedGpu, F::LightRaster],
        ),
        row(
            "order: rule 4 before 5 (small card without a denoiser)",
            auto(),
            GraphicsCapabilities {
                nrd: false,
                ..radeon("6600 XT", 5.4)
            },
            P1080,
            (Raster, D::None, U::Tsr, Balanced),
            &[F::NoDenoiser, F::LightRaster],
        ),
        row(
            "no renderer at all",
            auto(),
            GraphicsCapabilities::default(),
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::NotVulkan],
        ),
        // A rung asked for by name is given wherever it can run at all.
        row(
            "RtFull by name on the integrated GPU",
            preset(P::RtFull),
            ryzen_integrated(),
            P1080,
            (RtFull, D::Nrd, U::Taa, NativeAa),
            &[],
        ),
        row(
            "RtLite by name on the CPU rasterizer",
            preset(P::RtLite),
            llvmpipe(),
            P1080,
            (RtLite, D::Nrd, U::Tsr, Balanced),
            &[],
        ),
        row(
            "RtFull by name on a 5.4 GiB card off the fast list",
            preset(P::RtFull),
            rtx("2060", 5.4),
            P1080,
            (RtFull, D_RR, U_RR, NativeAa),
            &[],
        ),
        row(
            "RtLite by name on the RTX 5070",
            preset(P::RtLite),
            rtx_5070(),
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[],
        ),
        row(
            "Raster by name on the RTX 5070",
            preset(P::Raster),
            rtx_5070(),
            P1440,
            (Raster, D::None, U_SR, Quality),
            &[],
        ),
        row(
            "RtFull by name without ray query",
            preset(P::RtFull),
            gtx_1060(),
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::NoRayQuery],
        ),
        row(
            "RtLite by name without a denoiser",
            preset(P::RtLite),
            GraphicsCapabilities {
                nrd: false,
                ..radeon("9070 XT", 15.9)
            },
            P1440,
            (Raster, D::None, U::Tsr, Quality),
            &[F::NoDenoiser],
        ),
        row(
            "RtFull by name off Vulkan",
            preset(P::RtFull),
            GraphicsCapabilities {
                vulkan: false,
                ..rtx_5070()
            },
            P1080,
            (Raster, D::None, U_SR, NativeAa),
            &[F::NotVulkan],
        ),
        row(
            "RtFull by name in an app built without ray traced lighting",
            preset(P::RtFull),
            GraphicsCapabilities {
                solari: false,
                ray_query: false,
                ..rtx_5070()
            },
            P1080,
            (Raster, D::None, U_SR, NativeAa),
            &[F::RayTracingNotCompiled],
        ),
        // The denoiser.
        row(
            "NRD + DLSS SR on an RTX card without DLSS-RR",
            auto(),
            GraphicsCapabilities {
                dlss_ray_reconstruction: false,
                ..rtx_5070()
            },
            P1440,
            (RtFull, D::Nrd, U_SR, Quality),
            &[],
        ),
        row(
            "NRD by name on an RTX card: DLSS SR follows it",
            denoiser(DenoiserChoice::Nrd),
            rtx_5070(),
            P1440,
            (RtFull, D::Nrd, U_SR, Quality),
            &[],
        ),
        row(
            "NRD by name, not in the app",
            denoiser(DenoiserChoice::Nrd),
            rtx("5070", 11.94),
            P1440,
            (RtFull, D_RR, U_RR, Quality),
            &[F::NrdUnavailable],
        ),
        row(
            "DLSS-RR by name on a Radeon",
            denoiser(DenoiserChoice::DlssRayReconstruction),
            radeon("9070 XT", 15.9),
            P1440,
            (RtFull, D::Nrd, U::Tsr, Quality),
            &[F::DlssRayReconstructionUnavailable],
        ),
        row(
            "a denoiser choice means nothing on Raster",
            GraphicsSettings {
                preset: P::Raster,
                denoiser: DenoiserChoice::Nrd,
                ..Default::default()
            },
            rtx("5070", 11.94),
            P1080,
            (Raster, D::None, U_SR, NativeAa),
            &[],
        ),
        // FSR: never under `Auto`, and by name only on a Radeon under Windows with the DLL.
        row(
            "FSR is not Auto's choice on a Radeon under Windows, Raster",
            auto(),
            GraphicsCapabilities {
                nrd: false,
                fsr: Some(FsrLevel::Fsr4),
                ..radeon("9070 XT", 15.9)
            },
            P2160,
            (Raster, D::None, U::Tsr, Performance),
            &[F::NoDenoiser],
        ),
        row(
            "FSR is not Auto's choice on a Radeon under Windows, traced",
            auto(),
            GraphicsCapabilities {
                fsr: Some(FsrLevel::Fsr4),
                ..radeon("9070 XT", 15.9)
            },
            P2160,
            (RtFull, D::Nrd, U::Tsr, Performance),
            &[],
        ),
        row(
            "FSR is not Auto's choice even when it is the only upscaler",
            auto(),
            GraphicsCapabilities {
                tsr: false,
                fsr: Some(FsrLevel::Fsr4),
                ..radeon("9070 XT", 15.9)
            },
            P2160,
            (RtFull, D::Nrd, U::Taa, NativeAa),
            &[F::NoUpscaler],
        ),
        row(
            "FSR by name on a Radeon under Windows with the DLL",
            upscaler(P::Auto, UpscalerChoice::Fsr),
            GraphicsCapabilities {
                fsr: Some(FsrLevel::Fsr4),
                ..radeon("9070 XT", 15.9)
            },
            P1440,
            (RtFull, D::Nrd, U::Fsr, Quality),
            &[],
        ),
        row(
            "FSR 3.1 by name on an older Radeon, Raster",
            upscaler(P::Raster, UpscalerChoice::Fsr),
            GraphicsCapabilities {
                fsr: Some(FsrLevel::Fsr3),
                ..radeon("580", 7.9)
            },
            P1440,
            (Raster, D::None, U::Fsr, Quality),
            &[],
        ),
        row(
            "FSR by name on a Radeon under Linux, or without the DLL",
            upscaler(P::Auto, UpscalerChoice::Fsr),
            radeon("9070 XT", 15.9),
            P1440,
            (RtFull, D::Nrd, U::Tsr, Quality),
            &[F::FsrUnavailable],
        ),
        row(
            "FSR by name on a GeForce with the DLL: TSR, not DLSS",
            upscaler(P::Raster, UpscalerChoice::Fsr),
            GraphicsCapabilities {
                fsr: Some(FsrLevel::Fsr3),
                ..rtx_5070()
            },
            P1440,
            (Raster, D::None, U::Tsr, Quality),
            &[F::FsrNeedsRadeon],
        ),
        row(
            "FSR by name on a GeForce without TSR in the app",
            upscaler(P::Raster, UpscalerChoice::Fsr),
            GraphicsCapabilities {
                tsr: false,
                ..rtx_5070()
            },
            P1440,
            (Raster, D::None, U_SR, Quality),
            &[F::FsrUnavailable],
        ),
        row(
            "FSR by name where DLSS-RR denoises",
            upscaler(P::Auto, UpscalerChoice::Fsr),
            GraphicsCapabilities {
                fsr: Some(FsrLevel::Fsr3),
                ..rtx_5070()
            },
            P1440,
            (RtFull, D_RR, U_RR, Quality),
            &[F::RayReconstructionUpscales],
        ),
        // The other upscaler choices.
        row(
            "TSR by name where DLSS-RR denoises",
            upscaler(P::Auto, UpscalerChoice::Tsr),
            rtx_5070(),
            P1440,
            (RtFull, D_RR, U_RR, Quality),
            &[F::RayReconstructionUpscales],
        ),
        row(
            "TSR by name after NRD on an RTX card",
            GraphicsSettings {
                upscaler: UpscalerChoice::Tsr,
                denoiser: DenoiserChoice::Nrd,
                ..Default::default()
            },
            rtx_5070(),
            P1440,
            (RtFull, D::Nrd, U::Tsr, Quality),
            &[],
        ),
        row(
            "TSR by name, not in the app, on an RTX card",
            upscaler(P::Raster, UpscalerChoice::Tsr),
            GraphicsCapabilities {
                tsr: false,
                ..rtx_5070()
            },
            P1440,
            (Raster, D::None, U_SR, Quality),
            &[F::TsrUnavailable],
        ),
        row(
            "TSR by name, not in the app, with nothing else",
            upscaler(P::Raster, UpscalerChoice::Tsr),
            GraphicsCapabilities {
                tsr: false,
                ..gtx_1060()
            },
            P1440,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::TsrUnavailable, F::NoUpscaler],
        ),
        row(
            "TSR by name at native keeps TSR",
            upscaler(P::Raster, UpscalerChoice::Tsr),
            gtx_1060(),
            P1080,
            (Raster, D::None, U::Tsr, NativeAa),
            &[],
        ),
        row(
            "DLSS by name on a Radeon",
            upscaler(P::Raster, UpscalerChoice::Dlss),
            radeon("9070 XT", 15.9),
            P1440,
            (Raster, D::None, U::Tsr, Quality),
            &[F::DlssUnavailable],
        ),
        row(
            "DLSS by name where DLSS-RR denoises is DLSS-RR",
            upscaler(P::Auto, UpscalerChoice::Dlss),
            rtx_5070(),
            P1440,
            (RtFull, D_RR, U_RR, Quality),
            &[],
        ),
        row(
            "upscaling off on Raster at 4K",
            upscaler(P::Raster, UpscalerChoice::Off),
            rtx_5070(),
            P2160,
            (Raster, D::None, U::Taa, NativeAa),
            &[],
        ),
        row(
            "upscaling off with DLSS-RR: DLSS-RR at native",
            upscaler(P::Auto, UpscalerChoice::Off),
            rtx_5070(),
            P2160,
            (RtFull, D_RR, U_RR, NativeAa),
            &[],
        ),
        row(
            "upscaling off after NRD: TAA at native",
            GraphicsSettings {
                upscaler: UpscalerChoice::Off,
                denoiser: DenoiserChoice::Nrd,
                ..Default::default()
            },
            rtx_5070(),
            P2160,
            (RtFull, D::Nrd, U::Taa, NativeAa),
            &[],
        ),
        row(
            "no upscaler in the app at 4K",
            preset(P::Raster),
            GraphicsCapabilities {
                tsr: false,
                ..gtx_1060()
            },
            P2160,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::NoUpscaler],
        ),
        // The internal resolution under `Auto`, per rung, at 1080p, 1440p and 4K.
        row(
            "Raster 1080p without DLSS: TAA at native",
            preset(P::Raster),
            gtx_1060(),
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[],
        ),
        row(
            "Raster 1440p without DLSS: TSR Quality",
            preset(P::Raster),
            gtx_1060(),
            P1440,
            (Raster, D::None, U::Tsr, Quality),
            &[],
        ),
        row(
            "Raster 4K without DLSS: TSR Performance",
            preset(P::Raster),
            gtx_1060(),
            P2160,
            (Raster, D::None, U::Tsr, Performance),
            &[],
        ),
        row(
            "Raster 1080p with DLSS: DLSS SR at native",
            preset(P::Raster),
            rtx_5070(),
            P1080,
            (Raster, D::None, U_SR, NativeAa),
            &[],
        ),
        row(
            "Raster 4K with DLSS: DLSS SR Performance",
            preset(P::Raster),
            rtx_5070(),
            P2160,
            (Raster, D::None, U_SR, Performance),
            &[],
        ),
        row(
            "RtLite 1080p: Balanced (the plan's value, not the 5070's Quality)",
            preset(P::RtLite),
            radeon("7700 XT", 11.9),
            P1080,
            (RtLite, D::Nrd, U::Tsr, Balanced),
            &[],
        ),
        row(
            "RtLite 1440p: Balanced",
            preset(P::RtLite),
            radeon("7700 XT", 11.9),
            P1440,
            (RtLite, D::Nrd, U::Tsr, Balanced),
            &[],
        ),
        row(
            "RtLite 4K: Performance",
            preset(P::RtLite),
            rtx_5070(),
            P2160,
            (RtLite, D_RR, U_RR, Performance),
            &[],
        ),
        row(
            "RtFull 1080p: native",
            preset(P::RtFull),
            rtx_5070(),
            P1080,
            (RtFull, D_RR, U_RR, NativeAa),
            &[],
        ),
        row(
            "RtFull 1440p: Quality",
            preset(P::RtFull),
            rtx_5070(),
            P1440,
            (RtFull, D_RR, U_RR, Quality),
            &[],
        ),
        row(
            "RtFull 4K: Performance",
            preset(P::RtFull),
            rtx_5070(),
            P2160,
            (RtFull, D_RR, U_RR, Performance),
            &[],
        ),
        row(
            "RtFull 1080p after NRD without DLSS: TAA at native",
            preset(P::RtFull),
            radeon("9070 XT", 15.9),
            P1080,
            (RtFull, D::Nrd, U::Taa, NativeAa),
            &[],
        ),
        row(
            "ultrawide 1440p: Balanced, one step under 2560x1440, not 4K's Performance",
            preset(P::RtFull),
            rtx_5070(),
            UVec2::new(3440, 1440),
            (RtFull, D_RR, U_RR, Balanced),
            &[],
        ),
        // A quality asked for by name.
        row(
            "Performance by name at 1080p",
            quality(P::Raster, UpscaleQuality::Performance),
            rtx_5070(),
            P1080,
            (Raster, D::None, U_SR, Performance),
            &[],
        ),
        row(
            "native by name at 4K",
            quality(P::RtFull, UpscaleQuality::NativeAa),
            rtx_5070(),
            P2160,
            (RtFull, D_RR, U_RR, NativeAa),
            &[],
        ),
        row(
            "native by name at 4K without DLSS: TAA, and nothing was refused",
            quality(P::Raster, UpscaleQuality::NativeAa),
            gtx_1060(),
            P2160,
            (Raster, D::None, U::Taa, NativeAa),
            &[],
        ),
        row(
            "Balanced by name on RtFull",
            quality(P::RtFull, UpscaleQuality::Balanced),
            radeon("9070 XT", 15.9),
            P1440,
            (RtFull, D::Nrd, U::Tsr, Balanced),
            &[],
        ),
        row(
            "Quality by name with upscaling off: off wins",
            GraphicsSettings {
                preset: P::Raster,
                upscaler: UpscalerChoice::Off,
                upscale_quality: UpscaleQuality::Quality,
                ..Default::default()
            },
            rtx_5070(),
            P1440,
            (Raster, D::None, U::Taa, NativeAa),
            &[],
        ),
        // Raster's light table: `Auto` on an integrated GPU or a card under 5.5 GiB (M3-1, M2-12).
        row(
            "light Raster at 1440p: Performance",
            auto(),
            ryzen_integrated(),
            P1440,
            (Raster, D::None, U::Tsr, Performance),
            &[F::IntegratedGpu, F::LightRaster],
        ),
        row(
            "light Raster on a 4 GB RTX laptop: DLSS SR at Balanced, not DLAA",
            auto(),
            rtx("3050 Laptop GPU", 3.9),
            P1080,
            (Raster, D::None, U_SR, Balanced),
            &[F::LowVideoMemory, F::LightRaster],
        ),
        row(
            "light Raster on a 4 GB card without ray query",
            auto(),
            GraphicsCapabilities {
                ray_query: false,
                ..card("NVIDIA GeForce GTX 1050 Ti", GpuVendor::Nvidia, 3.9)
            },
            P1080,
            (Raster, D::None, U::Tsr, Balanced),
            &[F::NoRayQuery, F::LightRaster],
        ),
        row(
            "light Raster: an app without TSR renders native with TAA",
            auto(),
            GraphicsCapabilities {
                tsr: false,
                ..ryzen_integrated()
            },
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::IntegratedGpu, F::LightRaster, F::NoUpscaler],
        ),
        row(
            "light Raster: the player's quality by name wins",
            quality(P::Auto, UpscaleQuality::Quality),
            ryzen_integrated(),
            P1080,
            (Raster, D::None, U::Tsr, Quality),
            &[F::IntegratedGpu, F::LightRaster],
        ),
        row(
            "light Raster: upscaling off by name renders native",
            upscaler(P::Auto, UpscalerChoice::Off),
            ryzen_integrated(),
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::IntegratedGpu, F::LightRaster],
        ),
        row(
            "Raster by name on the integrated GPU keeps the full table",
            preset(P::Raster),
            ryzen_integrated(),
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[],
        ),
        // The slow-RT list: cards with ray tracing that `Auto` sends to Raster's full table (M2-1).
        row(
            "slow RT: RTX 3050 8 GB",
            auto(),
            rtx("3050", 7.9),
            P1080,
            (Raster, D::None, U_SR, NativeAa),
            &[F::SlowRayTracing],
        ),
        row(
            "slow RT: RTX 4050 Laptop GPU 6 GB",
            auto(),
            rtx("4050 Laptop GPU", 5.9),
            P1080,
            (Raster, D::None, U_SR, NativeAa),
            &[F::SlowRayTracing],
        ),
        row(
            "slow RT: RX 6600",
            auto(),
            radeon("6600", 7.9),
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::SlowRayTracing],
        ),
        row(
            "slow RT: RX 7600 (RADV)",
            auto(),
            GraphicsCapabilities {
                adapter_name: "AMD Radeon RX 7600 (RADV NAVI33)".into(),
                ..radeon("7600", 7.9)
            },
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::SlowRayTracing],
        ),
        row(
            "slow RT: Arc A580",
            auto(),
            GraphicsCapabilities {
                nrd: true,
                ..card("Intel(R) Arc(tm) A580 Graphics (DG2)", GpuVendor::Intel, 7.9)
            },
            P1080,
            (Raster, D::None, U::Taa, NativeAa),
            &[F::SlowRayTracing],
        ),
        row(
            "not slow: RX 6600 XT reaches RtLite",
            auto(),
            radeon("6600 XT", 7.9),
            P1080,
            (RtLite, D::Nrd, U::Tsr, Balanced),
            &[F::VideoMemoryBelowRtFull],
        ),
        row(
            "not slow: Arc A770 16 GB reaches RtLite",
            auto(),
            GraphicsCapabilities {
                nrd: true,
                ..card("Intel(R) Arc(tm) A770 Graphics (DG2)", GpuVendor::Intel, 15.9)
            },
            P1080,
            (RtLite, D::Nrd, U::Tsr, Balanced),
            &[F::NotOnFastRtList],
        ),
        row(
            "slow RT by name: RtLite asked on an RTX 3050 is given",
            preset(P::RtLite),
            rtx("3050", 7.9),
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[],
        ),
        // `Auto`'s denoiser on RtLite: NRD + DLSS SR on RTX 20/30, Ray Reconstruction on RTX 40/50 (M1-1).
        row(
            "RtLite on an RTX 3060 with NRD in the app: NRD and DLSS SR",
            auto(),
            GraphicsCapabilities {
                nrd: true,
                ..rtx("3060", 11.9)
            },
            P1080,
            (RtLite, D::Nrd, U_SR, Balanced),
            &[F::NotOnFastRtList, F::NrdOnOlderRtx],
        ),
        row(
            "RtLite on an RTX 2060 6 GB with NRD in the app: NRD and DLSS SR",
            auto(),
            GraphicsCapabilities {
                nrd: true,
                ..rtx("2060", 5.9)
            },
            P1080,
            (RtLite, D::Nrd, U_SR, Balanced),
            &[F::VideoMemoryBelowRtFull, F::NrdOnOlderRtx],
        ),
        row(
            "RtLite on an RTX 4060 with NRD in the app keeps Ray Reconstruction",
            auto(),
            GraphicsCapabilities {
                nrd: true,
                ..rtx("4060", 7.9)
            },
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[F::VideoMemoryBelowRtFull],
        ),
        row(
            "Ray Reconstruction by name on an RTX 3060 is given",
            GraphicsSettings {
                denoiser: DenoiserChoice::DlssRayReconstruction,
                ..Default::default()
            },
            GraphicsCapabilities {
                nrd: true,
                ..rtx("3060", 11.9)
            },
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[F::NotOnFastRtList],
        ),
        row(
            "RtLite on an RTX 3060 without NRD in the app keeps Ray Reconstruction",
            auto(),
            rtx("3060", 11.9),
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[F::NotOnFastRtList],
        ),
        // The fast-RT list's additions (M1-6, M2-10).
        row(
            "fast RT: RTX 3090 24 GB reaches RtFull and keeps Ray Reconstruction",
            auto(),
            GraphicsCapabilities {
                nrd: true,
                ..rtx("3090", 23.9)
            },
            P1080,
            (RtFull, D_RR, U_RR, NativeAa),
            &[],
        ),
        row(
            "fast RT: RTX 3080 Ti 12 GB reaches RtFull",
            auto(),
            rtx("3080 Ti", 11.9),
            P1080,
            (RtFull, D_RR, U_RR, NativeAa),
            &[],
        ),
        row(
            "fast RT: an RTX 4080 Laptop GPU 12 GB is one rung lower",
            auto(),
            rtx("4080 Laptop GPU", 11.9),
            P1080,
            (RtLite, D_RR, U_RR, Balanced),
            &[F::NotOnFastRtList],
        ),
        row(
            "fast RT: an RX 9070 XT that RADV names by its chip",
            auto(),
            GraphicsCapabilities {
                adapter_name: "AMD Radeon Graphics (RADV GFX1201)".into(),
                ..radeon("9070 XT", 15.9)
            },
            P1080,
            (RtFull, D::Nrd, U::Taa, NativeAa),
            &[],
        ),
        // Ultrawide (M2-11).
        row(
            "ultrawide 1440p on Raster: Balanced",
            preset(P::Raster),
            rtx_5070(),
            UVec2::new(3440, 1440),
            (Raster, D::None, U_SR, Balanced),
            &[],
        ),
        row(
            "ultrawide 1440p on RtLite: Balanced",
            preset(P::RtLite),
            radeon("7700 XT", 11.9),
            UVec2::new(3440, 1440),
            (RtLite, D::Nrd, U::Tsr, Balanced),
            &[],
        ),
        row(
            "ultrawide 1600p is over the bucket: Performance",
            preset(P::RtFull),
            rtx_5070(),
            UVec2::new(3840, 1600),
            (RtFull, D_RR, U_RR, Performance),
            &[],
        ),
    ]
}

#[test]
fn resolve_table() {
    let rows = rows();
    assert!(rows.len() >= 30, "the table has {} rows", rows.len());

    let mut failures = Vec::new();
    for row in &rows {
        let resolved = resolve(&row.settings, &row.capabilities, row.output);
        let expected = ResolvedGraphics {
            rung: row.rung,
            upscaler: row.upscaler,
            denoiser: row.denoiser,
            quality: row.quality,
            raster: row.raster,
            render_scale: row.quality.scale(),
            fallbacks: row.fallbacks.to_vec(),
            unfitted_meshes: 0,
            skipped_deforming_meshes: 0,
        };
        if resolved != expected {
            failures.push(format!(
                "{}:\n  expected {expected:?}\n       got {resolved:?}",
                row.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} rows failed\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}

/// FSR is never the result unless it was asked for by name, whatever the machine.
#[test]
fn fsr_only_by_name() {
    let machines = [
        rtx_5070(),
        ryzen_integrated(),
        llvmpipe(),
        gtx_1060(),
        radeon("9070 XT", 15.9),
        radeon("580", 7.9),
    ];
    let presets = [
        GraphicsPreset::Auto,
        GraphicsPreset::Raster,
        GraphicsPreset::RtLite,
        GraphicsPreset::RtFull,
    ];
    let upscalers = [
        UpscalerChoice::Auto,
        UpscalerChoice::Off,
        UpscalerChoice::Tsr,
        UpscalerChoice::Dlss,
    ];
    for machine in machines {
        for fsr in [None, Some(FsrLevel::Fsr3), Some(FsrLevel::Fsr4)] {
            for tsr in [false, true] {
                let capabilities = GraphicsCapabilities {
                    fsr,
                    tsr,
                    ..machine.clone()
                };
                for preset in presets {
                    for upscaler in upscalers {
                        let settings = GraphicsSettings {
                            preset,
                            upscaler,
                            ..Default::default()
                        };
                        let resolved = resolve(&settings, &capabilities, P2160);
                        assert_ne!(
                            resolved.upscaler,
                            ResolvedUpscaler::Fsr,
                            "{settings:?} on {capabilities:?}"
                        );
                    }
                }
            }
        }
    }
}

/// Whatever is asked for on whatever machine, the result can run there.
#[test]
fn resolved_is_always_runnable() {
    let machines = [
        GraphicsCapabilities::default(),
        rtx_5070(),
        rtx("2060", 5.4),
        ryzen_integrated(),
        llvmpipe(),
        gtx_1060(),
        radeon("9070 XT", 15.9),
        GraphicsCapabilities {
            fsr: Some(FsrLevel::Fsr4),
            nrd: false,
            tsr: false,
            ..radeon("9070 XT", 15.9)
        },
        GraphicsCapabilities {
            dlss_ray_reconstruction: false,
            ..rtx("3060", 11.9)
        },
    ];
    let presets = [
        GraphicsPreset::Auto,
        GraphicsPreset::Raster,
        GraphicsPreset::RtLite,
        GraphicsPreset::RtFull,
    ];
    let upscalers = [
        UpscalerChoice::Auto,
        UpscalerChoice::Off,
        UpscalerChoice::Tsr,
        UpscalerChoice::Dlss,
        UpscalerChoice::Fsr,
    ];
    let qualities = [
        UpscaleQuality::Auto,
        UpscaleQuality::NativeAa,
        UpscaleQuality::Quality,
        UpscaleQuality::Balanced,
        UpscaleQuality::Performance,
    ];
    let denoisers = [
        DenoiserChoice::Auto,
        DenoiserChoice::DlssRayReconstruction,
        DenoiserChoice::Nrd,
    ];
    for capabilities in &machines {
        for preset in presets {
            for upscaler in upscalers {
                for upscale_quality in qualities {
                    for denoiser in denoisers {
                        let settings = GraphicsSettings {
                            preset,
                            upscaler,
                            upscale_quality,
                            denoiser,
                            overrides: GraphicsOverrides::default(),
                            ..Default::default()
                        };
                        let resolved = resolve(&settings, capabilities, P1440);
                        let context = format!("{settings:?} on {capabilities:?} gave {resolved:?}");

                        if resolved.rung.is_traced() {
                            assert!(
                                capabilities.vulkan
                                    && capabilities.solari
                                    && capabilities.ray_query,
                                "{context}"
                            );
                        }
                        // A traced rung always has a denoiser, Raster never, and each is one the machine has.
                        match resolved.denoiser {
                            ResolvedDenoiser::None => {
                                assert!(!resolved.rung.is_traced(), "{context}");
                            }
                            ResolvedDenoiser::DlssRayReconstruction => assert!(
                                resolved.rung.is_traced() && capabilities.dlss_ray_reconstruction,
                                "{context}"
                            ),
                            ResolvedDenoiser::Nrd => {
                                assert!(resolved.rung.is_traced() && capabilities.nrd, "{context}");
                            }
                        }
                        // DLSS-RR is denoiser and upscaler together or neither.
                        assert_eq!(
                            resolved.denoiser == ResolvedDenoiser::DlssRayReconstruction,
                            resolved.upscaler == ResolvedUpscaler::DlssRayReconstruction,
                            "{context}"
                        );
                        match resolved.upscaler {
                            ResolvedUpscaler::Taa => {
                                assert_eq!(
                                    resolved.quality,
                                    ResolvedQuality::NativeAa,
                                    "{context}"
                                );
                            }
                            ResolvedUpscaler::Tsr => assert!(capabilities.tsr, "{context}"),
                            ResolvedUpscaler::DlssSuperResolution => {
                                assert!(capabilities.dlss_super_resolution, "{context}");
                            }
                            ResolvedUpscaler::DlssRayReconstruction => {}
                            ResolvedUpscaler::Fsr => assert!(
                                capabilities.fsr.is_some() && capabilities.vendor == GpuVendor::Amd,
                                "{context}"
                            ),
                        }
                        assert_eq!(resolved.render_scale, resolved.quality.scale(), "{context}");
                        // A rung asked for by name and not given always comes with its reason.
                        let asked = match preset {
                            GraphicsPreset::Auto => None,
                            GraphicsPreset::Raster => Some(GraphicsRung::Raster),
                            GraphicsPreset::RtLite => Some(GraphicsRung::RtLite),
                            GraphicsPreset::RtFull => Some(GraphicsRung::RtFull),
                        };
                        if asked.is_some_and(|asked| asked != resolved.rung) {
                            assert!(!resolved.fallbacks.is_empty(), "{context}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fast_rt_list() {
    for name in [
        "NVIDIA GeForce RTX 4070",
        "NVIDIA GeForce RTX 4070 Ti SUPER",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 5070",
        "NVIDIA GeForce RTX 5080",
        "AMD Radeon RX 9070",
        "AMD Radeon RX 9070 XT (RADV GFX1201)",
        "AMD Radeon RX 7900 XT",
        "AMD Radeon RX 7900 XTX (RADV NAVI31)",
        "AMD Radeon Graphics (RADV GFX1201)",
        "NVIDIA GeForce RTX 3080 Ti",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 3090 Ti",
    ] {
        assert!(on_fast_rt_list(name), "{name}");
    }
    for name in [
        "NVIDIA GeForce RTX 2080 Ti",
        "NVIDIA GeForce RTX 3060",
        "NVIDIA GeForce RTX 3080",
        "NVIDIA GeForce RTX 3080 Ti Laptop GPU",
        "NVIDIA GeForce RTX 4080 Laptop GPU",
        "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        "NVIDIA GeForce RTX 4060 Ti",
        "AMD Radeon Graphics (RADV GFX1200)",
        "NVIDIA GeForce RTX 5060 Ti",
        "NVIDIA RTX A4000",
        "NVIDIA RTX 4000 Ada Generation",
        "NVIDIA GeForce GTX 1070",
        "AMD Radeon RX 6600 XT",
        "AMD Radeon RX 7700 XT",
        "AMD Radeon RX 7900 GRE",
        "AMD Radeon RX 9060 XT",
        "AMD Ryzen 5 9600X 6-Core Processor (RADV RAPHAEL_MENDOCINO)",
        "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
        "llvmpipe (LLVM 23.1.5, 256 bits)",
        "",
    ] {
        assert!(!on_fast_rt_list(name), "{name}");
    }
}

#[test]
fn slow_rt_list() {
    for name in [
        "Intel(R) Arc(tm) A310 Graphics (DG2)",
        "Intel(R) Arc(tm) A380 Graphics (DG2)",
        "Intel(R) Arc(tm) A580 Graphics (DG2)",
        "Intel(R) Arc(tm) A570M Graphics (DG2)",
        "NVIDIA GeForce RTX 2050",
        "NVIDIA GeForce RTX 3050",
        "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "NVIDIA GeForce RTX 4050 Laptop GPU",
        "NVIDIA GeForce RTX 5050 Laptop GPU",
        "AMD Radeon RX 6400",
        "AMD Radeon RX 6500 XT",
        "AMD Radeon RX 6600",
        "AMD Radeon RX 6600M",
        "AMD Radeon RX 7600",
        "AMD Radeon RX 7600 (RADV NAVI33)",
    ] {
        assert!(on_slow_rt_list(name), "{name}");
        assert!(!on_fast_rt_list(name), "{name}");
    }
    for name in [
        "Intel(R) Arc(tm) A750 Graphics (DG2)",
        "Intel(R) Arc(tm) A770 Graphics (DG2)",
        "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
        "NVIDIA GeForce RTX 2060",
        "NVIDIA GeForce RTX 3060",
        "NVIDIA GeForce RTX 4060 Laptop GPU",
        "NVIDIA GeForce RTX 5050",
        "NVIDIA RTX A2000",
        "AMD Radeon RX 6600 XT",
        "AMD Radeon RX 6650 XT",
        "AMD Radeon RX 6700 XT",
        "AMD Radeon RX 7600 XT",
        "AMD Radeon RX 580 Series (RADV POLARIS10)",
        "AMD Ryzen 5 9600X 6-Core Processor (RADV RAPHAEL_MENDOCINO)",
        "llvmpipe (LLVM 23.1.5, 256 bits)",
        "",
    ] {
        assert!(!on_slow_rt_list(name), "{name}");
    }
}

#[test]
fn fallbacks_read_as_sentences() {
    assert_eq!(
        Fallback::IntegratedGpu.to_string(),
        "Raster: the adapter is an integrated or virtual GPU"
    );
}

#[test]
fn settings_round_trip_through_serde() {
    let settings = GraphicsSettings {
        preset: GraphicsPreset::RtLite,
        upscaler: UpscalerChoice::Tsr,
        upscale_quality: UpscaleQuality::Balanced,
        denoiser: DenoiserChoice::Nrd,
        overrides: GraphicsOverrides {
            ssr: Some(false),
            max_bounces: Some(2),
            baked_gi: Some(false),
            planar_reflections: Some(false),
            ..Default::default()
        },
        ambient_while_baked: 2.5,
    };
    let text = ron::to_string(&settings).unwrap();
    assert_eq!(ron::from_str::<GraphicsSettings>(&text).unwrap(), settings);
    // Settings saved before planar reflections existed still read.
    let old = text.replace(",planar_reflections:Some(false)", "");
    assert_ne!(old, text);
    assert_eq!(
        ron::from_str::<GraphicsSettings>(&old).unwrap().overrides.planar_reflections,
        None
    );
}

/// The iGPU rule for mirrors: `Auto` on an integrated GPU gives the light table, whose planar reflections
/// render at a lower resolution; a capable card gets the full table's; asking for `Raster` by name on the
/// integrated GPU gives the full table's too.
#[test]
fn planar_reflections_follow_the_raster_level() {
    use crate::raster::raster_with;
    let mirrors = |settings: &GraphicsSettings, capabilities: &GraphicsCapabilities| {
        let resolved = resolve(settings, capabilities, P1080);
        assert_eq!(resolved.rung, GraphicsRung::Raster);
        raster_with(resolved.raster, &settings.overrides)
            .planar_reflections
            .expect("Raster reflects mirrors")
    };
    let integrated = mirrors(&auto(), &ryzen_integrated());
    let by_name = mirrors(&preset(GraphicsPreset::Raster), &ryzen_integrated());
    let capable = mirrors(&preset(GraphicsPreset::Raster), &rtx_5070());
    assert!(integrated.resolution_scale < capable.resolution_scale);
    assert_eq!(by_name, capable);
    assert_eq!(integrated.max_mirrors, capable.max_mirrors);
}
