//! What the player asked for: [`GraphicsSettings`].

use bevy_ecs::prelude::*;
use bevy_reflect::{std_traits::ReflectDefault, Reflect, ReflectDeserialize, ReflectSerialize};
use serde::{Deserialize, Serialize};

/// The graphics settings a game saves and a settings menu edits.
///
/// Every field has an `Auto`, and [`resolve`](crate::resolve) turns the settings into what this machine will
/// actually run.
#[derive(Resource, Reflect, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[reflect(Resource, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphicsSettings {
    /// Which rung of the ladder to render with.
    pub preset: GraphicsPreset,
    /// Which upscaler brings the internal resolution up to the output's.
    pub upscaler: UpscalerChoice,
    /// How far below the output's resolution the frame is rendered.
    pub upscale_quality: UpscaleQuality,
    /// Which denoiser cleans the ray traced presets' lighting. Ignored by [`GraphicsPreset::Raster`].
    pub denoiser: DenoiserChoice,
    /// Single effects forced on or off, whatever the preset says.
    pub overrides: GraphicsOverrides,
    /// The brightness of the ambient light every [`GraphicsCamera`] gets while a baked indirect light is loaded
    /// ([`baked`](crate::baked)), in cd/m². It lights only what lies outside every baked box (a door jamb, a
    /// character in a doorway); inside one the bake replaces it. Default
    /// [`DEFAULT_AMBIENT_WHILE_BAKED`](crate::baked::DEFAULT_AMBIENT_WHILE_BAKED); settings saved before it
    /// existed read as that default.
    #[serde(default = "crate::baked::default_ambient_while_baked")]
    pub ambient_while_baked: f32,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            preset: Default::default(),
            upscaler: Default::default(),
            upscale_quality: Default::default(),
            denoiser: Default::default(),
            overrides: Default::default(),
            ambient_while_baked: crate::baked::DEFAULT_AMBIENT_WHILE_BAKED,
        }
    }
}

/// A rung of the graphics ladder, or `Auto` to pick one from the machine's capabilities.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Default, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum GraphicsPreset {
    /// The highest rung the static rules of [`resolve`](crate::resolve) allow on this machine. On an integrated
    /// GPU or a card under 5.5 GiB of video memory, Raster's light table at a lower resolution.
    #[default]
    Auto,
    /// Rasterized lighting: shadow maps, screen-space reflections and ambient occlusion, no ray tracing.
    Raster,
    /// Ray traced lighting with the cheap parameter set: two bounces that end in the world cache (the full set
    /// has three), at a lower internal resolution under [`UpscaleQuality::Auto`].
    RtLite,
    /// Ray traced lighting with the full parameter set.
    RtFull,
}

/// The upscaler the player asked for.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Default, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum UpscalerChoice {
    /// DLSS where it is supported, otherwise TSR. Never FSR.
    #[default]
    Auto,
    /// No upscaling: the frame is rendered at the output's resolution and anti-aliased there.
    Off,
    /// The vendor-neutral temporal upscaler.
    Tsr,
    /// NVIDIA DLSS.
    Dlss,
    /// AMD FSR. Only on a Radeon under Windows, with FSR's DLL present.
    Fsr,
}

/// How far below the output's resolution the frame is rendered.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Default, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum UpscaleQuality {
    /// Chosen from the preset and the output's resolution.
    #[default]
    Auto,
    /// The output's resolution: the upscaler only anti-aliases.
    NativeAa,
    /// 67 % of the output's resolution on each axis.
    Quality,
    /// 58 % of the output's resolution on each axis.
    Balanced,
    /// 50 % of the output's resolution on each axis.
    Performance,
}

/// The denoiser the player asked for on the ray traced presets.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Default, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum DenoiserChoice {
    /// DLSS Ray Reconstruction where it is supported, otherwise NRD.
    #[default]
    Auto,
    /// NVIDIA DLSS Ray Reconstruction, which also upscales.
    DlssRayReconstruction,
    /// NVIDIA Real-time Denoisers (`ReBLUR`), which runs on any vendor's card.
    Nrd,
}

/// Single effects forced on or off. `None` leaves an effect as the preset sets it.
///
/// Only what a preset really switches is here; everything else stays the game's own components.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Default, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub struct GraphicsOverrides {
    /// Shadow maps on lights. The ray traced presets trace their shadows and ignore this.
    pub shadows: Option<bool>,
    /// Screen-space ambient occlusion. [`GraphicsPreset::Raster`] only.
    pub ssao: Option<bool>,
    /// Screen-space reflections. [`GraphicsPreset::Raster`] only.
    pub ssr: Option<bool>,
    /// Contact shadows. [`GraphicsPreset::Raster`] only.
    pub contact_shadows: Option<bool>,
    /// Bloom.
    pub bloom: Option<bool>,
    /// Reuse of light samples between pixels and frames (`ReSTIR`). Ray traced presets only.
    pub restir: Option<bool>,
    /// The longest light path, in bounces. Ray traced presets only.
    pub max_bounces: Option<u32>,
    /// Baked indirect light ([`baked`](crate::baked)): `Some(false)` hides a loaded bake and gives the cameras
    /// back the game's ambient light, for A/B comparisons. `Some(true)` is the same as `None`. Settings saved
    /// before it existed read as `None`.
    #[serde(default)]
    pub baked_gi: Option<bool>,
}

/// Marks a camera the graphics settings are applied to.
///
/// Cameras without it are left alone.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default, Clone)]
pub struct GraphicsCamera;
