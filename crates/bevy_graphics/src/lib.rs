#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

//! Tiered graphics: three presets, and the policy that picks one for the machine the app runs on.
//!
//! The ladder has three rungs. [`GraphicsRung::Raster`] is rasterized lighting. [`GraphicsRung::RtLite`] and
//! [`GraphicsRung::RtFull`] are ray traced lighting (`bevy_solari`'s realtime renderer) with a cheap and a
//! full parameter set, a denoiser, and usually an upscaler.
//!
//! - [`GraphicsSettings`] is what the player asked for. Every field has an `Auto`.
//! - [`GraphicsCapabilities`] is what the machine can do, read once from the renderer.
//! - [`resolve`] turns the two, with the output's size, into a [`ResolvedGraphics`]: what will run, and a
//!   reason for everything that differs from what was asked. It is a pure function.
//! - [`GraphicsCamera`] marks the cameras the result is applied to.
//!
//! Built without the `solari` feature, everything resolves to [`GraphicsRung::Raster`].

pub mod apply;
pub mod baked;
pub mod denoisers;
pub mod dlss;
pub mod meshes;
pub mod raster;
pub mod traced;
pub mod upscalers;

mod capabilities;
mod resolve;
mod settings;

pub use capabilities::{FsrLevel, GpuDeviceType, GpuVendor, GraphicsCapabilities};
pub use resolve::{
    resolve, Fallback, GraphicsRung, ResolvedDenoiser, ResolvedGraphics, ResolvedQuality,
    ResolvedUpscaler,
};
pub use settings::{
    DenoiserChoice, GraphicsCamera, GraphicsOverrides, GraphicsPreset, GraphicsSettings,
    UpscaleQuality, UpscalerChoice,
};

use bevy_app::{App, Plugin, PreStartup};
use bevy_ecs::prelude::*;

/// Adds [`GraphicsSettings`] and [`GraphicsCapabilities`] to the app and applies the settings to every
/// [`GraphicsCamera`].
///
/// Not part of `DefaultPlugins`: an app that sets up its own cameras and effects does not need it.
#[derive(Default)]
pub struct GraphicsPlugin;

impl Plugin for GraphicsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GraphicsSettings>()
            .init_resource::<GraphicsCapabilities>()
            .add_systems(
                PreStartup,
                capabilities::detect_capabilities.in_set(GraphicsSystems::DetectCapabilities),
            );

        // Each part of the ladder registers itself from its own file.
        apply::build(app);
        baked::build(app);
        upscalers::tsr::build(app);
        upscalers::fsr::build(app);
        denoisers::nrd::build(app);
    }
}

/// The system sets of [`GraphicsPlugin`].
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum GraphicsSystems {
    /// Fills [`GraphicsCapabilities`] from the renderer, in [`PreStartup`]. It runs there and not when the
    /// plugin is built because the renderer, and what the DLSS plugin learns from it, exist only once every
    /// plugin has finished.
    ///
    /// A system that sets a capability the renderer cannot report ([`GraphicsCapabilities::tsr`],
    /// [`GraphicsCapabilities::fsr`], [`GraphicsCapabilities::nrd`]) runs in [`PreStartup`] after this set.
    DetectCapabilities,
}
