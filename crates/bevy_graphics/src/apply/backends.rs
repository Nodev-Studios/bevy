//! How an upscaler or a denoiser that lives outside this crate's core gets onto the ladder's cameras.
//!
//! The apply layer knows how to put DLSS and TAA on a camera. TSR, FSR and NRD each have a camera component of
//! their own, in a crate feature of their own; their module registers a [`CameraBackend`] in
//! [`GraphicsBackends`] from its `build` function, and the apply layer calls it:
//!
//! ```ignore
//! pub(crate) fn build(app: &mut App) {
//!     app.world_mut().resource_mut::<GraphicsBackends>().tsr = Some(CameraBackend {
//!         insert: |camera, resolved| {
//!             camera.insert(Tsr::at_scale(resolved.render_scale));
//!         },
//!         remove: |camera| {
//!             camera.remove::<Tsr>();
//!         },
//!     });
//!     // ... and the system that sets `GraphicsCapabilities::tsr`.
//! }
//! ```

use crate::ResolvedGraphics;
use bevy_ecs::prelude::*;

/// The upscalers and the denoiser the apply layer reaches through a hook, each `None` until its module
/// registers itself.
///
/// [`GraphicsPlugin`](crate::GraphicsPlugin) inserts it before any of those modules is built.
#[derive(Resource, Default)]
pub struct GraphicsBackends {
    /// The vendor-neutral temporal upscaler, used when [`ResolvedGraphics::upscaler`] is
    /// [`ResolvedUpscaler::Tsr`](crate::ResolvedUpscaler::Tsr).
    pub tsr: Option<CameraBackend>,
    /// FSR, used when [`ResolvedGraphics::upscaler`] is [`ResolvedUpscaler::Fsr`](crate::ResolvedUpscaler::Fsr).
    pub fsr: Option<CameraBackend>,
    /// NRD, used when [`ResolvedGraphics::denoiser`] is
    /// [`ResolvedDenoiser::Nrd`](crate::ResolvedDenoiser::Nrd).
    pub nrd: Option<CameraBackend>,
}

/// What the apply layer calls to put one upscaler or denoiser on a camera and to take it off again.
#[derive(Clone, Copy)]
pub struct CameraBackend {
    /// Puts the backend's components on the camera for these settings, or updates them. Called when the
    /// backend becomes the camera's choice and again whenever [`ResolvedGraphics`] or the player's overrides
    /// change while it stays the choice, so it must be safe to repeat.
    ///
    /// The camera already has `Hdr`, `Msaa::Off`, the depth, motion vector and deferred prepasses, and
    /// `TemporalJitter` with `MipBias`. An upscaler reads the internal resolution from
    /// [`ResolvedGraphics::render_scale`].
    pub insert: fn(&mut EntityCommands, &ResolvedGraphics),
    /// Removes from the camera everything [`Self::insert`] put there. Called once when the backend stops being
    /// the camera's choice.
    pub remove: fn(&mut EntityCommands),
}
