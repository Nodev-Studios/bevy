//! NRD, the denoiser of the ray traced rungs where DLSS Ray Reconstruction cannot run, on the ladder's
//! cameras.
//!
//! Stub: filled by work package B3, which also sets
//! [`GraphicsCapabilities::nrd`](crate::GraphicsCapabilities::nrd) from its `build` function.

use bevy_app::App;

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built.
pub(crate) fn build(_app: &mut App) {}
