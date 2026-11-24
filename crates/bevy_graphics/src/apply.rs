//! The apply layer: recomputes [`ResolvedGraphics`](crate::ResolvedGraphics) when the settings or the output
//! change, and puts the result onto every [`GraphicsCamera`](crate::GraphicsCamera) and onto the lights,
//! switching presets at run time.
//!
//! Stub: filled by work package C1, together with [`meshes`](crate::meshes), [`raster`](crate::raster),
//! [`traced`](crate::traced) and [`dlss`](crate::dlss), which it wires up from its `build` function.

use bevy_app::App;

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built.
pub(crate) fn build(_app: &mut App) {}
