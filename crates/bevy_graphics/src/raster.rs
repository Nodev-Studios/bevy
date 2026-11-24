//! What the `Raster` rung puts on a camera and on the lights: a table over effects Bevy already has.

use crate::GraphicsOverrides;
use bevy_light::ShadowFilteringMethod;
use bevy_pbr::{
    ContactShadows, ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel,
    ScreenSpaceReflections,
};
use bevy_post_process::bloom::Bloom;

/// The effects of [`GraphicsRung::Raster`](crate::GraphicsRung::Raster). `None` leaves an effect off.
#[derive(Clone)]
pub struct RasterEffects {
    /// How shadow map edges are filtered.
    pub shadow_filter: ShadowFilteringMethod,
    /// Screen-space shadows where shadow maps lack detail. They are also switched on for every light that
    /// casts shadow maps, because a light only gets them when it asks.
    pub contact_shadows: Option<ContactShadows>,
    /// Screen-space ambient occlusion.
    pub ambient_occlusion: Option<ScreenSpaceAmbientOcclusion>,
    /// Screen-space reflections. Where they find nothing, the reflection probes the game placed and then the
    /// environment map answer, as in any Bevy app.
    pub reflections: Option<ScreenSpaceReflections>,
    /// Bloom. The traced rungs use the same value: a rung changes the lighting, not the lens.
    pub bloom: Option<Bloom>,
}

/// The table of the Raster rung.
///
/// Measured on the testbed's `interior` and `many-lights` scenes at 1920x1080 with TAA on the RTX 5070
/// (2026-11-24, GPU time of the frame from the render diagnostics, medians of three runs). With Bevy's defaults
/// for every effect the frame took 2.75 ms of GPU on `interior` (3.22 ms median frame on a quiet machine)
/// against the rung's gate of 2.8 ms; with these cuts it takes about 2.25 ms. What each cut saves and costs is
/// beside it. On frames rendered at 3840x2160 the cuts together (contact shadows off) differ from the defaults
/// by 0.71-0.73 points on `interior`, where two runs of the defaults differ by 0.67, and by 0.13 on
/// `many-lights` (two runs: 0.06).
pub fn raster() -> RasterEffects {
    RasterEffects {
        // Every anti-aliasing the ladder uses is temporal, which is what this filter is made for.
        // `Hardware2x2` would save 0.03 ms on `interior`.
        shadow_filter: ShadowFilteringMethod::Temporal,
        // 4 linear steps over the same 0.3 m: 0.07 ms of the deferred lighting pass on `interior`, against
        // 0.29 ms at Bevy's 16 (0.15 at 8). Switching them off left no visible shape in C1's difference image of
        // that scene, but they stay on for the contacts shadow maps miss in a game (feet, thin props).
        contact_shadows: Some(ContactShadows {
            linear_steps: 4,
            ..Default::default()
        }),
        // Medium (8 samples a pixel, against High's 18): 0.22 ms against 0.37 on `interior`. On TAA frames
        // rendered at 3840x2160 (supersampled for 1080p) it differs from High by 0.53-0.62 points on `interior`,
        // where two runs of High differ by 0.67, and by 0.08-0.10 on `many-lights` (two runs: 0.06).
        ambient_occlusion: Some(ScreenSpaceAmbientOcclusion {
            quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Medium,
            ..Default::default()
        }),
        // 8 linear and 3 bisection steps, against Bevy's 10 and 5: 0.46 ms against 0.59 on `interior`, whose
        // glossy floor fills most of the frame. The one cut that shows above run-to-run noise: 0.12-0.13 points
        // on `many-lights` against 0.06 between two runs of the defaults (`interior`: 0.52-0.63 against 0.67).
        reflections: Some(ScreenSpaceReflections {
            linear_steps: 8,
            bisection_steps: 3,
            ..Default::default()
        }),
        // 0.10 ms; it is the lens, not the lighting, so every rung keeps it.
        bloom: Some(Bloom::NATURAL),
    }
}

/// The table with the player's overrides applied: an effect forced off is `None`, one forced on keeps the
/// table's value or, where the table has it off, the effect's own default.
pub fn raster_with(overrides: &GraphicsOverrides) -> RasterEffects {
    let table = raster();
    RasterEffects {
        shadow_filter: table.shadow_filter,
        contact_shadows: switch(table.contact_shadows, overrides.contact_shadows),
        ambient_occlusion: switch(table.ambient_occlusion, overrides.ssao),
        reflections: switch(table.reflections, overrides.ssr),
        bloom: switch(table.bloom, overrides.bloom),
    }
}

/// Bloom on a traced rung, with the player's override applied.
pub fn traced_bloom(overrides: &GraphicsOverrides) -> Option<Bloom> {
    switch(raster().bloom, overrides.bloom)
}

fn switch<T: Default>(table: Option<T>, forced: Option<bool>) -> Option<T> {
    match forced {
        None => table,
        Some(false) => None,
        Some(true) => Some(table.unwrap_or_default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_switch_single_effects() {
        let all_on = raster_with(&GraphicsOverrides::default());
        assert!(all_on.contact_shadows.is_some());
        assert!(all_on.ambient_occlusion.is_some());
        assert!(all_on.reflections.is_some());
        assert!(all_on.bloom.is_some());

        let no_ssr = raster_with(&GraphicsOverrides {
            ssr: Some(false),
            ..Default::default()
        });
        assert!(no_ssr.reflections.is_none());
        assert!(no_ssr.ambient_occlusion.is_some());

        let all_off = raster_with(&GraphicsOverrides {
            ssao: Some(false),
            ssr: Some(false),
            contact_shadows: Some(false),
            bloom: Some(false),
            ..Default::default()
        });
        assert!(all_off.contact_shadows.is_none());
        assert!(all_off.ambient_occlusion.is_none());
        assert!(all_off.reflections.is_none());
        assert!(all_off.bloom.is_none());
        assert!(traced_bloom(&GraphicsOverrides {
            bloom: Some(false),
            ..Default::default()
        })
        .is_none());
    }
}
