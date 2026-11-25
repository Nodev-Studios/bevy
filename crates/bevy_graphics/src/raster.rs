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
/// against the rung's gate of 2.8 ms. E2's cuts took it to about 2.25 ms; O3's three further cuts (contact
/// shadows off, SSAO Low, SSR 6 / 3 steps) to 2.02 ms (`many-lights`: 1.51 to 1.35), measured as the 10th
/// percentile of each pass over 300 frames, interleaved with the earlier table. Against Bevy's defaults rendered at
/// 3840x2160 and reduced to 1080p, O3's cuts move the error from 1.11 to 1.13 points on `interior` and from 0.479 to
/// 0.483 on `many-lights`; flicker and the motion metric stay within run-to-run noise.
pub fn raster() -> RasterEffects {
    RasterEffects {
        // Every anti-aliasing the ladder uses is temporal, which is what this filter is made for.
        // `Hardware2x2` saves 0.013 ms on `interior` and nothing measurable on `many-lights` (O3).
        shadow_filter: ShadowFilteringMethod::Temporal,
        // Off: 4 steps cost 0.06-0.07 ms of the deferred lighting pass on both scenes, and switching them off
        // changes neither scene's error (1.106 against 1.105, 0.478 against 0.479) nor its flicker. The testbed
        // has no feet or thin props, the contacts shadow maps miss, so a game with them may want them back
        // (`GraphicsOverrides::contact_shadows`, which turns them on at 4 steps).
        contact_shadows: None,
        // Low (4 samples a pixel, against Medium's 8; High is 18): 0.155 ms against 0.217 on `interior`, 0.151
        // against 0.213 on `many-lights`. TAA resolves the extra noise: error and flicker do not move on either
        // scene.
        ambient_occlusion: Some(ScreenSpaceAmbientOcclusion {
            quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Low,
            ..Default::default()
        }),
        // 6 linear and 3 bisection steps, against E2's 8 / 3 and Bevy's 10 / 5: 0.404 ms against 0.460 on
        // `interior`, whose glossy floor fills most of the frame, 0.219 against 0.252 on `many-lights`. The error
        // rises 0.025 points on `interior` and 0.006 on `many-lights`; it is the cut that shows most, on the
        // upper rim of the metal sphere, inside the speckle two runs of the earlier table already differ by.
        reflections: Some(ScreenSpaceReflections {
            linear_steps: 6,
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
        // Forced on where the table has them off: E2's 4 steps (0.07 ms on `interior` at 1080p), not Bevy's
        // 16 (0.29 ms).
        contact_shadows: match overrides.contact_shadows {
            Some(true) => Some(table.contact_shadows.unwrap_or(ContactShadows {
                linear_steps: 4,
                ..Default::default()
            })),
            forced => switch(table.contact_shadows, forced),
        },
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
        // Forced on, an effect is on whether or not the table has it.
        let all_on = raster_with(&GraphicsOverrides {
            ssao: Some(true),
            ssr: Some(true),
            contact_shadows: Some(true),
            bloom: Some(true),
            ..Default::default()
        });
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
