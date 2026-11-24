//! What the `Raster` rung puts on a camera and on the lights: a table over effects Bevy already has.

use crate::GraphicsOverrides;
use bevy_light::ShadowFilteringMethod;
use bevy_pbr::{ContactShadows, ScreenSpaceAmbientOcclusion, ScreenSpaceReflections};
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
// to tune: Bevy's defaults for each effect, until the rung is measured against its budget.
pub fn raster() -> RasterEffects {
    RasterEffects {
        // Every anti-aliasing the ladder uses is temporal, which is what this filter is made for.
        shadow_filter: ShadowFilteringMethod::Temporal,
        contact_shadows: Some(ContactShadows::default()),
        ambient_occlusion: Some(ScreenSpaceAmbientOcclusion::default()),
        reflections: Some(ScreenSpaceReflections::default()),
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
