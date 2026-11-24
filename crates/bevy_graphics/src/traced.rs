//! What the `RtLite` and `RtFull` rungs put on a camera: the two parameter sets of `SolariLighting`.
//!
//! Both rungs are one renderer, `bevy_solari`'s realtime path. `lite()` cuts every light path at its first
//! bounce, which then ends in the world cache, and spends less on direct light and on keeping the cache
//! fresh; `full()` is the renderer as its authors ship it.

#[cfg(feature = "solari")]
pub use solari::*;

#[cfg(feature = "solari")]
mod solari {
    use crate::{GraphicsOverrides, GraphicsRung};
    use bevy_solari::realtime::SolariLighting;

    /// The parameter set of [`GraphicsRung::RtLite`].
    pub fn lite() -> SolariLighting {
        SolariLighting {
            restir: false,
            // Read only by ReSTIR.
            confidence_weight_cap: 8.0,
            // to tune: half of `full`'s light candidates at the primary hit.
            primary_di_samples: 4,
            // Never read: with one bounce there is no second hit to light directly.
            secondary_di_samples: 4,
            // The first bounce is forced to end in the world cache, so one ray per pixel carries all the
            // indirect light and reflections.
            max_bounces: 1,
            world_cache_max_temporal_samples: 32.0,
            world_cache_direct_light_sample_count: 32,
            world_cache_max_gi_ray_distance: 50.0,
            // to tune: half of `full`'s, so the cache takes twice as long to follow a change of light.
            world_cache_cell_updates_soft_target: 20_000,
            world_cache_position_base_cell_size: 0.15,
            world_cache_position_lod_scale: 15.0,
            world_cache_force_opaque: true,
            // No temporal history on the first frame.
            reset: true,
        }
    }

    /// The parameter set of [`GraphicsRung::RtFull`]: `bevy_solari`'s own defaults, written out so a change
    /// upstream shows up here as a decision and not as a silent change of the rung.
    pub fn full() -> SolariLighting {
        SolariLighting {
            // to tune: whether reuse pays for itself depends on the denoiser.
            restir: false,
            confidence_weight_cap: 8.0,
            primary_di_samples: 8,
            secondary_di_samples: 4,
            max_bounces: 3,
            world_cache_max_temporal_samples: 32.0,
            world_cache_direct_light_sample_count: 32,
            world_cache_max_gi_ray_distance: 50.0,
            world_cache_cell_updates_soft_target: 40_000,
            world_cache_position_base_cell_size: 0.15,
            world_cache_position_lod_scale: 15.0,
            world_cache_force_opaque: true,
            // No temporal history on the first frame.
            reset: true,
        }
    }

    /// The lighting a traced rung puts on a camera, with the player's overrides applied. `None` for
    /// [`GraphicsRung::Raster`].
    pub fn solari_lighting(
        rung: GraphicsRung,
        overrides: &GraphicsOverrides,
    ) -> Option<SolariLighting> {
        let mut lighting = match rung {
            GraphicsRung::Raster => return None,
            GraphicsRung::RtLite => lite(),
            GraphicsRung::RtFull => full(),
        };
        if let Some(restir) = overrides.restir {
            lighting.restir = restir;
        }
        if let Some(max_bounces) = overrides.max_bounces {
            // The renderer needs one bounce to produce any indirect light at all.
            lighting.max_bounces = max_bounces.max(1);
        }
        Some(lighting)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn full_is_solaris_default() {
            let (ours, theirs) = (full(), SolariLighting::default());
            assert_eq!(ours.restir, theirs.restir);
            assert_eq!(ours.confidence_weight_cap, theirs.confidence_weight_cap);
            assert_eq!(ours.primary_di_samples, theirs.primary_di_samples);
            assert_eq!(ours.secondary_di_samples, theirs.secondary_di_samples);
            assert_eq!(ours.max_bounces, theirs.max_bounces);
            assert_eq!(
                ours.world_cache_max_temporal_samples,
                theirs.world_cache_max_temporal_samples
            );
            assert_eq!(
                ours.world_cache_direct_light_sample_count,
                theirs.world_cache_direct_light_sample_count
            );
            assert_eq!(
                ours.world_cache_max_gi_ray_distance,
                theirs.world_cache_max_gi_ray_distance
            );
            assert_eq!(
                ours.world_cache_cell_updates_soft_target,
                theirs.world_cache_cell_updates_soft_target
            );
            assert_eq!(
                ours.world_cache_position_base_cell_size,
                theirs.world_cache_position_base_cell_size
            );
            assert_eq!(
                ours.world_cache_position_lod_scale,
                theirs.world_cache_position_lod_scale
            );
            assert_eq!(
                ours.world_cache_force_opaque,
                theirs.world_cache_force_opaque
            );
            assert_eq!(ours.reset, theirs.reset);
        }

        #[test]
        fn lite_is_one_bounce_without_restir() {
            let lite = lite();
            assert!(!lite.restir);
            assert_eq!(lite.max_bounces, 1);
            assert!(lite.primary_di_samples < full().primary_di_samples);
            assert_eq!(
                lite.world_cache_cell_updates_soft_target * 2,
                full().world_cache_cell_updates_soft_target
            );
        }

        #[test]
        fn overrides_reach_the_traced_rungs_only() {
            let overrides = GraphicsOverrides {
                restir: Some(true),
                max_bounces: Some(0),
                ..Default::default()
            };
            assert!(solari_lighting(GraphicsRung::Raster, &overrides).is_none());
            let lite = solari_lighting(GraphicsRung::RtLite, &overrides).unwrap();
            assert!(lite.restir);
            assert_eq!(lite.max_bounces, 1);
            let full =
                solari_lighting(GraphicsRung::RtFull, &GraphicsOverrides::default()).unwrap();
            assert_eq!(full.max_bounces, 3);
        }
    }
}
