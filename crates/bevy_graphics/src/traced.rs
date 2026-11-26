//! What the `RtLite` and `RtFull` rungs put on a camera: the two parameter sets of `SolariLighting`.
//!
//! Both rungs are one renderer, `bevy_solari`'s realtime path. `lite()` ends every light path in the world
//! cache after two bounces instead of three and updates half as many cache cells; it samples direct light as
//! `full()` does, and neither uses `ReSTIR`. What makes `RtLite` cheaper is mostly its lower internal resolution
//! under `Auto` (`resolve`'s `auto_quality`). `full()` is the renderer as its authors ship it.

#[cfg(feature = "solari")]
pub use solari::*;

#[cfg(feature = "solari")]
mod solari {
    use crate::{GraphicsOverrides, GraphicsRung};
    use bevy_solari::realtime::SolariLighting;

    /// The parameter set of [`GraphicsRung::RtLite`].
    ///
    /// Measured on the testbed's `interior` and `many-lights` at 1920x1080 output, Balanced (58 %), on the
    /// RTX 5070 (2026-11-24, medians of three runs, error against Solari's path tracer with bloom off on both).
    /// The first set (2 light candidates, 1 bounce) erred 2.85 / 4.08 % with DLSS-RR and 2.98 / 4.00 % with
    /// NRD + TSR; this one errs 2.02 / 2.79 % and 2.55 / 3.56 %, for 0.43 ms of lighting pass instead of 0.22 on
    /// `interior`. The rung's whole frame stays far under its gate; what makes it lighter than `full` is now the
    /// internal resolution, one bounce fewer and half the world cache updates (`full` has `ReSTIR` off too).
    pub fn lite() -> SolariLighting {
        SolariLighting {
            // Off: with DLSS-RR on `interior` it nearly doubles the still camera's flicker (0.14 → 0.24 %)
            // without lowering the error (2.02 → 2.04 %), for 0.7-0.85 ms more lighting. On `many-lights` it
            // brings 4 candidates to 2.73 %, which 8 candidates without it nearly match (2.79 %) for 0.05 ms.
            // Under NRD it changes nothing (B3).
            restir: false,
            // Read only by ReSTIR.
            confidence_weight_cap: 8.0,
            // 8, as `full`: against 4 it lowers `many-lights`' error from 3.33 to 2.79 % (DLSS-RR) and 3.65 to
            // 3.56 % (NRD) for 0.05 ms, and leaves `interior` where it is. Against 2, `many-lights` 4.08 → 2.79.
            primary_di_samples: 8,
            secondary_di_samples: 4,
            // 2: a second bounce before the world cache takes `interior`'s error from 2.85 to 2.02 % (DLSS-RR) and
            // 2.98 to 2.55 % (NRD) for 0.14 ms; a third adds 0.02 ms and moves it 0.07 points (DLSS-RR) or nothing.
            max_bounces: 2,
            world_cache_max_temporal_samples: 32.0,
            world_cache_direct_light_sample_count: 32,
            world_cache_max_gi_ray_distance: 50.0,
            // Half of `full`'s. Updating as many cells as `full` does not shorten the lag after a light switches
            // off (1.22 against 1.06 % five frames after, DLSS-RR; 5.83 against 5.74 % with NRD) nor change
            // the world cache pass's time (C1). Measured when this was a stochastic soft target; upstream #26626
            // made it a hard per-frame limit taken in turns. To tune: remeasure the lag on the new world cache.
            world_cache_max_cell_updates_per_frame: 20_000,
            // 0.15 m, as `full`: 0.3 m raises `interior`'s error from 2.85 to 4.51 % for 0.01 ms; 0.1 m lowers it
            // 0.03 points for 0.02 ms.
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
            // Off with either denoiser. With NRD it changes neither error nor flicker (B3, and 2.32 against 2.30 %
            // on `interior` at 2560x1440 Quality here). With DLSS-RR at 2560x1440 Quality (2026-11-24, RTX 5070)
            // it lowers `many-lights`' error from 2.55 to 2.23 % but doubles `interior`'s flicker (0.13 → 0.24 %)
            // without lowering its error (1.91 → 1.94 %), and `interior` is the scene shaped like the game's rooms.
            restir: false,
            confidence_weight_cap: 8.0,
            primary_di_samples: 8,
            secondary_di_samples: 4,
            max_bounces: 3,
            world_cache_max_temporal_samples: 32.0,
            world_cache_direct_light_sample_count: 32,
            world_cache_max_gi_ray_distance: 50.0,
            world_cache_max_cell_updates_per_frame: 40_000,
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
                ours.world_cache_max_cell_updates_per_frame,
                theirs.world_cache_max_cell_updates_per_frame
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
        fn lite_traces_less_than_full() {
            let (lite, full) = (lite(), full());
            assert!(!lite.restir);
            assert!(lite.max_bounces < full.max_bounces);
            assert!(lite.primary_di_samples <= full.primary_di_samples);
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
