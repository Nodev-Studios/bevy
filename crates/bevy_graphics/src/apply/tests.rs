//! The apply layer on a bare `World`: no renderer, no window. What a camera and a light carry after each
//! switch is all these tests look at.

use super::*;
use crate::{Fallback, GraphicsOverrides};
use bevy_ecs::system::SystemId;

#[derive(Component)]
struct TestTsr;
#[derive(Component)]
struct TestFsr;
#[derive(Component)]
struct TestNrd;

fn test_backend<C: Component + Default>() -> CameraBackend {
    CameraBackend {
        insert: |camera, _| {
            camera.insert(C::default());
        },
        remove: |camera| {
            camera.remove::<C>();
        },
    }
}

impl Default for TestTsr {
    fn default() -> Self {
        Self
    }
}
impl Default for TestFsr {
    fn default() -> Self {
        Self
    }
}
impl Default for TestNrd {
    fn default() -> Self {
        Self
    }
}

struct Ladder {
    world: World,
    cameras: SystemId,
    lights: [SystemId; 3],
}

impl Ladder {
    fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(GraphicsSettings::default());
        world.insert_resource(GraphicsBackends {
            tsr: Some(test_backend::<TestTsr>()),
            fsr: Some(test_backend::<TestFsr>()),
            nrd: Some(test_backend::<TestNrd>()),
        });
        world.insert_resource(resolved(
            GraphicsRung::Raster,
            ResolvedUpscaler::Taa,
            ResolvedDenoiser::None,
            ResolvedQuality::NativeAa,
        ));
        let cameras = world.register_system(apply_to_cameras);
        let lights = [
            world.register_system(apply_to_lights::<PointLight>),
            world.register_system(apply_to_lights::<SpotLight>),
            world.register_system(apply_to_lights::<DirectionalLight>),
        ];
        Self {
            world,
            cameras,
            lights,
        }
    }

    fn camera(&mut self) -> Entity {
        self.world.spawn((Camera3d::default(), GraphicsCamera)).id()
    }

    fn apply(&mut self, state: &State) {
        self.world.insert_resource(state.resolved.clone());
        self.world.resource_mut::<GraphicsSettings>().overrides = state.overrides;
        self.world.run_system(self.cameras).unwrap();
        for lights in self.lights {
            self.world.run_system(lights).unwrap();
        }
    }

    fn has<C: Component>(&self, entity: Entity) -> bool {
        self.world.entity(entity).contains::<C>()
    }

    /// The names of the entity's components: ids differ from one `World` to the next.
    fn components(&self, entity: Entity) -> Vec<String> {
        let mut components: Vec<_> = self
            .world
            .entity(entity)
            .archetype()
            .components()
            .iter()
            .map(|id| {
                self.world
                    .components()
                    .get_info(*id)
                    .unwrap()
                    .name()
                    .to_string()
            })
            .collect();
        components.sort();
        components
    }
}

fn resolved(
    rung: GraphicsRung,
    upscaler: ResolvedUpscaler,
    denoiser: ResolvedDenoiser,
    quality: ResolvedQuality,
) -> ResolvedGraphics {
    ResolvedGraphics {
        rung,
        upscaler,
        denoiser,
        quality,
        render_scale: quality.scale(),
        fallbacks: Vec::<Fallback>::new(),
        unfitted_meshes: 0,
        skipped_deforming_meshes: 0,
    }
}

#[derive(Clone)]
struct State {
    resolved: ResolvedGraphics,
    overrides: GraphicsOverrides,
}

impl State {
    fn raster() -> Self {
        Self {
            resolved: resolved(
                GraphicsRung::Raster,
                ResolvedUpscaler::Taa,
                ResolvedDenoiser::None,
                ResolvedQuality::NativeAa,
            ),
            overrides: GraphicsOverrides::default(),
        }
    }

    fn traced(rung: GraphicsRung) -> Self {
        Self {
            // NRD with TAA needs no cargo feature of this crate, so it is the traced state every build of
            // the tests can apply.
            resolved: resolved(
                rung,
                ResolvedUpscaler::Taa,
                ResolvedDenoiser::Nrd,
                ResolvedQuality::NativeAa,
            ),
            overrides: GraphicsOverrides::default(),
        }
    }

    fn with(mut self, overrides: GraphicsOverrides) -> Self {
        self.overrides = overrides;
        self
    }
}

/// Every state `resolve` can return that this build of the crate can put on a camera, with a few overrides.
fn every_state() -> Vec<State> {
    let mut pairs = vec![
        (
            GraphicsRung::Raster,
            ResolvedDenoiser::None,
            ResolvedUpscaler::Taa,
        ),
        (
            GraphicsRung::Raster,
            ResolvedDenoiser::None,
            ResolvedUpscaler::Tsr,
        ),
        (
            GraphicsRung::Raster,
            ResolvedDenoiser::None,
            ResolvedUpscaler::Fsr,
        ),
    ];
    if cfg!(feature = "dlss") {
        pairs.push((
            GraphicsRung::Raster,
            ResolvedDenoiser::None,
            ResolvedUpscaler::DlssSuperResolution,
        ));
    }
    for rung in [GraphicsRung::RtLite, GraphicsRung::RtFull] {
        for upscaler in [
            ResolvedUpscaler::Taa,
            ResolvedUpscaler::Tsr,
            ResolvedUpscaler::Fsr,
        ] {
            pairs.push((rung, ResolvedDenoiser::Nrd, upscaler));
        }
        if cfg!(feature = "dlss") {
            pairs.push((
                rung,
                ResolvedDenoiser::Nrd,
                ResolvedUpscaler::DlssSuperResolution,
            ));
            pairs.push((
                rung,
                ResolvedDenoiser::DlssRayReconstruction,
                ResolvedUpscaler::DlssRayReconstruction,
            ));
        }
    }

    let overrides = [
        GraphicsOverrides::default(),
        GraphicsOverrides {
            shadows: Some(false),
            ssao: Some(false),
            ssr: Some(false),
            contact_shadows: Some(false),
            bloom: Some(false),
            restir: None,
            max_bounces: None,
        },
        GraphicsOverrides {
            restir: Some(true),
            max_bounces: Some(2),
            ..Default::default()
        },
    ];
    let mut states = Vec::new();
    for (rung, denoiser, upscaler) in pairs {
        for quality in [ResolvedQuality::NativeAa, ResolvedQuality::Performance] {
            // `resolve` never upscales with plain TAA.
            if upscaler == ResolvedUpscaler::Taa && quality != ResolvedQuality::NativeAa {
                continue;
            }
            for overrides in overrides {
                states.push(State {
                    resolved: resolved(rung, upscaler, denoiser, quality),
                    overrides,
                });
            }
        }
    }
    states
}

/// What must hold on a ladder camera in every state. Each line is a trap someone fell into.
fn assert_invariants(ladder: &Ladder, camera: Entity, state: &State) {
    let context = format!("{:?} with {:?}", state.resolved, state.overrides);
    let traced = state.resolved.rung.is_traced();
    let usages = ladder.world.get::<CameraMainTextureUsages>(camera).unwrap();
    let storage = usages.0.contains(TextureUsages::STORAGE_BINDING);

    // Every material in the app is deferred, and a deferred material on a camera with a depth prepass and
    // no deferred prepass panics.
    assert!(ladder.has::<DepthPrepass>(camera), "{context}");
    assert!(ladder.has::<DeferredPrepass>(camera), "{context}");
    // Deferred rendering has no MSAA.
    assert_eq!(
        ladder.world.get::<Msaa>(camera),
        Some(&Msaa::Off),
        "{context}"
    );
    // A storage-bindable main texture cannot be created in the 8-bit format a camera without `Hdr` uses.
    assert!(ladder.has::<Hdr>(camera), "{context}");
    assert_eq!(storage, traced, "{context}");
    // Jitter without something to resolve it shakes the picture; every upscaler needs motion vectors.
    assert!(ladder.has::<MotionVectorPrepass>(camera), "{context}");
    assert!(ladder.has::<TemporalJitter>(camera), "{context}");

    // Exactly one temporal anti-aliasing.
    #[cfg_attr(
        not(feature = "dlss"),
        expect(unused_mut, reason = "Only DLSS adds to the list.")
    )]
    let mut anti_aliasing = vec![
        (
            ResolvedUpscaler::Taa,
            ladder.has::<TemporalAntiAliasing>(camera),
        ),
        (ResolvedUpscaler::Tsr, ladder.has::<TestTsr>(camera)),
        (ResolvedUpscaler::Fsr, ladder.has::<TestFsr>(camera)),
    ];
    #[cfg(feature = "dlss")]
    {
        use bevy_anti_alias::dlss::{
            Dlss, DlssPerfQualityMode, DlssRayReconstructionFeature, DlssSuperResolutionFeature,
        };
        let super_resolution = ladder.world.get::<Dlss<DlssSuperResolutionFeature>>(camera);
        let ray_reconstruction = ladder
            .world
            .get::<Dlss<DlssRayReconstructionFeature>>(camera);
        anti_aliasing.push((
            ResolvedUpscaler::DlssSuperResolution,
            super_resolution.is_some(),
        ));
        anti_aliasing.push((
            ResolvedUpscaler::DlssRayReconstruction,
            ray_reconstruction.is_some(),
        ));
        // DLSS's own `Auto` picks the internal resolution by itself.
        let modes = super_resolution
            .map(|dlss| dlss.perf_quality_mode)
            .into_iter()
            .chain(ray_reconstruction.map(|dlss| dlss.perf_quality_mode));
        for mode in modes {
            assert!(
                mode == crate::dlss::perf_quality_mode(state.resolved.quality)
                    && mode != DlssPerfQualityMode::Auto,
                "{context}"
            );
        }
    }
    for (upscaler, present) in anti_aliasing {
        assert_eq!(
            present,
            upscaler == state.resolved.upscaler,
            "{upscaler:?} in {context}"
        );
    }
    assert_eq!(
        ladder.has::<TestNrd>(camera),
        state.resolved.denoiser == ResolvedDenoiser::Nrd,
        "{context}"
    );

    // The rung's lighting.
    #[cfg(feature = "solari")]
    {
        use bevy_core_pipeline::prepass::{DeferredPrepassDoubleBuffer, DepthPrepassDoubleBuffer};
        use bevy_solari::realtime::SolariLighting;

        let lighting = ladder.world.get::<SolariLighting>(camera);
        assert_eq!(lighting.is_some(), traced, "{context}");
        if let Some(lighting) = lighting {
            let expected =
                crate::traced::solari_lighting(state.resolved.rung, &state.overrides).unwrap();
            assert_eq!(lighting.max_bounces, expected.max_bounces, "{context}");
            assert_eq!(lighting.restir, expected.restir, "{context}");
            assert_eq!(
                lighting.primary_di_samples, expected.primary_di_samples,
                "{context}"
            );
        } else {
            assert!(!ladder.has::<DeferredPrepassDoubleBuffer>(camera));
            assert!(!ladder.has::<DepthPrepassDoubleBuffer>(camera));
        }
    }

    // The effects of the rung, each under its override.
    let on = |forced: Option<bool>| !traced && forced != Some(false);
    assert_eq!(
        ladder.has::<ScreenSpaceAmbientOcclusion>(camera),
        on(state.overrides.ssao),
        "{context}"
    );
    assert_eq!(
        ladder.has::<NormalPrepass>(camera),
        on(state.overrides.ssao),
        "{context}"
    );
    assert_eq!(
        ladder.has::<ScreenSpaceReflections>(camera),
        on(state.overrides.ssr),
        "{context}"
    );
    assert_eq!(
        ladder.has::<ContactShadows>(camera),
        on(state.overrides.contact_shadows),
        "{context}"
    );
    assert_eq!(
        ladder.has::<Bloom>(camera),
        state.overrides.bloom != Some(false),
        "{context}"
    );
    assert_eq!(
        ladder.world.get::<ShadowFilteringMethod>(camera).copied(),
        (!traced).then_some(ShadowFilteringMethod::Temporal),
        "{context}"
    );
}

#[test]
fn every_state_holds_its_invariants_from_a_fresh_camera() {
    for state in every_state() {
        let mut ladder = Ladder::new();
        let camera = ladder.camera();
        ladder.apply(&state);
        assert_invariants(&ladder, camera, &state);
    }
}

/// Switching presets at run time: from every state to every other one, the camera ends up exactly as a
/// camera that was given the second state from the start, and nothing the first state added is left behind.
#[test]
fn every_switch_ends_like_a_fresh_camera() {
    let states = every_state();
    for to in &states {
        let mut fresh = Ladder::new();
        let fresh_camera = fresh.camera();
        fresh.apply(to);
        let expected = fresh.components(fresh_camera);

        for from in &states {
            let mut ladder = Ladder::new();
            let camera = ladder.camera();
            ladder.apply(from);
            ladder.apply(to);
            assert_invariants(&ladder, camera, to);
            assert_eq!(
                ladder.components(camera),
                expected,
                "from {:?} with {:?} to {:?} with {:?}",
                from.resolved,
                from.overrides,
                to.resolved,
                to.overrides
            );
        }
    }
}

#[test]
fn the_preset_cycle_returns_to_the_first_raster() {
    let mut ladder = Ladder::new();
    let camera = ladder.camera();
    ladder.apply(&State::raster());
    let first = ladder.components(camera);
    for state in [
        State::traced(GraphicsRung::RtLite),
        State::traced(GraphicsRung::RtFull),
        State::raster(),
    ] {
        ladder.apply(&state);
        assert_invariants(&ladder, camera, &state);
    }
    assert_eq!(ladder.components(camera), first);
    assert_eq!(
        ladder
            .world
            .get::<AppliedGraphics>(camera)
            .and_then(AppliedGraphics::rung),
        Some(GraphicsRung::Raster)
    );
}

#[test]
fn a_camera_without_the_marker_is_left_alone() {
    let mut ladder = Ladder::new();
    let camera = ladder.world.spawn(Camera3d::default()).id();
    let before = ladder.components(camera);
    ladder.apply(&State::traced(GraphicsRung::RtFull));
    assert_eq!(ladder.components(camera), before);
}

#[test]
fn the_games_own_effect_values_survive_a_cycle() {
    let mut ladder = Ladder::new();
    let own_bloom = Bloom {
        intensity: 0.42,
        ..Bloom::NATURAL
    };
    let own_reflections = ScreenSpaceReflections {
        linear_steps: 7,
        ..Default::default()
    };
    let camera = ladder
        .world
        .spawn((
            Camera3d::default(),
            GraphicsCamera,
            own_bloom,
            own_reflections,
        ))
        .id();
    let bloom = |ladder: &Ladder| ladder.world.get::<Bloom>(camera).map(|b| b.intensity);
    let reflection_steps = |ladder: &Ladder| {
        ladder
            .world
            .get::<ScreenSpaceReflections>(camera)
            .map(|ssr| ssr.linear_steps)
    };

    ladder.apply(&State::raster());
    assert_eq!(bloom(&ladder), Some(0.42));
    assert_eq!(reflection_steps(&ladder), Some(7));

    // A traced rung has no screen-space reflections; an override takes the bloom away.
    ladder.apply(
        &State::traced(GraphicsRung::RtFull).with(GraphicsOverrides {
            bloom: Some(false),
            ..Default::default()
        }),
    );
    assert_eq!(bloom(&ladder), None);
    assert_eq!(reflection_steps(&ladder), None);

    ladder.apply(&State::raster());
    assert_eq!(bloom(&ladder), Some(0.42));
    assert_eq!(reflection_steps(&ladder), Some(7));
}

#[test]
fn the_games_own_normal_prepass_and_storage_usage_stay() {
    let mut ladder = Ladder::new();
    let camera = ladder
        .world
        .spawn((
            Camera3d::default(),
            GraphicsCamera,
            NormalPrepass,
            CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        ))
        .id();
    for state in [
        State::raster(),
        State::traced(GraphicsRung::RtLite),
        State::raster(),
    ] {
        ladder.apply(&state);
        assert!(ladder.has::<NormalPrepass>(camera));
        assert!(ladder
            .world
            .get::<CameraMainTextureUsages>(camera)
            .unwrap()
            .0
            .contains(TextureUsages::STORAGE_BINDING));
    }
}

#[test]
fn shadow_maps_go_off_on_a_traced_rung_and_come_back() {
    let mut ladder = Ladder::new();
    let shadowed = ladder
        .world
        .spawn(PointLight {
            shadow_maps_enabled: true,
            ..Default::default()
        })
        .id();
    let unshadowed = ladder.world.spawn(SpotLight::default()).id();
    let sun = ladder
        .world
        .spawn(DirectionalLight {
            shadow_maps_enabled: true,
            contact_shadows_enabled: true,
            ..Default::default()
        })
        .id();
    let flags = |ladder: &Ladder| {
        let point = ladder.world.get::<PointLight>(shadowed).unwrap();
        let spot = ladder.world.get::<SpotLight>(unshadowed).unwrap();
        let sun = ladder.world.get::<DirectionalLight>(sun).unwrap();
        [
            (point.shadow_maps_enabled, point.contact_shadows_enabled),
            (spot.shadow_maps_enabled, spot.contact_shadows_enabled),
            (sun.shadow_maps_enabled, sun.contact_shadows_enabled),
        ]
    };

    // Raster: contact shadows join the shadow maps, and a light without shadows stays without.
    let raster = [(true, true), (false, false), (true, true)];
    ladder.apply(&State::raster());
    assert_eq!(flags(&ladder), raster);

    // Traced: no shadow map anywhere; the flags the game set itself are untouched.
    for rung in [GraphicsRung::RtLite, GraphicsRung::RtFull] {
        ladder.apply(&State::traced(rung));
        assert_eq!(
            flags(&ladder),
            [(false, false), (false, false), (false, true)]
        );
    }

    ladder.apply(&State::raster());
    assert_eq!(flags(&ladder), raster);

    // The overrides: no shadow maps at all, and no contact shadows added by the ladder.
    ladder.apply(&State::raster().with(GraphicsOverrides {
        shadows: Some(false),
        ..Default::default()
    }));
    assert_eq!(
        flags(&ladder),
        [(false, false), (false, false), (false, true)]
    );
    ladder.apply(&State::raster().with(GraphicsOverrides {
        contact_shadows: Some(false),
        ..Default::default()
    }));
    assert_eq!(
        flags(&ladder),
        [(true, false), (false, false), (true, true)]
    );

    ladder.apply(&State::raster());
    assert_eq!(flags(&ladder), raster);
    let state = ladder.world.get::<LightShadowState>(shadowed).unwrap();
    assert!(state.own_shadow_maps_enabled());
    assert!(!state.own_contact_shadows_enabled());
}

#[test]
fn a_light_the_game_changes_while_traced_keeps_the_games_wish() {
    let mut ladder = Ladder::new();
    let light = ladder
        .world
        .spawn(PointLight {
            shadow_maps_enabled: true,
            ..Default::default()
        })
        .id();
    let late = ladder.world.spawn(PointLight::default()).id();
    ladder.apply(&State::traced(GraphicsRung::RtFull));

    // The game switches shadows on for a light that had none, and a new shadowed light appears.
    ladder
        .world
        .get_mut::<PointLight>(late)
        .unwrap()
        .shadow_maps_enabled = true;
    let spawned = ladder
        .world
        .spawn(PointLight {
            shadow_maps_enabled: true,
            ..Default::default()
        })
        .id();
    ladder.apply(&State::traced(GraphicsRung::RtFull));
    for entity in [light, late, spawned] {
        assert!(
            !ladder
                .world
                .get::<PointLight>(entity)
                .unwrap()
                .shadow_maps_enabled
        );
    }

    ladder.apply(&State::raster());
    for entity in [light, late, spawned] {
        assert!(
            ladder
                .world
                .get::<PointLight>(entity)
                .unwrap()
                .shadow_maps_enabled
        );
    }
}

#[test]
fn lights_are_not_marked_changed_when_nothing_changes() {
    let mut ladder = Ladder::new();
    let light = ladder
        .world
        .spawn(PointLight {
            shadow_maps_enabled: true,
            ..Default::default()
        })
        .id();
    ladder.apply(&State::traced(GraphicsRung::RtLite));
    ladder.world.clear_trackers();
    let before = ladder.world.entity(light).get_change_ticks::<PointLight>();
    ladder.apply(&State::traced(GraphicsRung::RtLite));
    ladder.apply(&State::traced(GraphicsRung::RtLite));
    let after = ladder.world.entity(light).get_change_ticks::<PointLight>();
    assert_eq!(
        before.map(|ticks| ticks.changed),
        after.map(|ticks| ticks.changed)
    );
}
