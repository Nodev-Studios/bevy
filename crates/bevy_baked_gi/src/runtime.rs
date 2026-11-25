use crate::{
    light_power, BakedGiManifest, BakedGiManifestLoader, BakedLightKind, BakedRegion, GiBakeRegion,
    GiBakeSkipEmission, LightKey,
};
use alloc::vec::Vec;
use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{AssetApp, AssetId, AssetServer, Assets, Handle};
use bevy_camera::visibility::{InheritedVisibility, Visibility};
use bevy_ecs::{
    change_detection::{DetectChanges, DetectChangesMut},
    component::Component,
    entity::Entity,
    hierarchy::ChildOf,
    lifecycle::RemovedComponents,
    name::Name,
    query::{AnyOf, Changed, Or, With},
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Local, Query, Res, ResMut},
};
use bevy_light::{
    DirectionalLight, IrradianceVolume, LightProbe, PointLight, RectLight, SpotLight,
};
use bevy_math::Vec3;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::renderer::RenderDevice;
use bevy_transform::{
    components::{GlobalTransform, Transform},
    TransformSystems,
};
use tracing::warn;
use wgpu_types::{Features, Limits};

/// Loads baked irradiance volumes and keeps their intensity in step with the lights.
///
/// Add a [`BakedGi`] entity naming a manifest; the plugin spawns one [`LightProbe`] + [`IrradianceVolume`] child
/// per baked region and reports what happened in [`BakedGiStatus`]. It needs no ray tracing.
pub struct BakedGiPlugin;

impl Plugin for BakedGiPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<BakedGiManifest>()
            .init_asset_loader::<BakedGiManifestLoader>()
            .init_resource::<BakedGiStatus>()
            .register_type::<GiBakeRegion>()
            .register_type::<GiBakeSkipEmission>()
            .add_systems(
                PostUpdate,
                (
                    spawn_baked_volumes,
                    check_stale_regions,
                    update_volume_intensities,
                    update_status,
                )
                    .chain()
                    .after(TransformSystems::Propagate),
            );
    }
}

/// The root of a loaded bake: its children are the baked volumes of the manifest it names.
#[derive(Component, Clone, Debug)]
#[require(Transform, Visibility)]
pub struct BakedGi(pub Handle<BakedGiManifest>);

/// One baked volume, a child of its [`BakedGi`]: region `region` of the manifest.
#[derive(Component, Clone, Copy, Debug)]
pub struct BakedGiVolume {
    /// The index of the region in the manifest's `regions`.
    pub region: usize,
}

/// What the runtime did with the bake.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BakedGiStatus {
    /// No bake is loaded: no [`BakedGi`] entity, its manifest is still loading, or it failed to load.
    #[default]
    Absent,
    /// The volumes are spawned.
    Loaded,
    /// The volumes are spawned, but a [`GiBakeRegion`] of the level is not where the bake put it: rebake.
    Stale,
    /// The device cannot bind arrays of textures, so Bevy would use only one volume per view: the bake is not
    /// loaded and the game keeps the lighting it had without one.
    Degraded,
}

/// Where a root stands.
#[derive(Component)]
struct BakedGiState {
    manifest: AssetId<BakedGiManifest>,
    degraded: bool,
    stale: bool,
}

/// Bevy's light probe shader binds up to 16 texture bindings for the standard material and one per light probe in
/// one array; with fewer elements per stage it falls back to a single volume. This mirrors the crate-private
/// check `binding_arrays_are_usable` of `bevy_pbr`'s `light_probe/mod.rs`, minus its Adreno 610 exception.
pub fn binding_arrays_are_usable(limits: &Limits, features: Features) -> bool {
    const STANDARD_MATERIAL_FRAGMENT_SHADER_MIN_TEXTURE_BINDINGS: u32 = 16;
    let needed = STANDARD_MATERIAL_FRAGMENT_SHADER_MIN_TEXTURE_BINDINGS
        + bevy_pbr::MAX_VIEW_LIGHT_PROBES as u32;
    limits.max_binding_array_elements_per_shader_stage >= needed
        && limits.max_binding_array_sampler_elements_per_shader_stage >= needed
        && features.contains(
            Features::TEXTURE_BINDING_ARRAY
                | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        )
}

/// The intensity of a baked volume when each of the manifest's lights shines at `scales` times its baked power
/// (0 for a light that is gone or hidden): `static + Σ weight × scale`. A region no light reached keeps 1.
pub fn region_intensity(region: &BakedRegion, scales: &[f32]) -> f32 {
    if region.bounce <= 0.0 {
        return 1.0;
    }
    region.static_weight
        + region
            .light_weights
            .iter()
            .zip(scales)
            .map(|(weight, scale)| weight * scale)
            .sum::<f32>()
}

fn spawn_baked_volumes(
    mut commands: Commands,
    roots: Query<(Entity, &BakedGi, Option<&BakedGiState>)>,
    manifests: Res<Assets<BakedGiManifest>>,
    asset_server: Option<Res<AssetServer>>,
    render_device: Option<Res<RenderDevice>>,
    volumes: Query<(Entity, &ChildOf), With<BakedGiVolume>>,
    mut failed: Local<HashSet<AssetId<BakedGiManifest>>>,
) {
    for (root, baked, state) in &roots {
        let id = baked.0.id();
        if state.is_some_and(|state| state.manifest == id) {
            continue;
        }
        let Some(manifest) = manifests.get(id) else {
            if let Some(server) = &asset_server
                && server.load_state(id).is_failed()
                && failed.insert(id)
            {
                warn!("baked GI manifest {:?} failed to load", baked.0.path());
            }
            continue;
        };
        for (volume, parent) in &volumes {
            if parent.parent() == root {
                commands.entity(volume).despawn();
            }
        }
        let degraded = render_device
            .as_ref()
            .is_some_and(|device| !binding_arrays_are_usable(&device.limits(), device.features()));
        if degraded {
            warn!(
                "baked GI not loaded: this device cannot bind arrays of light probe textures, so Bevy would \
                 light every view with one volume"
            );
        } else {
            for (index, region) in manifest.regions.iter().enumerate() {
                commands.spawn((
                    BakedGiVolume { region: index },
                    LightProbe {
                        falloff: Vec3::from_array(region.falloff),
                    },
                    IrradianceVolume {
                        voxels: manifest.volumes.get(index).cloned().unwrap_or_default(),
                        // The component defaults to 0; the light-following system sets the real value.
                        intensity: 1.0,
                        // A lightmapped mesh has its indirect light from its lightmap.
                        affects_lightmapped_meshes: false,
                    },
                    region.transform(),
                    Name::new(region.name.clone()),
                    ChildOf(root),
                ));
            }
        }
        commands.entity(root).insert(BakedGiState {
            manifest: id,
            degraded,
            stale: false,
        });
    }
}

/// Region markers left in the level are bake input; at run time they only tell whether the level moved them.
fn check_stale_regions(
    mut roots: Query<(&BakedGi, &mut BakedGiState)>,
    markers: Query<(&Name, &GlobalTransform), With<GiBakeRegion>>,
    changed_markers: Query<(), (With<GiBakeRegion>, Changed<GlobalTransform>)>,
    manifests: Res<Assets<BakedGiManifest>>,
) {
    let markers_changed = !changed_markers.is_empty();
    for (baked, mut state) in &mut roots {
        if !markers_changed && !state.is_changed() {
            continue;
        }
        let Some(manifest) = manifests.get(baked.0.id()) else {
            continue;
        };
        let mut stale = false;
        for (name, transform) in &markers {
            let Some(region) = manifest.regions.iter().find(|r| r.name == name.as_str()) else {
                warn!("baked GI is stale: region {name} is not in the bake; rebake");
                stale = true;
                continue;
            };
            let (scale, rotation, translation) = transform.to_scale_rotation_translation();
            let baked = region.transform();
            if translation.distance(baked.translation) > 1e-3
                || scale.distance(baked.scale) > 1e-3
                || rotation.angle_between(baked.rotation) > 1e-3
            {
                warn!("baked GI is stale: region {name} moved since the bake; rebake");
                stale = true;
            }
        }
        if state.stale != stale {
            state.bypass_change_detection().stale = stale;
        }
    }
}

type AnyLight<'a> = AnyOf<(
    &'a PointLight,
    &'a SpotLight,
    &'a RectLight,
    &'a DirectionalLight,
)>;

type LightChanged = (
    Or<(
        Changed<PointLight>,
        Changed<SpotLight>,
        Changed<RectLight>,
        Changed<DirectionalLight>,
        Changed<GlobalTransform>,
        Changed<InheritedVisibility>,
    )>,
    Or<(
        With<PointLight>,
        With<SpotLight>,
        With<RectLight>,
        With<DirectionalLight>,
    )>,
);

/// Each live, visible light's key and power. Two lights at one key add up.
fn live_light_powers<'a>(
    lights: impl Iterator<
        Item = (
            &'a GlobalTransform,
            Option<&'a InheritedVisibility>,
            (
                Option<&'a PointLight>,
                Option<&'a SpotLight>,
                Option<&'a RectLight>,
                Option<&'a DirectionalLight>,
            ),
        ),
    >,
) -> HashMap<LightKey, f32> {
    let mut powers = HashMap::default();
    for (transform, visibility, (point, spot, rect, directional)) in lights {
        if visibility.is_some_and(|visibility| !visibility.get()) {
            continue;
        }
        let (kind, power) = if let Some(light) = spot {
            (
                BakedLightKind::Spot,
                light_power(light.color, light.intensity),
            )
        } else if let Some(light) = point {
            (
                BakedLightKind::Point,
                light_power(light.color, light.intensity),
            )
        } else if let Some(light) = rect {
            (
                BakedLightKind::Rect,
                light_power(light.color, light.intensity),
            )
        } else if let Some(light) = directional {
            (
                BakedLightKind::Directional,
                light_power(light.color, light.illuminance),
            )
        } else {
            continue;
        };
        *powers.entry(LightKey::new(kind, transform)).or_insert(0.0) += power;
    }
    powers
}

fn update_volume_intensities(
    roots: Query<&BakedGi>,
    mut volumes: Query<(&BakedGiVolume, &ChildOf, &mut IrradianceVolume)>,
    lights: Query<(&GlobalTransform, Option<&InheritedVisibility>, AnyLight)>,
    changed_lights: Query<(), LightChanged>,
    added_volumes: Query<(), bevy_ecs::query::Added<BakedGiVolume>>,
    mut removed: (
        RemovedComponents<PointLight>,
        RemovedComponents<SpotLight>,
        RemovedComponents<RectLight>,
        RemovedComponents<DirectionalLight>,
    ),
    manifests: Res<Assets<BakedGiManifest>>,
    mut warned: Local<HashSet<LightKey>>,
) {
    let removed_count = removed.0.read().count()
        + removed.1.read().count()
        + removed.2.read().count()
        + removed.3.read().count();
    if changed_lights.is_empty() && added_volumes.is_empty() && removed_count == 0 {
        return;
    }

    let live = live_light_powers(lights.iter());
    let mut baked_keys: HashSet<LightKey> = HashSet::default();
    let mut scales_by_manifest: HashMap<AssetId<BakedGiManifest>, Vec<f32>> = HashMap::default();
    for (volume, parent, mut irradiance) in &mut volumes {
        let Ok(baked) = roots.get(parent.parent()) else {
            continue;
        };
        let Some(manifest) = manifests.get(baked.0.id()) else {
            continue;
        };
        let scales = scales_by_manifest.entry(baked.0.id()).or_insert_with(|| {
            manifest
                .lights
                .iter()
                .map(|light| live.get(&light.key).copied().unwrap_or(0.0) / light.power)
                .collect()
        });
        baked_keys.extend(manifest.lights.iter().map(|light| light.key));
        let Some(region) = manifest.regions.get(volume.region) else {
            continue;
        };
        let intensity = region_intensity(region, scales);
        if irradiance.intensity != intensity {
            irradiance.intensity = intensity;
        }
    }

    for (key, power) in live {
        if power > 0.0 && !baked_keys.is_empty() && !baked_keys.contains(&key) && warned.insert(key)
        {
            warn!(
                "{key:?} shines but was not in the bake (absent, hidden or off when baked): its bounce light \
                 needs a rebake"
            );
        }
    }
}

fn update_status(roots: Query<&BakedGiState>, mut status: ResMut<BakedGiStatus>) {
    let next = if roots.iter().any(|state| state.degraded) {
        BakedGiStatus::Degraded
    } else if roots.iter().any(|state| state.stale) {
        BakedGiStatus::Stale
    } else if roots.is_empty() {
        BakedGiStatus::Absent
    } else {
        BakedGiStatus::Loaded
    };
    status.set_if_neq(next);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BakeProvenance, BakedLight, MANIFEST_FORMAT};
    use bevy_app::App;
    use bevy_asset::AssetPlugin;
    use bevy_image::Image;

    fn region(name: &str, bounce: f64, static_weight: f32, light_weights: Vec<f32>) -> BakedRegion {
        BakedRegion {
            name: name.into(),
            file: format!("{name}.ktx2"),
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [4.0, 3.2, 4.0],
            falloff: [0.0; 3],
            resolution: [12, 10, 12],
            invalid_voxels: 0,
            bounce,
            static_weight,
            light_weights,
        }
    }

    /// A lamp as the app would have it after visibility propagation, which this test app does not run.
    fn spot_at(x: f32) -> (SpotLight, GlobalTransform, Transform, InheritedVisibility) {
        (
            SpotLight {
                intensity: 1000.0,
                ..Default::default()
            },
            GlobalTransform::from_xyz(x, 2.9, 0.0),
            Transform::from_xyz(x, 2.9, 0.0),
            InheritedVisibility::VISIBLE,
        )
    }

    #[test]
    fn intensity_formula_weighs_every_light() {
        // Room B is lit 70 % by its own lamp, 20 % by A's lamp through a door, 10 % by a glowing panel.
        let b = region("b", 5.0, 0.1, vec![0.2, 0.7]);
        assert!((region_intensity(&b, &[1.0, 1.0]) - 1.0).abs() < 1e-6);
        // A's lamp switched off (removed): B dims by A's share.
        assert!((region_intensity(&b, &[0.0, 1.0]) - 0.8).abs() < 1e-6);
        // B's own lamp at half power.
        assert!((region_intensity(&b, &[1.0, 0.5]) - 0.65).abs() < 1e-6);
        // A region no light reached stays at 1 whatever the lights do.
        let dark = region("dark", 0.0, 0.0, vec![0.0, 0.0]);
        assert_eq!(region_intensity(&dark, &[0.0, 0.0]), 1.0);
        // A lampless region lit only by emissive meshes does not follow the lamps.
        let glow = region("glow", 2.0, 1.0, vec![0.0, 0.0]);
        assert_eq!(region_intensity(&glow, &[0.0, 0.0]), 1.0);
    }

    #[test]
    fn degraded_when_binding_arrays_are_too_small() {
        let features = Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        let mut limits = Limits::default();
        limits.max_binding_array_elements_per_shader_stage = 1000;
        limits.max_binding_array_sampler_elements_per_shader_stage = 1000;
        assert!(binding_arrays_are_usable(&limits, features));
        assert!(!binding_arrays_are_usable(
            &limits,
            Features::TEXTURE_BINDING_ARRAY
        ));
        limits.max_binding_array_sampler_elements_per_shader_stage = 8;
        assert!(!binding_arrays_are_usable(&limits, features));
    }

    fn app_with_bake() -> (App, Entity) {
        let mut app = App::new();
        app.add_plugins((AssetPlugin::default(), BakedGiPlugin))
            .init_asset::<Image>();
        let a_key = LightKey::new(
            BakedLightKind::Spot,
            &GlobalTransform::from_xyz(0.0, 2.9, 0.0),
        );
        let b_key = LightKey::new(
            BakedLightKind::Spot,
            &GlobalTransform::from_xyz(5.0, 2.9, 0.0),
        );
        let manifest = BakedGiManifest {
            format: MANIFEST_FORMAT,
            scene_hash: "0".into(),
            bake: BakeProvenance {
                baker: "test".into(),
                adapter: "none".into(),
                driver: "none".into(),
                mode: "full".into(),
                samples: 1,
                seed: 0,
                invalid_back_face_share: 0.25,
                texel_format: "Rgba16Float".into(),
            },
            lights: vec![
                BakedLight {
                    key: a_key,
                    power: 1000.0,
                },
                BakedLight {
                    key: b_key,
                    power: 1000.0,
                },
            ],
            regions: vec![
                region("a", 9.0, 0.0, vec![0.9, 0.1]),
                region("b", 5.0, 0.1, vec![0.2, 0.7]),
                region("c", 0.0, 0.0, vec![0.0, 0.0]),
            ],
            volumes: Vec::new(),
        };
        let handle = app
            .world_mut()
            .resource_mut::<Assets<BakedGiManifest>>()
            .add(manifest);
        app.world_mut().spawn(spot_at(0.0));
        let b_lamp = app.world_mut().spawn(spot_at(5.0)).id();
        app.world_mut().spawn(BakedGi(handle));
        app.update();
        (app, b_lamp)
    }

    fn intensities(app: &mut App) -> Vec<f32> {
        let mut found: Vec<(usize, f32)> = app
            .world_mut()
            .query::<(&BakedGiVolume, &IrradianceVolume)>()
            .iter(app.world())
            .map(|(volume, irradiance)| (volume.region, irradiance.intensity))
            .collect();
        found.sort_by_key(|(region, _)| *region);
        found.into_iter().map(|(_, intensity)| intensity).collect()
    }

    #[test]
    fn runtime_spawns_volumes_and_follows_lamps() {
        let (mut app, b_lamp) = app_with_bake();
        app.update();
        assert_eq!(
            *app.world().resource::<BakedGiStatus>(),
            BakedGiStatus::Loaded
        );
        let all_on = intensities(&mut app);
        assert_eq!(all_on.len(), 3);
        assert!(all_on.iter().all(|i| (i - 1.0).abs() < 1e-5), "{all_on:?}");

        // Despawning B's lamp removes its share from both rooms; the unlit region C keeps 1.
        app.world_mut().despawn(b_lamp);
        app.update();
        let b_off = intensities(&mut app);
        assert!((b_off[0] - 0.9).abs() < 1e-5, "{b_off:?}");
        assert!((b_off[1] - 0.3).abs() < 1e-5, "{b_off:?}");
        assert_eq!(b_off[2], 1.0);
    }

    #[test]
    fn hidden_and_dimmed_lamps_count_by_their_live_power() {
        let (mut app, b_lamp) = app_with_bake();
        app.update();
        app.world_mut()
            .entity_mut(b_lamp)
            .insert(InheritedVisibility::HIDDEN);
        app.update();
        assert!((intensities(&mut app)[1] - 0.3).abs() < 1e-5);
        app.world_mut()
            .entity_mut(b_lamp)
            .insert(InheritedVisibility::VISIBLE)
            .get_mut::<SpotLight>()
            .unwrap()
            .intensity = 500.0;
        app.update();
        assert!((intensities(&mut app)[1] - 0.65).abs() < 1e-5);
    }

    #[test]
    fn a_moved_region_marker_makes_the_bake_stale() {
        let (mut app, _) = app_with_bake();
        app.world_mut().spawn((
            GiBakeRegion::default(),
            Name::new("a"),
            GlobalTransform::from(region("a", 0.0, 0.0, vec![]).transform()),
        ));
        app.update();
        assert_eq!(
            *app.world().resource::<BakedGiStatus>(),
            BakedGiStatus::Loaded
        );
        app.world_mut().spawn((
            GiBakeRegion::default(),
            Name::new("b"),
            GlobalTransform::from_xyz(0.5, 0.0, 0.0)
                .mul_transform(Transform::from_scale(Vec3::new(4.0, 3.2, 4.0))),
        ));
        app.update();
        assert_eq!(
            *app.world().resource::<BakedGiStatus>(),
            BakedGiStatus::Stale
        );
    }
}
