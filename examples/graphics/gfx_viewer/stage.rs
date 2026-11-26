//! Opening a scene: despawning the last one, spawning the next, placing the camera, and giving a glTF file a sun
//! and a sky when it brings no light of its own.

use crate::{
    baked as testbed_baked,
    catalog::{fs_asset_path, Catalog, Source},
    contract, scenes, Args, ViewerCamera,
};
use bevy::{
    camera::{primitives::Aabb, Exposure},
    camera_controller::free_camera::{FreeCamera, FreeCameraState},
    graphics::baked::GiBakeRegion,
    light::{
        atmosphere::ScatteringMedium, light_consts::lux, Atmosphere, AtmosphereEnvironmentMapLight,
    },
    pbr::AtmosphereSettings,
    prelude::*,
    world_serialization::{WorldAssetRoot, WorldInstanceReady},
};
use std::collections::HashSet;

/// Everything a scene spawned; despawned when another scene opens.
#[derive(Component)]
pub struct SceneEntity;

/// The sun the viewer gives a glTF scene that has no light.
#[derive(Component)]
pub struct DefaultSun;

/// The direction of the default sun: high and from the side, so it reaches into a courtyard such as Sponza's.
/// to tune.
pub fn default_sun_transform() -> Transform {
    Transform::from_xyz(0.0, 0.0, 0.0).looking_to(Vec3::new(-0.35, -1.0, -0.25), Vec3::Y)
}

/// The default sun's light: unfiltered sunlight, as the atmosphere expects. to tune.
pub fn default_sun() -> DirectionalLight {
    DirectionalLight {
        illuminance: lux::RAW_SUNLIGHT,
        shadow_maps_enabled: true,
        ..default()
    }
}

/// The exposure of a glTF scene under the default sun. to tune: the atmosphere example's 13 makes a shaded
/// courtyard dark under Raster, which has no bounce light without a bake.
const SUN_EV100: f32 = 12.0;

/// The scene that is open.
#[derive(Resource, Default)]
pub struct Stage {
    /// The scene asked for; [`open_requested_scene`] opens it.
    pub requested: Option<usize>,
    /// The index in the [`Catalog`] of the open scene.
    pub index: usize,
    /// Its name.
    pub name: String,
    /// The assets it waits for.
    pub assets: Vec<UntypedHandle>,
    /// Where the camera starts.
    pub start: Transform,
    /// The camera's vertical field of view.
    pub fov: f32,
    /// The glTF scene's root, while it is a glTF scene.
    pub gltf_root: Option<Entity>,
    /// The glTF scene's instance is spawned.
    pub instance_ready: bool,
    /// Everything is placed: the camera stands where it starts, the default lights are there.
    pub placed: bool,
    /// The world-space bounds of the scene's meshes, once known.
    pub bounds: Option<(Vec3, Vec3)>,
    /// The viewer gave this scene the default sun (and the bake must too).
    pub default_sun: bool,
    /// How many times a scene was opened, so systems can tell a scene from the next.
    pub generation: u32,
    /// A file of the scene failed to load.
    pub failed: bool,
}

/// Notes when a file of the open scene fails to load, for the overlay.
pub fn watch_loading(server: Res<AssetServer>, mut stage: ResMut<Stage>) {
    if stage.failed {
        return;
    }
    let failed = stage.assets.iter().any(|asset| {
        matches!(
            server.recursive_dependency_load_state(asset.id()),
            bevy::asset::RecursiveDependencyLoadState::Failed(_)
        )
    });
    if failed {
        stage.failed = true;
    }
}

impl Stage {
    /// Whether the open scene is a glTF file.
    pub fn is_gltf(&self) -> bool {
        self.gltf_root.is_some()
    }
}

/// Opens the scene [`Stage::requested`] names.
pub fn open_requested_scene(world: &mut World) {
    let Some(index) = world.resource_mut::<Stage>().requested.take() else {
        return;
    };
    let entry = world.resource::<Catalog>().entries[index].clone();
    info!("gfx_viewer: opening scene {}", entry.name);

    let old: Vec<Entity> = world
        .query_filtered::<Entity, With<SceneEntity>>()
        .iter(world)
        .collect();
    for entity in old {
        if let Ok(entity) = world.get_entity_mut(entity) {
            entity.despawn();
        }
    }
    world.insert_resource(ClearColor(Color::BLACK));
    world.insert_resource(GlobalAmbientLight::NONE);
    world.resource_mut::<Args>().scene = entry.name.clone();

    // Only what has a place in the world is the scene's: resources and cached systems are entities too.
    let before: HashSet<Entity> = world
        .query_filtered::<Entity, With<Transform>>()
        .iter(world)
        .collect();
    let mut stage = Stage {
        index,
        name: entry.name.clone(),
        generation: world.resource::<Stage>().generation + 1,
        ..default()
    };
    match &entry.source {
        Source::Builtin(name) => {
            let scene = scenes::spawn(name, world)
                .or_else(|| contract::Contract::scene(name, world))
                .or_else(|| testbed_baked::Baked::scene(name, world))
                .expect("every builtin name is a testbed scene");
            // The scene's bake regions, as the testbed places them.
            let scene = testbed_baked::Baked::default().prepare(world, scene);
            stage.start = (scene.orbit)(0);
            stage.fov = scene.fov;
            stage.assets = scene.assets;
            stage.instance_ready = true;
            stage.placed = true;
        }
        Source::Gltf(file) => {
            let handle: Handle<WorldAsset> = world
                .resource::<AssetServer>()
                .load(GltfAssetLabel::Scene(0).from_asset(fs_asset_path(file)));
            stage.assets = vec![handle.clone().untyped()];
            stage.fov = 1.0;
            stage.start = Transform::from_xyz(0.0, 1.6, 3.0).looking_at(Vec3::ZERO, Vec3::Y);
            let root = world
                .spawn((Name::new(entry.name.clone()), WorldAssetRoot(handle)))
                .observe(|_: On<WorldInstanceReady>, mut stage: ResMut<Stage>| {
                    stage.instance_ready = true;
                })
                .id();
            stage.gltf_root = Some(root);
        }
    }
    let after: Vec<Entity> = world
        .query_filtered::<Entity, With<Transform>>()
        .iter(world)
        .collect();
    for entity in after.into_iter().filter(|entity| !before.contains(entity)) {
        world.entity_mut(entity).insert(SceneEntity);
    }

    let gltf = stage.gltf_root.is_some();
    let (start, fov) = (stage.start, stage.fov);
    *world.resource_mut::<Stage>() = stage;
    place_camera(world, start, fov, 1.0);
    let camera = world
        .query_filtered::<Entity, With<ViewerCamera>>()
        .single(world)
        .expect("the viewer has one camera");
    if gltf {
        // The sky: an atmosphere lights and reflects on Raster through its environment map, and Solari traces
        // it on the traced rungs.
        let medium = world
            .resource_mut::<Assets<ScatteringMedium>>()
            .add(ScatteringMedium::earth(256, 256));
        world.spawn((SceneEntity, Atmosphere::earth(medium)));
        world.entity_mut(camera).insert((
            AtmosphereSettings::default(),
            AtmosphereEnvironmentMapLight::default(),
            Exposure { ev100: SUN_EV100 },
        ));
    } else {
        world
            .entity_mut(camera)
            .remove::<(AtmosphereSettings, AtmosphereEnvironmentMapLight)>()
            .insert(Exposure::default());
    }
}

/// Puts the camera at `start`, and restarts the free camera there (it keeps its own yaw and pitch).
pub fn place_camera(world: &mut World, start: Transform, fov: f32, speed: f32) {
    let Ok(camera) = world
        .query_filtered::<Entity, With<ViewerCamera>>()
        .single(world)
    else {
        return;
    };
    world.entity_mut(camera).insert((
        start,
        Projection::Perspective(PerspectiveProjection { fov, ..default() }),
        FreeCamera {
            walk_speed: 3.0 * speed,
            run_speed: 12.0 * speed,
            ..default()
        },
        FreeCameraState::default(),
    ));
}

/// Once a glTF scene's instance is spawned and its meshes have bounds: places the camera from the bounds, and
/// gives the scene the default sun when it has no light.
pub fn place_gltf_scene(world: &mut World) {
    let stage = world.resource::<Stage>();
    let (Some(root), true, false) = (stage.gltf_root, stage.instance_ready, stage.placed) else {
        return;
    };
    let mut meshes =
        world.query_filtered::<(Entity, Option<&Aabb>, &GlobalTransform), With<Mesh3d>>();
    let mut parents = world.query::<&ChildOf>();
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut count = 0;
    for (entity, aabb, transform) in meshes.iter(world) {
        if !descends_from(world, &mut parents, entity, root) {
            continue;
        }
        // Bounds come a frame after the mesh loads.
        let Some(aabb) = aabb else {
            return;
        };
        let (center, half) = (Vec3::from(aabb.center), Vec3::from(aabb.half_extents));
        for corner in 0..8 {
            let sign = Vec3::new(
                if corner & 1 == 0 { -1.0 } else { 1.0 },
                if corner & 2 == 0 { -1.0 } else { 1.0 },
                if corner & 4 == 0 { -1.0 } else { 1.0 },
            );
            let point = transform.transform_point(center + sign * half);
            min = min.min(point);
            max = max.max(point);
        }
        count += 1;
    }
    if count == 0 {
        return;
    }

    let mut lights = world.query_filtered::<Entity, Or<(
        With<PointLight>,
        With<SpotLight>,
        With<RectLight>,
        With<DirectionalLight>,
    )>>();
    let has_light = lights
        .iter(world)
        .collect::<Vec<_>>()
        .into_iter()
        .any(|light| descends_from(world, &mut parents, light, root));
    if !has_light {
        world.spawn((
            SceneEntity,
            DefaultSun,
            Name::new("default sun"),
            default_sun(),
            default_sun_transform(),
        ));
    }

    // The box `B` bakes, placed now so that a bake loaded later can tell whether it still fits the scene.
    let (region_transform, region) = gltf_region(min, max);
    world.spawn((
        SceneEntity,
        Name::new("glTF bounds"),
        region,
        region_transform,
    ));

    let (start, speed) = start_from_bounds(min, max);
    let fov = world.resource::<Stage>().fov;
    {
        let mut stage = world.resource_mut::<Stage>();
        stage.bounds = Some((min, max));
        stage.start = start;
        stage.placed = true;
        stage.default_sun = !has_light;
    }
    place_camera(world, start, fov, speed);
    info!(
        "gfx_viewer: glTF scene placed: {count} meshes within {min} .. {max}, {}",
        if has_light {
            "with its own lights"
        } else {
            "no light of its own: default sun and sky added"
        }
    );
}

/// Where the camera starts in a scene with these bounds, and how fast it moves. A place (wider than 8 m) is
/// seen from inside, at standing height above its floor, looking along its longest side; an object from outside.
/// to tune.
pub fn start_from_bounds(min: Vec3, max: Vec3) -> (Transform, f32) {
    let size = max - min;
    let center = (min + max) / 2.0;
    if size.x.max(size.z) > 8.0 {
        let eye = Vec3::new(center.x, min.y + (0.15 * size.y).clamp(1.0, 2.5), center.z);
        let along = if size.x >= size.z { Vec3::X } else { Vec3::Z };
        let speed = (size.x.max(size.z) / 30.0).clamp(0.5, 4.0);
        (
            Transform::from_translation(eye).looking_to(along, Vec3::Y),
            speed,
        )
    } else {
        let radius = (size.length() / 2.0).max(0.1);
        let eye = center + Vec3::new(0.0, 0.35 * radius, 1.7 * radius);
        (
            Transform::from_translation(eye).looking_at(center, Vec3::Y),
            (radius / 3.0).clamp(0.1, 2.0),
        )
    }
}

/// Whether `entity` is `root` or one of its descendants.
fn descends_from(
    world: &World,
    parents: &mut QueryState<&ChildOf>,
    mut entity: Entity,
    root: Entity,
) -> bool {
    loop {
        if entity == root {
            return true;
        }
        match parents.get(world, entity) {
            Ok(child_of) => entity = child_of.parent(),
            Err(_) => return false,
        }
    }
}

/// The bake regions of the open scene: the testbed's for its scenes, the meshes' bounds for a glTF file.
pub fn bake_regions(world: &mut World) -> Vec<(Transform, GiBakeRegion)> {
    let mut regions = world.query::<(&Transform, &GiBakeRegion)>();
    regions.iter(world).map(|(t, r)| (*t, *r)).collect()
}

/// One box around a glTF scene's bounds, with a voxel spacing that keeps the bake small enough to wait for:
/// about 48 voxels along its longest side, never finer than the testbed's 0.33 m. to tune.
pub fn gltf_region(min: Vec3, max: Vec3) -> (Transform, GiBakeRegion) {
    let size = max - min;
    let spacing = (size.max_element() / 48.0).max(GiBakeRegion::default().spacing);
    (
        GiBakeRegion::padded_transform(min, max),
        GiBakeRegion {
            spacing,
            ..default()
        },
    )
}
