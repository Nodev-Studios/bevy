//! Opening a scene: despawning the last one, spawning the next, placing the camera, and giving a glTF file a sun
//! and a sky when it brings no light of its own.

use crate::{
    baked as testbed_baked,
    catalog::{fs_asset_path, Catalog, GltfScene, Source, SunChoice},
    contract, scenes, Args, ViewerCamera,
};
use bevy::{
    camera::{primitives::Aabb, Exposure},
    camera_controller::free_camera::{FreeCamera, FreeCameraState},
    graphics::{apply::CameraCut, baked::GiBakeRegion},
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

/// The sun a scene's entry asks for, as a light and its transform.
pub fn manifest_sun(direction: Vec3, illuminance: f32) -> (DirectionalLight, Transform) {
    (
        DirectionalLight {
            illuminance,
            ..default_sun()
        },
        Transform::default().looking_to(direction.normalize_or(Vec3::NEG_Y), Vec3::Y),
    )
}

/// The exposure of a glTF scene under the default sun. to tune: the atmosphere example's 13 makes a shaded
/// courtyard dark under Raster, which has no bounce light without a bake.
pub const SUN_EV100: f32 = 12.0;

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
    /// The glTF scene's instances are spawned (one per file).
    pub instance_ready: bool,
    /// How many of the glTF scene's files are spawned, and how many it has.
    pub instances: (usize, usize),
    /// The glTF scene's files and sun, while it is a glTF scene (the bake's child loads the same).
    pub gltf: Option<GltfScene>,
    /// Everything is placed: the camera stands where it starts, the default lights are there.
    pub placed: bool,
    /// The world-space bounds of the scene's meshes, once known.
    pub bounds: Option<(Vec3, Vec3)>,
    /// The viewer gave this scene a sun (the default one or its entry's; the bake must have it too).
    pub default_sun: bool,
    /// The exposure the scene opened with (EV100), for the Render panel's reset.
    pub exposure: f32,
    /// How many times a scene was opened, so systems can tell a scene from the next.
    pub generation: u32,
    /// A file of the scene failed to load.
    pub failed: bool,
    /// Why the scene is not (all) there: the first load error, or that the file has no mesh.
    pub error: Option<String>,
    /// Frames since the glTF scene's instances were spawned.
    pub frames_since_ready: u32,
}

/// Frames after a glTF scene is spawned before meshes still without bounds are left out of its placement (bounds
/// come a frame after a mesh; a mesh with no positions never gets them). to tune.
const BOUNDS_WAIT_FRAMES: u32 = 30;

/// Frames after a scene is placed before the allocator is asked to give freed memory back: the last scene's
/// assets and render resources are dropped over the frames after the switch. to tune.
const TRIM_AFTER_FRAMES: u32 = 60;

/// Once per opened scene, hands the memory the last scene freed back to the system. glibc keeps freed heap pages
/// otherwise (0.5 to 1.3 GB per switch between the big scenes, measured), so the process only grew.
pub fn release_freed_memory(stage: Res<Stage>, mut waited: Local<(u32, u32)>) {
    if waited.0 != stage.generation {
        *waited = (stage.generation, 0);
    }
    if !stage.placed || waited.1 > TRIM_AFTER_FRAMES {
        return;
    }
    waited.1 += 1;
    if waited.1 == TRIM_AFTER_FRAMES {
        trim_heap();
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[expect(unsafe_code, reason = "glibc's malloc_trim has no safe binding in the dependencies")]
fn trim_heap() {
    unsafe extern "C" {
        fn malloc_trim(pad: usize) -> i32;
    }
    // SAFETY: malloc_trim takes no pointer and is thread-safe.
    unsafe { malloc_trim(0) };
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn trim_heap() {}

/// When the GPU last ran out of memory: the validation errors that follow it for this long (the objects that could
/// not be created, used once more before the scene is gone) are let pass too. to tune.
const AFTER_OUT_OF_MEMORY: std::time::Duration = std::time::Duration::from_secs(5);

/// The last time the GPU ran out of memory.
#[derive(Resource)]
struct GpuOutOfMemory(std::time::Instant);

/// What the viewer does on a render error: when the GPU runs out of memory (a scene too big for it), the scene is
/// closed and the Scene panel says why, and the viewer goes on; any other render error still quits, as Bevy's
/// default does.
pub fn render_error_handler() -> bevy::render::error_handler::RenderErrorHandler {
    use bevy::render::error_handler::{ErrorType, RenderErrorHandler, RenderErrorPolicy};
    RenderErrorHandler(|error, main_world, _| {
        let recent = main_world
            .get_resource::<GpuOutOfMemory>()
            .is_some_and(|at| at.0.elapsed() < AFTER_OUT_OF_MEMORY);
        match error.ty {
            ErrorType::OutOfMemory => {
                close_scene(main_world, "the GPU ran out of memory opening it");
                main_world.insert_resource(GpuOutOfMemory(std::time::Instant::now()));
                RenderErrorPolicy::Ignore
            }
            ErrorType::Validation if recent => RenderErrorPolicy::Ignore,
            _ => {
                error!("gfx_viewer: quitting on a render error: {:?}", error.ty);
                main_world.write_message(AppExit::error());
                RenderErrorPolicy::StopRendering
            }
        }
    })
}

/// Empties the stage after the open scene could not be shown, and keeps the reason for the Scene panel.
fn close_scene(world: &mut World, reason: &str) {
    let entities: Vec<Entity> = world
        .query_filtered::<Entity, With<SceneEntity>>()
        .iter(world)
        .collect();
    for entity in entities {
        if let Ok(entity) = world.get_entity_mut(entity) {
            entity.despawn();
        }
    }
    let mut stage = world.resource_mut::<Stage>();
    warn!("gfx_viewer: {} closed: {reason}", stage.name);
    // Dropping the handles unloads the scene's meshes and images.
    stage.assets.clear();
    stage.placed = true;
    stage.failed = true;
    stage.error = Some(reason.to_owned());
}

/// Notes when a file of the open scene fails to load, for the overlay.
pub fn watch_loading(server: Res<AssetServer>, mut stage: ResMut<Stage>) {
    if stage.failed {
        return;
    }
    let error = stage.assets.iter().find_map(|asset| {
        match server.recursive_dependency_load_state(asset.id()) {
            bevy::asset::RecursiveDependencyLoadState::Failed(error) => Some(error.to_string()),
            _ => None,
        }
    });
    if let Some(error) = error {
        warn!("gfx_viewer: {} failed to load: {error}", stage.name);
        stage.failed = true;
        stage.error = Some(error);
        // Nothing more will come: the scene is as placed as it gets (the overlay says a file failed).
        stage.placed = true;
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
        Source::Gltf(scene) => {
            let root = world
                .spawn((
                    Name::new(entry.name.clone()),
                    Transform::IDENTITY,
                    Visibility::default(),
                ))
                .id();
            for (file, transform) in &scene.files {
                let handle: Handle<WorldAsset> = world
                    .resource::<AssetServer>()
                    .load(GltfAssetLabel::Scene(0).from_asset(fs_asset_path(file)));
                stage.assets.push(handle.clone().untyped());
                world
                    .spawn((WorldAssetRoot(handle), *transform, ChildOf(root)))
                    .observe(|_: On<WorldInstanceReady>, mut stage: ResMut<Stage>| {
                        stage.instances.0 += 1;
                        stage.instance_ready = stage.instances.0 >= stage.instances.1;
                    });
            }
            stage.instances = (0, scene.files.len());
            stage.fov = 1.0;
            stage.start = Transform::from_xyz(0.0, 1.6, 3.0).looking_at(Vec3::ZERO, Vec3::Y);
            stage.gltf_root = Some(root);
            stage.gltf = Some(scene.clone());
        }
    }
    let after: Vec<Entity> = world
        .query_filtered::<Entity, With<Transform>>()
        .iter(world)
        .collect();
    for entity in after.into_iter().filter(|entity| !before.contains(entity)) {
        world.entity_mut(entity).insert(SceneEntity);
    }

    let gltf = stage.gltf.clone();
    let (start, fov) = (stage.start, stage.fov);
    *world.resource_mut::<Stage>() = stage;
    place_camera(world, start, fov, 1.0);
    let camera = world
        .query_filtered::<Entity, With<ViewerCamera>>()
        .single(world)
        .expect("the viewer has one camera");
    let sky = gltf
        .as_ref()
        .is_some_and(|scene| scene.sun != SunChoice::Off);
    let exposure = match &gltf {
        Some(scene) => scene.exposure.unwrap_or(if sky {
            SUN_EV100
        } else {
            Exposure::default().ev100
        }),
        None => Exposure::default().ev100,
    };
    world.resource_mut::<Stage>().exposure = exposure;
    if sky {
        // The sky: an atmosphere lights and reflects on Raster through its environment map, and Solari traces
        // it on the traced rungs.
        let medium = world
            .resource_mut::<Assets<ScatteringMedium>>()
            .add(ScatteringMedium::earth(256, 256));
        world.spawn((SceneEntity, Atmosphere::earth(medium)));
        world.entity_mut(camera).insert((
            AtmosphereSettings::default(),
            AtmosphereEnvironmentMapLight::default(),
        ));
    } else {
        world
            .entity_mut(camera)
            .remove::<(AtmosphereSettings, AtmosphereEnvironmentMapLight)>();
    }
    world
        .entity_mut(camera)
        .insert(Exposure { ev100: exposure });
}

/// Puts the camera at `start`, and restarts the free camera there (it keeps its own yaw and pitch). It is a cut:
/// every temporal history on the camera drops (Solari's world cache would otherwise keep the light of the last
/// scene when the rung does not change).
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
        CameraCut,
    ));
}

/// Once a glTF scene's instance is spawned and its meshes have bounds: places the camera from the bounds, and
/// gives the scene the default sun when it has no light.
pub fn place_gltf_scene(world: &mut World) {
    let mut stage = world.resource_mut::<Stage>();
    let (Some(root), true, false) = (stage.gltf_root, stage.instance_ready, stage.placed) else {
        return;
    };
    stage.frames_since_ready += 1;
    let waited = stage.frames_since_ready > BOUNDS_WAIT_FRAMES;
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
        // Bounds come a frame after the mesh loads; a mesh that never gets them is left out.
        let Some(aabb) = aabb else {
            if waited {
                continue;
            }
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
        if waited {
            // Nothing to show: say so and stop waiting, the viewer stays usable.
            let mut stage = world.resource_mut::<Stage>();
            warn!("gfx_viewer: {} has no mesh to show", stage.name);
            stage.error = Some("the file has no mesh to show".into());
            stage.placed = true;
        }
        return;
    }

    let mut lights = world.query_filtered::<Entity, Or<(
        With<PointLight>,
        With<SpotLight>,
        With<RectLight>,
        With<DirectionalLight>,
    )>>();
    let own_lights: Vec<Entity> = lights
        .iter(world)
        .collect::<Vec<_>>()
        .into_iter()
        .filter(|light| descends_from(world, &mut parents, *light, root))
        .collect();
    let has_light = !own_lights.is_empty();
    let has_sun = own_lights
        .iter()
        .any(|light| world.get::<DirectionalLight>(*light).is_some());
    let scene = world
        .resource::<Stage>()
        .gltf
        .clone()
        .unwrap_or_else(|| GltfScene {
            files: Vec::new(),
            exposure: None,
            camera: None,
            sun: SunChoice::Auto,
        });
    let sun = match scene.sun {
        SunChoice::Auto if !has_light => Some((default_sun(), default_sun_transform())),
        SunChoice::Sun {
            direction,
            illuminance,
        } if !has_sun => Some(manifest_sun(direction.into(), illuminance)),
        _ => None,
    };
    let gave_sun = sun.is_some();
    if let Some((light, transform)) = sun {
        world.spawn((
            SceneEntity,
            DefaultSun,
            Name::new("viewer sun"),
            light,
            transform,
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
    let start = scene.camera.unwrap_or(start);
    let fov = world.resource::<Stage>().fov;
    {
        let mut stage = world.resource_mut::<Stage>();
        stage.bounds = Some((min, max));
        stage.start = start;
        stage.placed = true;
        stage.default_sun = gave_sun;
    }
    place_camera(world, start, fov, speed);
    info!(
        "gfx_viewer: glTF scene placed: {count} meshes within {min} .. {max}, {}",
        match (has_light, gave_sun) {
            (_, true) => "the viewer added a sun",
            (true, false) => "with its own lights",
            (false, false) => "no light at all (its entry says sun: Off)",
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
