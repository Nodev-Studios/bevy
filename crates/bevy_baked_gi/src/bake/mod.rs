//! The baker: traces the level with `bevy_solari` and writes one irradiance volume per [`GiBakeRegion`].
//!
//! Add [`GiBakePlugin`] to an app that spawns the level (no camera is needed, nor a window). The plugin makes
//! every mesh traceable, swaps the material of [`GiBakeSkipEmission`] entities for one without emission, waits
//! until the level's assets are loaded and Solari's scene holds every light, bakes in short GPU submits, writes
//! the volumes and the manifest to [`GiBakeSettings::output`], and exits the app: code 0 when it wrote a bake,
//! 2 when it refused to.
//!
//! It refuses when nothing is traceable, when the level has no light source at all, and (in full mode) when a
//! light it would record as on reaches no region: those are mistakes, not bakes. A region with no lamp of its own,
//! a bake with every lamp removed, or a furnace lit only by emissive walls are correct bakes.

mod directions;
mod gpu;
mod post;

pub use directions::{
    fibonacci_directions, pcg_hash, sample_direction, voxel_rotation, voxel_seed,
};

use crate::{
    encode_ktx2_rgba16f, hash::mesh_entity_hash, light_power, scene_hash, BakeProvenance,
    BakedGiManifest, BakedLight, BakedLightKind, BakedRegion, GiBakeRegion, GiBakeSkipEmission,
    LightKey, SceneHash, MANIFEST_FORMAT,
};
use alloc::{format, string::String, sync::Arc, vec::Vec};
use bevy_app::{App, AppExit, Last, Plugin, Update};
use bevy_asset::{embedded_asset, AssetId, AssetServer, Assets, Handle};
use bevy_camera::visibility::InheritedVisibility;
use bevy_color::LinearRgba;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    name::Name,
    query::{Has, With, Without},
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Local, Query, ResMut},
    world::World,
};
use bevy_image::Image;
use bevy_light::{DirectionalLight, PointLight, RectLight, SpotLight};
use bevy_math::Vec3;
use bevy_mesh::{Mesh, Mesh3d, PrimitiveTopology};
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    init_gpu_resource,
    renderer::{RenderDevice, RenderGraph, RenderGraphSystems},
    RenderApp, RenderStartup,
};
use bevy_solari::{
    scene::{
        RaytracingLightKind, RaytracingMesh3d, RaytracingSceneBindings, RaytracingScenePlugin,
    },
    SolariPlugins,
};
use bevy_transform::components::GlobalTransform;
use gpu::{BakeExchange, BakeJob, CanonicalKey, GpuVoxel, PATHS_PER_THREAD};
use std::path::PathBuf;
use tracing::{error, info, warn};

/// Bakes the level's regions and exits. See the [module docs](self).
pub struct GiBakePlugin {
    /// What to bake and where to write it.
    pub settings: GiBakeSettings,
}

/// What a bake does.
#[derive(Resource, Clone, Debug)]
pub struct GiBakeSettings {
    /// The folder the volumes and `manifest.gi.ron` are written to. Created if missing.
    pub output: PathBuf,
    /// Paths per voxel: a multiple of 256. to tune: the default is the first count that met gate 4 on the
    /// graphics testbed's `interior` (two seeds 1.80 % apart, RTX 5070, 2026-11-25; 3.57 % at 262 144).
    pub samples: u32,
    /// The seed of the direction rotations and of the paths. Two seeds are two independent bakes.
    pub seed: u32,
    /// What the paths gather.
    pub mode: GiBakeMode,
    /// A voxel more than this share of whose first hits are back faces is inside geometry. to tune: DDGI's rule.
    pub invalid_back_face_share: f32,
}

impl Default for GiBakeSettings {
    fn default() -> Self {
        Self {
            output: PathBuf::from("gi"),
            samples: 1 << 20,
            seed: 0,
            mode: GiBakeMode::Full,
            invalid_back_face_share: 0.25,
        }
    }
}

/// What the paths gather.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GiBakeMode {
    /// Every bounce: the bake a level ships.
    #[default]
    Full,
    /// Only the emission of the first surface each direction hits: a test of units and layout.
    FirstHitEmission,
}

impl GiBakeMode {
    fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::FirstHitEmission => "first-hit-emission",
        }
    }
}

/// What a finished bake measured. Inserted, and logged, before the app exits.
#[derive(Resource, Clone, Debug)]
pub struct GiBakeSummary {
    /// The level's hash, as recorded in the manifest.
    pub scene_hash: SceneHash,
    /// Seconds from the first submit to the last readback.
    pub seconds: f64,
    /// Voxels over all regions.
    pub voxels: usize,
    /// Voxels found inside geometry.
    pub invalid_voxels: u32,
    /// Voxels that could not be filled (a region with no valid voxel).
    pub unfilled_voxels: usize,
    /// Mesh entities traced.
    pub traced_meshes: usize,
    /// Bottom-level acceleration structures: one per distinct traceable mesh.
    pub blas_count: usize,
    /// Analytic lights in Solari's light array.
    pub analytic_lights: usize,
    /// Emissive meshes in Solari's light array.
    pub emissive_meshes: usize,
    /// The share of first rays that hit nothing.
    pub first_ray_miss_share: f64,
    /// GPU submits.
    pub chunks: u32,
    /// The longest submit, in milliseconds (submit to readback).
    pub slowest_chunk_ms: f64,
}

impl Plugin for GiBakePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "bake.wesl");
        if !app.is_plugin_added::<RaytracingScenePlugin>() {
            app.add_plugins(RaytracingScenePlugin);
        }
        app.insert_resource(self.settings.clone())
            .init_resource::<BakeExchange>()
            .init_resource::<BakeProgress>()
            .add_systems(Update, (make_traceable, skip_emission))
            .add_systems(Last, drive_bake);
    }

    fn finish(&self, app: &mut App) {
        let exchange = app.world().resource::<BakeExchange>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        let features = render_app.world().resource::<RenderDevice>().features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            exchange.0.lock().unwrap().outcome = Some(Err(format!(
                "this GPU cannot trace rays for Solari: it lacks {:?}",
                SolariPlugins::required_wgpu_features().difference(features)
            )));
            return;
        }
        render_app
            .insert_resource(exchange)
            .add_systems(
                RenderStartup,
                gpu::init_bake_pipeline.after(init_gpu_resource::<RaytracingSceneBindings>),
            )
            .add_systems(
                RenderGraph,
                gpu::run_bake.in_set(RenderGraphSystems::Finish),
            );
    }
}

/// The material of this entity was swapped for one without emission.
#[derive(Component)]
struct EmissionSkipped;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Stage {
    #[default]
    Waiting,
    Baking,
    Done,
}

/// What the main world tracks between frames.
#[derive(Resource, Default)]
struct BakeProgress {
    stage: Stage,
    settled_frames: u32,
    traced_meshes: usize,
    job: Option<JobRecord>,
}

/// What the main world keeps of the job it sent, to interpret the outcome.
struct JobRecord {
    scene_hash: SceneHash,
    regions: Vec<RegionRecord>,
    analytic: HashMap<Entity, (LightKey, f32)>,
    summary: GiBakeSummary,
}

struct RegionRecord {
    name: String,
    file: String,
    region: GiBakeRegion,
    transform: GlobalTransform,
    resolution: bevy_math::UVec3,
    voxels: core::ops::Range<usize>,
}

/// Main frames the level must stay unchanged, fully loaded, before it is baked. to tune.
const SETTLE_FRAMES: u32 = 30;

/// Solari's test, on the main-world mesh: what it can build an acceleration structure for.
fn is_traceable(mesh: &Mesh) -> bool {
    let Ok(attributes) = mesh.try_attributes() else {
        return false;
    };
    let wanted = [
        Mesh::ATTRIBUTE_POSITION,
        Mesh::ATTRIBUTE_NORMAL,
        Mesh::ATTRIBUTE_UV_0,
        Mesh::ATTRIBUTE_TANGENT,
    ];
    mesh.primitive_topology() == PrimitiveTopology::TriangleList
        && attributes
            .map(|(attribute, _)| (attribute.id, attribute.format))
            .eq(wanted.iter().map(|a| (a.id, a.format)))
        && mesh.try_indices().is_ok()
}

fn make_traceable(
    mut commands: Commands,
    meshes: Query<
        (Entity, &Mesh3d),
        (
            With<MeshMaterial3d<StandardMaterial>>,
            Without<RaytracingMesh3d>,
        ),
    >,
) {
    for (entity, mesh) in &meshes {
        commands
            .entity(entity)
            .insert(RaytracingMesh3d(mesh.0.clone()));
    }
}

fn skip_emission(
    mut commands: Commands,
    marked: Query<
        (Entity, &MeshMaterial3d<StandardMaterial>),
        (With<GiBakeSkipEmission>, Without<EmissionSkipped>),
    >,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut swapped: Local<HashMap<AssetId<StandardMaterial>, Handle<StandardMaterial>>>,
) {
    for (entity, material) in &marked {
        let handle = match swapped.get(&material.0.id()) {
            Some(handle) => handle.clone(),
            None => {
                let Some(original) = materials.get(&material.0) else {
                    continue;
                };
                let dark = StandardMaterial {
                    emissive: LinearRgba::BLACK,
                    emissive_texture: None,
                    ..original.clone()
                };
                let handle = materials.add(dark);
                swapped.insert(material.0.id(), handle.clone());
                handle
            }
        };
        commands
            .entity(entity)
            .insert((MeshMaterial3d(handle), EmissionSkipped));
    }
}

/// Counts the traced meshes once every mesh is traceable-marked and every asset they use has loaded.
fn traced_scene_ready(world: &mut World) -> Option<usize> {
    let mut query = world.query::<(
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
        Has<RaytracingMesh3d>,
        Has<GiBakeSkipEmission>,
        Has<EmissionSkipped>,
    )>();
    let meshes = world.resource::<Assets<Mesh>>();
    let materials = world.resource::<Assets<StandardMaterial>>();
    let images = world.resource::<Assets<Image>>();
    let server = world.resource::<AssetServer>();
    let mut count = 0;
    for (mesh, material, traced, marked, skipped) in query.iter(world) {
        if !traced || (marked && !skipped) || meshes.get(&mesh.0).is_none() {
            return None;
        }
        let material = materials.get(&material.0)?;
        for texture in [
            &material.base_color_texture,
            &material.emissive_texture,
            &material.metallic_roughness_texture,
            &material.normal_map_texture,
        ]
        .into_iter()
        .flatten()
        {
            if images.get(texture).is_none() && !server.load_state(texture).is_failed() {
                return None;
            }
        }
        count += 1;
    }
    Some(count)
}

fn refuse(world: &mut World, reason: &str) {
    error!("baked GI: refusing to bake: {reason}");
    world.resource_mut::<BakeProgress>().stage = Stage::Done;
    world.write_message(AppExit::from_code(2));
}

fn drive_bake(world: &mut World) {
    match world.resource::<BakeProgress>().stage {
        Stage::Waiting => {
            let ready = traced_scene_ready(world);
            let mut progress = world.resource_mut::<BakeProgress>();
            match ready {
                Some(count) if count == progress.traced_meshes => progress.settled_frames += 1,
                Some(count) => {
                    progress.traced_meshes = count;
                    progress.settled_frames = 0;
                }
                None => progress.settled_frames = 0,
            }
            if progress.settled_frames >= SETTLE_FRAMES {
                send_job(world);
            }
        }
        Stage::Baking => {
            let exchange = world.resource::<BakeExchange>().clone();
            let outcome = exchange.0.lock().unwrap().outcome.take();
            if let Some(outcome) = outcome {
                match outcome {
                    Ok(outcome) => finish_bake(world, outcome),
                    Err(reason) => refuse(world, &reason),
                }
            }
        }
        Stage::Done => {}
    }
}

fn file_name(name: &str) -> String {
    let stem: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{stem}.ktx2")
}

fn kind_rank(kind: BakedLightKind) -> u8 {
    match kind {
        BakedLightKind::Directional => 0,
        BakedLightKind::Point => 1,
        BakedLightKind::Spot => 2,
        BakedLightKind::Rect => 3,
    }
}

fn send_job(world: &mut World) {
    let settings = world.resource::<GiBakeSettings>().clone();
    if settings.samples == 0 || !settings.samples.is_multiple_of(PATHS_PER_THREAD) {
        return refuse(
            world,
            &format!("samples must be a positive multiple of {PATHS_PER_THREAD}"),
        );
    }
    let hash = scene_hash(world);

    // Regions, sorted by name so the voxel order (and with it every seed) does not depend on spawn order.
    let mut regions = Vec::new();
    let mut query = world.query::<(&GiBakeRegion, &GlobalTransform, Option<&Name>)>();
    for (region, transform, name) in query.iter(world) {
        let Some(name) = name else {
            return refuse(world, "a GiBakeRegion has no Name");
        };
        regions.push((name.as_str().to_owned(), *region, *transform));
    }
    regions.sort_by(|a, b| a.0.cmp(&b.0));
    if regions.is_empty() {
        return refuse(world, "the level has no GiBakeRegion");
    }
    let mut files = HashSet::new();
    let mut records = Vec::new();
    let mut voxels = Vec::new();
    let mut voxel_region = Vec::new();
    for (index, (name, region, transform)) in regions.into_iter().enumerate() {
        let file = file_name(&name);
        if !files.insert(file.clone()) {
            return refuse(world, &format!("two regions share the file name {file}"));
        }
        let (scale, ..) = transform.to_scale_rotation_translation();
        let resolution = region.resolution(scale);
        let first = voxels.len();
        for z in 0..resolution.z {
            for y in 0..resolution.y {
                for x in 0..resolution.x {
                    // Voxel centres where Bevy's shader samples them: (i + 0.5) / R of the unit cube.
                    let unit = (bevy_math::UVec3::new(x, y, z).as_vec3() + 0.5)
                        / resolution.as_vec3()
                        - 0.5;
                    let position = transform.transform_point(unit);
                    let rotation = voxel_rotation(settings.seed, voxels.len() as u32);
                    voxels.push(GpuVoxel {
                        position: position.extend(1.0).to_array(),
                        rotation: [
                            rotation.x_axis.extend(0.0).to_array(),
                            rotation.y_axis.extend(0.0).to_array(),
                            rotation.z_axis.extend(0.0).to_array(),
                        ],
                    });
                    voxel_region.push(index as u32);
                }
            }
        }
        records.push(RegionRecord {
            name,
            file,
            region,
            transform,
            resolution,
            voxels: first..voxels.len(),
        });
    }

    // Lights: what Solari's light array will hold (visible analytic lights and visible emissive meshes).
    let visible = |visibility: Option<&InheritedVisibility>| visibility.is_none_or(|v| v.get());
    let mut light_keys = HashMap::default();
    let mut analytic = HashMap::default();
    let mut lights = world.query::<(
        Entity,
        &GlobalTransform,
        Option<&InheritedVisibility>,
        Option<&PointLight>,
        Option<&SpotLight>,
        Option<&RectLight>,
        Option<&DirectionalLight>,
    )>();
    for (entity, transform, visibility, point, spot, rect, directional) in lights.iter(world) {
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
        if !visible(visibility) {
            continue;
        }
        let key = LightKey::new(kind, transform);
        light_keys.insert(
            entity,
            CanonicalKey {
                kind: kind_rank(kind),
                position_cm: key.position_cm,
                content: 0,
            },
        );
        analytic.insert(entity, (key, power));
    }
    let mut emissive_meshes = 0;
    let mut blas = HashSet::new();
    let mut traced_meshes = 0;
    let mut untraceable = 0;
    let mut meshes = world.query::<(
        Entity,
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
        &GlobalTransform,
        Option<&InheritedVisibility>,
        Has<GiBakeSkipEmission>,
    )>();
    let mesh_assets = world.resource::<Assets<Mesh>>();
    let materials = world.resource::<Assets<StandardMaterial>>();
    let images = world.resource::<Assets<Image>>();
    for (entity, mesh, material, transform, visibility, skip) in meshes.iter(world) {
        if !visible(visibility) {
            continue;
        }
        if !mesh_assets.get(&mesh.0).is_some_and(is_traceable) {
            untraceable += 1;
            continue;
        }
        traced_meshes += 1;
        blas.insert(mesh.0.id());
        let emissive = materials.get(&material.0).is_some_and(|m| {
            Vec3::new(m.emissive.red, m.emissive.green, m.emissive.blue) != Vec3::ZERO
        });
        if emissive {
            emissive_meshes += 1;
            let cm = (transform.translation() * 100.0).round();
            light_keys.insert(
                entity,
                CanonicalKey {
                    kind: 4,
                    position_cm: [cm.x as i32, cm.y as i32, cm.z as i32],
                    content: mesh_entity_hash(
                        transform,
                        mesh,
                        material,
                        skip,
                        Some(mesh_assets),
                        Some(materials),
                        Some(images),
                    ),
                },
            );
        }
    }
    info!(
        "baked GI: {} regions, {} voxels, {} traced meshes, {} BLAS (one per distinct traceable mesh), {} analytic \
         lights, {} emissive meshes, {} meshes Solari cannot trace (skipped)",
        records.len(),
        voxels.len(),
        traced_meshes,
        blas.len(),
        analytic.len(),
        emissive_meshes,
        untraceable
    );
    if blas.is_empty() {
        return refuse(world, "no mesh can be traced (0 BLAS)");
    }
    if analytic.values().all(|(_, power)| *power <= 0.0) && emissive_meshes == 0 {
        return refuse(
            world,
            "the level has no light source: no lit analytic light and no emissive mesh",
        );
    }

    let directions = fibonacci_directions(settings.samples)
        .into_iter()
        .map(|d| d.extend(0.0).to_array())
        .collect();
    let summary = GiBakeSummary {
        scene_hash: hash,
        seconds: 0.0,
        voxels: voxels.len(),
        invalid_voxels: 0,
        unfilled_voxels: 0,
        traced_meshes,
        blas_count: blas.len(),
        analytic_lights: analytic.len(),
        emissive_meshes,
        first_ray_miss_share: 0.0,
        chunks: 0,
        slowest_chunk_ms: 0.0,
    };
    let job = BakeJob {
        mode: match settings.mode {
            GiBakeMode::Full => 0,
            GiBakeMode::FirstHitEmission => 1,
        },
        samples: settings.samples,
        seed: settings.seed,
        invalid_back_face_share: settings.invalid_back_face_share,
        voxels,
        voxel_region,
        region_count: records.len(),
        directions,
        light_keys,
    };
    world.resource::<BakeExchange>().0.lock().unwrap().job = Some(Arc::new(job));
    let mut progress = world.resource_mut::<BakeProgress>();
    progress.stage = Stage::Baking;
    progress.job = Some(JobRecord {
        scene_hash: hash,
        regions: records,
        analytic,
        summary,
    });
}

fn finish_bake(world: &mut World, outcome: gpu::BakeOutcome) {
    let settings = world.resource::<GiBakeSettings>().clone();
    let record = world.resource_mut::<BakeProgress>().job.take().unwrap();
    let accumulated = &outcome.accumulated;
    if accumulated.rays > 0 && accumulated.misses == accumulated.rays {
        return refuse(
            world,
            "every first ray missed: Solari's scene holds no geometry",
        );
    }

    // The manifest's lights: the analytic lights that were on, in canonical order (sorted by key).
    let mut lights = Vec::new();
    let mut light_columns = Vec::new();
    for (column, (entity, kind)) in outcome.canonical.iter().enumerate() {
        if *kind == RaytracingLightKind::EmissiveMesh {
            continue;
        }
        let Some((key, power)) = record.analytic.get(entity) else {
            continue;
        };
        if *power > 0.0 {
            lights.push(BakedLight {
                key: *key,
                power: *power,
            });
            light_columns.push(column);
        }
    }
    let light_count = outcome.canonical.len();
    if settings.mode == GiBakeMode::Full {
        for (light, column) in lights.iter().zip(&light_columns) {
            let reached: f64 = accumulated.region_tallies.iter().map(|t| t[*column]).sum();
            if reached <= 0.0 {
                return refuse(
                    world,
                    &format!("{:?} is on but its light reaches no region", light.key),
                );
            }
        }
    }

    let mut regions = Vec::new();
    let mut volumes = Vec::new();
    let mut summary = record.summary.clone();
    for (index, region) in record.regions.iter().enumerate() {
        let (volume, invalid, unfilled) = post::region_volume(
            accumulated,
            region.voxels.clone(),
            region.resolution,
            settings.samples,
            settings.invalid_back_face_share,
        );
        if unfilled > 0 {
            warn!(
                "baked GI: region {} has no voxel outside geometry; {unfilled} voxels stay black",
                region.name
            );
        }
        summary.invalid_voxels += invalid;
        summary.unfilled_voxels += unfilled;
        let tallies = &accumulated.region_tallies[index];
        let bounce: f64 = tallies.iter().sum();
        let share = |value: f64| {
            if bounce > 0.0 {
                (value / bounce) as f32
            } else {
                0.0
            }
        };
        let light_weights: Vec<f32> = light_columns.iter().map(|c| share(tallies[*c])).collect();
        let static_weight = if bounce > 0.0 {
            1.0 - light_weights.iter().sum::<f32>()
        } else {
            0.0
        };
        let (scale, rotation, translation) = region.transform.to_scale_rotation_translation();
        regions.push(BakedRegion {
            name: region.name.clone(),
            file: region.file.clone(),
            translation: translation.to_array(),
            rotation: rotation.to_array(),
            scale: scale.to_array(),
            falloff: region.region.falloff.to_array(),
            resolution: region.resolution.to_array(),
            invalid_voxels: invalid,
            bounce,
            static_weight,
            light_weights,
        });
        volumes.push(encode_ktx2_rgba16f(&volume));
    }
    debug_assert_eq!(
        light_count + 1,
        accumulated
            .region_tallies
            .first()
            .map_or(light_count + 1, Vec::len)
    );

    let manifest = BakedGiManifest {
        format: MANIFEST_FORMAT,
        scene_hash: format!("{}", record.scene_hash),
        bake: BakeProvenance {
            baker: format!("bevy_baked_gi {}", env!("CARGO_PKG_VERSION")),
            adapter: outcome.adapter.clone(),
            driver: outcome.driver.clone(),
            mode: settings.mode.name().into(),
            samples: settings.samples,
            seed: settings.seed,
            invalid_back_face_share: settings.invalid_back_face_share,
            texel_format: "Rgba16Float".into(),
        },
        lights,
        regions,
        volumes: Vec::new(),
    };
    let text = match ron::ser::to_string_pretty(&manifest, ron::ser::PrettyConfig::default()) {
        Ok(text) => text,
        Err(error) => return refuse(world, &format!("could not write the manifest: {error}")),
    };
    let written = std::fs::create_dir_all(&settings.output).and_then(|()| {
        for (region, bytes) in manifest.regions.iter().zip(&volumes) {
            std::fs::write(settings.output.join(&region.file), bytes)?;
        }
        std::fs::write(settings.output.join("manifest.gi.ron"), text + "\n")
    });
    if let Err(error) = written {
        return refuse(
            world,
            &format!("could not write {}: {error}", settings.output.display()),
        );
    }

    summary.seconds = outcome.seconds;
    summary.chunks = outcome.chunks;
    summary.slowest_chunk_ms = outcome.slowest_chunk_ms;
    summary.first_ray_miss_share = accumulated.misses as f64 / accumulated.rays.max(1) as f64;
    info!(
        "baked GI: wrote {} regions to {} in {:.2} s ({} submits, slowest {:.1} ms) on {} ({}); {} of {} voxels \
         inside geometry, {:.4} % of first rays missed; scene hash {}",
        manifest.regions.len(),
        settings.output.display(),
        summary.seconds,
        summary.chunks,
        summary.slowest_chunk_ms,
        outcome.adapter,
        outcome.driver,
        summary.invalid_voxels,
        summary.voxels,
        100.0 * summary.first_ray_miss_share,
        record.scene_hash
    );
    world.insert_resource(summary);
    world.resource_mut::<BakeProgress>().stage = Stage::Done;
    world.write_message(AppExit::Success);
}
