//! The testbed's checks of Solari's denoiser contract (`SolariDenoiserInputs`): the `--debug
//! diffuse|specular|hitdist` views, the comparison of the legacy colour with the composed colour in the same frame,
//! and the `contract` scene, whose every surface is also known here as an analytic shape so that what the lighting
//! pass wrote can be compared with what it should have written.
//!
//! `--contract-check` reads back, on one frame in the middle of the warm-up, the lit colour the lighting pass left
//! in the view target (the legacy colour), the colour compose wrote over it, the two radiance textures, the view
//! depth and the guides. It checks them and writes the result to the `contract` section of `report.json`; a clause
//! that fails is a failure of the run. The readback waits for the GPU, which is why it is not in a measured frame.

use crate::{drive, finish, ActiveScene, Args, Flags, Mode, Phase, Report, Run, Scene};
use bevy::{
    anti_alias::ray_reconstruction::{
        DenoiserSystems, ViewDenoiserGuideTextures, ViewDenoiserInputTextures,
        DENOISER_MISS_HIT_DISTANCE, DENOISER_SKY_VIEW_DEPTH,
    },
    core_pipeline::{
        core_3d::main_opaque_pass_3d,
        schedule::{Core3d, Core3dSystems},
        tonemapping::{DebandDither, Tonemapping},
    },
    prelude::*,
    render::{
        render_resource::{
            binding_types::{texture_2d, texture_storage_2d},
            BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, MapMode, PipelineCache, PollType, ShaderStages,
            StorageTextureAccess, TexelCopyBufferInfo, TexelCopyBufferLayout, Texture,
            TextureFormat, TextureSampleType,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        view::ViewTarget,
        ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
    },
    shader::Shader,
    solari::realtime::SolariDenoiserInputs,
};
use core::f32::consts::TAU;
use serde_json::{json, Map, Value};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};

/// The usage text of this file's flags, printed by `--help` after the raw flags.
pub const HELP: &str = "\
  --contract                     solari: write Solari's denoiser contract (SolariDenoiserInputs) and compose the
                                 frame from it
  --contract-check               solari: read back the legacy and composed colour and the contract textures of one
                                 warm-up frame, check them, and write a `contract` section in report.json; a clause
                                 that fails fails the run (implies --contract). No --taa or DLSS: a jittered frame
                                 does not match the scene's analytic shapes pixel for pixel
  --debug diffuse|specular|hitdist
                                 solari: show a contract texture in place of the frame, with no tonemapping
                                 (implies --contract): radiance as x / (1 + x); hit distance as 1 - exp(-d / 5 m)
                                 in red for the diffuse lobe and green for the specular lobe, blue where the ray
                                 left the scene
  --scene contract               a closed room of analytic shapes for --contract-check: a metallic-1 sphere, a matte
                                 back wall, a smooth dielectric floor and a flat mirror
";

/// Which contract texture `--debug` shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DebugView {
    /// Demodulated diffuse radiance.
    Diffuse,
    /// Demodulated specular radiance.
    Specular,
    /// Both hit distances.
    HitDistance,
}

impl DebugView {
    fn entry_point(self) -> &'static str {
        match self {
            DebugView::Diffuse => "diffuse_view",
            DebugView::Specular => "specular_view",
            DebugView::HitDistance => "hitdist_view",
        }
    }
}

/// This file's flags.
#[derive(Resource, Clone)]
pub struct Contract {
    /// The camera gets `SolariDenoiserInputs`.
    pub inputs: bool,
    /// `--contract-check`.
    pub check: bool,
    /// `--debug`.
    pub debug: Option<DebugView>,
}

impl Contract {
    /// Takes this file's flags off the command line. Flags nobody takes are a usage error.
    pub fn parse(flags: &mut Flags) -> Self {
        let contract = flags.switch("--contract");
        let check = flags.switch("--contract-check");
        let debug = flags.choice(
            "--debug",
            &[
                ("diffuse", DebugView::Diffuse),
                ("specular", DebugView::Specular),
                ("hitdist", DebugView::HitDistance),
            ],
        );
        Contract {
            inputs: contract || check || debug.is_some(),
            check,
            debug,
        }
    }

    /// Adds this file's plugins and systems, after the default plugins and Solari's.
    pub fn build(&self, app: &mut App, args: &Args) {
        if !self.inputs {
            return;
        }
        // `Args::parse` runs before this file's flags are known, so the combinations are refused here: the run
        // stops before it draws, with the reason in its report.
        let mut refusals = Vec::new();
        if args.mode != Mode::Solari {
            refusals.push("--contract, --contract-check and --debug need --mode solari");
        }
        if (self.check || self.debug.is_some()) && (args.taa || args.dlss_rr || args.dlss_sr) {
            refusals.push(
                "--contract-check and --debug read the render-resolution textures of an unjittered frame: \
                 no --taa, --dlss-rr or --dlss-sr",
            );
        }
        if !refusals.is_empty() {
            let world = app.world_mut();
            let mut report = world.resource_mut::<Report>();
            report
                .failures
                .extend(refusals.into_iter().map(String::from));
            world.resource_mut::<Run>().phase = Phase::Finishing;
            return;
        }

        if self.check {
            let shared = Arc::new(Shared::default());
            app.insert_resource(CheckState {
                shared: shared.clone(),
                capture_frame: args.warmup / 2,
                requested: false,
            })
            .add_systems(
                Update,
                (request_readback, collect_readback)
                    .chain()
                    .after(drive)
                    .before(finish),
            );
            app.sub_app_mut(RenderApp)
                .insert_resource(Readback {
                    shared,
                    active: false,
                    mapping: false,
                    mapped: Arc::new(AtomicUsize::new(0)),
                    copies: Vec::new(),
                })
                .add_systems(ExtractSchedule, start_readback)
                .add_systems(
                    Core3d,
                    (
                        copy_legacy.in_set(DenoiserSystems::Denoise),
                        copy_composed
                            .in_set(ContractSystems::Capture)
                            .after(DenoiserSystems::Compose)
                            .before(main_opaque_pass_3d)
                            .in_set(Core3dSystems::MainPass),
                    ),
                )
                .add_systems(Render, finish_readback.in_set(RenderSystems::Cleanup));
        }

        if let Some(view) = self.debug {
            let shader = app
                .world_mut()
                .resource_mut::<Assets<Shader>>()
                .add(Shader::from_wgsl(
                    DEBUG_VIEW_SHADER,
                    "gfx_testbed/contract_debug_view.wgsl",
                ));
            app.sub_app_mut(RenderApp)
                .insert_resource(DebugViewShader(shader, view))
                .add_systems(RenderStartup, init_debug_view_pipeline)
                .add_systems(
                    Core3d,
                    show_debug_view
                        .after(ContractSystems::Capture)
                        .after(DenoiserSystems::Compose)
                        .before(main_opaque_pass_3d)
                        .in_set(Core3dSystems::MainPass),
                );
        }
    }

    /// Spawns the scene called `name` if this file owns it.
    pub fn scene(name: &str, world: &mut World) -> Option<Scene> {
        (name == "contract").then(|| {
            world
                .run_system_cached(contract_scene)
                .expect("the contract scene takes only resources the default plugins provide")
        })
    }

    /// Adds this file's components to the camera, after the raw flags or the ladder gave it theirs.
    pub fn camera(&self, args: &Args, camera: &mut EntityWorldMut) {
        if !self.inputs || args.mode != Mode::Solari {
            return;
        }
        camera.insert(SolariDenoiserInputs);
        if self.debug.is_some() {
            // The view shows the texture's values themselves, so nothing may curve or dither them.
            camera.insert((Tonemapping::None, DebandDither::Disabled));
        }
    }
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ContractSystems {
    /// Reads the view target after compose, before anything else writes it.
    Capture,
}

// The contract scene. Every surface is an axis-aligned box or a sphere, spawned from the same numbers the
// check traces against.

/// The inside of the room: x from -3 to 3, y (up) from 0 to 3, z from -4 (the matte wall) to 3.
const ROOM_MIN: Vec3 = Vec3::new(-3.0, 0.0, -4.0);
const ROOM_MAX: Vec3 = Vec3::new(3.0, 3.0, 3.0);
/// Thickness of the boxes the room is made of.
const SLAB: f32 = 0.1;
const SPHERE_CENTER: Vec3 = Vec3::new(1.0, 0.6, -1.4);
const SPHERE_RADIUS: f32 = 0.6;
/// The mirror stands 2 cm proud of the left wall: its reflecting face is at x = -2.98.
const MIRROR_MIN: Vec3 = Vec3::new(-3.0, 0.4, -3.3);
const MIRROR_MAX: Vec3 = Vec3::new(-2.98, 2.4, -0.6);
const CONTRACT_FOV: f32 = 1.0;
const CONTRACT_ORBIT_FRAMES: f32 = 720.0;

/// What a surface of the contract scene is, for the check.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Surface {
    /// Smooth dielectric (roughness 0): its specular lobe is a delta, so its specular hit distance is known.
    Floor,
    /// The matte back wall (roughness 1, dielectric).
    MatteWall,
    /// The other walls and the ceiling.
    Wall,
    /// Metallic 1, roughness 0.3: no diffuse lobe at all.
    Sphere,
    /// Metallic 1, roughness 0: a flat perfect mirror.
    Mirror,
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    /// An axis-aligned box of these half extents around the entity's translation.
    Box(Vec3),
    /// A sphere of this radius around the entity's translation.
    Sphere(f32),
}

/// An entity of the contract scene, with the analytic shape the check traces.
#[derive(Component, Clone, Copy, Debug)]
pub struct ContractObject {
    surface: Surface,
    shape: Shape,
}

fn contract_orbit(frame: u32) -> Transform {
    let angle = frame as f32 * (TAU / CONTRACT_ORBIT_FRAMES);
    Transform::from_xyz(
        0.8 * ops::sin(angle),
        1.6,
        2.4 + 0.3 * (ops::cos(angle) - 1.0),
    )
    .looking_at(Vec3::new(0.0, 0.9, -2.0), Vec3::Y)
}

fn contract_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) -> Scene {
    let mut material = |color: Color, metallic: f32, roughness: f32| {
        materials.add(StandardMaterial {
            base_color: color,
            metallic,
            perceptual_roughness: roughness,
            ..default()
        })
    };
    let floor = material(Color::srgb(0.25, 0.25, 0.25), 0.0, 0.0);
    let matte = material(Color::srgb(0.8, 0.8, 0.8), 0.0, 1.0);
    let wall = material(Color::srgb(0.6, 0.55, 0.5), 0.0, 0.9);
    let gold = material(Color::srgb(1.0, 0.78, 0.34), 1.0, 0.3);
    let mirror = material(Color::srgb(0.95, 0.95, 0.95), 1.0, 0.0);

    let mut spawn_box =
        |surface: Surface, min: Vec3, max: Vec3, material: &Handle<StandardMaterial>| {
            let half = (max - min) / 2.0;
            let mesh = Cuboid::from_size(max - min)
                .mesh()
                .build()
                .with_generated_tangents()
                .expect("a cuboid mesh has what tangent generation needs");
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material.clone()),
                Transform::from_translation(min + half),
                ContractObject {
                    surface,
                    shape: Shape::Box(half),
                },
            ));
        };
    let (lo, hi, s) = (ROOM_MIN, ROOM_MAX, SLAB);
    let outer_lo = lo - Vec3::splat(s);
    let outer_hi = hi + Vec3::splat(s);
    spawn_box(
        Surface::Floor,
        Vec3::new(outer_lo.x, lo.y - s, outer_lo.z),
        Vec3::new(outer_hi.x, lo.y, outer_hi.z),
        &floor,
    );
    spawn_box(
        Surface::Wall,
        Vec3::new(outer_lo.x, hi.y, outer_lo.z),
        Vec3::new(outer_hi.x, hi.y + s, outer_hi.z),
        &wall,
    );
    spawn_box(
        Surface::MatteWall,
        Vec3::new(lo.x, lo.y, lo.z - s),
        Vec3::new(hi.x, hi.y, lo.z),
        &matte,
    );
    spawn_box(
        Surface::Wall,
        Vec3::new(lo.x, lo.y, hi.z),
        Vec3::new(hi.x, hi.y, hi.z + s),
        &wall,
    );
    spawn_box(
        Surface::Wall,
        Vec3::new(lo.x - s, lo.y, outer_lo.z),
        Vec3::new(lo.x, hi.y, outer_hi.z),
        &wall,
    );
    spawn_box(
        Surface::Wall,
        Vec3::new(hi.x, lo.y, outer_lo.z),
        Vec3::new(hi.x + s, hi.y, outer_hi.z),
        &wall,
    );
    spawn_box(Surface::Mirror, MIRROR_MIN, MIRROR_MAX, &mirror);

    let sphere = Sphere::new(SPHERE_RADIUS)
        .mesh()
        .uv(128, 64)
        .with_generated_tangents()
        .expect("a sphere mesh has what tangent generation needs");
    commands.spawn((
        Mesh3d(meshes.add(sphere)),
        MeshMaterial3d(gold),
        Transform::from_translation(SPHERE_CENTER),
        ContractObject {
            surface: Surface::Sphere,
            shape: Shape::Sphere(SPHERE_RADIUS),
        },
    ));

    for (position, intensity) in [
        (Vec3::new(-0.5, 2.6, -1.0), 300_000.0),
        (Vec3::new(2.0, 2.4, 1.5), 150_000.0),
    ] {
        commands.spawn((
            PointLight {
                intensity,
                range: 15.0,
                shadow_maps_enabled: true,
                ..default()
            },
            Transform::from_translation(position),
        ));
    }

    Scene {
        orbit: contract_orbit,
        fov: CONTRACT_FOV,
        assets: Vec::new(),
    }
}

// The readback.

/// What the main world and the render world share about the readback.
#[derive(Default)]
struct Shared {
    /// Set by the main world on the capture frame; taken by the render world when it extracts that frame.
    request: AtomicBool,
    /// The textures read back, or why there are none.
    result: Mutex<Option<Result<Vec<Plane>, String>>>,
}

#[derive(Resource)]
struct CheckState {
    shared: Arc<Shared>,
    capture_frame: u32,
    requested: bool,
}

/// Placed in the report's failures when the readback is asked for, and taken out when it arrives, so that a run
/// whose readback never came back cannot pass.
const READBACK_PENDING: &str =
    "--contract-check: the readback of the contract textures never arrived";

fn request_readback(mut state: ResMut<CheckState>, run: Res<Run>, mut report: ResMut<Report>) {
    if !state.requested && run.phase == Phase::Running && run.frame == state.capture_frame {
        state.requested = true;
        state.shared.request.store(true, Ordering::SeqCst);
        report.failures.push(READBACK_PENDING.into());
    }
}

fn collect_readback(
    state: Res<CheckState>,
    run: Res<Run>,
    args: Res<Args>,
    scene: Option<Res<ActiveScene>>,
    objects: Query<(&ContractObject, &GlobalTransform)>,
    mut report: ResMut<Report>,
) {
    let Some(result) = state.shared.result.lock().unwrap().take() else {
        return;
    };
    report
        .failures
        .retain(|failure| failure != READBACK_PENDING);
    let planes = match result {
        Ok(planes) => planes,
        Err(error) => {
            report.failures.push(format!("--contract-check: {error}"));
            return;
        }
    };

    let objects: Vec<(ContractObject, Vec3)> = objects
        .iter()
        .map(|(object, transform)| (*object, transform.translation()))
        .collect();
    let geometry = match (&scene, args.orbit, objects.is_empty()) {
        (Some(scene), false, false) => Some(Geometry {
            camera: (scene.0.orbit)(args.pose),
            fov: scene.0.fov,
            objects,
        }),
        _ => None,
    };
    let (section, failures) = check(&planes, geometry.as_ref(), state.capture_frame);
    report.sections.insert("contract".into(), section);
    report.failures.extend(failures);
    info!(
        "--contract-check ran on frame {} of {:?}",
        state.capture_frame, run.phase
    );
}

struct Copy {
    name: &'static str,
    format: TextureFormat,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    buffer: Buffer,
}

#[derive(Resource)]
struct Readback {
    shared: Arc<Shared>,
    /// This frame is the capture frame.
    active: bool,
    mapping: bool,
    mapped: Arc<AtomicUsize>,
    copies: Vec<Copy>,
}

fn start_readback(mut readback: ResMut<Readback>) {
    if readback.shared.request.swap(false, Ordering::SeqCst) {
        readback.active = true;
    }
}

fn copy_texture(
    readback: &mut Readback,
    device: &RenderDevice,
    ctx: &mut RenderContext,
    name: &'static str,
    texture: &Texture,
) {
    let format = texture.format();
    let bytes_per_pixel = format
        .block_copy_size(None)
        .expect("the contract's textures are uncompressed colour formats");
    let bytes_per_row = (texture.width() * bytes_per_pixel).next_multiple_of(256);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some(name),
        size: u64::from(bytes_per_row * texture.height()),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    ctx.command_encoder().copy_texture_to_buffer(
        texture.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    readback.copies.push(Copy {
        name,
        format,
        width: texture.width(),
        height: texture.height(),
        bytes_per_row,
        buffer,
    });
}

fn copy_legacy(
    view: ViewQuery<&ViewTarget, With<ViewDenoiserInputTextures>>,
    mut readback: ResMut<Readback>,
    device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    if !readback.active || !readback.copies.is_empty() {
        return;
    }
    copy_texture(
        &mut readback,
        &device,
        &mut ctx,
        "legacy",
        view.into_inner().main_texture(),
    );
}

fn copy_composed(
    view: ViewQuery<(
        &ViewTarget,
        &ViewDenoiserInputTextures,
        &ViewDenoiserGuideTextures,
    )>,
    mut readback: ResMut<Readback>,
    device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    // Only after the legacy colour of the same frame was copied.
    if !readback.active || readback.copies.len() != 1 {
        return;
    }
    let (target, inputs, guides) = view.into_inner();
    for (name, texture) in [
        ("composed", target.main_texture()),
        ("diffuse", &inputs.diffuse_radiance_hit_distance.texture),
        ("specular", &inputs.specular_radiance_hit_distance.texture),
        ("view_depth", &inputs.view_depth.texture),
        ("normal_roughness", &guides.normal_roughness.texture),
    ] {
        copy_texture(&mut readback, &device, &mut ctx, name, texture);
    }
}

/// Number of textures one readback copies.
const COPIES: usize = 6;

/// Maps the copies once the frame that made them is submitted, and waits for them.
fn finish_readback(mut readback: ResMut<Readback>, device: Res<RenderDevice>) {
    if !readback.active {
        return;
    }
    if readback.copies.len() != COPIES {
        *readback.shared.result.lock().unwrap() = Some(Err(format!(
            "the capture frame copied {} textures instead of {COPIES}: the view had no contract textures",
            readback.copies.len()
        )));
        readback.active = false;
        readback.copies.clear();
        return;
    }
    if !readback.mapping {
        readback.mapping = true;
        for copy in &readback.copies {
            let mapped = readback.mapped.clone();
            copy.buffer
                .slice(..)
                .map_async(MapMode::Read, move |result| {
                    if result.is_ok() {
                        mapped.fetch_add(1, Ordering::SeqCst);
                    }
                });
        }
    }
    // The frame's commands are submitted by now; waiting keeps the copies from lingering into measured frames.
    let _ = device.poll(PollType::wait_indefinitely());
    if readback.mapped.load(Ordering::SeqCst) != COPIES {
        return;
    }
    let planes = readback.copies.iter().map(Plane::read).collect();
    *readback.shared.result.lock().unwrap() = Some(Ok(planes));
    readback.active = false;
    readback.copies.clear();
}

/// A texture read back, as `f32` channels.
struct Plane {
    name: &'static str,
    width: u32,
    height: u32,
    channels: usize,
    data: Vec<f32>,
}

impl Plane {
    fn read(copy: &Copy) -> Self {
        let bytes = copy
            .buffer
            .slice(..)
            .get_mapped_range()
            .expect("the buffer was mapped");
        let channels = match copy.format {
            TextureFormat::Rgba16Float => 4,
            TextureFormat::R32Float => 1,
            other => unreachable!("the contract's textures have no {other:?}"),
        };
        let mut data = Vec::with_capacity((copy.width * copy.height) as usize * channels);
        for y in 0..copy.height {
            let row = &bytes[(y * copy.bytes_per_row) as usize..];
            for i in 0..copy.width as usize * channels {
                data.push(match copy.format {
                    TextureFormat::Rgba16Float => {
                        half_to_f32(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]))
                    }
                    _ => f32::from_le_bytes([
                        row[i * 4],
                        row[i * 4 + 1],
                        row[i * 4 + 2],
                        row[i * 4 + 3],
                    ]),
                });
            }
        }
        Plane {
            name: copy.name,
            width: copy.width,
            height: copy.height,
            channels,
            data,
        }
    }

    fn texel(&self, pixel: usize) -> &[f32] {
        &self.data[pixel * self.channels..(pixel + 1) * self.channels]
    }
}

/// Every 16-bit float is exactly a 32-bit float: the sign, exponent and mantissa move to their wider fields.
fn half_to_f32(half: u16) -> f32 {
    let sign = u32::from(half & 0x8000) << 16;
    let exponent = u32::from((half >> 10) & 0x1f);
    let mantissa = u32::from(half & 0x3ff);
    match exponent {
        // Subnormal: mantissa times 2^-24.
        0 => f32::from_bits(sign | 0x3f80_0000) * mantissa as f32 * f32::from_bits(0x3380_0000),
        // Infinity or NaN.
        31 => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}

// The check.

/// The scene as the check traces it.
struct Geometry {
    camera: Transform,
    fov: f32,
    objects: Vec<(ContractObject, Vec3)>,
}

/// A ray's first hit on the analytic scene.
#[derive(Clone, Copy)]
struct Hit {
    object: usize,
    distance: f32,
    normal: Vec3,
}

impl Geometry {
    fn trace(&self, origin: Vec3, direction: Vec3) -> Option<Hit> {
        let mut nearest: Option<Hit> = None;
        for (index, (object, center)) in self.objects.iter().enumerate() {
            let hit = match object.shape {
                Shape::Box(half) => ray_box(origin, direction, *center - half, *center + half),
                Shape::Sphere(radius) => ray_sphere(origin, direction, *center, radius),
            };
            if let Some((distance, normal)) = hit
                && nearest.is_none_or(|nearest| distance < nearest.distance)
            {
                nearest = Some(Hit {
                    object: index,
                    distance,
                    normal,
                });
            }
        }
        nearest
    }

    /// The camera ray through a point of the image, `pixel` in pixels from the top-left corner.
    fn camera_ray(&self, pixel: Vec2, size: Vec2) -> (Vec3, Vec3) {
        let ndc = Vec2::new(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0);
        let half_height = ops::tan(self.fov / 2.0);
        let view = Vec3::new(
            ndc.x * half_height * size.x / size.y,
            ndc.y * half_height,
            -1.0,
        );
        (
            self.camera.translation,
            (self.camera.rotation * view).normalize(),
        )
    }
}

/// Where a ray from outside enters the box, ignoring hits closer than 1 mm.
fn ray_box(origin: Vec3, direction: Vec3, min: Vec3, max: Vec3) -> Option<(f32, Vec3)> {
    let inverse = direction.recip();
    let t0 = (min - origin) * inverse;
    let t1 = (max - origin) * inverse;
    let near = t0.min(t1);
    let far = t0.max(t1);
    let entry = near.max_element();
    if entry > far.min_element() || entry < 1e-3 {
        return None;
    }
    let axis = if entry == near.x {
        Vec3::X
    } else if entry == near.y {
        Vec3::Y
    } else {
        Vec3::Z
    };
    Some((entry, -axis * direction.dot(axis).signum()))
}

fn ray_sphere(origin: Vec3, direction: Vec3, center: Vec3, radius: f32) -> Option<(f32, Vec3)> {
    let offset = origin - center;
    let b = offset.dot(direction);
    let discriminant = b * b - (offset.length_squared() - radius * radius);
    if discriminant < 0.0 {
        return None;
    }
    let distance = -b - discriminant.sqrt();
    (distance >= 1e-3).then(|| (distance, (origin + direction * distance - center) / radius))
}

/// A clause: whether it passed, and what it counted.
fn clause(pass: bool, mut counts: Map<String, Value>) -> Value {
    counts.insert("pass".into(), json!(pass));
    Value::Object(counts)
}

/// Relative error tolerated between a hit distance read back and the traced one (plan section 5, row B1).
const HIT_DISTANCE_TOLERANCE: f32 = 0.05;
/// Relative error tolerated between the view depth read back and the traced one: the sphere is a mesh of 128 by 64
/// segments, whose faces are within 0.03 % of its radius from the analytic sphere.
const VIEW_DEPTH_TOLERANCE: f32 = 0.01;
/// Fewer checked pixels than this make a clause about a surface meaningless.
const MIN_PIXELS: usize = 1000;

fn check(planes: &[Plane], geometry: Option<&Geometry>, frame: u32) -> (Value, Vec<String>) {
    let plane = |name: &str| {
        planes
            .iter()
            .find(|plane| plane.name == name)
            .expect("every texture of the readback is there")
    };
    let (legacy, composed) = (plane("legacy"), plane("composed"));
    let (diffuse, specular) = (plane("diffuse"), plane("specular"));
    let view_depth = plane("view_depth");
    let (width, height) = (legacy.width, legacy.height);
    let pixels = (width * height) as usize;

    let mut clauses = Map::new();

    // Plan section 5, row B1: every pixel within 0.01 * max(a, b) + 1e-3, in each colour channel.
    let (mut outside, mut worst_excess, mut worst_pixel, mut max_difference) =
        (0usize, 0f32, None, 0f32);
    for pixel in 0..pixels {
        for channel in 0..3 {
            let (a, b) = (legacy.texel(pixel)[channel], composed.texel(pixel)[channel]);
            let difference = (a - b).abs();
            let excess = difference - (0.01 * a.max(b) + 1e-3);
            max_difference = max_difference.max(difference);
            // A NaN on either side counts as outside.
            if excess.is_nan() || excess > 0.0 {
                outside += 1;
                if excess.is_nan() || excess > worst_excess {
                    worst_excess = excess;
                    worst_pixel = Some([pixel as u32 % width, pixel as u32 / width]);
                }
            }
        }
    }
    clauses.insert(
        "legacy_vs_composed".into(),
        clause(
            outside == 0,
            json_map(json!({
                "channels": pixels * 3,
                "channels_outside_tolerance": outside,
                "max_abs_difference": max_difference,
                "worst_excess": worst_excess,
                "worst_pixel": worst_pixel,
            })),
        ),
    );

    // No NaN, no infinity, nothing negative; at most one lobe traced per pixel; nothing shaded where there is no
    // geometry.
    let (mut non_finite, mut negative, mut both_lobes, mut beyond_miss, mut sky_with_values) =
        (0, 0, 0, 0, 0);
    for pixel in 0..pixels {
        let (d, s, z) = (
            diffuse.texel(pixel),
            specular.texel(pixel),
            view_depth.texel(pixel)[0],
        );
        non_finite += d
            .iter()
            .chain(s)
            .chain([&z])
            .filter(|v| !v.is_finite())
            .count();
        negative += d.iter().chain(s).chain([&z]).filter(|v| **v < 0.0).count();
        if d[3] > 0.0 && s[3] > 0.0 {
            both_lobes += 1;
        }
        if d[3] > DENOISER_MISS_HIT_DISTANCE || s[3] > DENOISER_MISS_HIT_DISTANCE {
            beyond_miss += 1;
        }
        if z >= DENOISER_SKY_VIEW_DEPTH && d.iter().chain(s).any(|v| *v != 0.0) {
            sky_with_values += 1;
        }
    }
    clauses.insert(
        "finite".into(),
        clause(
            non_finite == 0 && negative == 0,
            json_map(json!({ "non_finite_values": non_finite, "negative_values": negative })),
        ),
    );
    clauses.insert(
        "one_lobe_per_pixel".into(),
        clause(
            both_lobes == 0 && beyond_miss == 0 && sky_with_values == 0,
            json_map(json!({
                "pixels_with_both_hit_distances": both_lobes,
                "pixels_beyond_miss_distance": beyond_miss,
                "sky_pixels_with_values": sky_with_values,
            })),
        ),
    );

    if let Some(geometry) = geometry {
        check_geometry(geometry, planes, &mut clauses);
    } else {
        clauses.insert(
            "geometry".into(),
            json!({ "evaluated": false, "why": "only the contract scene with a static camera has known shapes" }),
        );
    }

    let failures = clauses
        .iter()
        .filter(|(_, value)| value.get("pass") == Some(&Value::Bool(false)))
        .map(|(name, value)| format!("--contract-check: {name} failed: {value}"))
        .collect();
    (
        json!({ "frame": frame, "size": [width, height], "clauses": clauses }),
        failures,
    )
}

fn json_map(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => unreachable!("built from an object literal"),
    }
}

/// Per surface: how many pixels were checked, and how many broke the clause.
#[derive(Default)]
struct Tally {
    checked: usize,
    failed: usize,
    worst: Option<Value>,
    /// The smallest and largest expected value checked.
    range: Option<(f32, f32)>,
}

impl Tally {
    fn record(&mut self, ok: bool, detail: impl FnOnce() -> Value) {
        self.checked += 1;
        if !ok {
            self.failed += 1;
            if self.worst.is_none() {
                self.worst = Some(detail());
            }
        }
    }

    fn expect(&mut self, value: f32) {
        let (min, max) = self.range.get_or_insert((value, value));
        *min = min.min(value);
        *max = max.max(value);
    }

    fn json(&self) -> Value {
        json!({
            "checked": self.checked,
            "failed": self.failed,
            "expected_range": self.range.map(|(min, max)| [min, max]),
            "first_failure": self.worst,
        })
    }
}

fn check_geometry(geometry: &Geometry, planes: &[Plane], clauses: &mut Map<String, Value>) {
    let plane = |name: &str| {
        planes
            .iter()
            .find(|plane| plane.name == name)
            .expect("every texture of the readback is there")
    };
    let (diffuse, specular) = (plane("diffuse"), plane("specular"));
    let (view_depth, normal_roughness) = (plane("view_depth"), plane("normal_roughness"));
    let (width, height) = (diffuse.width, diffuse.height);
    let size = Vec2::new(width as f32, height as f32);
    let forward = geometry.camera.forward().as_vec3();
    let room_diagonal = (ROOM_MAX - ROOM_MIN).length();

    let mut metal_diffuse = Tally::default();
    let mut floor_specular = Tally::default();
    let mut mirror_specular = Tally::default();
    let mut mirror_guides = Tally::default();
    let mut depth = Tally::default();
    let mut matte_one_lobe = Tally::default();
    let mut matte_diffuse_sampled = 0usize;
    let mut diffuse_range = Tally::default();

    for y in 0..height {
        for x in 0..width {
            let pixel = (y * width + x) as usize;
            let center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let (origin, direction) = geometry.camera_ray(center, size);
            let Some(hit) = geometry.trace(origin, direction) else {
                continue;
            };
            let surface = geometry.objects[hit.object].0.surface;
            let (d, s) = (diffuse.texel(pixel), specular.texel(pixel));

            // Every diffuse hit distance is a ray inside the closed room.
            diffuse_range.record(
                d[3] == 0.0 || (d[3] > 0.0 && d[3] <= room_diagonal),
                || json!({ "pixel": [x, y], "diffuse_hit_distance": d[3] }),
            );

            // The surface masks, eroded by 2 pixels: silhouettes and the sphere's facets do not count.
            let interior = [(-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0)]
                .iter()
                .all(|(dx, dy)| {
                    let (o, dir) = geometry.camera_ray(center + Vec2::new(*dx, *dy), size);
                    geometry.trace(o, dir).map(|h| h.object) == Some(hit.object)
                });
            if !interior {
                continue;
            }

            let expected_depth = hit.distance * direction.dot(forward);
            let measured_depth = view_depth.texel(pixel)[0];
            depth.expect(expected_depth);
            depth.record(
                (measured_depth - expected_depth).abs() <= VIEW_DEPTH_TOLERANCE * expected_depth,
                || json!({ "pixel": [x, y], "surface": format!("{surface:?}"), "view_depth": measured_depth, "traced": expected_depth }),
            );

            match surface {
                Surface::Sphere | Surface::Mirror => {
                    metal_diffuse.record(d.iter().all(|v| *v == 0.0), || {
                        json!({ "pixel": [x, y], "surface": format!("{surface:?}"), "diffuse": d })
                    });
                }
                Surface::MatteWall => {
                    matte_one_lobe.record((d[3] > 0.0) != (s[3] > 0.0), || {
                        json!({ "pixel": [x, y], "diffuse_hit_distance": d[3], "specular_hit_distance": s[3] })
                    });
                    if d[3] > 0.0 {
                        matte_diffuse_sampled += 1;
                    }
                }
                _ => {}
            }

            // Where the specular lobe is a delta, the specular ray is the reflected view ray, and its length is
            // known. Pixels whose reflection crosses an edge within a pixel are left out.
            if matches!(surface, Surface::Floor | Surface::Mirror) {
                let point = origin + direction * hit.distance;
                let reflect = |dir: Vec3| dir - 2.0 * dir.dot(hit.normal) * hit.normal;
                let reflected = geometry.trace(point, reflect(direction));
                let stable = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
                    .iter()
                    .all(|(dx, dy)| {
                        let (o, dir) = geometry.camera_ray(center + Vec2::new(*dx, *dy), size);
                        let Some(near) = geometry.trace(o, dir) else {
                            return false;
                        };
                        let neighbour = geometry.trace(o + dir * near.distance, reflect(dir));
                        neighbour.map(|h| h.object) == reflected.map(|h| h.object)
                    });
                if surface == Surface::Mirror {
                    let n = normal_roughness.texel(pixel);
                    mirror_guides.record(
                        Vec3::new(n[0], n[1], n[2]).dot(hit.normal) >= 0.999 && n[3] <= 1e-3,
                        || json!({ "pixel": [x, y], "normal_roughness": n, "mirror_normal": hit.normal }),
                    );
                }
                let (Some(reflected), true) = (reflected, stable) else {
                    continue;
                };
                let tally = if surface == Surface::Floor {
                    &mut floor_specular
                } else {
                    &mut mirror_specular
                };
                // The floor's ray took its diffuse lobe on about half the pixels; the mirror has no other lobe.
                if s[3] == 0.0 && surface == Surface::Floor {
                    continue;
                }
                tally.expect(reflected.distance);
                tally.record(
                    (s[3] - reflected.distance).abs() <= HIT_DISTANCE_TOLERANCE * reflected.distance,
                    || json!({ "pixel": [x, y], "specular_hit_distance": s[3], "traced": reflected.distance }),
                );
            }
        }
    }

    let enough = |tally: &Tally| tally.checked >= MIN_PIXELS;
    let mut insert = |name: &str, pass: bool, value: Value| {
        clauses.insert(name.into(), clause(pass, json_map(value)));
    };
    insert(
        "metal_diffuse_zero",
        enough(&metal_diffuse) && metal_diffuse.failed == 0,
        json!({ "what": "diffuse radiance and hit distance are 0 on the metallic-1 sphere and the mirror", "pixels": metal_diffuse.json() }),
    );
    insert(
        "floor_specular_hit_distance",
        enough(&floor_specular) && floor_specular.failed == 0,
        json!({ "what": "specular hit distance on the smooth floor within 5 % of the traced reflection", "pixels": floor_specular.json() }),
    );
    insert(
        "diffuse_hit_distance",
        diffuse_range.failed == 0 && enough(&matte_one_lobe) && matte_one_lobe.failed == 0,
        json!({
            "what": "every diffuse hit distance is 0 or a ray inside the room; on the matte wall exactly one lobe has a hit distance",
            "room_diagonal": room_diagonal,
            "all_pixels": diffuse_range.json(),
            "matte_wall": matte_one_lobe.json(),
            "matte_wall_diffuse_sampled": matte_diffuse_sampled,
        }),
    );
    insert(
        "mirror",
        enough(&mirror_specular)
            && mirror_specular.failed == 0
            && enough(&mirror_guides)
            && mirror_guides.failed == 0,
        json!({
            "what": "on the flat mirror: all light is specular, its hit distance is the traced mirror-to-surface \
                     distance, and the guides are the mirror's own (normal, roughness 0), not replaced",
            "specular_hit_distance": mirror_specular.json(),
            "guides": mirror_guides.json(),
        }),
    );
    insert(
        "view_depth",
        enough(&depth) && depth.failed == 0,
        json!({ "what": "view depth within 1 % of the traced depth", "pixels": depth.json() }),
    );
}

// The debug views.

const DEBUG_VIEW_SHADER: &str = r"
@group(0) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var diffuse: texture_2d<f32>;
@group(0) @binding(2) var specular: texture_2d<f32>;

// Keep in sync with DENOISER_MISS_HIT_DISTANCE.
const MISS = 10000.0;

fn show_radiance(radiance: vec3<f32>) -> vec4<f32> {
    return vec4(radiance / (1.0 + radiance), 1.0);
}

fn show_distance(distance: f32) -> f32 {
    return select(1.0 - exp(-distance / 5.0), 0.0, distance <= 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn diffuse_view(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(diffuse)) { return; }
    textureStore(output, id.xy, show_radiance(textureLoad(diffuse, id.xy, 0).rgb));
}

@compute @workgroup_size(8, 8, 1)
fn specular_view(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(specular)) { return; }
    textureStore(output, id.xy, show_radiance(textureLoad(specular, id.xy, 0).rgb));
}

@compute @workgroup_size(8, 8, 1)
fn hitdist_view(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(diffuse)) { return; }
    let d = textureLoad(diffuse, id.xy, 0).a;
    let s = textureLoad(specular, id.xy, 0).a;
    textureStore(output, id.xy, vec4(show_distance(d), show_distance(s), select(0.0, 1.0, max(d, s) >= MISS), 1.0));
}
";

#[derive(Resource)]
struct DebugViewShader(Handle<Shader>, DebugView);

#[derive(Resource)]
struct DebugViewPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

fn init_debug_view_pipeline(
    mut commands: Commands,
    shader: Res<DebugViewShader>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "gfx_testbed_contract_debug_view",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("gfx_testbed_contract_debug_view".into()),
        layout: vec![layout.clone()],
        shader: shader.0.clone(),
        entry_point: Some(shader.1.entry_point().into()),
        ..default()
    });
    commands.insert_resource(DebugViewPipeline { layout, pipeline });
}

fn show_debug_view(
    view: ViewQuery<(&ViewTarget, &ViewDenoiserInputTextures)>,
    debug: Res<DebugViewPipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (target, inputs) = view.into_inner();
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(debug.pipeline) else {
        return;
    };
    let bind_group = device.create_bind_group(
        "gfx_testbed_contract_debug_view",
        &pipeline_cache.get_bind_group_layout(&debug.layout),
        &BindGroupEntries::sequential((
            target.main_texture_view(),
            &inputs.diffuse_radiance_hit_distance.default_view,
            &inputs.specular_radiance_hit_distance.default_view,
        )),
    );
    let size = inputs.view_depth.texture.size();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("gfx_testbed_contract_debug_view"),
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
}
