//! Bakes irradiance volumes for test scenes with `bevy_baked_gi`, and checks bakes against CPU references.
//!
//! ```text
//! gi_bake bake SCENE OUT [--seed N] [--samples N] [--first-hit] [--move-wall]
//!                                                  without --samples, each region's path count from its noise
//! gi_bake verify SCENE MANIFEST [--move-wall]      exit 1 when MANIFEST is stale for SCENE (no GPU used)
//! gi_bake furnace-check OUT RADIANCE               every texel of every side against RADIANCE
//! gi_bake asymmetric-check OUT                     the centre voxel against the same direction set on the CPU
//! gi_bake sphere-check OUT                         the inverted sphere against its closed form
//! gi_bake compare OUT_A OUT_B                      relative RMS between two bakes of one scene
//! gi_bake quantize OUT                             what Rgb9e5 would cost against the stored Rgba16Float
//! ```
//!
//! Scenes: `furnace` (a closed emissive cube), `asymmetric-furnace` (its six walls at six radiances), `sphere`
//! (an inverted diffuse sphere around a point light), `interior` (the graphics testbed's room) and `synthetic`
//! (twenty rooms off a corridor, generated from a fixed seed). A bake needs a GPU with ray queries; run it with
//! no window, e.g. through the fork's `tools/gfx-headless.sh`, and with `BEVY_ASSET_ROOT` at the repository root.

#[path = "gfx_testbed/scenes.rs"]
mod scenes;

use bevy::{
    app::PluginsState,
    asset::RenderAssetUsages,
    baked_gi::{
        bake::{
            fibonacci_directions, sample_direction, voxel_rotation, GiBakeMode, GiBakePlugin,
            GiBakeSettings,
        },
        scene_hash, BakedGiManifest, BakedRegion, GiBakeRegion, FACE_NAMES,
    },
    image::{CompressedImageFormats, ImageSampler, ImageType},
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    window::ExitCondition,
    winit::WinitPlugin,
};
use chacha20::ChaCha8Rng;
use core::f64::consts::PI;
use rand::{RngExt, SeedableRng};
use std::path::{Path, PathBuf};

/// What `gfx_testbed/scenes.rs` tells its caller about a scene; this example reads none of it.
#[expect(
    dead_code,
    reason = "the testbed's scene description, unused when baking"
)]
pub struct Scene {
    orbit: fn(u32) -> Transform,
    fov: f32,
    assets: Vec<UntypedHandle>,
}

/// The testbed's switchable light; `interior` marks its spot with it.
#[derive(Component)]
pub struct ToggleLight;

/// Each wall's radiance in `asymmetric-furnace`, as a multiple of the furnace radiance, for the walls at
/// +X, −X, +Y, −Y, +Z, −Z: the plan's frozen values (gate 2), whose centre-voxel faces are at least 15.6 % apart.
const ASYMMETRIC_RADIANCES: [f32; 6] = [1.0, 0.15, 0.7, 0.25, 0.5, 0.35];
/// The furnaces' radiance L, in cd/m².
const FURNACE_RADIANCE: f32 = 1.0;
/// Half the furnace cube's side, in metres.
const FURNACE_HALF: f32 = 1.0;
/// The inverted sphere: radius, its base colour (linear), and the point light's power in lumens.
const SPHERE_RADIUS: f32 = 2.0;
const SPHERE_ALBEDO: f32 = 0.5;
const SPHERE_LUMENS: f32 = 10_000.0;
/// The six axis directions in `FACE_NAMES` order.
const AXES: [Vec3; 6] = [
    Vec3::X,
    Vec3::NEG_X,
    Vec3::Y,
    Vec3::NEG_Y,
    Vec3::Z,
    Vec3::NEG_Z,
];

fn main() -> AppExit {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .map(|v| v.parse::<u32>().expect("a number"))
    };
    let positional: Vec<&str> = {
        let mut out = Vec::new();
        let mut skip = false;
        for a in &args {
            if skip {
                skip = false;
            } else if a == "--seed" || a == "--samples" {
                skip = true;
            } else if !a.starts_with("--") {
                out.push(a.as_str());
            }
        }
        out
    };
    let result = match positional.as_slice() {
        ["bake", scene, out] => {
            return bake(
                scene,
                PathBuf::from(out),
                GiBakeSettings {
                    output: PathBuf::from(out),
                    samples: value("--samples").unwrap_or(GiBakeSettings::default().samples),
                    // A fixed count when one is given; otherwise each region's count from its own noise.
                    adaptive: if value("--samples").is_some() {
                        None
                    } else {
                        GiBakeSettings::default().adaptive
                    },
                    seed: value("--seed").unwrap_or(0),
                    mode: if flag("--first-hit") {
                        GiBakeMode::FirstHitEmission
                    } else {
                        GiBakeMode::Full
                    },
                    ..default()
                },
                flag("--move-wall"),
            );
        }
        ["verify", scene, manifest] => verify(scene, Path::new(manifest), flag("--move-wall")),
        ["furnace-check", out, radiance] => {
            furnace_check(Path::new(out), radiance.parse().unwrap())
        }
        ["asymmetric-check", out] => asymmetric_check(Path::new(out)),
        ["sphere-check", out] => sphere_check(Path::new(out)),
        ["compare", a, b] => compare(Path::new(a), Path::new(b)),
        ["quantize", out] => quantize(Path::new(out)),
        _ => Err("usage: see the doc comment at the top of examples/graphics/gi_bake.rs".into()),
    };
    match result {
        Ok(()) => AppExit::Success,
        Err(message) => {
            eprintln!("{message}");
            AppExit::error()
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Scenes

/// A mesh Solari can trace: it builds acceleration structures only for meshes with tangents.
fn traceable(mesh: impl Into<Mesh>) -> Mesh {
    mesh.into()
        .with_generated_tangents()
        .expect("primitive meshes have what tangent generation needs")
}

/// The gates' test material: diffuse only as far as Solari allows (`reflectance: 0`, roughness 1), base colour
/// given in linear values.
fn test_material(base: f32, emissive: f32) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::linear_rgb(base, base, base),
        emissive: LinearRgba::rgb(emissive, emissive, emissive),
        reflectance: 0.0,
        perceptual_roughness: 1.0,
        metallic: 0.0,
        ..default()
    }
}

fn spawn_scene(name: &str, world: &mut World, move_wall: bool) {
    match name {
        "furnace" => furnace(world, [1.0; 6], move_wall),
        "asymmetric-furnace" => furnace(world, ASYMMETRIC_RADIANCES, move_wall),
        "sphere" => sphere(world),
        "interior" => {
            scenes::spawn("interior", world).expect("the testbed has an interior scene");
            world.spawn((
                Name::new("interior"),
                GiBakeRegion::default(),
                GiBakeRegion::padded_transform(
                    Vec3::new(-7.0, 0.0, -4.0),
                    Vec3::new(7.0, 3.0, 4.0),
                ),
            ));
        }
        "synthetic" => synthetic(world),
        other => panic!("no scene called {other}"),
    }
}

/// A closed cube centred at the origin whose walls face in and glow at `radiances[wall] × L`, in
/// `FACE_NAMES` order of the wall's position. One region fills it, 9 voxels a side so one voxel sits at the
/// centre. `move_wall` moves the +X wall out by 5 cm (the staleness test).
fn furnace(world: &mut World, radiances: [f32; 6], move_wall: bool) {
    for (wall, axis) in AXES.into_iter().enumerate() {
        let mesh = world
            .resource_mut::<Assets<Mesh>>()
            .add(traceable(Plane3d::new(-axis, Vec2::splat(FURNACE_HALF))));
        let material = world
            .resource_mut::<Assets<StandardMaterial>>()
            .add(test_material(0.0, radiances[wall] * FURNACE_RADIANCE));
        let offset = if move_wall && wall == 0 { 0.05 } else { 0.0 };
        world.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_translation(axis * (FURNACE_HALF + offset)),
        ));
    }
    world.spawn((
        Name::new("furnace"),
        GiBakeRegion {
            spacing: 2.0 * FURNACE_HALF / 9.0,
            ..default()
        },
        Transform::from_scale(Vec3::splat(2.0 * FURNACE_HALF)),
    ));
}

/// A UV sphere seen from inside: normals flipped and every triangle's winding reversed, so its inner side is
/// its front side.
fn inverted_sphere() -> Mesh {
    let mut mesh = Sphere::new(SPHERE_RADIUS).mesh().uv(64, 32);
    if let Some(VertexAttributeValues::Float32x3(normals)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
    {
        for n in normals {
            *n = [-n[0], -n[1], -n[2]];
        }
    }
    let indices: Vec<u32> = mesh.indices().unwrap().iter().map(|i| i as u32).collect();
    let reversed = indices.chunks(3).flat_map(|t| [t[0], t[2], t[1]]).collect();
    mesh.insert_indices(Indices::U32(reversed));
    traceable(mesh)
}

/// The inverted sphere's surface area: the sum of its triangles, not 4πR².
fn inverted_sphere_area() -> f64 {
    let mesh = inverted_sphere();
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        unreachable!()
    };
    let indices: Vec<usize> = mesh.indices().unwrap().iter().collect();
    indices
        .chunks(3)
        .map(|t| {
            let p = |i: usize| Vec3::from_array(positions[t[i]]).as_dvec3();
            0.5 * (p(1) - p(0)).cross(p(2) - p(0)).length()
        })
        .sum()
}

/// An inverted diffuse sphere with a point light at its centre (gate 3), and a region of 9 voxels a side inside
/// it.
fn sphere(world: &mut World) {
    let mesh = world.resource_mut::<Assets<Mesh>>().add(inverted_sphere());
    let material = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(test_material(SPHERE_ALBEDO, 0.0));
    world.spawn((Mesh3d(mesh), MeshMaterial3d(material)));
    world.spawn((
        PointLight {
            intensity: SPHERE_LUMENS,
            range: 100.0,
            radius: 0.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::default(),
    ));
    world.spawn((
        Name::new("sphere"),
        GiBakeRegion {
            spacing: 0.25,
            ..default()
        },
        Transform::from_scale(Vec3::splat(2.2)),
    ));
}

/// A generic level (gate 5): a corridor with ten rooms on each side, each room 3–14 m along the corridor and
/// 3–25 m deep, 45 cm walls, a 1 m door per room, a 3 m ceiling; a spot in most rooms and every 6 m of corridor.
/// Generated from a fixed seed, so every run builds the same level.
fn synthetic(world: &mut World) {
    const WALL: f32 = 0.45;
    const HEIGHT: f32 = 3.0;
    const CORRIDOR_HALF: f32 = 1.2;
    const DOOR_HALF: f32 = 0.5;
    const DOOR_HEIGHT: f32 = 2.1;
    let mut rng = ChaCha8Rng::seed_from_u64(0x5eed_2026);
    let mut boxes: Vec<(Vec3, Vec3)> = Vec::new();
    let mut rooms: Vec<(String, Vec3, Vec3, bool)> = Vec::new();
    let near = CORRIDOR_HALF + WALL;
    let mut corridor_end: f32 = 0.0;
    let mut side_ends = Vec::new();

    for (side_index, side) in [1.0f32, -1.0].into_iter().enumerate() {
        // A box in side-local coordinates: x along the corridor, `l` outwards from its centre line.
        let mut push = |x0: f32, x1: f32, l0: f32, l1: f32, y0: f32, y1: f32| {
            let (z0, z1) = (side * l0, side * l1);
            boxes.push((Vec3::new(x0, y0, z0.min(z1)), Vec3::new(x1, y1, z0.max(z1))));
        };
        let sizes: Vec<(f32, f32)> = (0..10)
            .map(|_| (rng.random_range(3.0..=14.0), rng.random_range(3.0..=25.0)))
            .collect();
        let mut x = 0.0;
        for (i, &(width, depth)) in sizes.iter().enumerate() {
            let previous_depth = if i == 0 { depth } else { sizes[i - 1].1 };
            push(
                x - WALL,
                x,
                CORRIDOR_HALF,
                near + depth.max(previous_depth) + WALL,
                0.0,
                HEIGHT,
            );
            let door = x + width / 2.0;
            push(x, door - DOOR_HALF, CORRIDOR_HALF, near, 0.0, HEIGHT);
            push(
                door + DOOR_HALF,
                x + width,
                CORRIDOR_HALF,
                near,
                0.0,
                HEIGHT,
            );
            push(
                door - DOOR_HALF,
                door + DOOR_HALF,
                CORRIDOR_HALF,
                near,
                DOOR_HEIGHT,
                HEIGHT,
            );
            push(
                x - WALL,
                x + width + WALL,
                near + depth,
                near + depth + WALL,
                0.0,
                HEIGHT,
            );
            let (l0, l1) = (near, near + depth);
            let (z0, z1) = (side * l0, side * l1);
            rooms.push((
                format!("room-{}{:02}", if side > 0.0 { 'n' } else { 's' }, i),
                Vec3::new(x, 0.0, z0.min(z1)),
                Vec3::new(x + width, HEIGHT, z0.max(z1)),
                (i * 7 + side_index * 3) % 10 != 4,
            ));
            x += width + WALL;
        }
        // The wall after the last room.
        push(
            x - WALL,
            x,
            CORRIDOR_HALF,
            near + sizes[9].1 + WALL,
            0.0,
            HEIGHT,
        );
        side_ends.push((side, x));
        corridor_end = corridor_end.max(x);
    }
    // Close the shorter side's corridor wall up to the far end, and the corridor's two ends.
    for (side, end) in side_ends {
        if end < corridor_end {
            let (z0, z1) = (side * CORRIDOR_HALF, side * near);
            boxes.push((
                Vec3::new(end, 0.0, z0.min(z1)),
                Vec3::new(corridor_end, HEIGHT, z0.max(z1)),
            ));
        }
    }
    boxes.push((Vec3::new(-WALL, 0.0, -near), Vec3::new(0.0, HEIGHT, near)));
    boxes.push((
        Vec3::new(corridor_end - WALL, 0.0, -near),
        Vec3::new(corridor_end, HEIGHT, near),
    ));
    let reach = near + 25.0 + 2.0 * WALL;
    let floor = (
        Vec3::new(-1.0, -0.2, -reach),
        Vec3::new(corridor_end + 1.0, 0.0, reach),
    );
    let ceiling = (
        Vec3::new(-1.0, HEIGHT, -reach),
        Vec3::new(corridor_end + 1.0, HEIGHT + 0.2, reach),
    );

    let wall_material = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::linear_rgb(0.45, 0.45, 0.42),
            perceptual_roughness: 0.9,
            ..default()
        });
    let floor_material = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::linear_rgb(0.25, 0.25, 0.25),
            perceptual_roughness: 0.6,
            ..default()
        });
    let spawn_box =
        |world: &mut World, (min, max): (Vec3, Vec3), material: &Handle<StandardMaterial>| {
            let mesh = world
                .resource_mut::<Assets<Mesh>>()
                .add(traceable(Cuboid::from_corners(Vec3::ZERO, max - min)));
            world.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                Transform::from_translation((min + max) / 2.0),
            ));
        };
    for b in &boxes {
        spawn_box(world, *b, &wall_material);
    }
    spawn_box(world, floor, &floor_material);
    spawn_box(world, ceiling, &wall_material);

    let spot = |world: &mut World, at: Vec3, lumens: f32| {
        world.spawn((
            SpotLight {
                intensity: lumens,
                range: 20.0,
                outer_angle: 1.2,
                inner_angle: 0.6,
                shadow_maps_enabled: false,
                ..default()
            },
            Transform::from_translation(at).looking_to(Vec3::NEG_Y, Vec3::Z),
        ));
    };
    for (name, min, max, lit) in &rooms {
        if *lit {
            let area = (max.x - min.x) * (max.z - min.z);
            spot(
                world,
                Vec3::new((min.x + max.x) / 2.0, 2.9, (min.z + max.z) / 2.0),
                150.0 * area,
            );
        }
        world.spawn((
            Name::new(name.clone()),
            GiBakeRegion::default(),
            GiBakeRegion::padded_transform(*min, *max),
        ));
    }
    let mut x = 3.0;
    while x < corridor_end - 1.0 {
        spot(world, Vec3::new(x, 2.9, 0.0), 2000.0);
        x += 6.0;
    }
    world.spawn((
        Name::new("corridor"),
        GiBakeRegion::default(),
        GiBakeRegion::padded_transform(
            Vec3::new(0.0, 0.0, -CORRIDOR_HALF),
            Vec3::new(corridor_end - WALL, HEIGHT, CORRIDOR_HALF),
        ),
    ));
}

// ---------------------------------------------------------------------------------------------------------------
// Bake and verify

fn bake(scene: &str, out: PathBuf, settings: GiBakeSettings, move_wall: bool) -> AppExit {
    let scene = scene.to_owned();
    println!("baking {scene} to {}", out.display());
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
            GiBakePlugin { settings },
        ))
        .add_systems(Startup, move |world: &mut World| {
            spawn_scene(&scene, world, move_wall);
        })
        .set_runner(run_until_baked)
        .run()
}

/// `ScheduleRunnerPlugin`'s loop, except that the process ends as soon as the bake asks to exit, without shutting
/// the app down: one bake of `synthetic` on the RTX 5070 (2026-11-25) hung after writing its files, with the main
/// and render threads both parked, and did not exit in two minutes. The files are complete when the bake asks.
fn run_until_baked(mut app: App) -> AppExit {
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
    }
    app.finish();
    app.cleanup();
    loop {
        app.update();
        if let Some(exit) = app.should_exit() {
            std::process::exit(match exit {
                AppExit::Success => 0,
                AppExit::Error(code) => i32::from(code.get()),
            });
        }
    }
}

/// Builds the scene in an app with no renderer at all and compares its hash with the manifest's.
fn verify(scene: &str, manifest: &Path, move_wall: bool) -> Result<(), String> {
    let text =
        std::fs::read_to_string(manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let manifest: BakedGiManifest = ron::de::from_str(&text).map_err(|e| e.to_string())?;
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, TransformPlugin, AssetPlugin::default()))
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_asset::<Image>();
    let scene = scene.to_owned();
    app.add_systems(Startup, move |world: &mut World| {
        spawn_scene(&scene, world, move_wall);
    });
    app.finish();
    app.cleanup();
    app.update();
    app.update();
    let hash = format!("{}", scene_hash(app.world_mut()));
    println!("scene hash {hash}, manifest {}", manifest.scene_hash);
    if hash == manifest.scene_hash {
        println!("up to date");
        Ok(())
    } else {
        Err("stale: the scene changed since the bake; rebake".into())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Reading a bake back

fn read_manifest(out: &Path) -> Result<BakedGiManifest, String> {
    let path = out.join("manifest.gi.ron");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    ron::de::from_str(&text).map_err(|e| e.to_string())
}

/// A baked volume read back through Bevy's KTX2 loader, sampled the way `irradiance_volume.wesl` samples it.
struct Volume {
    size: UVec3,
    texels: Vec<Vec3>,
    region: BakedRegion,
}

impl Volume {
    fn load(out: &Path, region: &BakedRegion) -> Result<Self, String> {
        let bytes = std::fs::read(out.join(&region.file)).map_err(|e| e.to_string())?;
        let image = Image::from_buffer(
            &bytes,
            ImageType::Extension("ktx2"),
            CompressedImageFormats::NONE,
            false,
            ImageSampler::Default,
            RenderAssetUsages::default(),
        )
        .map_err(|e| e.to_string())?;
        let size = image.texture_descriptor.size;
        let data = image.data.as_ref().ok_or("no data")?;
        let texels = data
            .chunks(8)
            .map(|t| {
                let c = |i: usize| f16_to_f32(u16::from_le_bytes([t[2 * i], t[2 * i + 1]]));
                Vec3::new(c(0), c(1), c(2))
            })
            .collect();
        Ok(Self {
            size: UVec3::new(size.width, size.height, size.depth_or_array_layers),
            texels,
            region: region.clone(),
        })
    }

    fn texel(&self, p: IVec3) -> Vec3 {
        let p = p.clamp(IVec3::ZERO, self.size.as_ivec3() - 1).as_uvec3();
        self.texels[(p.x + self.size.x * (p.y + self.size.y * p.z)) as usize]
    }

    /// A linear-filtered, clamp-to-edge fetch at normalised coordinates, as the volume's sampler does.
    fn sample(&self, uvw: Vec3) -> Vec3 {
        let st = uvw * self.size.as_vec3() - 0.5;
        let base = st.floor();
        let f = st - base;
        let b = base.as_ivec3();
        let mut sum = Vec3::ZERO;
        for corner in 0..8 {
            let o = IVec3::new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let w = Vec3::select(o.cmpeq(IVec3::ONE), f, 1.0 - f);
            sum += w.x * w.y * w.z * self.texel(b + o);
        }
        sum
    }

    /// `irradiance_volume.wesl`'s lookup for one volume at `world` with normal `normal`, reading side `face`'s
    /// slot of the atlas for the axis of side `slot_of[face]`: the identity reads the bake as the shader does, a
    /// permutation is a decode with sides swapped.
    fn irradiance(&self, world: Vec3, normal: Vec3, slot_of: [usize; 6]) -> Vec3 {
        let resolution = self.size.as_vec3() / Vec3::new(1.0, 2.0, 3.0);
        let light_from_world = self.region.transform().compute_affine().inverse();
        let unit = light_from_world.transform_point3(world);
        let stp = ((unit + 0.5) * resolution).clamp(Vec3::splat(0.5), resolution - 0.5);
        let uvw = stp / self.size.as_vec3();
        let mut total = Vec3::ZERO;
        let nn = normal * normal;
        for axis in 0..3 {
            let face = 2 * axis + usize::from(normal[axis] < 0.0);
            let slot = slot_of[face];
            let (slot_axis, slot_negative) = (slot / 2, slot % 2 == 1);
            let offset = Vec3::new(
                0.0,
                if slot_negative { 0.5 } else { 0.0 },
                slot_axis as f32 / 3.0,
            );
            total += self.sample(uvw + offset) * nn[axis];
        }
        total
    }

    fn voxel_centre(&self, voxel: UVec3) -> Vec3 {
        let r = UVec3::from_array(self.region.resolution).as_vec3();
        let unit = (voxel.as_vec3() + 0.5) / r - 0.5;
        self.region.transform().transform_point(unit)
    }

    /// Every voxel's six sides, read the way the shader reads them.
    fn sides(&self) -> impl Iterator<Item = (UVec3, [Vec3; 6])> + '_ {
        let r = UVec3::from_array(self.region.resolution);
        (0..r.z).flat_map(move |z| {
            (0..r.y).flat_map(move |y| {
                (0..r.x).map(move |x| {
                    let v = UVec3::new(x, y, z);
                    let at = self.voxel_centre(v);
                    (v, AXES.map(|axis| self.irradiance(at, axis, IDENTITY)))
                })
            })
        })
    }
}

const IDENTITY: [usize; 6] = [0, 1, 2, 3, 4, 5];

/// IEEE 754 half to single precision.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa * exp2i(-24),
        31 => f32::INFINITY,
        e => (1.0 + mantissa / 1024.0) * exp2i(e - 15),
    }
}

fn luminance(c: Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

// ---------------------------------------------------------------------------------------------------------------
// Gate 1: units

fn furnace_check(out: &Path, radiance: f32) -> Result<(), String> {
    let manifest = read_manifest(out)?;
    let (mut worst, mut sum, mut count) = (0.0f64, 0.0f64, 0usize);
    for region in &manifest.regions {
        let volume = Volume::load(out, region)?;
        for (_, sides) in volume.sides() {
            for side in sides {
                for c in side.to_array() {
                    let error = (f64::from(c) / f64::from(radiance) - 1.0).abs();
                    worst = worst.max(error);
                    sum += error;
                    count += 1;
                }
            }
        }
        println!(
            "region {}: {} invalid voxels",
            region.name, region.invalid_voxels
        );
    }
    println!(
        "furnace: {count} channel values over every voxel side; mean |error| {:.4} %, worst {:.4} % (gate: 1 %)",
        100.0 * sum / count as f64,
        100.0 * worst
    );
    if worst <= 0.01 {
        Ok(())
    } else {
        Err("FAIL".into())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Gate 2: layout

/// Each wall's exact share of the cosine lobe of a side at the cube's centre, (1/π) ∫ max(0, ω·a) dω over the wall,
/// by the midpoint rule on a 600 × 600 grid of each wall.
fn exact_shares() -> [[f64; 6]; 6] {
    let n = 600;
    let mut shares = [[0.0; 6]; 6];
    for (wall, axis) in AXES.iter().enumerate() {
        let axis = axis.as_dvec3();
        let (u, v) = (axis.any_orthonormal_pair().0, axis.any_orthonormal_pair().1);
        let step = 2.0 / n as f64;
        for i in 0..n {
            for j in 0..n {
                let p = axis
                    + u * (-1.0 + (i as f64 + 0.5) * step)
                    + v * (-1.0 + (j as f64 + 0.5) * step);
                let r2 = p.length_squared();
                let w = p / r2.sqrt();
                let solid_angle = w.dot(axis) * step * step / r2;
                for (side, a) in AXES.iter().enumerate() {
                    shares[side][wall] += w.dot(a.as_dvec3()).max(0.0) * solid_angle / PI;
                }
            }
        }
    }
    shares
}

fn asymmetric_check(out: &Path) -> Result<(), String> {
    // Before anything is read: the expected faces and their separation.
    let shares = exact_shares();
    let exact: Vec<f64> = (0..6)
        .map(|side| {
            (0..6)
                .map(|wall| shares[side][wall] * f64::from(ASYMMETRIC_RADIANCES[wall]))
                .sum()
        })
        .collect();
    println!(
        "form factors at the centre: facing wall {:.4}, side wall {:.4}, opposite {:.4}",
        shares[0][0], shares[0][2], shares[0][1]
    );
    let mut closest = (f64::MAX, 0, 0);
    for a in 0..6 {
        for b in a + 1..6 {
            let gap = (exact[a] - exact[b]).abs() / exact[a].max(exact[b]);
            if gap < closest.0 {
                closest = (gap, a, b);
            }
        }
    }
    println!(
        "exact centre faces (× L): {}; closest pair {} / {} {:.1} % apart (needs ≥ 10 %)",
        (0..6)
            .map(|s| format!("{} {:.4}", FACE_NAMES[s], exact[s]))
            .collect::<Vec<_>>()
            .join(", "),
        FACE_NAMES[closest.1],
        FACE_NAMES[closest.2],
        100.0 * closest.0
    );
    if closest.0 < 0.10 {
        return Err("the radiances do not separate the faces by 10 %".into());
    }

    let manifest = read_manifest(out)?;
    let region = &manifest.regions[0];
    let volume = Volume::load(out, region)?;
    let r = UVec3::from_array(region.resolution);
    if r % 2 != UVec3::ONE {
        return Err("the region needs an odd resolution for a voxel at the centre".into());
    }
    let centre = r / 2;
    let global_index = centre.x + r.x * (centre.y + r.y * centre.z);
    let at = volume.voxel_centre(centre);
    println!("centre voxel {centre} (bake index {global_index}) at {at}");

    // The same estimator on the CPU, with the bake's own rotated direction set.
    let rotation = voxel_rotation(manifest.bake.seed, global_index);
    let samples = manifest.bake.samples;
    let mut cpu = [0.0f64; 6];
    for base in fibonacci_directions(samples) {
        let d = sample_direction(&rotation, base);
        let wall = {
            let a = d.abs();
            let axis = if a.x >= a.y && a.x >= a.z {
                0
            } else if a.y >= a.z {
                1
            } else {
                2
            };
            2 * axis + usize::from(d[axis] < 0.0)
        };
        let radiance = f64::from(ASYMMETRIC_RADIANCES[wall] * FURNACE_RADIANCE);
        for (side, axis) in AXES.iter().enumerate() {
            cpu[side] += 4.0 * f64::from(d.dot(*axis).max(0.0)) * radiance / f64::from(samples);
        }
    }

    let decode = |slot_of: [usize; 6]| -> [f64; 6] {
        core::array::from_fn(|side| f64::from(volume.irradiance(at, AXES[side], slot_of).x))
    };
    let worst = |gpu: [f64; 6]| {
        (0..6)
            .map(|s| (gpu[s] / cpu[s] - 1.0).abs())
            .fold(0.0, f64::max)
    };
    let gpu = decode(IDENTITY);
    for side in 0..6 {
        println!(
            "{}: bake {:.5}, CPU same directions {:.5} ({:+.3} %), exact {:.5}",
            FACE_NAMES[side],
            gpu[side],
            cpu[side],
            100.0 * (gpu[side] / cpu[side] - 1.0),
            exact[side] * f64::from(FURNACE_RADIANCE)
        );
    }
    let pass = worst(gpu) <= 0.01;
    println!("worst side {:.4} % (gate: 1 %)", 100.0 * worst(gpu));

    let mut controls_failed = 0;
    for (a, name_a) in FACE_NAMES.iter().enumerate() {
        for (b, name_b) in FACE_NAMES.iter().enumerate().skip(a + 1) {
            let mut slot_of = IDENTITY;
            slot_of.swap(a, b);
            let error = worst(decode(slot_of));
            if error > 0.01 {
                controls_failed += 1;
            }
            println!(
                "control: {} and {} swapped, worst side {:.1} % -> {}",
                name_a,
                name_b,
                100.0 * error,
                if error > 0.01 {
                    "fails, as it must"
                } else {
                    "PASSES (control broken)"
                }
            );
        }
    }
    println!("{controls_failed} of 15 swapped decodes fail");
    if pass && controls_failed == 15 {
        Ok(())
    } else {
        Err("FAIL".into())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Gate 3: the inverted sphere

/// Bevy's DFG table (the one Solari samples for `F_ab`): 64 × 64 (u = N·V, v = perceptual roughness), R16G16.
struct Dfg {
    texels: Vec<[f32; 2]>,
}

impl Dfg {
    fn load() -> Self {
        let image = Image::from_buffer(
            include_bytes!("../../crates/bevy_pbr/src/environment_map/dfg.ktx2"),
            ImageType::Extension("ktx2"),
            CompressedImageFormats::NONE,
            false,
            ImageSampler::Default,
            RenderAssetUsages::default(),
        )
        .expect("Bevy's DFG table loads");
        let texels = image
            .data
            .unwrap()
            .chunks(4)
            .map(|t| {
                let c = |i: usize| f16_to_f32(u16::from_le_bytes([t[2 * i], t[2 * i + 1]]));
                [c(0), c(1)]
            })
            .collect();
        Self { texels }
    }

    /// Bilinear, clamp to edge, at texel centres: what `textureSampleLevel` with Bevy's linear sampler returns.
    fn f_ab(&self, n_dot_v: f64, perceptual_roughness: f64) -> [f64; 2] {
        let s = (n_dot_v * 64.0 - 0.5).clamp(0.0, 63.0);
        let t = (perceptual_roughness * 64.0 - 0.5).clamp(0.0, 63.0);
        let (x0, y0) = (s.floor() as usize, t.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(63), (y0 + 1).min(63));
        let (fx, fy) = (s - x0 as f64, t - y0 as f64);
        let at = |x: usize, y: usize, c: usize| f64::from(self.texels[x + 64 * y][c]);
        core::array::from_fn(|c| {
            (at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx) * (1.0 - fy)
                + (at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx) * fy
        })
    }
}

/// Solari's `evaluate_brdf` (`brdf.wesl`) for the test material (metallic 0, `reflectance: 0` so F0 = 0, roughness
/// 1), including the N·L factor, in a frame where the normal is +Z.
fn solari_brdf(dfg: &Dfg, wo: bevy::math::DVec3, wi: bevy::math::DVec3, albedo: f64) -> f64 {
    let (n_dot_l, n_dot_v) = (wi.z, wo.z);
    if n_dot_l < 0.0001 || n_dot_v < 0.0001 {
        return 0.0;
    }
    let f_ab = dfg.f_ab(n_dot_v.max(0.0001), 1.0);
    // F0 = 0: the dielectric specular lobe's albedo is F_ab.y; the diffuse lobe gets what is left.
    let diffuse = (1.0 - f_ab[1]) * albedo / PI * n_dot_l;
    let h = (wi + wo).normalize();
    let (n_dot_h, l_dot_h) = (h.z.min(1.0), wi.dot(h));
    if n_dot_h < 0.0001 || l_dot_h < 0.0001 {
        return diffuse;
    }
    let roughness = 1.0f64;
    let d = {
        let a = n_dot_h * roughness;
        let k = roughness / (1.0 - n_dot_h * n_dot_h + a * a);
        k * k / PI
    };
    let v = {
        let a2 = roughness * roughness;
        let lv = n_dot_l * ((n_dot_v - a2 * n_dot_v) * n_dot_v + a2).sqrt();
        let ll = n_dot_v * ((n_dot_l - a2 * n_dot_l) * n_dot_l + a2).sqrt();
        0.5 / (lv + ll)
    };
    let fresnel = (1.0 - l_dot_h).clamp(0.0, 1.0).powi(5);
    diffuse + d * v * fresnel * n_dot_l
}

fn sphere_check(out: &Path) -> Result<(), String> {
    use bevy::math::DVec3;
    let dfg = Dfg::load();
    let albedo = f64::from(SPHERE_ALBEDO);
    let area = inverted_sphere_area();
    let flux = f64::from(SPHERE_LUMENS);
    let direct = flux / area;

    // (a) The gate's closed form with ρ_eff: the material's albedo, cosine-weighted over the hemisphere, where
    // the albedo at N·V = μ is (1 − F_ab.y(μ)) × base + F_ab.y(μ).
    let steps = 2000;
    let rho_eff: f64 = (0..steps)
        .map(|i| {
            let mu = (i as f64 + 0.5) / steps as f64;
            let s = dfg.f_ab(mu, 1.0)[1];
            ((1.0 - s) * albedo + s) * 2.0 * mu / steps as f64
        })
        .sum();
    let closed_form = flux * rho_eff / (area * (1.0 - rho_eff)) / PI;
    println!(
        "sphere: A = {area:.4} m² (triangles; 4πR² = {:.4}), Φ = {flux} lm, E_direct = Φ/A = {direct:.4} lx",
        4.0 * PI * f64::from(SPHERE_RADIUS).powi(2)
    );
    println!(
        "ρ_eff = {rho_eff:.5} (base {albedo}, F_ab.y at μ = 1: {:.5}, at μ = 0.5: {:.5}); closed form \
         Φρ/(A(1−ρ)) / π = {closed_form:.4} cd/m² (stored convention E/π)",
        dfg.f_ab(1.0, 1.0)[1],
        dfg.f_ab(0.5, 1.0)[1]
    );

    // (b) Cross-check: the steady state of a perfect sphere with Solari's angle-dependent BRDF. Inside a sphere
    // the angle a ray leaves a wall at equals the angle it arrives at the next wall, so the outgoing radiance
    // depends on μ alone: L(μo) = brdf(μo ← normal) E_direct + ∫ brdf(μo ← ωi) L(μi) dωi.
    let bins = 64;
    let phis = 128;
    let mu = |i: usize| (i as f64 + 0.5) / bins as f64;
    let dir = |m: f64, phi: f64| {
        DVec3::new(
            (1.0 - m * m).sqrt() * phi.cos(),
            (1.0 - m * m).sqrt() * phi.sin(),
            m,
        )
    };
    let mut kernel = vec![vec![0.0; bins]; bins];
    for (o, row) in kernel.iter_mut().enumerate() {
        let wo = dir(mu(o), 0.0);
        for (i, k) in row.iter_mut().enumerate() {
            for p in 0..phis {
                let phi = (p as f64 + 0.5) / phis as f64 * 2.0 * PI;
                *k += solari_brdf(&dfg, wo, dir(mu(i), phi), albedo)
                    * (1.0 / bins as f64)
                    * (2.0 * PI / phis as f64);
            }
        }
    }
    let first: Vec<f64> = (0..bins)
        .map(|o| solari_brdf(&dfg, dir(mu(o), 0.0), DVec3::Z, albedo) * direct)
        .collect();
    let mut radiance = first.clone();
    for _ in 0..200 {
        radiance = (0..bins)
            .map(|o| first[o] + (0..bins).map(|i| kernel[o][i] * radiance[i]).sum::<f64>())
            .collect();
    }
    let l_of = |m: f64| radiance[((m * bins as f64) as usize).min(bins - 1)];

    let manifest = read_manifest(out)?;
    let region = &manifest.regions[0];
    let volume = Volume::load(out, region)?;
    let directions = fibonacci_directions(4096);
    let radius = f64::from(SPHERE_RADIUS);
    let (mut sum_bake, mut sum_b, mut count, mut worst_a, mut worst_b) =
        (0.0, 0.0, 0usize, 0.0f64, 0.0f64);
    for (voxel, sides) in volume.sides() {
        let p = volume.voxel_centre(voxel).as_dvec3();
        let mut expected = [0.0f64; 6];
        for d in &directions {
            let w = d.as_dvec3();
            // Where the ray leaves the sphere, and the angle to the wall's inward normal there.
            let b = p.dot(w);
            let t = -b + (b * b - p.length_squared() + radius * radius).sqrt();
            let y = p + w * t;
            let m = y.dot(w) / radius;
            for (side, axis) in AXES.iter().enumerate() {
                expected[side] +=
                    4.0 * w.dot(axis.as_dvec3()).max(0.0) * l_of(m) / directions.len() as f64;
            }
        }
        for side in 0..6 {
            let bake = f64::from(luminance(sides[side]));
            sum_bake += bake;
            sum_b += expected[side];
            count += 1;
            worst_a = worst_a.max((bake / closed_form - 1.0).abs());
            worst_b = worst_b.max((bake / expected[side] - 1.0).abs());
        }
    }
    let mean_bake = sum_bake / count as f64;
    let mean_b = sum_b / count as f64;
    println!(
        "steady state, centre voxel L(μ=1) = {:.4}; mean over the region's voxel sides {mean_b:.4}",
        l_of(1.0)
    );
    let mean_a_error = mean_bake / closed_form - 1.0;
    println!(
        "bake: mean over {count} voxel sides {mean_bake:.4}: {:+.3} % against the closed form (gate: 2 %), worst \
         side {:.3} % (gate: 6 %); against the steady state: mean {:+.3} %, worst {:.3} %; {} invalid voxels",
        100.0 * mean_a_error,
        100.0 * worst_a,
        100.0 * (mean_bake / mean_b - 1.0),
        100.0 * worst_b,
        region.invalid_voxels
    );
    if mean_a_error.abs() <= 0.02 && worst_a <= 0.06 {
        Ok(())
    } else {
        Err("FAIL".into())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Gate 4: noise and vendors

/// Per-texel relative difference of luminance, (b − a) / ((a + b) / 2), RMS over the texels (voxel sides) whose
/// luminance in `a` is above 1 % of `a`'s brightest texel in that volume; pooled over all regions, and per region.
fn compare(a: &Path, b: &Path) -> Result<(), String> {
    let manifest_a = read_manifest(a)?;
    let manifest_b = read_manifest(b)?;
    let (mut sum, mut count) = (0.0f64, 0usize);
    let mut worst_region = (0.0f64, String::new());
    for region in &manifest_a.regions {
        let other = manifest_b
            .regions
            .iter()
            .find(|r| r.name == region.name)
            .ok_or_else(|| format!("{} is missing from {}", region.name, b.display()))?;
        let va = Volume::load(a, region)?;
        let vb = Volume::load(b, other)?;
        let sides_a: Vec<f32> = va.sides().flat_map(|(_, s)| s.map(luminance)).collect();
        let sides_b: Vec<f32> = vb.sides().flat_map(|(_, s)| s.map(luminance)).collect();
        let max = sides_a.iter().copied().fold(0.0f32, f32::max);
        let (mut region_sum, mut region_count) = (0.0f64, 0usize);
        for (x, y) in sides_a.iter().zip(&sides_b) {
            if *x <= 0.01 * max {
                continue;
            }
            let rel = f64::from((y - x) / ((x + y) / 2.0));
            region_sum += rel * rel;
            region_count += 1;
        }
        let rms = (region_sum / region_count.max(1) as f64).sqrt();
        if rms > worst_region.0 {
            worst_region = (rms, region.name.clone());
        }
        sum += region_sum;
        count += region_count;
    }
    let rms = (sum / count.max(1) as f64).sqrt();
    println!(
        "relative RMS over {count} texels above 1 % of their volume's max: {:.3} %; worst region {} {:.3} %",
        100.0 * rms,
        worst_region.1,
        100.0 * worst_region.0
    );
    println!(
        "adapters: {} / {}",
        manifest_a.bake.adapter, manifest_b.bake.adapter
    );
    Ok(())
}

/// 2^n, exact, for n in -126..=127 (built from the bits: `powi` is not portable across platforms).
fn exp2i(n: i32) -> f32 {
    f32::from_bits(((n + 127) as u32) << 23)
}

/// Rgb9e5 as Vulkan's `E5B9G9R9_UFLOAT_PACK32` stores it (shared exponent, 9-bit mantissas), back to f32.
fn rgb9e5_round_trip(c: Vec3) -> Vec3 {
    let max_value = 65408.0f32;
    let c = c.clamp(Vec3::ZERO, Vec3::splat(max_value));
    let max = c.max_element();
    if max == 0.0 {
        return Vec3::ZERO;
    }
    // floor(log2(max)), read from the float's exponent bits (a subnormal reads as -127)
    let floor_log2 = ((max.to_bits() >> 23) & 0xff) as i32 - 127;
    let exponent = floor_log2.max(-16) + 1 + 15;
    let mut scale = exp2i(exponent - 15 - 9);
    if (max / scale + 0.5).floor() as i32 == 512 {
        scale *= 2.0;
    }
    (c / scale + 0.5).floor() * scale
}

fn quantize(out: &Path) -> Result<(), String> {
    let manifest = read_manifest(out)?;
    let (mut sum, mut worst, mut count) = (0.0f64, 0.0f64, 0usize);
    for region in &manifest.regions {
        let volume = Volume::load(out, region)?;
        let max = volume
            .texels
            .iter()
            .map(|t| t.max_element())
            .fold(0.0f32, f32::max);
        for t in &volume.texels {
            for (q, v) in rgb9e5_round_trip(*t)
                .to_array()
                .into_iter()
                .zip(t.to_array())
            {
                if v <= 0.01 * max {
                    continue;
                }
                let rel = f64::from((q - v) / v);
                sum += rel * rel;
                worst = worst.max(rel.abs());
                count += 1;
            }
        }
    }
    println!(
        "Rgb9e5 against the stored Rgba16Float, per channel above 1 % of the volume's max: RMS {:.3} %, worst {:.3} % \
         over {count} values (Rgba16Float itself rounds to within 0.049 %)",
        100.0 * (sum / count.max(1) as f64).sqrt(),
        100.0 * worst
    );
    Ok(())
}
