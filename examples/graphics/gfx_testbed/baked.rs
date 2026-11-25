//! The testbed's baked indirect light (`bevy_baked_gi`, through `bevy_graphics`): its scenes, the flags that bake
//! a scene or load its bake, and the views its image gates read.
//!
//! - `--bake-gi DIR` bakes the scene into `DIR` in a windowless app and exits (a binary built with the
//!   `bevy_baked_gi_bake` feature). `--baked-gi DIR` loads that bake into the run.
//! - `--without-toggle-light` and `--lights-off NAME,...` spawn the scene without lights, for references and
//!   for exact bakes; they act the same in a bake and in a run.
//! - `--gi-coverage` is a debug view of the bake alone: every material white and matte, no analytic light,
//!   every volume at intensity 1, no tonemapping. Beside each capture it writes `mask-<capture>.png`: the pixels
//!   whose surface lies inside a baked box, found by tracing the scene's triangles on the CPU. A black pixel of
//!   the capture inside the mask is a surface the bake does not reach.
//! - In `gi-furnace` with a bake loaded, the run also writes the CPU's prediction of the frame
//!   (`predicted.png`), the same with each axis's two sides swapped (`predicted-swap-{x,y,z}.png`), and one mask
//!   per side of the sphere's pixels whose normal faces that wall (`mask-{px,nx,py,ny,pz,nz}.png`).
//!
//! `report.json` gets a `gi` section: the bake's status, the boxes in the camera's frustum, the device limits
//! that decide whether Bevy can bind them, and the scenes' measurement rectangles.

use crate::{ActiveScene, Args, Flags, Phase, Report, Run, Scene, TestbedCamera, ToggleLight};
use bevy::{
    asset::{io::AssetSourceBuilder, RenderAssetUsages},
    camera::{
        primitives::{Aabb, Frustum},
        Exposure,
    },
    graphics::{
        baked::{BakedGi, BakedGiManifest, BakedGiPlugin, BakedGiStatus, GiBakeRegion},
        GraphicsSettings,
    },
    image::{CompressedImageFormats, ImageSampler, ImageType},
    light::IrradianceVolume,
    math::Affine3A,
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        renderer::RenderDevice,
        view::{DebandDither, Tonemapping},
    },
};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// The usage text of this file's flags, printed by `--help` after the others.
pub const HELP: &str = "
Baked indirect light (bevy_baked_gi):
  --baked-gi DIR                 load DIR/manifest.gi.ron, a bake of this scene
  --bake-gi DIR                  bake the scene into DIR and exit, without a window (needs a binary built with
                                 the bevy_baked_gi_bake feature)
  --bake-seed N                  the bake's rotation seed (default 0)
  --bake-samples N               a fixed path count per voxel, instead of each region's from its own noise
  --gi-one-box                   bake the two-rooms scenes under one box spanning both rooms
  --gi-skip-diffuser             diffuser: mark the diffuser GiBakeSkipEmission
  --without-toggle-light         spawn the scene without the fixture --light-toggle-at switches off
  --lights-off NAME,...          spawn the scene without these named lights
  --gi-pin-intensity             hold every baked volume at intensity 1, whatever the lights do
  --gi-coverage                  the bake alone: white matte materials, no analytic light, volumes at
                                 intensity 1, exposure 0 EV, no tonemapping; writes mask-<capture>.png
  Scenes: two-rooms (0.45 m wall), two-rooms-thin (0.10 m), two-rooms-door (0.45 m, 1 m door), rooms (12 rooms
  off a corridor), gi-furnace (gate 2's asymmetric furnace with a white sphere left out of the bake),
  diffuser (a bake-only room with a spot and an emissive diffuser)
";

/// The asset source a loaded bake is read from.
const SOURCE: &str = "gi";
/// How far inside a baked box a surface must lie to be in the coverage mask, in metres: the GPU and the CPU
/// tracer may disagree by a little at a box face.
const MASK_MARGIN: f32 = 0.01;
/// The share of a normal along an axis above which a sphere pixel faces that axis's wall (gate 6). to tune.
const FACING: f32 = 0.8;
/// Pixels a gate-6 region mask is eroded by, so no pixel near a silhouette or a region border counts. to tune.
const ERODE: i32 = 2;

/// The radiance of the furnace walls in gate 2's frozen proportions, for the walls at +X, −X, +Y, −Y, +Z, −Z.
const FURNACE_RADIANCES: [f32; 6] = [1.0, 0.15, 0.7, 0.25, 0.5, 0.35];
/// Half the furnace's side, as in `gi_bake`'s.
const FURNACE_HALF: f32 = 1.0;
/// The white sphere in the furnace.
const FURNACE_SPHERE: f32 = 0.35;
/// The exposure of the scenes whose frames are predicted on the CPU: 1 / 1.2 of the radiance.
const PREDICTED_EV100: f32 = 0.0;

/// A thing the bake leaves out: it moves at run time (gate 6's sphere).
#[derive(Component)]
pub struct GiDynamic;

/// The diffuser of the `diffuser` scene.
#[derive(Component)]
pub struct Diffuser;

/// This file's flags.
#[derive(Resource, Clone, Default)]
pub struct Baked {
    load: Option<PathBuf>,
    bake: Option<PathBuf>,
    bake_seed: u32,
    bake_samples: Option<u32>,
    one_box: bool,
    skip_diffuser: bool,
    without_toggle_light: bool,
    lights_off: Vec<String>,
    pin_intensity: bool,
    coverage: bool,
}

impl Baked {
    /// Takes this file's flags off the command line.
    pub fn parse(flags: &mut Flags) -> Self {
        let dir = |flags: &mut Flags, name: &str| {
            flags.value(name).map(|value| {
                let path = PathBuf::from(value);
                std::path::absolute(&path).unwrap_or(path)
            })
        };
        let baked = Baked {
            load: dir(flags, "--baked-gi"),
            bake: dir(flags, "--bake-gi"),
            bake_seed: flags.parsed("--bake-seed").unwrap_or(0),
            bake_samples: flags.parsed("--bake-samples"),
            one_box: flags.switch("--gi-one-box"),
            skip_diffuser: flags.switch("--gi-skip-diffuser"),
            without_toggle_light: flags.switch("--without-toggle-light"),
            lights_off: flags
                .value("--lights-off")
                .map(|names| names.split(',').map(str::to_owned).collect())
                .unwrap_or_default(),
            pin_intensity: flags.switch("--gi-pin-intensity"),
            coverage: flags.switch("--gi-coverage"),
        };
        if baked.load.is_some() && baked.bake.is_some() {
            flags.error("--bake-gi bakes and exits: no --baked-gi with it");
        }
        if baked.bake_samples.is_some_and(|n| n == 0 || n % 256 != 0) {
            flags.error("--bake-samples must be a positive multiple of 256");
        }
        if baked.coverage && baked.load.is_none() {
            flags.error("--gi-coverage shows a bake: give --baked-gi");
        }
        baked
    }

    /// With `--bake-gi`: bakes the scene and returns the process's exit code; `None` otherwise.
    pub fn bake(&self, args: &Args) -> Option<AppExit> {
        let out = self.bake.clone()?;
        #[cfg(feature = "bevy_baked_gi_bake")]
        {
            Some(bake::run(self.clone(), args.scene.clone(), out))
        }
        #[cfg(not(feature = "bevy_baked_gi_bake"))]
        {
            let _ = (args, out);
            eprintln!(
                "gfx_testbed: --bake-gi needs a binary built with the bevy_baked_gi_bake feature"
            );
            Some(AppExit::from_code(2))
        }
    }

    /// Registers the folder of `--baked-gi` as an asset source. Called before the default plugins.
    pub fn register_source(&self, app: &mut App) {
        if let Some(dir) = &self.load {
            app.register_asset_source(
                SOURCE,
                AssetSourceBuilder::platform_default(&dir.to_string_lossy(), None),
            );
        }
    }

    /// Adds this file's systems, after the ladder's plugins.
    pub fn build(&self, app: &mut App, _args: &Args) {
        app.insert_resource(self.clone());
        if self.load.is_none() {
            return;
        }
        if !app.is_plugin_added::<BakedGiPlugin>() {
            app.add_plugins(BakedGiPlugin);
        }
        if self.coverage
            && let Some(mut settings) = app.world_mut().get_resource_mut::<GraphicsSettings>()
        {
            // Nothing but the volumes may light or darken the coverage view.
            let overrides = &mut settings.overrides;
            overrides.ssao = Some(false);
            overrides.ssr = Some(false);
            overrides.contact_shadows = Some(false);
            overrides.bloom = Some(false);
        }
        if self.pin_intensity || self.coverage {
            app.add_systems(Last, pin_intensity);
        }
        app.add_systems(Update, record.after(crate::drive).before(crate::finish));
    }

    /// Spawns the scene called `name` if this file owns it.
    pub fn scene(name: &str, world: &mut World) -> Option<Scene> {
        Some(match name {
            "two-rooms" => two_rooms(world, 0.45, false),
            "two-rooms-thin" => two_rooms(world, 0.10, false),
            "two-rooms-door" => two_rooms(world, 0.45, true),
            "rooms" => rooms(world),
            "gi-furnace" => furnace(world),
            "diffuser" => diffuser(world),
            _ => return None,
        })
    }

    /// Gives the camera what this file's views need.
    pub fn camera(&self, args: &Args, camera: &mut EntityWorldMut) {
        if self.coverage || args.scene == "gi-furnace" {
            // A frame the CPU predicts or reads for black: linear light times the exposure, clamped, sRGB.
            camera.insert((
                Exposure {
                    ev100: PREDICTED_EV100,
                },
                Tonemapping::None,
                DebandDither::Disabled,
            ));
        }
    }

    /// Shapes the spawned scene for this run: lights removed, region markers, the bake loaded, and the coverage
    /// view's materials. Returns the scene with the bake among the assets it waits for.
    pub fn prepare(&self, world: &mut World, mut scene: Scene) -> Scene {
        if let Err(failure) = self.prepare_level(world, &scene_name(world), false) {
            world.resource_mut::<Report>().failures.push(failure);
        }
        if self.load.is_some() {
            let manifest: Handle<BakedGiManifest> = world
                .resource::<AssetServer>()
                .load(format!("{SOURCE}://manifest.gi.ron"));
            scene.assets.push(manifest.clone().untyped());
            world.spawn((Name::new("baked GI"), BakedGi(manifest)));
        }
        if self.coverage {
            let lights: Vec<Entity> = world
                .query_filtered::<Entity, Or<(
                    With<PointLight>,
                    With<SpotLight>,
                    With<RectLight>,
                    With<DirectionalLight>,
                )>>()
                .iter(world)
                .collect();
            for light in lights {
                world.despawn(light);
            }
            for (_, material) in world.resource_mut::<Assets<StandardMaterial>>().iter_mut() {
                *material = StandardMaterial {
                    base_color: Color::WHITE,
                    perceptual_roughness: 1.0,
                    reflectance: 0.0,
                    metallic: 0.0,
                    ..default()
                };
            }
        }
        scene
    }

    /// What a bake and a run of the same flags share: the lights left out and the region markers. A bake also
    /// leaves out what moves, and marks the diffuser when asked.
    fn prepare_level(&self, world: &mut World, name: &str, baking: bool) -> Result<(), String> {
        if self.without_toggle_light {
            let toggles: Vec<Entity> = world
                .query_filtered::<Entity, With<ToggleLight>>()
                .iter(world)
                .collect();
            if toggles.is_empty() {
                return Err("--without-toggle-light: the scene marks no fixture".into());
            }
            for light in toggles {
                world.despawn(light);
            }
        }
        for wanted in &self.lights_off {
            let found: Vec<Entity> = world
                .query::<(Entity, &Name)>()
                .iter(world)
                .filter(|(_, name)| name.as_str() == wanted)
                .map(|(entity, _)| entity)
                .collect();
            if found.is_empty() {
                return Err(format!(
                    "--lights-off: the scene has no light called {wanted}"
                ));
            }
            for light in found {
                world.despawn(light);
            }
        }
        for (region_name, transform, region) in regions(name, self.one_box) {
            world.spawn((Name::new(region_name), region, transform));
        }
        if baking {
            let dynamic: Vec<Entity> = world
                .query_filtered::<Entity, With<GiDynamic>>()
                .iter(world)
                .collect();
            for entity in dynamic {
                world.despawn(entity);
            }
        }
        if self.skip_diffuser {
            let diffusers: Vec<Entity> = world
                .query_filtered::<Entity, With<Diffuser>>()
                .iter(world)
                .collect();
            if diffusers.is_empty() {
                return Err("--gi-skip-diffuser: the scene has no diffuser".into());
            }
            for entity in diffusers {
                world
                    .entity_mut(entity)
                    .insert(bevy::graphics::baked::GiBakeSkipEmission);
            }
        }
        Ok(())
    }
}

fn scene_name(world: &World) -> String {
    world.resource::<Args>().scene.clone()
}

// ---------------------------------------------------------------------------------------------------------------
// Regions

/// The boxes a scene is baked in: (name, transform, region).
fn regions(scene: &str, one_box: bool) -> Vec<(String, Transform, GiBakeRegion)> {
    let padded = |name: &str, min: Vec3, max: Vec3| {
        (
            name.to_owned(),
            GiBakeRegion::padded_transform(min, max),
            GiBakeRegion::default(),
        )
    };
    match scene {
        "interior" => vec![padded(
            "interior",
            Vec3::new(-7.0, 0.0, -4.0),
            Vec3::new(7.0, 3.0, 4.0),
        )],
        "two-rooms" | "two-rooms-thin" | "two-rooms-door" => {
            let half = two_rooms_wall(scene) / 2.0;
            if one_box {
                vec![padded(
                    "both",
                    Vec3::new(-ROOM_LENGTH - half, 0.0, -ROOM_HALF_WIDTH),
                    Vec3::new(ROOM_LENGTH + half, ROOM_HEIGHT, ROOM_HALF_WIDTH),
                )]
            } else {
                vec![
                    padded(
                        "room-a",
                        Vec3::new(-ROOM_LENGTH - half, 0.0, -ROOM_HALF_WIDTH),
                        Vec3::new(-half, ROOM_HEIGHT, ROOM_HALF_WIDTH),
                    ),
                    padded(
                        "room-b",
                        Vec3::new(half, 0.0, -ROOM_HALF_WIDTH),
                        Vec3::new(ROOM_LENGTH + half, ROOM_HEIGHT, ROOM_HALF_WIDTH),
                    ),
                ]
            }
        }
        "rooms" => {
            let mut out = Vec::new();
            for i in 0..ROOMS_PER_SIDE {
                let x0 = rooms_wall_x(i) + WALL;
                for (side, letter) in [(1.0f32, 'n'), (-1.0, 's')] {
                    let (z0, z1) = (side * CORRIDOR_OUTER, side * (CORRIDOR_OUTER + ROOM_DEPTH));
                    out.push(padded(
                        &format!("room-{letter}{i}"),
                        Vec3::new(x0, 0.0, z0.min(z1)),
                        Vec3::new(x0 + ROOM_SIZE, ROOM_HEIGHT, z0.max(z1)),
                    ));
                }
            }
            out.push(padded(
                "corridor",
                Vec3::new(WALL, 0.0, -CORRIDOR_HALF),
                Vec3::new(rooms_wall_x(ROOMS_PER_SIDE), ROOM_HEIGHT, CORRIDOR_HALF),
            ));
            out
        }
        "gi-furnace" => vec![(
            "furnace".into(),
            Transform::from_scale(Vec3::splat(2.0 * FURNACE_HALF)),
            GiBakeRegion {
                spacing: 2.0 * FURNACE_HALF / 9.0,
                ..default()
            },
        )],
        "diffuser" => vec![padded(
            "room",
            Vec3::new(-2.5, 0.0, -2.0),
            Vec3::new(2.5, ROOM_HEIGHT, 2.0),
        )],
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Scenes

const WALL: f32 = 0.45;
const ROOM_HEIGHT: f32 = 3.0;
const ROOM_LENGTH: f32 = 5.0;
const ROOM_HALF_WIDTH: f32 = 2.0;
/// Room A's lamp, in lumens. Room B's is [`B_LAMP_SHARE`] of it.
const A_LAMP: f32 = 1_000_000.0;
/// to tune: gate 9a's frozen 5 % of A's intensity.
const B_LAMP_SHARE: f32 = 0.05;
/// The door of `two-rooms-door`, along the shared wall (z) and its height.
const DOOR_Z: (f32, f32) = (0.5, 1.5);
const DOOR_HEIGHT: f32 = 2.1;
/// The part of the shared wall `report.json` names `wall`: away from the door and the corners.
const WALL_Z: (f32, f32) = (-1.6, 0.2);
const WALL_Y: (f32, f32) = (0.3, 2.6);

fn two_rooms_wall(scene: &str) -> f32 {
    if scene == "two-rooms-thin" {
        0.10
    } else {
        WALL
    }
}

/// A mesh Solari can trace: it builds acceleration structures only for meshes with tangents.
fn traceable(mesh: impl Into<Mesh>) -> Mesh {
    mesh.into()
        .with_generated_tangents()
        .expect("primitive meshes have what tangent generation needs")
}

fn matte(world: &mut World, linear: f32) -> Handle<StandardMaterial> {
    world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::linear_rgb(linear, linear, linear),
            perceptual_roughness: 0.9,
            ..default()
        })
}

/// A wall, floor or ceiling slab from `min` to `max`.
fn slab(world: &mut World, min: Vec3, max: Vec3, material: &Handle<StandardMaterial>) {
    let mesh = world
        .resource_mut::<Assets<Mesh>>()
        .add(traceable(Cuboid::from_corners(Vec3::ZERO, max - min)));
    world.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material.clone()),
        Transform::from_translation((min + max) / 2.0),
    ));
}

fn spot(world: &mut World, name: &str, at: Vec3, lumens: f32) -> Entity {
    world
        .spawn((
            Name::new(name.to_owned()),
            SpotLight {
                intensity: lumens,
                range: 20.0,
                inner_angle: 0.6,
                outer_angle: 1.1,
                shadow_maps_enabled: true,
                ..default()
            },
            Transform::from_translation(at).looking_to(Vec3::NEG_Y, Vec3::Z),
        ))
        .id()
}

/// Two rooms of 5 by 4 m and 3 m high on either side of one wall (x = 0) of `wall` metres, A at −x with its
/// lamp (`lamp-a`, the fixture `--light-toggle-at` switches off) and B at +x with a lamp at 5 % of A's
/// (`lamp-b`). With `door`, a 1 m by 2.1 m opening joins them near one end.
///
/// The still camera stands in B and looks at the shared wall; poses 1 to 480 walk from B through the door into
/// A; pose 1000 stands in A and looks at the shared wall.
fn two_rooms(world: &mut World, wall: f32, door: bool) -> Scene {
    let walls = matte(world, 0.6);
    let floor = matte(world, 0.3);
    let (half, outer) = (wall / 2.0, ROOM_LENGTH + wall / 2.0);
    let (w, h) = (ROOM_HALF_WIDTH, ROOM_HEIGHT);
    slab(
        world,
        Vec3::new(-outer - WALL, -0.2, -w - WALL),
        Vec3::new(outer + WALL, 0.0, w + WALL),
        &floor,
    );
    slab(
        world,
        Vec3::new(-outer - WALL, h, -w - WALL),
        Vec3::new(outer + WALL, h + 0.2, w + WALL),
        &walls,
    );
    for side in [-1.0f32, 1.0] {
        // The far walls along x, and the long walls along z.
        let (x0, x1) = (side * outer, side * (outer + WALL));
        slab(
            world,
            Vec3::new(x0.min(x1), 0.0, -w - WALL),
            Vec3::new(x0.max(x1), h, w + WALL),
            &walls,
        );
        let (z0, z1) = (side * w, side * (w + WALL));
        slab(
            world,
            Vec3::new(-outer, 0.0, z0.min(z1)),
            Vec3::new(outer, h, z0.max(z1)),
            &walls,
        );
    }
    if door {
        slab(
            world,
            Vec3::new(-half, 0.0, -w),
            Vec3::new(half, h, DOOR_Z.0),
            &walls,
        );
        slab(
            world,
            Vec3::new(-half, 0.0, DOOR_Z.1),
            Vec3::new(half, h, w),
            &walls,
        );
        slab(
            world,
            Vec3::new(-half, DOOR_HEIGHT, DOOR_Z.0),
            Vec3::new(half, h, DOOR_Z.1),
            &walls,
        );
    } else {
        slab(
            world,
            Vec3::new(-half, 0.0, -w),
            Vec3::new(half, h, w),
            &walls,
        );
    }
    let a = spot(
        world,
        "lamp-a",
        Vec3::new(-half - ROOM_LENGTH / 2.0, 2.9, 0.0),
        A_LAMP,
    );
    world.entity_mut(a).insert(ToggleLight);
    spot(
        world,
        "lamp-b",
        Vec3::new(half + ROOM_LENGTH / 2.0, 2.9, 0.0),
        A_LAMP * B_LAMP_SHARE,
    );
    Scene {
        orbit: if door {
            two_rooms_door_orbit
        } else {
            two_rooms_orbit
        },
        fov: 1.0,
        assets: Vec::new(),
    }
}

fn two_rooms_orbit(frame: u32) -> Transform {
    if frame >= 1000 {
        Transform::from_xyz(-4.6, 1.5, 0.0).looking_to(Vec3::X, Vec3::Y)
    } else {
        Transform::from_xyz(4.6, 1.5, 0.0).looking_to(Vec3::NEG_X, Vec3::Y)
    }
}

/// Frames of the walk from B through the door into A.
const DOOR_WALK_FRAMES: f32 = 480.0;

fn two_rooms_door_orbit(frame: u32) -> Transform {
    if frame == 0 || frame >= 1000 {
        return two_rooms_orbit(frame);
    }
    let t = (frame as f32 / DOOR_WALK_FRAMES).min(1.0);
    let z = (DOOR_Z.0 + DOOR_Z.1) / 2.0 * (t * 8.0).min(1.0);
    Transform::from_xyz(4.6 - 9.2 * t, 1.5, z).looking_to(Vec3::NEG_X, Vec3::Y)
}

const ROOMS_PER_SIDE: usize = 6;
const ROOM_SIZE: f32 = 4.0;
const ROOM_DEPTH: f32 = 4.5;
const CORRIDOR_HALF: f32 = 1.2;
const CORRIDOR_OUTER: f32 = CORRIDOR_HALF + WALL;

/// Where the wall before room `i` starts along the corridor.
fn rooms_wall_x(i: usize) -> f32 {
    i as f32 * (ROOM_SIZE + WALL)
}

/// Twelve rooms of 4 by 4.5 m, six on each side of a 2.4 m corridor, 45 cm walls, a 1 m door per room, a spot in
/// every room and every 6 m of corridor: thirteen boxes, all in the frustum of the still camera at the corridor's
/// start. Generic; it is the layout council member 1 proposed for the probe cap.
fn rooms(world: &mut World) -> Scene {
    let walls = matte(world, 0.6);
    let floor = matte(world, 0.3);
    let end = rooms_wall_x(ROOMS_PER_SIDE);
    let reach = CORRIDOR_OUTER + ROOM_DEPTH + WALL;
    slab(
        world,
        Vec3::new(0.0, -0.2, -reach),
        Vec3::new(end + WALL, 0.0, reach),
        &floor,
    );
    slab(
        world,
        Vec3::new(0.0, ROOM_HEIGHT, -reach),
        Vec3::new(end + WALL, ROOM_HEIGHT + 0.2, reach),
        &walls,
    );
    for i in 0..=ROOMS_PER_SIDE {
        let x = rooms_wall_x(i);
        if i == 0 || i == ROOMS_PER_SIDE {
            slab(
                world,
                Vec3::new(x, 0.0, -reach),
                Vec3::new(x + WALL, ROOM_HEIGHT, reach),
                &walls,
            );
        } else {
            for side in [-1.0f32, 1.0] {
                let (z0, z1) = (side * CORRIDOR_OUTER, side * reach);
                slab(
                    world,
                    Vec3::new(x, 0.0, z0.min(z1)),
                    Vec3::new(x + WALL, ROOM_HEIGHT, z0.max(z1)),
                    &walls,
                );
            }
        }
    }
    for side in [-1.0f32, 1.0] {
        let (b0, b1) = (side * (reach - WALL), side * reach);
        slab(
            world,
            Vec3::new(WALL, 0.0, b0.min(b1)),
            Vec3::new(end, ROOM_HEIGHT, b0.max(b1)),
            &walls,
        );
        let (z0, z1) = (side * CORRIDOR_HALF, side * CORRIDOR_OUTER);
        let (z0, z1) = (z0.min(z1), z0.max(z1));
        for i in 0..ROOMS_PER_SIDE {
            let x0 = rooms_wall_x(i) + WALL;
            let door = x0 + ROOM_SIZE / 2.0;
            slab(
                world,
                Vec3::new(x0, 0.0, z0),
                Vec3::new(door - 0.5, ROOM_HEIGHT, z1),
                &walls,
            );
            slab(
                world,
                Vec3::new(door + 0.5, 0.0, z0),
                Vec3::new(x0 + ROOM_SIZE, ROOM_HEIGHT, z1),
                &walls,
            );
            slab(
                world,
                Vec3::new(door - 0.5, DOOR_HEIGHT, z0),
                Vec3::new(door + 0.5, ROOM_HEIGHT, z1),
                &walls,
            );
            let letter = if side > 0.0 { 'n' } else { 's' };
            spot(
                world,
                &format!("lamp-{letter}{i}"),
                Vec3::new(door, 2.9, side * (CORRIDOR_OUTER + ROOM_DEPTH / 2.0)),
                300_000.0,
            );
        }
    }
    let mut x = 3.0;
    let mut k = 0;
    while x < end {
        spot(
            world,
            &format!("lamp-corridor{k}"),
            Vec3::new(x, 2.9, 0.0),
            100_000.0,
        );
        x += 6.0;
        k += 1;
    }
    Scene {
        orbit: rooms_orbit,
        fov: 1.0,
        assets: Vec::new(),
    }
}

/// Frames of one walk down the corridor and back.
const ROOMS_WALK_FRAMES: f32 = 1440.0;

/// Walks the corridor and back, looking from side to side into the rooms. Frame 0 stands at the corridor's
/// start and looks down it.
fn rooms_orbit(frame: u32) -> Transform {
    let phase = frame as f32 / ROOMS_WALK_FRAMES * core::f32::consts::TAU;
    let x = 1.5 + (rooms_wall_x(ROOMS_PER_SIDE) - 3.0) * 0.5 * (1.0 - ops::cos(phase));
    let yaw = 1.1 * ops::sin(phase * 6.0);
    let direction = Vec3::new(ops::cos(yaw), 0.0, ops::sin(yaw));
    Transform::from_xyz(x, 1.6, 0.0).looking_to(direction, Vec3::Y)
}

/// Gate 2's asymmetric furnace (walls glowing at gate 2's frozen proportions of 1 cd/m², black otherwise),
/// with a white diffuse sphere at its centre that the bake leaves out. Pose 0 looks at the sphere from the
/// (+, +, +) corner, any other pose from the (−, −, −) corner, so every side of the sphere faces one of them.
fn furnace(world: &mut World) -> Scene {
    let axes = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    for (wall, axis) in axes.into_iter().enumerate() {
        let mesh = world
            .resource_mut::<Assets<Mesh>>()
            .add(traceable(Plane3d::new(-axis, Vec2::splat(FURNACE_HALF))));
        let material = world
            .resource_mut::<Assets<StandardMaterial>>()
            .add(test_material(0.0, FURNACE_RADIANCES[wall]));
        world.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_translation(axis * FURNACE_HALF),
        ));
    }
    let mesh = world
        .resource_mut::<Assets<Mesh>>()
        .add(traceable(Sphere::new(FURNACE_SPHERE).mesh().uv(128, 64)));
    let material = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(test_material(1.0, 0.0));
    world.spawn((Mesh3d(mesh), MeshMaterial3d(material), GiDynamic));
    Scene {
        orbit: furnace_orbit,
        fov: 0.7,
        assets: Vec::new(),
    }
}

fn furnace_orbit(frame: u32) -> Transform {
    let corner = if frame == 0 { 0.8 } else { -0.8 };
    Transform::from_translation(Vec3::splat(corner)).looking_at(Vec3::ZERO, Vec3::Y)
}

/// The gates' test material: diffuse as far as Bevy and Solari allow (`reflectance: 0`, roughness 1), base
/// colour and emission given in linear values.
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

/// The diffuser's spot, in lumens; the diffuser emits as much (gate 13).
const DIFFUSER_LUMENS: f32 = 100_000.0;
const DIFFUSER_HALF: f32 = 0.3;

/// A bake-only room of 5 by 4 m with a spot (`lamp`) under a ceiling diffuser authored as
/// `emissive = lumens / (π · area)` with the spot's lumens (gate 13).
fn diffuser(world: &mut World) -> Scene {
    let walls = matte(world, 0.6);
    let (x, z, h) = (2.5, 2.0, ROOM_HEIGHT);
    slab(
        world,
        Vec3::new(-x - WALL, -0.2, -z - WALL),
        Vec3::new(x + WALL, 0.0, z + WALL),
        &walls,
    );
    slab(
        world,
        Vec3::new(-x - WALL, h, -z - WALL),
        Vec3::new(x + WALL, h + 0.2, z + WALL),
        &walls,
    );
    for side in [-1.0f32, 1.0] {
        let (x0, x1) = (side * x, side * (x + WALL));
        slab(
            world,
            Vec3::new(x0.min(x1), 0.0, -z - WALL),
            Vec3::new(x0.max(x1), h, z + WALL),
            &walls,
        );
        let (z0, z1) = (side * z, side * (z + WALL));
        slab(
            world,
            Vec3::new(-x, 0.0, z0.min(z1)),
            Vec3::new(x, h, z0.max(z1)),
            &walls,
        );
    }
    let area = (2.0 * DIFFUSER_HALF) * (2.0 * DIFFUSER_HALF);
    let radiance = DIFFUSER_LUMENS / (core::f32::consts::PI * area);
    let mesh = world
        .resource_mut::<Assets<Mesh>>()
        .add(traceable(Plane3d::new(
            Vec3::NEG_Y,
            Vec2::splat(DIFFUSER_HALF),
        )));
    let material = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::linear_rgb(0.8, 0.8, 0.8),
            emissive: LinearRgba::rgb(radiance, radiance, radiance),
            perceptual_roughness: 0.9,
            ..default()
        });
    world.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, 2.85, 0.0),
        Diffuser,
    ));
    // Under the diffuser and pointing down, so the diffuser does not shade it.
    spot(world, "lamp", Vec3::new(0.0, 2.7, 0.0), DIFFUSER_LUMENS);
    Scene {
        orbit: |_| Transform::from_xyz(2.0, 1.6, 1.5).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y),
        fov: 1.0,
        assets: Vec::new(),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Run-time views

fn pin_intensity(mut volumes: Query<&mut IrradianceVolume>) {
    for mut volume in &mut volumes {
        if volume.intensity != 1.0 {
            volume.intensity = 1.0;
        }
    }
}

/// The frames `main.rs` captures, by name: the orbit's eight and `a`.
fn capture_name(args: &Args, frame: u32) -> Option<String> {
    let measured = args.warmup..args.warmup + args.frames;
    if frame == measured.end {
        return Some("a".into());
    }
    let step = (args.frames / crate::ORBIT_CAPTURES).max(1);
    (args.orbit
        && measured.contains(&frame)
        && (frame - args.warmup).is_multiple_of(step)
        && (frame - args.warmup) / step < crate::ORBIT_CAPTURES)
        .then(|| format!("orbit-{frame:05}"))
}

/// Writes the masks and predictions of the captured frames, and the `gi` section of `report.json`.
fn record(
    world: &mut World,
    mut triangles: Local<Option<Vec<MeshTriangles>>>,
    mut captures: Local<Vec<Value>>,
) {
    let run = world.resource::<Run>();
    if run.phase != Phase::Running {
        return;
    }
    let frame = run.frame;
    let args = world.resource::<Args>().clone();
    let baked = world.resource::<Baked>().clone();
    let Some(name) = capture_name(&args, frame) else {
        return;
    };
    let Ok((transform, frustum)) = world
        .query_filtered::<(&Transform, &Frustum), With<TestbedCamera>>()
        .single(world)
        .map(|(transform, frustum)| (*transform, frustum.clone()))
    else {
        return;
    };
    let fov = world.resource::<ActiveScene>().0.fov;
    let camera = PinholeCamera::new(transform, fov, args.res);
    let boxes: Vec<(String, Affine3A)> = world
        .query_filtered::<(&Name, &GlobalTransform), With<IrradianceVolume>>()
        .iter(world)
        .map(|(name, transform)| (name.to_string(), transform.affine()))
        .collect();
    let unit = Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5));
    let in_view: Vec<&str> = boxes
        .iter()
        .filter(|(_, affine)| frustum.intersects_obb(&unit, affine, true, false))
        .map(|(name, _)| name.as_str())
        .collect();
    let mut failures: Vec<String> = Vec::new();
    let mut entry = json!({
        "capture": format!("{name}.png"),
        "frame": frame,
        "boxes": boxes.len(),
        "boxes_in_view": in_view.len(),
        "boxes_in_view_names": in_view,
    });

    if baked.coverage {
        let triangles = triangles.get_or_insert_with(|| collect_triangles(world));
        let hits = trace(&camera, triangles);
        let (mask, inside) = coverage_mask(&camera, &hits, &boxes);
        let file = format!("mask-{name}.png");
        failures.extend(save_mask(&args.out.join(&file), args.res, &mask).err());
        entry["mask"] = json!(file);
        entry["mask_pixels"] = json!(inside);
    }
    if args.scene.starts_with("two-rooms") {
        let half = two_rooms_wall(&args.scene) / 2.0;
        let x = if transform.translation.x > 0.0 {
            half
        } else {
            -half
        };
        let corners = [
            Vec3::new(x, WALL_Y.0, WALL_Z.0),
            Vec3::new(x, WALL_Y.1, WALL_Z.0),
            Vec3::new(x, WALL_Y.0, WALL_Z.1),
            Vec3::new(x, WALL_Y.1, WALL_Z.1),
        ];
        if let Some(rect) = camera.project_rect(&corners) {
            entry["wall_region"] = json!(rect);
        }
    }
    if args.scene == "gi-furnace" && name == "a" {
        match predict_furnace(&camera, baked.load.as_deref()) {
            Ok(prediction) => {
                let mut counts = Map::new();
                for (file, image) in &prediction.images {
                    failures.extend(save_rgb(&args.out.join(file), args.res, image).err());
                }
                for (face, mask) in prediction.masks.iter().enumerate() {
                    let file = format!("mask-{}.png", FACE_FILES[face]);
                    failures.extend(save_mask(&args.out.join(&file), args.res, mask).err());
                    counts.insert(
                        FACE_FILES[face].into(),
                        json!(mask.iter().filter(|m| **m).count()),
                    );
                }
                entry["prediction"] = json!({
                    "files": prediction.images.iter().map(|(file, _)| file.clone()).collect::<Vec<_>>(),
                    "region_pixels": counts,
                    "facing": FACING,
                    "eroded_by": ERODE,
                    "ev100": PREDICTED_EV100,
                });
            }
            Err(error) => world
                .resource_mut::<Report>()
                .failures
                .push(format!("gate 6 prediction: {error}")),
        }
    }
    captures.push(entry);
    world.resource_mut::<Report>().failures.extend(failures);

    if name == "a" {
        let status = format!("{:?}", *world.resource::<BakedGiStatus>());
        let device = world.resource::<RenderDevice>();
        let (limits, features) = (device.limits(), device.features());
        let section = json!({
            "dir": baked.load,
            "status": status,
            "pinned_intensity": baked.pin_intensity || baked.coverage,
            "coverage_view": baked.coverage,
            "lights_off": baked.lights_off,
            "without_toggle_light": baked.without_toggle_light,
            "binding_arrays_are_usable":
                bevy::graphics::baked::binding_arrays_are_usable(&limits, features),
            "max_view_light_probes": bevy::pbr::MAX_VIEW_LIGHT_PROBES,
            "limits": {
                "max_binding_array_elements_per_shader_stage": limits.max_binding_array_elements_per_shader_stage,
                "max_binding_array_sampler_elements_per_shader_stage":
                    limits.max_binding_array_sampler_elements_per_shader_stage,
                "max_sampled_textures_per_shader_stage": limits.max_sampled_textures_per_shader_stage,
                "max_storage_buffers_per_shader_stage": limits.max_storage_buffers_per_shader_stage,
            },
            "features": {
                "texture_binding_array": features.contains(bevy::render::settings::WgpuFeatures::TEXTURE_BINDING_ARRAY),
                "sampled_texture_and_storage_buffer_array_non_uniform_indexing": features.contains(
                    bevy::render::settings::WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                ),
            },
            "captures": *captures,
        });
        world
            .resource_mut::<Report>()
            .sections
            .insert("gi".into(), section);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The CPU's view of a frame

/// The testbed camera as a pinhole: one ray through each pixel's centre.
struct PinholeCamera {
    transform: Transform,
    tan_half: f32,
    aspect: f32,
    size: UVec2,
}

impl PinholeCamera {
    fn new(transform: Transform, fov: f32, size: UVec2) -> Self {
        Self {
            transform,
            tan_half: ops::tan(fov / 2.0),
            aspect: size.x as f32 / size.y as f32,
            size,
        }
    }

    fn ray(&self, x: u32, y: u32) -> Vec3 {
        let ndc_x = 2.0 * (x as f32 + 0.5) / self.size.x as f32 - 1.0;
        let ndc_y = 1.0 - 2.0 * (y as f32 + 0.5) / self.size.y as f32;
        let local = Vec3::new(
            ndc_x * self.tan_half * self.aspect,
            ndc_y * self.tan_half,
            -1.0,
        );
        (self.transform.rotation * local).normalize()
    }

    /// The pixel rectangle `[x0, y0, x1, y1]` (ends excluded) inside the projection of `points`, all of which
    /// must be in front of the camera.
    fn project_rect(&self, points: &[Vec3]) -> Option<[u32; 4]> {
        let inverse = self.transform.compute_affine().inverse();
        let mut pixels = Vec::new();
        for point in points {
            let p = inverse.transform_point3(*point);
            if p.z >= 0.0 {
                return None;
            }
            let ndc = Vec2::new(
                p.x / (-p.z * self.tan_half * self.aspect),
                p.y / (-p.z * self.tan_half),
            );
            pixels.push(Vec2::new(
                (ndc.x + 1.0) / 2.0 * self.size.x as f32,
                (1.0 - ndc.y) / 2.0 * self.size.y as f32,
            ));
        }
        let min = pixels.iter().fold(Vec2::MAX, |a, b| a.min(*b));
        let max = pixels.iter().fold(Vec2::MIN, |a, b| a.max(*b));
        let clamp = |v: f32, limit: u32| v.clamp(0.0, limit as f32) as u32;
        Some([
            clamp(min.x.ceil(), self.size.x),
            clamp(min.y.ceil(), self.size.y),
            clamp(max.x.floor(), self.size.x),
            clamp(max.y.floor(), self.size.y),
        ])
    }
}

/// One mesh's triangles in world space, with their bounds.
struct MeshTriangles {
    min: Vec3,
    max: Vec3,
    triangles: Vec<[Vec3; 3]>,
}

fn collect_triangles(world: &mut World) -> Vec<MeshTriangles> {
    let entities: Vec<(Handle<Mesh>, GlobalTransform)> = world
        .query::<(&Mesh3d, &GlobalTransform)>()
        .iter(world)
        .map(|(mesh, transform)| (mesh.0.clone(), *transform))
        .collect();
    let meshes = world.resource::<Assets<Mesh>>();
    let mut out = Vec::new();
    for (handle, transform) in entities {
        let Some(mesh) = meshes.get(&handle) else {
            continue;
        };
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            continue;
        };
        let positions: Vec<Vec3> = positions
            .iter()
            .map(|p| transform.transform_point(Vec3::from_array(*p)))
            .collect();
        let indices: Vec<usize> = match mesh.indices() {
            Some(Indices::U16(i)) => i.iter().map(|i| *i as usize).collect(),
            Some(Indices::U32(i)) => i.iter().map(|i| *i as usize).collect(),
            None => (0..positions.len()).collect(),
        };
        let triangles: Vec<[Vec3; 3]> = indices
            .chunks_exact(3)
            .map(|t| [positions[t[0]], positions[t[1]], positions[t[2]]])
            .collect();
        let min = positions.iter().fold(Vec3::MAX, |a, b| a.min(*b));
        let max = positions.iter().fold(Vec3::MIN, |a, b| a.max(*b));
        out.push(MeshTriangles {
            min,
            max,
            triangles,
        });
    }
    out
}

/// The nearest hit of a ray: the distance and which mesh, or `None`.
type Hit = Option<(f32, usize)>;

fn ray_box(origin: Vec3, inverse: Vec3, min: Vec3, max: Vec3) -> bool {
    let t0 = (min - origin) * inverse;
    let t1 = (max - origin) * inverse;
    let near = t0.min(t1).max_element();
    let far = t0.max(t1).min_element();
    far >= near.max(0.0)
}

/// Möller–Trumbore, both sides.
fn ray_triangle(origin: Vec3, direction: Vec3, [a, b, c]: &[Vec3; 3]) -> Option<f32> {
    let (e1, e2) = (*b - *a, *c - *a);
    let p = direction.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / det;
    let s = origin - *a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = direction.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 1e-4).then_some(t)
}

/// Every pixel's nearest hit, rows split over the machine's threads.
fn trace(camera: &PinholeCamera, meshes: &[MeshTriangles]) -> Vec<Hit> {
    let (width, height) = (camera.size.x as usize, camera.size.y as usize);
    let mut hits = vec![None; width * height];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let rows_per = height.div_ceil(threads);
    let origin = camera.transform.translation;
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in hits.chunks_mut(rows_per * width).enumerate() {
            scope.spawn(move || {
                for (i, hit) in chunk.iter_mut().enumerate() {
                    let pixel = chunk_index * rows_per * width + i;
                    let direction = camera.ray((pixel % width) as u32, (pixel / width) as u32);
                    let inverse = direction.recip();
                    let mut best: Hit = None;
                    for (index, mesh) in meshes.iter().enumerate() {
                        if !ray_box(origin, inverse, mesh.min, mesh.max) {
                            continue;
                        }
                        for triangle in &mesh.triangles {
                            if let Some(t) = ray_triangle(origin, direction, triangle)
                                && best.is_none_or(|(b, _)| t < b)
                            {
                                best = Some((t, index));
                            }
                        }
                    }
                    *hit = best;
                }
            });
        }
    });
    hits
}

/// The pixels whose surface lies inside a baked box by [`MASK_MARGIN`], eroded by one pixel against neighbours
/// on another mesh or outside: those are the pixels the coverage gate reads. Also returns their count.
fn coverage_mask(
    camera: &PinholeCamera,
    hits: &[Hit],
    boxes: &[(String, Affine3A)],
) -> (Vec<bool>, usize) {
    let (width, height) = (camera.size.x as i32, camera.size.y as i32);
    let inverses: Vec<(Affine3A, Vec3)> = boxes
        .iter()
        .map(|(_, affine)| {
            let scale = affine.matrix3.to_cols_array_2d();
            let scale = Vec3::new(
                Vec3::from_array(scale[0]).length(),
                Vec3::from_array(scale[1]).length(),
                Vec3::from_array(scale[2]).length(),
            );
            (affine.inverse(), Vec3::splat(0.5) - MASK_MARGIN / scale)
        })
        .collect();
    let inside: Vec<bool> = hits
        .iter()
        .enumerate()
        .map(|(pixel, hit)| {
            let Some((t, _)) = hit else {
                return false;
            };
            let direction =
                camera.ray((pixel as i32 % width) as u32, (pixel as i32 / width) as u32);
            let point = camera.transform.translation + direction * *t;
            inverses
                .iter()
                .any(|(inverse, limit)| inverse.transform_point3(point).abs().cmplt(*limit).all())
        })
        .collect();
    let mut mask = vec![false; inside.len()];
    let mut count = 0;
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let mesh = hits[i].map(|(_, m)| m);
            let keep = inside[i]
                && [(-1, 0), (1, 0), (0, -1), (0, 1)].iter().all(|(dx, dy)| {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= width || ny >= height {
                        return true;
                    }
                    let n = (ny * width + nx) as usize;
                    inside[n] && hits[n].map(|(_, m)| m) == mesh
                });
            mask[i] = keep;
            count += usize::from(keep);
        }
    }
    (mask, count)
}

fn save_rgba(path: &Path, size: UVec2, data: Vec<u8>) -> Result<(), String> {
    let image = Image::new(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        // The screenshots' format: the bytes are written as they are.
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD,
    );
    image
        .try_into_dynamic()
        .map_err(|error| error.to_string())
        .and_then(|image| {
            image
                .to_rgb8()
                .save(path)
                .map_err(|error| error.to_string())
        })
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn save_mask(path: &Path, size: UVec2, mask: &[bool]) -> Result<(), String> {
    let data = mask
        .iter()
        .flat_map(|m| if *m { [255; 4] } else { [0, 0, 0, 255] })
        .collect();
    save_rgba(path, size, data)
}

fn save_rgb(path: &Path, size: UVec2, pixels: &[[u8; 3]]) -> Result<(), String> {
    let data = pixels
        .iter()
        .flat_map(|[r, g, b]| [*r, *g, *b, 255])
        .collect();
    save_rgba(path, size, data)
}

// ---------------------------------------------------------------------------------------------------------------
// Gate 6: the furnace frame predicted from the bake

/// The six sides in `FACE_NAMES` order, as file names.
const FACE_FILES: [&str; 6] = ["px", "nx", "py", "ny", "pz", "nz"];

struct Prediction {
    images: Vec<(String, Vec<[u8; 3]>)>,
    masks: [Vec<bool>; 6],
}

/// A baked volume read back from its KTX2 file, sampled as `irradiance_volume.wesl` samples it. The same
/// reading as `gi_bake`'s checks, which gates 1 and 2 validated.
struct Volume {
    size: UVec3,
    texels: Vec<Vec3>,
    light_from_world: Affine3A,
}

impl Volume {
    fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join("manifest.gi.ron");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let manifest: BakedGiManifest = ron::de::from_str(&text).map_err(|e| e.to_string())?;
        let [region] = manifest.regions.as_slice() else {
            return Err("the furnace bake must have exactly one region".into());
        };
        let bytes = std::fs::read(dir.join(&region.file)).map_err(|e| e.to_string())?;
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
            light_from_world: region.transform().compute_affine().inverse(),
        })
    }

    fn texel(&self, p: IVec3) -> Vec3 {
        let p = p.clamp(IVec3::ZERO, self.size.as_ivec3() - 1).as_uvec3();
        self.texels[(p.x + self.size.x * (p.y + self.size.y * p.z)) as usize]
    }

    /// A linear-filtered, clamp-to-edge fetch at normalised coordinates.
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

    /// `irradiance_volume.wesl:62-74` at `world` for `normal`; `swap` reads that axis's two sides swapped.
    fn irradiance(&self, world: Vec3, normal: Vec3, swap: Option<usize>) -> Vec3 {
        let resolution = self.size.as_vec3() / Vec3::new(1.0, 2.0, 3.0);
        let unit = self.light_from_world.transform_point3(world);
        let stp = ((unit + 0.5) * resolution).clamp(Vec3::splat(0.5), resolution - 0.5);
        let uvw = stp / self.size.as_vec3();
        let nn = normal * normal;
        let mut total = Vec3::ZERO;
        for axis in 0..3 {
            let negative = (normal[axis] < 0.0) != (swap == Some(axis));
            let offset = Vec3::new(0.0, if negative { 0.5 } else { 0.0 }, axis as f32 / 3.0);
            total += self.sample(uvw + offset) * nn[axis];
        }
        total
    }
}

/// IEEE 754 half to single precision.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}

/// What the display gets for linear light `c` with [`Tonemapping::None`]: clamped, sRGB-encoded, 8 bits.
fn encode(c: Vec3) -> [u8; 3] {
    c.to_array().map(|v| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    })
}

/// The furnace frame as the shader would draw the sphere from the bake: albedo 1 times the volume at each
/// pixel's surface point and normal, times the exposure. Black off the sphere.
fn predict_furnace(camera: &PinholeCamera, dir: Option<&Path>) -> Result<Prediction, String> {
    let volume = Volume::load(dir.ok_or("no --baked-gi")?)?;
    let exposure = Exposure {
        ev100: PREDICTED_EV100,
    }
    .exposure();
    let (width, height) = (camera.size.x as i32, camera.size.y as i32);
    let origin = camera.transform.translation;
    let mut surface: Vec<Option<(Vec3, Vec3)>> = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let d = camera.ray(x as u32, y as u32);
            // |o + t d|² = r² with |d| = 1.
            let b = origin.dot(d);
            let c = origin.length_squared() - FURNACE_SPHERE * FURNACE_SPHERE;
            let disc = b * b - c;
            surface.push((disc >= 0.0).then(|| {
                let p = origin + d * (-b - disc.sqrt());
                (p, p.normalize())
            }));
        }
    }
    let mut images = Vec::new();
    for (file, swap) in [
        ("predicted.png", None),
        ("predicted-swap-x.png", Some(0)),
        ("predicted-swap-y.png", Some(1)),
        ("predicted-swap-z.png", Some(2)),
    ] {
        let pixels = surface
            .iter()
            .map(|hit| match hit {
                Some((p, n)) => encode(volume.irradiance(*p, *n, swap) * exposure),
                None => [0; 3],
            })
            .collect();
        images.push((file.to_owned(), pixels));
    }
    let face = |hit: &Option<(Vec3, Vec3)>| {
        let (_, n) = (*hit)?;
        (0..3).find_map(|axis| {
            (n[axis].abs() >= FACING).then_some(2 * axis + usize::from(n[axis] < 0.0))
        })
    };
    let faces: Vec<Option<usize>> = surface.iter().map(face).collect();
    let masks = core::array::from_fn(|wanted| {
        let mut mask = vec![false; faces.len()];
        for y in 0..height {
            for x in 0..width {
                mask[(y * width + x) as usize] = (-ERODE..=ERODE).all(|dy| {
                    (-ERODE..=ERODE).all(|dx| {
                        let (nx, ny) = (x + dx, y + dy);
                        nx >= 0
                            && ny >= 0
                            && nx < width
                            && ny < height
                            && faces[(ny * width + nx) as usize] == Some(wanted)
                    })
                });
            }
        }
        mask
    });
    Ok(Prediction { images, masks })
}

// ---------------------------------------------------------------------------------------------------------------
// Baking

#[cfg(feature = "bevy_baked_gi_bake")]
mod bake {
    use super::Baked;
    use bevy::{
        app::PluginsState,
        baked_gi::bake::{GiBakePlugin, GiBakeSettings},
        prelude::*,
        window::ExitCondition,
        winit::WinitPlugin,
    };
    use std::path::PathBuf;

    /// Bakes `scene` with `baked`'s flags into `out` and ends the process.
    pub fn run(baked: Baked, scene: String, out: PathBuf) -> AppExit {
        println!("baking {scene} to {}", out.display());
        let settings = GiBakeSettings {
            output: out,
            samples: baked
                .bake_samples
                .unwrap_or(GiBakeSettings::default().samples),
            adaptive: if baked.bake_samples.is_some() {
                None
            } else {
                GiBakeSettings::default().adaptive
            },
            seed: baked.bake_seed,
            ..default()
        };
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
                let spawned = crate::scenes::spawn(&scene, world)
                    .or_else(|| Baked::scene(&scene, world))
                    .is_some();
                let prepared = if spawned {
                    baked.prepare_level(world, &scene, true)
                } else {
                    Err(format!("there is no scene called {scene}"))
                };
                if let Err(error) = prepared {
                    eprintln!("gfx_testbed: {error}");
                    std::process::exit(2);
                }
            })
            .set_runner(run_until_baked)
            .run()
    }

    /// `gi_bake`'s runner: the process ends as soon as the bake asks to exit, because one bake there hung in
    /// the app's shutdown after its files were written (`examples/graphics/gi_bake.rs`).
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
}
