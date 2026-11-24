//! The testbed's scenes. Each is built in code or from `assets/`, so a run needs no download, and each is the
//! same on every run: nothing here depends on time or on an unseeded random number.
//!
//! A scene spawns plain `Mesh3d` entities whose meshes already have what Solari needs (positions, normals, UVs,
//! tangents, indices), and lights with shadow maps as a rasterized game would set them. `main.rs` adapts both to
//! the rendering mode.

use crate::{Scene, ToggleLight};
use bevy::{
    image::{ImageAddressMode, ImageLoaderSettings},
    mesh::VertexAttributeValues,
    prelude::*,
};
use chacha20::ChaCha8Rng;
use core::f32::consts::TAU;
use rand::{RngExt, SeedableRng};

/// Spawns the scene called `name`, or returns `None` when this file has no such scene.
pub fn spawn(name: &str, world: &mut World) -> Option<Scene> {
    let scene = match name {
        "interior" => world.run_system_cached(interior),
        "many-lights" => world.run_system_cached(many_lights),
        _ => return None,
    };
    Some(scene.expect("scene systems take only resources the default plugins provide"))
}

/// A mesh Solari can trace: it builds an acceleration structure only for meshes that carry tangents.
fn traceable(mesh: impl Into<Mesh>) -> Mesh {
    mesh.into()
        .with_generated_tangents()
        .expect("primitive meshes have the attributes tangent generation needs")
}

/// Frames of one full turn of the `interior` orbit: half a degree per frame, a brisk walk around the room.
const INTERIOR_ORBIT_FRAMES: f32 = 720.0;

fn interior_orbit(frame: u32) -> Transform {
    let angle = frame as f32 * (TAU / INTERIOR_ORBIT_FRAMES);
    Transform::from_xyz(3.0 * ops::sin(angle), 1.6, 3.0 * ops::cos(angle))
        .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y)
}

/// A closed room, 14 m by 8 m and 3 m high, lit only by its fixtures: a ceiling spot, a ceiling rect light and a
/// point light, as in the rooms of the game this ladder is for. Nothing comes from a sky, so every lit pixel is
/// direct light from a fixture or light bounced off a wall.
///
/// The room and its three lights are council member 3's `m3_local_lights` example, which was run under Solari,
/// the reference tracer and the rasterizer; the furniture is added for the denoisers and upscalers to have a
/// mirror-like surface, a textured floor and a corner that only bounced light reaches.
fn interior(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) -> Scene {
    let mut matte = |color: Color, roughness: f32| {
        materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: roughness,
            ..default()
        })
    };
    let grey = matte(Color::srgb(0.7, 0.7, 0.7), 0.8);
    let red = matte(Color::srgb(0.8, 0.15, 0.1), 0.6);
    let green = matte(Color::srgb(0.15, 0.6, 0.2), 0.9);

    // The floor's checker repeats once per two metres: detail for temporal filters to keep or smear.
    let checker: Handle<Image> = asset_server
        .load_builder()
        .with_settings::<ImageLoaderSettings>(|settings| {
            settings
                .sampler
                .get_or_init_descriptor()
                .set_address_mode(ImageAddressMode::Repeat);
        })
        .load("textures/uv_checker_bw.png");
    let floor_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.75, 0.75, 0.75),
        base_color_texture: Some(checker.clone()),
        perceptual_roughness: 0.45,
        ..default()
    });
    let mut floor = traceable(Plane3d::new(Vec3::Y, Vec2::new(7.0, 4.0)));
    match floor.attribute_mut(Mesh::ATTRIBUTE_UV_0) {
        Some(VertexAttributeValues::Float32x2(uvs)) => {
            for uv in uvs {
                *uv = [uv[0] * 7.0, uv[1] * 4.0];
            }
        }
        _ => unreachable!("a plane mesh has two-component UVs"),
    }
    commands.spawn((Mesh3d(meshes.add(floor)), MeshMaterial3d(floor_material)));

    let walls = [
        (Vec3::NEG_Y, Vec2::new(7.0, 4.0), Vec3::new(0.0, 3.0, 0.0)),
        (Vec3::Z, Vec2::new(7.0, 1.5), Vec3::new(0.0, 1.5, -4.0)),
        (Vec3::NEG_Z, Vec2::new(7.0, 1.5), Vec3::new(0.0, 1.5, 4.0)),
        (Vec3::X, Vec2::new(1.5, 4.0), Vec3::new(-7.0, 1.5, 0.0)),
        (Vec3::NEG_X, Vec2::new(1.5, 4.0), Vec3::new(7.0, 1.5, 0.0)),
    ];
    for (normal, half_size, position) in walls {
        commands.spawn((
            Mesh3d(meshes.add(traceable(Plane3d::new(normal, half_size)))),
            MeshMaterial3d(grey.clone()),
            Transform::from_translation(position),
        ));
    }

    let cube = meshes.add(traceable(Cuboid::from_length(0.8)));
    for x in [-4.0, 0.0, 4.0] {
        commands.spawn((
            Mesh3d(cube.clone()),
            MeshMaterial3d(red.clone()),
            Transform::from_xyz(x + 0.6, 0.4, -1.0).with_rotation(Quat::from_rotation_y(0.5)),
        ));
    }
    // Everything the orbit passes over stays under the camera's 1.6 m.
    commands.spawn((
        Mesh3d(meshes.add(traceable(Sphere::new(0.5).mesh().uv(64, 32)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.9, 0.9),
            metallic: 1.0,
            perceptual_roughness: 0.2,
            ..default()
        })),
        Transform::from_xyz(-1.0, 0.5, 0.2),
    ));
    commands.spawn((
        Mesh3d(meshes.add(traceable(Cuboid::new(0.5, 1.0, 0.5)))),
        MeshMaterial3d(green),
        Transform::from_xyz(1.6, 0.5, 0.6).with_rotation(Quat::from_rotation_y(-0.3)),
    ));
    commands.spawn((
        Mesh3d(meshes.add(traceable(Cuboid::new(0.6, 3.0, 0.6)))),
        MeshMaterial3d(grey.clone()),
        Transform::from_xyz(-5.5, 1.5, -2.5),
    ));
    // A partition that hides the far right corner from the rect light: that corner is lit by bounces alone.
    commands.spawn((
        Mesh3d(meshes.add(traceable(Cuboid::new(0.2, 3.0, 3.0)))),
        MeshMaterial3d(grey),
        Transform::from_xyz(5.6, 1.5, -2.5),
    ));

    commands.spawn((
        PointLight {
            intensity: 400_000.0,
            range: 12.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-4.0, 2.0, 0.0),
    ));
    commands.spawn((
        SpotLight {
            intensity: 1_500_000.0,
            range: 12.0,
            inner_angle: 0.5,
            outer_angle: 0.7,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 2.9, 0.0).looking_to(Vec3::NEG_Y, Vec3::Z),
        // The fixture in the middle of every view: switching it off changes most of the frame.
        ToggleLight,
    ));
    commands.spawn((
        RectLight {
            intensity: 600_000.0,
            width: 1.2,
            height: 0.6,
            range: 12.0,
            ..default()
        },
        Transform::from_xyz(4.0, 2.9, 0.0).looking_to(Vec3::NEG_Y, Vec3::Z),
    ));

    Scene {
        orbit: interior_orbit,
        fov: 1.2,
        assets: vec![checker.untyped()],
    }
}

/// Frames of one full turn of the `many-lights` orbit.
const MANY_LIGHTS_ORBIT_FRAMES: f32 = 1440.0;

fn many_lights_orbit(frame: u32) -> Transform {
    let angle = frame as f32 * (TAU / MANY_LIGHTS_ORBIT_FRAMES);
    Transform::from_xyz(42.0 * ops::sin(angle), 16.0, 42.0 * ops::cos(angle))
        .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y)
}

/// An open field of 3000 boxes under a 16 by 16 grid of coloured point lights: the stress case for choosing
/// which light to sample (traced) and for light clustering (raster).
///
/// The lights are analytic so that the rasterizer and the tracers light the same scene; none has a shadow map,
/// because 256 cube shadow maps are not something a rasterized game would ask for.
fn many_lights(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) -> Scene {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    commands.spawn((
        Mesh3d(meshes.add(traceable(Plane3d::new(Vec3::Y, Vec2::splat(70.0))))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.5, 0.5),
            perceptual_roughness: 0.9,
            ..default()
        })),
    ));

    let cube = meshes.add(traceable(Cuboid::default()));
    for _ in 0..3000 {
        let scale = Vec3::new(
            rng.random_range(0.3..=2.0),
            rng.random_range(0.3..=2.0),
            rng.random_range(0.3..=2.0),
        );
        commands.spawn((
            Mesh3d(cube.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(rng.random(), rng.random(), rng.random()),
                perceptual_roughness: rng.random_range(0.1..=1.0),
                ..default()
            })),
            Transform::from_xyz(
                rng.random_range(-60.0..=60.0),
                scale.y / 2.0,
                rng.random_range(-60.0..=60.0),
            )
            .with_scale(scale),
        ));
    }

    for x in 0..16 {
        for z in 0..16 {
            let mut light = commands.spawn((
                PointLight {
                    color: Color::hsl(rng.random_range(0.0..360.0), 0.9, 0.6),
                    intensity: 300_000.0,
                    range: 14.0,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_xyz((x as f32 - 7.5) * 7.0, 3.0, (z as f32 - 7.5) * 7.0),
            ));
            // One light beside the centre of the field, in front of the still camera.
            if (x, z) == (8, 8) {
                light.insert(ToggleLight);
            }
        }
    }

    Scene {
        orbit: many_lights_orbit,
        fov: 1.0,
        assets: Vec::new(),
    }
}
