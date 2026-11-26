//! The testbed's mirror scene: what a flat mirror must show on every rung.
//!
//! `mirrors`: a room 8 m by 6 m and 3 m high with a wall mirror straight ahead of the camera and a long mirror
//! on the right wall, seen at a grazing angle. The wall mirror shows what stands behind the camera (a red and
//! a blue box, a striped panel), which screen-space reflections cannot find; the side mirror shows the left
//! half of the room, mostly off screen. Lit by a ceiling point light and a spot, both with shadow maps; nothing
//! comes from a sky. Deterministic: no time, no random numbers.

use crate::Scene;
use bevy::{mesh::VertexAttributeValues, prelude::*};
use core::f32::consts::{FRAC_PI_2, TAU};

/// Spawns the scene called `name`, or returns `None` when this file has no such scene.
pub fn spawn(name: &str, world: &mut World) -> Option<Scene> {
    let scene = match name {
        "mirrors" => world.run_system_cached(mirrors),
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

/// Frames of one sweep of the camera: from left to right and back, 1.2 m wide.
const MIRRORS_ORBIT_FRAMES: f32 = 480.0;

fn mirrors_orbit(frame: u32) -> Transform {
    let angle = frame as f32 * (TAU / MIRRORS_ORBIT_FRAMES);
    Transform::from_xyz(0.6 * ops::sin(angle), 1.6, 1.2)
        .looking_at(Vec3::new(0.3, 1.4, -3.0), Vec3::Y)
}

fn mirrors(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) -> Scene {
    // A mirror: a smooth, white, fully metallic surface. The ladder finds it by its material and its flat mesh.
    let mirror = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.95),
        metallic: 1.0,
        perceptual_roughness: 0.02,
        ..default()
    });
    let mut matte = |color: Color| {
        materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.8,
            ..default()
        })
    };
    let wall = matte(Color::srgb(0.75, 0.73, 0.7));
    let floor = matte(Color::srgb(0.45, 0.4, 0.35));
    let red = matte(Color::srgb(0.8, 0.12, 0.08));
    let blue = matte(Color::srgb(0.1, 0.25, 0.85));
    let green = matte(Color::srgb(0.15, 0.65, 0.2));
    let yellow = matte(Color::srgb(0.9, 0.75, 0.1));
    let dark = matte(Color::srgb(0.05, 0.05, 0.05));

    // The room: floor, ceiling and four walls, facing in.
    let (half_x, half_z, height) = (4.0, 3.0, 3.0);
    let surfaces = [
        (Vec3::Y, Vec2::new(half_x, half_z), Vec3::ZERO, floor.clone()),
        (Vec3::NEG_Y, Vec2::new(half_x, half_z), Vec3::Y * height, wall.clone()),
        (Vec3::Z, Vec2::new(half_x, height / 2.0), Vec3::new(0.0, height / 2.0, -half_z), wall.clone()),
        (Vec3::NEG_Z, Vec2::new(half_x, height / 2.0), Vec3::new(0.0, height / 2.0, half_z), wall.clone()),
        (Vec3::X, Vec2::new(height / 2.0, half_z), Vec3::new(-half_x, height / 2.0, 0.0), wall.clone()),
        (Vec3::NEG_X, Vec2::new(height / 2.0, half_z), Vec3::new(half_x, height / 2.0, 0.0), wall.clone()),
    ];
    for (normal, half_size, position, material) in surfaces {
        commands.spawn((
            Mesh3d(meshes.add(traceable(Plane3d::new(normal, half_size)))),
            MeshMaterial3d(material),
            Transform::from_translation(position),
        ));
    }

    // The wall mirror ahead of the camera, 2.4 m by 1.6 m, 1 cm off the back wall, in a dark frame.
    commands.spawn((
        Mesh3d(meshes.add(traceable(Plane3d::new(Vec3::Z, Vec2::new(1.2, 0.8))))),
        MeshMaterial3d(mirror.clone()),
        Transform::from_xyz(0.3, 1.5, -half_z + 0.01),
    ));
    commands.spawn((
        Mesh3d(meshes.add(traceable(Cuboid::new(2.6, 1.8, 0.01)))),
        MeshMaterial3d(dark.clone()),
        Transform::from_xyz(0.3, 1.5, -half_z + 0.004),
    ));
    // The side mirror on the right wall, 3 m long, seen at a grazing angle.
    commands.spawn((
        Mesh3d(meshes.add(traceable(Plane3d::new(Vec3::NEG_X, Vec2::new(0.8, 1.5))))),
        MeshMaterial3d(mirror),
        Transform::from_xyz(half_x - 0.01, 1.3, -1.0),
    ));

    // Behind the camera: only the mirrors show these.
    let cube = meshes.add(traceable(Cuboid::from_length(0.8)));
    commands.spawn((
        Mesh3d(cube.clone()),
        MeshMaterial3d(red.clone()),
        Transform::from_xyz(-1.0, 0.4, 2.3).with_rotation(Quat::from_rotation_y(0.4)),
    ));
    commands.spawn((
        Mesh3d(meshes.add(traceable(Cuboid::new(0.6, 1.8, 0.6)))),
        MeshMaterial3d(blue),
        Transform::from_xyz(1.4, 0.9, 2.2),
    ));
    // A striped panel on the wall behind the camera: sharp edges show whether the reflection is sharp and
    // lands where it should.
    let stripe = meshes.add(traceable(Plane3d::new(Vec3::NEG_Z, Vec2::new(0.12, 0.6))));
    for i in 0..7 {
        commands.spawn((
            Mesh3d(stripe.clone()),
            MeshMaterial3d(if i % 2 == 0 { yellow.clone() } else { dark.clone() }),
            Transform::from_xyz(-0.72 + i as f32 * 0.24, 1.6, half_z - 0.01),
        ));
    }
    // The left half of the room, off screen at the start: the side mirror shows it.
    commands.spawn((
        Mesh3d(meshes.add(traceable(Cuboid::new(0.7, 1.2, 0.7)))),
        MeshMaterial3d(green),
        Transform::from_xyz(-3.0, 0.6, -0.5).with_rotation(Quat::from_rotation_y(0.3)),
    ));
    commands.spawn((
        Mesh3d(meshes.add(traceable(Sphere::new(0.45).mesh().uv(48, 24)))),
        MeshMaterial3d(yellow),
        Transform::from_xyz(-2.6, 0.45, -2.2),
    ));
    // Something in front of the mirror, seen both directly and reflected.
    commands.spawn((
        Mesh3d(cube),
        MeshMaterial3d(red),
        Transform::from_xyz(0.9, 0.4, -1.8).with_rotation(Quat::from_rotation_y(-0.5)),
    ));
    // A floor tile pattern, so the floor seen in the mirrors has detail.
    let mut tiles = traceable(Plane3d::new(Vec3::Y, Vec2::new(1.5, 1.5)));
    if let Some(VertexAttributeValues::Float32x3(positions)) = tiles.attribute_mut(Mesh::ATTRIBUTE_POSITION) {
        for position in positions {
            position[1] = 0.002;
        }
    }
    commands.spawn((
        Mesh3d(meshes.add(tiles)),
        MeshMaterial3d(matte(Color::srgb(0.85, 0.85, 0.85))),
        Transform::from_xyz(-1.5, 0.0, 0.5).with_rotation(Quat::from_rotation_y(FRAC_PI_2 / 2.0)),
    ));

    commands.spawn((
        PointLight {
            intensity: 300_000.0,
            range: 10.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-1.0, 2.7, 0.5),
    ));
    commands.spawn((
        SpotLight {
            intensity: 900_000.0,
            range: 10.0,
            inner_angle: 0.6,
            outer_angle: 0.9,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(1.5, 2.9, 1.5).looking_to(Vec3::NEG_Y, Vec3::Z),
    ));

    Scene {
        orbit: mirrors_orbit,
        fov: 1.1,
        assets: Vec::new(),
    }
}
