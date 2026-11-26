use super::*;
use bevy_camera::CameraProjection;
use bevy_math::{Quat, Vec2};
use bevy_mesh::{MeshBuilder, Meshable};
use bevy_shape::{Cuboid, Plane3d, Sphere};

fn wall_mirror() -> WorldMirror {
    // A 2 x 1 m mirror on the wall z = 0, reflecting towards +z, centred at (0, 1, 0).
    let rect = measure_mirror_mesh(&Plane3d::new(Vec3::Y, Vec2::new(1.0, 0.5)).mesh().build())
        .expect("a plane is flat");
    let transform = Transform::from_translation(Vec3::new(0.0, 1.0, 0.0))
        .with_rotation(Quat::from_rotation_x(core::f32::consts::FRAC_PI_2));
    rect.to_world(&transform.compute_affine()).unwrap()
}

fn main_camera() -> (Mat4, PerspectiveProjection) {
    let transform = Transform::from_xyz(0.5, 1.2, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    let projection = PerspectiveProjection {
        aspect_ratio: 16.0 / 9.0,
        ..Default::default()
    };
    (transform.to_matrix(), projection)
}

fn project(clip_from_world: Mat4, point: Vec3) -> Vec4 {
    clip_from_world * point.extend(1.0)
}

/// Inside the clip volume of a reverse-Z projection: |x|, |y| <= w and 0 <= z <= w.
fn is_drawn(clip: Vec4) -> bool {
    clip.w > 0.0 && clip.x.abs() <= clip.w && clip.y.abs() <= clip.w && clip.z >= 0.0 && clip.z <= clip.w
}

#[test]
fn a_plane_is_flat_and_curved_or_closed_meshes_are_not() {
    let rect = measure_mirror_mesh(&Plane3d::new(Vec3::Y, Vec2::new(1.0, 0.5)).mesh().build()).unwrap();
    assert!(rect.normal.abs_diff_eq(Vec3::Y, 1e-6));
    assert!(rect.center.abs_diff_eq(Vec3::ZERO, 1e-6));
    let mut halves = [rect.half_u.length(), rect.half_v.length()];
    halves.sort_by(f32::total_cmp);
    assert!((halves[0] - 0.5).abs() < 1e-5 && (halves[1] - 1.0).abs() < 1e-5);

    assert!(measure_mirror_mesh(&Sphere::new(1.0).mesh().build()).is_none());
    assert!(measure_mirror_mesh(&Cuboid::new(1.0, 1.0, 0.01).mesh().build()).is_none());

    // One vertex lifted 1 % of the diagonal off the plane: not flat.
    let mut bent = Plane3d::new(Vec3::Y, Vec2::new(1.0, 0.5)).mesh().build();
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        bent.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        positions[0][1] += 0.022;
    }
    assert!(measure_mirror_mesh(&bent).is_none());

    // One normal tilted 5 degrees: not flat.
    let mut tilted = Plane3d::new(Vec3::Y, Vec2::new(1.0, 0.5)).mesh().build();
    if let Some(VertexAttributeValues::Float32x3(normals)) =
        tilted.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
    {
        normals[0] = (Quat::from_rotation_x(5f32.to_radians()) * Vec3::Y).to_array();
    }
    assert!(measure_mirror_mesh(&tilted).is_none());
}

#[test]
fn the_world_rectangle_follows_the_transform() {
    let mirror = wall_mirror();
    assert!(mirror.normal().abs_diff_eq(Vec3::Z, 1e-5));
    assert!((mirror.signed_distance(Vec3::new(3.0, 2.0, 4.0)) - 4.0).abs() < 1e-5);
    let (dual_u, dual_v) = mirror.dual_axes();
    for corner in mirror.corners() {
        let offset = corner - mirror.center;
        assert!((offset.dot(dual_u).abs() - 1.0).abs() < 1e-5);
        assert!((offset.dot(dual_v).abs() - 1.0).abs() < 1e-5);
    }
    // A non-uniform scale and a shear-free rotation keep the edges at +-1.
    let rect = measure_mirror_mesh(&Plane3d::new(Vec3::Y, Vec2::new(1.0, 0.5)).mesh().build()).unwrap();
    let transform = Transform::from_scale(Vec3::new(3.0, 1.0, 0.5))
        .with_rotation(Quat::from_rotation_z(0.3))
        .with_translation(Vec3::new(1.0, 2.0, 3.0));
    let world = rect.to_world(&transform.compute_affine()).unwrap();
    let (dual_u, dual_v) = world.dual_axes();
    for corner in world.corners() {
        let offset = corner - world.center;
        assert!((offset.dot(dual_u).abs() - 1.0).abs() < 1e-4);
        assert!((offset.dot(dual_v).abs() - 1.0).abs() < 1e-4);
    }
    // A mesh mirrored by a negative scale keeps the side its normals point to.
    let flipped = rect
        .to_world(&Transform::from_scale(Vec3::new(-1.0, 1.0, 1.0)).compute_affine())
        .unwrap();
    assert!(flipped.normal().abs_diff_eq(Vec3::Y, 1e-5));
}

#[test]
fn the_oblique_projection_clips_behind_the_mirror_and_keeps_the_room() {
    let mirror = wall_mirror();
    let (world_from_view, projection) = main_camera();
    let (mirror_world_from_view, mirror_projection) =
        mirror_camera(world_from_view, &projection, &mirror, 0.002, 16.0 / 9.0);
    // The reflection of a reflection is the camera itself: the transform is the reflected one.
    assert!((mirror.reflection() * mirror_world_from_view).abs_diff_eq(world_from_view, 1e-4));
    assert!(mirror_world_from_view.determinant() < 0.0);

    let clip_from_world = mirror_projection.get_clip_from_view() * mirror_world_from_view.inverse();
    let main_clip_from_world = projection.get_clip_from_view() * world_from_view.inverse();

    // Something in the room, in front of the mirror, is drawn; so is something behind the main camera.
    assert!(is_drawn(project(clip_from_world, Vec3::new(0.2, 1.0, 2.0))));
    assert!(is_drawn(project(clip_from_world, Vec3::new(0.3, 1.1, 8.0))));
    // Something behind the mirror (in the next room, or the wall's back), which the mirror camera, standing
    // behind the wall, has in front of it: not drawn.
    assert!(!is_drawn(project(clip_from_world, Vec3::new(0.1, 1.0, -0.5))));
    assert!(!is_drawn(project(clip_from_world, Vec3::new(0.1, 1.0, -0.001))));
    // The mirror's own surface is not drawn into its reflection.
    assert!(!is_drawn(project(clip_from_world, Vec3::new(0.1, 1.0, 0.0))));
    // A point on the mirror lands where the main camera sees it: the texture is sampled at the pixel.
    for point in [Vec3::new(0.3, 1.2, 0.0), Vec3::new(-0.8, 0.6, 0.0)] {
        let mirror_clip = project(clip_from_world, point);
        let main_clip = project(main_clip_from_world, point);
        let (a, b) = (mirror_clip.xy() / mirror_clip.w, main_clip.xy() / main_clip.w);
        assert!(a.abs_diff_eq(b, 1e-4), "{a} against {b}");
    }
    // Depth falls with distance from the mirror camera along a ray: nearer things win the depth test.
    let near = project(clip_from_world, Vec3::new(0.0, 1.0, 1.0));
    let far = project(clip_from_world, Vec3::new(0.0, 1.0, 4.0));
    assert!(near.z / near.w > far.z / far.w);
}

#[test]
fn coverage_is_the_share_of_the_screen() {
    let mirror = wall_mirror();
    let (world_from_view, projection) = main_camera();
    let clip_from_world = projection.get_clip_from_view() * world_from_view.inverse();
    let coverage = screen_coverage(clip_from_world, &mirror.corners());
    assert!(coverage > 0.02 && coverage < 0.5, "{coverage}");

    // Seen from 1 cm, the mirror fills the screen.
    let close = Transform::from_xyz(0.0, 1.0, 0.01).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    let close_clip = projection.get_clip_from_view() * close.to_matrix().inverse();
    assert!((screen_coverage(close_clip, &mirror.corners()) - 1.0).abs() < 1e-5);

    // Seen from far away, it is small.
    let far = Transform::from_xyz(0.0, 1.0, 200.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    let far_clip = projection.get_clip_from_view() * far.to_matrix().inverse();
    assert!(screen_coverage(far_clip, &mirror.corners()) < 0.001);
}

#[test]
fn the_budget_keeps_the_largest_mirrors() {
    let candidates = vec![(0.1, 'a'), (0.4, 'b'), (0.0001, 'c'), (0.3, 'd'), (0.2, 'e')];
    let chosen: Vec<char> = choose_mirrors(candidates.clone(), 2, 0.002)
        .into_iter()
        .map(|(_, mirror)| mirror)
        .collect();
    assert_eq!(chosen, ['b', 'd']);
    // Too small to be worth a render: left out even when the budget has room.
    let chosen: Vec<char> = choose_mirrors(candidates.clone(), 8, 0.002)
        .into_iter()
        .map(|(_, mirror)| mirror)
        .collect();
    assert_eq!(chosen, ['b', 'd', 'e', 'a']);
    assert!(choose_mirrors(candidates, 0, 0.0).is_empty());
}
