use alloc::vec::Vec;
use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_math::{UVec3, Vec3};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_transform::components::Transform;

/// A box of the level to bake into one irradiance volume.
///
/// Like a [`LightProbe`](bevy_light::LightProbe), the box is a unit cube centred on the origin, placed and scaled
/// by the entity's transform. The entity also needs a `Name`: it names the volume's file and its manifest entry,
/// so it must be unique among the regions.
///
/// Bevy gives a fragment on a box face a probe weight of 0, so a box should not end exactly on a floor, a ceiling
/// or a wall's surface; [`GiBakeRegion::padded_transform`] builds the box of a room from its inner bounds.
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
#[require(Transform)]
pub struct GiBakeRegion {
    /// The largest distance between two voxel centres along any axis, in metres.
    pub spacing: f32,
    /// Copied to the volume's [`LightProbe::falloff`](bevy_light::LightProbe::falloff): regions of one space
    /// that overlap blend over this share of the box instead of stepping. `ZERO` (the default) lets the runtime
    /// pick it: [`overlap_falloffs`] blends across every overlap with another region and leaves a region that
    /// overlaps nothing as it is.
    pub falloff: Vec3,
}

impl Default for GiBakeRegion {
    fn default() -> Self {
        Self {
            // to tune: the plan's first spacing; gate 7 tries 0.25 m if it misses at this one
            spacing: 0.33,
            falloff: Vec3::ZERO,
        }
    }
}

impl GiBakeRegion {
    /// How far the box reaches past a room's walls. to tune: small enough that it never reaches the far face of
    /// a wall that is at least 0.3 m thick and centred on the room's bound.
    pub const HORIZONTAL_PADDING: f32 = 0.03;
    /// How far the box reaches below the floor and above the ceiling. to tune: puts the floor and the ceiling
    /// inside the box, where Bevy weighs the volume fully.
    pub const VERTICAL_PADDING: f32 = 0.10;

    /// The transform of a region that covers the room whose inner bounds are `min`..`max` (Y up), padded by
    /// [`Self::HORIZONTAL_PADDING`] and [`Self::VERTICAL_PADDING`].
    pub fn padded_transform(min: Vec3, max: Vec3) -> Transform {
        let pad = Vec3::new(
            Self::HORIZONTAL_PADDING,
            Self::VERTICAL_PADDING,
            Self::HORIZONTAL_PADDING,
        );
        let (min, max) = (min - pad, max + pad);
        Transform::from_translation((min + max) / 2.0).with_scale(max - min)
    }

    /// The number of voxels along each axis of a box of `size` metres: at least 2, so the volume interpolates.
    pub fn resolution(&self, size: Vec3) -> UVec3 {
        // The small slack keeps a size that is an exact multiple of the spacing from rounding up a voxel.
        let cells = (size.abs() / self.spacing - 1e-4).ceil();
        cells.as_uvec3().max(UVec3::splat(2))
    }
}

/// Bakes this entity without its emission.
///
/// A lamp that a rasterizer draws as an analytic light (a spot) and shows as an emissive mesh (a diffuser) would
/// otherwise reach the bake twice: once through the light, and once through the mesh's emission. The baker swaps
/// the marked entity's material for a copy with black emission before the scene reaches the tracer, so the mesh
/// leaves the light list and still reflects like the original.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
pub struct GiBakeSkipEmission;

/// The falloff of each box (`Transform`s of unit cubes, as [`GiBakeRegion`]s place them) that blends it into the
/// boxes it overlaps without a seam.
///
/// Bevy weighs overlapping volumes by their falloff and normalises the sum, so with a falloff of 0 a fragment
/// steps from A alone to (A + B) / 2 where it enters B. Here every face of a box that lies inside another box gets
/// a falloff on its axis as wide as the overlap, so the box's weight reaches 0 at that face and the blend is
/// continuous. The falloff is symmetric (Bevy's), so the opposite face fades too; where no other box is there the
/// normalised sum is still the box alone. Boxes that only touch, or overlap nothing, get `ZERO`. Rotated boxes are
/// left at `ZERO`: the overlap is measured on axis-aligned boxes. to tune: the cap of 0.5, a box's whole half.
pub fn overlap_falloffs(boxes: &[Transform]) -> Vec<Vec3> {
    const EPSILON: f32 = 1e-4;
    let aligned = |t: &Transform| t.rotation.angle_between(bevy_math::Quat::IDENTITY) < 1e-4;
    let bounds = |t: &Transform| {
        let half = t.scale.abs() / 2.0;
        (t.translation - half, t.translation + half)
    };
    boxes
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let mut falloff = Vec3::ZERO;
            if !aligned(a) {
                return falloff;
            }
            let (a_min, a_max) = bounds(a);
            let size = a_max - a_min;
            for (j, b) in boxes.iter().enumerate() {
                if i == j || !aligned(b) {
                    continue;
                }
                let (b_min, b_max) = bounds(b);
                let overlaps =
                    (0..3).all(|k| a_min[k] < b_max[k] - EPSILON && b_min[k] < a_max[k] - EPSILON);
                if !overlaps {
                    continue;
                }
                for k in 0..3 {
                    let strictly_inside = |x: f32| b_min[k] + EPSILON < x && x < b_max[k] - EPSILON;
                    let mut depth: f32 = 0.0;
                    if strictly_inside(a_max[k]) {
                        depth = depth.max(a_max[k] - b_min[k].max(a_min[k]));
                    }
                    if strictly_inside(a_min[k]) {
                        depth = depth.max(b_max[k].min(a_max[k]) - a_min[k]);
                    }
                    if size[k] > 0.0 {
                        falloff[k] = falloff[k].max((depth / size[k]).min(0.5));
                    }
                }
            }
            falloff
        })
        .collect()
}

/// Leaves this entity, and everything under it in the hierarchy, out of the bake.
///
/// For whatever moves or comes and goes at run time: characters, physics props, doors that open, a lamp the
/// player carries. Baked in place, their occlusion and bounce would stay in the static volumes after they move
/// (a dark patch where a door stood). The baker does not trace their meshes and switches their lights off while it
/// bakes, and the [`scene_hash`](crate::scene_hash) ignores them, so moving one never makes a bake stale. At run
/// time they are lit by the volumes like any other mesh.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
pub struct GiBakeExclude;

/// A stable id for a light, which the bake records and the run-time matches first.
///
/// Without one, a live light is matched to the baked light of the same kind nearest to it, within
/// [`LightKey::POSITION_TOLERANCE`](crate::LightKey::POSITION_TOLERANCE) (a directional light: its direction,
/// within [`LightKey::DIRECTION_TOLERANCE`](crate::LightKey::DIRECTION_TOLERANCE)). Give an id to a light that
/// moves farther than that and keeps its bounce light: a lamp that sways or rides a door, a sun turned for the
/// time of day. Ids are unique per light kind within a level.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
pub struct GiBakeLight(pub u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_puts_floor_and_walls_inside_the_box() {
        let transform =
            GiBakeRegion::padded_transform(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 3.0, 2.0));
        let half = transform.scale / 2.0;
        let min = transform.translation - half;
        let max = transform.translation + half;
        assert!((min.y + 0.10).abs() < 1e-6 && (max.y - 3.10).abs() < 1e-6);
        assert!((min.x + 0.03).abs() < 1e-6 && (max.z - 2.03).abs() < 1e-6);
    }

    /// The weight Bevy gives a point of a box (`light_probe.wesl`): per axis
    /// `saturate((1 − 2|p|) / (2 · max(falloff, 1e-4)))`, the smallest of the three; 0 outside.
    fn weight(t: &Transform, falloff: Vec3, point: Vec3) -> f32 {
        let p = (point - t.translation) / t.scale;
        let axis = (Vec3::ONE - 2.0 * p.abs()) / (2.0 * falloff.max(Vec3::splat(1e-4)));
        axis.clamp(Vec3::ZERO, Vec3::ONE).min_element()
    }

    #[test]
    fn an_l_corridor_blends_without_a_seam() {
        // Corridor A along x, corridor B along z, sharing the 2 × 2 m corner square at x 8..10, z 0..2.
        let a = Transform::from_xyz(5.0, 1.5, 1.0).with_scale(Vec3::new(10.0, 3.0, 2.0));
        let b = Transform::from_xyz(9.0, 1.5, 5.0).with_scale(Vec3::new(2.0, 3.0, 10.0));
        let lone = Transform::from_xyz(50.0, 1.5, 0.0).with_scale(Vec3::splat(3.0));
        let falloffs = overlap_falloffs(&[a, b, lone]);
        // A's z faces lie inside B (its far z face at 2 m, inside B's 0..10), B's x face at 8 lies inside A.
        assert_eq!(falloffs[0], Vec3::new(0.0, 0.0, 0.5));
        assert_eq!(falloffs[1], Vec3::new(0.5, 0.0, 0.0));
        assert_eq!(falloffs[2], Vec3::ZERO);
        // B's share of the blend, (A, B) values 0 and 1, on lines that cross the overlap's edges: no step.
        let blend = |point: Vec3| {
            let (wa, wb) = (
                weight(&a, falloffs[0], point),
                weight(&b, falloffs[1], point),
            );
            wb / (wa + wb)
        };
        for z in [0.3, 1.0, 1.7] {
            let mut previous = blend(Vec3::new(7.0, 1.5, z));
            assert_eq!(previous, 0.0);
            for step in 1..=200 {
                let x = 7.0 + 2.9 * step as f32 / 200.0;
                let next = blend(Vec3::new(x, 1.5, z));
                assert!(
                    (next - previous).abs() < 0.05,
                    "x {x} z {z}: {previous} → {next}"
                );
                previous = next;
            }
        }
        for x in [8.3, 9.0, 9.7] {
            let mut previous = blend(Vec3::new(x, 1.5, 0.1));
            for step in 1..=200 {
                let z = 0.1 + 2.9 * step as f32 / 200.0;
                let next = blend(Vec3::new(x, 1.5, z));
                assert!(
                    (next - previous).abs() < 0.05,
                    "x {x} z {z}: {previous} → {next}"
                );
                previous = next;
            }
            assert_eq!(previous, 1.0);
        }
        // Without the falloffs the old seam: B's share jumps from 0 to ½ where the overlap starts.
        let hard = |point: Vec3| {
            let (wa, wb) = (weight(&a, Vec3::ZERO, point), weight(&b, Vec3::ZERO, point));
            wb / (wa + wb)
        };
        assert_eq!(hard(Vec3::new(7.99, 1.5, 1.0)), 0.0);
        assert_eq!(hard(Vec3::new(8.01, 1.5, 1.0)), 0.5);
    }

    #[test]
    fn rooms_padded_on_either_side_of_a_wall_do_not_blend() {
        let a =
            GiBakeRegion::padded_transform(Vec3::new(-5.0, 0.0, -2.0), Vec3::new(-0.05, 3.0, 2.0));
        let b =
            GiBakeRegion::padded_transform(Vec3::new(0.05, 0.0, -2.0), Vec3::new(5.0, 3.0, 2.0));
        assert_eq!(overlap_falloffs(&[a, b]), vec![Vec3::ZERO, Vec3::ZERO]);
    }

    #[test]
    fn resolution_rounds_up_and_keeps_two_voxels() {
        let region = GiBakeRegion {
            spacing: 0.5,
            ..Default::default()
        };
        assert_eq!(
            region.resolution(Vec3::new(2.0, 0.1, 2.2)),
            UVec3::new(4, 2, 5)
        );
    }
}
