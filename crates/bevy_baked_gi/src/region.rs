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
    /// that overlap blend over this share of the box instead of stepping.
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
