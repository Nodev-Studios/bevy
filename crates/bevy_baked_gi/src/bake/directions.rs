//! The directions a voxel's paths start in, computed on the CPU and uploaded, so a test can rebuild exactly the
//! directions a bake used (GPU `sin`, `cos` and `sqrt` are not correctly rounded; integer hashing is exact).

use alloc::vec::Vec;
use bevy_math::{DQuat, Mat3, Vec3};
use core::f64::consts::{PI, TAU};

/// The PCG hash (Jarzynski and Olano, "Hash Functions for GPU Rendering", 2020). `bake.wesl` has the same
/// function, so CPU and GPU derive the same seeds.
pub fn pcg_hash(value: u32) -> u32 {
    let state = value.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// The seed of voxel `voxel` (its index over all regions of the bake) under bake seed `seed`.
pub fn voxel_seed(seed: u32, voxel: u32) -> u32 {
    pcg_hash(seed ^ pcg_hash(voxel))
}

/// A uniform number in [0, 1) from the top 24 bits of `hash`.
fn unit(hash: u32) -> f64 {
    f64::from(hash >> 8) / f64::from(1u32 << 24)
}

/// `n` directions spread evenly over the sphere: the spherical Fibonacci set, point `i` at height
/// `1 − (2i + 1)/n` and turned by the golden angle from the one before.
pub fn fibonacci_directions(n: u32) -> Vec<Vec3> {
    let golden_angle = PI * (3.0 - 5.0f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - (2.0 * f64::from(i) + 1.0) / f64::from(n);
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = golden_angle * f64::from(i);
            Vec3::new((r * phi.cos()) as f32, (r * phi.sin()) as f32, z as f32)
        })
        .collect()
}

/// The rotation of voxel `voxel`'s direction set: a uniformly random rotation (Shoemake's method) from the
/// voxel's seed. Two bakes with the same seed rotate every voxel the same way; two seeds are "two rotations".
pub fn voxel_rotation(seed: u32, voxel: u32) -> Mat3 {
    let h0 = pcg_hash(voxel_seed(seed, voxel) ^ 0x9e37_79b9);
    let h1 = pcg_hash(h0);
    let h2 = pcg_hash(h1);
    let (u1, u2, u3) = (unit(h0), unit(h1), unit(h2));
    let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
    let quat = DQuat::from_xyzw(
        a * (TAU * u2).sin(),
        a * (TAU * u2).cos(),
        b * (TAU * u3).sin(),
        b * (TAU * u3).cos(),
    )
    .normalize();
    Mat3::from_quat(quat.as_quat())
}

/// The direction of sample `sample` of the voxel rotated by `rotation`, as `bake.wesl` computes it.
pub fn sample_direction(rotation: &Mat3, base: Vec3) -> Vec3 {
    (*rotation * base).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotated_fibonacci_set_integrates_cosine_lobes() {
        // A uniform furnace: each side's estimate 4/N Σ max(0, ω·n) must be 1.
        let base = fibonacci_directions(2048);
        for voxel in 0..20 {
            let rotation = voxel_rotation(7, voxel);
            for axis in [
                Vec3::X,
                Vec3::NEG_X,
                Vec3::Y,
                Vec3::NEG_Y,
                Vec3::Z,
                Vec3::NEG_Z,
            ] {
                let sum: f64 = base
                    .iter()
                    .map(|d| f64::from(sample_direction(&rotation, *d).dot(axis).max(0.0)))
                    .sum();
                let estimate = 4.0 * sum / 2048.0;
                assert!((estimate - 1.0).abs() < 2e-3, "{estimate}");
            }
        }
    }

    #[test]
    fn rotations_are_orthonormal_and_differ_by_seed() {
        let a = voxel_rotation(1, 5);
        let b = voxel_rotation(2, 5);
        assert!((a * a.transpose()).abs_diff_eq(Mat3::IDENTITY, 1e-5));
        assert!((a.determinant() - 1.0).abs() < 1e-5);
        assert!(!a.abs_diff_eq(b, 1e-3));
        assert_eq!(voxel_rotation(1, 5), a);
    }
}
