//! Baked reflection probes, the CPU half: where a cubemap texel looks, the GGX prefilter, and the KTX2 cubemap.
//!
//! **What a texel holds.** Bevy reprojects a reflection probe onto its box (`ParallaxCorrection::Auto`): the
//! shader intersects the reflected ray with the box in probe space and samples the cubemap at that point, z
//! negated (`environment_map.wesl`, `compute_cubemap_sample_dir`). So texel `c` of the cubemap stands for the point
//! `q = 0.5 · (c.x, c.y, −c.z) / max|c|` of the unit box, and the capture traces from the capture origin towards
//! that point's world position. With the origin at the box's centre this is exactly the light Bevy's
//! reprojection assumes; a capture moved off the centre (the centre inside geometry) is off by the move.
//!
//! **Prefilter.** Level `m` of `L` holds the GGX lobe of perceptual roughness `m / (L − 1)`, the level Bevy's
//! shader samples for a roughness (`radiance_level = perceptual_roughness · (levels − 1)`); the roughness is
//! clamped to 0.089 before squaring, as `perceptualRoughnessToRoughness` does. Each texel sums the GGX lobe with
//! `N = V = R` (Karis, "Real Shading in Unreal Engine 4", 2013), weighted by N·L, from a box-filtered pyramid of
//! level 0 at the level filtered importance sampling picks (Křivánek and Colbert, 2008), as Bevy's
//! `environment_filter.wesl` does on the GPU.

use alloc::vec::Vec;
use bevy_math::{
    ops::{self, FloatPow},
    Vec3,
};
use core::f32::consts::PI;

/// How each region's reflection probe is captured and filtered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiBakeReflections {
    /// Texels along a face's side at level 0, a power of two.
    pub size: u32,
    /// Paths per texel at level 0.
    pub samples: u32,
    /// Mip levels, the last one `size >> (levels − 1)` texels wide.
    pub levels: u32,
    /// GGX samples per texel of each prefiltered level.
    pub prefilter_samples: u32,
}

impl Default for GiBakeReflections {
    fn default() -> Self {
        Self {
            // to tune: member 3's 128² (about 0.5 MB per probe with its mips)
            size: 128,
            // to tune: level 0 is only read by mirrors; rougher levels average many level-0 texels
            samples: 512,
            // to tune: stops at 4×4, so the roughest lobes are not drawn from 1-texel faces that cannot blend
            levels: 6,
            // to tune: filtered importance sampling reads a blurrier source level when samples are sparse
            prefilter_samples: 512,
        }
    }
}

impl GiBakeReflections {
    /// Whether the settings describe a cubemap that can be written.
    pub fn is_valid(&self) -> bool {
        self.size.is_power_of_two()
            && self.size >= 4
            && self.levels >= 1
            && (self.size >> (self.levels - 1)) >= 1
            && self.samples > 0
            && self.prefilter_samples > 0
    }
}

/// The cubemap coordinate (not normalised) of face `face` (+X, −X, +Y, −Y, +Z, −Z) at `(s, t)` ∈ [0, 1]², `t`
/// down the rows: the inverse of the cube lookup of Vulkan, D3D and wgpu.
pub fn face_direction(face: usize, s: f32, t: f32) -> Vec3 {
    let (sc, tc) = (2.0 * s - 1.0, 2.0 * t - 1.0);
    match face {
        0 => Vec3::new(1.0, -tc, -sc),
        1 => Vec3::new(-1.0, -tc, sc),
        2 => Vec3::new(sc, 1.0, tc),
        3 => Vec3::new(sc, -1.0, -tc),
        4 => Vec3::new(sc, -tc, 1.0),
        _ => Vec3::new(-sc, -tc, -1.0),
    }
}

/// The face and `(s, t)` a cubemap lookup of `direction` reads.
pub fn direction_face(direction: Vec3) -> (usize, f32, f32) {
    let a = direction.abs();
    let (face, sc, tc, ma) = if a.x >= a.y && a.x >= a.z {
        if direction.x > 0.0 {
            (0, -direction.z, -direction.y, a.x)
        } else {
            (1, direction.z, -direction.y, a.x)
        }
    } else if a.y >= a.z {
        if direction.y > 0.0 {
            (2, direction.x, direction.z, a.y)
        } else {
            (3, direction.x, -direction.z, a.y)
        }
    } else if direction.z > 0.0 {
        (4, direction.x, -direction.y, a.z)
    } else {
        (5, -direction.x, -direction.y, a.z)
    };
    (face, 0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0))
}

/// The point of the unit box (probe space) that cubemap coordinate `c` stands for under Bevy's parallax
/// correction.
pub fn box_point(c: Vec3) -> Vec3 {
    let p = Vec3::new(c.x, c.y, -c.z);
    0.5 * p / p.abs().max_element()
}

/// One face of one level: `n × n` linear RGB texels, row by row.
type Face = Vec<[f32; 3]>;

/// A cubemap level: six faces of `n × n`.
#[derive(Clone)]
struct Level {
    n: usize,
    faces: [Face; 6],
}

impl Level {
    /// Bilinear within the face, clamped at its edges.
    fn sample(&self, direction: Vec3) -> Vec3 {
        let (face, s, t) = direction_face(direction);
        let n = self.n as f32;
        let x = (s * n - 0.5).clamp(0.0, n - 1.0);
        let y = (t * n - 0.5).clamp(0.0, n - 1.0);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.n - 1), (y0 + 1).min(self.n - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let texel = |x: usize, y: usize| Vec3::from_array(self.faces[face][y * self.n + x]);
        let top = texel(x0, y0).lerp(texel(x1, y0), fx);
        let bottom = texel(x0, y1).lerp(texel(x1, y1), fx);
        top.lerp(bottom, fy)
    }

    /// The next level of a box-filtered pyramid.
    fn half(&self) -> Level {
        let n = (self.n / 2).max(1);
        let faces = core::array::from_fn(|face| {
            let source = &self.faces[face];
            let mut out = vec![[0.0; 3]; n * n];
            for y in 0..n {
                for x in 0..n {
                    let mut sum = Vec3::ZERO;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (sx, sy) = ((2 * x + dx).min(self.n - 1), (2 * y + dy).min(self.n - 1));
                        sum += Vec3::from_array(source[sy * self.n + sx]);
                    }
                    out[y * n + x] = (sum / 4.0).to_array();
                }
            }
            out
        });
        Level { n, faces }
    }
}

/// The base-2 radical inverse of `i` (Van der Corput).
fn radical_inverse(i: u32) -> f32 {
    i.reverse_bits() as f32 * 2.328_306_4e-10
}

/// The GGX normal distribution at `n_dot_h` for `alpha` = roughness².
fn ggx_d(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d)
}

/// Prefilters a captured cubemap (`size × size` per face, `faces[face][row · size + column]`) into `levels`
/// levels, level 0 unchanged. Returns each level's faces.
pub fn prefilter(size: usize, faces: [Face; 6], levels: u32, samples: u32) -> Vec<[Face; 6]> {
    let mut pyramid = vec![Level { n: size, faces }];
    while pyramid.last().unwrap().n > 1 {
        let next = pyramid.last().unwrap().half();
        pyramid.push(next);
    }
    let texel_solid_angle = 4.0 * PI / (6.0 * (size * size) as f32);
    let max_lod = (pyramid.len() - 1) as f32;
    let sample_at = |direction: Vec3, lod: f32| {
        let lod = lod.clamp(0.0, max_lod);
        let low = lod.floor() as usize;
        let high = (low + 1).min(pyramid.len() - 1);
        pyramid[low]
            .sample(direction)
            .lerp(pyramid[high].sample(direction), lod - low as f32)
    };

    let mut out = vec![pyramid[0].faces.clone()];
    for level in 1..levels {
        let n = (size >> level).max(1);
        let perceptual = level as f32 / (levels - 1) as f32;
        let alpha = perceptual.clamp(0.089, 1.0).squared();
        // The lobe's samples in a frame around +Z, shared by every texel of the level.
        let lobe: Vec<(Vec3, f32, f32)> = (0..samples)
            .filter_map(|i| {
                let (u, v) = ((i as f32 + 0.5) / samples as f32, radical_inverse(i));
                let cos_h = ((1.0 - u) / (1.0 + (alpha * alpha - 1.0) * u)).sqrt();
                let sin_h = (1.0 - cos_h * cos_h).max(0.0).sqrt();
                let phi = 2.0 * PI * v;
                let h = Vec3::new(sin_h * ops::cos(phi), sin_h * ops::sin(phi), cos_h);
                // L = reflect(−V, H) with V = N = +Z.
                let l = 2.0 * cos_h * h - Vec3::Z;
                if l.z <= 0.0 {
                    return None;
                }
                // pdf(L) = D · (N·H) / (4 · V·H), and V·H = N·H when V = N.
                let pdf = ggx_d(cos_h, alpha) / 4.0;
                let lod = 0.5 * ops::log2(1.0 / (samples as f32 * pdf * texel_solid_angle));
                Some((l, l.z, lod))
            })
            .collect();
        let faces = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..6)
                .map(|face| {
                    let (lobe, sample_at) = (&lobe, &sample_at);
                    scope.spawn(move || {
                        let mut texels = vec![[0.0; 3]; n * n];
                        for y in 0..n {
                            for x in 0..n {
                                let s = (x as f32 + 0.5) / n as f32;
                                let t = (y as f32 + 0.5) / n as f32;
                                let normal = face_direction(face, s, t).normalize();
                                let (tangent, bitangent) = normal.any_orthonormal_pair();
                                let mut sum = Vec3::ZERO;
                                let mut weight = 0.0;
                                for (l, n_dot_l, lod) in lobe {
                                    let direction = tangent * l.x + bitangent * l.y + normal * l.z;
                                    sum += sample_at(direction, *lod) * *n_dot_l;
                                    weight += n_dot_l;
                                }
                                texels[y * n + x] = (sum / weight).to_array();
                            }
                        }
                        texels
                    })
                })
                .collect();
            let faces: Vec<Face> = handles
                .into_iter()
                .map(|handle| handle.join().expect("a prefilter thread panicked"))
                .collect();
            <[Face; 6]>::try_from(faces).ok().unwrap()
        });
        out.push(faces);
    }
    out
}

/// Writes prefiltered levels as a KTX2 cubemap in `Rgba16Float`, alpha 1, values clamped to the half-float
/// range.
pub fn encode_ktx2_cubemap(size: u32, levels: &[[Face; 6]]) -> Vec<u8> {
    const KTX2_IDENTIFIER: [u8; 12] = [
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
    ];
    const VK_FORMAT_R16G16B16A16_SFLOAT: u32 = 97;
    let half = |value: f32| half::f16::from_f32(value.clamp(0.0, 65_504.0)).to_bits();
    let one = half::f16::from_f32(1.0).to_bits();
    let level_bytes: Vec<Vec<u8>> = levels
        .iter()
        .map(|faces| {
            faces
                .iter()
                .flat_map(|face| face.iter())
                .flat_map(|rgb| {
                    [half(rgb[0]), half(rgb[1]), half(rgb[2]), one]
                        .into_iter()
                        .flat_map(u16::to_le_bytes)
                })
                .collect()
        })
        .collect();
    let dfd_block = crate::volume::rgba16f_dfd_block();
    let dfd_total = 4 + dfd_block.len() as u32;

    const HEADER: u32 = 12 + 9 * 4;
    const INDEX: u32 = 4 * 4 + 2 * 8;
    let level_index = 3 * 8 * levels.len() as u32;
    let dfd_offset = HEADER + INDEX + level_index;
    // The spec stores the smallest level first; each starts on a multiple of the texel size (8).
    let mut offsets = vec![0u64; levels.len()];
    let mut end = u64::from(dfd_offset + dfd_total);
    for (level, bytes) in level_bytes.iter().enumerate().rev() {
        end = end.next_multiple_of(8);
        offsets[level] = end;
        end += bytes.len() as u64;
    }

    let mut out = Vec::with_capacity(end as usize);
    out.extend_from_slice(&KTX2_IDENTIFIER);
    for value in [
        VK_FORMAT_R16G16B16A16_SFLOAT,
        2, // type size
        size,
        size,
        0, // depth
        0, // layers
        6, // faces
        levels.len() as u32,
        0, // no supercompression
    ] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&dfd_offset.to_le_bytes());
    out.extend_from_slice(&dfd_total.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // no key/value data
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // no supercompression global data
    out.extend_from_slice(&0u64.to_le_bytes());
    for (offset, bytes) in offsets.iter().zip(&level_bytes) {
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    }
    out.extend_from_slice(&dfd_total.to_le_bytes());
    out.extend_from_slice(&dfd_block);
    for (level, bytes) in level_bytes.iter().enumerate().rev() {
        out.resize(offsets[level] as usize, 0);
        out.extend_from_slice(bytes);
    }
    out
}

/// The cubemap's file name for a region's volume file `volume` (`room.ktx2` → `room.refl.ktx2`).
pub(crate) fn file_name(volume: &str) -> String {
    let stem = volume.strip_suffix(".ktx2").unwrap_or(volume);
    format!("{stem}.refl.ktx2")
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_image::{CompressedImageFormats, Image, ImageSampler, ImageType};
    use wgpu_types::{TextureFormat, TextureViewDimension};

    #[test]
    fn face_lookup_inverts_the_texel_direction() {
        for face in 0..6 {
            for (s, t) in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.3), (0.25, 0.95)] {
                let (back, bs, bt) = direction_face(face_direction(face, s, t) * 3.0);
                assert_eq!(back, face);
                assert!((bs - s).abs() < 1e-6 && (bt - t).abs() < 1e-6);
            }
        }
        // The face centres point along the axes in Bevy's face order.
        let axes = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
        for (face, axis) in axes.into_iter().enumerate() {
            assert_eq!(face_direction(face, 0.5, 0.5), axis);
        }
    }

    #[test]
    fn box_points_lie_on_the_unit_box_with_z_negated() {
        // Bevy negates z after reprojecting, so cubemap +Z looks at the box's −Z face.
        assert_eq!(box_point(Vec3::Z), Vec3::new(0.0, 0.0, -0.5));
        assert_eq!(
            box_point(Vec3::new(2.0, 1.0, 0.0)),
            Vec3::new(0.5, 0.25, 0.0)
        );
    }

    fn constant_faces(n: usize, value: impl Fn(usize) -> [f32; 3]) -> [Face; 6] {
        core::array::from_fn(|face| vec![value(face); n * n])
    }

    #[test]
    fn prefilter_keeps_a_uniform_cubemap_uniform() {
        let levels = prefilter(16, constant_faces(16, |_| [2.0, 1.0, 0.5]), 4, 64);
        assert_eq!(levels.len(), 4);
        for (level, faces) in levels.iter().enumerate() {
            assert_eq!(faces[0].len(), (16 >> level) * (16 >> level));
            for texel in faces.iter().flatten() {
                for (got, want) in texel.iter().zip([2.0, 1.0, 0.5]) {
                    assert!((got - want).abs() < 1e-4, "level {level}: {texel:?}");
                }
            }
        }
    }

    #[test]
    fn rougher_levels_spread_a_bright_face() {
        // Only +Y is lit: a sharp level keeps the other faces' centres dark, the roughest one lets light in.
        let levels = prefilter(
            16,
            constant_faces(16, |f| [(f == 2) as u8 as f32; 3]),
            5,
            256,
        );
        let centre = |level: usize, face: usize| {
            let n = 16 >> level;
            levels[level][face][(n / 2) * n + n / 2][0]
        };
        assert!(centre(1, 0) < 0.01);
        assert!(centre(4, 0) > 0.05 && centre(4, 0) < centre(4, 2));
        assert!(centre(4, 3) < centre(4, 0));
    }

    #[test]
    fn ktx2_cubemap_round_trips_through_bevys_loader() {
        let size = 8;
        let faces: [Face; 6] = core::array::from_fn(|face| {
            (0..size * size)
                .map(|i| [face as f32, i as f32 * 0.5, 1.0])
                .collect()
        });
        let levels = vec![
            faces.clone(),
            constant_faces(4, |f| [f as f32 + 10.0, 0.0, 0.0]),
        ];
        let bytes = encode_ktx2_cubemap(size as u32, &levels);
        let image = Image::from_buffer(
            &bytes,
            ImageType::Extension("ktx2"),
            CompressedImageFormats::NONE,
            false,
            ImageSampler::Default,
            Default::default(),
        )
        .expect("Bevy's KTX2 loader reads the cubemap");
        assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba16Float);
        assert_eq!(image.texture_descriptor.mip_level_count, 2);
        assert_eq!(image.texture_descriptor.size.depth_or_array_layers, 6);
        assert_eq!(
            image.texture_view_descriptor.as_ref().unwrap().dimension,
            Some(TextureViewDimension::Cube)
        );
        // Mip-major: level 0's six faces, then level 1's.
        let data = image.data.as_ref().unwrap();
        let read = |texel: usize, channel: usize| {
            let i = texel * 8 + channel * 2;
            half::f16::from_bits(u16::from_le_bytes([data[i], data[i + 1]])).to_f32()
        };
        assert_eq!(read(3 * 64 + 5, 0), 3.0);
        assert_eq!(read(3 * 64 + 5, 1), 2.5);
        assert_eq!(read(6 * 64 + 4 * 16, 0), 14.0);
    }

    #[test]
    fn cubemap_file_sits_beside_the_volume() {
        assert_eq!(file_name("room_a.ktx2"), "room_a.refl.ktx2");
    }
}
