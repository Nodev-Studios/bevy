use alloc::vec::Vec;
use bevy_math::UVec3;

/// The six sides of a voxel in the order this crate stores them: +X, −X, +Y, −Y, +Z, −Z.
pub const FACE_NAMES: [&str; 6] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];

/// One baked volume before it is written: per voxel, each side's linear RGB in [`FACE_NAMES`] order.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeData {
    /// Voxels along each axis.
    pub resolution: UVec3,
    /// Indexed by `x + Rx * (y + Ry * z)`.
    pub faces: Vec<[[f32; 3]; 6]>,
}

impl VolumeData {
    /// The index into `faces` of the voxel at `voxel`.
    pub fn index(&self, voxel: UVec3) -> usize {
        let r = self.resolution;
        (voxel.x + r.x * (voxel.y + r.y * voxel.z)) as usize
    }
}

/// Where side `face` ([`FACE_NAMES`] order) of the voxel at `voxel` sits in the atlas of a volume of
/// `resolution` voxels.
///
/// This follows the shader (`irradiance_volume.wesl`, which adds half the atlas height for a negative side), not
/// the module docs of `irradiance_volume.rs`, which put the negative sides in the lower half and disagree with it.
/// The atlas is `(Rx, 2 Ry, 3 Rz)`: X sides in the first third along Z, then Y, then Z; positive sides in rows
/// `0..Ry`, negative sides in rows `Ry..2 Ry`.
pub fn atlas_texel(voxel: UVec3, face: usize, resolution: UVec3) -> UVec3 {
    let axis = (face / 2) as u32;
    let negative = face % 2 == 1;
    UVec3::new(
        voxel.x,
        voxel.y + if negative { resolution.y } else { 0 },
        voxel.z + axis * resolution.z,
    )
}

// KTX2 constants (Khronos KTX 2.0 specification).
const KTX2_IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];
const VK_FORMAT_R16G16B16A16_SFLOAT: u32 = 97;
const KHR_DF_MODEL_RGBSDA: u8 = 1;
const KHR_DF_PRIMARIES_BT709: u8 = 1;
const KHR_DF_TRANSFER_LINEAR: u8 = 1;
const KHR_DF_SAMPLE_DATATYPE_FLOAT: u8 = 0x80;
const KHR_DF_SAMPLE_DATATYPE_SIGNED: u8 = 0x40;

/// Writes `volume` as a KTX2 3D texture in `Rgba16Float`, in the atlas layout Bevy's irradiance volumes read
/// (see [`atlas_texel`]). Alpha is 1.
///
/// Half floats keep about three significant digits over the whole range light needs. `Rgb9e5` would halve the
/// size, but its shared exponent costs a dim channel next to a bright one most of its precision.
pub fn encode_ktx2_rgba16f(volume: &VolumeData) -> Vec<u8> {
    let r = volume.resolution;
    let (width, height, depth) = (r.x, 2 * r.y, 3 * r.z);
    let texel_count = (width * height * depth) as usize;
    let mut texels = vec![[0u16; 4]; texel_count];
    let one = half::f16::from_f32(1.0).to_bits();
    for z in 0..r.z {
        for y in 0..r.y {
            for x in 0..r.x {
                let voxel = UVec3::new(x, y, z);
                let sides = &volume.faces[volume.index(voxel)];
                for (face, rgb) in sides.iter().enumerate() {
                    let t = atlas_texel(voxel, face, r);
                    let i = (t.x + width * (t.y + height * t.z)) as usize;
                    texels[i] = [
                        half::f16::from_f32(rgb[0]).to_bits(),
                        half::f16::from_f32(rgb[1]).to_bits(),
                        half::f16::from_f32(rgb[2]).to_bits(),
                        one,
                    ];
                }
            }
        }
    }
    let level: Vec<u8> = texels
        .iter()
        .flat_map(|t| t.iter().flat_map(|c| c.to_le_bytes()))
        .collect();

    // The data format descriptor: one basic block with four 16-bit float samples (R, G, B, A).
    let mut dfd_block = Vec::new();
    dfd_block.extend_from_slice(&0u32.to_le_bytes()); // vendor 0 (Khronos), descriptor type 0 (basic)
    dfd_block.extend_from_slice(&2u16.to_le_bytes()); // version 2
    dfd_block.extend_from_slice(&(24u16 + 16 * 4).to_le_bytes());
    dfd_block.extend_from_slice(&[
        KHR_DF_MODEL_RGBSDA,
        KHR_DF_PRIMARIES_BT709,
        KHR_DF_TRANSFER_LINEAR,
        0, // flags: alpha straight
    ]);
    dfd_block.extend_from_slice(&[0, 0, 0, 0]); // texel block 1x1x1x1
    dfd_block.extend_from_slice(&[8, 0, 0, 0, 0, 0, 0, 0]); // 8 bytes in plane 0
    for (i, channel) in [0u8, 1, 2, 15].into_iter().enumerate() {
        dfd_block.extend_from_slice(&(16 * i as u16).to_le_bytes()); // bit offset
        dfd_block.push(15); // bit length - 1
        dfd_block.push(channel | KHR_DF_SAMPLE_DATATYPE_FLOAT | KHR_DF_SAMPLE_DATATYPE_SIGNED);
        dfd_block.extend_from_slice(&[0, 0, 0, 0]); // sample position
        dfd_block.extend_from_slice(&(-1.0f32).to_bits().to_le_bytes());
        dfd_block.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
    }
    let dfd_total = 4 + dfd_block.len() as u32;

    const HEADER: u32 = 12 + 9 * 4;
    const INDEX: u32 = 4 * 4 + 2 * 8;
    const LEVEL_INDEX: u32 = 3 * 8;
    let dfd_offset = HEADER + INDEX + LEVEL_INDEX;
    // Level data starts on a multiple of the texel size (8) and of 4.
    let level_offset = (dfd_offset + dfd_total).next_multiple_of(8);

    let mut out = Vec::with_capacity(level_offset as usize + level.len());
    out.extend_from_slice(&KTX2_IDENTIFIER);
    for value in [
        VK_FORMAT_R16G16B16A16_SFLOAT,
        2, // type size
        width,
        height,
        depth,
        0, // layers
        1, // faces
        1, // levels
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
    out.extend_from_slice(&(level_offset as u64).to_le_bytes());
    out.extend_from_slice(&(level.len() as u64).to_le_bytes());
    out.extend_from_slice(&(level.len() as u64).to_le_bytes());
    out.extend_from_slice(&dfd_total.to_le_bytes());
    out.extend_from_slice(&dfd_block);
    out.resize(level_offset as usize, 0);
    out.extend_from_slice(&level);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_image::{CompressedImageFormats, Image, ImageSampler, ImageType};
    use wgpu_types::{TextureDimension, TextureFormat};

    fn test_volume() -> VolumeData {
        let resolution = UVec3::new(3, 2, 4);
        let count = (resolution.x * resolution.y * resolution.z) as usize;
        let faces = (0..count)
            .map(|v| {
                core::array::from_fn(|f| {
                    let base = (v * 6 + f) as f32;
                    [base * 0.25, base * 0.5 + 0.125, 1000.0 - base]
                })
            })
            .collect();
        VolumeData { resolution, faces }
    }

    #[test]
    fn ktx2_round_trips_through_bevys_loader() {
        let volume = test_volume();
        let bytes = encode_ktx2_rgba16f(&volume);
        let image = Image::from_buffer(
            &bytes,
            ImageType::Extension("ktx2"),
            CompressedImageFormats::NONE,
            false,
            ImageSampler::Default,
            Default::default(),
        )
        .expect("Bevy's KTX2 loader reads the volume");
        assert_eq!(image.texture_descriptor.dimension, TextureDimension::D3);
        assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba16Float);
        let r = volume.resolution;
        let size = image.texture_descriptor.size;
        assert_eq!(
            (size.width, size.height, size.depth_or_array_layers),
            (r.x, 2 * r.y, 3 * r.z)
        );
        let data = image.data.as_ref().unwrap();
        for z in 0..r.z {
            for y in 0..r.y {
                for x in 0..r.x {
                    let voxel = UVec3::new(x, y, z);
                    for face in 0..6 {
                        let t = atlas_texel(voxel, face, r);
                        let i = 8 * (t.x + r.x * (t.y + 2 * r.y * t.z)) as usize;
                        for c in 0..3 {
                            let bits = u16::from_le_bytes([data[i + 2 * c], data[i + 2 * c + 1]]);
                            let got = half::f16::from_bits(bits).to_f32();
                            let want = volume.faces[volume.index(voxel)][face][c];
                            assert!((got - want).abs() <= want.abs() * 1e-3, "{got} vs {want}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn atlas_puts_negative_sides_in_the_upper_half() {
        let r = UVec3::new(4, 5, 6);
        assert_eq!(atlas_texel(UVec3::new(1, 2, 3), 0, r), UVec3::new(1, 2, 3));
        assert_eq!(atlas_texel(UVec3::new(1, 2, 3), 1, r), UVec3::new(1, 7, 3));
        assert_eq!(atlas_texel(UVec3::new(1, 2, 3), 2, r), UVec3::new(1, 2, 9));
        assert_eq!(atlas_texel(UVec3::new(1, 2, 3), 5, r), UVec3::new(1, 7, 15));
    }
}
