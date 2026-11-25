//! What the CPU does with the kernel's output: sums each voxel's batches in a fixed order (so a bake is
//! deterministic whatever the chunking), decides which voxels are inside geometry, fills those from their
//! neighbours, and turns the per-light tallies into each region's weights.

use crate::VolumeData;
use alloc::vec::Vec;
use bevy_math::{IVec3, UVec3};

/// Everything summed from the kernel's output, over all regions.
#[derive(Clone, Debug, Default)]
pub(crate) struct Accumulated {
    /// Per voxel, each side's linear RGB in [`crate::FACE_NAMES`] order, already divided by the sample count.
    pub faces: Vec<[[f64; 3]; 6]>,
    /// Per voxel, how many first hits were back faces.
    pub back_faces: Vec<u32>,
    /// First rays that hit nothing, over all voxels.
    pub misses: u64,
    /// First rays traced, over all voxels.
    pub rays: u64,
    /// Per region, the luminance gathered from each light in canonical order, then the static share last. Only
    /// valid voxels count.
    pub region_tallies: Vec<Vec<f64>>,
}

/// One chunk's output as read back.
pub(crate) struct ChunkOutput<'a> {
    /// `threads × 6` sides, `[r, g, b, unused]`.
    pub faces: &'a [[f32; 4]],
    /// `threads` × `[back faces, misses]`.
    pub hits: &'a [[u32; 2]],
    /// `threads × (light_count + 1)`.
    pub tallies: &'a [f32],
}

impl Accumulated {
    pub(crate) fn new(voxel_count: usize, region_count: usize, light_count: usize) -> Self {
        Self {
            faces: vec![[[0.0; 3]; 6]; voxel_count],
            back_faces: vec![0; voxel_count],
            misses: 0,
            rays: 0,
            region_tallies: vec![vec![0.0; light_count + 1]; region_count],
        }
    }

    /// Adds the voxels `first..first + count` from a chunk whose threads are `voxel × batches + batch`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the bake's fixed parameters, passed once per chunk"
    )]
    pub(crate) fn add_chunk(
        &mut self,
        first: usize,
        count: usize,
        batches: usize,
        samples: u32,
        invalid_back_face_share: f32,
        voxel_region: &[u32],
        output: &ChunkOutput,
    ) {
        let stride = self.region_tallies.first().map_or(1, Vec::len);
        for local in 0..count {
            let voxel = first + local;
            let mut sides = [[0.0f64; 3]; 6];
            let (mut back_faces, mut misses) = (0u32, 0u32);
            for batch in 0..batches {
                let thread = local * batches + batch;
                for (side, sum) in sides.iter_mut().enumerate() {
                    let value = output.faces[thread * 6 + side];
                    for c in 0..3 {
                        sum[c] += f64::from(value[c]);
                    }
                }
                back_faces += output.hits[thread][0];
                misses += output.hits[thread][1];
            }
            for side in &mut sides {
                for c in side.iter_mut() {
                    *c /= f64::from(samples);
                }
            }
            self.faces[voxel] = sides;
            self.back_faces[voxel] = back_faces;
            self.misses += u64::from(misses);
            self.rays += u64::from(samples);

            if !is_valid(back_faces, samples, invalid_back_face_share) {
                continue;
            }
            let tallies = &mut self.region_tallies[voxel_region[voxel] as usize];
            for batch in 0..batches {
                let thread = local * batches + batch;
                let row = &output.tallies[thread * stride..(thread + 1) * stride];
                for (total, value) in tallies.iter_mut().zip(row) {
                    *total += f64::from(*value);
                }
            }
        }
    }
}

/// A voxel is inside geometry when more than `share` of its first hits are back faces: from inside a closed
/// mesh every ray leaves through a back face. 0.25 is DDGI's rule (to tune).
pub(crate) fn is_valid(back_faces: u32, samples: u32, share: f32) -> bool {
    (back_faces as f32) <= share * samples as f32
}

/// Gate 4's statistic between two bakes of one region: the RMS, over the sides of voxels valid in both whose
/// luminance in `a` is above 1 % of the region's largest, of (a − b) / ((a + b) / 2). 0 when no side qualifies.
pub(crate) fn region_noise(
    a: &Accumulated,
    b: &Accumulated,
    range: core::ops::Range<usize>,
    samples: u32,
    invalid_back_face_share: f32,
) -> f64 {
    let luminance = |c: [f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let valid = |voxel: usize| {
        is_valid(a.back_faces[voxel], samples, invalid_back_face_share)
            && is_valid(b.back_faces[voxel], samples, invalid_back_face_share)
    };
    let pairs: Vec<(f64, f64)> = range
        .filter(|voxel| valid(*voxel))
        .flat_map(|voxel| {
            (0..6).map(move |side| {
                (
                    luminance(a.faces[voxel][side]),
                    luminance(b.faces[voxel][side]),
                )
            })
        })
        .collect();
    let max = pairs.iter().map(|(x, _)| *x).fold(0.0, f64::max);
    let (sum, count) = pairs.iter().filter(|(x, _)| *x > 0.01 * max).fold(
        (0.0, 0usize),
        |(sum, count), (x, y)| {
            let relative = (x - y) / ((x + y) / 2.0);
            (sum + relative * relative, count + 1)
        },
    );
    if count == 0 {
        0.0
    } else {
        (sum / count as f64).sqrt()
    }
}

/// Replaces each invalid voxel with the mean of its valid face neighbours, growing outwards from the valid ones
/// one layer per pass, inside this volume only (so light never crosses into another region). Each pass reads
/// only the previous pass's values, so the result does not depend on the visiting order.
///
/// Returns the voxels left unfilled: all of them when the volume has no valid voxel.
pub(crate) fn dilate(volume: &mut VolumeData, valid: &mut [bool]) -> usize {
    let r = volume.resolution.as_ivec3();
    let neighbours = [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ];
    loop {
        let mut filled = Vec::new();
        for z in 0..r.z {
            for y in 0..r.y {
                for x in 0..r.x {
                    let voxel = IVec3::new(x, y, z);
                    let index = volume.index(voxel.as_uvec3());
                    if valid[index] {
                        continue;
                    }
                    let mut sum = [[0.0f64; 3]; 6];
                    let mut count = 0u32;
                    for offset in neighbours {
                        let n = voxel + offset;
                        if n.cmplt(IVec3::ZERO).any() || n.cmpge(r).any() {
                            continue;
                        }
                        let n_index = volume.index(n.as_uvec3());
                        if !valid[n_index] {
                            continue;
                        }
                        count += 1;
                        for (s, side) in sum.iter_mut().zip(&volume.faces[n_index]) {
                            for c in 0..3 {
                                s[c] += f64::from(side[c]);
                            }
                        }
                    }
                    if count > 0 {
                        let mean = sum.map(|side| side.map(|c| (c / f64::from(count)) as f32));
                        filled.push((index, mean));
                    }
                }
            }
        }
        if filled.is_empty() {
            return valid.iter().filter(|v| !**v).count();
        }
        for (index, value) in filled {
            volume.faces[index] = value;
            valid[index] = true;
        }
    }
}

/// The volume of the region whose voxels are `range` of `accumulated`, with invalid voxels filled.
/// Returns it, the invalid voxel count, and the voxels that could not be filled.
pub(crate) fn region_volume(
    accumulated: &Accumulated,
    range: core::ops::Range<usize>,
    resolution: UVec3,
    samples: u32,
    invalid_back_face_share: f32,
) -> (VolumeData, u32, usize) {
    let faces = accumulated.faces[range.clone()]
        .iter()
        .map(|sides| sides.map(|side| side.map(|c| c as f32)))
        .collect();
    let mut valid: Vec<bool> = accumulated.back_faces[range]
        .iter()
        .map(|back_faces| is_valid(*back_faces, samples, invalid_back_face_share))
        .collect();
    let invalid = valid.iter().filter(|v| !**v).count() as u32;
    let mut volume = VolumeData { resolution, faces };
    let unfilled = dilate(&mut volume, &mut valid);
    (volume, invalid, unfilled)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accumulated(values: &[f64]) -> Accumulated {
        let mut accumulated = Accumulated::new(values.len(), 1, 0);
        for (voxel, value) in values.iter().enumerate() {
            accumulated.faces[voxel] = [[*value; 3]; 6];
        }
        accumulated
    }

    #[test]
    fn region_noise_is_gate_fours_relative_rms() {
        // Relative differences +0.2 and −0.2 on every side of the two bright voxels; the third is below 1 % of
        // the region's max and the fourth is inside geometry in one bake, so neither counts.
        let a = accumulated(&[1.1, 0.9, 0.001, 5.0]);
        let mut b = accumulated(&[0.9, 1.1, 0.5, 1.0]);
        b.back_faces[3] = 100;
        let noise = region_noise(&a, &b, 0..4, 100, 0.25);
        assert!((noise - 0.2).abs() < 1e-12, "{noise}");
        assert_eq!(region_noise(&a, &a, 0..4, 100, 0.25), 0.0);
    }

    #[test]
    fn dilation_fills_from_the_same_volume_in_layers() {
        let resolution = UVec3::new(3, 1, 1);
        let mut volume = VolumeData {
            resolution,
            faces: vec![[[0.0; 3]; 6], [[0.0; 3]; 6], [[3.0; 3]; 6]],
        };
        let mut valid = vec![false, false, true];
        assert_eq!(dilate(&mut volume, &mut valid), 0);
        assert_eq!(volume.faces[1], [[3.0; 3]; 6]);
        assert_eq!(volume.faces[0], [[3.0; 3]; 6]);
    }

    #[test]
    fn a_volume_with_no_valid_voxel_stays_unfilled() {
        let mut volume = VolumeData {
            resolution: UVec3::new(2, 1, 1),
            faces: vec![[[0.0; 3]; 6]; 2],
        };
        assert_eq!(dilate(&mut volume, &mut [false, false]), 2);
    }

    #[test]
    fn chunks_sum_batches_and_skip_invalid_voxels_in_tallies() {
        // Two voxels, two batches, one light: voxel 1 is inside geometry.
        let faces = vec![[1.0f32, 2.0, 3.0, 0.0]; 2 * 2 * 6];
        let hits = vec![[0u32, 1], [0, 0], [3, 0], [2, 0]];
        let tallies = vec![1.0f32, 0.5, 2.0, 0.25, 10.0, 10.0, 10.0, 10.0];
        let output = ChunkOutput {
            faces: &faces,
            hits: &hits,
            tallies: &tallies,
        };
        let mut accumulated = Accumulated::new(2, 1, 1);
        accumulated.add_chunk(0, 2, 2, 8, 0.25, &[0, 0], &output);
        assert_eq!(accumulated.faces[0][3], [0.25, 0.5, 0.75]);
        assert_eq!(accumulated.back_faces, vec![0, 5]);
        assert_eq!(accumulated.misses, 1);
        assert_eq!(accumulated.region_tallies[0], vec![3.0, 0.75]);
    }
}
