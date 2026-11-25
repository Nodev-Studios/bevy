//! Baked reflection probes, the GPU half: captures each region's level-0 cubemap with `reflection.wesl`, after the
//! volumes are baked (their voxels tell whether the box's centre is inside geometry).

use super::{
    gpu::GpuVoxel,
    post::{is_valid, Accumulated},
    reflection::GiBakeReflections,
};
use alloc::{format, string::String, vec, vec::Vec};
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::prelude::*;
use bevy_math::{Affine3A, Vec3};
use bevy_render::{
    render_resource::{
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
        },
        BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntries, Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
        ComputePipelineDescriptor, MapMode, PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue},
};
use bevy_solari::scene::RaytracingSceneBindings;
use bytemuck::{Pod, Zeroable};
use core::ops::Range;
use std::time::Instant;
use tracing::{info, warn};

const WORKGROUP_SIZE: u32 = 64;
/// Each submit aims at this long on the GPU, as the volume passes do. to tune.
const TARGET_CHUNK_MS: f64 = 30.0;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuParams {
    texel_offset: u32,
    texel_count: u32,
    size: u32,
    samples: u32,
    light_count: u32,
    seed: u32,
    probe: u32,
    _padding: u32,
    origin: [f32; 4],
    box_columns: [[f32; 4]; 4],
}

/// One region's probe as the main world asks for it.
pub(crate) struct ProbeJob {
    /// The region box: world from the unit cube.
    pub world_from_box: Affine3A,
}

/// What the main world asks for besides the volumes.
pub(crate) struct ReflectionJob {
    pub settings: GiBakeReflections,
    /// One per region, in the job's region order.
    pub probes: Vec<ProbeJob>,
}

/// One captured probe.
pub(crate) struct ProbeCapture {
    /// Where it was captured.
    pub origin: Vec3,
    /// Level 0: per face, `size × size` texels row by row.
    pub faces: [Vec<[f32; 3]>; 6],
    /// The share of first hits that were back faces at `origin`.
    pub back_face_share: f32,
    /// The box's centre was inside geometry, so the capture moved to the nearest valid voxel.
    pub moved: bool,
}

#[derive(Resource)]
pub(crate) struct ReflectionPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

pub(crate) fn init_reflection_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    scene: Res<RaytracingSceneBindings>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "gi_bake_reflection_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("gi_bake_reflection".into()),
        layout: vec![scene.bind_group_layout.clone(), layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "reflection.wesl"),
        ..Default::default()
    });
    commands.insert_resource(ReflectionPipeline { layout, pipeline });
}

impl ReflectionPipeline {
    /// The compiled pipeline and its layout, once ready.
    pub(crate) fn get<'a>(
        &self,
        cache: &'a PipelineCache,
    ) -> Option<(&'a ComputePipeline, BindGroupLayout)> {
        let pipeline = cache.get_compute_pipeline(self.pipeline)?;
        Some((pipeline, cache.get_bind_group_layout(&self.layout)))
    }
}

/// What the capture shares with the volume passes.
pub(crate) struct Capture<'a> {
    pub pipeline: &'a ComputePipeline,
    pub layout: BindGroupLayout,
    pub scene_group: &'a BindGroup,
    pub canonical: &'a Buffer,
    pub light_info: &'a Buffer,
    pub light_count: u32,
    pub seed: u32,
    pub invalid_back_face_share: f32,
    pub device: &'a RenderDevice,
    pub queue: &'a RenderQueue,
}

impl Capture<'_> {
    /// Captures every region's probe. A region whose centre is inside geometry (more than the voxel rule's share
    /// of back-face first hits) is captured again at its valid voxel nearest the centre.
    pub(crate) fn probes(
        &self,
        job: &ReflectionJob,
        voxels: &[GpuVoxel],
        region_voxels: &[Range<usize>],
        region_samples: &[u32],
        accumulated: &Accumulated,
    ) -> Result<Vec<ProbeCapture>, String> {
        let started = Instant::now();
        let mut texels_per_chunk = 1024u32; // to tune: about 30 ms on the RTX 5070 at 512 paths
        let mut captures = Vec::new();
        for (index, probe) in job.probes.iter().enumerate() {
            let centre = Vec3::from(probe.world_from_box.translation);
            let mut capture =
                self.capture(job, probe, index as u32, centre, &mut texels_per_chunk)?;
            if capture.back_face_share > self.invalid_back_face_share {
                let nearest = region_voxels[index]
                    .clone()
                    .filter(|&voxel| {
                        is_valid(
                            accumulated.back_faces[voxel],
                            region_samples[index],
                            self.invalid_back_face_share,
                        )
                    })
                    .map(|voxel| Vec3::from_slice(&voxels[voxel].position[..3]))
                    .min_by(|a, b| a.distance(centre).total_cmp(&b.distance(centre)));
                match nearest {
                    Some(origin) => {
                        capture =
                            self.capture(job, probe, index as u32, origin, &mut texels_per_chunk)?;
                        capture.moved = true;
                    }
                    None => warn!(
                        "baked GI: probe {index}: the box's centre is inside geometry and no voxel is valid; \
                         captured at the centre"
                    ),
                }
            }
            captures.push(capture);
        }
        info!(
            "baked GI: captured {} reflection probes ({}² per face, {} paths per texel) in {:.2} s",
            captures.len(),
            job.settings.size,
            job.settings.samples,
            started.elapsed().as_secs_f64()
        );
        Ok(captures)
    }

    fn capture(
        &self,
        job: &ReflectionJob,
        probe: &ProbeJob,
        index: u32,
        origin: Vec3,
        texels_per_chunk: &mut u32,
    ) -> Result<ProbeCapture, String> {
        let size = job.settings.size;
        let samples = job.settings.samples;
        let total = 6 * size * size;
        let mut texels: Vec<[f32; 4]> = Vec::with_capacity(total as usize);
        let affine = probe.world_from_box;
        let box_columns = [
            affine.matrix3.x_axis.extend(0.0).to_array(),
            affine.matrix3.y_axis.extend(0.0).to_array(),
            affine.matrix3.z_axis.extend(0.0).to_array(),
            affine.translation.extend(1.0).to_array(),
        ];
        let mut offset = 0;
        while offset < total {
            let count = (*texels_per_chunk).min(total - offset);
            let params = GpuParams {
                texel_offset: offset,
                texel_count: count,
                size,
                samples,
                light_count: self.light_count,
                seed: self.seed,
                probe: index,
                _padding: 0,
                origin: origin.extend(1.0).to_array(),
                box_columns,
            };
            let started = Instant::now();
            texels.extend(self.chunk(&params)?);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            offset += count;
            // As the volume passes: towards the target time, at most double or halve per chunk.
            let scale = (TARGET_CHUNK_MS / elapsed_ms.max(0.01)).clamp(0.5, 2.0);
            *texels_per_chunk = ((f64::from(count) * scale) as u32).clamp(WORKGROUP_SIZE, 1 << 20);
        }
        let per_face = (size * size) as usize;
        let faces = core::array::from_fn(|face| {
            texels[face * per_face..(face + 1) * per_face]
                .iter()
                .map(|t| [t[0], t[1], t[2]])
                .collect()
        });
        let back_faces: f64 = texels.iter().map(|t| f64::from(t[3])).sum();
        Ok(ProbeCapture {
            origin,
            faces,
            back_face_share: (back_faces / (f64::from(total) * f64::from(samples))) as f32,
            moved: false,
        })
    }

    fn chunk(&self, params: &GpuParams) -> Result<Vec<[f32; 4]>, String> {
        let device = self.device;
        let bytes = u64::from(params.texel_count) * 16;
        let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gi_bake_reflection_params"),
            contents: bytemuck::bytes_of(params),
            usage: BufferUsages::UNIFORM,
        });
        let output = device.create_buffer(&BufferDescriptor {
            label: Some("gi_bake_reflection_texels"),
            size: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("gi_bake_reflection_staging"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(
            "gi_bake_reflection_group",
            &self.layout,
            &BindGroupEntries::sequential((
                uniform.as_entire_binding(),
                self.canonical.as_entire_binding(),
                self.light_info.as_entire_binding(),
                output.as_entire_binding(),
            )),
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gi_bake_reflection"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("gi_bake_reflection"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.pipeline);
            pass.set_bind_group(0, self.scene_group, &[]);
            pass.set_bind_group(1, &group, &[]);
            pass.dispatch_workgroups(params.texel_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(MapMode::Read, |result| {
            if let Err(error) = result {
                warn!("could not map a reflection capture buffer: {error}");
            }
        });
        device
            .poll(wgpu_types::PollType::wait_indefinitely())
            .map_err(|error| {
                format!("the GPU failed while capturing a reflection probe: {error}")
            })?;
        let values: Vec<[f32; 4]> = {
            let view = staging
                .slice(..)
                .get_mapped_range()
                .expect("the buffer was mapped for reading");
            bytemuck::cast_slice(&view).to_vec()
        };
        staging.unmap();
        Ok(values)
    }
}

/// The label of a probe's region for logs; kept beside the capture so logs name regions, not indices.
pub(crate) fn describe(name: &str, capture: &ProbeCapture) -> String {
    format!(
        "baked GI: region {name}: reflection probe at ({:.2}, {:.2}, {:.2}){}, {:.1} % back-face first hits",
        capture.origin.x,
        capture.origin.y,
        capture.origin.z,
        if capture.moved {
            ", moved off the box's centre (inside geometry)"
        } else {
            ""
        },
        100.0 * capture.back_face_share
    )
}
