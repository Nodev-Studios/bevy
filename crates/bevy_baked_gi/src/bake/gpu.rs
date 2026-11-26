//! The render-world half of the baker: waits until Solari's scene matches what the main world sent, then runs
//! the kernel chunk by chunk and hands the sums back.

use super::{
    directions::{fibonacci_directions, voxel_rotation},
    post::{region_noise, Accumulated, ChunkOutput},
    reflection_gpu::{Capture, ProbeCapture, ReflectionJob, ReflectionPipeline},
    GiBakeAdaptive,
};
use alloc::{format, string::String, sync::Arc, vec, vec::Vec};
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::{
    entity::Entity,
    prelude::*,
    system::{Local, Res},
};
use bevy_math::Mat3;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::{
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
        },
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
        BufferDescriptor, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
        CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
        PipelineCache, ShaderStages,
    },
    renderer::{RenderAdapterInfo, RenderDevice, RenderQueue},
    sync_world::MainEntity,
};
use bevy_solari::scene::{RaytracingLightKind, RaytracingSceneBindings};
use bytemuck::{Pod, Zeroable};
use core::ops::Range;
use std::{sync::Mutex, time::Instant};
use tracing::{info, warn};

/// Paths each thread sums. Fixed, so a voxel's threads (`samples / PATHS_PER_THREAD`) and the order its sums are
/// added in depend only on the sample count, never on the chunking, and a chunk keeps the GPU full however many
/// paths a voxel takes. to tune: 256 is what the first bakes used (4096 paths over 16 threads).
pub(crate) const PATHS_PER_THREAD: u32 = 256;
const WORKGROUP_SIZE: u32 = 64;
/// Frames the render world waits once Solari's light array matches the main world's lights, for the last BLAS
/// and TLAS builds to land. to tune: BLASes are built the frame their mesh is prepared; this is slack.
const SETTLE_FRAMES: u32 = 10;
/// Frames after which a light array that never matches stops the bake.
const GIVE_UP_FRAMES: u32 = 2000;
/// Each submit aims at this long on the GPU, so the desktop sharing the GPU stays responsive. to tune.
const TARGET_CHUNK_MS: f64 = 30.0;
/// The output buffers of one chunk stay under this size.
const MAX_CHUNK_BUFFER_BYTES: u64 = 64 << 20;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct GpuVoxel {
    pub position: [f32; 4],
    pub rotation: [[f32; 4]; 3],
}

/// A rotation as the kernel reads it: three columns.
pub(crate) fn rotation_columns(rotation: Mat3) -> [[f32; 4]; 3] {
    [
        rotation.x_axis.extend(0.0).to_array(),
        rotation.y_axis.extend(0.0).to_array(),
        rotation.z_axis.extend(0.0).to_array(),
    ]
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuParams {
    voxel_count: u32,
    voxel_global_offset: u32,
    samples: u32,
    batches: u32,
    light_count: u32,
    mode: u32,
    seed: u32,
    _padding: u32,
}

/// Orders lights the same way in every run: by kind, then where they are, then their content.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CanonicalKey {
    pub kind: u8,
    pub position_cm: [i32; 3],
    pub content: u64,
}

/// What the main world asks the render world to bake.
pub(crate) struct BakeJob {
    pub mode: u32,
    pub samples: u32,
    pub seed: u32,
    pub invalid_back_face_share: f32,
    pub voxels: Vec<GpuVoxel>,
    pub voxel_region: Vec<u32>,
    pub region_count: usize,
    /// Each region's voxels, contiguous in `voxels`.
    pub region_voxels: Vec<Range<usize>>,
    /// `None`: every region at `samples` paths.
    pub adaptive: Option<GiBakeAdaptive>,
    /// Every light Solari should have, by main-world entity.
    pub light_keys: HashMap<Entity, CanonicalKey>,
    /// Position and power of the lights next-event estimation picks by power / distance² (point, spot and
    /// rect lights); the others are picked at 1 / light count.
    pub positioned_lights: HashMap<Entity, [f32; 4]>,
    /// The reflection probes to capture after the volumes; `None`: no probes.
    pub reflections: Option<ReflectionJob>,
}

/// What came back.
pub(crate) struct BakeOutcome {
    pub accumulated: Accumulated,
    /// Paths per voxel in each region.
    pub region_samples: Vec<u32>,
    /// In adaptive mode, each region's noise between the two pilots.
    pub region_noise: Option<Vec<f64>>,
    /// Each region's reflection probe, in region order; empty when the job asked for none.
    pub reflections: Vec<ProbeCapture>,
    /// The lights in canonical order (the tallies' order): main-world entity and kind.
    pub canonical: Vec<(Entity, RaytracingLightKind)>,
    pub adapter: String,
    pub driver: String,
    pub seconds: f64,
    pub chunks: u32,
    pub slowest_chunk_ms: f64,
}

#[derive(Default)]
pub(crate) struct Exchange {
    pub job: Option<Arc<BakeJob>>,
    pub outcome: Option<Result<BakeOutcome, String>>,
}

/// Shared by both worlds: the main world puts a job in, the render world puts the outcome back.
#[derive(Resource, Clone, Default)]
pub(crate) struct BakeExchange(pub Arc<Mutex<Exchange>>);

#[derive(Resource)]
pub(crate) struct BakePipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

pub(crate) fn init_bake_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    scene: Res<RaytracingSceneBindings>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "gi_bake_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("gi_bake".into()),
        layout: vec![scene.bind_group_layout.clone(), layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "bake.wesl"),
        ..Default::default()
    });
    commands.insert_resource(BakePipeline { layout, pipeline });
}

#[derive(Default)]
pub(crate) struct WaitState {
    settled: u32,
    waited: u32,
}

/// Runs after this frame's TLAS build was submitted, so the kernel traces it.
pub(crate) fn run_bake(
    exchange: Res<BakeExchange>,
    pipeline: Option<Res<BakePipeline>>,
    reflection_pipeline: Option<Res<ReflectionPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    scene: Res<RaytracingSceneBindings>,
    main_entities: Query<&MainEntity>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    adapter: Res<RenderAdapterInfo>,
    mut wait: Local<WaitState>,
) {
    let job = {
        let exchange = exchange.0.lock().unwrap();
        if exchange.outcome.is_some() {
            return;
        }
        let Some(job) = exchange.job.clone() else {
            return;
        };
        job
    };
    wait.waited += 1;

    let reflection = reflection_pipeline
        .as_ref()
        .and_then(|p| p.get(&pipeline_cache));
    let ready = pipeline
        .as_ref()
        .and_then(|p| pipeline_cache.get_compute_pipeline(p.pipeline))
        .is_some()
        && (job.reflections.is_none() || reflection.is_some())
        && scene.bind_group.is_some();
    let lights: Vec<(RaytracingLightKind, Option<Entity>)> = scene
        .light_sources()
        .map(|(kind, entity)| (kind, main_entities.get(entity).ok().map(MainEntity::id)))
        .collect();
    // An emissive mesh over 65 535 triangles is several light sources (chunks) of one entity
    let light_entities: HashSet<Option<Entity>> = lights.iter().map(|(_, e)| *e).collect();
    let matches = light_entities.len() == job.light_keys.len()
        && lights
            .iter()
            .all(|(_, entity)| entity.is_some_and(|e| job.light_keys.contains_key(&e)));
    if !(ready && matches) {
        wait.settled = 0;
        if wait.waited.is_multiple_of(240) {
            info!(
                "waiting for the traced scene: pipeline and bind group {}, {} of {} lights in Solari's array",
                if ready { "ready" } else { "not ready" },
                lights.len(),
                job.light_keys.len()
            );
        }
        if wait.waited >= GIVE_UP_FRAMES {
            exchange.0.lock().unwrap().outcome = Some(Err(format!(
                "the traced scene never matched the level: Solari has {} lights, the level {}",
                lights.len(),
                job.light_keys.len()
            )));
        }
        return;
    }
    wait.settled += 1;
    if wait.settled < SETTLE_FRAMES {
        return;
    }

    let pipeline = pipeline.unwrap();
    let mut canonical: Vec<(CanonicalKey, u32, Entity, RaytracingLightKind)> = lights
        .iter()
        .enumerate()
        .map(|(index, (kind, entity))| {
            let entity = entity.unwrap();
            (job.light_keys[&entity], index as u32, entity, *kind)
        })
        .collect();
    canonical.sort_by_key(|(key, ..)| *key);

    let light_info: Vec<[f32; 4]> = canonical
        .iter()
        .map(|(_, _, entity, _)| {
            job.positioned_lights
                .get(entity)
                .copied()
                .unwrap_or([0.0, 0.0, 0.0, -1.0])
        })
        .collect();
    let canonical_to_solari: Vec<u32> = canonical.iter().map(|(_, index, ..)| *index).collect();
    let outcome = PassContext::new(
        &job,
        &canonical_to_solari,
        &light_info,
        &pipeline,
        &pipeline_cache,
        &scene,
        &device,
        &queue,
    )
    .and_then(|mut context| {
        let (accumulated, region_samples, region_noise) = bake_level(&mut context)?;
        let reflections = match (&job.reflections, reflection) {
            (Some(reflections), Some((pipeline, layout))) => Capture {
                pipeline,
                layout,
                scene_group: context.scene_group,
                canonical: &context.canonical,
                light_info: &context.light_info,
                light_count: context.light_count as u32,
                seed: job.seed,
                invalid_back_face_share: job.invalid_back_face_share,
                device: &device,
                queue: &queue,
            }
            .probes(
                reflections,
                &job.voxels,
                &job.region_voxels,
                &region_samples,
                &accumulated,
            )?,
            _ => Vec::new(),
        };
        Ok(BakeOutcome {
            accumulated,
            region_samples,
            region_noise,
            reflections,
            canonical: canonical.iter().map(|(_, _, e, k)| (*e, *k)).collect(),
            adapter: adapter.name.clone(),
            driver: format!("{} {}", adapter.driver, adapter.driver_info),
            seconds: context.started.elapsed().as_secs_f64(),
            chunks: context.chunks,
            slowest_chunk_ms: context.slowest_ms,
        })
    });
    exchange.0.lock().unwrap().outcome = Some(outcome);
}

fn storage(device: &RenderDevice, label: &str, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn staging(device: &RenderDevice, label: &str, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read<T: Pod>(buffer: &Buffer, count: usize) -> Vec<T> {
    let slice = buffer.slice(..);
    let view = slice
        .get_mapped_range()
        .expect("the buffer was mapped for reading");
    let values: &[T] = bytemuck::cast_slice(&view);
    values[..count].to_vec()
}

/// What every pass shares: the pipeline, the scene and the light buffers.
struct PassContext<'a> {
    job: &'a BakeJob,
    light_count: usize,
    canonical: Buffer,
    light_info: Buffer,
    compute_pipeline: &'a bevy_render::render_resource::ComputePipeline,
    scene_group: &'a bevy_render::render_resource::BindGroup,
    layout: bevy_render::render_resource::BindGroupLayout,
    device: &'a RenderDevice,
    queue: &'a RenderQueue,
    /// The rotated sets' base directions, one buffer per path count.
    directions: HashMap<u32, Buffer>,
    /// Paths in the next chunk: adapts towards `TARGET_CHUNK_MS` across passes and regions.
    chunk_paths: u64,
    started: Instant,
    chunks: u32,
    slowest_ms: f64,
    last_log: Instant,
}

impl<'a> PassContext<'a> {
    fn new(
        job: &'a BakeJob,
        canonical_to_solari: &[u32],
        light_info: &[[f32; 4]],
        pipeline: &BakePipeline,
        pipeline_cache: &'a PipelineCache,
        scene: &'a RaytracingSceneBindings,
        device: &'a RenderDevice,
        queue: &'a RenderQueue,
    ) -> Result<Self, String> {
        let compute_pipeline = pipeline_cache
            .get_compute_pipeline(pipeline.pipeline)
            .ok_or("the bake pipeline is not compiled")?;
        let scene_group = scene.bind_group.as_ref().ok_or("no scene bind group")?;
        let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
        let mut canonical = canonical_to_solari.to_vec();
        if canonical.is_empty() {
            canonical.push(0);
        }
        let canonical = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gi_bake_canonical_lights"),
            contents: bytemuck::cast_slice(&canonical),
            usage: BufferUsages::STORAGE,
        });
        let mut light_info = light_info.to_vec();
        if light_info.is_empty() {
            light_info.push([0.0; 4]);
        }
        let light_info = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gi_bake_light_info"),
            contents: bytemuck::cast_slice(&light_info),
            usage: BufferUsages::STORAGE,
        });
        Ok(Self {
            job,
            light_count: canonical_to_solari.len(),
            canonical,
            light_info,
            compute_pipeline,
            scene_group,
            layout,
            device,
            queue,
            directions: HashMap::default(),
            // to tune: 64 voxels of 4096 paths, short on the slowest GPU here
            chunk_paths: 64 * 4096,
            started: Instant::now(),
            chunks: 0,
            slowest_ms: 0.0,
            last_log: Instant::now(),
        })
    }

    /// Bakes every region at its own path count, with `voxels`' rotations and `seed`.
    fn pass(
        &mut self,
        label: &str,
        voxels: &[GpuVoxel],
        seed: u32,
        region_samples: &[u32],
    ) -> Result<Accumulated, String> {
        let job = self.job;
        let mut accumulated = Accumulated::new(voxels.len(), job.region_count, self.light_count);
        let total_paths: u64 = job
            .region_voxels
            .iter()
            .zip(region_samples)
            .map(|(range, samples)| range.len() as u64 * u64::from(*samples))
            .sum();
        let mut done_paths = 0u64;
        for (range, &samples) in job.region_voxels.iter().zip(region_samples) {
            let mut first = range.start;
            while first < range.end {
                let count =
                    self.chunk(&mut accumulated, voxels, first, range.end, samples, seed)?;
                first += count;
                done_paths += count as u64 * u64::from(samples);
                if self.last_log.elapsed().as_secs_f64() > 5.0 {
                    self.last_log = Instant::now();
                    info!(
                        "baked GI: {label} {:.0} % of its paths, {:.1} s",
                        100.0 * done_paths as f64 / total_paths as f64,
                        self.started.elapsed().as_secs_f64()
                    );
                }
            }
        }
        Ok(accumulated)
    }

    /// Runs one chunk of voxels starting at `first`, all at `samples` paths; returns how many voxels it took.
    fn chunk(
        &mut self,
        accumulated: &mut Accumulated,
        voxels: &[GpuVoxel],
        first: usize,
        end: usize,
        samples: u32,
        seed: u32,
    ) -> Result<usize, String> {
        let job = self.job;
        let device = self.device;
        let stride = self.light_count as u64 + 1;
        let batches = samples / PATHS_PER_THREAD;
        let bytes_per_voxel = u64::from(batches) * (6 * 16 + stride * 4 + 8);
        let max_chunk = ((MAX_CHUNK_BUFFER_BYTES / bytes_per_voxel) as usize)
            .clamp(1, (65_535 * WORKGROUP_SIZE / batches).max(1) as usize);
        let count = ((self.chunk_paths / u64::from(samples)) as usize)
            .clamp(1, max_chunk)
            .min(end - first);
        let directions = self.directions.entry(samples).or_insert_with(|| {
            let base: Vec<[f32; 4]> = fibonacci_directions(samples)
                .into_iter()
                .map(|d| d.extend(0.0).to_array())
                .collect();
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("gi_bake_directions"),
                contents: bytemuck::cast_slice(&base),
                usage: BufferUsages::STORAGE,
            })
        });

        let threads = count as u64 * u64::from(batches);
        let params = GpuParams {
            voxel_count: count as u32,
            voxel_global_offset: first as u32,
            samples,
            batches,
            light_count: self.light_count as u32,
            mode: job.mode,
            seed,
            _padding: 0,
        };
        let params = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gi_bake_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let voxel_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gi_bake_voxels"),
            contents: bytemuck::cast_slice(&voxels[first..first + count]),
            usage: BufferUsages::STORAGE,
        });
        let sizes = [threads * 6 * 16, threads * stride * 4, threads * 8];
        let faces = storage(device, "gi_bake_faces", sizes[0]);
        let tallies = storage(device, "gi_bake_tallies", sizes[1]);
        let hits = storage(device, "gi_bake_first_hits", sizes[2]);
        let staged = [
            staging(device, "gi_bake_faces_staging", sizes[0]),
            staging(device, "gi_bake_tallies_staging", sizes[1]),
            staging(device, "gi_bake_first_hits_staging", sizes[2]),
        ];
        let group = device.create_bind_group(
            "gi_bake_group",
            &self.layout,
            &BindGroupEntries::sequential((
                params.as_entire_binding(),
                voxel_buffer.as_entire_binding(),
                directions.as_entire_binding(),
                self.canonical.as_entire_binding(),
                faces.as_entire_binding(),
                tallies.as_entire_binding(),
                hits.as_entire_binding(),
                self.light_info.as_entire_binding(),
            )),
        );

        let chunk_start = Instant::now();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gi_bake"),
        });
        encoder.clear_buffer(&tallies, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("gi_bake"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.compute_pipeline);
            pass.set_bind_group(0, self.scene_group, &[]);
            pass.set_bind_group(1, &group, &[]);
            pass.dispatch_workgroups((threads as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        for ((source, target), size) in [&faces, &tallies, &hits]
            .into_iter()
            .zip(&staged)
            .zip(sizes)
        {
            encoder.copy_buffer_to_buffer(source, 0, target, 0, size.max(16));
        }
        self.queue.submit([encoder.finish()]);
        for buffer in &staged {
            buffer.slice(..).map_async(MapMode::Read, |result| {
                if let Err(error) = result {
                    warn!("could not map a bake buffer: {error}");
                }
            });
        }
        device
            .poll(wgpu_types::PollType::wait_indefinitely())
            .map_err(|error| format!("the GPU failed while baking: {error}"))?;
        let elapsed_ms = chunk_start.elapsed().as_secs_f64() * 1000.0;

        let threads = threads as usize;
        let face_values: Vec<[f32; 4]> = read(&staged[0], threads * 6);
        let tally_values: Vec<f32> = read(&staged[1], threads * stride as usize);
        let hit_values: Vec<[u32; 2]> = read(&staged[2], threads);
        for buffer in &staged {
            buffer.unmap();
        }
        accumulated.add_chunk(
            first,
            count,
            batches as usize,
            samples,
            job.invalid_back_face_share,
            &job.voxel_region,
            &ChunkOutput {
                faces: &face_values,
                hits: &hit_values,
                tallies: &tally_values,
            },
        );

        self.chunks += 1;
        self.slowest_ms = self.slowest_ms.max(elapsed_ms);
        // The next chunk aims at the target time; at most double or halve, so one odd chunk cannot swing it.
        let scale = (TARGET_CHUNK_MS / elapsed_ms.max(0.01)).clamp(0.5, 2.0);
        let paths = (count as u64 * u64::from(samples)) as f64;
        self.chunk_paths = ((paths * scale) as u64).max(u64::from(PATHS_PER_THREAD));
        Ok(count)
    }
}

/// The fixed rotations of the two pilots: not the bake's seed, so every rotation of a level gets the same path
/// counts.
const PILOT_SEEDS: [u32; 2] = [0x7069_6c01, 0x7069_6c02];

/// Bakes the level: in adaptive mode two pilots measure each region's noise and set its path count first.
fn bake_level(
    context: &mut PassContext,
) -> Result<(Accumulated, Vec<u32>, Option<Vec<f64>>), String> {
    let job = context.job;
    let Some(adaptive) = job.adaptive else {
        let samples = vec![job.samples; job.region_count];
        let accumulated = context.pass("bake", &job.voxels, job.seed, &samples)?;
        return Ok((accumulated, samples, None));
    };
    let pilot_samples = vec![adaptive.pilot_samples; job.region_count];
    let mut pilots = Vec::new();
    for (index, seed) in PILOT_SEEDS.into_iter().enumerate() {
        let voxels: Vec<GpuVoxel> = job
            .voxels
            .iter()
            .enumerate()
            .map(|(i, voxel)| GpuVoxel {
                position: voxel.position,
                rotation: rotation_columns(voxel_rotation(seed, i as u32)),
            })
            .collect();
        let label = format!("pilot {} of 2", index + 1);
        pilots.push(context.pass(&label, &voxels, seed, &pilot_samples)?);
    }
    let noise: Vec<f64> = job
        .region_voxels
        .iter()
        .map(|range| {
            region_noise(
                &pilots[0],
                &pilots[1],
                range.clone(),
                adaptive.pilot_samples,
                job.invalid_back_face_share,
            )
        })
        .collect();
    let samples: Vec<u32> = noise.iter().map(|n| adaptive.samples_for(*n)).collect();
    let accumulated = context.pass("bake", &job.voxels, job.seed, &samples)?;
    Ok((accumulated, samples, Some(noise)))
}
