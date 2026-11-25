//! The render-world half of the baker: waits until Solari's scene matches what the main world sent, then runs
//! the kernel chunk by chunk and hands the sums back.

use super::post::{Accumulated, ChunkOutput};
use alloc::{format, string::String, sync::Arc, vec::Vec};
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::{
    entity::Entity,
    prelude::*,
    system::{Local, Res},
};
use bevy_platform::collections::HashMap;
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
use std::{sync::Mutex, time::Instant};
use tracing::{info, warn};

/// Threads per voxel: each sums `samples / BATCHES` paths. Fixed, so the sums do not depend on the chunking.
pub(crate) const BATCHES: u32 = 16;
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
    pub directions: Vec<[f32; 4]>,
    /// Every light Solari should have, by main-world entity.
    pub light_keys: HashMap<Entity, CanonicalKey>,
}

/// What came back.
pub(crate) struct BakeOutcome {
    pub accumulated: Accumulated,
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

    let ready = pipeline
        .as_ref()
        .and_then(|p| pipeline_cache.get_compute_pipeline(p.pipeline))
        .is_some()
        && scene.bind_group.is_some();
    let lights: Vec<(RaytracingLightKind, Option<Entity>)> = scene
        .light_sources()
        .map(|(kind, entity)| (kind, main_entities.get(entity).ok().map(MainEntity::id)))
        .collect();
    let matches = lights.len() == job.light_keys.len()
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

    let outcome = bake_chunks(
        &job,
        &canonical
            .iter()
            .map(|(_, index, ..)| *index)
            .collect::<Vec<_>>(),
        &pipeline,
        &pipeline_cache,
        &scene,
        &device,
        &queue,
    )
    .map(
        |(accumulated, seconds, chunks, slowest_chunk_ms)| BakeOutcome {
            accumulated,
            canonical: canonical.iter().map(|(_, _, e, k)| (*e, *k)).collect(),
            adapter: adapter.name.clone(),
            driver: format!("{} {}", adapter.driver, adapter.driver_info),
            seconds,
            chunks,
            slowest_chunk_ms,
        },
    );
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

#[expect(
    clippy::type_complexity,
    reason = "returned once, destructured at the call"
)]
fn bake_chunks(
    job: &BakeJob,
    canonical_to_solari: &[u32],
    pipeline: &BakePipeline,
    pipeline_cache: &PipelineCache,
    scene: &RaytracingSceneBindings,
    device: &RenderDevice,
    queue: &RenderQueue,
) -> Result<(Accumulated, f64, u32, f64), String> {
    let compute_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.pipeline)
        .ok_or("the bake pipeline is not compiled")?;
    let scene_group = scene.bind_group.as_ref().ok_or("no scene bind group")?;
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    let light_count = canonical_to_solari.len();
    let stride = light_count as u64 + 1;
    let total = job.voxels.len();

    let directions = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("gi_bake_directions"),
        contents: bytemuck::cast_slice(&job.directions),
        usage: BufferUsages::STORAGE,
    });
    let mut canonical = canonical_to_solari.to_vec();
    if canonical.is_empty() {
        canonical.push(0);
    }
    let canonical = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("gi_bake_canonical_lights"),
        contents: bytemuck::cast_slice(&canonical),
        usage: BufferUsages::STORAGE,
    });

    let bytes_per_voxel = u64::from(BATCHES) * (6 * 16 + stride * 4 + 8);
    let max_chunk = ((MAX_CHUNK_BUFFER_BYTES / bytes_per_voxel) as usize)
        .clamp(1, (65_535 * WORKGROUP_SIZE / BATCHES) as usize);
    // to tune: small enough that the first submit is short on the slowest GPU here
    let mut chunk = 64usize.min(max_chunk);
    let mut accumulated = Accumulated::new(total, job.region_count, light_count);
    let start_time = Instant::now();
    let (mut first, mut chunks, mut slowest_ms, mut last_log) =
        (0usize, 0u32, 0f64, Instant::now());

    while first < total {
        let count = chunk.min(total - first);
        let threads = count as u64 * u64::from(BATCHES);
        let params = GpuParams {
            voxel_count: count as u32,
            voxel_global_offset: first as u32,
            samples: job.samples,
            batches: BATCHES,
            light_count: light_count as u32,
            mode: job.mode,
            seed: job.seed,
            _padding: 0,
        };
        let params = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gi_bake_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let voxels = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gi_bake_voxels"),
            contents: bytemuck::cast_slice(&job.voxels[first..first + count]),
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
            &layout,
            &BindGroupEntries::sequential((
                params.as_entire_binding(),
                voxels.as_entire_binding(),
                directions.as_entire_binding(),
                canonical.as_entire_binding(),
                faces.as_entire_binding(),
                tallies.as_entire_binding(),
                hits.as_entire_binding(),
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
            pass.set_pipeline(compute_pipeline);
            pass.set_bind_group(0, scene_group, &[]);
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
        queue.submit([encoder.finish()]);
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
            BATCHES as usize,
            job.samples,
            job.invalid_back_face_share,
            &job.voxel_region,
            &ChunkOutput {
                faces: &face_values,
                hits: &hit_values,
                tallies: &tally_values,
            },
        );

        first += count;
        chunks += 1;
        slowest_ms = slowest_ms.max(elapsed_ms);
        // The next chunk aims at the target time; at most double or halve, so one odd chunk cannot swing it.
        let scale = (TARGET_CHUNK_MS / elapsed_ms.max(0.01)).clamp(0.5, 2.0);
        chunk = ((chunk as f64 * scale) as usize).clamp(1, max_chunk);
        if last_log.elapsed().as_secs_f64() > 5.0 {
            last_log = Instant::now();
            info!(
                "baked {first} of {total} voxels ({:.0} %), {:.1} s",
                100.0 * first as f64 / total as f64,
                start_time.elapsed().as_secs_f64()
            );
        }
    }
    Ok((
        accumulated,
        start_time.elapsed().as_secs_f64(),
        chunks,
        slowest_ms,
    ))
}
