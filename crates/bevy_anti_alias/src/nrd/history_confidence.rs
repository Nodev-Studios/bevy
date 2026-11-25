//! The history confidence [`Nrd`](super::Nrd) gives `ReBLUR`: how far the history it accumulated can
//! still be trusted, so that a change of light shows in a few frames instead of the tens `ReBLUR`'s
//! history spans.
//!
//! NRD's README prescribes a gradient: last frame's samples traced again with this frame's scene,
//! against what they gave last frame. Solari keeps neither last frame's random numbers nor its
//! lights, so this estimate compares instead this frame's noisy lighting with what NRD showed last
//! frame at the same surface, both summed over tiles of [`TILE_SIZE`] pixels and their neighbours.
//! The sums of a still scene agree within their noise; after a light switches, they differ until
//! NRD has caught up. It needs only the denoiser's own textures, so it lives beside NRD and the
//! lighting pass does not change.
//!
//! One confidence serves both lobes: Solari traces one lobe per pixel, and NRD's documentation
//! advises a single "lighting" confidence for that case.

use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
    render_resource::{
        binding_types::{texture_2d, texture_storage_2d, uniform_buffer},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, CommandEncoder, ComputePassDescriptor, ComputePipelineDescriptor,
        PipelineCache, ShaderStages, ShaderType, StorageTextureAccess, TextureDescriptor,
        TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
        TextureViewDescriptor, UniformBuffer,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::CachedTexture,
};
use bevy_utils::default;

use crate::ray_reconstruction::{
    ViewDenoisedRadianceTextures, ViewDenoiserInputTextures, DENOISER_SKY_VIEW_DEPTH,
};

/// Pixels on each side of a tile. Keep in sync with `history_confidence.wesl`.
const TILE_SIZE: u32 = 16;

/// The settings of the estimate, see `history_confidence.wesl` for each field.
#[derive(ShaderType, Clone, Copy, Default)]
struct HistoryConfidenceParams {
    history_valid: u32,
    radius: u32,
    depth_tolerance: f32,
    change_kept: f32,
    change_dropped: f32,
    luminance_floor: f32,
    sky_depth: f32,
}

/// The settings; `history_valid` is filled in per frame.
///
/// Chosen on the testbed's `interior` at 2560x1440 on the RTX 5070 (2026-11-24, NRD + TAA, bloom off), among
/// six candidates, as the one with the lowest error against the path-traced reference whose light-switch lag
/// stays under 2.6 %: error 2.409 % (2.414 without the estimate), flicker 0.134 % (0.137), lag five frames
/// after a light goes off 1.717 % (6.511). A smaller radius or lower thresholds lowered the lag to 1.39 % but
/// raised the error to 2.55–2.67 %: `ReBLUR` shortens its history every frame the confidence is under 1, so
/// noise in the estimate must map to exactly 1.
// to tune: on other scenes and resolutions; the lag and error move in opposite directions.
const PARAMS: HistoryConfidenceParams = HistoryConfidenceParams {
    history_valid: 0,
    radius: 3,
    depth_tolerance: 0.05,
    change_kept: 0.2,
    change_dropped: 0.45,
    luminance_floor: 0.01,
    sky_depth: DENOISER_SKY_VIEW_DEPTH * 0.5,
};

/// The two passes of the estimate.
#[derive(Resource)]
pub(super) struct HistoryConfidencePipelines {
    gather_layout: BindGroupLayoutDescriptor,
    gather: CachedComputePipelineId,
    resolve_layout: BindGroupLayoutDescriptor,
    resolve: CachedComputePipelineId,
}

pub(super) fn init_history_confidence_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let unfilterable = || texture_2d(TextureSampleType::Float { filterable: false });
    let gather_layout = BindGroupLayoutDescriptor::new(
        "nrd_history_confidence_gather_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                unfilterable(),
                unfilterable(),
                unfilterable(),
                unfilterable(),
                unfilterable(),
                unfilterable(),
                unfilterable(),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<HistoryConfidenceParams>(false),
            ),
        ),
    );
    let resolve_layout = BindGroupLayoutDescriptor::new(
        "nrd_history_confidence_resolve_bind_group_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (8, uniform_buffer::<HistoryConfidenceParams>(false)),
                (9, unfilterable()),
                (
                    10,
                    texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                ),
            ),
        ),
    );
    let shader = load_embedded_asset!(asset_server.as_ref(), "history_confidence.wesl");
    let queue = |label: &'static str, layout: &BindGroupLayoutDescriptor, entry_point| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(entry_point),
            ..default()
        })
    };
    let gather = queue(
        "nrd_history_confidence_gather_pipeline",
        &gather_layout,
        "gather".into(),
    );
    let resolve = queue(
        "nrd_history_confidence_resolve_pipeline",
        &resolve_layout,
        "resolve".into(),
    );
    commands.insert_resource(HistoryConfidencePipelines {
        gather_layout,
        gather,
        resolve_layout,
        resolve,
    });
}

/// The textures of one view's estimate.
pub(super) struct HistoryConfidence {
    /// Per tile: this frame's luminance, last frame's, and the pixels counted.
    tiles: CachedTexture,
    /// What NRD reads, one texel per tile.
    confidence: CachedTexture,
    /// The view depth of the frame NRD's last output belongs to.
    previous_view_depth: CachedTexture,
    params: UniformBuffer<HistoryConfidenceParams>,
    /// False until a frame has been denoised and its view depth kept, and again after a reset.
    history_valid: bool,
}

impl HistoryConfidence {
    pub(super) fn new(
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        size: UVec2,
    ) -> Self {
        let tile_count = (size + (TILE_SIZE - 1)) / TILE_SIZE;
        let create_texture = |label, size: UVec2, format, usage| {
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
            let default_view = texture.create_view(&TextureViewDescriptor::default());
            CachedTexture {
                texture,
                default_view,
            }
        };
        let confidence = create_texture(
            "nrd_history_confidence",
            tile_count,
            TextureFormat::Rgba8Unorm,
            TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST,
        );
        // Full confidence until the first estimate: wgpu would start the texture at 0, a reset.
        render_queue.write_texture(
            confidence.texture.as_image_copy(),
            &vec![u8::MAX; (tile_count.x * tile_count.y * 4) as usize],
            bevy_render::render_resource::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile_count.x * 4),
                rows_per_image: None,
            },
            tile_count.to_extents(),
        );
        Self {
            tiles: create_texture(
                "nrd_history_confidence_tiles",
                tile_count,
                TextureFormat::Rgba32Float,
                TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            ),
            confidence,
            previous_view_depth: create_texture(
                "nrd_history_confidence_previous_view_depth",
                size,
                TextureFormat::R32Float,
                TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            ),
            params: UniformBuffer::default(),
            history_valid: false,
        }
    }

    /// The texture NRD reads.
    pub(super) fn texture(&self) -> &TextureView {
        &self.confidence.default_view
    }

    /// Makes the next estimate give full confidence: last frame's output is not NRD's history.
    pub(super) fn invalidate(&mut self) {
        self.history_valid = false;
    }

    /// Records the estimate for this frame, before NRD runs. `outputs` must still hold NRD's
    /// output of the previous frame.
    pub(super) fn estimate(
        &mut self,
        pipelines: Option<&HistoryConfidencePipelines>,
        pipeline_cache: &PipelineCache,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        encoder: &mut CommandEncoder,
        diagnostics: Option<&DiagnosticsRecorder>,
        inputs: &ViewDenoiserInputTextures,
        motion_vectors: &TextureView,
        outputs: &ViewDenoisedRadianceTextures,
    ) {
        let Some((pipelines, gather, resolve)) = pipelines.and_then(|pipelines| {
            Some((
                pipelines,
                pipeline_cache.get_compute_pipeline(pipelines.gather)?,
                pipeline_cache.get_compute_pipeline(pipelines.resolve)?,
            ))
        }) else {
            // The confidence texture keeps its last values; no view depth is kept for next frame.
            self.history_valid = false;
            return;
        };

        self.params.set(HistoryConfidenceParams {
            history_valid: self.history_valid as u32,
            ..PARAMS
        });
        self.params.write_buffer(render_device, render_queue);
        let Some(params) = self.params.binding() else {
            return;
        };

        let tile_count = self.tiles.texture.size();
        let gather_bind_group = render_device.create_bind_group(
            "nrd_history_confidence_gather_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.gather_layout),
            &BindGroupEntries::sequential((
                &inputs.diffuse_radiance_hit_distance.default_view,
                &inputs.specular_radiance_hit_distance.default_view,
                &inputs.view_depth.default_view,
                motion_vectors,
                &outputs.diffuse.default_view,
                &outputs.specular.default_view,
                &self.previous_view_depth.default_view,
                &self.tiles.default_view,
                params.clone(),
            )),
        );
        let resolve_bind_group = render_device.create_bind_group(
            "nrd_history_confidence_resolve_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.resolve_layout),
            &BindGroupEntries::with_indices((
                (8, params),
                (9, &self.tiles.default_view),
                (10, &self.confidence.default_view),
            )),
        );

        let time_span = diagnostics.time_span(encoder, "nrd_history_confidence");
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("nrd_history_confidence"),
            timestamp_writes: None,
        });
        if self.history_valid {
            pass.set_pipeline(gather);
            pass.set_bind_group(0, &gather_bind_group, &[]);
            pass.dispatch_workgroups(tile_count.width, tile_count.height, 1);
        }
        pass.set_pipeline(resolve);
        pass.set_bind_group(0, &resolve_bind_group, &[]);
        pass.dispatch_workgroups(
            tile_count.width.div_ceil(8),
            tile_count.height.div_ceil(8),
            1,
        );
        drop(pass);

        // NRD's output of this frame will be compared against this frame's surfaces.
        encoder.copy_texture_to_texture(
            inputs.view_depth.texture.as_image_copy(),
            self.previous_view_depth.texture.as_image_copy(),
            inputs.view_depth.texture.size(),
        );
        time_span.end(encoder);
        self.history_valid = true;
    }
}
