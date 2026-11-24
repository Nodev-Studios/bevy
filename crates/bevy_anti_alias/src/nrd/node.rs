use super::{Nrd, NrdCropPipeline, NrdState, ViewNrd};
use crate::ray_reconstruction::{
    ViewDenoisedRadianceTextures, ViewDenoiserGuideTextures, ViewDenoiserInputTextures,
    DENOISER_SKY_VIEW_DEPTH,
};
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::system::Res;
use bevy_image::ToExtents;
use bevy_render::{
    camera::TemporalJitter,
    diagnostic::RecordDiagnostics,
    render_resource::{
        BindGroupEntries, CommandEncoder, ComputePassDescriptor, PipelineCache, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    texture::CachedTexture,
    view::ExtractedView,
};
use nrd_wgpu::{DenoiserInputs, DenoiserOutputs, FrameSettings};
use tracing::error;

/// Runs NRD on one view: the noisy lighting in, the denoised lighting out.
pub(super) fn nrd(
    view: ViewQuery<(
        &Nrd,
        &ViewNrd,
        &ExtractedView,
        Option<&TemporalJitter>,
        &ViewPrepassTextures,
        &ViewDenoiserGuideTextures,
        &ViewDenoiserInputTextures,
        &ViewDenoisedRadianceTextures,
    )>,
    crop_pipeline: Option<Res<NrdCropPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut ctx: RenderContext,
) {
    let (nrd, view_nrd, view, jitter, prepass, guides, inputs, outputs) = view.into_inner();
    let mut state = view_nrd.state.lock().unwrap();
    let Some(nrd_state) = state.as_mut() else {
        // `prepare_nrd` takes the outputs away next frame; until then they hold the noisy light.
        copy_noisy(ctx.command_encoder(), inputs, outputs);
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "nrd");

    let Some(motion_vectors) = motion_vectors(
        nrd_state,
        view_nrd,
        prepass,
        crop_pipeline.as_deref(),
        &pipeline_cache,
        &render_device,
        ctx.command_encoder(),
    ) else {
        copy_noisy(ctx.command_encoder(), inputs, outputs);
        time_span.end(ctx.command_encoder());
        return;
    };

    let frame = FrameSettings {
        view_to_clip: view.clip_from_view.to_cols_array(),
        world_to_view: view.world_from_view.to_matrix().inverse().to_cols_array(),
        // The sign DLSS takes from the same component (`dlss/node.rs`).
        jitter: jitter.map_or([0.0; 2], |jitter| (-jitter.offset).to_array()),
        // Bevy's motion vectors are current minus previous position, in uv.
        motion_vector_scale: [-1.0, -1.0],
        frame_index: nrd_state.frame_index,
        // Anything nearer than the sky marker is denoised.
        denoising_range: DENOISER_SKY_VIEW_DEPTH * 0.5,
        reset_history: nrd.reset,
    };
    let result = nrd_state.denoiser.denoise(
        render_device.wgpu_device(),
        &render_queue,
        ctx.command_encoder(),
        &frame,
        &DenoiserInputs {
            diffuse: &inputs.diffuse_radiance_hit_distance.default_view,
            specular: &inputs.specular_radiance_hit_distance.default_view,
            normal_roughness: &guides.normal_roughness.default_view,
            view_depth: &inputs.view_depth.default_view,
            motion_vectors: &motion_vectors,
        },
        &DenoiserOutputs {
            diffuse: &outputs.diffuse.default_view,
            specular: &outputs.specular.default_view,
        },
    );
    nrd_state.frame_index = nrd_state.frame_index.wrapping_add(1);
    if let Err(error) = result {
        error!("NRD failed; this view shows the noisy lighting from now on: {error}");
        *state = None;
        copy_noisy(ctx.command_encoder(), inputs, outputs);
    }
    time_span.end(ctx.command_encoder());
}

/// The view's motion vectors at the render resolution, or `None` while the pass that cuts them to
/// it is still compiling.
fn motion_vectors(
    state: &mut NrdState,
    view_nrd: &ViewNrd,
    prepass: &ViewPrepassTextures,
    crop_pipeline: Option<&NrdCropPipeline>,
    pipeline_cache: &PipelineCache,
    render_device: &RenderDevice,
    encoder: &mut CommandEncoder,
) -> Option<bevy_render::render_resource::TextureView> {
    let prepass = &prepass.motion_vectors.as_ref()?.texture;
    let prepass_size = prepass.texture.size();
    if prepass_size.width == view_nrd.size.x && prepass_size.height == view_nrd.size.y {
        return Some(prepass.default_view.clone());
    }

    // Under an upscaler the prepass texture has the output's size and the frame is drawn in its
    // top-left corner; NRD wants textures of the render resolution.
    let crop_pipeline = crop_pipeline?;
    let pipeline = pipeline_cache.get_compute_pipeline(crop_pipeline.pipeline)?;
    let cropped = state.motion_vectors.get_or_insert_with(|| {
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("nrd_motion_vectors"),
            size: view_nrd.size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let default_view = texture.create_view(&TextureViewDescriptor::default());
        CachedTexture {
            texture,
            default_view,
        }
    });
    let bind_group = render_device.create_bind_group(
        "nrd_crop_motion_vectors_bind_group",
        &pipeline_cache.get_bind_group_layout(&crop_pipeline.bind_group_layout),
        &BindGroupEntries::sequential((&prepass.default_view, &cropped.default_view)),
    );
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("nrd_crop_motion_vectors"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(view_nrd.size.x.div_ceil(8), view_nrd.size.y.div_ceil(8), 1);
    drop(pass);
    Some(cropped.default_view.clone())
}

/// Fills the outputs with the noisy lighting, for a frame NRD does not denoise.
fn copy_noisy(
    encoder: &mut CommandEncoder,
    inputs: &ViewDenoiserInputTextures,
    outputs: &ViewDenoisedRadianceTextures,
) {
    for (input, output) in [
        (&inputs.diffuse_radiance_hit_distance, &outputs.diffuse),
        (&inputs.specular_radiance_hit_distance, &outputs.specular),
    ] {
        encoder.copy_texture_to_texture(
            input.texture.as_image_copy(),
            output.texture.as_image_copy(),
            output.texture.size(),
        );
    }
}
