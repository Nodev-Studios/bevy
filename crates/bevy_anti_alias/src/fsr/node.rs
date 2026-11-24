use super::{prepare::FsrViewContext, Fsr};
use bevy_camera::MainPassResolutionOverride;
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::prelude::*;
use bevy_math::ops;
use bevy_render::{
    camera::TemporalJitter,
    diagnostic::RecordDiagnostics,
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, Msaa, ViewTarget},
};
use bevy_time::Time;
use fsr_wgpu::FsrRenderParameters;
use tracing::{error, warn};

pub fn fsr(
    view: ViewQuery<(
        &Fsr,
        &FsrViewContext,
        &ExtractedView,
        &MainPassResolutionOverride,
        &TemporalJitter,
        &ViewTarget,
        &ViewPrepassTextures,
        &Msaa,
    )>,
    time: Res<Time>,
    mut ctx: RenderContext,
) {
    let (fsr, context, view, render_size, jitter, view_target, prepass_textures, msaa) =
        view.into_inner();

    if *msaa != Msaa::Off {
        warn!("FSR requires MSAA to be disabled");
        return;
    }
    let (Some(depth), Some(motion_vectors)) =
        (&prepass_textures.depth, &prepass_textures.motion_vectors)
    else {
        return;
    };

    let view_target = view_target.post_process_write();
    let render_size = render_size.0;
    // Bevy's reverse-Z perspective projections, the only ones `extract_fsr` lets through, keep the near plane in
    // `w_axis.z` and 1 / tan(vertical fov / 2) in `y_axis.y`.
    let near_plane = view.clip_from_view.w_axis.z;
    let fov_y = 2.0 * ops::atan(1.0 / view.clip_from_view.y_axis.y);

    let mut upscaler = context.upscaler.lock().unwrap();
    let parameters = FsrRenderParameters {
        color: view_target.source_texture,
        depth: &depth.texture.texture,
        motion_vectors: &motion_vectors.texture.texture,
        output: view_target.destination_texture,
        // Bevy shifts the projection so that a pixel samples the scene at its centre plus
        // `TemporalJitter::offset`; FSR, like DLSS, expects the centre minus its jitter (AMD's sample passes the
        // negation of the offset it renders with, as TSR's FSR 3.1 does).
        jitter_offset: (-jitter.offset).to_array(),
        // Bevy's motion vectors are UV offsets from the previous frame to this one; FSR wants render pixels
        // pointing back to the previous frame, as DLSS does.
        motion_vector_scale: (-render_size.as_vec2()).to_array(),
        render_size: render_size.to_array(),
        reset: fsr.reset || context.reset,
        // FSR 3.1's source refuses values under 1 ms (TSR clamps the same way), and the first frame's is 0.
        frame_time_delta_ms: (time.delta_secs() * 1000.0).max(1.0),
        camera_near: near_plane,
        camera_fov_y: fov_y,
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "fsr");
    match upscaler.render(&parameters, ctx.command_encoder()) {
        Ok(command_buffer) => ctx.add_command_buffer(command_buffer),
        Err(render_error) => bevy_utils::once!(error!("FSR did not run: {render_error}")),
    }
    time_span.end(ctx.command_encoder());
}
