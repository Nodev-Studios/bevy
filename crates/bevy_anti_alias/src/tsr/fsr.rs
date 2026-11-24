//! TSR's FSR 3.1 algorithm: AMD's upscaler from `vendor/wgpu_ffx`, recorded into Bevy's command encoder.

use super::Tsr;
use bevy_camera::MainPassResolutionOverride;
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::prelude::*;
use bevy_math::UVec2;
use bevy_render::{
    camera::TemporalJitter,
    diagnostic::RecordDiagnostics,
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, Msaa, ViewTarget},
};
use bevy_time::Time;
use tracing::{error, warn};
use wgpu_ffx::{FsrContext, FsrDispatchInfo, FsrView};

/// Whether FSR runs on this device, and its pipelines once a camera has asked for them.
#[derive(Resource)]
pub struct TsrFsr {
    pub supported: bool,
    /// Built by the first camera with [`TsrAlgorithm::Fsr3`](super::TsrAlgorithm::Fsr3), so that an app that
    /// never asks for it compiles none of its shaders.
    pub context: Option<FsrContext>,
}

/// A view's FSR state: its textures and history, sized for the output.
#[derive(Component)]
pub struct TsrFsrView {
    pub view: FsrView,
    pub output_size: UVec2,
}

pub fn tsr_fsr(
    view: ViewQuery<(
        &Tsr,
        &mut TsrFsrView,
        &ExtractedView,
        &MainPassResolutionOverride,
        &TemporalJitter,
        &ViewTarget,
        &ViewPrepassTextures,
        &Msaa,
    )>,
    fsr: Res<TsrFsr>,
    time: Res<Time>,
    mut ctx: RenderContext,
) {
    let (tsr, mut fsr_view, view, render_size, jitter, view_target, prepass_textures, msaa) =
        view.into_inner();

    if *msaa != Msaa::Off {
        warn!("TSR requires MSAA to be disabled");
        return;
    }
    let (Some(context), Some(depth), Some(motion_vectors)) = (
        &fsr.context,
        &prepass_textures.depth,
        &prepass_textures.motion_vectors,
    ) else {
        return;
    };

    let view_target = view_target.post_process_write();
    let render_size = render_size.0.as_vec2();
    // Bevy's reverse-Z perspective projections, the only ones `extract_tsr` lets through, keep the near plane in
    // `w_axis.z` and 1 / tan(vertical fov / 2) in `y_axis.y`.
    let near_plane = view.clip_from_view.w_axis.z;
    let fov_y = 2.0 * (1.0 / view.clip_from_view.y_axis.y).atan();

    let info = FsrDispatchInfo {
        color: (**view_target.source_texture).clone(),
        depth: (*depth.texture.texture).clone(),
        motion_vectors: (*motion_vectors.texture.texture).clone(),
        output: (**view_target.destination_texture).clone(),
        // Bevy shifts the projection so that a pixel samples the scene at its centre plus
        // `TemporalJitter::offset`; FSR, like DLSS, expects the centre minus its jitter.
        jitter_offset: (-jitter.offset).to_array(),
        // Bevy's motion vectors are UV offsets from the previous frame to this one; FSR wants render pixels
        // pointing back to the previous frame, as DLSS does.
        motion_vector_scale: (-render_size).to_array(),
        render_size: render_size.as_uvec2().to_array(),
        upscale_size: fsr_view.output_size.to_array(),
        // FSR only reads it to smooth its frame-to-frame luma, and refuses values under 1 ms.
        frame_time_delta: (time.delta_secs() * 1000.0).max(1.0),
        reset_history: tsr.reset,
        // FSR names a reverse-Z depth's planes the other way round: its "near" is the far plane, at infinity.
        camera_near: f32::INFINITY,
        camera_far: near_plane,
        camera_fov_y: fov_y,
        view_space_to_meters_factor: 1.0,
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "tsr");
    if let Err(dispatch_error) = context.dispatch(&mut fsr_view.view, ctx.command_encoder(), &info)
    {
        bevy_utils::once!(error!("TSR's FSR 3.1 did not run: {dispatch_error}"));
    }
    time_span.end(ctx.command_encoder());
}
