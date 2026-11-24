use super::{
    prepare::{TsrContext, TsrTextures},
    TsrPipelines,
};
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::system::Res;
use bevy_math::UVec2;
use bevy_render::{
    diagnostic::RecordDiagnostics,
    render_resource::{
        BindGroup, BindGroupEntries, ComputePassDescriptor, ComputePipeline, PipelineCache,
    },
    renderer::{RenderContext, ViewQuery},
    view::{Msaa, ViewTarget},
};
use tracing::warn;

pub fn tsr(
    view: ViewQuery<(
        &TsrContext,
        &TsrTextures,
        &ViewTarget,
        &ViewPrepassTextures,
        &Msaa,
    )>,
    pipelines: Res<TsrPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (context, textures, view_target, prepass_textures, msaa) = view.into_inner();

    if *msaa != Msaa::Off {
        warn!("TSR requires MSAA to be disabled");
        return;
    }
    let (Some(ids), Some(depth), Some(motion_vectors)) = (
        pipelines.ids,
        &prepass_textures.depth,
        &prepass_textures.motion_vectors,
    ) else {
        return;
    };
    // Until the pipelines are compiled the frame stays at the render resolution, in the main texture's corner.
    let (Some(convert_pipeline), Some(activate_pipeline), Some(upscale_pipeline)) = (
        pipeline_cache.get_compute_pipeline(ids.convert),
        pipeline_cache.get_compute_pipeline(ids.activate),
        pipeline_cache.get_compute_pipeline(ids.upscale),
    ) else {
        return;
    };
    let Some(uniform) = context.uniform.buffer() else {
        return;
    };

    let view_target = view_target.post_process_write();

    let render_device = ctx.render_device();
    let convert_bind_group = render_device.create_bind_group(
        "tsr_convert_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipelines.convert_layout),
        &BindGroupEntries::sequential((
            uniform.as_entire_binding(),
            view_target.source,
            &depth.texture.default_view,
            &motion_vectors.texture.default_view,
            &textures.ycocg_color.default_view,
            &textures.motion_depth.default_view,
        )),
    );
    let activate_bind_group = render_device.create_bind_group(
        "tsr_activate_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipelines.activate_layout),
        &BindGroupEntries::sequential((
            uniform.as_entire_binding(),
            &textures.previous_luma_history.default_view,
            &textures.motion_depth.default_view,
            &textures.ycocg_color.default_view,
            &textures.motion_depth_clip.default_view,
            &textures.luma_history.default_view,
            &pipelines.point_clamp_sampler,
        )),
    );
    let upscale_bind_group = render_device.create_bind_group(
        "tsr_upscale_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipelines.upscale_layout),
        &BindGroupEntries::sequential((
            uniform.as_entire_binding(),
            &textures.previous_history.default_view,
            &textures.motion_depth_clip.default_view,
            &textures.ycocg_color.default_view,
            &textures.history.default_view,
            view_target.destination,
            &pipelines.linear_clamp_sampler,
        )),
    );

    let workgroups = |size: UVec2| (size + 7) / 8;
    let passes: [(&str, &ComputePipeline, &BindGroup, UVec2); 3] = [
        (
            "tsr_convert",
            convert_pipeline,
            &convert_bind_group,
            workgroups(context.render_size),
        ),
        (
            "tsr_activate",
            activate_pipeline,
            &activate_bind_group,
            workgroups(context.render_size),
        ),
        (
            "tsr_upscale",
            upscale_pipeline,
            &upscale_bind_group,
            workgroups(context.output_size),
        ),
    ];

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "tsr");
    for (label, pipeline, bind_group, groups) in passes {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
        let pass_span = diagnostics.pass_span(&mut pass, label);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(groups.x, groups.y, 1);
        pass_span.end(&mut pass);
    }
    time_span.end(ctx.command_encoder());
}
