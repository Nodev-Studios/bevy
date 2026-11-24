use super::{Fsr, FsrSdkResource};
use bevy_camera::{Camera3d, CameraMainTextureUsages, MainPassResolutionOverride};
use bevy_core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass};
use bevy_diagnostic::FrameCount;
use bevy_ecs::prelude::*;
use bevy_math::{ops, UVec2, Vec2, Vec4Swizzles};
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    render_resource::TextureUsages,
    view::ExtractedView,
};
use fsr_wgpu::FsrUpscaler;
use std::sync::Mutex;
use tracing::{error, info};

/// A view's FSR context, sized for its output.
#[derive(Component)]
pub struct FsrViewContext {
    pub upscaler: Mutex<FsrUpscaler>,
    /// The output size the context was made for.
    pub output_size: UVec2,
    /// The render size of the frame being prepared.
    pub render_size: UVec2,
    /// The history belongs to another render size or to no frame at all.
    pub reset: bool,
}

pub fn prepare_fsr(
    mut query: Query<
        (
            Entity,
            &ExtractedView,
            &Fsr,
            &mut Camera3d,
            &mut CameraMainTextureUsages,
            &mut TemporalJitter,
            &mut MipBias,
            Option<&mut FsrViewContext>,
        ),
        (With<DepthPrepass>, With<MotionVectorPrepass>),
    >,
    sdk: Res<FsrSdkResource>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    for (
        entity,
        view,
        fsr,
        mut camera_3d,
        mut main_texture_usages,
        mut jitter,
        mut mip_bias,
        context,
    ) in &mut query
    {
        // FSR writes its output as a storage image and reads depth as a sampled image.
        main_texture_usages.0 |= TextureUsages::STORAGE_BINDING;
        let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
        depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
        camera_3d.depth_texture_usages = depth_texture_usages.into();

        let output_size = view.viewport.zw();
        let render_size = (output_size.as_vec2() * fsr.render_scale.clamp(0.0, 1.0))
            .round()
            .as_uvec2()
            .max(UVec2::ONE);

        let mut new_context = None;
        let context = match context {
            Some(context) if context.output_size == output_size => {
                let context = context.into_inner();
                // The node has used the previous frame's reset.
                context.reset = false;
                context
            }
            _ => {
                // The inputs are output-sized, with the image in their top-left render rectangle: the maximum
                // render size is the textures' size, as for TSR's FSR 3.1.
                match FsrUpscaler::new(&sdk.0, output_size.to_array(), output_size.to_array()) {
                    Ok(upscaler) => {
                        info!(
                            "FSR: {}x{} context runs FSR {} (provider id {:#x})",
                            output_size.x,
                            output_size.y,
                            upscaler.provider_version().name,
                            upscaler.provider_version().id
                        );
                        new_context.insert(FsrViewContext {
                            upscaler: Mutex::new(upscaler),
                            output_size,
                            render_size,
                            reset: true,
                        })
                    }
                    Err(context_error) => {
                        bevy_utils::once!(error!("FSR context not created: {context_error}"));
                        commands
                            .entity(entity)
                            .remove::<(FsrViewContext, MainPassResolutionOverride)>();
                        continue;
                    }
                }
            }
        };
        if context.render_size != render_size {
            context.render_size = render_size;
            context.reset = true;
        }

        let upscaler = context.upscaler.get_mut().unwrap();
        jitter.offset = match upscaler.jitter(frame_count.0, render_size.x) {
            Ok(offset) => Vec2::from(offset),
            Err(jitter_error) => {
                bevy_utils::once!(error!("FSR jitter: {jitter_error}"));
                Vec2::ZERO
            }
        };
        // AMD's mip bias for FSR: log2(render / output) - 1.
        mip_bias.0 = ops::log2(render_size.x as f32 / output_size.x as f32) - 1.0;

        let mut entity_commands = commands.entity(entity);
        entity_commands.insert(MainPassResolutionOverride(render_size));
        if let Some(new_context) = new_context {
            entity_commands.insert(new_context);
        }
    }
}
