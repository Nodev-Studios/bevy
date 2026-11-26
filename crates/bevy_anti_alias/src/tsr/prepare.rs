use super::{
    fsr::{TsrFsr, TsrFsrView},
    Tsr, TsrAlgorithm, TsrFsr3Failure, TsrPipelines,
};
use crate::backend_failure::{capture_device_errors, is_forced, provoke_validation_error};
use bevy_camera::{CameraMainTextureUsages, MainPassResolutionOverride};
use bevy_core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass};
use bevy_diagnostic::FrameCount;
use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::{ops, UVec2, Vec2, Vec4Swizzles};
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    render_resource::{
        PipelineCache, ShaderType, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages, UniformBuffer,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::{CachedTexture, TextureCache},
    view::ExtractedView,
};
use tracing::info;

/// `TsrParams` in `params.wesl`.
#[derive(ShaderType, Clone, Default)]
pub struct TsrUniform {
    render_size: UVec2,
    display_size: UVec2,
    render_size_rcp: Vec2,
    display_size_rcp: Vec2,
    jitter_offset: Vec2,
    camera_fov_angle_hor: f32,
    reset: u32,
}

/// What TSR keeps on a view from one frame to the next.
#[derive(Component)]
pub struct TsrContext {
    pub render_size: UVec2,
    pub output_size: UVec2,
    pub uniform: UniformBuffer<TsrUniform>,
}

/// TSR's textures for this frame. The history pairs swap roles every frame.
#[derive(Component)]
pub struct TsrTextures {
    pub ycocg_color: CachedTexture,
    pub motion_depth: CachedTexture,
    pub motion_depth_clip: CachedTexture,
    pub previous_luma_history: CachedTexture,
    pub luma_history: CachedTexture,
    pub previous_history: CachedTexture,
    pub history: CachedTexture,
}

pub fn prepare_tsr(
    mut query: Query<
        (
            Entity,
            &ExtractedView,
            &Tsr,
            &mut CameraMainTextureUsages,
            &mut TemporalJitter,
            &mut MipBias,
            Option<&mut TsrContext>,
            Option<&TsrFsrView>,
        ),
        (With<DepthPrepass>, With<MotionVectorPrepass>),
    >,
    mut pipelines: ResMut<TsrPipelines>,
    mut fsr: ResMut<TsrFsr>,
    fsr_failure: Res<TsrFsr3Failure>,
    pipeline_cache: Res<PipelineCache>,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    for (entity, view, tsr, mut main_texture_usages, mut jitter, mut mip_bias, context, fsr_view) in
        &mut query
    {
        // The upscale pass writes its output into the main texture as a storage texture.
        main_texture_usages.0 |= TextureUsages::STORAGE_BINDING;

        let output_size = view.viewport.zw();
        let render_size = (output_size.as_vec2() * tsr.render_scale.clamp(0.0, 1.0))
            .round()
            .as_uvec2()
            .max(UVec2::ONE);
        let upscale = output_size.as_vec2() / render_size.as_vec2();

        // FSR 2's rule: about eight jitter positions fall in each output pixel per cycle, so the cycle is longer
        // the further the frame is upscaled. Halton (2, 3), as Bevy's TAA uses.
        // to tune: SGSR 2's sample leaves the sequence to the application.
        let phases = (8.0 * upscale.x * upscale.x).ceil() as u32;
        let index = frame_count.0 % phases + 1;
        jitter.offset = Vec2::new(halton(index, 2), halton(index, 3)) - 0.5;
        // FSR 2's texture LOD bias for this ratio; at native resolution it is TAA's -1.
        mip_bias.0 = ops::log2(1.0 / upscale.x) - 1.0;

        let mut entity_commands = commands.entity(entity);
        entity_commands.insert(MainPassResolutionOverride(render_size));

        if tsr.algorithm == TsrAlgorithm::Fsr3 {
            if fsr.supported && fsr_failure.0.has_failed() {
                // FSR 3.1 failed on this device (here or in `tsr_fsr`): SGSR 2 from now on.
                fsr.supported = false;
                fsr.context = None;
            }
            if fsr.supported && fsr.context.is_none() {
                let device = render_device.wgpu_device();
                let (context, error) = capture_device_errors(device, || {
                    let context = wgpu_ffx::FsrContext::new(device);
                    if is_forced("tsr-fsr3") {
                        provoke_validation_error(device);
                    }
                    context
                });
                match error {
                    None => fsr.context = Some(context),
                    Some(error) => {
                        fsr_failure.0.report("TSR's FSR 3.1", error);
                        fsr.supported = false;
                    }
                }
            }
            if let (true, Some(context)) = (fsr.supported, &fsr.context) {
                entity_commands.remove::<(TsrContext, TsrTextures)>();
                // The inputs are output-sized, with the image in their top-left render rectangle: FSR's own
                // dynamic-resolution layout, with the maximum render size set to the textures' size.
                if fsr_view.is_none_or(|fsr_view| fsr_view.output_size != output_size) {
                    let (view, error) = capture_device_errors(render_device.wgpu_device(), || {
                        wgpu_ffx::FsrContext::create_view(
                            context,
                            &render_queue,
                            output_size.to_array(),
                            output_size.to_array(),
                        )
                    });
                    if let Some(error) = error {
                        fsr_failure.0.report("TSR's FSR 3.1", error);
                        entity_commands.remove::<TsrFsrView>();
                        // The next frame runs SGSR 2.
                        continue;
                    }
                    entity_commands.insert(TsrFsrView {
                        view,
                        output_size,
                        render_size: UVec2::ZERO,
                    });
                }
                continue;
            }
            bevy_utils::once!(info!(
                "TSR runs SGSR 2: its FSR 3.1 needs Vulkan, with wgpu's SPIR-V passthrough"
            ));
        }
        entity_commands.remove::<TsrFsrView>();
        pipelines.queue(&pipeline_cache);

        let fresh = context.as_ref().is_none_or(|context| {
            context.render_size != render_size || context.output_size != output_size
        });
        let mut new_context = None;
        let context = match context {
            Some(context) if !fresh => context.into_inner(),
            _ => new_context.insert(TsrContext {
                render_size,
                output_size,
                uniform: UniformBuffer::default(),
            }),
        };
        context.uniform.set(TsrUniform {
            render_size,
            display_size: output_size,
            render_size_rcp: 1.0 / render_size.as_vec2(),
            display_size_rcp: 1.0 / output_size.as_vec2(),
            // Bevy shifts the projection so that a pixel samples the scene at its centre plus
            // `TemporalJitter::offset`; SGSR 2 expects the centre minus its jitter. DLSS takes the same negation.
            jitter_offset: -jitter.offset,
            // `clip_from_view.x_axis.x` is 1 / tan(horizontal fov / 2) for Bevy's perspective projections, the
            // only ones `extract_tsr` lets through.
            camera_fov_angle_hor: 1.0 / view.clip_from_view.x_axis.x,
            // Textures that are new this frame hold nothing yet.
            reset: u32::from(fresh || tsr.reset),
        });
        context.uniform.write_buffer(&render_device, &render_queue);

        let mut texture = |label, size: UVec2, format| {
            texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: Some(label),
                    size: size.to_extents(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                },
            )
        };
        let ycocg_color = texture("tsr_ycocg_color", render_size, TextureFormat::R32Uint);
        let motion_depth = texture("tsr_motion_depth", render_size, TextureFormat::Rgba16Float);
        let motion_depth_clip = texture(
            "tsr_motion_depth_clip",
            render_size,
            TextureFormat::Rgba16Float,
        );
        let luma_history_1 = texture("tsr_luma_history_1", render_size, TextureFormat::R32Uint);
        let luma_history_2 = texture("tsr_luma_history_2", render_size, TextureFormat::R32Uint);
        let history_1 = texture("tsr_history_1", output_size, TextureFormat::Rgba16Float);
        let history_2 = texture("tsr_history_2", output_size, TextureFormat::Rgba16Float);
        let ((previous_luma_history, luma_history), (previous_history, history)) =
            if frame_count.0.is_multiple_of(2) {
                ((luma_history_2, luma_history_1), (history_2, history_1))
            } else {
                ((luma_history_1, luma_history_2), (history_1, history_2))
            };

        entity_commands.insert(TsrTextures {
            ycocg_color,
            motion_depth,
            motion_depth_clip,
            previous_luma_history,
            luma_history,
            previous_history,
            history,
        });
        if let Some(context) = new_context {
            entity_commands.insert(context);
        }
    }
}

/// Element `index` (from 1) of the Halton sequence in `base`, in `[0, 1)`.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut fraction = 1.0;
    let mut result = 0.0;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}
