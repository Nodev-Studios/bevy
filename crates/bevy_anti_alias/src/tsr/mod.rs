//! Temporal super resolution (TSR): a vendor-neutral temporal upscaler.
//!
//! TSR renders the frame below the output's resolution with a moving sub-pixel jitter and rebuilds the full
//! resolution from the current frame and the reprojected history, like DLSS Super Resolution, on any GPU with
//! compute shaders and no vendor SDK. It is Qualcomm's Snapdragon Game Super Resolution 2 (SGSR 2), its 3-pass
//! compute variant, ported to WESL from <https://github.com/SnapdragonStudios/snapdragon-gsr> (BSD-3-Clause; the
//! notice is in each shader file).
//!
//! # Usage
//! 1. Check for `Option<Res<TsrSupported>>` at run time: it exists when the device can run TSR.
//! 2. Add [`Tsr`] to a 3D camera with [`Msaa::Off`](bevy_render::view::Msaa::Off), and no other temporal
//!    anti-aliasing or upscaler.
//! 3. Custom rendering code, including third party crates, should account for the optional
//!    `MainPassResolutionOverride`, as with DLSS (see the `custom_render_phase` example).
//!
//! SGSR 2 compares the frame before and after its transparent objects to keep them from ghosting. Bevy keeps no
//! copy of the frame before its transparent pass, so that part is left out: transparent objects that write no
//! motion vectors may leave trails.

mod extract;
mod node;
mod prepare;

use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    prepass::{DepthPrepass, MotionVectorPrepass},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    render_resource::{
        binding_types::{
            sampler, texture_2d, texture_depth_2d, texture_storage_2d, uniform_buffer,
        },
        AddressMode, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedComputePipelineId,
        ComputePipelineDescriptor, DownlevelFlags, FilterMode, PipelineCache, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, StorageTextureAccess, TextureFormat,
        TextureSampleType, TextureUsages,
    },
    renderer::{RenderAdapter, RenderDevice},
    view::prepare_view_targets,
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::{load_shader_library, Shader};
use bevy_utils::default;
use prepare::TsrUniform;
use tracing::info;

/// Adds [`Tsr`], when the device can run it. Part of [`AntiAliasPlugin`](crate::AntiAliasPlugin).
#[derive(Default)]
pub struct TsrPlugin;

impl Plugin for TsrPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "params.wesl");
        embedded_asset!(app, "convert.wesl");
        embedded_asset!(app, "activate.wesl");
        embedded_asset!(app, "upscale.wesl");

        app.register_type::<Tsr>();
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        if !device_runs_tsr(render_app.world().resource::<RenderAdapter>()) {
            info!("TSR is not supported on this device: it needs compute shaders and storage textures");
            return;
        }

        render_app
            .add_systems(RenderStartup, init_tsr_pipelines)
            .add_systems(ExtractSchedule, extract::extract_tsr)
            .add_systems(
                Render,
                prepare::prepare_tsr
                    .in_set(RenderSystems::PrepareViews)
                    .before(prepare_view_targets),
            )
            .add_systems(Core3d, node::tsr.in_set(Core3dSystems::EarlyPostProcess));

        app.insert_resource(TsrSupported);
    }
}

/// Whether the device has what TSR's passes use: compute shaders, and write-only storage textures of the two
/// formats it writes (`R32Uint` and `Rgba16Float`, which is also the format of an [`Hdr`] camera's main texture,
/// the upscale pass's output). Both formats are storage formats in WebGPU's core, so in practice this is every
/// device with compute shaders: everything but WebGL 2.
fn device_runs_tsr(adapter: &RenderAdapter) -> bool {
    adapter
        .get_downlevel_capabilities()
        .flags
        .contains(DownlevelFlags::COMPUTE_SHADERS)
        && [TextureFormat::R32Uint, TextureFormat::Rgba16Float]
            .into_iter()
            .all(|format| {
                adapter
                    .get_texture_format_features(format)
                    .allowed_usages
                    .contains(TextureUsages::STORAGE_BINDING)
            })
}

/// Camera component that renders below the output's resolution and upscales with TSR.
///
/// It replaces [`TemporalAntiAliasing`](crate::taa::TemporalAntiAliasing): a camera has one or the other, and
/// it needs [`Msaa::Off`](bevy_render::view::Msaa::Off). Only perspective cameras are upscaled.
///
/// Only available when the [`TsrSupported`] resource exists.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default, Clone)]
#[require(TemporalJitter, MipBias, DepthPrepass, MotionVectorPrepass, Hdr)]
pub struct Tsr {
    /// The internal resolution as a share of the output's on each axis, from just above 0 to 1. At 1 TSR only
    /// anti-aliases.
    ///
    /// `2.0 / 3.0` is what DLSS calls Quality, `0.58` Balanced and `0.5` Performance.
    pub render_scale: f32,
    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,
}

impl Default for Tsr {
    fn default() -> Self {
        Self {
            render_scale: 2.0 / 3.0,
            reset: false,
        }
    }
}

/// When TSR can run on the current device, this resource exists in the main world. Otherwise it is absent.
#[derive(Resource, Clone, Copy, Debug)]
pub struct TsrSupported;

/// The bind group layouts and samplers of TSR's three passes, and their pipelines once a camera has asked for
/// them.
#[derive(Resource)]
pub struct TsrPipelines {
    convert_layout: BindGroupLayoutDescriptor,
    activate_layout: BindGroupLayoutDescriptor,
    upscale_layout: BindGroupLayoutDescriptor,
    point_clamp_sampler: Sampler,
    linear_clamp_sampler: Sampler,
    shaders: [Handle<Shader>; 3],
    /// Queued by the first camera with [`Tsr`], so that an app that never upscales compiles none of them.
    ids: Option<TsrPipelineIds>,
}

#[derive(Clone, Copy)]
struct TsrPipelineIds {
    convert: CachedComputePipelineId,
    activate: CachedComputePipelineId,
    upscale: CachedComputePipelineId,
}

impl TsrPipelines {
    fn queue(&mut self, pipeline_cache: &PipelineCache) -> TsrPipelineIds {
        *self.ids.get_or_insert_with(|| {
            let queue = |label: &'static str,
                         layout: &BindGroupLayoutDescriptor,
                         shader: &Handle<Shader>| {
                pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(label.into()),
                    layout: vec![layout.clone()],
                    shader: shader.clone(),
                    ..default()
                })
            };
            TsrPipelineIds {
                convert: queue(
                    "tsr_convert_pipeline",
                    &self.convert_layout,
                    &self.shaders[0],
                ),
                activate: queue(
                    "tsr_activate_pipeline",
                    &self.activate_layout,
                    &self.shaders[1],
                ),
                upscale: queue(
                    "tsr_upscale_pipeline",
                    &self.upscale_layout,
                    &self.shaders[2],
                ),
            }
        })
    }
}

fn init_tsr_pipelines(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
) {
    let clamp_sampler = |label, filter| {
        render_device.create_sampler(&SamplerDescriptor {
            label: Some(label),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: filter,
            min_filter: filter,
            ..default()
        })
    };

    let convert_layout = BindGroupLayoutDescriptor::new(
        "tsr_convert_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<TsrUniform>(false),
                // The main texture, the depth and the motion vectors, drawn in their top-left render rectangle.
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_depth_2d(),
                texture_2d(TextureSampleType::Float { filterable: false }),
                // YCoCg colour and motion-depth, render-sized.
                texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let activate_layout = BindGroupLayoutDescriptor::new(
        "tsr_activate_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<TsrUniform>(false),
                // Previous luma history, motion-depth and YCoCg colour.
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Uint),
                // Motion, depth clip and luma flag; luma history.
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::WriteOnly),
                sampler(SamplerBindingType::NonFiltering),
            ),
        ),
    );
    let upscale_layout = BindGroupLayoutDescriptor::new(
        "tsr_upscale_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<TsrUniform>(false),
                // Previous history, output-sized; motion, depth clip and luma flag; YCoCg colour.
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Uint),
                // History and the camera's main texture, output-sized.
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    commands.insert_resource(TsrPipelines {
        convert_layout,
        activate_layout,
        upscale_layout,
        point_clamp_sampler: clamp_sampler("tsr_point_clamp_sampler", FilterMode::Nearest),
        linear_clamp_sampler: clamp_sampler("tsr_linear_clamp_sampler", FilterMode::Linear),
        shaders: [
            load_embedded_asset!(asset_server.as_ref(), "convert.wesl"),
            load_embedded_asset!(asset_server.as_ref(), "activate.wesl"),
            load_embedded_asset!(asset_server.as_ref(), "upscale.wesl"),
        ],
        ids: None,
    });
}
