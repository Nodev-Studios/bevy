//! NVIDIA's Real-time Denoisers (NRD): `ReBLUR` as the denoiser of ray traced lighting.
//!
//! NRD runs on any GPU the renderer drives through Vulkan, NVIDIA or not. It is the denoiser for
//! a ray traced lighting pass where DLSS Ray Reconstruction cannot run.
//!
//! The NRD library and its shaders come from the NRD SDK, which is not part of this repository;
//! `vendor/nrd_wgpu/README.md` in the repository says how to fetch it and build with it, and
//! what its licence asks of an app that ships it.
//!
//! # Usage
//! 1. Enable Bevy's `nrd` feature, with the NRD SDK fetched.
//! 2. Give a camera that has `bevy_solari`'s `SolariLighting` both `SolariDenoiserInputs` (the
//!    lighting pass then writes [`ViewDenoiserInputTextures`]) and [`Nrd`]. NRD reads no specular
//!    motion vectors: `bevy_solari`'s `SolariSkipSpecularMotionVectors` saves the lighting pass the
//!    work of writing them.
//! 3. Resolve the denoised frame with temporal anti-aliasing or an upscaler, as with any ray
//!    traced lighting.
//!
//! Each frame [`Nrd`] reads the textures of [`crate::ray_reconstruction`] in
//! [`DenoiserSystems::Denoise`] and writes [`ViewDenoisedRadianceTextures`], which the lighting
//! pass composes into the view's lit color. Before `ReBLUR` runs, [`Nrd`] estimates how far its
//! history can still be trusted, so that a change of light shows within a few frames (see
//! `history_confidence.rs`). A view where NRD cannot run (see
//! [`Nrd::required_wgpu_features`]) logs why once and shows the noisy lighting.
//!
//! NRD's shaders and pipelines are created once per render device, inside wgpu error scopes, when the first
//! view asks for NRD. If the driver refuses them, or NRD fails later, [`NrdFailure`] says so in the main world
//! and no view tries NRD again: a settings layer (`bevy_graphics`) then puts another denoiser or rasterized
//! lighting on the camera. When the render size changes, NRD's textures follow once the new size has held for
//! a few frames, without compiling anything; the view shows the noisy lighting until then.

mod history_confidence;
mod node;

use crate::{
    backend_failure::BackendFailure,
    ray_reconstruction::{
        DenoiserSystems, ViewDenoisedRadianceTextures, ViewDenoiserGuideTextures,
        ViewDenoiserInputTextures,
    },
};
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer};
use bevy_camera::{Camera, MainPassResolutionOverride};
use bevy_core_pipeline::schedule::Core3d;
use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        binding_types::{texture_2d, texture_storage_2d},
        BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedComputePipelineId,
        ComputePipelineDescriptor, PipelineCache, ShaderStages, StorageTextureAccess,
        TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
        TextureViewDescriptor,
    },
    renderer::{RenderDevice, RenderQueue},
    settings::WgpuFeatures,
    sync_world::RenderEntity,
    texture::CachedTexture,
    ExtractSchedule, MainWorld, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_utils::default;
use history_confidence::{init_history_confidence_pipelines, HistoryConfidence};
use nrd_wgpu::{Denoiser, DenoiserKind, DenoiserPipelines, InputFormats, TexturePrecision};
use std::sync::Mutex;
use tracing::info;

/// Adds [`Nrd`]. Part of [`AntiAliasPlugin`](crate::AntiAliasPlugin) when the `nrd` feature is on.
#[derive(Default)]
pub struct NrdPlugin;

impl Plugin for NrdPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "crop_motion_vectors.wesl");
        embedded_asset!(app, "history_confidence.wesl");

        let failure = NrdFailure::default();
        app.insert_resource(failure.clone());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(failure)
            .init_resource::<NrdDevicePipelines>()
            .add_systems(
                RenderStartup,
                (init_nrd_crop_pipeline, init_history_confidence_pipelines),
            )
            .add_systems(ExtractSchedule, extract_nrd)
            .add_systems(Render, prepare_nrd.in_set(RenderSystems::PrepareResources))
            .add_systems(Core3d, node::nrd.in_set(DenoiserSystems::Denoise));
    }
}

/// A component for a 3d camera with ray traced lighting, to denoise that lighting with NRD's
/// `ReBLUR`.
///
/// The camera must also make its lighting pass write [`ViewDenoiserInputTextures`]
/// (`bevy_solari`'s `SolariDenoiserInputs`); without them NRD does nothing.
#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component, Default, Clone)]
pub struct Nrd {
    /// The precision of the textures `ReBLUR` keeps its history in.
    pub precision: NrdPrecision,
    /// Set to true to drop the accumulated history, after a camera cut for example. It is set
    /// back to false after the frame it applied to.
    pub reset: bool,
}

/// The precision of the textures [`Nrd`] accumulates in.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Default, Clone, PartialEq)]
pub enum NrdPrecision {
    /// 16-bit floats, as NRD asks for. Both drivers tried (NVIDIA 620.73, Mesa RADV 26.3) round
    /// toward zero when they store a 16-bit float, so `ReBLUR`'s history settles below its target.
    /// Behind TAA on the RTX 5070 at 2560x1440 (2026-11-24) the frame's mean luma came out 1.2 to
    /// 2.0 points darker than with [`Self::Full`] on three scenes, and its error against the
    /// path-traced reference 0.03 to 0.38 points higher.
    Half,
    /// 32-bit floats wherever NRD asks for 16-bit ones. Measured as for [`Self::Half`], the mean
    /// luma was within 0.4 % of the reference on two scenes and 1.6 % dark on the third, where
    /// DLSS Ray Reconstruction is 1.5 % dark. About 1.6 times the memory of [`Self::Half`]: 647
    /// against 401 MiB of textures at 2560x1440, 288 MiB at 1707x960. Needs
    /// `WgpuFeatures::FLOAT32_FILTERABLE`.
    #[default]
    Full,
}

impl NrdPrecision {
    fn texture_precision(self) -> TexturePrecision {
        match self {
            Self::Half => TexturePrecision::Half,
            Self::Full => TexturePrecision::Full,
        }
    }
}

/// Set when NRD failed on this render device, in both the main and the render world (see
/// [`backend_failure`](crate::backend_failure)): the driver refused its shaders, pipelines or textures, or a frame
/// failed. From then on no view runs NRD; views with [`Nrd`] show the noisy lighting until the camera gets
/// another denoiser.
#[derive(Resource, Clone, Default, Debug)]
pub struct NrdFailure(pub BackendFailure);

impl Nrd {
    /// The wgpu features the render device needs for [`Nrd`] at `precision`.
    ///
    /// NRD's shaders are SPIR-V handed to the driver unchanged (`PASSTHROUGH_SHADERS`), they store
    /// to formats base WebGPU does not allow as storage textures
    /// (`TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`), and [`NrdPrecision::Full`] filters 32-bit
    /// float textures (`FLOAT32_FILTERABLE`). All three are Vulkan-only here. Bevy's default
    /// render settings ask for every feature the adapter has, so a Vulkan device normally has
    /// them.
    pub fn required_wgpu_features(precision: NrdPrecision) -> WgpuFeatures {
        Denoiser::required_features(precision.texture_precision())
    }
}

/// The NRD denoiser of one view.
#[derive(Component)]
pub struct ViewNrd {
    /// The size and precision the view renders at, and its outputs have.
    size: UVec2,
    precision: NrdPrecision,
    /// Frames the view has had a size or precision its denoiser does not have yet. The denoiser follows once
    /// it reaches [`RESIZE_AFTER_STABLE_FRAMES`]; until then the view shows the noisy lighting.
    unsettled_frames: u32,
    /// `None` once creating or running the denoiser failed: the view then shows the noisy lighting.
    state: Mutex<Option<NrdState>>,
}

impl ViewNrd {
    /// Whether the denoiser is at the view's size and precision, so it can run this frame.
    fn is_settled(state: &NrdState, size: UVec2, precision: NrdPrecision) -> bool {
        state.denoiser.size() == size.to_array()
            && state.denoiser.precision() == precision.texture_precision()
    }
}

/// How many frames in a row a view must keep a new render size before NRD's textures follow it. A window being
/// dragged, or a quality slider being moved, changes the size every frame; remaking 300-650 MiB of textures each
/// time would stall every one of those frames. to tune: 4 frames is about 70 ms at 60 Hz.
const RESIZE_AFTER_STABLE_FRAMES: u32 = 4;

struct NrdState {
    denoiser: Denoiser,
    frame_index: u32,
    /// The view's motion vectors cut to the render resolution, when the prepass texture is
    /// larger (an upscaler is on).
    motion_vectors: Option<CachedTexture>,
    history_confidence: HistoryConfidence,
}

/// NRD's pipelines for the render device, shared by every view, created when the first view asks for NRD.
#[derive(Resource, Default)]
struct NrdDevicePipelines(Option<DenoiserPipelines>);

/// The formats of what the lighting pass hands NRD.
fn input_formats(
    inputs: &ViewDenoiserInputTextures,
    guides: &ViewDenoiserGuideTextures,
) -> InputFormats {
    InputFormats {
        diffuse: inputs.diffuse_radiance_hit_distance.texture.format(),
        specular: inputs.specular_radiance_hit_distance.texture.format(),
        normal_roughness: guides.normal_roughness.texture.format(),
        view_depth: inputs.view_depth.texture.format(),
        // The prepass's, or the cut copy of `motion_vectors`, which has the same format.
        motion_vectors: TextureFormat::Rg16Float,
        history_confidence: Some(HistoryConfidence::FORMAT),
    }
}

fn extract_nrd(
    mut commands: Commands,
    mut main_world: ResMut<MainWorld>,
    cleanup_query: Query<Has<Nrd>>,
) {
    let mut cameras = main_world.query::<(RenderEntity, &Camera, Option<&mut Nrd>)>();
    for (entity, camera, nrd) in cameras.iter_mut(&mut main_world) {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        match nrd {
            Some(mut nrd) if camera.is_active => {
                entity_commands.insert(nrd.clone());
                nrd.reset = false;
            }
            _ if cleanup_query.get(entity) == Ok(true) => {
                entity_commands.remove::<(Nrd, ViewNrd, ViewDenoisedRadianceTextures)>();
            }
            _ => {}
        }
    }
}

/// Creates the denoiser of each view with [`Nrd`] and the textures it writes. When the view's render resolution
/// or the precision changes, the outputs follow at once and NRD's own textures once the change has held for
/// [`RESIZE_AFTER_STABLE_FRAMES`].
fn prepare_nrd(
    mut views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&MainPassResolutionOverride>,
        &Nrd,
        Option<&mut ViewNrd>,
        Option<&ViewDenoiserInputTextures>,
        Option<&ViewDenoiserGuideTextures>,
    )>,
    mut device_pipelines: ResMut<NrdDevicePipelines>,
    failure: Res<NrdFailure>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut commands: Commands,
) {
    for (entity, camera, resolution_override, nrd, view_nrd, inputs, guides) in &mut views {
        let mut entity_commands = commands.entity(entity);
        // The lighting pass inserts its textures one frame after the camera asks for them.
        let (Some(inputs), Some(guides), Some(mut size)) =
            (inputs, guides, camera.physical_viewport_size)
        else {
            entity_commands.remove::<(ViewNrd, ViewDenoisedRadianceTextures)>();
            continue;
        };
        if let Some(MainPassResolutionOverride(resolution_override)) = resolution_override {
            size = *resolution_override;
        }
        if failure.0.has_failed() {
            // NRD failed on this device: no view tries it again.
            entity_commands.remove::<(ViewNrd, ViewDenoisedRadianceTextures)>();
            continue;
        }
        let precision = nrd.precision;
        let formats = input_formats(inputs, guides);

        let Some(mut view_nrd) = view_nrd else {
            let pipelines = match &device_pipelines.0 {
                Some(pipelines) => Ok(pipelines.clone()),
                None => DenoiserPipelines::new(
                    render_device.wgpu_device(),
                    DenoiserKind::ReblurDiffuseSpecular,
                )
                .inspect(|pipelines| device_pipelines.0 = Some(pipelines.clone())),
            };
            let denoiser = pipelines.and_then(|pipelines| {
                pipelines.warm_up(
                    render_device.wgpu_device(),
                    precision.texture_precision(),
                    &formats,
                )?;
                Denoiser::with_pipelines(
                    render_device.wgpu_device(),
                    &pipelines,
                    size.to_array(),
                    precision.texture_precision(),
                )
            });
            match denoiser {
                Ok(denoiser) => {
                    info!(
                        "NRD: ReBLUR at {}x{}, {:?} precision, {:.1} MiB of textures",
                        size.x,
                        size.y,
                        precision,
                        denoiser.texture_bytes() as f64 / (1024.0 * 1024.0)
                    );
                    entity_commands.insert((
                        denoised_textures(&render_device, size),
                        ViewNrd {
                            size,
                            precision,
                            unsettled_frames: 0,
                            state: Mutex::new(Some(NrdState {
                                denoiser,
                                frame_index: 0,
                                motion_vectors: None,
                                history_confidence: HistoryConfidence::new(
                                    &render_device,
                                    &render_queue,
                                    size,
                                ),
                            })),
                        },
                    ));
                }
                Err(error) => {
                    failure.0.report("NRD", error);
                    entity_commands.remove::<ViewDenoisedRadianceTextures>();
                }
            }
            continue;
        };

        let ViewNrd {
            size: view_size,
            precision: view_precision,
            unsettled_frames,
            state,
        } = &mut *view_nrd;
        let Some(nrd_state) = state.get_mut().unwrap() else {
            entity_commands.remove::<ViewDenoisedRadianceTextures>();
            continue;
        };
        if (*view_size, *view_precision) != (size, precision) {
            // The outputs and the confidence estimate are small and follow at once; NRD's textures wait.
            entity_commands.insert(denoised_textures(&render_device, size));
            nrd_state.history_confidence =
                HistoryConfidence::new(&render_device, &render_queue, size);
            nrd_state.motion_vectors = None;
            *view_size = size;
            *view_precision = precision;
            *unsettled_frames = 0;
        }
        if ViewNrd::is_settled(nrd_state, size, precision) {
            continue;
        }
        *unsettled_frames += 1;
        if *unsettled_frames < RESIZE_AFTER_STABLE_FRAMES {
            continue;
        }
        let device = render_device.wgpu_device();
        let pipelines = nrd_state.denoiser.pipelines().clone();
        let resized = pipelines
            .warm_up(device, precision.texture_precision(), &formats)
            .and_then(|()| {
                nrd_state
                    .denoiser
                    .resize(device, size.to_array(), precision.texture_precision())
            });
        match resized {
            Ok(()) => info!(
                "NRD: ReBLUR resized to {}x{}, {:?} precision, {:.1} MiB of textures",
                size.x,
                size.y,
                precision,
                nrd_state.denoiser.texture_bytes() as f64 / (1024.0 * 1024.0)
            ),
            Err(error) => {
                failure.0.report("NRD", error);
                *state.get_mut().unwrap() = None;
                entity_commands.remove::<ViewDenoisedRadianceTextures>();
            }
        }
    }
}

/// The two textures NRD writes for the lighting pass to compose.
fn denoised_textures(render_device: &RenderDevice, size: UVec2) -> ViewDenoisedRadianceTextures {
    let create_texture = |label| {
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: nrd_wgpu::OUTPUT_FORMAT,
            // `COPY_DST`: a frame NRD does not denoise gets the noisy lighting copied in.
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let default_view = texture.create_view(&TextureViewDescriptor::default());
        CachedTexture {
            texture,
            default_view,
        }
    };
    ViewDenoisedRadianceTextures {
        diffuse: create_texture("nrd_denoised_diffuse"),
        specular: create_texture("nrd_denoised_specular"),
    }
}

/// The pass that cuts the motion vectors to the render resolution, see
/// [`NrdState::motion_vectors`].
#[derive(Resource)]
struct NrdCropPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

fn init_nrd_crop_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "nrd_crop_motion_vectors_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_storage_2d(TextureFormat::Rg16Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("nrd_crop_motion_vectors_pipeline".into()),
        layout: vec![bind_group_layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "crop_motion_vectors.wesl"),
        entry_point: Some("crop".into()),
        ..default()
    });
    commands.insert_resource(NrdCropPipeline {
        bind_group_layout,
        pipeline,
    });
}
