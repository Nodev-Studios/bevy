//! NVIDIA's Real-time Denoisers (NRD): ReBLUR as the denoiser of ray traced lighting.
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
//!    lighting pass then writes [`ViewDenoiserInputTextures`]) and [`Nrd`].
//! 3. Resolve the denoised frame with temporal anti-aliasing or an upscaler, as with any ray
//!    traced lighting.
//!
//! Each frame [`Nrd`] reads the textures of [`crate::ray_reconstruction`] in
//! [`DenoiserSystems::Denoise`] and writes [`ViewDenoisedRadianceTextures`], which the lighting
//! pass composes into the view's lit color. A view where NRD cannot run (see
//! [`Nrd::required_wgpu_features`]) logs why once and shows the noisy lighting.

mod node;

use crate::ray_reconstruction::{
    DenoiserSystems, ViewDenoisedRadianceTextures, ViewDenoiserInputTextures,
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
    renderer::RenderDevice,
    settings::WgpuFeatures,
    sync_world::RenderEntity,
    texture::CachedTexture,
    ExtractSchedule, MainWorld, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_utils::default;
use nrd_wgpu::{Denoiser, DenoiserKind, TexturePrecision};
use std::sync::Mutex;
use tracing::{error, info};

/// Adds [`Nrd`]. Part of [`AntiAliasPlugin`](crate::AntiAliasPlugin) when the `nrd` feature is on.
#[derive(Default)]
pub struct NrdPlugin;

impl Plugin for NrdPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "crop_motion_vectors.wesl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_nrd_crop_pipeline)
            .add_systems(ExtractSchedule, extract_nrd)
            .add_systems(Render, prepare_nrd.in_set(RenderSystems::PrepareResources))
            .add_systems(Core3d, node::nrd.in_set(DenoiserSystems::Denoise));
    }
}

/// A component for a 3d camera with ray traced lighting, to denoise that lighting with NRD's
/// ReBLUR.
///
/// The camera must also make its lighting pass write [`ViewDenoiserInputTextures`]
/// (`bevy_solari`'s `SolariDenoiserInputs`); without them NRD does nothing.
#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component, Default, Clone)]
pub struct Nrd {
    /// The precision of the textures ReBLUR keeps its history in.
    pub precision: NrdPrecision,
    /// Set to true to drop the accumulated history, after a camera cut for example. It is set
    /// back to false after the frame it applied to.
    pub reset: bool,
}

/// The precision of the textures [`Nrd`] accumulates in.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Default, Clone, PartialEq)]
pub enum NrdPrecision {
    /// 16-bit floats, as NRD asks for. Measured on the RTX 5070 and on Mesa RADV, both drivers
    /// round toward zero when they store a 16-bit float, so ReBLUR's history settles below its
    /// target: the denoised light comes out a few percent dark (`vendor/nrd_wgpu/README.md`).
    Half,
    /// 32-bit floats wherever NRD asks for 16-bit ones: no darkening, about 1.6 times the memory
    /// of [`Self::Half`]. Needs `WgpuFeatures::FLOAT32_FILTERABLE`.
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

/// The NRD denoiser of one view, at one size and precision.
#[derive(Component)]
pub struct ViewNrd {
    size: UVec2,
    precision: NrdPrecision,
    /// `None` once creating or running the denoiser failed: the view then shows the noisy
    /// lighting until its size or precision changes.
    state: Mutex<Option<NrdState>>,
}

struct NrdState {
    denoiser: Denoiser,
    frame_index: u32,
    /// The view's motion vectors cut to the render resolution, when the prepass texture is
    /// larger (an upscaler is on).
    motion_vectors: Option<CachedTexture>,
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

/// Creates the denoiser of each view with [`Nrd`] and the textures it writes, and recreates them
/// when the view's render resolution or the precision changes.
fn prepare_nrd(
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&MainPassResolutionOverride>,
        &Nrd,
        Option<&ViewNrd>,
        Has<ViewDenoiserInputTextures>,
    )>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (entity, camera, resolution_override, nrd, view_nrd, has_inputs) in &views {
        let mut entity_commands = commands.entity(entity);
        // The lighting pass inserts its textures one frame after the camera asks for them.
        let (true, Some(mut size)) = (has_inputs, camera.physical_viewport_size) else {
            entity_commands.remove::<(ViewNrd, ViewDenoisedRadianceTextures)>();
            continue;
        };
        if let Some(MainPassResolutionOverride(resolution_override)) = resolution_override {
            size = *resolution_override;
        }

        if let Some(view_nrd) = view_nrd
            && view_nrd.size == size
            && view_nrd.precision == nrd.precision
        {
            if view_nrd.state.lock().unwrap().is_none() {
                entity_commands.remove::<ViewDenoisedRadianceTextures>();
            }
            continue;
        }

        let denoiser = Denoiser::new(
            render_device.wgpu_device(),
            DenoiserKind::ReblurDiffuseSpecular,
            size.to_array(),
            nrd.precision.texture_precision(),
        );
        let state = match denoiser {
            Ok(denoiser) => {
                info!(
                    "NRD: ReBLUR at {}x{}, {:?} precision, {:.1} MiB of textures",
                    size.x,
                    size.y,
                    nrd.precision,
                    denoiser.texture_bytes() as f64 / (1024.0 * 1024.0)
                );
                let create_texture = |label| {
                    let texture = render_device.create_texture(&TextureDescriptor {
                        label: Some(label),
                        size: size.to_extents(),
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: nrd_wgpu::OUTPUT_FORMAT,
                        // `COPY_DST`: a frame NRD cannot denoise gets the noisy lighting copied in.
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
                entity_commands.insert(ViewDenoisedRadianceTextures {
                    diffuse: create_texture("nrd_denoised_diffuse"),
                    specular: create_texture("nrd_denoised_specular"),
                });
                Some(NrdState {
                    denoiser,
                    frame_index: 0,
                    motion_vectors: None,
                })
            }
            Err(error) => {
                error!("NRD cannot denoise this view, which shows the noisy lighting: {error}");
                entity_commands.remove::<ViewDenoisedRadianceTextures>();
                None
            }
        };
        entity_commands.insert(ViewNrd {
            size,
            precision: nrd.precision,
            state: Mutex::new(state),
        });
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
