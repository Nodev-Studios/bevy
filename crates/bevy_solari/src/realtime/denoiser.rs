use super::SolariLighting;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_anti_alias::dlss::{Dlss, DlssRayReconstructionFeature};
use bevy_anti_alias::ray_reconstruction::{
    ViewDenoisedRadianceTextures, ViewDenoiserGuideTextures, ViewDenoiserInputTextures,
};
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_camera::MainPassResolutionOverride;
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, With},
    reflect::ReflectComponent,
    resource::Resource,
    system::{Commands, Query, Res},
};
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{texture_2d, texture_depth_2d, texture_storage_2d, uniform_buffer},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat,
        TextureSampleType, TextureUsages, TextureViewDescriptor,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    texture::CachedTexture,
    view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use bevy_utils::default;

/// A component for a 3d camera entity with [`SolariLighting`], to make the lighting pass write
/// what a denoiser other than DLSS Ray Reconstruction needs.
///
/// With this component the lighting pass fills, each frame, the
/// [`ViewDenoiserInputTextures`] and [`ViewDenoiserGuideTextures`] of the view: the light each
/// surface reflects, split into its diffuse and specular lobes, divided by the lobe's reflectance
/// and paired with the length of the ray that gathered it. The types in
/// [`bevy_anti_alias::ray_reconstruction`] document the encoding and where a denoiser runs.
///
/// The view's lit color is then written by a compose step after the denoiser, from the textures
/// the denoiser returns. With no denoiser the compose step reads the noisy split, and the image is
/// the one [`SolariLighting`] renders without this component.
///
/// # Mirrors
///
/// The split is always made at the surface the view ray hit, and the guides always describe that
/// surface. On a flat perfect mirror this means: all of the light is in the specular texture, its
/// hit distance is the distance from the mirror to what is seen in it, and normal, roughness and
/// depth are the mirror's own. A denoiser places the reflection from that hit distance.
///
/// This differs from a view with DLSS Ray Reconstruction alone, where the guides of a mirror are
/// replaced by those of the surface seen in it. A camera with both this component and DLSS Ray
/// Reconstruction gets the unreplaced guides, and denoises mirror reflections less well.
#[derive(Component, Reflect, Clone, Copy, Default)]
#[reflect(Component, Default, Clone)]
pub struct SolariDenoiserInputs;

/// Creates and frees the textures the lighting pass writes for a denoiser.
pub fn prepare_solari_denoiser_textures(
    #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))] query: Query<
        (
            Entity,
            &ExtractedCamera,
            Option<&MainPassResolutionOverride>,
            Has<SolariDenoiserInputs>,
            Option<&ViewDenoiserGuideTextures>,
            Option<&ViewDenoiserInputTextures>,
        ),
        With<SolariLighting>,
    >,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] query: Query<
        (
            Entity,
            &ExtractedCamera,
            Option<&MainPassResolutionOverride>,
            Has<SolariDenoiserInputs>,
            Option<&ViewDenoiserGuideTextures>,
            Option<&ViewDenoiserInputTextures>,
            Has<Dlss<DlssRayReconstructionFeature>>,
        ),
        With<SolariLighting>,
    >,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for query_item in &query {
        #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
        let (entity, camera, resolution_override, wants_inputs, guides, inputs) = query_item;
        #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
        let wants_guides = wants_inputs;

        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        let (entity, camera, resolution_override, wants_inputs, guides, inputs, has_dlss_rr) =
            query_item;
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        let wants_guides = wants_inputs || has_dlss_rr;

        let Some(mut view_size) = camera.physical_viewport_size else {
            continue;
        };
        if let Some(MainPassResolutionOverride(resolution_override)) = resolution_override {
            view_size = *resolution_override;
        }

        let create_texture = |label, format, extra_usages| {
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING
                    | extra_usages,
                view_formats: &[],
            });
            let default_view = texture.create_view(&TextureViewDescriptor::default());
            CachedTexture {
                texture,
                default_view,
            }
        };
        let has_size = |texture: &CachedTexture| {
            UVec2::new(texture.texture.width(), texture.texture.height()) == view_size
        };

        let mut entity_commands = commands.entity(entity);

        match (wants_guides, guides) {
            (true, Some(guides)) if has_size(&guides.depth) => {}
            (true, _) => {
                entity_commands.insert(ViewDenoiserGuideTextures {
                    diffuse_albedo: create_texture(
                        "solari_lighting_diffuse_albedo",
                        TextureFormat::Rgba8Unorm,
                        TextureUsages::empty(),
                    ),
                    specular_albedo: create_texture(
                        "solari_lighting_specular_albedo",
                        TextureFormat::Rgba8Unorm,
                        TextureUsages::empty(),
                    ),
                    normal_roughness: create_texture(
                        "solari_lighting_normal_roughness",
                        TextureFormat::Rgba16Float,
                        TextureUsages::empty(),
                    ),
                    depth: create_texture(
                        "solari_lighting_dlss_rr_depth",
                        TextureFormat::R32Float,
                        TextureUsages::empty(),
                    ),
                    specular_motion_vectors: create_texture(
                        "solari_lighting_specular_motion_vectors",
                        TextureFormat::Rg16Float,
                        TextureUsages::empty(),
                    ),
                });
            }
            (false, Some(_)) => {
                entity_commands.remove::<ViewDenoiserGuideTextures>();
            }
            (false, None) => {}
        }

        match (wants_inputs, inputs) {
            (true, Some(inputs)) if has_size(&inputs.view_depth) => {}
            (true, _) => {
                // `COPY_SRC` lets a tool read back what a denoiser is given.
                entity_commands.insert(ViewDenoiserInputTextures {
                    diffuse_radiance_hit_distance: create_texture(
                        "solari_lighting_diffuse_radiance_hit_distance",
                        TextureFormat::Rgba16Float,
                        TextureUsages::COPY_SRC,
                    ),
                    specular_radiance_hit_distance: create_texture(
                        "solari_lighting_specular_radiance_hit_distance",
                        TextureFormat::Rgba16Float,
                        TextureUsages::COPY_SRC,
                    ),
                    view_depth: create_texture(
                        "solari_lighting_view_depth",
                        TextureFormat::R32Float,
                        TextureUsages::COPY_SRC,
                    ),
                });
            }
            (false, Some(_)) => {
                entity_commands.remove::<ViewDenoiserInputTextures>();
            }
            (false, None) => {}
        }
    }
}

/// The pipeline of the compose step, see [`solari_compose`].
#[derive(Resource)]
pub struct SolariComposePipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

pub fn init_solari_compose_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "solari_compose_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Uint),
                texture_depth_2d(),
                uniform_buffer::<ViewUniform>(true),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );

    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("solari_compose_pipeline".into()),
        layout: vec![bind_group_layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "compose.wesl"),
        entry_point: Some("compose".into()),
        ..default()
    });

    commands.insert_resource(SolariComposePipeline {
        bind_group_layout,
        pipeline,
    });
}

/// Writes the lit color of a view with [`SolariDenoiserInputs`] from the radiance its denoiser
/// returned, or from the noisy radiance when the view has no denoiser.
pub fn solari_compose(
    view: ViewQuery<
        (
            &ViewTarget,
            &ViewPrepassTextures,
            &ViewUniformOffset,
            &ViewDenoiserInputTextures,
            Option<&ViewDenoisedRadianceTextures>,
        ),
        With<SolariLighting>,
    >,
    compose_pipeline: Option<Res<SolariComposePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    view_uniforms: Res<ViewUniforms>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (view_target, view_prepass_textures, view_uniform_offset, inputs, denoised) =
        view.into_inner();

    let Some(compose_pipeline) = compose_pipeline else {
        return;
    };
    let (Some(pipeline), Some(gbuffer), Some(depth_buffer), Some(view_uniforms_binding)) = (
        pipeline_cache.get_compute_pipeline(compose_pipeline.pipeline),
        view_prepass_textures.deferred_view(),
        view_prepass_textures.depth_only_view(),
        view_uniforms.uniforms.binding(),
    ) else {
        return;
    };

    let (diffuse_radiance, specular_radiance) = match denoised {
        Some(denoised) => (&denoised.diffuse, &denoised.specular),
        None => (
            &inputs.diffuse_radiance_hit_distance,
            &inputs.specular_radiance_hit_distance,
        ),
    };

    let bind_group = render_device.create_bind_group(
        "solari_compose_bind_group",
        &pipeline_cache.get_bind_group_layout(&compose_pipeline.bind_group_layout),
        &BindGroupEntries::sequential((
            view_target.main_texture_view(),
            gbuffer,
            depth_buffer,
            view_uniforms_binding,
            &diffuse_radiance.default_view,
            &specular_radiance.default_view,
        )),
    );

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("solari_compose"),
            timestamp_writes: None,
        });
    let d = diagnostics.time_span(&mut pass, "solari_compose");

    let view_size = inputs.view_depth.texture.size();
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
    pass.dispatch_workgroups(view_size.width.div_ceil(8), view_size.height.div_ceil(8), 1);

    d.end(&mut pass);
}
