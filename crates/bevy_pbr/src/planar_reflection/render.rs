//! The render side of planar reflections: the per-view uniform and textures the deferred lighting and
//! screen-space reflection passes bind at group 3 on views that reflect a mirror this frame.

use bevy_app::App;
use bevy_asset::AssetId;
use bevy_ecs::{prelude::*, query::QueryItem, system::lifetimeless::Read};
use bevy_image::Image;
use bevy_math::{Mat4, Vec4};
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        AddressMode, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        DynamicUniformBuffer, FilterMode, PipelineCache, Sampler, SamplerBindingType,
        SamplerDescriptor, ShaderStages, ShaderType, TextureSampleType,
    },
    renderer::{RenderDevice, RenderQueue},
    sync_component::SyncComponent,
    texture::{FallbackImageZero, GpuImage},
    view::ExtractedView,
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_utils::default;

use super::{ViewPlanarMirrors, MAX_PLANAR_MIRRORS};

/// The shader definition of a pass that reads planar reflections, and the group it binds them at.
pub const PLANAR_REFLECTIONS_SHADER_DEF: &str = "PLANAR_REFLECTIONS";

pub(super) fn build(app: &mut App) {
    app.add_plugins(ExtractComponentPlugin::<ViewPlanarMirrors>::default());
}

pub(super) fn finish(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_gpu_resource::<PlanarMirrorsBuffer>()
        .add_systems(RenderStartup, init_planar_reflections_layout)
        .add_systems(
            Render,
            (
                prepare_planar_mirror_uniforms.in_set(RenderSystems::PrepareResources),
                prepare_planar_reflection_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            ),
        );
}

/// One mirror as the shader reads it.
#[derive(ShaderType, Clone, Copy, Default, Debug)]
pub struct GpuPlanarMirror {
    clip_from_world: Mat4,
    plane: Vec4,
    center: Vec4,
    dual_u: Vec4,
    dual_v: Vec4,
    uv_rect: Vec4,
}

/// The mirrors of one view.
#[derive(ShaderType, Clone, Copy, Default, Debug)]
pub struct GpuPlanarMirrors {
    mirrors: [GpuPlanarMirror; MAX_PLANAR_MIRRORS],
    count: u32,
    normal_distortion: f32,
    _padding: Vec4,
}

/// The mirrors a view reflects this frame, in the render world.
#[derive(Component, Clone, Debug)]
pub struct ExtractedPlanarMirrors {
    images: Vec<AssetId<Image>>,
    uniform: GpuPlanarMirrors,
}

impl SyncComponent<RenderApp> for ViewPlanarMirrors {
    type Target = (
        ExtractedPlanarMirrors,
        PlanarMirrorsUniformOffset,
        PlanarReflectionsBindGroup,
    );
}

impl ExtractComponent<RenderApp> for ViewPlanarMirrors {
    type QueryData = Read<ViewPlanarMirrors>;
    type QueryFilter = ();
    type Out = ExtractedPlanarMirrors;

    fn extract_component(view: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        // A view with no mirror this frame keeps the passes it has without them.
        if view.mirrors.is_empty() {
            return None;
        }
        let mut uniform = GpuPlanarMirrors {
            count: view.mirrors.len().min(MAX_PLANAR_MIRRORS) as u32,
            normal_distortion: view.normal_distortion,
            ..default()
        };
        for (gpu, mirror) in uniform.mirrors.iter_mut().zip(&view.mirrors) {
            let (dual_u, dual_v) = mirror.rect.dual_axes();
            *gpu = GpuPlanarMirror {
                clip_from_world: mirror.clip_from_world,
                plane: mirror.rect.plane,
                center: mirror.rect.center.extend(1.0),
                dual_u: dual_u.extend(0.0),
                dual_v: dual_v.extend(0.0),
                uv_rect: mirror.uv_rect,
            };
        }
        Some(ExtractedPlanarMirrors {
            images: view
                .mirrors
                .iter()
                .take(MAX_PLANAR_MIRRORS)
                .map(|mirror| mirror.image.id())
                .collect(),
            uniform,
        })
    }
}

/// The layout of group 3 and the sampler of the mirror textures.
#[derive(Resource)]
pub struct PlanarReflectionsLayout {
    /// Group 3 of the passes that read planar reflections.
    pub layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
}

/// The uniform of every view's mirrors.
#[derive(Resource, Default)]
pub struct PlanarMirrorsBuffer(DynamicUniformBuffer<GpuPlanarMirrors>);

/// Where a view's mirrors are in [`PlanarMirrorsBuffer`].
#[derive(Component, Clone, Copy)]
pub struct PlanarMirrorsUniformOffset(u32);

/// Group 3 of a view that reflects mirrors this frame, with its dynamic offset. A pass specialized with
/// [`PLANAR_REFLECTIONS_SHADER_DEF`] binds it.
#[derive(Component)]
pub struct PlanarReflectionsBindGroup {
    /// The bind group.
    pub bind_group: BindGroup,
    /// Its one dynamic offset.
    pub offset: u32,
}

/// The layout of group 3 of a pass specialized with [`PLANAR_REFLECTIONS_SHADER_DEF`].
pub fn planar_reflections_layout() -> BindGroupLayoutDescriptor {
    let filterable = TextureSampleType::Float { filterable: true };
    BindGroupLayoutDescriptor::new(
        "planar_reflections_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<GpuPlanarMirrors>(true),
                texture_2d(filterable),
                texture_2d(filterable),
                texture_2d(filterable),
                texture_2d(filterable),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    )
}

fn init_planar_reflections_layout(mut commands: Commands, render_device: Res<RenderDevice>) {
    let layout = planar_reflections_layout();
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("planar_reflections_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    commands.insert_resource(PlanarReflectionsLayout { layout, sampler });
}

fn prepare_planar_mirror_uniforms(
    mut commands: Commands,
    views: Query<(Entity, &ExtractedPlanarMirrors), With<ExtractedView>>,
    mut buffer: ResMut<PlanarMirrorsBuffer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(mut writer) = buffer
        .0
        .get_writer(views.iter().len(), &render_device, &render_queue)
    else {
        return;
    };
    for (view, mirrors) in &views {
        let offset = writer.write(&mirrors.uniform);
        commands
            .entity(view)
            .insert(PlanarMirrorsUniformOffset(offset));
    }
}

fn prepare_planar_reflection_bind_groups(
    mut commands: Commands,
    views: Query<(Entity, &ExtractedPlanarMirrors, &PlanarMirrorsUniformOffset)>,
    buffer: Res<PlanarMirrorsBuffer>,
    layout: Option<Res<PlanarReflectionsLayout>>,
    pipeline_cache: Res<PipelineCache>,
    images: Res<RenderAssets<GpuImage>>,
    fallback: Res<FallbackImageZero>,
    render_device: Res<RenderDevice>,
) {
    let (Some(binding), Some(layout)) = (buffer.0.binding(), layout) else {
        return;
    };
    for (view, mirrors, offset) in &views {
        // A texture not on the GPU yet (the frame its mirror camera was made) reads as black (zero).
        let texture = |index: usize| {
            mirrors
                .images
                .get(index)
                .and_then(|id| images.get(*id))
                .map_or(&fallback.texture_view, |image| &image.texture_view)
        };
        let bind_group = render_device.create_bind_group(
            "planar_reflections_bind_group",
            &pipeline_cache.get_bind_group_layout(&layout.layout),
            &BindGroupEntries::sequential((
                binding.clone(),
                texture(0),
                texture(1),
                texture(2),
                texture(3),
                &layout.sampler,
            )),
        );
        commands.entity(view).insert(PlanarReflectionsBindGroup {
            bind_group,
            offset: offset.0,
        });
    }
}
