//! The traced light for the surfaces the rasterizer still shades forward on a view with [`SolariLighting`]:
//! glass (the transmissive pass) and blended surfaces (the transparent pass), drawn over the traced image.
//!
//! Without it they are lit by the rasterized lights, which have no shadow maps on a traced view, and by an
//! unoccluded ambient and environment, so a glass in the shade glows as if it stood in the sun. Through
//! `bevy_pbr`'s [`TracedForwardLight`] hook their fragments instead
//!
//! - trace one shadow ray toward each light that reaches them (the specular part of direct light),
//! - take the light arriving at them, direct and indirect, from the world cache (the diffuse part), adding the
//!   cell when the cache lacks it, and
//! - see the environment's reflection only where a ray along the reflection leaves the scene.
//!
//! The shader is `forward_light.wesl`; the bind group holds this frame's TLAS, the view's world cache and its
//! lighting constants.
//!
//! [`SolariLighting`]: super::SolariLighting

use super::prepare::{SolariLightingResources, WORLD_CACHE_SIZE};
use crate::scene::RaytracingSceneBindings;
use bevy_ecs::{
    entity::Entity,
    system::{Commands, Query, Res},
};
use bevy_pbr::{TracedForwardLightBindGroup, TracedForwardLightLayout};
use bevy_render::{
    render_resource::{
        binding_types::{acceleration_structure, storage_buffer_sized, uniform_buffer_sized},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, PipelineCache,
        ShaderStages,
    },
    renderer::RenderDevice,
};
use bevy_shader::ShaderDefVal;

pub fn traced_forward_light_layout() -> TracedForwardLightLayout {
    TracedForwardLightLayout {
        layout: BindGroupLayoutDescriptor::new(
            "solari_forward_light_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    acceleration_structure(),
                    storage_buffer_sized(false, None),
                    uniform_buffer_sized(false, None),
                ),
            ),
        ),
        shader_defs: vec![ShaderDefVal::UInt(
            "WORLD_CACHE_SIZE".into(),
            WORLD_CACHE_SIZE as u32,
        )],
    }
}

/// This frame's bind group on every view with Solari's resources, once the scene is ready to trace.
pub fn prepare_traced_forward_light_bind_groups(
    views: Query<(Entity, &SolariLightingResources)>,
    scene_bindings: Res<RaytracingSceneBindings>,
    layout: Res<TracedForwardLightLayout>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (entity, resources) in &views {
        let Some(tlas) = scene_bindings.current_tlas() else {
            commands
                .entity(entity)
                .remove::<TracedForwardLightBindGroup>();
            continue;
        };
        let bind_group = render_device.create_bind_group(
            "solari_forward_light_bind_group",
            &pipeline_cache.get_bind_group_layout(&layout.layout),
            &BindGroupEntries::sequential((
                tlas.as_binding(),
                resources.world_cache.as_entire_binding(),
                resources.constants.as_entire_binding(),
            )),
        );
        commands
            .entity(entity)
            .insert(TracedForwardLightBindGroup(bind_group));
    }
}
