use super::{
    prepare::SolariLightingResources, SolariDenoiserInputs, SolariLighting,
    SolariSkipSpecularMotionVectors,
};
use bevy_anti_alias::ray_reconstruction::{ViewDenoiserGuideTextures, ViewDenoiserInputTextures};
use bevy_camera::Camera;
use bevy_ecs::{
    query::Has,
    system::{Commands, ResMut},
};
use bevy_pbr::{deferred::SkipDeferredLighting, TracedForwardLight, TracedForwardLightBindGroup};
use bevy_render::{sync_world::RenderEntity, MainWorld};

pub fn extract_solari_lighting(mut main_world: ResMut<MainWorld>, mut commands: Commands) {
    let mut cameras_3d = main_world.query::<(
        RenderEntity,
        &Camera,
        Option<&mut SolariLighting>,
        Has<SolariDenoiserInputs>,
        Has<SolariSkipSpecularMotionVectors>,
    )>();

    for (entity, camera, solari_lighting, denoiser_inputs, skip_specular_motion_vectors) in
        cameras_3d.iter_mut(&mut main_world)
    {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if let Some(mut solari_lighting) = solari_lighting
            && camera.is_active
        {
            entity_commands.insert((solari_lighting.clone(), SkipDeferredLighting, TracedForwardLight));
            solari_lighting.reset = false;
            if denoiser_inputs {
                entity_commands.insert(SolariDenoiserInputs);
            } else {
                entity_commands.remove::<SolariDenoiserInputs>();
            }
            if denoiser_inputs && skip_specular_motion_vectors {
                entity_commands.insert(SolariSkipSpecularMotionVectors);
            } else {
                entity_commands.remove::<SolariSkipSpecularMotionVectors>();
            }
        } else {
            entity_commands.remove::<(
                SolariLighting,
                SolariLightingResources,
                SkipDeferredLighting,
                TracedForwardLight,
                TracedForwardLightBindGroup,
                SolariDenoiserInputs,
                SolariSkipSpecularMotionVectors,
                ViewDenoiserGuideTextures,
                ViewDenoiserInputTextures,
            )>();
        }
    }
}
