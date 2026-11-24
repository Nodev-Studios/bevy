use super::{prepare::TsrContext, Tsr};
use bevy_camera::{Camera, Hdr, MainPassResolutionOverride, Projection};
use bevy_ecs::{
    query::{Has, With},
    system::{Commands, Query, ResMut},
};
use bevy_render::{sync_world::RenderEntity, MainWorld};

pub fn extract_tsr(
    mut commands: Commands,
    mut main_world: ResMut<MainWorld>,
    cleanup_query: Query<Has<Tsr>>,
) {
    let mut cameras_3d = main_world
        .query_filtered::<(RenderEntity, &Camera, &Projection, Option<&mut Tsr>), With<Hdr>>();

    for (entity, camera, camera_projection, tsr) in cameras_3d.iter_mut(&mut main_world) {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if let Some(mut tsr) = tsr
            && camera.is_active
            && camera_projection.is_perspective()
        {
            entity_commands.insert(tsr.clone());
            tsr.reset = false;
        } else if cleanup_query.get(entity) == Ok(true) {
            entity_commands.remove::<(Tsr, TsrContext, MainPassResolutionOverride)>();
        }
    }
}
