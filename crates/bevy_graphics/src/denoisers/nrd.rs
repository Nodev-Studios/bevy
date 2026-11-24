//! NRD, the denoiser of the ray traced rungs where DLSS Ray Reconstruction cannot run, on the ladder's
//! cameras.
//!
//! Built with the `nrd` feature, this module sets [`GraphicsCapabilities::nrd`] and registers NRD's
//! [`CameraBackend`]: a camera whose resolved denoiser is [`ResolvedDenoiser::Nrd`](crate::ResolvedDenoiser::Nrd)
//! gets `bevy_anti_alias`'s [`Nrd`](bevy_anti_alias::nrd::Nrd) and `bevy_solari`'s
//! [`SolariDenoiserInputs`](bevy_solari::realtime::SolariDenoiserInputs), which makes the lighting pass write
//! what NRD reads. Without the feature it does nothing, and [`GraphicsCapabilities::nrd`] stays `false`.

use bevy_app::App;

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built.
#[cfg(not(feature = "nrd"))]
pub(crate) fn build(_app: &mut App) {}

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built.
#[cfg(feature = "nrd")]
pub(crate) fn build(app: &mut App) {
    use crate::{
        apply::backends::{CameraBackend, GraphicsBackends},
        GraphicsSystems,
    };
    use bevy_anti_alias::nrd::Nrd;
    use bevy_app::PreStartup;
    use bevy_ecs::prelude::*;
    use bevy_solari::realtime::SolariDenoiserInputs;

    app.world_mut().resource_mut::<GraphicsBackends>().nrd = Some(CameraBackend {
        insert: |camera, _resolved| {
            camera.insert((Nrd::default(), SolariDenoiserInputs));
        },
        remove: |camera| {
            camera.remove::<(Nrd, SolariDenoiserInputs)>();
        },
    });
    app.add_systems(
        PreStartup,
        detect_nrd.after(GraphicsSystems::DetectCapabilities),
    );
}

/// NRD can run when the renderer is on Vulkan, the device can trace rays for the lighting NRD denoises, and it
/// has the wgpu features NRD's shaders need at the precision [`Nrd`](bevy_anti_alias::nrd::Nrd) uses by default
/// (`PASSTHROUGH_SHADERS`, `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`, `FLOAT32_FILTERABLE`; see
/// [`Nrd::required_wgpu_features`](bevy_anti_alias::nrd::Nrd::required_wgpu_features)).
///
/// It does not check that the driver accepts NRD's storage images declared without a format, which Vulkan
/// leaves optional and wgpu never enables: both drivers tried (NVIDIA on Linux, Mesa RADV) run them.
#[cfg(feature = "nrd")]
fn detect_nrd(
    mut capabilities: bevy_ecs::prelude::ResMut<crate::GraphicsCapabilities>,
    device: Option<bevy_ecs::prelude::Res<bevy_render::renderer::RenderDevice>>,
) {
    use bevy_anti_alias::nrd::{Nrd, NrdPrecision};
    let has_features = device.is_some_and(|device| {
        device
            .features()
            .contains(Nrd::required_wgpu_features(NrdPrecision::default()))
    });
    capabilities.nrd = capabilities.vulkan && capabilities.ray_query && has_features;
}
