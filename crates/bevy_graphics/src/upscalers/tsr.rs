//! TSR, the vendor-neutral temporal upscaler, on the ladder's cameras.
//!
//! [`Tsr`] lives in `bevy_anti_alias` and is in every build. [`GraphicsCapabilities::tsr`] is true when
//! `bevy_anti_alias` found the device able to run it ([`TsrSupported`]): compute shaders and write-only storage
//! textures of `R32Uint` and `Rgba16Float`, which is every device but WebGL 2's. The camera gets `Tsr`'s default
//! algorithm, AMD's FSR 3.1, which needs Vulkan; elsewhere TSR runs SGSR 2 in its place.

use crate::{
    apply::{
        backends::{CameraBackend, GraphicsBackends},
        GraphicsApplySystems,
    },
    GraphicsCapabilities, GraphicsSystems, ResolvedGraphics,
};
use bevy_anti_alias::tsr::{Tsr, TsrSupported};
use bevy_app::{App, PreStartup};
use bevy_ecs::prelude::*;

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built.
pub(crate) fn build(app: &mut App) {
    app.world_mut().resource_mut::<GraphicsBackends>().tsr = Some(CameraBackend {
        insert: insert_tsr,
        remove: |camera| {
            camera.remove::<Tsr>();
        },
    });
    app.add_systems(
        PreStartup,
        detect_tsr
            .after(GraphicsSystems::DetectCapabilities)
            .before(GraphicsApplySystems::Resolve),
    );
}

/// Puts [`Tsr`] on the camera at the resolved scale. A camera that already has it keeps its history unless the
/// scale changes, because the apply layer calls this again whenever anything else about the camera changes.
fn insert_tsr(camera: &mut EntityCommands, resolved: &ResolvedGraphics) {
    let render_scale = resolved.render_scale;
    camera
        .entry::<Tsr>()
        .and_modify(move |mut tsr| {
            if tsr.render_scale != render_scale {
                tsr.render_scale = render_scale;
            }
        })
        .or_insert(Tsr {
            render_scale,
            ..Default::default()
        });
}

fn detect_tsr(
    supported: Option<Res<TsrSupported>>,
    mut capabilities: ResMut<GraphicsCapabilities>,
) {
    capabilities.tsr = supported.is_some();
}
