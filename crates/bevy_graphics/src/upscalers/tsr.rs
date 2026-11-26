//! TSR, the vendor-neutral temporal upscaler, on the ladder's cameras.
//!
//! [`Tsr`] lives in `bevy_anti_alias` and is in every build. [`GraphicsCapabilities::tsr`] is true when
//! `bevy_anti_alias` found the device able to run it ([`TsrSupported`]): compute shaders and write-only storage
//! textures of `R32Uint` and `Rgba16Float`, which is every device but WebGL 2's. The camera gets `Tsr`'s default
//! algorithm, AMD's FSR 3.1, which needs Vulkan; elsewhere TSR runs SGSR 2 in its place. If FSR 3.1 fails on the
//! device while the app runs, `bevy_anti_alias` switches TSR to SGSR 2 ([`TsrFsr3Failure`]); this module clears
//! [`TsrSupported::fsr3`] and records [`Fallback::TsrFsr3Failed`].

use crate::{
    apply::{
        backends::{CameraBackend, GraphicsBackendFailures, GraphicsBackends},
        GraphicsApplySystems,
    },
    Fallback, GraphicsCapabilities, GraphicsSystems, ResolvedGraphics,
};
use bevy_anti_alias::tsr::{Tsr, TsrFsr3Failure, TsrSupported};
use bevy_app::{App, PostUpdate, PreStartup};
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
    )
    .add_systems(
        PostUpdate,
        watch_tsr_fsr3_failure.before(GraphicsApplySystems::Resolve),
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
                // The history is at the old scale; FSR 3.1 would otherwise blend it in.
                tsr.reset = true;
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

/// Clears [`TsrSupported::fsr3`] and records [`Fallback::TsrFsr3Failed`] once TSR's FSR 3.1 failed on the device.
fn watch_tsr_fsr3_failure(
    failure: Option<Res<TsrFsr3Failure>>,
    supported: Option<ResMut<TsrSupported>>,
    mut failures: ResMut<GraphicsBackendFailures>,
) {
    if !failure.is_some_and(|failure| failure.0.has_failed())
        || failures.failures().contains(&Fallback::TsrFsr3Failed)
    {
        return;
    }
    if let Some(mut supported) = supported {
        supported.fsr3 = false;
    }
    failures.record(Fallback::TsrFsr3Failed);
}
