//! FSR on the ladder's cameras.
//!
//! [`GraphicsCapabilities::fsr`] is `Some` only where `bevy_anti_alias` loaded FSR's DLL for this device
//! ([`FsrSupported`]): with the `fsr` feature, under Windows, on an AMD GPU, with the DLLs next to the executable.
//! The level is the provider the DLL chose: FSR 4 on the Radeons it supports, FSR 3.1 on the others.
//!
//! FSR is only used when asked for by name; `resolve` never picks it under `Auto` until it has run on real
//! hardware. Where it cannot run, a request for it falls back to TSR, whose default algorithm is FSR 3.1 from
//! AMD's source. If it fails on the device while the app runs (`FsrFailure`), the capability is cleared and the
//! failure recorded (`Fallback::FsrFailed`), so the camera gets TSR.

#[cfg(feature = "fsr")]
use crate::{
    apply::{
        backends::{CameraBackend, GraphicsBackendFailures, GraphicsBackends},
        GraphicsApplySystems,
    },
    Fallback, FsrLevel, GraphicsCapabilities, GraphicsSystems, ResolvedGraphics,
};
#[cfg(feature = "fsr")]
use bevy_anti_alias::fsr::{Fsr, FsrFailure, FsrSupported};
use bevy_app::App;
#[cfg(feature = "fsr")]
use bevy_app::{PostUpdate, PreStartup};
#[cfg(feature = "fsr")]
use bevy_ecs::prelude::*;

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built.
#[cfg(feature = "fsr")]
pub(crate) fn build(app: &mut App) {
    app.world_mut().resource_mut::<GraphicsBackends>().fsr = Some(CameraBackend {
        insert: insert_fsr,
        remove: |camera| {
            camera.remove::<Fsr>();
        },
    });
    app.add_systems(
        PreStartup,
        detect_fsr
            .after(GraphicsSystems::DetectCapabilities)
            .before(GraphicsApplySystems::Resolve),
    )
    .add_systems(
        PostUpdate,
        watch_fsr_failure.before(GraphicsApplySystems::Resolve),
    );
}

/// Clears [`GraphicsCapabilities::fsr`] and records [`Fallback::FsrFailed`] once FSR failed on the device.
#[cfg(feature = "fsr")]
fn watch_fsr_failure(
    failure: Option<Res<FsrFailure>>,
    mut capabilities: ResMut<GraphicsCapabilities>,
    mut failures: ResMut<GraphicsBackendFailures>,
) {
    if !failure.is_some_and(|failure| failure.0.has_failed())
        || failures.failures().contains(&Fallback::FsrFailed)
    {
        return;
    }
    capabilities.fsr = None;
    failures.record(Fallback::FsrFailed);
}

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built. Without the `fsr` feature there is no
/// FSR: [`GraphicsCapabilities::fsr`](crate::GraphicsCapabilities::fsr) stays `None`.
#[cfg(not(feature = "fsr"))]
pub(crate) fn build(_app: &mut App) {}

/// Puts [`Fsr`] on the camera at the resolved scale. A camera that already has it keeps its history unless the
/// scale changes, because the apply layer calls this again whenever anything else about the camera changes.
#[cfg(feature = "fsr")]
fn insert_fsr(camera: &mut EntityCommands, resolved: &ResolvedGraphics) {
    let render_scale = resolved.render_scale;
    camera
        .entry::<Fsr>()
        .and_modify(move |mut fsr| {
            if fsr.render_scale != render_scale {
                fsr.render_scale = render_scale;
                fsr.reset = true;
            }
        })
        .or_insert(Fsr {
            render_scale,
            ..Default::default()
        });
}

#[cfg(feature = "fsr")]
fn detect_fsr(
    supported: Option<Res<FsrSupported>>,
    mut capabilities: ResMut<GraphicsCapabilities>,
) {
    capabilities.fsr = supported.map(|supported| fsr_level(supported.major_version()));
}

/// FSR 4 when the DLL's provider is version 4 or later; FSR 3.1 for its fallback, and for a name that does not
/// parse, since only FSR 4 needs telling apart.
#[cfg(feature = "fsr")]
fn fsr_level(major_version: Option<u32>) -> FsrLevel {
    match major_version {
        Some(major) if major >= 4 => FsrLevel::Fsr4,
        _ => FsrLevel::Fsr3,
    }
}

#[cfg(all(test, feature = "fsr"))]
mod tests {
    use super::*;

    #[test]
    fn provider_versions_map_to_levels() {
        assert_eq!(fsr_level(Some(4)), FsrLevel::Fsr4);
        assert_eq!(fsr_level(Some(5)), FsrLevel::Fsr4);
        assert_eq!(fsr_level(Some(3)), FsrLevel::Fsr3);
        assert_eq!(fsr_level(None), FsrLevel::Fsr3);
    }

    /// Without FSR's DLL (any machine but a Radeon under Windows with the DLLs), the capability stays empty.
    #[test]
    fn no_dll_no_fsr() {
        let mut app = App::new();
        app.init_resource::<GraphicsCapabilities>()
            .add_systems(bevy_app::Update, detect_fsr);
        app.update();
        assert_eq!(app.world().resource::<GraphicsCapabilities>().fsr, None);

        app.insert_resource(FsrSupported {
            provider_name: "4.1.2".into(),
            provider_id: 1,
        });
        app.update();
        assert_eq!(
            app.world().resource::<GraphicsCapabilities>().fsr,
            Some(FsrLevel::Fsr4)
        );
    }
}
