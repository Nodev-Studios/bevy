//! NRD, the denoiser of the ray traced rungs where DLSS Ray Reconstruction cannot run, on the ladder's
//! cameras.
//!
//! Built with the `nrd` feature, this module sets
//! [`GraphicsCapabilities::nrd`](crate::GraphicsCapabilities::nrd) and registers NRD's
//! [`CameraBackend`](crate::apply::backends::CameraBackend): a camera whose resolved denoiser is
//! [`ResolvedDenoiser::Nrd`](crate::ResolvedDenoiser::Nrd) gets `bevy_anti_alias`'s `Nrd` and `bevy_solari`'s
//! `SolariDenoiserInputs`, which makes the lighting pass write what NRD reads, and
//! `SolariSkipSpecularMotionVectors`, which spares it what NRD does not read. Without the feature it does
//! nothing, and the capability stays `false`.
//!
//! If NRD fails on the device while the app runs (`bevy_anti_alias`'s `NrdFailure`), the capability is cleared
//! and the failure recorded ([`Fallback::NrdFailed`](crate::Fallback::NrdFailed)), so the ladder steps down to
//! DLSS Ray Reconstruction or to `Raster`.

use bevy_app::App;

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built.
#[cfg(not(feature = "nrd"))]
pub(crate) fn build(_app: &mut App) {}

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built.
#[cfg(feature = "nrd")]
pub(crate) fn build(app: &mut App) {
    on::build(app);
}

#[cfg(feature = "nrd")]
mod on {
    use crate::{
        apply::{
            backends::{CameraBackend, GraphicsBackendFailures, GraphicsBackends},
            GraphicsApplySystems,
        },
        Fallback, GraphicsCapabilities, GraphicsSystems,
    };
    use bevy_anti_alias::nrd::{Nrd, NrdFailure, NrdPrecision};
    use bevy_app::{App, PostUpdate, PreStartup};
    use bevy_ecs::prelude::*;
    use bevy_render::{
        renderer::{RenderAdapter, RenderDevice},
        settings::WgpuFeatures,
    };
    use bevy_solari::realtime::{SolariDenoiserInputs, SolariSkipSpecularMotionVectors};
    use tracing::info;

    /// The precision the ladder gives [`Nrd`]: [`NrdPrecision::Full`] where the device can filter 32-bit float
    /// textures, [`NrdPrecision::Half`] where it cannot.
    #[derive(Resource, Clone, Copy, Debug)]
    pub(crate) struct NrdLadderPrecision(pub NrdPrecision);

    pub(super) fn build(app: &mut App) {
        app.world_mut().resource_mut::<GraphicsBackends>().nrd = Some(CameraBackend {
            // NRD reads no specular motion vectors, so the lighting pass skips the mirror tracking that
            // writes them. `Nrd` cannot require it: `bevy_anti_alias` cannot name `bevy_solari`'s types.
            insert: |camera, _resolved| {
                camera.insert((SolariDenoiserInputs, SolariSkipSpecularMotionVectors));
                // A camera that has `Nrd` keeps it, and its history.
                camera.queue(|mut camera: EntityWorldMut| {
                    if camera.contains::<Nrd>() {
                        return;
                    }
                    let precision = camera
                        .world()
                        .get_resource::<NrdLadderPrecision>()
                        .map_or_else(NrdPrecision::default, |precision| precision.0);
                    camera.insert(Nrd {
                        precision,
                        ..Default::default()
                    });
                });
            },
            remove: |camera| {
                camera.remove::<(Nrd, SolariDenoiserInputs, SolariSkipSpecularMotionVectors)>();
            },
        });
        app.add_systems(
            PreStartup,
            detect_nrd.after(GraphicsSystems::DetectCapabilities),
        )
        .add_systems(
            PostUpdate,
            watch_nrd_failure.before(GraphicsApplySystems::Resolve),
        );
    }

    /// What NRD can run at on a device, or `None` when it cannot run.
    ///
    /// NRD needs Vulkan, ray traced lighting to denoise, the wgpu features of
    /// [`Nrd::required_wgpu_features`] at [`NrdPrecision::Half`] (`PASSTHROUGH_SHADERS`,
    /// `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`), and the Vulkan features
    /// `shaderStorageImageReadWithoutFormat` and `shaderStorageImageWriteWithoutFormat`: NRD's SPIR-V declares
    /// its storage images without a format. wgpu never enables those two, so a device that does not even
    /// support them is refused; one that supports them ran NRD on both drivers tried (NVIDIA on Linux, Mesa
    /// RADV). `FLOAT32_FILTERABLE` makes the precision [`NrdPrecision::Full`].
    pub(crate) fn nrd_support(
        vulkan: bool,
        ray_query: bool,
        features: WgpuFeatures,
        storage_images_without_format: Option<bool>,
    ) -> Option<NrdPrecision> {
        let runs = !cfg!(feature = "force_disable_nrd")
            && vulkan
            && ray_query
            && features.contains(Nrd::required_wgpu_features(NrdPrecision::Half))
            && storage_images_without_format == Some(true);
        runs.then(|| {
            if features.contains(Nrd::required_wgpu_features(NrdPrecision::Full)) {
                NrdPrecision::Full
            } else {
                NrdPrecision::Half
            }
        })
    }

    fn detect_nrd(
        mut capabilities: ResMut<GraphicsCapabilities>,
        device: Option<Res<RenderDevice>>,
        adapter: Option<Res<RenderAdapter>>,
        mut commands: Commands,
    ) {
        let support = nrd_support(
            capabilities.vulkan,
            capabilities.ray_query,
            device.map_or(WgpuFeatures::empty(), |device| device.features()),
            adapter.and_then(|adapter| storage_images_without_format(&adapter)),
        );
        capabilities.nrd = support.is_some();
        if let Some(precision) = support {
            if precision == NrdPrecision::Half {
                info!("NRD runs at half precision: the device cannot filter 32-bit float textures");
            }
            commands.insert_resource(NrdLadderPrecision(precision));
        }
    }

    /// Whether the Vulkan device supports storage images declared without a format, for both reading and
    /// writing. `None` when the renderer is not on Vulkan.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[expect(
        unsafe_code,
        reason = "wgpu does not expose these Vulkan features; they are read from the raw adapter."
    )]
    fn storage_images_without_format(adapter: &RenderAdapter) -> Option<bool> {
        // SAFETY: the raw adapter is only read from, and the guard is dropped before this function returns.
        let raw_adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
        let instance = raw_adapter.shared_instance().raw_instance();
        // SAFETY: the physical device handle belongs to this instance, and both outlive the call because
        // `raw_adapter` keeps them alive.
        let features =
            unsafe { instance.get_physical_device_features(raw_adapter.raw_physical_device()) };
        Some(
            features.shader_storage_image_read_without_format != 0
                && features.shader_storage_image_write_without_format != 0,
        )
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    fn storage_images_without_format(_adapter: &RenderAdapter) -> Option<bool> {
        None
    }

    /// Clears [`GraphicsCapabilities::nrd`] and records [`Fallback::NrdFailed`] once NRD failed on the device.
    pub(crate) fn watch_nrd_failure(
        failure: Option<Res<NrdFailure>>,
        mut capabilities: ResMut<GraphicsCapabilities>,
        mut failures: ResMut<GraphicsBackendFailures>,
    ) {
        if !failure.is_some_and(|failure| failure.0.has_failed())
            || failures.failures().contains(&Fallback::NrdFailed)
        {
            return;
        }
        capabilities.nrd = false;
        failures.record(Fallback::NrdFailed);
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const BASE: WgpuFeatures = WgpuFeatures::PASSTHROUGH_SHADERS
            .union(WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);

        #[test]
        fn storage_images_without_format_are_required() {
            let full = BASE | WgpuFeatures::FLOAT32_FILTERABLE;
            if cfg!(feature = "force_disable_nrd") {
                assert_eq!(nrd_support(true, true, full, Some(true)), None);
                return;
            }
            assert_eq!(
                nrd_support(true, true, full, Some(true)),
                Some(NrdPrecision::Full)
            );
            assert_eq!(nrd_support(true, true, full, Some(false)), None);
            assert_eq!(nrd_support(true, true, full, None), None);
            assert_eq!(nrd_support(false, true, full, Some(true)), None);
            assert_eq!(nrd_support(true, false, full, Some(true)), None);
        }

        /// NRD failing on the device takes the ladder from `RtFull` with NRD down to `Raster`, with the reason.
        #[test]
        fn a_failed_nrd_steps_down_to_raster() {
            use crate::{
                apply::resolve_graphics, GpuDeviceType, GpuVendor, GraphicsPreset, GraphicsRung,
                GraphicsSettings, ResolvedDenoiser, ResolvedGraphics,
            };
            use bevy_anti_alias::backend_failure::BackendFailure;
            let capabilities = GraphicsCapabilities {
                adapter_name: "AMD Radeon RX 7900 XTX".into(),
                vendor: GpuVendor::Amd,
                device_type: GpuDeviceType::Discrete,
                vulkan: true,
                video_memory_bytes: Some(24 << 30),
                solari: true,
                ray_query: true,
                tsr: true,
                nrd: true,
                ..Default::default()
            };
            let failure = NrdFailure(BackendFailure::default());
            let mut app = App::new();
            app.insert_resource(capabilities)
                .insert_resource(GraphicsSettings {
                    preset: GraphicsPreset::RtFull,
                    ..Default::default()
                })
                .insert_resource(failure.clone())
                .init_resource::<GraphicsBackendFailures>()
                .add_systems(
                    bevy_app::Update,
                    (watch_nrd_failure, resolve_graphics).chain(),
                );
            app.update();
            let resolved = app.world().resource::<ResolvedGraphics>().clone();
            assert_eq!(resolved.rung, GraphicsRung::RtFull);
            assert_eq!(resolved.denoiser, ResolvedDenoiser::Nrd);

            failure.0.report("NRD", "a test");
            app.update();
            let resolved = app.world().resource::<ResolvedGraphics>().clone();
            assert_eq!(resolved.rung, GraphicsRung::Raster);
            assert!(resolved.fallbacks.contains(&Fallback::NoDenoiser));
            assert!(resolved.fallbacks.contains(&Fallback::NrdFailed));
            assert!(!app.world().resource::<GraphicsCapabilities>().nrd);
        }

        #[test]
        fn no_float32_filtering_gives_half_precision() {
            if cfg!(feature = "force_disable_nrd") {
                return;
            }
            assert_eq!(
                nrd_support(true, true, BASE, Some(true)),
                Some(NrdPrecision::Half)
            );
            assert_eq!(
                nrd_support(true, true, WgpuFeatures::PASSTHROUGH_SHADERS, Some(true)),
                None
            );
        }
    }
}
