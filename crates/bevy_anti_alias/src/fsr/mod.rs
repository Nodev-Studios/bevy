//! AMD FSR upscaling: FSR 4 on the Radeons that support it, FSR 3.1 on other Radeons, through the signed Vulkan
//! DLLs of AMD's FSR SDK 2.4.0 (`vendor/fsr_wgpu`).
//!
//! Requires Windows, the Vulkan backend, an AMD GPU, and `amd_fidelityfx_loader_vk.dll` and
//! `amd_fidelityfx_upscaler_vk.dll` next to the executable (from the SDK's `Kits/FidelityFX/signedbin/`; they
//! are binary-redistributable). Anywhere else the plugin does nothing and [`FsrSupported`] is absent. The FSR 3.1
//! algorithm also runs on every Vulkan GPU, from AMD's source, as [`Tsr`](crate::tsr::Tsr)'s default.
//!
//! **It has never run**: there is no Windows machine with a Radeon where it was written. See
//! `vendor/fsr_wgpu/README.md` for what is known and what is not.
//!
//! # Usage
//! 1. Enable Bevy's `fsr` feature. `DefaultPlugins` then holds [`FsrInitPlugin`], which must come before
//!    [`RenderPlugin`](bevy_render::RenderPlugin) because it changes the Vulkan device.
//! 2. Check for `Option<Res<FsrSupported>>` at run time; it says which FSR the DLL chose for this GPU.
//! 3. Add [`Fsr`] to a 3D camera with [`Msaa::Off`](bevy_render::view::Msaa::Off), and no other temporal
//!    anti-aliasing or upscaler.
//! 4. Custom rendering code, including third party crates, should account for the optional
//!    `MainPassResolutionOverride`, as with DLSS (see the `custom_render_phase` example).

mod extract;
mod node;
mod prepare;

use alloc::sync::Arc;
use bevy_app::{App, Plugin};
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    prepass::{DepthPrepass, MotionVectorPrepass},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    renderer::{
        raw_vulkan_init::{AdditionalVulkanFeatures, RawVulkanInitSettings},
        RenderDevice,
    },
    view::prepare_view_targets,
    ExtractSchedule, Render, RenderApp, RenderSystems,
};
use fsr_wgpu::FsrSdk;
use tracing::{info, warn};

/// Prepares the Vulkan device for FSR. Must be added before [`RenderPlugin`](bevy_render::RenderPlugin), which
/// `DefaultPlugins` does when the `fsr` feature is on.
///
/// It changes the device only on Windows, on an AMD adapter, and when FSR's loader DLL is found.
#[derive(Default)]
pub struct FsrInitPlugin;

impl Plugin for FsrInitPlugin {
    #[expect(
        unsafe_code,
        reason = "FSR's DLL needs device extensions wgpu does not enable."
    )]
    fn build(&self, app: &mut App) {
        if !fsr_wgpu::AVAILABLE {
            return;
        }
        let mut raw_vulkan_settings = app
            .world_mut()
            .get_resource_or_init::<RawVulkanInitSettings>();
        // SAFETY: `register_device_extensions` only adds extensions and features the adapter supports.
        unsafe {
            raw_vulkan_settings.add_create_device_callback(|args, adapter, features| {
                if !fsr_wgpu::is_amd(adapter) || !fsr_wgpu::loader_dll_present() {
                    return;
                }
                let changes = fsr_wgpu::register_device_extensions(args, adapter);
                info!("FSR: device prepared ({changes})");
                features.insert::<FsrDevice>();
            });
        }
    }
}

/// Marks a Vulkan device [`FsrInitPlugin`] prepared.
struct FsrDevice;

/// Enables FSR when the device can run it. Part of [`AntiAliasPlugin`](crate::AntiAliasPlugin); needs
/// [`FsrInitPlugin`].
#[derive(Default)]
pub struct FsrPlugin;

impl Plugin for FsrPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Fsr>();
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        if !render_app
            .world()
            .get_resource::<AdditionalVulkanFeatures>()
            .is_some_and(AdditionalVulkanFeatures::has::<FsrDevice>)
        {
            return;
        }
        let device = render_app
            .world()
            .resource::<RenderDevice>()
            .wgpu_device()
            .clone();
        let (sdk, provider) = match FsrSdk::new(&device).and_then(|sdk| {
            let provider = sdk.probe()?;
            Ok((sdk, provider))
        }) {
            Ok(loaded) => loaded,
            Err(error) => {
                warn!("FSR is not available: {error}");
                return;
            }
        };
        info!(
            "FSR: the DLL runs FSR {} (provider id {:#x}) on this GPU",
            provider.name, provider.id
        );

        render_app
            .insert_resource(FsrSdkResource(sdk))
            .add_systems(ExtractSchedule, extract::extract_fsr)
            .add_systems(
                Render,
                prepare::prepare_fsr
                    .in_set(RenderSystems::PrepareViews)
                    .before(prepare_view_targets),
            )
            .add_systems(Core3d, node::fsr.in_set(Core3dSystems::EarlyPostProcess));
        app.insert_resource(FsrSupported {
            provider_name: provider.name,
            provider_id: provider.id,
        });
    }
}

/// Camera component that renders below the output's resolution and upscales with FSR.
///
/// It replaces [`TemporalAntiAliasing`](crate::taa::TemporalAntiAliasing): a camera has one or the other, and
/// it needs [`Msaa::Off`](bevy_render::view::Msaa::Off). Only perspective cameras are upscaled.
///
/// Only available when the [`FsrSupported`] resource exists.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default, Clone)]
#[require(TemporalJitter, MipBias, DepthPrepass, MotionVectorPrepass, Hdr)]
pub struct Fsr {
    /// The internal resolution as a share of the output's on each axis, from just above 0 to 1. At 1 FSR only
    /// anti-aliases.
    ///
    /// AMD's named modes are `2.0 / 3.0` (Quality, 1.5x), `1.0 / 1.7` (Balanced) and `0.5` (Performance).
    pub render_scale: f32,
    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,
}

impl Default for Fsr {
    fn default() -> Self {
        Self {
            render_scale: 2.0 / 3.0,
            reset: false,
        }
    }
}

/// Exists in the main world when FSR runs on this machine.
#[derive(Resource, Clone, Debug)]
pub struct FsrSupported {
    /// The FSR the DLL chose for this GPU, as it names it: `4.x` is FSR 4, `3.1.x` its fallback.
    pub provider_name: String,
    /// The DLL's id for that provider.
    pub provider_id: u64,
}

impl FsrSupported {
    /// The major version of the FSR the DLL chose, from [`Self::provider_name`].
    pub fn major_version(&self) -> Option<u32> {
        fsr_wgpu::ProviderVersion {
            id: self.provider_id,
            name: self.provider_name.clone(),
        }
        .major()
    }
}

#[derive(Resource)]
struct FsrSdkResource(Arc<FsrSdk>);
