//! What the machine can do: [`GraphicsCapabilities`], and the system that reads it from the renderer.

use bevy_ecs::prelude::*;
use bevy_reflect::{std_traits::ReflectDefault, Reflect, ReflectDeserialize, ReflectSerialize};
use bevy_render::renderer::{RenderAdapter, RenderAdapterInfo, RenderDevice};
use serde::{Deserialize, Serialize};
use wgpu_types::{Backend, DeviceType};

/// What the machine the app runs on can do, as far as the graphics ladder cares.
///
/// [`GraphicsPlugin`](crate::GraphicsPlugin) fills it once, in
/// [`GraphicsSystems::DetectCapabilities`](crate::GraphicsSystems::DetectCapabilities); until then, and in an
/// app without a renderer, it is the default: nothing supported.
///
/// [`resolve`](crate::resolve) reads nothing else about the machine, so a test or a tool can build one by hand
/// to ask what another machine would run.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(Resource, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphicsCapabilities {
    /// The adapter's name as its driver reports it, like `NVIDIA GeForce RTX 5070`.
    pub adapter_name: String,
    /// Who made the adapter.
    pub vendor: GpuVendor,
    /// What kind of device the adapter is.
    pub device_type: GpuDeviceType,
    /// The driver's name, like `NVIDIA` or `radv`.
    pub driver: String,
    /// The driver's version text, like `620.73.04` or `Mesa 26.3.1`.
    pub driver_info: String,
    /// The renderer runs on Vulkan. Everything above [`GraphicsRung::Raster`](crate::GraphicsRung::Raster)
    /// needs it.
    pub vulkan: bool,
    /// The size in bytes of the adapter's largest device-local memory heap: its video memory.
    ///
    /// `None` when it could not be read (the renderer is not on Vulkan, or the system is neither Linux nor
    /// Windows). A card sold as "12 GB" reads a little under 12 GiB here.
    pub video_memory_bytes: Option<u64>,
    /// `bevy_graphics` was built with its `solari` feature, so the ray traced presets exist in this app.
    pub solari: bool,
    /// The device has every feature ray traced lighting needs
    /// (`SolariPlugins::required_wgpu_features`). Always `false` without [`Self::solari`].
    ///
    /// On its own this says little about speed: integrated GPUs and CPU rasterizers have it too.
    pub ray_query: bool,
    /// DLSS Super Resolution works on this machine.
    pub dlss_super_resolution: bool,
    /// DLSS Ray Reconstruction works on this machine.
    pub dlss_ray_reconstruction: bool,
    /// The vendor-neutral temporal upscaler is in this app. Set by its module, [`upscalers::tsr`](crate::upscalers::tsr).
    pub tsr: bool,
    /// The FSR version this machine can run. `Some` only under Windows with FSR's DLL loaded; set by
    /// [`upscalers::fsr`](crate::upscalers::fsr).
    pub fsr: Option<FsrLevel>,
    /// The NRD denoiser is in this app. Set by its module, [`denoisers::nrd`](crate::denoisers::nrd).
    pub nrd: bool,
}

/// Who made a graphics adapter, from its PCI vendor id.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Default, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum GpuVendor {
    /// NVIDIA (`0x10de`).
    Nvidia,
    /// AMD (`0x1002`).
    Amd,
    /// Intel (`0x8086`).
    Intel,
    /// Anyone else, with the id. A CPU rasterizer is here too: Mesa's reports `0x10005`.
    Other(u32),
    /// Not read yet, or the app has no renderer.
    #[default]
    Unknown,
}

impl GpuVendor {
    /// The vendor a PCI vendor id belongs to.
    pub fn from_pci_id(id: u32) -> Self {
        match id {
            0x10de => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            other => Self::Other(other),
        }
    }
}

/// What kind of device a graphics adapter is.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Default, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum GpuDeviceType {
    /// A graphics card with its own memory.
    Discrete,
    /// A GPU that shares the system's memory.
    Integrated,
    /// A GPU offered by a virtual machine.
    Virtual,
    /// A rasterizer that runs on the CPU.
    Cpu,
    /// Unknown, not read yet, or the app has no renderer.
    #[default]
    Other,
}

impl From<DeviceType> for GpuDeviceType {
    fn from(device_type: DeviceType) -> Self {
        match device_type {
            DeviceType::DiscreteGpu => Self::Discrete,
            DeviceType::IntegratedGpu => Self::Integrated,
            DeviceType::VirtualGpu => Self::Virtual,
            DeviceType::Cpu => Self::Cpu,
            DeviceType::Other => Self::Other,
        }
    }
}

/// The generation of FSR a machine can run.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[reflect(Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum FsrLevel {
    /// FSR 3.1, which FSR's DLL gives every card FSR 4 does not support.
    Fsr3,
    /// FSR 4.
    Fsr4,
}

/// Fills [`GraphicsCapabilities`] from the renderer's adapter and device.
pub(crate) fn detect_capabilities(world: &mut World) {
    let mut capabilities = GraphicsCapabilities {
        solari: cfg!(feature = "solari"),
        ..Default::default()
    };

    if let Some(info) = world.get_resource::<RenderAdapterInfo>() {
        capabilities.adapter_name.clone_from(&info.name);
        capabilities.vendor = GpuVendor::from_pci_id(info.vendor);
        capabilities.device_type = info.device_type.into();
        capabilities.driver.clone_from(&info.driver);
        capabilities.driver_info.clone_from(&info.driver_info);
        capabilities.vulkan = info.backend == Backend::Vulkan;
    }
    if let Some(adapter) = world.get_resource::<RenderAdapter>() {
        capabilities.video_memory_bytes = video_memory_bytes(adapter);
    }
    if let Some(device) = world.get_resource::<RenderDevice>() {
        capabilities.ray_query = has_ray_tracing_features(device);
    }
    #[cfg(feature = "dlss")]
    {
        use bevy_anti_alias::dlss::{DlssRayReconstructionSupported, DlssSuperResolutionSupported};
        capabilities.dlss_super_resolution =
            world.contains_resource::<DlssSuperResolutionSupported>();
        capabilities.dlss_ray_reconstruction =
            world.contains_resource::<DlssRayReconstructionSupported>();
    }

    world.insert_resource(capabilities);
}

#[cfg(feature = "solari")]
fn has_ray_tracing_features(device: &RenderDevice) -> bool {
    device
        .features()
        .contains(bevy_solari::SolariPlugins::required_wgpu_features())
}

#[cfg(not(feature = "solari"))]
fn has_ray_tracing_features(_device: &RenderDevice) -> bool {
    false
}

/// The size of the adapter's largest device-local heap. wgpu does not expose memory heaps, so this asks Vulkan.
#[cfg(any(target_os = "linux", target_os = "windows"))]
#[expect(
    unsafe_code,
    reason = "Reading the adapter's memory heaps needs the raw Vulkan handles."
)]
fn video_memory_bytes(adapter: &RenderAdapter) -> Option<u64> {
    // SAFETY: the raw adapter is only read from, and the guard is dropped before this function returns.
    let raw_adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let instance = raw_adapter.shared_instance().raw_instance();
    // SAFETY: the physical device handle belongs to this instance, and both outlive the call because
    // `raw_adapter` keeps them alive.
    let memory = unsafe {
        instance.get_physical_device_memory_properties(raw_adapter.raw_physical_device())
    };
    memory
        .memory_heaps_as_slice()
        .iter()
        .filter(|heap| heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn video_memory_bytes(_adapter: &RenderAdapter) -> Option<u64> {
    None
}
