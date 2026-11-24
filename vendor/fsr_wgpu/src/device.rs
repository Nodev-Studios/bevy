//! What FSR's DLL needs from the Vulkan device, added while wgpu creates it.
//!
//! The SDK documents no device requirement for its Vulkan DLLs. What is enabled here comes from two places in
//! FSR SDK 2.4.0: the extension names inside `amd_fidelityfx_upscaler_vk.dll` (`VK_KHR_cooperative_matrix`,
//! `VK_EXT_shader_float8`, `VK_KHR_shader_float16_int8`, `VK_EXT_subgroup_size_control`,
//! `VK_KHR_synchronization2`, ...), and the device AMD's own sample creates
//! (`Kits/Cauldron2/framework/render/vk/device_vk.cpp`), which also asks for `VK_KHR_compute_shader_derivatives`
//! because FSR 4's INT8 model uses quad operations in compute shaders. That sample also turns on every Vulkan 1.2
//! feature the device has; the ones wgpu never chains (8-bit storage, scalar block layout, standard uniform buffer
//! layout, subgroup extended types, separate depth/stencil layouts) are turned on here the same way.
//!
//! wgpu builds the feature structs of the extensions it enables itself *before* this callback runs, and their
//! fields are private. Chaining a second struct of the same type is invalid Vulkan, so a feature whose struct
//! wgpu already chains stays as wgpu set it. That leaves out, on any Vulkan 1.3 driver:
//! - `computeFullSubgroups` (wgpu sets only `subgroupSizeControl`);
//! - `shaderFloat16` and the 16-bit storage features, unless the app asks wgpu for `SHADER_F16`;
//! - `cooperativeMatrix`, if wgpu enabled `VK_KHR_cooperative_matrix` itself (it does when the app asks for
//!   `EXPERIMENTAL_COOPERATIVE_MATRIX`, and then sets the feature from that request);
//! - the core features of `VkPhysicalDeviceFeatures`, among them `shaderStorageImageWriteWithoutFormat`.

use ash::vk;
use core::ffi::{c_void, CStr};
use std::fmt;
use wgpu::hal::vulkan::{Adapter, CreateDeviceCallbackArgs};

use crate::ffi::{
    VkPhysicalDeviceShaderFloat8FeaturesEXT,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
};

/// AMD's PCI vendor id.
pub const AMD_VENDOR_ID: u32 = 0x1002;

const COOPERATIVE_MATRIX: &CStr = c"VK_KHR_cooperative_matrix";
const SHADER_FLOAT8: &CStr = c"VK_EXT_shader_float8";
const COMPUTE_SHADER_DERIVATIVES: &CStr = c"VK_KHR_compute_shader_derivatives";
const SYNCHRONIZATION_2: &CStr = c"VK_KHR_synchronization2";

/// What [`register_device_extensions`] changed, for the log.
#[derive(Clone, Debug, Default)]
pub struct DeviceChanges {
    /// Extensions added to the device.
    pub extensions: Vec<&'static CStr>,
    /// Feature bits turned on, by their Vulkan names.
    pub features: Vec<&'static str>,
    /// Features the adapter has that were left as wgpu set them, because wgpu already chains their struct.
    pub left_to_wgpu: Vec<&'static str>,
}

impl fmt::Display for DeviceChanges {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let extensions: Vec<_> = self
            .extensions
            .iter()
            .map(|name| name.to_string_lossy())
            .collect();
        write!(
            f,
            "extensions added: [{}]; features turned on: [{}]; left as wgpu set them: [{}]",
            extensions.join(", "),
            self.features.join(", "),
            self.left_to_wgpu.join(", ")
        )
    }
}

/// Whether the adapter is one FSR's DLL is offered on: an AMD GPU.
pub fn is_amd(adapter: &Adapter) -> bool {
    // SAFETY: a query on the adapter's own physical device, which `adapter` keeps alive.
    let properties = unsafe {
        adapter
            .shared_instance()
            .raw_instance()
            .get_physical_device_properties(adapter.raw_physical_device())
    };
    properties.vendor_id == AMD_VENDOR_ID
}

/// Adds to the device wgpu is creating the extensions and features FSR's DLL uses, where the adapter has them and
/// wgpu has not already chained their feature struct (see the module's documentation). Nothing is removed.
///
/// Call it from [`Adapter::open_with_callback`]'s callback (Bevy's `RawVulkanInitSettings`). The feature structs
/// it chains are leaked: a few dozen bytes once per device, because the create info borrows them past this call.
///
/// # Safety
/// `args` must be the callback's arguments for a device on `adapter`.
pub unsafe fn register_device_extensions(
    args: &mut CreateDeviceCallbackArgs,
    adapter: &Adapter,
) -> DeviceChanges {
    let instance = adapter.shared_instance().raw_instance();
    let physical_device = adapter.raw_physical_device();
    // SAFETY: queries on the adapter's own physical device.
    let (api_version, supported) = unsafe {
        (
            instance
                .get_physical_device_properties(physical_device)
                .api_version,
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap_or_default(),
        )
    };
    let supports = |name: &CStr| {
        supported
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(name))
    };
    // Extensions already in the list were enabled by wgpu (which then chains their struct) or by an earlier
    // callback; either way their features are not this function's to set.
    let listed = |args: &CreateDeviceCallbackArgs, name: &CStr| args.extensions.contains(&name);
    let mut changes = DeviceChanges::default();

    // Cooperative matrices: the matrix cores both of FSR 4's models (FP8 on RDNA 4, INT8 before it) run on.
    let mut cooperative_matrix = false;
    if supports(COOPERATIVE_MATRIX) {
        let mut query = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        // SAFETY: `query` is a valid struct for this physical device's features query.
        unsafe { query_features(instance, physical_device, (&raw mut query).cast()) };
        if query.cooperative_matrix == vk::TRUE {
            if listed(args, COOPERATIVE_MATRIX) {
                changes.left_to_wgpu.push("cooperativeMatrix");
            } else {
                let feature = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
                    .cooperative_matrix(true);
                // SAFETY: a feature the device supports, of an extension added in the same step.
                unsafe { chain(args, COOPERATIVE_MATRIX, feature, &mut changes) };
                changes.features.push("cooperativeMatrix");
                cooperative_matrix = true;
            }
        }
    }

    // 8-bit floats, for FSR 4's FP8 model (RDNA 4).
    if supports(SHADER_FLOAT8) && !listed(args, SHADER_FLOAT8) {
        let mut query = float8_features(false, false);
        // SAFETY: as above.
        unsafe { query_features(instance, physical_device, (&raw mut query).cast()) };
        if query.shaderFloat8 == vk::TRUE {
            let matrices = cooperative_matrix && query.shaderFloat8CooperativeMatrix == vk::TRUE;
            // SAFETY: as above; the matrix bit only with the cooperative matrix feature it depends on.
            unsafe {
                chain(
                    args,
                    SHADER_FLOAT8,
                    float8_features(true, matrices),
                    &mut changes,
                )
            };
            changes.features.push("shaderFloat8");
            if matrices {
                changes.features.push("shaderFloat8CooperativeMatrix");
            }
        }
    }

    // Quad operations in compute shaders, which FSR 4's INT8 model uses (AMD's sample enables them for it).
    if supports(COMPUTE_SHADER_DERIVATIVES) && !listed(args, COMPUTE_SHADER_DERIVATIVES) {
        // The KHR extension's struct and type value are the NV one's, which is the name `ash` 0.38 knows.
        let mut query = vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default();
        // SAFETY: as above.
        unsafe { query_features(instance, physical_device, (&raw mut query).cast()) };
        if query.compute_derivative_group_linear == vk::TRUE
            || query.compute_derivative_group_quads == vk::TRUE
        {
            let feature = vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
                .compute_derivative_group_linear(query.compute_derivative_group_linear == vk::TRUE)
                .compute_derivative_group_quads(query.compute_derivative_group_quads == vk::TRUE);
            // SAFETY: as above.
            unsafe { chain(args, COMPUTE_SHADER_DERIVATIVES, feature, &mut changes) };
            changes.features.push("computeDerivativeGroup*");
        }
    }

    // `vkCmdPipelineBarrier2`. Core in Vulkan 1.3 but still off unless enabled; wgpu 30 never chains it.
    let core_synchronization_2 = api_version >= vk::API_VERSION_1_3;
    if (core_synchronization_2 || supports(SYNCHRONIZATION_2)) && !listed(args, SYNCHRONIZATION_2) {
        let mut query = vk::PhysicalDeviceSynchronization2Features::default();
        // SAFETY: as above.
        unsafe { query_features(instance, physical_device, (&raw mut query).cast()) };
        if query.synchronization2 == vk::TRUE {
            let feature =
                vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
            let extension = (!core_synchronization_2).then_some(SYNCHRONIZATION_2);
            // SAFETY: as above.
            unsafe { chain_feature(args, extension, feature, &mut changes) };
            changes.features.push("synchronization2");
        }
    }

    // Vulkan 1.2 features AMD's sample turns on wholesale (it chains the whole `VkPhysicalDeviceVulkan12Features`
    // back as queried) and wgpu 30 never chains: chained here with every bit the device reports.
    if api_version >= vk::API_VERSION_1_2 {
        // SAFETY: each is a 1.2 core feature struct no one else chains, holding only bits the device supports.
        unsafe {
            chain_supported::<vk::PhysicalDevice8BitStorageFeatures>(
                instance,
                physical_device,
                args,
                &mut changes,
                "8-bit storage",
            );
            chain_supported::<vk::PhysicalDeviceScalarBlockLayoutFeatures>(
                instance,
                physical_device,
                args,
                &mut changes,
                "scalarBlockLayout",
            );
            chain_supported::<vk::PhysicalDeviceUniformBufferStandardLayoutFeatures>(
                instance,
                physical_device,
                args,
                &mut changes,
                "uniformBufferStandardLayout",
            );
            chain_supported::<vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures>(
                instance,
                physical_device,
                args,
                &mut changes,
                "shaderSubgroupExtendedTypes",
            );
            chain_supported::<vk::PhysicalDeviceSeparateDepthStencilLayoutsFeatures>(
                instance,
                physical_device,
                args,
                &mut changes,
                "separateDepthStencilLayouts",
            );
        }
    }

    changes
}

/// Queries a core feature struct and chains it as queried: every bit the device supports turned on.
///
/// # Safety
/// `T` must be a Vulkan feature struct whose `Default` sets its `sType`, of the device's core version, that no
/// other struct in the chain duplicates.
unsafe fn chain_supported<T: Default + 'static>(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    args: &mut CreateDeviceCallbackArgs,
    changes: &mut DeviceChanges,
    name: &'static str,
) {
    let mut feature = T::default();
    // SAFETY: forwarded; the struct's `pNext` is null from `Default`, and reset after the query.
    unsafe {
        query_features(instance, physical_device, (&raw mut feature).cast());
        (*(&raw mut feature).cast::<vk::BaseOutStructure>()).p_next = core::ptr::null_mut();
        chain_feature(args, None, feature, changes);
    }
    changes.features.push(name);
}

fn float8_features(
    float8: bool,
    cooperative_matrix: bool,
) -> VkPhysicalDeviceShaderFloat8FeaturesEXT {
    VkPhysicalDeviceShaderFloat8FeaturesEXT {
        sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
        pNext: core::ptr::null_mut(),
        shaderFloat8: float8.into(),
        shaderFloat8CooperativeMatrix: cooperative_matrix.into(),
    }
}

/// Fills one feature struct from the physical device.
///
/// # Safety
/// `feature` must point to a Vulkan feature struct with its `sType` set and a null `pNext`.
unsafe fn query_features(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    feature: *mut c_void,
) {
    let mut features2 = vk::PhysicalDeviceFeatures2 {
        p_next: feature,
        ..Default::default()
    };
    // SAFETY: the caller guarantees the chain is one valid struct.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
}

/// Adds `extension` and chains `feature` into the device's create info.
///
/// # Safety
/// The device must support both.
unsafe fn chain<T: 'static>(
    args: &mut CreateDeviceCallbackArgs,
    extension: &'static CStr,
    feature: T,
    changes: &mut DeviceChanges,
) {
    // SAFETY: forwarded.
    unsafe { chain_feature(args, Some(extension), feature, changes) };
}

/// # Safety
/// `T` must be a Vulkan feature struct (`sType`, then `pNext`) that no other struct in the chain duplicates, and
/// the device must support the extension and the features it turns on.
unsafe fn chain_feature<T: 'static>(
    args: &mut CreateDeviceCallbackArgs,
    extension: Option<&'static CStr>,
    feature: T,
    changes: &mut DeviceChanges,
) {
    if let Some(extension) = extension {
        args.extensions.push(extension);
        changes.extensions.push(extension);
    }
    let feature = Box::into_raw(Box::new(feature));
    // SAFETY: every Vulkan struct starts with `sType` (4 bytes, padded to 8) and then `pNext`; the box is leaked,
    // so it outlives the create info that points to it.
    unsafe {
        let base = feature.cast::<vk::BaseOutStructure>();
        (*base).p_next = args.create_info.p_next.cast_mut().cast();
    }
    args.create_info.p_next = feature.cast_const().cast();
}
