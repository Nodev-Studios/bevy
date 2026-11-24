//! The part of AMD's FidelityFX API (FSR SDK 2.4.0) this crate calls, mirrored by hand, and one Vulkan struct
//! `ash` 0.38 does not have.
//!
//! The types and values below follow the SDK's public headers `ffx_api.h`, `ffx_api_types.h`,
//! `vk/ffx_api_vk.h` and `ffx_upscale.h`, which carry this notice:
//!
//! > Copyright (C) 2026 Advanced Micro Devices, Inc.
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//! > documentation files(the "Software"), to deal in the Software without restriction, including without
//! > limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and /or sell copies of
//! > the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
//! > conditions :
//! >
//! > The above copyright notice and this permission notice shall be included in all copies or substantial
//! > portions of the Software.
//! >
//! > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
//! > LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
//! > EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
//! > IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//! > USE OR OTHER DEALINGS IN THE SOFTWARE.
//!
//! Every size, offset and value here is listed once in [`LAYOUT`] and [`CONSTANTS`]. The `const` assertions at
//! the bottom check the Rust side for whatever target is being built (the Windows cross-build included), and the
//! test `layout_matches_sdk_headers` compiles the same table as C++ `static_assert`s against the fetched SDK's
//! headers, for Linux and for the Windows MSVC ABI.

#![expect(
    non_camel_case_types,
    non_snake_case,
    reason = "The names are the C API's, so that they can be checked against its headers."
)]

use core::ffi::{c_char, c_void};

pub type ffxReturnCode_t = u32;
pub type ffxContext = *mut c_void;
pub type ffxStructType_t = u64;

/// `wchar_t`: 16 bits on Windows, the only system the DLLs exist for.
#[cfg(windows)]
pub type WChar = u16;
#[cfg(not(windows))]
pub type WChar = u32;

pub type ffxApiMessage = Option<unsafe extern "C" fn(message_type: u32, message: *const WChar)>;

pub type PfnFfxCreateContext = unsafe extern "C" fn(
    context: *mut ffxContext,
    desc: *mut ffxApiHeader,
    mem_cb: *const c_void,
) -> ffxReturnCode_t;
pub type PfnFfxDestroyContext =
    unsafe extern "C" fn(context: *mut ffxContext, mem_cb: *const c_void) -> ffxReturnCode_t;
pub type PfnFfxQuery =
    unsafe extern "C" fn(context: *mut ffxContext, desc: *mut ffxApiHeader) -> ffxReturnCode_t;
pub type PfnFfxDispatch =
    unsafe extern "C" fn(context: *mut ffxContext, desc: *const ffxApiHeader) -> ffxReturnCode_t;

pub const FFX_API_RETURN_OK: ffxReturnCode_t = 0;

pub const FFX_API_EFFECT_ID_UPSCALE: u64 = 0x0001_0000;
pub const FFX_API_BACKEND_ID_VK: u64 = 0x0100_0000;

pub const FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: u64 = 6;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK: u64 = FFX_API_BACKEND_ID_VK | 0x03;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: u64 = FFX_API_EFFECT_ID_UPSCALE;
pub const FFX_API_DISPATCH_DESC_TYPE_UPSCALE: u64 = FFX_API_EFFECT_ID_UPSCALE | 0x01;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: u64 =
    FFX_API_EFFECT_ID_UPSCALE | 0x04;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: u64 = FFX_API_EFFECT_ID_UPSCALE | 0x05;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION: u64 = FFX_API_EFFECT_ID_UPSCALE | 0x0b;

/// `FFX_UPSCALER_VERSION` of the headers this crate follows: 4.1.2, packed as `major << 22 | minor << 12 | patch`.
pub const FFX_UPSCALER_VERSION: u32 = (4 << 22) | (1 << 12) | 2;

pub const FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
pub const FFX_UPSCALE_ENABLE_DEPTH_INVERTED: u32 = 1 << 3;
pub const FFX_UPSCALE_ENABLE_DEPTH_INFINITE: u32 = 1 << 4;
pub const FFX_UPSCALE_ENABLE_AUTO_EXPOSURE: u32 = 1 << 5;
pub const FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION: u32 = 1 << 6;
pub const FFX_UPSCALE_ENABLE_DEBUG_CHECKING: u32 = 1 << 7;

pub const FFX_API_RESOURCE_TYPE_TEXTURE2D: u32 = 2;
pub const FFX_API_RESOURCE_FLAGS_NONE: u32 = 0;
pub const FFX_API_RESOURCE_USAGE_READ_ONLY: u32 = 0;
pub const FFX_API_RESOURCE_USAGE_UAV: u32 = 1 << 1;
pub const FFX_API_RESOURCE_USAGE_DEPTHTARGET: u32 = 1 << 2;
pub const FFX_API_RESOURCE_STATE_UNORDERED_ACCESS: u32 = 1 << 1;
pub const FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ: u32 = (1 << 3) | (1 << 2);

pub const FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT: u32 = 3;
pub const FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT: u32 = 4;
pub const FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT: u32 = 16;
pub const FFX_API_SURFACE_FORMAT_R16G16_FLOAT: u32 = 18;
pub const FFX_API_SURFACE_FORMAT_R32_FLOAT: u32 = 28;

/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT`, from the Khronos Vulkan headers.
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT: i32 = 1_000_567_000;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ffxApiHeader {
    pub type_: ffxStructType_t,
    pub pNext: *mut ffxApiHeader,
}

impl ffxApiHeader {
    pub const fn new(type_: ffxStructType_t) -> Self {
        Self {
            type_,
            pNext: core::ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FfxApiDimensions2D {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FfxApiFloatCoords2D {
    pub x: f32,
    pub y: f32,
}

/// The C type has unions over `width`/`size`, `height`/`stride` and `depth`/`alignment`; this crate only
/// describes textures.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FfxApiResourceDescription {
    pub type_: u32,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mipCount: u32,
    pub flags: u32,
    pub usage: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxApiResource {
    pub resource: *mut c_void,
    pub description: FfxApiResourceDescription,
    pub state: u32,
}

impl FfxApiResource {
    pub const NULL: Self = Self {
        resource: core::ptr::null_mut(),
        description: FfxApiResourceDescription {
            type_: 0,
            format: 0,
            width: 0,
            height: 0,
            depth: 0,
            mipCount: 0,
            flags: 0,
            usage: 0,
        },
        state: FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ,
    };
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkDeviceContext {
    pub vkDevice: ash::vk::Device,
    pub vkPhysicalDevice: ash::vk::PhysicalDevice,
    pub vkDeviceProcAddr: ash::vk::PFN_vkGetDeviceProcAddr,
}

#[repr(C)]
pub struct ffxCreateBackendVKDesc {
    pub header: ffxApiHeader,
    pub deviceContext: VkDeviceContext,
}

#[repr(C)]
pub struct ffxCreateContextDescUpscale {
    pub header: ffxApiHeader,
    pub flags: u32,
    pub maxRenderSize: FfxApiDimensions2D,
    pub maxUpscaleSize: FfxApiDimensions2D,
    pub fpMessage: ffxApiMessage,
}

#[repr(C)]
pub struct ffxCreateContextDescUpscaleVersion {
    pub header: ffxApiHeader,
    pub version: u32,
}

#[repr(C)]
pub struct ffxDispatchDescUpscale {
    pub header: ffxApiHeader,
    pub commandList: *mut c_void,
    pub color: FfxApiResource,
    pub depth: FfxApiResource,
    pub motionVectors: FfxApiResource,
    pub exposure: FfxApiResource,
    pub reactive: FfxApiResource,
    pub transparencyAndComposition: FfxApiResource,
    pub output: FfxApiResource,
    pub jitterOffset: FfxApiFloatCoords2D,
    pub motionVectorScale: FfxApiFloatCoords2D,
    pub renderSize: FfxApiDimensions2D,
    pub upscaleSize: FfxApiDimensions2D,
    pub enableSharpening: bool,
    pub sharpness: f32,
    pub frameTimeDelta: f32,
    pub preExposure: f32,
    pub reset: bool,
    pub cameraNear: f32,
    pub cameraFar: f32,
    pub cameraFovAngleVertical: f32,
    pub viewSpaceToMetersFactor: f32,
    pub flags: u32,
}

#[repr(C)]
pub struct ffxQueryDescUpscaleGetJitterPhaseCount {
    pub header: ffxApiHeader,
    pub renderWidth: u32,
    pub displayWidth: u32,
    pub pOutPhaseCount: *mut i32,
}

#[repr(C)]
pub struct ffxQueryDescUpscaleGetJitterOffset {
    pub header: ffxApiHeader,
    pub index: i32,
    pub phaseCount: i32,
    pub pOutX: *mut f32,
    pub pOutY: *mut f32,
}

#[repr(C)]
pub struct ffxQueryGetProviderVersion {
    pub header: ffxApiHeader,
    pub versionId: u64,
    pub versionName: *const c_char,
}

/// `VkPhysicalDeviceShaderFloat8FeaturesEXT` (`VK_EXT_shader_float8`), which FSR 4's FP8 model on RDNA 4 uses.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkPhysicalDeviceShaderFloat8FeaturesEXT {
    pub sType: i32,
    pub pNext: *mut c_void,
    pub shaderFloat8: ash::vk::Bool32,
    pub shaderFloat8CooperativeMatrix: ash::vk::Bool32,
}

/// Lists a C struct's size and the offset of each listed field as `(C expression, value)` rows, and asserts the
/// same numbers on the Rust mirror at compile time.
macro_rules! layout_table {
    ($($c:ident = $rust:ty, $size:literal { $($c_field:tt: $rust_field:ident @ $offset:literal),* $(,)? })*) => {
        /// `sizeof` and `offsetof` of every mirrored C struct, as C expressions with their expected values.
        #[cfg(test)]
        pub const LAYOUT: &[(&str, usize)] = &[
            $(
                (concat!("sizeof(", stringify!($c), ")"), $size),
                $((concat!("offsetof(", stringify!($c), ", ", stringify!($c_field), ")"), $offset),)*
            )*
        ];
        $(
            const _: () = assert!(size_of::<$rust>() == $size, concat!("size of ", stringify!($c)));
            $(
                const _: () = assert!(
                    core::mem::offset_of!($rust, $rust_field) == $offset,
                    concat!("offset of ", stringify!($c), ".", stringify!($c_field))
                );
            )*
        )*
    };
}

layout_table! {
    ffxApiHeader = ffxApiHeader, 16 { type: type_ @ 0, pNext: pNext @ 8 }
    FfxApiDimensions2D = FfxApiDimensions2D, 8 { width: width @ 0, height: height @ 4 }
    FfxApiFloatCoords2D = FfxApiFloatCoords2D, 8 { x: x @ 0, y: y @ 4 }
    FfxApiResourceDescription = FfxApiResourceDescription, 32 {
        type: type_ @ 0, format: format @ 4, width: width @ 8, height: height @ 12, depth: depth @ 16,
        mipCount: mipCount @ 20, flags: flags @ 24, usage: usage @ 28,
    }
    FfxApiResource = FfxApiResource, 48 { resource: resource @ 0, description: description @ 8, state: state @ 40 }
    VkDeviceContext = VkDeviceContext, 24 {
        vkDevice: vkDevice @ 0, vkPhysicalDevice: vkPhysicalDevice @ 8, vkDeviceProcAddr: vkDeviceProcAddr @ 16,
    }
    ffxCreateBackendVKDesc = ffxCreateBackendVKDesc, 40 { header: header @ 0, deviceContext: deviceContext @ 16 }
    ffxCreateContextDescUpscale = ffxCreateContextDescUpscale, 48 {
        header: header @ 0, flags: flags @ 16, maxRenderSize: maxRenderSize @ 20,
        maxUpscaleSize: maxUpscaleSize @ 28, fpMessage: fpMessage @ 40,
    }
    ffxCreateContextDescUpscaleVersion = ffxCreateContextDescUpscaleVersion, 24 { header: header @ 0, version: version @ 16 }
    ffxDispatchDescUpscale = ffxDispatchDescUpscale, 432 {
        header: header @ 0, commandList: commandList @ 16, color: color @ 24, depth: depth @ 72,
        motionVectors: motionVectors @ 120, exposure: exposure @ 168, reactive: reactive @ 216,
        transparencyAndComposition: transparencyAndComposition @ 264, output: output @ 312,
        jitterOffset: jitterOffset @ 360, motionVectorScale: motionVectorScale @ 368, renderSize: renderSize @ 376,
        upscaleSize: upscaleSize @ 384, enableSharpening: enableSharpening @ 392, sharpness: sharpness @ 396,
        frameTimeDelta: frameTimeDelta @ 400, preExposure: preExposure @ 404, reset: reset @ 408,
        cameraNear: cameraNear @ 412, cameraFar: cameraFar @ 416, cameraFovAngleVertical: cameraFovAngleVertical @ 420,
        viewSpaceToMetersFactor: viewSpaceToMetersFactor @ 424, flags: flags @ 428,
    }
    ffxQueryDescUpscaleGetJitterPhaseCount = ffxQueryDescUpscaleGetJitterPhaseCount, 32 {
        header: header @ 0, renderWidth: renderWidth @ 16, displayWidth: displayWidth @ 20,
        pOutPhaseCount: pOutPhaseCount @ 24,
    }
    ffxQueryDescUpscaleGetJitterOffset = ffxQueryDescUpscaleGetJitterOffset, 40 {
        header: header @ 0, index: index @ 16, phaseCount: phaseCount @ 20, pOutX: pOutX @ 24, pOutY: pOutY @ 32,
    }
    ffxQueryGetProviderVersion = ffxQueryGetProviderVersion, 32 {
        header: header @ 0, versionId: versionId @ 16, versionName: versionName @ 24,
    }
    VkPhysicalDeviceShaderFloat8FeaturesEXT = VkPhysicalDeviceShaderFloat8FeaturesEXT, 24 {
        sType: sType @ 0, pNext: pNext @ 8, shaderFloat8: shaderFloat8 @ 16,
        shaderFloat8CooperativeMatrix: shaderFloat8CooperativeMatrix @ 20,
    }
}

/// Every constant above that comes from the headers, as `(C expression, value)`.
#[cfg(test)]
pub const CONSTANTS: &[(&str, u64)] = &[
    ("FFX_API_RETURN_OK", FFX_API_RETURN_OK as u64),
    ("FFX_API_EFFECT_ID_UPSCALE", FFX_API_EFFECT_ID_UPSCALE),
    ("FFX_API_BACKEND_ID_VK", FFX_API_BACKEND_ID_VK),
    (
        "FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION",
        FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION,
    ),
    (
        "FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK",
        FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK,
    ),
    (
        "FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE",
        FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
    ),
    (
        "FFX_API_DISPATCH_DESC_TYPE_UPSCALE",
        FFX_API_DISPATCH_DESC_TYPE_UPSCALE,
    ),
    (
        "FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT",
        FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT,
    ),
    (
        "FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET",
        FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET,
    ),
    (
        "FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION",
        FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION,
    ),
    ("FFX_UPSCALER_VERSION", FFX_UPSCALER_VERSION as u64),
    (
        "FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE",
        FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE as u64,
    ),
    (
        "FFX_UPSCALE_ENABLE_DEPTH_INVERTED",
        FFX_UPSCALE_ENABLE_DEPTH_INVERTED as u64,
    ),
    (
        "FFX_UPSCALE_ENABLE_DEPTH_INFINITE",
        FFX_UPSCALE_ENABLE_DEPTH_INFINITE as u64,
    ),
    (
        "FFX_UPSCALE_ENABLE_AUTO_EXPOSURE",
        FFX_UPSCALE_ENABLE_AUTO_EXPOSURE as u64,
    ),
    (
        "FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION",
        FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION as u64,
    ),
    (
        "FFX_UPSCALE_ENABLE_DEBUG_CHECKING",
        FFX_UPSCALE_ENABLE_DEBUG_CHECKING as u64,
    ),
    (
        "FFX_API_RESOURCE_TYPE_TEXTURE2D",
        FFX_API_RESOURCE_TYPE_TEXTURE2D as u64,
    ),
    (
        "FFX_API_RESOURCE_FLAGS_NONE",
        FFX_API_RESOURCE_FLAGS_NONE as u64,
    ),
    (
        "FFX_API_RESOURCE_USAGE_READ_ONLY",
        FFX_API_RESOURCE_USAGE_READ_ONLY as u64,
    ),
    (
        "FFX_API_RESOURCE_USAGE_UAV",
        FFX_API_RESOURCE_USAGE_UAV as u64,
    ),
    (
        "FFX_API_RESOURCE_USAGE_DEPTHTARGET",
        FFX_API_RESOURCE_USAGE_DEPTHTARGET as u64,
    ),
    (
        "FFX_API_RESOURCE_STATE_UNORDERED_ACCESS",
        FFX_API_RESOURCE_STATE_UNORDERED_ACCESS as u64,
    ),
    (
        "FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ",
        FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ as u64,
    ),
    (
        "FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT",
        FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT as u64,
    ),
    (
        "FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT",
        FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT as u64,
    ),
    (
        "FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT",
        FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT as u64,
    ),
    (
        "FFX_API_SURFACE_FORMAT_R16G16_FLOAT",
        FFX_API_SURFACE_FORMAT_R16G16_FLOAT as u64,
    ),
    (
        "FFX_API_SURFACE_FORMAT_R32_FLOAT",
        FFX_API_SURFACE_FORMAT_R32_FLOAT as u64,
    ),
    (
        "VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT",
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT as u64,
    ),
];

#[cfg(test)]
mod tests {
    use super::{CONSTANTS, LAYOUT};
    use std::{env, fmt::Write as _, fs, path::PathBuf, process::Command};

    /// Compiles [`LAYOUT`] and [`CONSTANTS`] as C++ `static_assert`s against the headers of the SDK in `FSR_SDK`
    /// (`tools/fetch-sdks.sh fsr`): for this machine's ABI with the system C++ compiler, and for Windows' (MSVC,
    /// x86-64) with `clang++` and the MSVC headers `cargo xwin` caches. Prints why and passes when the SDK or a
    /// compiler is missing, so that a build without the SDK still tests.
    #[test]
    fn layout_matches_sdk_headers() {
        let Some(sdk) = env::var_os("FSR_SDK").map(PathBuf::from) else {
            println!("skipped: FSR_SDK is not set (tools/fetch-sdks.sh fsr)");
            return;
        };
        let vulkan_include = env::var_os("VULKAN_SDK")
            .map(|vulkan| PathBuf::from(vulkan).join("include"))
            .unwrap_or_else(|| sdk.join("../vulkan-headers/include"));
        let kits = sdk.join("Kits/FidelityFX");

        let mut source = String::from(
            "#include <cstddef>\n#include <cstdint>\n#include \"ffx_api.h\"\n#include \"ffx_api_types.h\"\n\
             #include \"vk/ffx_api_vk.h\"\n#include \"ffx_upscale.h\"\n",
        );
        for (expression, value) in LAYOUT {
            writeln!(
                source,
                "static_assert({expression} == {value}, \"{expression}\");"
            )
            .unwrap();
        }
        for (expression, value) in CONSTANTS {
            writeln!(
                source,
                "static_assert((uint64_t)({expression}) == {value}ull, \"{expression}\");"
            )
            .unwrap();
        }
        let file = env::temp_dir().join(format!("fsr_wgpu_layout_{}.cpp", std::process::id()));
        fs::write(&file, source).unwrap();

        let includes = [
            kits.join("api/include"),
            kits.join("upscalers/include"),
            vulkan_include,
        ];
        let compile = |compiler: &str, extra: &[String]| {
            let mut command = Command::new(compiler);
            command.args(["-std=c++17", "-fsyntax-only"]).args(extra);
            for include in &includes {
                command.arg("-I").arg(include);
            }
            let output = command.arg(&file).output();
            match output {
                Ok(output) if output.status.success() => true,
                Ok(output) => panic!(
                    "{compiler} {extra:?}: the mirror does not match the SDK's headers\n{}",
                    String::from_utf8_lossy(&output.stderr)
                ),
                Err(error) => {
                    println!("skipped {compiler}: {error}");
                    false
                }
            }
        };

        // `ffx_api.h` marks its functions `__declspec(dllexport)`, which GCC on Linux does not know.
        if compile("c++", &["-D__declspec(x)=".to_string()]) {
            println!(
                "checked {} sizes and offsets and {} constants for this machine's ABI",
                LAYOUT.len(),
                CONSTANTS.len()
            );
        }
        let xwin = env::var_os("XWIN_CACHE_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/cargo-xwin/xwin"))
            });
        match xwin.filter(|xwin| xwin.join("crt/include").is_dir()) {
            Some(xwin) => {
                let mut extra = vec![
                    "--target=x86_64-pc-windows-msvc".to_string(),
                    "-fms-extensions".to_string(),
                ];
                for dir in [
                    "crt/include",
                    "sdk/include/ucrt",
                    "sdk/include/um",
                    "sdk/include/shared",
                ] {
                    extra.push(format!("-isystem{}", xwin.join(dir).display()));
                }
                if compile("clang++", &extra) {
                    println!("checked the same for x86_64-pc-windows-msvc");
                }
            }
            None => println!("skipped the Windows ABI: no MSVC headers from cargo xwin"),
        }
        let _ = fs::remove_file(&file);
    }
}
