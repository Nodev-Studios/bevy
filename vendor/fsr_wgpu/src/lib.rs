//! AMD FSR upscaling (FSR 4, or FSR 3.1 where FSR 4 cannot run) for [`wgpu`] on Vulkan, through the signed DLLs
//! of AMD's FSR SDK 2.4.0 and wgpu's raw Vulkan handles. It is to FSR what `dlss_wgpu` is to DLSS.
//!
//! - [`register_device_extensions`] adds what the DLL uses to the device while wgpu creates it.
//! - [`FsrSdk`] loads `amd_fidelityfx_loader_vk.dll`, which loads `amd_fidelityfx_upscaler_vk.dll` and picks the
//!   newest FSR this device runs; [`FsrSdk::probe`] says which one that is.
//! - [`FsrUpscaler`] is one upscaling context: jitter, then [`FsrUpscaler::render`] into a command buffer.
//!
//! The DLLs exist for Windows only. Elsewhere, and with the `mock` feature, everything compiles and
//! [`FsrSdk::new`] fails. Nothing of the SDK is compiled in: the DLLs are loaded at run time and must sit next
//! to the executable. See the README.

mod device;
mod ffi;

pub use device::{is_amd, register_device_extensions, DeviceChanges, AMD_VENDOR_ID};

use ash::vk::Handle as _;
use core::ffi::{c_void, CStr};
use ffi::*;
use std::{iter, sync::Arc};
use wgpu::{
    hal::api::Vulkan, CommandBuffer, CommandEncoder, CommandEncoderDescriptor, Device, Texture,
    TextureFormat, TextureTransition, TextureUsages, TextureUses,
};

/// The loader DLL the application loads; it finds `amd_fidelityfx_upscaler_vk.dll` next to the executable or next
/// to itself.
pub const LOADER_DLL: &str = "amd_fidelityfx_loader_vk.dll";
/// The DLL with the upscalers, loaded by [`LOADER_DLL`].
pub const UPSCALER_DLL: &str = "amd_fidelityfx_upscaler_vk.dll";

/// Whether this build can run FSR at all: on Windows, without the `mock` feature.
pub const AVAILABLE: bool = cfg!(windows) && !cfg!(feature = "mock");

/// Whether FSR can be loaded here: [`AVAILABLE`] and [`LOADER_DLL`] found. Cheap enough to call while choosing
/// whether to change a device for FSR.
pub fn loader_dll_present() -> bool {
    // SAFETY: as in `FsrSdk::new`; the library is unloaded again at once.
    AVAILABLE && unsafe { libloading::Library::new(LOADER_DLL) }.is_ok()
}

/// Why FSR cannot run, or what failed.
#[derive(Debug, thiserror::Error)]
pub enum FsrError {
    /// This build or system has no FSR: not Windows, or built with the `mock` feature.
    #[error("FSR is not available in this build: {0}")]
    Unavailable(&'static str),
    /// The loader DLL is missing or does not have the expected functions.
    #[error("could not load {LOADER_DLL}: {0}")]
    Load(#[from] libloading::Error),
    /// The device is not a Vulkan device.
    #[error("FSR needs a Vulkan device")]
    NotVulkan,
    /// A texture has a format this crate does not describe to FSR.
    #[error("FSR is not given textures of format {0:?}")]
    UnsupportedFormat(TextureFormat),
    /// An FSR API call returned an error code.
    #[error("{call} returned {code} ({})", return_code_name(*.code))]
    Api {
        /// The call and the descriptor it was given.
        call: &'static str,
        /// `ffxReturnCode_t`.
        code: u32,
    },
}

fn return_code_name(code: u32) -> &'static str {
    match code {
        1 => "error",
        2 => "unknown descriptor type",
        3 => "runtime error",
        4 => "no provider",
        5 => "out of memory",
        6 => "invalid parameter",
        7 => "descriptor type too new for the provider",
        _ => "unknown code",
    }
}

fn check(call: &'static str, code: ffxReturnCode_t) -> Result<(), FsrError> {
    if code == FFX_API_RETURN_OK {
        Ok(())
    } else {
        Err(FsrError::Api { call, code })
    }
}

/// The FSR implementation a context runs, as the DLL names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderVersion {
    /// The provider's id.
    pub id: u64,
    /// Its name for display, like `4.1.2` or `3.1.6`.
    pub name: String,
}

impl ProviderVersion {
    /// The major version: the first number in [`Self::name`].
    pub fn major(&self) -> Option<u32> {
        self.name
            .split(|c: char| !c.is_ascii_digit())
            .find(|part| !part.is_empty())
            .and_then(|part| part.parse().ok())
    }
}

/// The FSR API, loaded from the loader DLL, for one wgpu device.
pub struct FsrSdk {
    create_context: PfnFfxCreateContext,
    destroy_context: PfnFfxDestroyContext,
    query: PfnFfxQuery,
    dispatch: PfnFfxDispatch,
    device: Device,
    device_context: VkDeviceContext,
    // Last, so that it is unloaded after everything above is gone.
    _library: libloading::Library,
}

// SAFETY: the function pointers and Vulkan handles are plain values; the FSR API may be called from any thread as
// long as one context is not used from two at once, which `FsrUpscaler`'s `&mut self` methods ensure.
unsafe impl Send for FsrSdk {}
// SAFETY: as above.
unsafe impl Sync for FsrSdk {}

impl FsrSdk {
    /// Loads the FSR API for `device`, which must have been created with [`register_device_extensions`].
    ///
    /// Fails off Windows, with the `mock` feature, when [`LOADER_DLL`] is not found, or when `device` is not a
    /// Vulkan device.
    pub fn new(device: &Device) -> Result<Arc<Self>, FsrError> {
        if cfg!(feature = "mock") {
            return Err(FsrError::Unavailable(
                "built with fsr_wgpu's `mock` feature",
            ));
        }
        if !cfg!(windows) {
            return Err(FsrError::Unavailable(
                "AMD ships FSR's DLLs for Windows only",
            ));
        }
        // SAFETY: the loader DLL is AMD's signed binary, whose initialisation has no preconditions.
        let library = unsafe { libloading::Library::new(LOADER_DLL) }?;
        // SAFETY: the symbols have these types in `ffx_api.h`.
        let (create_context, destroy_context, query, dispatch) = unsafe {
            (
                *library.get::<PfnFfxCreateContext>(b"ffxCreateContext\0")?,
                *library.get::<PfnFfxDestroyContext>(b"ffxDestroyContext\0")?,
                *library.get::<PfnFfxQuery>(b"ffxQuery\0")?,
                *library.get::<PfnFfxDispatch>(b"ffxDispatch\0")?,
            )
        };
        // SAFETY: the handles are only copied out; `FsrSdk` keeps a clone of `device`, so they outlive it.
        let device_context = unsafe {
            let hal_device = device.as_hal::<Vulkan>().ok_or(FsrError::NotVulkan)?;
            VkDeviceContext {
                vkDevice: hal_device.raw_device().handle(),
                vkPhysicalDevice: hal_device.raw_physical_device(),
                vkDeviceProcAddr: hal_device
                    .shared_instance()
                    .raw_instance()
                    .fp_v1_0()
                    .get_device_proc_addr,
            }
        };
        Ok(Arc::new(Self {
            create_context,
            destroy_context,
            query,
            dispatch,
            device: device.clone(),
            device_context,
            _library: library,
        }))
    }

    /// Which FSR the DLL picks for this device: creates a context for a 1280x720 output, asks for its provider
    /// and destroys it.
    pub fn probe(self: &Arc<Self>) -> Result<ProviderVersion, FsrError> {
        // to tune: no FSR provider documents a minimum size; 720p is a size every one is meant for.
        let size = [1280, 720];
        FsrUpscaler::new(self, size, size).map(|upscaler| upscaler.provider_version().clone())
    }
}

/// The inputs and output of one [`FsrUpscaler::render`].
///
/// All four textures must be the same size or larger than [`FsrUpscaler`]'s maximum render size, with the image
/// in their top-left `render_size` rectangle; `output` gets the upscaled image in its top-left upscale-size
/// rectangle. The inputs need `TEXTURE_BINDING`, the output `STORAGE_BINDING`.
pub struct FsrRenderParameters<'a> {
    /// Linear HDR colour, jittered.
    pub color: &'a Texture,
    /// Reverse-Z depth with an infinite far plane, jittered.
    pub depth: &'a Texture,
    /// Motion vectors at the render resolution, without jitter.
    pub motion_vectors: &'a Texture,
    /// Where the upscaled frame goes.
    pub output: &'a Texture,
    /// The jitter in render pixels, in FSR's sign: the negation of the sub-pixel offset the camera rendered with.
    pub jitter_offset: [f32; 2],
    /// Turns `motion_vectors` into render pixels pointing from this frame to the previous one.
    pub motion_vector_scale: [f32; 2],
    /// The size of the rendered rectangle.
    pub render_size: [u32; 2],
    /// Throw the history away (a camera cut).
    pub reset: bool,
    /// Milliseconds since the previous frame.
    pub frame_time_delta_ms: f32,
    /// The camera's near plane distance.
    pub camera_near: f32,
    /// The vertical field of view, in radians.
    pub camera_fov_y: f32,
}

/// One FSR upscaling context: a fixed output size and a maximum render size.
pub struct FsrUpscaler {
    sdk: Arc<FsrSdk>,
    context: ffxContext,
    upscale_size: [u32; 2],
    max_render_size: [u32; 2],
    provider: ProviderVersion,
}

// SAFETY: the context is only used through `&mut self` or `&self` queries the API allows from any thread.
unsafe impl Send for FsrUpscaler {}
// SAFETY: as above.
unsafe impl Sync for FsrUpscaler {}

impl FsrUpscaler {
    /// Creates a context that upscales to `upscale_size` from at most `max_render_size`.
    ///
    /// The input is described as linear HDR with reverse-Z depth and an infinite far plane, Bevy's conventions;
    /// FSR computes the exposure itself.
    pub fn new(
        sdk: &Arc<FsrSdk>,
        upscale_size: [u32; 2],
        max_render_size: [u32; 2],
    ) -> Result<Self, FsrError> {
        let mut backend = ffxCreateBackendVKDesc {
            header: ffxApiHeader::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK),
            deviceContext: sdk.device_context,
        };
        let mut version = ffxCreateContextDescUpscaleVersion {
            header: ffxApiHeader::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION),
            version: FFX_UPSCALER_VERSION,
        };
        version.header.pNext = (&raw mut backend).cast();
        let mut create = ffxCreateContextDescUpscale {
            header: ffxApiHeader::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE),
            flags: FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE
                | FFX_UPSCALE_ENABLE_DEPTH_INVERTED
                | FFX_UPSCALE_ENABLE_DEPTH_INFINITE
                // AMD's sample and documentation recommend it; Bevy's HDR colour is exposed but not normalised.
                | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE
                // The render rectangle sits inside larger textures and its size changes with the quality.
                | FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION
                // to tune: FSR has never run here; its input checks go to the log until a run comes back clean.
                | FFX_UPSCALE_ENABLE_DEBUG_CHECKING,
            maxRenderSize: FfxApiDimensions2D {
                width: max_render_size[0],
                height: max_render_size[1],
            },
            maxUpscaleSize: FfxApiDimensions2D {
                width: upscale_size[0],
                height: upscale_size[1],
            },
            fpMessage: Some(log_message),
        };
        create.header.pNext = (&raw mut version).cast();

        let mut context: ffxContext = core::ptr::null_mut();
        // SAFETY: every descriptor in the chain is alive until the call returns; the DLL copies what it keeps.
        check("ffxCreateContext(upscale)", unsafe {
            (sdk.create_context)(&mut context, (&raw mut create).cast(), core::ptr::null())
        })?;

        let mut upscaler = Self {
            sdk: Arc::clone(sdk),
            context,
            upscale_size,
            max_render_size,
            provider: ProviderVersion {
                id: 0,
                name: String::new(),
            },
        };
        let mut query = ffxQueryGetProviderVersion {
            header: ffxApiHeader::new(FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION),
            versionId: 0,
            versionName: core::ptr::null(),
        };
        // SAFETY: a query on the live context; the name is copied at once because the SDK documents that version
        // names live in global memory that later queries may overwrite.
        unsafe {
            check(
                "ffxQuery(provider version)",
                (sdk.query)(&mut upscaler.context, (&raw mut query).cast()),
            )?;
            upscaler.provider = ProviderVersion {
                id: query.versionId,
                name: if query.versionName.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(query.versionName)
                        .to_string_lossy()
                        .into_owned()
                },
            };
        }
        Ok(upscaler)
    }

    /// The FSR implementation this context runs.
    pub fn provider_version(&self) -> &ProviderVersion {
        &self.provider
    }

    /// The output size the context was created for.
    pub fn upscale_size(&self) -> [u32; 2] {
        self.upscale_size
    }

    /// The largest render size the context accepts.
    pub fn max_render_size(&self) -> [u32; 2] {
        self.max_render_size
    }

    /// FSR's jitter for `frame`, in render pixels within `[-0.5, 0.5]`, from its own sequence for this ratio of
    /// render to output width. The camera renders with the scene shifted by this much; FSR is then given its
    /// negation ([`FsrRenderParameters::jitter_offset`]), as AMD's sample does.
    pub fn jitter(&mut self, frame: u32, render_width: u32) -> Result<[f32; 2], FsrError> {
        let mut phase_count = 0_i32;
        let mut phase_query = ffxQueryDescUpscaleGetJitterPhaseCount {
            header: ffxApiHeader::new(FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT),
            renderWidth: render_width,
            displayWidth: self.upscale_size[0],
            pOutPhaseCount: &mut phase_count,
        };
        // SAFETY: a query on the live context, writing to a local.
        check("ffxQuery(jitter phase count)", unsafe {
            (self.sdk.query)(&mut self.context, (&raw mut phase_query).cast())
        })?;
        let mut offset = [0.0_f32; 2];
        let [x, y] = &mut offset;
        let mut offset_query = ffxQueryDescUpscaleGetJitterOffset {
            header: ffxApiHeader::new(FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET),
            index: (frame % phase_count.max(1) as u32) as i32,
            phaseCount: phase_count,
            pOutX: x,
            pOutY: y,
        };
        // SAFETY: as above.
        check("ffxQuery(jitter offset)", unsafe {
            (self.sdk.query)(&mut self.context, (&raw mut offset_query).cast())
        })?;
        Ok(offset)
    }

    /// Records the upscale into a command buffer of its own, which must be submitted right after `command_encoder`.
    ///
    /// The textures' state changes go on `command_encoder`, as `dlss_wgpu` does.
    pub fn render(
        &mut self,
        parameters: &FsrRenderParameters,
        command_encoder: &mut CommandEncoder,
    ) -> Result<CommandBuffer, FsrError> {
        let color = resource(parameters.color, false)?;
        let depth = resource(parameters.depth, false)?;
        let motion_vectors = resource(parameters.motion_vectors, false)?;
        let output = resource(parameters.output, true)?;

        command_encoder.transition_resources(
            iter::empty(),
            [
                (parameters.color, TextureUses::RESOURCE),
                (parameters.depth, TextureUses::RESOURCE),
                (parameters.motion_vectors, TextureUses::RESOURCE),
                (parameters.output, TextureUses::STORAGE_READ_WRITE),
            ]
            .into_iter()
            .map(|(texture, state)| TextureTransition {
                texture,
                selector: None,
                state,
            }),
        );

        let mut dispatch = ffxDispatchDescUpscale {
            header: ffxApiHeader::new(FFX_API_DISPATCH_DESC_TYPE_UPSCALE),
            commandList: core::ptr::null_mut(),
            color,
            depth,
            motionVectors: motion_vectors,
            exposure: FfxApiResource::NULL,
            reactive: FfxApiResource::NULL,
            transparencyAndComposition: FfxApiResource::NULL,
            output,
            jitterOffset: FfxApiFloatCoords2D {
                x: parameters.jitter_offset[0],
                y: parameters.jitter_offset[1],
            },
            motionVectorScale: FfxApiFloatCoords2D {
                x: parameters.motion_vector_scale[0],
                y: parameters.motion_vector_scale[1],
            },
            renderSize: FfxApiDimensions2D {
                width: parameters.render_size[0],
                height: parameters.render_size[1],
            },
            upscaleSize: FfxApiDimensions2D {
                width: self.upscale_size[0],
                height: self.upscale_size[1],
            },
            enableSharpening: false,
            sharpness: 0.0,
            frameTimeDelta: parameters.frame_time_delta_ms,
            // Bevy's colour is not pre-exposed for FSR.
            preExposure: 1.0,
            reset: parameters.reset,
            // FSR orders the two planes itself from the depth flags (`FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
            // _INFINITE`); AMD's sample passes the camera's near and far. Bevy's far plane is at infinity, given
            // as the largest finite float so that FSR's arithmetic stays finite.
            cameraNear: parameters.camera_near,
            cameraFar: f32::MAX,
            cameraFovAngleVertical: parameters.camera_fov_y,
            viewSpaceToMetersFactor: 1.0,
            flags: 0,
        };

        let depth_image = depth.resource;
        let mut fsr_encoder = self
            .sdk
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("fsr_upscale"),
            });
        let mut result = Ok(());
        // SAFETY: the raw command buffer is recording (wgpu opens it for `as_hal_mut`); the images belong to this
        // device; the depth image's layout is restored before the closure returns, so wgpu's tracking holds.
        unsafe {
            let hal_device = self
                .sdk
                .device
                .as_hal::<Vulkan>()
                .ok_or(FsrError::NotVulkan)?;
            let raw_device = hal_device.raw_device();
            fsr_encoder.as_hal_mut::<Vulkan, _, _>(|encoder| {
                let Some(encoder) = encoder else {
                    result = Err(FsrError::NotVulkan);
                    return;
                };
                let command_buffer = encoder.raw_handle();
                // wgpu keeps a sampled depth texture in DEPTH_STENCIL_READ_ONLY_OPTIMAL; FSR's Vulkan backend
                // reads every input in SHADER_READ_ONLY_OPTIMAL, the layout AMD's sample gives depth too.
                depth_layout_barrier(raw_device, command_buffer, depth_image.cast(), false);
                dispatch.commandList = command_buffer.as_raw() as *mut c_void;
                result = check(
                    "ffxDispatch(upscale)",
                    (self.sdk.dispatch)(&mut self.context, (&raw const dispatch).cast()),
                );
                depth_layout_barrier(raw_device, command_buffer, depth_image.cast(), true);
            });
        }
        result?;
        Ok(fsr_encoder.finish())
    }
}

impl Drop for FsrUpscaler {
    fn drop(&mut self) {
        // SAFETY: the GPU may still use the context's resources; the device is idle before they are freed.
        unsafe {
            if let Some(hal_device) = self.sdk.device.as_hal::<Vulkan>() {
                let _ = hal_device.raw_device().device_wait_idle();
            }
            let code = (self.sdk.destroy_context)(&mut self.context, core::ptr::null());
            if code != FFX_API_RETURN_OK {
                log::error!(
                    "ffxDestroyContext(upscale) returned {code} ({})",
                    return_code_name(code)
                );
            }
        }
    }
}

/// Moves a depth image between wgpu's sampled-depth layout and the one FSR reads it in.
///
/// # Safety
/// `command_buffer` must be recording on `device`, and `image` must be a depth image of that device in the
/// `from` layout.
unsafe fn depth_layout_barrier(
    device: &ash::Device,
    command_buffer: ash::vk::CommandBuffer,
    image: *mut c_void,
    back_to_wgpu: bool,
) {
    use ash::vk;
    let (old_layout, new_layout) = if back_to_wgpu {
        (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        )
    } else {
        (
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )
    };
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(vk::Image::from_raw(image as u64))
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        });
    // SAFETY: forwarded from the caller.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Describes a wgpu texture to FSR as `ffx_api_vk.h`'s `ffxApiGetImageResourceDescriptionVK` would.
fn resource(texture: &Texture, output: bool) -> Result<FfxApiResource, FsrError> {
    let format = match texture.format() {
        TextureFormat::Rgba16Float => FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
        TextureFormat::Rgba32Float => FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT,
        TextureFormat::Rg11b10Ufloat => FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT,
        TextureFormat::Rg16Float => FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
        TextureFormat::Depth32Float => FFX_API_SURFACE_FORMAT_R32_FLOAT,
        other => return Err(FsrError::UnsupportedFormat(other)),
    };
    let mut usage = FFX_API_RESOURCE_USAGE_READ_ONLY;
    if texture.format().is_depth_stencil_format() {
        usage |= FFX_API_RESOURCE_USAGE_DEPTHTARGET;
    }
    if texture.usage().contains(TextureUsages::STORAGE_BINDING) {
        usage |= FFX_API_RESOURCE_USAGE_UAV;
    }
    // SAFETY: the handle is only passed to FSR while `texture` is borrowed by the caller.
    let image = unsafe {
        texture
            .as_hal::<Vulkan>()
            .ok_or(FsrError::NotVulkan)?
            .raw_handle()
    };
    Ok(FfxApiResource {
        resource: ash::vk::Handle::as_raw(image) as *mut c_void,
        description: FfxApiResourceDescription {
            type_: FFX_API_RESOURCE_TYPE_TEXTURE2D,
            format,
            width: texture.width(),
            height: texture.height(),
            depth: texture.depth_or_array_layers(),
            mipCount: texture.mip_level_count(),
            flags: FFX_API_RESOURCE_FLAGS_NONE,
            usage,
        },
        state: if output {
            FFX_API_RESOURCE_STATE_UNORDERED_ACCESS
        } else {
            FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ
        },
    })
}

/// FSR's messages (`fpMessage`), into the log.
unsafe extern "C" fn log_message(message_type: u32, message: *const WChar) {
    if message.is_null() {
        return;
    }
    // SAFETY: FSR passes a NUL-terminated wide string that lives for the call.
    let units = unsafe {
        let mut length = 0;
        while *message.add(length) != 0 {
            length += 1;
        }
        core::slice::from_raw_parts(message, length)
    };
    #[cfg(windows)]
    let text = String::from_utf16_lossy(units);
    #[cfg(not(windows))]
    let text: String = units
        .iter()
        .map(|&unit| char::from_u32(unit).unwrap_or(char::REPLACEMENT_CHARACTER))
        .collect();
    if message_type == 0 {
        log::error!("FSR: {text}");
    } else {
        log::warn!("FSR: {text}");
    }
}

#[cfg(test)]
mod tests {
    use super::ProviderVersion;

    #[test]
    fn provider_major_version() {
        // The providers' names are `MAKE_VERSION_STRING(major, minor, patch)` in the SDK's open API code
        // (v1.1.4 `ffx-api/src/ffx_provider_fsr3upscale.cpp`); these three are the strings inside FSR SDK 2.4.0's
        // `amd_fidelityfx_upscaler_vk.dll`.
        for (name, major) in [
            ("4.1.2", Some(4)),
            ("3.1.6", Some(3)),
            ("2.3.4", Some(2)),
            ("", None),
        ] {
            let version = ProviderVersion {
                id: 0,
                name: name.into(),
            };
            assert_eq!(version.major(), major, "{name}");
        }
    }
}
