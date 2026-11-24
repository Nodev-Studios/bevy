//! The C interface of `shim/nrd_shim.h`, mirrored by hand. Keep the two in step.

use core::ffi::c_char;

#[repr(C)]
pub struct NrdwInstance {
    _opaque: [u8; 0],
}

pub const NRDW_DENOISER_REBLUR_DIFFUSE_SPECULAR: u32 = 0;

pub const NRDW_KIND_UNORM: u32 = 1;
pub const NRDW_KIND_SNORM: u32 = 2;
pub const NRDW_KIND_UINT: u32 = 3;
pub const NRDW_KIND_SINT: u32 = 4;
pub const NRDW_KIND_FLOAT: u32 = 5;
pub const NRDW_KIND_PACKED_10_10_10_2_UNORM: u32 = 6;
pub const NRDW_KIND_PACKED_11_11_10_UFLOAT: u32 = 7;

pub const NRDW_RESOURCE_MOTION_VECTORS: u32 = 1;
pub const NRDW_RESOURCE_NORMAL_ROUGHNESS: u32 = 2;
pub const NRDW_RESOURCE_VIEW_Z: u32 = 3;
pub const NRDW_RESOURCE_IN_DIFFUSE: u32 = 4;
pub const NRDW_RESOURCE_IN_SPECULAR: u32 = 5;
pub const NRDW_RESOURCE_OUT_DIFFUSE: u32 = 6;
pub const NRDW_RESOURCE_OUT_SPECULAR: u32 = 7;
pub const NRDW_RESOURCE_TRANSIENT_POOL: u32 = 8;
pub const NRDW_RESOURCE_PERMANENT_POOL: u32 = 9;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NrdwFormat {
    pub channels: u32,
    pub bits: u32,
    pub kind: u32,
}

#[repr(C)]
pub struct NrdwInfo {
    pub version_major: u32,
    pub version_minor: u32,
    pub version_build: u32,
    pub constants_and_samplers_set: u32,
    pub resources_set: u32,
    pub first_sampler_binding: u32,
    pub constants_binding: u32,
    pub first_texture_binding: u32,
    pub first_storage_binding: u32,
    pub constants_max_size: u32,
    pub sampler_count: u32,
    pub pipeline_count: u32,
    pub permanent_pool_size: u32,
    pub transient_pool_size: u32,
    pub normal_roughness_is_10_10_10_2: u32,
    pub entry_point: *const c_char,
}

#[repr(C)]
pub struct NrdwPipeline {
    pub spirv: *const u32,
    pub spirv_bytes: u64,
    pub texture_count: u32,
    pub storage_count: u32,
    pub name: *const c_char,
}

#[repr(C)]
#[derive(Default)]
pub struct NrdwPoolTexture {
    pub format: NrdwFormat,
    pub downsample_factor: u32,
}

#[repr(C)]
pub struct NrdwFrame {
    pub view_to_clip: [f32; 16],
    pub view_to_clip_previous: [f32; 16],
    pub world_to_view: [f32; 16],
    pub world_to_view_previous: [f32; 16],
    pub motion_vector_scale: [f32; 2],
    pub jitter: [f32; 2],
    pub jitter_previous: [f32; 2],
    pub width: u32,
    pub height: u32,
    pub denoising_range: f32,
    pub frame_index: u32,
    pub reset: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct NrdwReblurSettings {
    pub hit_distance_a: f32,
    pub hit_distance_b: f32,
    pub hit_distance_c: f32,
    pub max_accumulated_frames: u32,
    pub max_fast_accumulated_frames: u32,
    pub max_stabilized_frames: u32,
    pub diffuse_prepass_blur_radius: f32,
    pub specular_prepass_blur_radius: f32,
    pub min_blur_radius: f32,
    pub max_blur_radius: f32,
    pub hit_distance_reconstruction: u32,
    pub anti_firefly: u32,
    pub fast_history_clamping_sigma_scale: f32,
    pub antilag_luminance_sigma_scale: f32,
    pub antilag_luminance_sensitivity: f32,
}

#[repr(C)]
pub struct NrdwDispatch {
    pub name: *const c_char,
    pub pipeline: u32,
    pub grid_width: u32,
    pub grid_height: u32,
    pub resource_count: u32,
    pub constants: *const u8,
    pub constants_size: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct NrdwResource {
    pub kind: u32,
    pub is_storage: u32,
    pub pool_index: u32,
}

#[cfg(not(feature = "mock"))]
unsafe extern "C" {
    pub fn nrdw_create(denoiser: u32, info: *mut NrdwInfo) -> *mut NrdwInstance;
    pub fn nrdw_destroy(instance: *mut NrdwInstance);
    pub fn nrdw_sampler_is_linear(instance: *const NrdwInstance, index: u32) -> u32;
    pub fn nrdw_pipeline(instance: *const NrdwInstance, index: u32, pipeline: *mut NrdwPipeline);
    pub fn nrdw_pool_texture(
        instance: *const NrdwInstance,
        permanent: u32,
        index: u32,
        texture: *mut NrdwPoolTexture,
    );
    pub fn nrdw_reblur_defaults(settings: *mut NrdwReblurSettings);
    pub fn nrdw_set_reblur_settings(
        instance: *mut NrdwInstance,
        settings: *const NrdwReblurSettings,
    ) -> u32;
    pub fn nrdw_set_frame(instance: *mut NrdwInstance, frame: *const NrdwFrame) -> u32;
    pub fn nrdw_begin_dispatches(instance: *mut NrdwInstance) -> i32;
    pub fn nrdw_dispatch(instance: *const NrdwInstance, index: u32, dispatch: *mut NrdwDispatch);
    pub fn nrdw_dispatch_resource(
        instance: *const NrdwInstance,
        dispatch: u32,
        index: u32,
        resource: *mut NrdwResource,
    );
}

/// Stand-ins for the shim with the `mock` feature: no instance is ever created, so nothing past
/// [`nrdw_create`] is reached.
#[cfg(feature = "mock")]
pub use mock::*;

#[cfg(feature = "mock")]
#[expect(
    clippy::missing_safety_doc,
    reason = "the signatures mirror the shim's"
)]
mod mock {
    use super::*;

    pub unsafe fn nrdw_create(_denoiser: u32, _info: *mut NrdwInfo) -> *mut NrdwInstance {
        core::ptr::null_mut()
    }
    pub unsafe fn nrdw_destroy(_instance: *mut NrdwInstance) {}
    pub unsafe fn nrdw_sampler_is_linear(_instance: *const NrdwInstance, _index: u32) -> u32 {
        unreachable!()
    }
    pub unsafe fn nrdw_pipeline(
        _instance: *const NrdwInstance,
        _index: u32,
        _pipeline: *mut NrdwPipeline,
    ) {
        unreachable!()
    }
    pub unsafe fn nrdw_pool_texture(
        _instance: *const NrdwInstance,
        _permanent: u32,
        _index: u32,
        _texture: *mut NrdwPoolTexture,
    ) {
        unreachable!()
    }
    pub unsafe fn nrdw_reblur_defaults(_settings: *mut NrdwReblurSettings) {}
    pub unsafe fn nrdw_set_reblur_settings(
        _instance: *mut NrdwInstance,
        _settings: *const NrdwReblurSettings,
    ) -> u32 {
        unreachable!()
    }
    pub unsafe fn nrdw_set_frame(_instance: *mut NrdwInstance, _frame: *const NrdwFrame) -> u32 {
        unreachable!()
    }
    pub unsafe fn nrdw_begin_dispatches(_instance: *mut NrdwInstance) -> i32 {
        unreachable!()
    }
    pub unsafe fn nrdw_dispatch(
        _instance: *const NrdwInstance,
        _index: u32,
        _dispatch: *mut NrdwDispatch,
    ) {
        unreachable!()
    }
    pub unsafe fn nrdw_dispatch_resource(
        _instance: *const NrdwInstance,
        _dispatch: u32,
        _index: u32,
        _resource: *mut NrdwResource,
    ) {
        unreachable!()
    }
}
