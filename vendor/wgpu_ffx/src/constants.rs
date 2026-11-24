//! The constant buffer every pass binds. Its layout is `cbFSR3Upscaler` and the blocks after it in
//! `shaders/include/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h`; blocks this crate does not fill (the
//! generate-reactive and sharpening passes are not part of it) stay zero but keep their place.

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct Constants {
    pub fsr: FsrConstants,
    pub generate_auto_reactive: [f32; 4],
    pub rcas: [u32; 4],
    pub generate_reactive: [f32; 4],
    pub spd: SpdConstants,
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct FsrConstants {
    pub render_size: [u32; 2],
    pub previous_frame_render_size: [u32; 2],

    pub upscale_size: [u32; 2],
    pub previous_frame_upscale_size: [u32; 2],

    pub max_render_size: [u32; 2],
    pub max_upscale_size: [u32; 2],

    pub device_to_view_depth: [f32; 4],

    pub jitter_offset: [f32; 2],
    pub previous_frame_jitter_offset: [f32; 2],

    pub motion_vector_scale: [f32; 2],
    pub downscale_factor: [f32; 2],

    pub motion_vector_jitter_cancellation: [f32; 2],
    pub tan_half_fov: f32,
    pub jitter_phase_count: f32,

    pub delta_time: f32,
    pub delta_pre_exposure: f32,
    pub view_space_to_meters_factor: f32,
    pub frame_index: f32,

    pub velocity_factor: f32,
    pub reactiveness_scale: f32,
    pub shading_change_scale: f32,
    pub accumulation_added_per_frame: f32,
    pub min_disocclusion_accumulation: f32,
    pub _padding: [u32; 3],
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct SpdConstants {
    pub mips: u32,
    pub num_work_groups: u32,
    pub work_group_offset: [u32; 2],
    pub render_size: [u32; 2],
    pub _padding: [u32; 2],
}
