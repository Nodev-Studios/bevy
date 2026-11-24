//! AMD FidelityFX Super Resolution 3.1 (FSR 3.1) upscaler for wgpu 30, on Vulkan.
//!
//! A trimmed copy of [wgpu-ffx](https://github.com/cwfitzgerald/wgpu-ffx) (MIT, by the wgpu-ffx developers and
//! AMD), which runs AMD's FSR 3.1 upscaler from the GLSL of the FidelityFX SDK v1.1.4 (MIT). `README.md` lists
//! what differs from wgpu-ffx and why.
//!
//! It runs one configuration, the one its SPIR-V is compiled for: high-dynamic-range colour, reverse-Z depth with
//! an infinite far plane, and motion vectors at the render resolution without jitter. There is no sharpening pass,
//! no reactive-mask generation and no debug view.
//!
//! # Usage
//!
//! 1. Check [`FsrContext::is_supported`], then create an [`FsrContext`]. This builds the pipelines.
//! 2. Create an [`FsrView`] with [`FsrContext::create_view`] for the largest render and upscale sizes the view will
//!    use. This allocates the textures and holds the history.
//! 3. Each frame, record [`FsrContext::dispatch`] into a command encoder.
//!
//! The input textures may be larger than the render size: FSR reads the render-sized rectangle at their origin,
//! which is how dynamic resolution works in AMD's SDK.

mod constants;
mod lanczos2;
mod pass;
mod resources;
mod spd;
mod validation;

use std::mem;

use wgpu::util::DeviceExt as _;

use crate::{
    pass::{FsrPass, FsrPassKind},
    resources::{AccessType, FsrResourceName, OwnedBindingResource},
};

pub use validation::FsrDispatchError;

/// The compiled pipelines of FSR's seven passes. One per device; views share it.
pub struct FsrContext {
    device: wgpu::Device,
    passes: [FsrPass; 7],
}

/// Per-camera state for FSR: every internal texture, the history, and the frame-to-frame constants.
///
/// Create it with [`FsrContext::create_view`], and replace it to change its maximum sizes.
pub struct FsrView {
    constants: constants::Constants,
    resources: resources::FsrResources,

    max_render_size: [u32; 2],
    max_upscale_size: [u32; 2],

    first_execution: bool,
    frame_kind: FrameKind,
}

/// The formats FSR's passes both read and write as storage images.
const READ_WRITE_STORAGE_FORMATS: [wgpu::TextureFormat; 5] = [
    wgpu::TextureFormat::R8Unorm,
    wgpu::TextureFormat::R16Float,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba32Float,
];

impl FsrContext {
    /// Whether FSR runs on this adapter and device.
    ///
    /// The shaders are SPIR-V, loaded unvalidated through [`wgpu::Features::PASSTHROUGH_SHADERS`], so this needs
    /// Vulkan. They also clear textures, filter 32-bit float textures, and read and write
    /// [`READ_WRITE_STORAGE_FORMATS`] as storage images.
    pub fn is_supported(adapter: &wgpu::Adapter, device: &wgpu::Device) -> bool {
        let needed = wgpu::Features::PASSTHROUGH_SHADERS
            | wgpu::Features::CLEAR_TEXTURE
            | wgpu::Features::FLOAT32_FILTERABLE
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        adapter.get_info().backend == wgpu::Backend::Vulkan
            && device.features().contains(needed)
            && READ_WRITE_STORAGE_FORMATS.into_iter().all(|format| {
                adapter
                    .get_texture_format_features(format)
                    .flags
                    .contains(wgpu::TextureFormatFeatureFlags::STORAGE_READ_WRITE)
            })
    }

    /// Builds the pipelines. The device must pass [`FsrContext::is_supported`].
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            device: device.clone(),
            passes: FsrPassKind::ALL.map(|kind| FsrPass::new(device, kind)),
        }
    }

    /// Allocates a view's textures for the largest render and upscale sizes it will use.
    ///
    /// # Panics
    ///
    /// Panics if a dimension of `max_render_size` is below 2, or one of `max_upscale_size` is 0.
    pub fn create_view(
        &self,
        queue: &wgpu::Queue,
        max_render_size: [u32; 2],
        max_upscale_size: [u32; 2],
    ) -> FsrView {
        assert!(
            max_render_size[0] >= 2 && max_render_size[1] >= 2,
            "max_render_size must be at least [2, 2], got {max_render_size:?} \
             (internal half-resolution textures require dimensions >= 1)",
        );
        assert!(
            max_upscale_size[0] >= 1 && max_upscale_size[1] >= 1,
            "max_upscale_size must be at least [1, 1], got {max_upscale_size:?}",
        );
        FsrView {
            constants: constants::Constants {
                fsr: constants::FsrConstants {
                    max_render_size,
                    max_upscale_size,
                    velocity_factor: 1.0,
                    reactiveness_scale: 1.0,
                    shading_change_scale: 1.0,
                    accumulation_added_per_frame: 1.0 / 3.0,
                    min_disocclusion_accumulation: -1.0 / 3.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            resources: resources::FsrResources::new(
                &self.device,
                queue,
                max_render_size,
                max_upscale_size,
            ),
            max_render_size,
            max_upscale_size,
            first_execution: true,
            frame_kind: FrameKind::Even,
        }
    }

    /// Records FSR's passes into `encoder`, after checking `info`. The caller submits the encoder.
    pub fn dispatch(
        &self,
        view: &mut FsrView,
        encoder: &mut wgpu::CommandEncoder,
        info: &FsrDispatchInfo,
    ) -> Result<(), FsrDispatchError> {
        validation::check_dispatch(info, view.max_render_size, view.max_upscale_size)?;

        let reset_accumulation = info.reset_history || view.first_execution;
        view.first_execution = false;

        let fsrc = &mut view.constants.fsr;

        fsrc.previous_frame_jitter_offset = fsrc.jitter_offset;
        fsrc.previous_frame_upscale_size = fsrc.upscale_size;
        fsrc.previous_frame_render_size = fsrc.render_size;

        fsrc.jitter_offset = info.jitter_offset;
        fsrc.upscale_size = info.upscale_size;
        fsrc.render_size = info.render_size;
        fsrc.downscale_factor = [
            info.render_size[0] as f32 / info.upscale_size[0] as f32,
            info.render_size[1] as f32 / info.upscale_size[1] as f32,
        ];

        // compute the horizontal FOV for the shader from the vertical one.
        let aspect_ratio = info.render_size[0] as f32 / info.render_size[1] as f32;
        let camera_angle_horizontal =
            f32::atan(f32::tan(info.camera_fov_y * 0.5) * aspect_ratio) * 2.0;
        fsrc.tan_half_fov = f32::tan(camera_angle_horizontal * 0.5);
        fsrc.view_space_to_meters_factor = info.view_space_to_meters_factor;
        fsrc.device_to_view_depth = device_to_view_depth(info);

        // Bevy-style HDR colour is not pre-exposed: the exposure stays 1.
        fsrc.delta_pre_exposure = 1.0;

        // The shaders want motion vectors in UV units of the render size.
        fsrc.motion_vector_scale =
            std::array::from_fn(|i| info.motion_vector_scale[i] / info.render_size[i] as f32);

        let jitter_phase_count =
            get_jitter_phase_count(info.render_size[0], info.upscale_size[0]) as f32;
        // AMD's SDK moves the count one step a frame towards a new target, so that a resolution change does not
        // jump.
        if reset_accumulation || fsrc.jitter_phase_count == 0.0 {
            fsrc.jitter_phase_count = jitter_phase_count;
        } else if jitter_phase_count > fsrc.jitter_phase_count {
            fsrc.jitter_phase_count += 1.0;
        } else if jitter_phase_count < fsrc.jitter_phase_count {
            fsrc.jitter_phase_count -= 1.0;
        }

        // convert delta time to seconds and clamp to [0, 1].
        fsrc.delta_time = f32::clamp(info.frame_time_delta * 0.001, 0.0, 1.0);

        if reset_accumulation {
            fsrc.frame_index = 0.0;
        } else {
            fsrc.frame_index += 1.0;
        }

        view.constants.spd = constants::SpdConstants::new(info.render_size[0], info.render_size[1]);

        let resources = &view.resources;
        if reset_accumulation {
            // The previous accumulation is what this frame reads.
            let accumulation = resources.to_binding(
                info,
                &pass::ResourceAccess {
                    name: FsrResourceName::AccumulationPrevious,
                    access_type: AccessType::Srv,
                    mip: None,
                },
                view.frame_kind,
            );
            let OwnedBindingResource::View(accumulation) = accumulation else {
                unreachable!("the accumulation is a texture")
            };
            encoder.clear_texture(
                accumulation.texture(),
                &wgpu::ImageSubresourceRange::default(),
            );

            // Frame info starts as exposure -1, log luma 1.
            let mut clear_values_exposure = vec![-1.0f32, 1.0, 0.0, 0.0];
            clear_values_exposure.resize(64, 0.0);
            let staging_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("FSR3 Frame Info Staging"),
                        contents: bytemuck::cast_slice(&clear_values_exposure),
                        usage: wgpu::BufferUsages::COPY_SRC,
                    });
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &resources.frame_info,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        // With reverse-Z the reconstructed depth keeps the nearest sample by `atomicMax`, so it starts at 0 (far).
        // wgpu-ffx cleared both buffers with a WGSL compute pass; zero needs none.
        encoder.clear_buffer(&resources.reconstructed_previous_depth, 0, None);
        encoder.clear_buffer(&resources.spd_atomic_counter, 0, None);
        encoder.clear_texture(&resources.spd_mips, &wgpu::ImageSubresourceRange::default());

        let constants_staging_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FSR3 Constants Staging"),
                    contents: bytemuck::bytes_of(&view.constants),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });
        encoder.copy_buffer_to_buffer(
            &constants_staging_buffer,
            0,
            &resources.constant_buffer,
            0,
            mem::size_of::<constants::Constants>() as wgpu::BufferAddress,
        );

        let groups =
            |[width, height]: [u32; 2], tile: u32| [width.div_ceil(tile), height.div_ceil(tile)];
        let render = groups(info.render_size, 8);
        let half_render = groups([info.render_size[0] / 2, info.render_size[1] / 2], 8);
        let spd = groups(info.render_size, 64);
        let upscale = groups(info.upscale_size, 8);

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FSR3"),
            timestamp_writes: None,
        });
        for (pass, workgroups) in
            self.passes
                .iter()
                .zip([render, spd, spd, half_render, render, render, upscale])
        {
            pass.dispatch(
                &self.device,
                &mut compute_pass,
                resources,
                info,
                view.frame_kind,
                workgroups,
            );
        }
        drop(compute_pass);

        view.frame_kind.advance();

        Ok(())
    }
}

/// The factors that turn a reverse-Z, infinite-far device depth back into view depth (AMD's
/// `setupDeviceDepthToViewSpaceDepthParams`, for that one case).
fn device_to_view_depth(info: &FsrDispatchInfo) -> [f32; 4] {
    // With inverted infinite depth `camera_near` is infinite and `camera_far` is the near plane.
    let near_plane = info.camera_far;
    let aspect = info.render_size[0] as f32 / info.render_size[1] as f32;
    let cot_half_fov_y = 1.0 / f32::tan(0.5 * info.camera_fov_y);
    [
        -f32::EPSILON,
        near_plane,
        aspect / cot_half_fov_y,
        1.0 / cot_half_fov_y,
    ]
}

/// The number of jitter phases AMD's SDK uses for this ratio: `8 · (display / render)²`, truncated.
pub fn get_jitter_phase_count(render_width: u32, display_width: u32) -> u32 {
    (8.0 * f32::powi(display_width as f32 / render_width as f32, 2)) as u32
}

/// Per-frame dispatch parameters for the FSR3 upscaler.
pub struct FsrDispatchInfo {
    /// The jittered colour, `Rgba16Float`, at least `render_size`, with `TEXTURE_BINDING`.
    pub color: wgpu::Texture,
    /// The depth, `Depth32Float` reverse-Z, at least `render_size`, with `TEXTURE_BINDING`.
    pub depth: wgpu::Texture,
    /// Motion vectors at the render resolution, `Rg16Float`, at least `render_size`, with `TEXTURE_BINDING`.
    pub motion_vectors: wgpu::Texture,
    /// The output, `Rgba16Float`, at least `upscale_size`, with `STORAGE_BINDING`.
    pub output: wgpu::Texture,

    /// The sub-pixel jitter offset applied to the camera, in render pixels.
    pub jitter_offset: [f32; 2],
    /// What turns a stored motion vector into pixels of the render size, pointing from this frame to the last.
    pub motion_vector_scale: [f32; 2],

    /// The resolution that was used for rendering the input resources.
    pub render_size: [u32; 2],
    /// The resolution that the upscaler will output.
    pub upscale_size: [u32; 2],

    /// The time elapsed since the last frame (expressed in milliseconds).
    pub frame_time_delta: f32,
    /// A boolean value which when set to true, indicates the camera has moved discontinuously.
    pub reset_history: bool,
    /// `f32::INFINITY` or `f32::MAX`: the far plane of a reverse-Z, infinite-far depth.
    pub camera_near: f32,
    /// The distance to the near plane.
    pub camera_far: f32,
    /// The camera angle field of view in the vertical direction (expressed in radians).
    pub camera_fov_y: f32,
    /// The scale factor to convert view space units to meters
    pub view_space_to_meters_factor: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum FrameKind {
    #[default]
    Even,
    Odd,
}

impl FrameKind {
    fn advance(&mut self) {
        *self = match self {
            FrameKind::Even => FrameKind::Odd,
            FrameKind::Odd => FrameKind::Even,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two frames on the first Vulkan adapter, with inputs larger than the render size as Bevy passes them, and
    /// no wgpu error. Needs a Vulkan GPU.
    #[test]
    fn dispatch_smoke() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no Vulkan adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: adapter.features(),
            required_limits: adapter.limits(),
            // SAFETY: the test only uses passthrough shaders, which this crate's SPIR-V is written for.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }))
        .expect("no device");
        assert!(FsrContext::is_supported(&adapter, &device));
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let output_size = [1280u32, 720u32];
        let render_size = [640u32, 360u32];
        let texture = |format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: output_size[0],
                    height: output_size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let sampled = wgpu::TextureUsages::TEXTURE_BINDING;
        let color = texture(wgpu::TextureFormat::Rgba16Float, sampled);
        let depth = texture(wgpu::TextureFormat::Depth32Float, sampled);
        let motion_vectors = texture(wgpu::TextureFormat::Rg16Float, sampled);
        let output = texture(
            wgpu::TextureFormat::Rgba16Float,
            sampled | wgpu::TextureUsages::STORAGE_BINDING,
        );

        let context = FsrContext::new(&device);
        let mut view = context.create_view(&queue, output_size, output_size);
        for frame in 0..2 {
            let mut encoder = device.create_command_encoder(&Default::default());
            context
                .dispatch(
                    &mut view,
                    &mut encoder,
                    &FsrDispatchInfo {
                        color: color.clone(),
                        depth: depth.clone(),
                        motion_vectors: motion_vectors.clone(),
                        output: output.clone(),
                        jitter_offset: [0.25 * frame as f32, -0.25],
                        motion_vector_scale: [-(render_size[0] as f32), -(render_size[1] as f32)],
                        render_size,
                        upscale_size: output_size,
                        frame_time_delta: 16.6,
                        reset_history: false,
                        camera_near: f32::INFINITY,
                        camera_far: 0.1,
                        camera_fov_y: std::f32::consts::FRAC_PI_4,
                        view_space_to_meters_factor: 1.0,
                    },
                )
                .expect("valid parameters");
            queue.submit([encoder.finish()]);
        }
        let error = pollster::block_on(errors.pop());
        assert!(error.is_none(), "{error:?}");
    }
}
