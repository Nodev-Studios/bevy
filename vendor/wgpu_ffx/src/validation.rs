use crate::FsrDispatchInfo;

/// Why [`FsrContext::dispatch`](crate::FsrContext::dispatch) refused its parameters.
#[derive(Debug, thiserror::Error)]
pub enum FsrDispatchError {
    #[error("Jitter offset [{x}, {y}] is outside the expected range [-1.0, 1.0]")]
    JitterOffsetOutOfRange { x: f32, y: f32 },

    #[error("Motion vector scale [{x}, {y}] is greater than max size [{max_width}, {max_height}]")]
    MotionVectorScaleTooLarge {
        x: f32,
        y: f32,
        max_width: u32,
        max_height: u32,
    },

    #[error("Motion vector scale contains zero value: [{x}, {y}]")]
    MotionVectorScaleZero { x: f32, y: f32 },

    #[error(
        "Render size [{width}, {height}] is greater than the view's max render size [{max_width}, {max_height}]"
    )]
    RenderSizeTooLarge {
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },

    #[error("Render size contains zero dimension: [{width}, {height}]")]
    RenderSizeZero { width: u32, height: u32 },

    #[error(
        "Upscale size [{width}, {height}] is greater than the view's max upscale size [{max_width}, {max_height}]"
    )]
    UpscaleSizeTooLarge {
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },

    #[error("Upscale size contains zero dimension: [{width}, {height}]")]
    UpscaleSizeZero { width: u32, height: u32 },

    #[error(
        "Frame time delta {0}ms is less than 1.0ms - this value should be milliseconds (~16.6ms for 60fps)"
    )]
    FrameTimeDeltaTooLow(f32),

    #[error(
        "Camera near is {camera_near}, expected f32::MAX or infinity for reverse-Z infinite depth"
    )]
    InvertedInfiniteDepthNearNotMax { camera_near: f32 },

    #[error(
        "Camera far ({camera_far}) is very low (< 0.075), which may cause depth separation artifacts"
    )]
    InvertedDepthFarTooLow { camera_far: f32 },

    #[error("Camera vertical FOV angle must be greater than 0.0, got {0}")]
    CameraFovTooLow(f32),

    #[error(
        "Camera vertical FOV angle is {0} radians, which is greater than 180 degrees (π radians)"
    )]
    CameraFovTooHigh(f32),
}

pub fn check_dispatch(
    info: &FsrDispatchInfo,
    max_render_size: [u32; 2],
    max_upscale_size: [u32; 2],
) -> Result<(), FsrDispatchError> {
    if info.jitter_offset[0].abs() > 1.0 || info.jitter_offset[1].abs() > 1.0 {
        return Err(FsrDispatchError::JitterOffsetOutOfRange {
            x: info.jitter_offset[0],
            y: info.jitter_offset[1],
        });
    }

    if info.motion_vector_scale[0] > max_render_size[0] as f32
        || info.motion_vector_scale[1] > max_render_size[1] as f32
    {
        return Err(FsrDispatchError::MotionVectorScaleTooLarge {
            x: info.motion_vector_scale[0],
            y: info.motion_vector_scale[1],
            max_width: max_render_size[0],
            max_height: max_render_size[1],
        });
    }
    if info.motion_vector_scale[0] == 0.0 || info.motion_vector_scale[1] == 0.0 {
        return Err(FsrDispatchError::MotionVectorScaleZero {
            x: info.motion_vector_scale[0],
            y: info.motion_vector_scale[1],
        });
    }

    if info.render_size[0] > max_render_size[0] || info.render_size[1] > max_render_size[1] {
        return Err(FsrDispatchError::RenderSizeTooLarge {
            width: info.render_size[0],
            height: info.render_size[1],
            max_width: max_render_size[0],
            max_height: max_render_size[1],
        });
    }
    if info.render_size[0] == 0 || info.render_size[1] == 0 {
        return Err(FsrDispatchError::RenderSizeZero {
            width: info.render_size[0],
            height: info.render_size[1],
        });
    }

    if info.upscale_size[0] > max_upscale_size[0] || info.upscale_size[1] > max_upscale_size[1] {
        return Err(FsrDispatchError::UpscaleSizeTooLarge {
            width: info.upscale_size[0],
            height: info.upscale_size[1],
            max_width: max_upscale_size[0],
            max_height: max_upscale_size[1],
        });
    }
    if info.upscale_size[0] == 0 || info.upscale_size[1] == 0 {
        return Err(FsrDispatchError::UpscaleSizeZero {
            width: info.upscale_size[0],
            height: info.upscale_size[1],
        });
    }

    if info.frame_time_delta < 1.0 {
        return Err(FsrDispatchError::FrameTimeDeltaTooLow(
            info.frame_time_delta,
        ));
    }

    if info.camera_near != f32::MAX && info.camera_near != f32::INFINITY {
        return Err(FsrDispatchError::InvertedInfiniteDepthNearNotMax {
            camera_near: info.camera_near,
        });
    }
    if info.camera_far < 0.075 {
        return Err(FsrDispatchError::InvertedDepthFarTooLow {
            camera_far: info.camera_far,
        });
    }

    if info.camera_fov_y <= 0.0 {
        return Err(FsrDispatchError::CameraFovTooLow(info.camera_fov_y));
    }
    if info.camera_fov_y > std::f32::consts::PI {
        return Err(FsrDispatchError::CameraFovTooHigh(info.camera_fov_y));
    }

    Ok(())
}
