//! The crate's check: full ReBLUR frames (pack, every dispatch NRD asks for, unpack) on a synthetic noisy floor.
//!
//! A camera looks down at an infinite floor. Every frame each pixel samples either its diffuse or its specular
//! lobe, as a path tracer with one sample per pixel does, with a known mean and heavy noise. No GPU renders the
//! scene: depth, normals, motion vectors and the noisy radiance are computed on the CPU and uploaded, so the
//! only GPU work is the denoiser's.
//!
//! - `still` (the gate): a uniformly lit floor and a camera that does not move. After the last frame the
//!   denoised image must have the mean of the true radiance within 2 % and at least 10 times less variance
//!   than the noisy input, per channel and per lobe, and no frame may hold a NaN or an infinity.
//! - `alternates`: the same floor with the 5x5 hit distance reconstruction and without the stabilization
//!   pass, so that the two shaders ReBLUR swaps in for those settings run as well. Same variance and NaN
//!   checks.
//! - `pan` (a check of the conventions [`FrameSettings`] documents): the floor's light varies along x and the
//!   camera slides along x. Run with the documented motion vector scale and with the opposite sign; the
//!   documented one must give the lower error against the truth.
//!
//! Usage: `reblur_plane [--size WxH] [--frames N] [--precision full|half] [--hit-distance uniform|random]
//! [--timestamps outer|dispatches]`, adapter chosen with `WGPU_ADAPTER_NAME` (a substring). The defaults are
//! the gate. `--precision half` and `--hit-distance random` reproduce the two ways the mean was seen to drift
//! (see the README). `--timestamps` only profiles: it runs `still` and prints the GPU time of the whole
//! denoiser (`outer`: two timestamps around it), or of the whole denoiser and of every dispatch
//! (`dispatches`), each the median of the timed frames. It checks nothing.
//! `--resize-bench ROUNDS` only measures what a change of render size costs, alternating 2560x1440 and 1707x960
//! (DLSS Quality at 1440p): making a denoiser with new pipelines, as every size change did before
//! `Denoiser::resize`, against resizing one, each including its first frame. It also checks that a warmed-up
//! pipeline set needs no new pipeline in a frame, and that `NRD_WGPU_FORCE_DEVICE_ERROR` comes back as an error.
//! Exit code 0 when every check passes.

#![allow(
    clippy::disallowed_methods,
    clippy::needless_range_loop,
    reason = "a standalone check outside Bevy's workspace: no bevy_math, and the two lobes are indexed together"
)]

use nrd_wgpu::{
    Denoiser, DenoiserInputs, DenoiserKind, DenoiserOutputs, DenoiserPipelines, FrameSettings,
    HitDistanceReconstruction, InputFormats, NrdError, ReblurSettings, TexturePrecision,
    FORCE_DEVICE_ERROR_VARIABLE, OUTPUT_FORMAT,
};
use std::{
    process::ExitCode,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Instant,
};
use wgpu::{
    Backends, BufferDescriptor, BufferUsages, Device, DeviceDescriptor, Extent3d, Features,
    Instance, InstanceDescriptor, MapMode, PollType, Queue, TexelCopyBufferInfo,
    TexelCopyBufferLayout, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages,
};

const CAMERA_HEIGHT: f32 = 2.0;
/// The camera looks 45 degrees below the horizon with a 60 degree vertical field of view, so every pixel sees
/// the floor.
const CAMERA_PITCH: f32 = -core::f32::consts::FRAC_PI_4;
const FOV_Y: f32 = core::f32::consts::FRAC_PI_3;
const NEAR: f32 = 0.1;
const DIFFUSE: [f32; 3] = [0.6, 0.45, 0.3];
const SPECULAR: [f32; 3] = [0.25, 0.3, 0.4];
const ROUGHNESS: f32 = 0.5;
/// World units the camera slides per frame in the `pan` scenario.
const PAN_SPEED: f32 = 0.05;
/// World-space period of the light pattern in the `pan` scenario.
const PATTERN_PERIOD: f32 = 2.0;

/// The hit distance a sampled lobe reports.
#[derive(Clone, Copy, PartialEq)]
enum HitDistance {
    /// The same for every pixel and frame. ReBLUR weights samples by their hit distance, so only then is the
    /// mean it returns comparable with the mean it was given.
    Uniform,
    /// An independent random distance per pixel and frame.
    Random,
}

#[derive(Clone, Copy)]
struct Options {
    size: [u32; 2],
    frames: u32,
    precision: TexturePrecision,
    hit_distance: HitDistance,
    timestamps: Timestamps,
}

/// What `--timestamps` times on the GPU.
#[derive(Clone, Copy, PartialEq)]
enum Timestamps {
    Off,
    /// Two timestamps around the denoiser.
    Outer,
    /// The same, and the denoiser's own timestamps between its dispatches.
    Dispatches,
}

#[derive(Clone, Copy, PartialEq)]
enum Scenario {
    Still,
    /// As `Still`, with the settings that make ReBLUR use its other shader permutations.
    Alternates,
    /// The floor's light varies along x; the camera moves when `speed` is not 0.
    Pattern {
        speed: f32,
        motion_vector_scale: f32,
    },
}

struct Camera {
    x: f32,
    aspect: f32,
}

impl Camera {
    fn focal(&self) -> f32 {
        1.0 / (FOV_Y * 0.5).tan()
    }

    /// Bevy's projection: right-handed, reversed depth, infinite far plane. Column-major.
    fn view_to_clip(&self) -> [f32; 16] {
        let f = self.focal();
        [
            f / self.aspect,
            0.0,
            0.0,
            0.0, //
            0.0,
            f,
            0.0,
            0.0, //
            0.0,
            0.0,
            0.0,
            -1.0, //
            0.0,
            0.0,
            NEAR,
            0.0,
        ]
    }

    /// The camera's axes in world space: right, up and back (the camera looks along -back).
    fn axes(&self) -> [[f32; 3]; 3] {
        let (sin, cos) = CAMERA_PITCH.sin_cos();
        [[1.0, 0.0, 0.0], [0.0, cos, sin], [0.0, -sin, cos]]
    }

    fn position(&self) -> [f32; 3] {
        [self.x, CAMERA_HEIGHT, 0.0]
    }

    fn world_to_view(&self) -> [f32; 16] {
        let [right, up, back] = self.axes();
        let position = self.position();
        let translation = [
            -dot(right, position),
            -dot(up, position),
            -dot(back, position),
        ];
        [
            right[0],
            up[0],
            back[0],
            0.0, //
            right[1],
            up[1],
            back[1],
            0.0, //
            right[2],
            up[2],
            back[2],
            0.0, //
            translation[0],
            translation[1],
            translation[2],
            1.0,
        ]
    }

    /// The floor point seen through the centre of a pixel, and its distance along the view axis.
    fn floor_hit(&self, uv: [f32; 2]) -> ([f32; 3], f32) {
        let f = self.focal();
        let view = [
            (uv[0] * 2.0 - 1.0) * self.aspect / f,
            (1.0 - uv[1] * 2.0) / f,
            -1.0,
        ];
        let [right, up, back] = self.axes();
        let direction = [
            right[0] * view[0] + up[0] * view[1] + back[0] * view[2],
            right[1] * view[0] + up[1] * view[1] + back[1] * view[2],
            right[2] * view[0] + up[2] * view[1] + back[2] * view[2],
        ];
        let depth = -CAMERA_HEIGHT / direction[1];
        let position = self.position();
        (
            [
                position[0] + direction[0] * depth,
                0.0,
                position[2] + direction[2] * depth,
            ],
            depth,
        )
    }

    /// Where a world point lands on screen, in uv.
    fn project(&self, point: [f32; 3]) -> [f32; 2] {
        let [right, up, back] = self.axes();
        let position = self.position();
        let relative = [
            point[0] - position[0],
            point[1] - position[1],
            point[2] - position[2],
        ];
        let view = [dot(right, relative), dot(up, relative), dot(back, relative)];
        let f = self.focal();
        let ndc = [f / self.aspect * view[0] / -view[2], f * view[1] / -view[2]];
        [ndc[0] * 0.5 + 0.5, 0.5 - ndc[1] * 0.5]
    }
}

/// Whether a pixel samples its diffuse lobe this frame; it samples the specular one otherwise. An ordered 4x4
/// dither shifted by a new offset every frame, as NRD's documentation asks: every 2x2 block holds both lobes.
fn samples_diffuse(x: u32, y: u32, frame: u32) -> bool {
    const BAYER: [[u32; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    let dither = (BAYER[(y % 4) as usize][(x % 4) as usize] as f32 + 0.5) / 16.0;
    (dither + random(0, frame, 0)).fract() < 0.5
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The true diffuse radiance of a floor point, relative to [`DIFFUSE`].
fn pattern(scenario: Scenario, point: [f32; 3]) -> f32 {
    match scenario {
        Scenario::Still | Scenario::Alternates => 1.0,
        Scenario::Pattern { .. } => {
            1.0 + 0.5 * (core::f32::consts::TAU * point[0] / PATTERN_PERIOD).sin()
        }
    }
}

/// A uniform random number in 0..1 for a pixel, a frame and a stream (PCG's output hash).
fn random(pixel: u32, frame: u32, stream: u32) -> f32 {
    let mut state = pixel
        .wrapping_mul(747_796_405)
        .wrapping_add(frame.wrapping_mul(2_891_336_453))
        .wrapping_add(stream.wrapping_mul(1_181_783_497))
        .wrapping_add(12_345);
    state = state.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    ((word >> 22) ^ word) as f32 / u32::MAX as f32
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;
    if exponent >= 31 {
        // Too large for half precision (or already infinite or NaN).
        sign | 0x7c00 | if value.is_nan() { 0x200 } else { 0 }
    } else if exponent <= 0 {
        if exponent < -10 {
            sign
        } else {
            sign | ((mantissa | 0x0080_0000) >> (14 - exponent)) as u16
        }
    } else {
        sign | ((exponent as u16) << 10) | (mantissa >> 13) as u16
    }
}

fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // A subnormal half: normalize it.
            let shift = mantissa.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((mantissa << (shift + 13)) & 0x007f_ffff)
        }
        (31, _) => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

struct Gpu {
    device: Device,
    queue: Queue,
    errors: Arc<AtomicU32>,
}

fn create_texture(
    gpu: &Gpu,
    size: [u32; 2],
    format: TextureFormat,
    usage: TextureUsages,
) -> Texture {
    gpu.device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn upload<T: Copy>(gpu: &Gpu, texture: &Texture, texels: &[T]) {
    let bytes_per_texel = (core::mem::size_of_val(texels) / texels.len()) as u32;
    // SAFETY: the texel types used here are plain arrays of `u16` or `f32` without padding.
    let bytes = unsafe {
        core::slice::from_raw_parts(texels.as_ptr().cast::<u8>(), core::mem::size_of_val(texels))
    };
    gpu.queue.write_texture(
        texture.as_image_copy(),
        bytes,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(texture.width() * bytes_per_texel),
            rows_per_image: Some(texture.height()),
        },
        texture.size(),
    );
}

/// The last frame of one run: the noisy input, the denoised output and the truth, as RGB per pixel.
struct Run {
    truth: [Vec<[f32; 3]>; 2],
    noisy: [Vec<[f32; 3]>; 2],
    denoised: [Vec<[f32; 3]>; 2],
    non_finite: u64,
    shaders: Vec<(String, bool)>,
    /// Wall-clock time of a frame with no upload and no readback, from recording it to an idle GPU: the
    /// denoiser's three passes and the CPU work of recording them. The median and the fastest of the timed
    /// frames: the GPU is shared with the desktop, and the fastest frame is the one least disturbed.
    frame_milliseconds: [f64; 2],
    /// With `--timestamps`: the GPU time of the whole denoiser, the median of the timed frames.
    outer_milliseconds: f64,
    /// With `--timestamps dispatches`: each dispatch's GPU time, the median of the timed frames.
    dispatch_milliseconds: Vec<(String, f64)>,
    /// With `--timestamps`: wall-clock milliseconds per frame of 120 frames submitted back to back, without and
    /// with the denoiser's own timestamps, three rounds alternated, and the CPU time spent recording them.
    throughput: Vec<(bool, f64, f64)>,
    texture_bytes: u64,
}

fn run(gpu: &Gpu, options: Options, scenario: Scenario) -> Run {
    let Options {
        size,
        frames,
        precision,
        hit_distance,
        timestamps,
    } = options;
    let [width, height] = size;
    let pixels = (width * height) as usize;
    let input_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let output_usage = TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC;
    let diffuse_in = create_texture(gpu, size, TextureFormat::Rgba16Float, input_usage);
    let specular_in = create_texture(gpu, size, TextureFormat::Rgba16Float, input_usage);
    let normal_roughness = create_texture(gpu, size, TextureFormat::Rgba16Float, input_usage);
    let view_depth = create_texture(gpu, size, TextureFormat::R32Float, input_usage);
    let motion_vectors = create_texture(gpu, size, TextureFormat::Rg16Float, input_usage);
    let outputs = [
        create_texture(gpu, size, OUTPUT_FORMAT, output_usage),
        create_texture(gpu, size, OUTPUT_FORMAT, output_usage),
    ];
    let views = [
        &diffuse_in,
        &specular_in,
        &normal_roughness,
        &view_depth,
        &motion_vectors,
    ]
    .map(|texture| texture.create_view(&Default::default()));
    let output_views = [&outputs[0], &outputs[1]].map(|t| t.create_view(&Default::default()));

    // Rows of a texture-to-buffer copy must start at multiples of 256 bytes.
    let row_bytes = (width * 8).next_multiple_of(256);
    let readback = [0, 1].map(|_| {
        gpu.device.create_buffer(&BufferDescriptor {
            label: None,
            size: (row_bytes * height) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    });

    let (speed, motion_vector_scale) = match scenario {
        Scenario::Still | Scenario::Alternates => (0.0, -1.0),
        Scenario::Pattern {
            speed,
            motion_vector_scale,
        } => (speed, motion_vector_scale),
    };
    let aspect = width as f32 / height as f32;
    let mut denoiser = Denoiser::new(
        &gpu.device,
        DenoiserKind::ReblurDiffuseSpecular,
        size,
        precision,
    )
    .expect("the denoiser could not be created");
    if scenario == Scenario::Alternates {
        denoiser
            .set_reblur_settings(&ReblurSettings {
                hit_distance_reconstruction: HitDistanceReconstruction::Area5x5,
                max_stabilized_frames: 0,
                ..Default::default()
            })
            .expect("the settings were refused");
    }

    let inputs = DenoiserInputs {
        diffuse: &views[0],
        specular: &views[1],
        normal_roughness: &views[2],
        view_depth: &views[3],
        motion_vectors: &views[4],
        history_confidence: None,
    };
    let denoiser_outputs = DenoiserOutputs {
        diffuse: &output_views[0],
        specular: &output_views[1],
    };

    let normal = [0.0, 1.0, 0.0, ROUGHNESS].map(f32_to_f16);
    upload(gpu, &normal_roughness, &vec![normal; pixels]);

    let mut result = Run {
        truth: [vec![[0.0; 3]; pixels], vec![[0.0; 3]; pixels]],
        noisy: [vec![[0.0; 3]; pixels], vec![[0.0; 3]; pixels]],
        denoised: [vec![[0.0; 3]; pixels], vec![[0.0; 3]; pixels]],
        non_finite: 0,
        shaders: Vec::new(),
        frame_milliseconds: [0.0; 2],
        outer_milliseconds: 0.0,
        dispatch_milliseconds: Vec::new(),
        throughput: Vec::new(),
        texture_bytes: denoiser.texture_bytes(),
    };
    let mut diffuse_texels = vec![[0u16; 4]; pixels];
    let mut specular_texels = vec![[0u16; 4]; pixels];
    let mut depth_texels = vec![0f32; pixels];
    let mut motion_texels = vec![[0u16; 2]; pixels];

    for frame in 0..frames {
        let camera = Camera {
            x: speed * frame as f32,
            aspect,
        };
        let previous_camera = Camera {
            x: speed * frame.saturating_sub(1) as f32,
            aspect,
        };
        let last = frame + 1 == frames;

        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                let uv = [
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                ];
                let (point, depth) = camera.floor_hit(uv);
                depth_texels[index] = depth;

                // Stored as Bevy's motion vector prepass does: where the point is now minus where it was.
                let previous_uv = previous_camera.project(point);
                motion_texels[index] =
                    [uv[0] - previous_uv[0], uv[1] - previous_uv[1]].map(f32_to_f16);

                let light = pattern(scenario, point);
                let truth = [DIFFUSE.map(|c| c * light), SPECULAR];
                // One lobe per pixel and frame, each half of the time, so a sampled lobe carries twice its
                // radiance; the other lobe holds no radiance and a hit distance of 0.
                let lobe = if samples_diffuse(x, y, frame) { 0 } else { 1 };
                let noise = 4.0 * random(index as u32, frame, 1);
                let mut noisy = [[0.0; 4]; 2];
                for channel in 0..3 {
                    noisy[lobe][channel] = truth[lobe][channel] * noise;
                }
                noisy[lobe][3] = match hit_distance {
                    HitDistance::Uniform => 1.5,
                    HitDistance::Random => 0.5 + 2.0 * random(index as u32, frame, 2),
                };
                diffuse_texels[index] = noisy[0].map(f32_to_f16);
                specular_texels[index] = noisy[1].map(f32_to_f16);

                if last {
                    for lobe in 0..2 {
                        result.truth[lobe][index] = truth[lobe];
                        // What the denoiser was really given, after the rounding to half precision.
                        let texels = if lobe == 0 {
                            &diffuse_texels
                        } else {
                            &specular_texels
                        };
                        result.noisy[lobe][index] =
                            [0, 1, 2].map(|channel| f16_to_f32(texels[index][channel]));
                    }
                }
            }
        }
        upload(gpu, &diffuse_in, &diffuse_texels);
        upload(gpu, &specular_in, &specular_texels);
        upload(gpu, &view_depth, &depth_texels);
        upload(gpu, &motion_vectors, &motion_texels);

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        denoiser
            .denoise(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &frame_settings(&camera, motion_vector_scale, frame),
                &inputs,
                &denoiser_outputs,
            )
            .expect("the frame could not be recorded");
        for lobe in 0..2 {
            encoder.copy_texture_to_buffer(
                outputs[lobe].as_image_copy(),
                TexelCopyBufferInfo {
                    buffer: &readback[lobe],
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row_bytes),
                        rows_per_image: Some(height),
                    },
                },
                outputs[lobe].size(),
            );
        }
        gpu.queue.submit([encoder.finish()]);
        for buffer in &readback {
            buffer
                .slice(..)
                .map_async(MapMode::Read, |result| result.unwrap());
        }
        gpu.device.poll(PollType::wait_indefinitely()).unwrap();

        for lobe in 0..2 {
            {
                let data = readback[lobe].slice(..).get_mapped_range().unwrap();
                for y in 0..height {
                    let row = &data[(y * row_bytes) as usize..][..(width * 8) as usize];
                    for x in 0..width as usize {
                        let texel: [f32; 4] = core::array::from_fn(|channel| {
                            let at = x * 8 + channel * 2;
                            f16_to_f32(u16::from_le_bytes([row[at], row[at + 1]]))
                        });
                        if texel.iter().any(|value| !value.is_finite()) {
                            result.non_finite += 1;
                        }
                        if last {
                            result.denoised[lobe][y as usize * width as usize + x] =
                                [texel[0], texel[1], texel[2]];
                        }
                    }
                }
            }
            readback[lobe].unmap();
        }
    }

    result.shaders = denoiser
        .shaders()
        .map(|(name, used)| (name.to_owned(), used))
        .collect();

    // The cost of a frame, measured on frames of their own: the last inputs again, nothing uploaded or read.
    // The frames before the timed ones give the GPU time to leave its power-saving clocks.
    const WARM_UP_FRAMES: u32 = 60;
    const TIMED_FRAMES: u32 = 60;
    let camera = Camera {
        x: speed * (frames - 1) as f32,
        aspect,
    };
    let mut timings = Vec::new();
    let mut dispatch_timings: Vec<(String, Vec<f64>)> = Vec::new();
    let mut outer_timings = Vec::new();
    // Two timestamps around the whole denoiser, with `--timestamps`.
    let outer_timer = (timestamps != Timestamps::Off).then(|| {
        let buffer = |usage| {
            gpu.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 16,
                usage,
                mapped_at_creation: false,
            })
        };
        (
            gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            buffer(BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC),
            buffer(BufferUsages::COPY_DST | BufferUsages::MAP_READ),
        )
    });
    for frame in frames..frames + WARM_UP_FRAMES + TIMED_FRAMES {
        let timed = frame >= frames + WARM_UP_FRAMES;
        if timestamps == Timestamps::Dispatches && timed {
            denoiser
                .set_timestamps(&gpu.device, true)
                .expect("timestamps were refused");
        }
        let outer = outer_timer.as_ref().filter(|_| timed);
        let started = Instant::now();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        if let Some((queries, ..)) = outer {
            encoder.write_timestamp(queries, 0);
        }
        denoiser
            .denoise(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &frame_settings(&camera, motion_vector_scale, frame),
                &inputs,
                &denoiser_outputs,
            )
            .expect("the frame could not be recorded");
        if let Some((queries, resolved, readback)) = outer {
            encoder.write_timestamp(queries, 1);
            encoder.resolve_query_set(queries, 0..2, resolved, 0);
            encoder.copy_buffer_to_buffer(resolved, 0, readback, 0, 16);
        }
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(PollType::wait_indefinitely()).unwrap();
        if let Some((_, _, readback)) = outer {
            readback
                .slice(..)
                .map_async(MapMode::Read, |result| result.unwrap());
            gpu.device.poll(PollType::wait_indefinitely()).unwrap();
            let ticks: Vec<u64> = readback
                .slice(..)
                .get_mapped_range()
                .unwrap()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|bytes| u64::from_le_bytes(*bytes))
                .collect();
            readback.unmap();
            outer_timings.push(
                ticks[1].wrapping_sub(ticks[0]) as f64 * gpu.queue.get_timestamp_period() as f64
                    / 1e6,
            );
        }
        if timed {
            timings.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        if let Some(dispatches) = denoiser.read_timestamps(&gpu.device, &gpu.queue) {
            for (index, dispatch) in dispatches.into_iter().enumerate() {
                if dispatch_timings.len() <= index {
                    dispatch_timings.push((dispatch.name.clone(), Vec::new()));
                }
                dispatch_timings[index].1.push(dispatch.milliseconds);
            }
        }
    }
    timings.sort_by(f64::total_cmp);
    result.frame_milliseconds = [timings[timings.len() / 2], timings[0]];
    if timestamps != Timestamps::Off {
        const BACK_TO_BACK: u32 = 120;
        let mut frame = frames + WARM_UP_FRAMES + TIMED_FRAMES;
        for _round in 0..3 {
            for inner in [false, true] {
                denoiser
                    .set_timestamps(&gpu.device, inner)
                    .expect("timestamps were refused");
                let started = Instant::now();
                let mut recording = 0.0;
                for _ in 0..BACK_TO_BACK {
                    let recorded = Instant::now();
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    denoiser
                        .denoise(
                            &gpu.device,
                            &gpu.queue,
                            &mut encoder,
                            &frame_settings(&camera, motion_vector_scale, frame),
                            &inputs,
                            &denoiser_outputs,
                        )
                        .expect("the frame could not be recorded");
                    gpu.queue.submit([encoder.finish()]);
                    recording += recorded.elapsed().as_secs_f64() * 1000.0;
                    frame += 1;
                }
                gpu.device.poll(PollType::wait_indefinitely()).unwrap();
                let total = started.elapsed().as_secs_f64() * 1000.0;
                result.throughput.push((
                    inner,
                    total / BACK_TO_BACK as f64,
                    recording / BACK_TO_BACK as f64,
                ));
            }
        }
    }
    if !outer_timings.is_empty() {
        outer_timings.sort_by(f64::total_cmp);
        result.outer_milliseconds = outer_timings[outer_timings.len() / 2];
    }
    result.dispatch_milliseconds = dispatch_timings
        .into_iter()
        .map(|(name, mut times)| {
            times.sort_by(f64::total_cmp);
            (name, times[times.len() / 2])
        })
        .collect();
    result
}

fn frame_settings(camera: &Camera, motion_vector_scale: f32, frame: u32) -> FrameSettings {
    FrameSettings {
        view_to_clip: camera.view_to_clip(),
        world_to_view: camera.world_to_view(),
        jitter: [0.0, 0.0],
        motion_vector_scale: [motion_vector_scale; 2],
        frame_index: frame,
        denoising_range: 1000.0,
        reset_history: false,
    }
}

/// Mean and variance of one channel over the image.
fn mean_variance(image: &[[f32; 3]], channel: usize) -> (f64, f64) {
    let count = image.len() as f64;
    let mean = image.iter().map(|texel| texel[channel] as f64).sum::<f64>() / count;
    let variance = image
        .iter()
        .map(|texel| (texel[channel] as f64 - mean).powi(2))
        .sum::<f64>()
        / count;
    (mean, variance)
}

/// Root mean square error of the denoised diffuse radiance against the truth, relative to the truth's mean.
fn relative_error(run: &Run) -> f64 {
    let mut squared = 0.0;
    let mut truth_sum = 0.0;
    for (denoised, truth) in run.denoised[0].iter().zip(&run.truth[0]) {
        for channel in 0..3 {
            squared += (denoised[channel] as f64 - truth[channel] as f64).powi(2);
            truth_sum += truth[channel] as f64;
        }
    }
    let count = (run.truth[0].len() * 3) as f64;
    (squared / count).sqrt() / (truth_sum / count)
}

fn main() -> ExitCode {
    let mut options = Options {
        size: [640, 360],
        frames: 48,
        precision: TexturePrecision::Full,
        hit_distance: HitDistance::Uniform,
        timestamps: Timestamps::Off,
    };
    let mut resize_rounds = 0u32;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = arguments.next().unwrap_or_default();
        match (argument.as_str(), value.as_str()) {
            ("--size", size) => {
                let (width, height) = size.split_once('x').expect("--size takes WxH");
                options.size = [width.parse().unwrap(), height.parse().unwrap()];
            }
            ("--frames", frames) => {
                options.frames = frames.parse().expect("--frames takes a number")
            }
            ("--precision", "full") => options.precision = TexturePrecision::Full,
            ("--precision", "half") => options.precision = TexturePrecision::Half,
            ("--hit-distance", "uniform") => options.hit_distance = HitDistance::Uniform,
            ("--hit-distance", "random") => options.hit_distance = HitDistance::Random,
            ("--timestamps", "outer") => options.timestamps = Timestamps::Outer,
            ("--timestamps", "dispatches") => options.timestamps = Timestamps::Dispatches,
            ("--resize-bench", rounds) => {
                resize_rounds = rounds.parse().expect("--resize-bench takes a number")
            }
            _ => panic!("unknown argument {argument} {value}"),
        }
    }
    let Options {
        size,
        frames,
        precision,
        hit_distance,
        timestamps,
    } = options;
    assert!(frames >= 32, "the check needs at least 32 frames");

    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::VULKAN,
        ..InstanceDescriptor::new_without_display_handle()
    });
    let wanted = std::env::var("WGPU_ADAPTER_NAME")
        .unwrap_or_default()
        .to_lowercase();
    let adapter = pollster::block_on(instance.enumerate_adapters(Backends::VULKAN))
        .into_iter()
        .find(|adapter| adapter.get_info().name.to_lowercase().contains(&wanted))
        .expect("no Vulkan adapter matches WGPU_ADAPTER_NAME");
    let info = adapter.get_info();
    println!(
        "adapter: {} ({:?}), driver {} {}",
        info.name, info.device_type, info.driver, info.driver_info
    );
    let [major, minor, build] = Denoiser::library_version();
    println!(
        "NRD {major}.{minor}.{build}, {}x{}, {frames} frames per run, {} precision, {} hit distance",
        size[0],
        size[1],
        if precision == TexturePrecision::Full { "full" } else { "half" },
        if hit_distance == HitDistance::Uniform { "uniform" } else { "random" },
    );

    let mut required_features = Denoiser::required_features(precision);
    if timestamps != Timestamps::Off {
        required_features |= Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    }
    let missing = required_features - adapter.features();
    if !missing.is_empty() {
        println!("FAIL: the adapter lacks {missing:?}");
        return ExitCode::FAILURE;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: None,
        required_features,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("the device could not be created");
    let errors = Arc::new(AtomicU32::new(0));
    let error_count = errors.clone();
    device.on_uncaptured_error(Arc::new(move |error| {
        error_count.fetch_add(1, Ordering::Relaxed);
        println!("WGPU ERROR: {error}");
    }));
    let gpu = Gpu {
        device,
        queue,
        errors,
    };

    if resize_rounds > 0 {
        return resize_bench(&gpu, precision, resize_rounds);
    }

    if timestamps != Timestamps::Off {
        let still = run(&gpu, options, Scenario::Still);
        println!(
            "{:.3} ms on the GPU for the whole denoiser, median of 60 frames after 60 warm-up frames",
            still.outer_milliseconds
        );
        for (inner, milliseconds, recording) in &still.throughput {
            println!(
                "back to back, {} the denoiser's timestamps: {milliseconds:.3} ms per frame ({recording:.3} ms of it recording on the CPU)",
                if *inner { "with" } else { "without" }
            );
        }
        println!("GPU time per dispatch, same frames:");
        let mut total = 0.0;
        for (name, milliseconds) in &still.dispatch_milliseconds {
            println!("  {milliseconds:7.3} ms  {name}");
            total += milliseconds;
        }
        println!("  {total:7.3} ms  sum of the medians");
        println!(
            "{:.1} MiB in the denoiser's textures",
            still.texture_bytes as f64 / (1024.0 * 1024.0)
        );
        return ExitCode::SUCCESS;
    }

    let mut passed = true;
    let mut check = |name: &str, ok: bool| {
        println!("{} {name}", if ok { "pass" } else { "FAIL" });
        passed &= ok;
    };

    println!("\n== still: uniform light, camera at rest ==");
    let still = run(&gpu, options, Scenario::Still);
    println!(
        "{:.2} ms per frame, median of 60 frames after 60 warm-up frames; fastest {:.2} ms \
         (wall clock from recording a frame to an idle GPU, no upload or readback)",
        still.frame_milliseconds[0], still.frame_milliseconds[1]
    );
    println!(
        "{:.1} MiB in the denoiser's textures",
        still.texture_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("lobe     channel  truth   noisy mean  noisy var   denoised mean  denoised var  mean error  var ratio");
    let mut worst_mean_error: f64 = 0.0;
    let mut worst_variance_ratio = f64::INFINITY;
    for (lobe, name) in ["diffuse", "specular"].iter().enumerate() {
        for channel in 0..3 {
            let truth = still.truth[lobe][0][channel] as f64;
            let (noisy_mean, noisy_variance) = mean_variance(&still.noisy[lobe], channel);
            let (mean, variance) = mean_variance(&still.denoised[lobe], channel);
            let mean_error = (mean / truth - 1.0).abs();
            let variance_ratio = noisy_variance / variance;
            worst_mean_error = worst_mean_error.max(mean_error);
            worst_variance_ratio = worst_variance_ratio.min(variance_ratio);
            println!(
                "{name:8} {channel:7}  {truth:.4}  {noisy_mean:10.4}  {noisy_variance:9.5}  {mean:13.4}  {variance:12.3e}  {:9.2}%  {variance_ratio:9.0}x",
                mean_error * 100.0
            );
        }
    }
    check(
        &format!("variance down at least 10 times (worst {worst_variance_ratio:.0}x)"),
        worst_variance_ratio >= 10.0,
    );
    check(
        &format!(
            "mean within 2 % of the truth (worst {:.2} %)",
            worst_mean_error * 100.0
        ),
        worst_mean_error <= 0.02,
    );
    check(
        &format!(
            "no NaN or infinity in any frame ({} texels)",
            still.non_finite
        ),
        still.non_finite == 0,
    );

    println!("\n== alternates: 5x5 hit distance reconstruction, no stabilization pass ==");
    let alternates = run(&gpu, options, Scenario::Alternates);
    let mut worst_variance_ratio = f64::INFINITY;
    for lobe in 0..2 {
        for channel in 0..3 {
            let (_, noisy_variance) = mean_variance(&alternates.noisy[lobe], channel);
            let (_, variance) = mean_variance(&alternates.denoised[lobe], channel);
            worst_variance_ratio = worst_variance_ratio.min(noisy_variance / variance);
        }
    }
    check(
        &format!("variance down at least 10 times (worst {worst_variance_ratio:.0}x)"),
        worst_variance_ratio >= 10.0,
    );
    check(
        &format!(
            "no NaN or infinity in any frame ({} texels)",
            alternates.non_finite
        ),
        alternates.non_finite == 0,
    );
    println!("NRD's shaders for this denoiser, and whether a frame of the two runs above dispatched them:");
    for ((name, used), (_, used_by_alternates)) in still.shaders.iter().zip(&alternates.shaders) {
        let used = *used || *used_by_alternates;
        println!("  {} {name}", if used { "used  " } else { "unused" });
    }

    println!("\n== pan: light varies along x, camera slides along x ==");
    let pattern = |speed, motion_vector_scale| {
        run(
            &gpu,
            options,
            Scenario::Pattern {
                speed,
                motion_vector_scale,
            },
        )
    };
    let at_rest = pattern(0.0, -1.0);
    let documented = pattern(PAN_SPEED, -1.0);
    let opposite = pattern(PAN_SPEED, 1.0);
    let errors = [&at_rest, &documented, &opposite].map(relative_error);
    println!("diffuse error against the truth, relative RMS:");
    println!(
        "  camera at rest                          {:.2} %",
        errors[0] * 100.0
    );
    println!(
        "  moving, documented motion vector scale  {:.2} %",
        errors[1] * 100.0
    );
    println!(
        "  moving, opposite motion vector scale    {:.2} %",
        errors[2] * 100.0
    );
    check(
        "the documented motion vector convention tracks the moving floor better than its opposite",
        errors[1] < errors[2],
    );
    let non_finite = at_rest.non_finite + documented.non_finite + opposite.non_finite;
    check(
        &format!("no NaN or infinity in any frame ({non_finite} texels)"),
        non_finite == 0,
    );

    let error_count = gpu.errors.load(Ordering::Relaxed);
    check(&format!("no wgpu error ({error_count})"), error_count == 0);

    println!(
        "\n{}",
        if passed {
            "RESULT: pass"
        } else {
            "RESULT: FAIL"
        }
    );
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Inputs and outputs of one size, filled with a plain floor, for `--resize-bench`.
struct BenchTextures {
    views: [wgpu::TextureView; 5],
    outputs: [wgpu::TextureView; 2],
}

impl BenchTextures {
    fn new(gpu: &Gpu, size: [u32; 2]) -> Self {
        let input_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        let formats = [
            TextureFormat::Rgba16Float,
            TextureFormat::Rgba16Float,
            TextureFormat::Rgba16Float,
            TextureFormat::R32Float,
            TextureFormat::Rg16Float,
        ];
        let pixels = (size[0] * size[1]) as usize;
        let textures = formats.map(|format| create_texture(gpu, size, format, input_usage));
        let radiance = [0.5, 0.5, 0.5, 1.0].map(f32_to_f16);
        upload(gpu, &textures[0], &vec![radiance; pixels]);
        upload(gpu, &textures[1], &vec![radiance; pixels]);
        upload(
            gpu,
            &textures[2],
            &vec![[0.0, 1.0, 0.0, ROUGHNESS].map(f32_to_f16); pixels],
        );
        upload(gpu, &textures[3], &vec![-5.0f32; pixels]);
        upload(gpu, &textures[4], &vec![[0u16; 2]; pixels]);
        Self {
            views: textures.map(|texture| texture.create_view(&Default::default())),
            outputs: [0, 1].map(|_| {
                create_texture(gpu, size, OUTPUT_FORMAT, TextureUsages::STORAGE_BINDING)
                    .create_view(&Default::default())
            }),
        }
    }

    fn inputs(&self) -> DenoiserInputs<'_> {
        DenoiserInputs {
            diffuse: &self.views[0],
            specular: &self.views[1],
            normal_roughness: &self.views[2],
            view_depth: &self.views[3],
            motion_vectors: &self.views[4],
            history_confidence: None,
        }
    }
}

/// Records one frame, submits it and waits for the GPU.
fn bench_frame(gpu: &Gpu, denoiser: &mut Denoiser, textures: &BenchTextures, frame: u32) {
    let camera = Camera {
        x: 0.0,
        aspect: 16.0 / 9.0,
    };
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    denoiser
        .denoise(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &frame_settings(&camera, -1.0, frame),
            &textures.inputs(),
            &DenoiserOutputs {
                diffuse: &textures.outputs[0],
                specular: &textures.outputs[1],
            },
        )
        .expect("the frame could not be recorded");
    gpu.queue.submit([encoder.finish()]);
    gpu.device.poll(PollType::wait_indefinitely()).unwrap();
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// `--resize-bench`: what a change of render size costs before and after `Denoiser::resize`.
fn resize_bench(gpu: &Gpu, precision: TexturePrecision, rounds: u32) -> ExitCode {
    let sizes = [[2560, 1440], [1707, 960]];
    let textures = sizes.map(|size| BenchTextures::new(gpu, size));
    let formats = InputFormats::of(&textures[0].inputs());
    let milliseconds = |start: Instant| start.elapsed().as_secs_f64() * 1000.0;
    let mut passed = true;

    // The shared pipelines, warmed up once; a few frames at each size show none is created later.
    let pipelines = DenoiserPipelines::new(&gpu.device, DenoiserKind::ReblurDiffuseSpecular)
        .expect("the pipelines could not be created");
    pipelines
        .warm_up(&gpu.device, precision, &formats)
        .expect("the warm-up failed");
    let warmed = pipelines.pipeline_count();
    let mut denoiser = Denoiser::with_pipelines(&gpu.device, &pipelines, sizes[0], precision)
        .expect("the denoiser could not be created");
    let mut frame = 0;
    let mut steady = Vec::new();
    for _ in 0..30 {
        let start = Instant::now();
        bench_frame(gpu, &mut denoiser, &textures[0], frame);
        steady.push(milliseconds(start));
        frame += 1;
    }
    let after_frames = pipelines.pipeline_count();
    println!("{warmed} pipelines after the warm-up, {after_frames} after 30 frames");
    let ok = after_frames == warmed;
    println!(
        "{} a frame at the default settings creates no pipeline the warm-up did not",
        if ok { "pass" } else { "FAIL" }
    );
    passed &= ok;

    let mut recreate = Vec::new();
    let mut resize = Vec::new();
    for round in 0..rounds as usize {
        let to = (round + 1) % 2;
        // Before: a new denoiser with pipelines of its own at the new size (modules, pipelines, textures).
        let start = Instant::now();
        let mut fresh = Denoiser::new(
            &gpu.device,
            DenoiserKind::ReblurDiffuseSpecular,
            sizes[to],
            precision,
        )
        .expect("the denoiser could not be created");
        bench_frame(gpu, &mut fresh, &textures[to], 0);
        recreate.push(milliseconds(start));
        drop(fresh);

        // After: the same change through `resize`.
        let start = Instant::now();
        denoiser
            .resize(&gpu.device, sizes[to], precision)
            .expect("the resize failed");
        bench_frame(gpu, &mut denoiser, &textures[to], frame);
        resize.push(milliseconds(start));
        frame += 1;
    }
    let show = |values: &[f64]| {
        values
            .iter()
            .map(|value| format!("{value:.1}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    println!(
        "steady frame at 2560x1440: median {:.2} ms (wall clock, recording to an idle GPU)",
        median(steady)
    );
    println!(
        "size change with a new denoiser and new pipelines (before): median {:.1} ms; each round: {}",
        median(recreate.clone()),
        show(&recreate)
    );
    println!(
        "size change with Denoiser::resize (after): median {:.1} ms; each round: {}",
        median(resize.clone()),
        show(&resize)
    );

    // SAFETY: nothing else reads the environment on another thread here.
    unsafe { std::env::set_var(FORCE_DEVICE_ERROR_VARIABLE, "1") };
    let forced = DenoiserPipelines::new(&gpu.device, DenoiserKind::ReblurDiffuseSpecular);
    unsafe { std::env::remove_var(FORCE_DEVICE_ERROR_VARIABLE) };
    let ok = matches!(&forced, Err(NrdError::Device(_)));
    if let Err(error) = &forced {
        println!("forced device error: {error}");
    }
    println!(
        "{} a device error inside the pipelines' creation comes back as NrdError::Device",
        if ok { "pass" } else { "FAIL" }
    );
    passed &= ok;
    let errors = gpu.errors.load(Ordering::Relaxed);
    println!(
        "{} no wgpu error reached the uncaptured-error handler ({errors})",
        if errors == 0 { "pass" } else { "FAIL" }
    );
    passed &= errors == 0;
    println!("\nRESULT: {}", if passed { "pass" } else { "FAIL" });
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
