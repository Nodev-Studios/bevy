//! Runs NVIDIA's NRD denoisers as ordinary [`wgpu`] compute pipelines.
//!
//! NRD issues no graphics calls of its own: it hands out precompiled SPIR-V shaders, the textures it needs and,
//! every frame, a list of compute dispatches. [`Denoiser`] loads those shaders through wgpu's passthrough route,
//! owns the textures and replays the dispatch list into a [`wgpu::CommandEncoder`]. No raw Vulkan handle is used.
//!
//! The crate builds against the NRD SDK, which is not part of this repository: see the README.
//!
//! # Contract
//!
//! [`Denoiser::denoise`] reads [`DenoiserInputs`] and writes [`DenoiserOutputs`]; the two types say what each
//! texture holds. NRD's own encodings never leave this crate: a pack pass in front of NRD and an unpack pass
//! behind it convert to and from them.

mod ffi;

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    ffi::CStr,
    fmt,
    future::Future,
    num::NonZero,
    pin::pin,
    ptr::NonNull,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor,
    ComputePassTimestampWrites, ComputePipeline, ComputePipelineDescriptor, Device, ErrorFilter,
    Extent3d, Features, FilterMode, MapMode, PassthroughShaderEntryPoint, PipelineLayoutDescriptor,
    PollType, QuerySet, QuerySetDescriptor, QueryType, Queue, Sampler, SamplerBindingType,
    SamplerDescriptor, ShaderModule, ShaderModuleDescriptorPassthrough, ShaderStages,
    StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDimension,
};

/// Bytes of the uniform block shared by the pack and unpack passes.
const PARAMS_SIZE: u64 = 32;

/// The format of the two textures of [`DenoiserOutputs`].
pub const OUTPUT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

#[cfg(not(feature = "mock"))]
static PACK_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pack.spv"));
#[cfg(not(feature = "mock"))]
static UNPACK_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unpack.spv"));
#[cfg(feature = "mock")]
static PACK_SPIRV: &[u8] = &[];
#[cfg(feature = "mock")]
static UNPACK_SPIRV: &[u8] = &[];

/// The denoiser a [`Denoiser`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DenoiserKind {
    /// ReBLUR for diffuse and specular radiance, denoised together.
    ReblurDiffuseSpecular,
}

/// The precision of the floating point textures NRD accumulates in.
///
/// NRD asks for 16-bit float textures. Both drivers tried here (NVIDIA 620.73 and Mesa RADV 26.3) round toward
/// zero when a shader stores into one, so a value that is blended a little closer to its target every frame
/// stops short of it, by about one representable step per accumulated frame. NRD's shaders declare their images
/// without a format, which lets them run unchanged on 32-bit textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexturePrecision {
    /// 16-bit floats, as NRD asks. The denoised image comes out darker than its input, by 2 % after 32 frames
    /// and up to 4 % after 96 (measured on a uniform surface, see the README).
    Half,
    /// 32-bit floats wherever NRD asks for 16-bit ones: no darkening, about twice the texture memory. Needs
    /// `Features::FLOAT32_FILTERABLE`.
    Full,
}

/// How ReBLUR fills in the hit distance of a pixel whose lobe was not sampled this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HitDistanceReconstruction {
    /// Every pixel carries a valid hit distance for both lobes.
    Off,
    /// Take it from the 3x3 neighbourhood. The renderer must then sample each lobe often enough for every
    /// neighbourhood to hold a sample.
    Area3x3,
    /// Take it from the 5x5 neighbourhood.
    Area5x5,
}

/// ReBLUR's settings, the part of them this crate exposes. [`Default`] gives NRD's own defaults, except
/// [`hit_distance_reconstruction`](Self::hit_distance_reconstruction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReblurSettings {
    /// The three parameters of the function that normalizes a hit distance to 0..1 (a constant in world units, a
    /// scale by view depth, and a scale by roughness). The pack pass and ReBLUR use the same values.
    pub hit_distance_parameters: [f32; 3],
    /// Most frames the main history accumulates, up to 63.
    pub max_accumulated_frames: u32,
    /// Most frames the fast history accumulates. The main history is clamped to it.
    pub max_fast_accumulated_frames: u32,
    /// Most frames the final stabilization pass accumulates; 0 turns that pass off.
    pub max_stabilized_frames: u32,
    /// Radius in pixels of the spatial pass that runs before accumulation, for diffuse; 0 turns it off.
    pub diffuse_prepass_blur_radius: f32,
    /// The same radius for specular.
    pub specular_prepass_blur_radius: f32,
    /// Blur radius in pixels once the history has converged.
    pub min_blur_radius: f32,
    /// Blur radius in pixels on an empty history.
    pub max_blur_radius: f32,
    /// Defaults to [`HitDistanceReconstruction::Area3x3`], not to NRD's `Off`: the input contract marks a lobe
    /// that was not sampled with a hit distance of 0, which is only valid with reconstruction on.
    pub hit_distance_reconstruction: HitDistanceReconstruction,
    /// Suppresses isolated very bright pixels.
    pub anti_firefly: bool,
    /// How far, in standard deviations of the fast history's neighbourhood, the main history may sit from the
    /// fast one before it is clamped to it (1..=3). Smaller follows a change of light sooner and flickers more.
    pub fast_history_clamping_sigma_scale: f32,
    /// ReBLUR's anti-lag, which shortens the history where the luminance moved more than the noise explains.
    pub antilag: ReblurAntilag,
}

/// ReBLUR's anti-lag settings. NRD's documentation advises tuning them per renderer, and says history confidence
/// inputs, which this crate does not take yet, are the better cure for lag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReblurAntilag {
    /// The luminance change is first reduced by the local standard deviation times this (above 0). Smaller
    /// makes the anti-lag react to smaller changes.
    pub luminance_sigma_scale: f32,
    /// Sensitivity to the change that remains (above 0); smaller is more sensitive.
    pub luminance_sensitivity: f32,
}

impl Default for ReblurSettings {
    fn default() -> Self {
        let mut raw = ffi::NrdwReblurSettings::default();
        // SAFETY: the shim only writes the struct it is given.
        unsafe { ffi::nrdw_reblur_defaults(&mut raw) };
        Self {
            hit_distance_parameters: [raw.hit_distance_a, raw.hit_distance_b, raw.hit_distance_c],
            max_accumulated_frames: raw.max_accumulated_frames,
            max_fast_accumulated_frames: raw.max_fast_accumulated_frames,
            max_stabilized_frames: raw.max_stabilized_frames,
            diffuse_prepass_blur_radius: raw.diffuse_prepass_blur_radius,
            specular_prepass_blur_radius: raw.specular_prepass_blur_radius,
            min_blur_radius: raw.min_blur_radius,
            max_blur_radius: raw.max_blur_radius,
            hit_distance_reconstruction: HitDistanceReconstruction::Area3x3,
            anti_firefly: raw.anti_firefly != 0,
            fast_history_clamping_sigma_scale: raw.fast_history_clamping_sigma_scale,
            antilag: ReblurAntilag {
                luminance_sigma_scale: raw.antilag_luminance_sigma_scale,
                luminance_sensitivity: raw.antilag_luminance_sensitivity,
            },
        }
    }
}

impl TexturePrecision {
    /// The format to create in place of one NRD asks for.
    fn apply(self, format: TextureFormat) -> TextureFormat {
        match (self, format) {
            (Self::Full, TextureFormat::R16Float) => TextureFormat::R32Float,
            (Self::Full, TextureFormat::Rg16Float) => TextureFormat::Rg32Float,
            (Self::Full, TextureFormat::Rgba16Float) => TextureFormat::Rgba32Float,
            _ => format,
        }
    }
}

impl ReblurSettings {
    fn to_raw(self) -> ffi::NrdwReblurSettings {
        ffi::NrdwReblurSettings {
            hit_distance_a: self.hit_distance_parameters[0],
            hit_distance_b: self.hit_distance_parameters[1],
            hit_distance_c: self.hit_distance_parameters[2],
            max_accumulated_frames: self.max_accumulated_frames,
            max_fast_accumulated_frames: self.max_fast_accumulated_frames,
            max_stabilized_frames: self.max_stabilized_frames,
            diffuse_prepass_blur_radius: self.diffuse_prepass_blur_radius,
            specular_prepass_blur_radius: self.specular_prepass_blur_radius,
            min_blur_radius: self.min_blur_radius,
            max_blur_radius: self.max_blur_radius,
            hit_distance_reconstruction: match self.hit_distance_reconstruction {
                HitDistanceReconstruction::Off => 0,
                HitDistanceReconstruction::Area3x3 => 3,
                HitDistanceReconstruction::Area5x5 => 5,
            },
            anti_firefly: self.anti_firefly as u32,
            fast_history_clamping_sigma_scale: self.fast_history_clamping_sigma_scale,
            antilag_luminance_sigma_scale: self.antilag.luminance_sigma_scale,
            antilag_luminance_sensitivity: self.antilag.luminance_sensitivity,
        }
    }
}

/// What changes from one frame to the next. The denoiser remembers the previous frame's matrices and jitter
/// itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameSettings {
    /// The projection matrix **without jitter**, column-major, for column vectors (`glam`'s
    /// `Mat4::to_cols_array`). Either handedness, reversed depth and an infinite far plane are all understood.
    pub view_to_clip: [f32; 16],
    /// The world-to-view matrix, in the same layout.
    pub world_to_view: [f32; 16],
    /// The sub-pixel jitter of this frame in pixels, each component in -0.5..=0.5.
    pub jitter: [f32; 2],
    /// The motion vector texture is multiplied by this to give `previous uv - current uv`. A texture that stores
    /// `current uv - previous uv` (Bevy's motion vector prepass) takes `[-1.0, -1.0]`.
    pub motion_vector_scale: [f32; 2],
    /// Must grow by exactly one every frame.
    pub frame_index: u32,
    /// Pixels at or beyond this view depth, in world units, are not denoised and come out black.
    pub denoising_range: f32,
    /// Drops the accumulated history, after a camera cut for example.
    pub reset_history: bool,
}

/// The textures [`Denoiser::denoise`] reads. All are at the denoiser's size and none is in an encoding of NRD's.
#[derive(Clone, Copy, Debug)]
pub struct DenoiserInputs<'a> {
    /// `rgb`: diffuse radiance with the material demodulated (the surface's own emission left out). `a`: world
    /// distance from the surface to the first hit of the diffuse ray, or 0 where the diffuse lobe was not sampled
    /// in this pixel.
    pub diffuse: &'a TextureView,
    /// The same for the specular lobe.
    pub specular: &'a TextureView,
    /// `xyz`: world-space normal, unit length. `w`: perceptual roughness (the square root of the GGX alpha).
    pub normal_roughness: &'a TextureView,
    /// `r`: linear view-space depth in world units, of either sign. 0, a non-finite value or a depth beyond the
    /// denoising range marks a pixel with no surface.
    pub view_depth: &'a TextureView,
    /// `rg`: screen-space motion in uv units, without jitter. See [`FrameSettings::motion_vector_scale`].
    pub motion_vectors: &'a TextureView,
    /// `r`: how far the accumulated history of the previous frame can still be trusted, in 0..=1: 0 drops it,
    /// 1 keeps accumulating as NRD would without this input. ReBLUR reads it, for both lobes, at the uv of the
    /// previous frame with a linear filter, so it may be smaller than the denoiser; any filterable float format.
    /// NRD's README calls this input the best cure for lag after a change of light, which its own anti-lag
    /// (`ReblurSettings::antilag`) does not remove.
    ///
    /// Give it on every frame of a denoiser's life or on none: NRD runs other shaders with it.
    pub history_confidence: Option<&'a TextureView>,
}

/// The textures [`Denoiser::denoise`] writes: two [`OUTPUT_FORMAT`] textures with `STORAGE_BINDING` usage at the
/// denoiser's size. They may be the textures of [`DenoiserInputs::diffuse`] and
/// [`DenoiserInputs::specular`]: the inputs are read before the outputs are written.
#[derive(Clone, Copy, Debug)]
pub struct DenoiserOutputs<'a> {
    /// `rgb`: denoised demodulated diffuse radiance. `a`: the diffuse hit distance normalized to 0..1, an
    /// ambient occlusion term. Pixels with no surface are 0.
    pub diffuse: &'a TextureView,
    /// The same for the specular lobe; `a` is a specular occlusion term.
    pub specular: &'a TextureView,
}

/// Why a [`Denoiser`] could not be created or run.
#[derive(Debug)]
pub enum NrdError {
    /// The device was created without some of [`Denoiser::required_features`].
    MissingFeatures(Features),
    /// NRD refused to create the denoiser.
    CreationFailed,
    /// The NRD library was built with an option this crate does not handle.
    UnsupportedLibrary(&'static str),
    /// NRD asks for a texture format wgpu does not have.
    UnsupportedFormat,
    /// A texture has a format that cannot be bound where it is needed.
    UnsupportedTextureFormat(TextureFormat),
    /// NRD refused the settings.
    InvalidSettings(&'static str),
    /// A texture is not the size the denoiser was created for.
    SizeMismatch {
        /// The denoiser's size.
        expected: [u32; 2],
        /// The texture's size.
        found: [u32; 2],
    },
    /// A dispatch of NRD's uses a resource this crate does not provide.
    UnsupportedResource(String),
    /// wgpu or the driver refused to create a shader module, a pipeline or a texture (a wgpu validation,
    /// internal or out-of-memory error, caught in an error scope).
    Device(String),
}

impl fmt::Display for NrdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingFeatures(features) => {
                write!(f, "the device lacks the wgpu features {features:?}")
            }
            Self::CreationFailed => write!(f, "NRD could not create the denoiser"),
            Self::UnsupportedLibrary(what) => write!(f, "unsupported NRD build: {what}"),
            Self::UnsupportedFormat => write!(f, "NRD asks for a texture format wgpu lacks"),
            Self::UnsupportedTextureFormat(format) => {
                write!(f, "a texture of format {format:?} cannot be bound")
            }
            Self::InvalidSettings(what) => write!(f, "invalid settings: {what}"),
            Self::SizeMismatch { expected, found } => write!(
                f,
                "a texture is {}x{}, the denoiser was created for {}x{}",
                found[0], found[1], expected[0], expected[1]
            ),
            Self::Device(error) => write!(f, "the device refused NRD's {error}"),
            Self::UnsupportedResource(dispatch) => {
                write!(
                    f,
                    "NRD's dispatch \"{dispatch}\" uses a resource nrd_wgpu does not provide"
                )
            }
        }
    }
}

impl core::error::Error for NrdError {}

/// One binding of a pipeline's texture set.
#[derive(Clone, Copy)]
enum Slot {
    Uniform,
    Texture(TextureFormat),
    Storage(TextureFormat),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum PipelineId {
    Nrd(u32),
    Pack,
    Unpack,
}

struct NrdPipeline {
    module: ShaderModule,
    name: String,
    texture_count: u32,
    storage_count: u32,
}

/// A compute pipeline together with the layout its texture set was created from.
#[derive(Clone)]
struct CachedPipeline {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

/// One recorded dispatch, ready to be issued.
struct Step {
    /// NRD's name for the dispatch, or this crate's for the pack and unpack passes.
    name: &'static CStr,
    pipeline: ComputePipeline,
    resources: BindGroup,
    /// Byte offset of this dispatch's constants in the constant buffer; `None` for the pack and unpack passes.
    constants_offset: Option<u32>,
    workgroups: [u32; 2],
}

struct Previous {
    view_to_clip: [f32; 16],
    world_to_view: [f32; 16],
    jitter: [f32; 2],
}

/// The formats of the textures a frame's [`DenoiserInputs`] bind. With the precision they fix every pipeline a frame
/// uses, so [`DenoiserPipelines::warm_up`] can create those pipelines before the first frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputFormats {
    /// The format of [`DenoiserInputs::diffuse`].
    pub diffuse: TextureFormat,
    /// The format of [`DenoiserInputs::specular`].
    pub specular: TextureFormat,
    /// The format of [`DenoiserInputs::normal_roughness`].
    pub normal_roughness: TextureFormat,
    /// The format of [`DenoiserInputs::view_depth`].
    pub view_depth: TextureFormat,
    /// The format of [`DenoiserInputs::motion_vectors`].
    pub motion_vectors: TextureFormat,
    /// The format of [`DenoiserInputs::history_confidence`], `None` when the frames give none.
    pub history_confidence: Option<TextureFormat>,
}

impl InputFormats {
    /// The formats of `inputs`' textures.
    pub fn of(inputs: &DenoiserInputs) -> Self {
        let format = |view: &TextureView| view.texture().format();
        Self {
            diffuse: format(inputs.diffuse),
            specular: format(inputs.specular),
            normal_roughness: format(inputs.normal_roughness),
            view_depth: format(inputs.view_depth),
            motion_vectors: format(inputs.motion_vectors),
            history_confidence: inputs.history_confidence.map(format),
        }
    }
}

/// What NRD's library says about its shaders' bindings. The same for every instance of one library.
struct LibraryLayout {
    entry_point: String,
    resources_set: u32,
    constants_set: u32,
    first_texture_binding: u32,
    first_storage_binding: u32,
    first_sampler_binding: u32,
    constants_binding: u32,
    constants_size: u32,
    permanent_pool_size: u32,
    transient_pool_size: u32,
}

/// The part of a denoiser that does not depend on its size: NRD's shader modules, the pack and unpack passes, the
/// bind group layouts, the samplers and the compute pipelines. Create it once per device and share it (it is a
/// cheap [`Clone`]) between every [`Denoiser`] of that device, so that a new denoiser, or one that changes size
/// ([`Denoiser::resize`]), compiles nothing.
///
/// Every module and pipeline is created inside wgpu error scopes: a driver or wgpu refusing one comes back as
/// [`NrdError::Device`] instead of reaching the device's uncaptured-error handler.
#[derive(Clone)]
pub struct DenoiserPipelines(Arc<SharedPipelines>);

struct SharedPipelines {
    kind: DenoiserKind,
    layout: LibraryLayout,
    samplers: Vec<Sampler>,
    constants_layout: BindGroupLayout,
    nrd_pipelines: Vec<NrdPipeline>,
    pack_module: ShaderModule,
    unpack_module: ShaderModule,
    /// One pipeline per shader **and** per set of bound texture formats: wgpu fixes the formats in the layout.
    pipelines: Mutex<HashMap<(PipelineId, Vec<TextureFormat>), CachedPipeline>>,
}

/// The environment variable that makes [`DenoiserPipelines::new`] provoke a wgpu validation error inside its error
/// scope, so that an application's handling of a driver that refuses NRD can be tested on a driver that does not.
pub const FORCE_DEVICE_ERROR_VARIABLE: &str = "NRD_WGPU_FORCE_DEVICE_ERROR";

impl DenoiserPipelines {
    /// Loads NRD's shaders for `kind` and creates the layouts and samplers they need. No pipeline is created
    /// yet: [`warm_up`](Self::warm_up) creates those of a precision and a set of input formats, and anything a
    /// frame needs beyond them is created when the frame is recorded.
    pub fn new(device: &Device, kind: DenoiserKind) -> Result<Self, NrdError> {
        if cfg!(feature = "mock") {
            return Err(NrdError::UnsupportedLibrary(
                "built with the `mock` feature, without NRD",
            ));
        }
        let (guard, info) = create_instance(kind)?;
        let instance = guard.0;
        // SAFETY: NRD returns a NUL-terminated string that lives as long as the library.
        let entry_point = unsafe { CStr::from_ptr(info.entry_point) }
            .to_string_lossy()
            .into_owned();
        let layout = LibraryLayout {
            entry_point,
            resources_set: info.resources_set,
            constants_set: info.constants_and_samplers_set,
            first_texture_binding: info.first_texture_binding,
            first_storage_binding: info.first_storage_binding,
            first_sampler_binding: info.first_sampler_binding,
            constants_binding: info.constants_binding,
            constants_size: info.constants_max_size.next_multiple_of(16),
            permanent_pool_size: info.permanent_pool_size,
            transient_pool_size: info.transient_pool_size,
        };

        let (shared, error) = capture_device_errors(device, || {
            let samplers: Vec<Sampler> = (0..info.sampler_count)
                .map(|index| {
                    // SAFETY: `index` is below the sampler count NRD reported.
                    let linear =
                        unsafe { ffi::nrdw_sampler_is_linear(instance.as_ptr(), index) } != 0;
                    let filter = if linear {
                        FilterMode::Linear
                    } else {
                        FilterMode::Nearest
                    };
                    device.create_sampler(&SamplerDescriptor {
                        label: Some("nrd_sampler"),
                        mag_filter: filter,
                        min_filter: filter,
                        ..Default::default()
                    })
                })
                .collect();

            // The set that holds the samplers and the constant buffer is the same for every NRD pipeline.
            let mut constants_slots = Vec::new();
            for index in 0..info.sampler_count {
                constants_slots.push((
                    info.first_sampler_binding + index,
                    BindingType::Sampler(SamplerBindingType::Filtering),
                ));
            }
            constants_slots.push((
                info.constants_binding,
                BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
            ));
            let constants_layout = dense_layout(device, "nrd_constants_layout", &constants_slots);

            let nrd_pipelines = (0..info.pipeline_count)
                .map(|index| {
                    let mut raw = core::mem::MaybeUninit::<ffi::NrdwPipeline>::uninit();
                    // SAFETY: `index` is below the pipeline count NRD reported, and the shim fills the struct.
                    let raw = unsafe {
                        ffi::nrdw_pipeline(instance.as_ptr(), index, raw.as_mut_ptr());
                        raw.assume_init()
                    };
                    // SAFETY: NRD embeds each shader as `spirv_bytes` bytes of 4-byte SPIR-V words in static
                    // storage, and the name is a NUL-terminated string inside the instance.
                    let (spirv, name) = unsafe {
                        (
                            core::slice::from_raw_parts(raw.spirv, raw.spirv_bytes as usize / 4),
                            CStr::from_ptr(raw.name).to_string_lossy().into_owned(),
                        )
                    };
                    NrdPipeline {
                        module: passthrough_module(device, &name, &layout.entry_point, spirv),
                        name,
                        texture_count: raw.texture_count,
                        storage_count: raw.storage_count,
                    }
                })
                .collect();

            if std::env::var_os(FORCE_DEVICE_ERROR_VARIABLE).is_some() {
                // A buffer that is both mappable for reading and a storage buffer is a validation error.
                let _ = device.create_buffer(&BufferDescriptor {
                    label: Some("nrd_forced_device_error"),
                    size: 4,
                    usage: BufferUsages::MAP_READ | BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
            }

            SharedPipelines {
                kind,
                samplers,
                constants_layout,
                nrd_pipelines,
                pack_module: passthrough_module(
                    device,
                    "nrd_pack",
                    "main",
                    &spirv_words(PACK_SPIRV),
                ),
                unpack_module: passthrough_module(
                    device,
                    "nrd_unpack",
                    "main",
                    &spirv_words(UNPACK_SPIRV),
                ),
                pipelines: Mutex::new(HashMap::new()),
                layout,
            }
        });
        drop(guard);
        match error {
            Some(error) => Err(NrdError::Device(error)),
            None => Ok(Self(Arc::new(shared))),
        }
    }

    /// Creates every pipeline a frame at `precision` with inputs of these formats uses at the default ReBLUR
    /// settings, the first frame of a history and the frames after it. Pipelines that exist already are kept, so
    /// calling it again is cheap.
    pub fn warm_up(
        &self,
        device: &Device,
        precision: TexturePrecision,
        inputs: &InputFormats,
    ) -> Result<(), NrdError> {
        let missing = Denoiser::required_features(precision) - device.features();
        if !missing.is_empty() {
            return Err(NrdError::MissingFeatures(missing));
        }
        let shared = &self.0;
        let (guard, _) = create_instance(shared.kind)?;
        let instance = guard.0;

        let packed = |resource: u32| match resource {
            ffi::NRDW_RESOURCE_MOTION_VECTORS => Some(PACKED_MOTION_VECTORS_FORMAT),
            ffi::NRDW_RESOURCE_NORMAL_ROUGHNESS => Some(PACKED_NORMAL_ROUGHNESS_FORMAT),
            ffi::NRDW_RESOURCE_VIEW_Z => Some(PACKED_VIEW_Z_FORMAT),
            ffi::NRDW_RESOURCE_IN_DIFFUSE | ffi::NRDW_RESOURCE_IN_SPECULAR => {
                Some(PACKED_RADIANCE_FORMAT)
            }
            ffi::NRDW_RESOURCE_OUT_DIFFUSE | ffi::NRDW_RESOURCE_OUT_SPECULAR => {
                Some(precision.apply(PACKED_RADIANCE_FORMAT))
            }
            ffi::NRDW_RESOURCE_IN_DIFFUSE_CONFIDENCE
            | ffi::NRDW_RESOURCE_IN_SPECULAR_CONFIDENCE => inputs.history_confidence,
            _ => None,
        };
        let pool_format = |permanent: bool, index: u32| -> Option<TextureFormat> {
            let mut raw = ffi::NrdwPoolTexture::default();
            // SAFETY: `index` is below the pool size NRD reported for this library.
            unsafe {
                ffi::nrdw_pool_texture(instance.as_ptr(), permanent as u32, index, &mut raw);
            }
            convert_format(raw.format).map(|format| precision.apply(format))
        };

        // SAFETY: the shim copies the settings before returning.
        if unsafe {
            ffi::nrdw_set_reblur_settings(instance.as_ptr(), &ReblurSettings::default().to_raw())
        } == 0
        {
            return Err(NrdError::InvalidSettings("NRD refused the ReBLUR settings"));
        }
        let size = 256;
        for (frame_index, reset) in [(0, 2), (1, 0)] {
            let frame = ffi::NrdwFrame {
                view_to_clip: WARM_UP_PROJECTION,
                view_to_clip_previous: WARM_UP_PROJECTION,
                world_to_view: IDENTITY,
                world_to_view_previous: IDENTITY,
                motion_vector_scale: [-1.0, -1.0],
                jitter: [0.0; 2],
                jitter_previous: [0.0; 2],
                width: size,
                height: size,
                denoising_range: 1000.0,
                frame_index,
                reset,
                history_confidence: inputs.history_confidence.is_some() as u32,
            };
            // SAFETY: the instance is alive and the shim copies the frame before returning.
            if unsafe { ffi::nrdw_set_frame(instance.as_ptr(), &frame) } == 0 {
                return Err(NrdError::InvalidSettings("NRD refused the frame settings"));
            }
            // SAFETY: the dispatch list stays valid until the next call into the instance.
            let dispatch_count = unsafe { ffi::nrdw_begin_dispatches(instance.as_ptr()) };
            let dispatch_count = u32::try_from(dispatch_count)
                .map_err(|_| NrdError::InvalidSettings("NRD returned no dispatches"))?;
            for index in 0..dispatch_count {
                let mut raw = core::mem::MaybeUninit::<ffi::NrdwDispatch>::uninit();
                // SAFETY: `index` is below the dispatch count, and the shim fills the struct.
                let raw = unsafe {
                    ffi::nrdw_dispatch(instance.as_ptr(), index, raw.as_mut_ptr());
                    raw.assume_init()
                };
                let texture_count = shared.nrd_pipelines[raw.pipeline as usize].texture_count;
                let mut slots = Vec::with_capacity(raw.resource_count as usize);
                for resource_index in 0..raw.resource_count {
                    let mut resource = ffi::NrdwResource::default();
                    // SAFETY: both indices are in range for the current dispatch list.
                    unsafe {
                        ffi::nrdw_dispatch_resource(
                            instance.as_ptr(),
                            index,
                            resource_index,
                            &mut resource,
                        );
                    };
                    let format = match resource.kind {
                        ffi::NRDW_RESOURCE_TRANSIENT_POOL => {
                            pool_format(false, resource.pool_index)
                        }
                        ffi::NRDW_RESOURCE_PERMANENT_POOL => pool_format(true, resource.pool_index),
                        kind => packed(kind),
                    };
                    let Some(format) = format else {
                        // `denoise` reports what this dispatch lacks when a frame asks for it.
                        slots.clear();
                        break;
                    };
                    slots.push(shared.nrd_slot(resource_index, texture_count, format));
                }
                if !slots.is_empty() {
                    self.pipeline(device, PipelineId::Nrd(raw.pipeline), &slots, false)?;
                }
            }
        }
        drop(guard);

        let own_slots = |formats: &[(TextureFormat, bool)]| -> Vec<(u32, TextureFormat, bool)> {
            formats
                .iter()
                .enumerate()
                .map(|(index, (format, is_storage))| (index as u32 + 1, *format, *is_storage))
                .collect()
        };
        self.pipeline(
            device,
            PipelineId::Pack,
            &own_slots(&pack_formats(inputs)),
            true,
        )?;
        self.pipeline(
            device,
            PipelineId::Unpack,
            &own_slots(&unpack_formats(precision)),
            true,
        )?;
        Ok(())
    }

    /// How many compute pipelines exist, for every precision and set of input formats seen so far.
    pub fn pipeline_count(&self) -> usize {
        self.0.pipelines.lock().unwrap().len()
    }

    /// The pipeline of shader `id` for the texture formats of `slots`, created on first use inside wgpu error
    /// scopes.
    fn pipeline(
        &self,
        device: &Device,
        id: PipelineId,
        slots: &[(u32, TextureFormat, bool)],
        own: bool,
    ) -> Result<CachedPipeline, NrdError> {
        let shared = &self.0;
        let formats: Vec<TextureFormat> = slots.iter().map(|(_, format, _)| *format).collect();
        if let Some(cached) = shared.pipelines.lock().unwrap().get(&(id, formats.clone())) {
            return Ok(cached.clone());
        }

        let mut layout_slots = Vec::new();
        if own {
            layout_slots.push((0, Slot::Uniform));
        }
        for (binding, format, is_storage) in slots {
            let slot = if *is_storage {
                Slot::Storage(*format)
            } else {
                Slot::Texture(*format)
            };
            layout_slots.push((*binding, slot));
        }
        let mut entries = Vec::new();
        for (binding, slot) in layout_slots {
            let ty = match slot {
                Slot::Uniform => BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                Slot::Texture(format) => BindingType::Texture {
                    sample_type: format
                        .sample_type(None, Some(device.features()))
                        .ok_or(NrdError::UnsupportedTextureFormat(format))?,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                // Declared write-only whatever the shader does: Vulkan has one kind of storage image, and a
                // read-write declaration would need a format capability wgpu grants to few formats.
                Slot::Storage(format) => BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format,
                    view_dimension: TextureViewDimension::D2,
                },
            };
            entries.push((binding, ty));
        }

        let (module, entry_point, label) = match id {
            PipelineId::Nrd(index) => {
                let pipeline = &shared.nrd_pipelines[index as usize];
                (
                    &pipeline.module,
                    shared.layout.entry_point.as_str(),
                    pipeline.name.as_str(),
                )
            }
            PipelineId::Pack => (&shared.pack_module, "main", "nrd_pack"),
            PipelineId::Unpack => (&shared.unpack_module, "main", "nrd_unpack"),
        };
        let (cached, error) = capture_device_errors(device, || {
            let layout = dense_layout(device, "nrd_resources_layout", &entries);
            let mut sets = [None, None];
            if own {
                sets[0] = Some(&layout);
            } else {
                sets[shared.layout.resources_set as usize] = Some(&layout);
                sets[shared.layout.constants_set as usize] = Some(&shared.constants_layout);
            }
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: if own { &sets[..1] } else { &sets },
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
            CachedPipeline { pipeline, layout }
        });
        if let Some(error) = error {
            return Err(NrdError::Device(format!("pipeline {label}: {error}")));
        }
        shared
            .pipelines
            .lock()
            .unwrap()
            .insert((id, formats), cached.clone());
        Ok(cached)
    }
}

impl SharedPipelines {
    /// The binding of the `position`-th resource of an NRD dispatch: the textures it reads come first, then the
    /// ones it writes.
    fn nrd_slot(
        &self,
        position: u32,
        texture_count: u32,
        format: TextureFormat,
    ) -> (u32, TextureFormat, bool) {
        let is_storage = position >= texture_count;
        let binding = if is_storage {
            self.layout.first_storage_binding + position - texture_count
        } else {
            self.layout.first_texture_binding + position
        };
        (binding, format, is_storage)
    }
}

/// The formats of the textures this crate creates around NRD.
const PACKED_RADIANCE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
const PACKED_NORMAL_ROUGHNESS_FORMAT: TextureFormat = TextureFormat::Rgb10a2Unorm;
const PACKED_VIEW_Z_FORMAT: TextureFormat = TextureFormat::R32Float;
const PACKED_MOTION_VECTORS_FORMAT: TextureFormat = TextureFormat::Rg16Float;

/// The pack pass's bindings after its uniform block: the five inputs, then the five packed textures.
fn pack_formats(inputs: &InputFormats) -> [(TextureFormat, bool); 10] {
    [
        (inputs.diffuse, false),
        (inputs.specular, false),
        (inputs.normal_roughness, false),
        (inputs.view_depth, false),
        (inputs.motion_vectors, false),
        (PACKED_RADIANCE_FORMAT, true),
        (PACKED_RADIANCE_FORMAT, true),
        (PACKED_NORMAL_ROUGHNESS_FORMAT, true),
        (PACKED_VIEW_Z_FORMAT, true),
        (PACKED_MOTION_VECTORS_FORMAT, true),
    ]
}

/// The unpack pass's bindings after its uniform block: NRD's two outputs and the view depth, then the two outputs.
fn unpack_formats(precision: TexturePrecision) -> [(TextureFormat, bool); 5] {
    [
        (precision.apply(PACKED_RADIANCE_FORMAT), false),
        (precision.apply(PACKED_RADIANCE_FORMAT), false),
        (PACKED_VIEW_Z_FORMAT, false),
        (OUTPUT_FORMAT, true),
        (OUTPUT_FORMAT, true),
    ]
}

const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];
/// A reverse-Z infinite perspective (Bevy's), 90° vertical field of view, near plane 0.1: only to list a frame's
/// dispatches, which do not depend on it.
const WARM_UP_PROJECTION: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.1, 0.0,
];

/// The textures of a [`Denoiser`], all at its size.
struct SizedTextures {
    permanent_pool: Vec<TextureView>,
    transient_pool: Vec<TextureView>,
    packed_diffuse: TextureView,
    packed_specular: TextureView,
    packed_normal_roughness: TextureView,
    packed_view_z: TextureView,
    packed_motion_vectors: TextureView,
    denoised_diffuse: TextureView,
    denoised_specular: TextureView,
}

impl SizedTextures {
    fn new(
        device: &Device,
        instance: NonNull<ffi::NrdwInstance>,
        layout: &LibraryLayout,
        size: [u32; 2],
        precision: TexturePrecision,
    ) -> Result<Self, NrdError> {
        let pool = |permanent: bool, count: u32| -> Result<Vec<TextureView>, NrdError> {
            (0..count)
                .map(|index| {
                    let mut raw = ffi::NrdwPoolTexture::default();
                    // SAFETY: `index` is below the pool size NRD reported.
                    unsafe {
                        ffi::nrdw_pool_texture(
                            instance.as_ptr(),
                            permanent as u32,
                            index,
                            &mut raw,
                        );
                    };
                    let format = convert_format(raw.format).ok_or(NrdError::UnsupportedFormat)?;
                    let format = precision.apply(format);
                    let factor = raw.downsample_factor.max(1);
                    let label = if permanent {
                        "nrd_permanent"
                    } else {
                        "nrd_transient"
                    };
                    Ok(create_texture(
                        device,
                        label,
                        [size[0].div_ceil(factor), size[1].div_ceil(factor)],
                        format,
                    ))
                })
                .collect()
        };
        let (textures, error) = capture_device_errors(device, || -> Result<Self, NrdError> {
            Ok(Self {
                permanent_pool: pool(true, layout.permanent_pool_size)?,
                transient_pool: pool(false, layout.transient_pool_size)?,
                packed_diffuse: create_texture(
                    device,
                    "nrd_packed_diffuse",
                    size,
                    PACKED_RADIANCE_FORMAT,
                ),
                packed_specular: create_texture(
                    device,
                    "nrd_packed_specular",
                    size,
                    PACKED_RADIANCE_FORMAT,
                ),
                packed_normal_roughness: create_texture(
                    device,
                    "nrd_packed_normal_roughness",
                    size,
                    PACKED_NORMAL_ROUGHNESS_FORMAT,
                ),
                packed_view_z: create_texture(
                    device,
                    "nrd_packed_view_z",
                    size,
                    PACKED_VIEW_Z_FORMAT,
                ),
                packed_motion_vectors: create_texture(
                    device,
                    "nrd_packed_motion_vectors",
                    size,
                    PACKED_MOTION_VECTORS_FORMAT,
                ),
                // NRD also uses its two outputs as scratch space inside a frame, so they follow the precision too.
                denoised_diffuse: create_texture(
                    device,
                    "nrd_denoised_diffuse",
                    size,
                    precision.apply(PACKED_RADIANCE_FORMAT),
                ),
                denoised_specular: create_texture(
                    device,
                    "nrd_denoised_specular",
                    size,
                    precision.apply(PACKED_RADIANCE_FORMAT),
                ),
            })
        });
        match error {
            Some(error) => Err(NrdError::Device(format!("textures: {error}"))),
            None => textures,
        }
    }

    fn iter(&self) -> impl Iterator<Item = &TextureView> {
        self.permanent_pool
            .iter()
            .chain(&self.transient_pool)
            .chain([
                &self.packed_diffuse,
                &self.packed_specular,
                &self.packed_normal_roughness,
                &self.packed_view_z,
                &self.packed_motion_vectors,
                &self.denoised_diffuse,
                &self.denoised_specular,
            ])
    }
}

/// An NRD denoiser for one view: its NRD instance, which keeps the history's bookkeeping, and its textures, which
/// are at one size. [`resize`](Self::resize) changes the size without compiling anything.
pub struct Denoiser {
    instance: NonNull<ffi::NrdwInstance>,
    size: [u32; 2],
    precision: TexturePrecision,
    settings: ReblurSettings,
    previous: Option<Previous>,
    pipelines: DenoiserPipelines,
    /// NRD's shaders a frame of this denoiser has dispatched.
    used: HashSet<u32>,

    constants_stride: u32,
    constants_capacity: u32,
    constants: Option<(Buffer, BindGroup)>,
    params: Buffer,
    textures: SizedTextures,
    /// Bound at the binding numbers a shader leaves unused, see [`dense_layout`].
    filler: TextureView,
    timestamps: Option<Timestamps>,
}

/// The queries and buffers of [`Denoiser::set_timestamps`].
struct Timestamps {
    queries: QuerySet,
    /// Queries in `queries`: one more than the dispatches of the longest frame recorded so far.
    capacity: u32,
    resolved: Buffer,
    readback: Buffer,
    /// The dispatches of the last recorded frame, in order.
    names: Vec<&'static CStr>,
}

impl Timestamps {
    fn new(device: &Device, capacity: u32) -> Self {
        let buffer = |label, usage| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: capacity as u64 * 8,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            queries: device.create_query_set(&QuerySetDescriptor {
                label: Some("nrd_timestamps"),
                ty: QueryType::Timestamp,
                count: capacity,
            }),
            capacity,
            resolved: buffer(
                "nrd_timestamps_resolved",
                BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            ),
            readback: buffer(
                "nrd_timestamps_readback",
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            ),
            names: Vec::new(),
        }
    }
}

/// The GPU time of one dispatch of a frame, from [`Denoiser::read_timestamps`].
#[derive(Clone, Debug)]
pub struct DispatchTime {
    /// NRD's name for the dispatch, or `nrd_wgpu pack` / `nrd_wgpu unpack` for this crate's two passes.
    pub name: String,
    /// From the end of the previous dispatch (for the first one, the start of its pass) to the end of this one,
    /// so the barrier in front of the dispatch and its ramp-up are counted with it.
    pub milliseconds: f64,
}

// SAFETY: the NRD instance behind the pointer has no thread affinity, and every call that touches it takes
// `&mut self` (or happens in `drop`), so it is never used from two threads at once.
unsafe impl Send for Denoiser {}
// SAFETY: `&Denoiser` gives access to nothing that reaches the NRD instance.
unsafe impl Sync for Denoiser {}

impl Drop for Denoiser {
    fn drop(&mut self) {
        // SAFETY: the instance came from `nrdw_create` and is destroyed exactly once.
        unsafe { ffi::nrdw_destroy(self.instance.as_ptr()) };
    }
}

impl Denoiser {
    /// The wgpu features the device must have been created with.
    ///
    /// - `PASSTHROUGH_SHADERS`: NRD ships SPIR-V, which is handed to the driver as it is.
    /// - `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`: NRD writes to texture formats (single and dual channel,
    ///   10:10:10:2) that base WebGPU does not allow as storage textures.
    /// - `FLOAT32_FILTERABLE`, for [`TexturePrecision::Full`]: NRD samples its history with a linear filter.
    ///
    /// NRD's SPIR-V also declares its storage images without a format, which in Vulkan needs the device features
    /// `shaderStorageImageWriteWithoutFormat` and `shaderStorageImageReadWithoutFormat`. wgpu has no way to ask
    /// for them and never enables them. The shaders ran correctly without them on the two drivers tried (see the
    /// README); a driver is free not to. An application can at least check that the device supports them, from
    /// the raw Vulkan adapter.
    pub fn required_features(precision: TexturePrecision) -> Features {
        let features =
            Features::PASSTHROUGH_SHADERS | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        match precision {
            TexturePrecision::Half => features,
            TexturePrecision::Full => features | Features::FLOAT32_FILTERABLE,
        }
    }

    /// Creates a denoiser for textures of `size` pixels, with pipelines of its own that the first frames create.
    /// A denoiser that has [`DenoiserPipelines`] to share is made with [`with_pipelines`](Self::with_pipelines).
    pub fn new(
        device: &Device,
        kind: DenoiserKind,
        size: [u32; 2],
        precision: TexturePrecision,
    ) -> Result<Self, NrdError> {
        Self::with_pipelines(
            device,
            &DenoiserPipelines::new(device, kind)?,
            size,
            precision,
        )
    }

    /// Creates a denoiser for textures of `size` pixels that uses, and adds to, `pipelines`.
    pub fn with_pipelines(
        device: &Device,
        pipelines: &DenoiserPipelines,
        size: [u32; 2],
        precision: TexturePrecision,
    ) -> Result<Self, NrdError> {
        check_size_and_features(device, size, precision)?;
        let (guard, _) = create_instance(pipelines.0.kind)?;
        let layout = &pipelines.0.layout;
        let textures = SizedTextures::new(device, guard.0, layout, size, precision)?;
        let params = device.create_buffer(&BufferDescriptor {
            label: Some("nrd_pack_params"),
            size: PARAMS_SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut denoiser = Self {
            instance: guard.release(),
            size,
            precision,
            settings: ReblurSettings::default(),
            previous: None,
            pipelines: pipelines.clone(),
            used: HashSet::new(),
            constants_stride: layout
                .constants_size
                .next_multiple_of(device.limits().min_uniform_buffer_offset_alignment),
            constants_capacity: 0,
            constants: None,
            params,
            textures,
            filler: create_texture(device, "nrd_filler", [1, 1], TextureFormat::Rgba8Unorm),
            timestamps: None,
        };
        let settings = denoiser.settings;
        denoiser.set_reblur_settings(&settings)?;
        Ok(denoiser)
    }

    /// Changes the size or the precision of the denoiser's textures. Only the textures are made again: the NRD
    /// instance, its settings and the pipelines stay. The history is dropped, so the next frame starts a new one.
    /// On an error the denoiser keeps its old textures and size.
    pub fn resize(
        &mut self,
        device: &Device,
        size: [u32; 2],
        precision: TexturePrecision,
    ) -> Result<(), NrdError> {
        check_size_and_features(device, size, precision)?;
        // The old textures go first, so the two sets are never in memory together.
        let textures = SizedTextures::new(
            device,
            self.instance,
            &self.pipelines.0.layout,
            size,
            precision,
        )?;
        self.textures = textures;
        self.size = size;
        self.precision = precision;
        self.previous = None;
        Ok(())
    }

    /// The size, in pixels, of every texture this denoiser reads and writes.
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// The precision of the textures this denoiser accumulates in.
    pub fn precision(&self) -> TexturePrecision {
        self.precision
    }

    /// The pipelines this denoiser uses.
    pub fn pipelines(&self) -> &DenoiserPipelines {
        &self.pipelines
    }

    /// Bytes of GPU memory in the textures this denoiser owns: NRD's two pools and the packed and denoised
    /// copies around them.
    pub fn texture_bytes(&self) -> u64 {
        self.textures
            .iter()
            .map(|view| {
                let texture = view.texture();
                let texel = texture.format().block_copy_size(None).unwrap_or(0);
                texture.width() as u64 * texture.height() as u64 * texel as u64
            })
            .sum()
    }

    /// The version of the NRD library this crate was built against, as `[major, minor, build]`.
    pub fn library_version() -> [u32; 3] {
        // The version is a property of the library, but NRD only reports it through an instance.
        match create_instance(DenoiserKind::ReblurDiffuseSpecular) {
            Ok((_guard, info)) => [info.version_major, info.version_minor, info.version_build],
            Err(_) => [0; 3],
        }
    }

    /// NRD's shaders for this denoiser, as `file|macro=value|...`, each with whether a frame of this denoiser has
    /// dispatched it. Which ones a frame uses depends on the settings; some are debug views this crate never asks
    /// for.
    pub fn shaders(&self) -> impl Iterator<Item = (&str, bool)> {
        self.pipelines
            .0
            .nrd_pipelines
            .iter()
            .enumerate()
            .map(|(index, pipeline)| (pipeline.name.as_str(), self.used.contains(&(index as u32))))
    }

    /// The ReBLUR settings in use.
    pub fn reblur_settings(&self) -> &ReblurSettings {
        &self.settings
    }

    /// Changes the ReBLUR settings. They apply from the next [`denoise`](Self::denoise).
    pub fn set_reblur_settings(&mut self, settings: &ReblurSettings) -> Result<(), NrdError> {
        // NRD checks its settings with assertions that a release build compiles out, so the ranges that would
        // otherwise corrupt the history silently are checked here.
        if settings.max_accumulated_frames > 63 {
            return Err(NrdError::InvalidSettings(
                "max_accumulated_frames is at most 63",
            ));
        }
        if !(1.0..=3.0).contains(&settings.fast_history_clamping_sigma_scale) {
            return Err(NrdError::InvalidSettings(
                "fast_history_clamping_sigma_scale must be within 1..=3",
            ));
        }
        if settings
            .hit_distance_parameters
            .iter()
            .chain([
                &settings.antilag.luminance_sigma_scale,
                &settings.antilag.luminance_sensitivity,
            ])
            .any(|p| p.is_nan() || *p <= 0.0)
        {
            return Err(NrdError::InvalidSettings(
                "hit_distance_parameters and the anti-lag settings must be positive",
            ));
        }
        let raw = settings.to_raw();
        // SAFETY: the instance is alive and the shim copies the settings before returning.
        if unsafe { ffi::nrdw_set_reblur_settings(self.instance.as_ptr(), &raw) } == 0 {
            return Err(NrdError::InvalidSettings("NRD refused the ReBLUR settings"));
        }
        self.settings = *settings;
        Ok(())
    }

    /// Records one frame of denoising into `encoder`: the pack pass, every dispatch NRD asks for, and the unpack
    /// pass. Nothing runs until the encoder is submitted, and the frame's constants are uploaded through `queue`,
    /// so the encoder must be submitted before `denoise` is called again.
    ///
    /// A pipeline the frame needs that [`DenoiserPipelines::warm_up`] did not create is created here.
    pub fn denoise(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        frame: &FrameSettings,
        inputs: &DenoiserInputs,
        outputs: &DenoiserOutputs,
    ) -> Result<(), NrdError> {
        for view in [
            inputs.diffuse,
            inputs.specular,
            inputs.normal_roughness,
            inputs.view_depth,
            inputs.motion_vectors,
            outputs.diffuse,
            outputs.specular,
        ] {
            let found = [view.texture().width(), view.texture().height()];
            if found != self.size {
                return Err(NrdError::SizeMismatch {
                    expected: self.size,
                    found,
                });
            }
        }
        for view in [outputs.diffuse, outputs.specular] {
            if view.texture().format() != OUTPUT_FORMAT {
                return Err(NrdError::UnsupportedTextureFormat(view.texture().format()));
            }
        }
        if frame.jitter.iter().any(|j| !(-0.5..=0.5).contains(j)) {
            return Err(NrdError::InvalidSettings(
                "jitter must be within -0.5..=0.5",
            ));
        }
        if frame.motion_vector_scale.contains(&0.0) {
            return Err(NrdError::InvalidSettings("motion_vector_scale cannot be 0"));
        }
        if frame.denoising_range.is_nan() || frame.denoising_range <= 0.0 {
            return Err(NrdError::InvalidSettings(
                "denoising_range must be positive",
            ));
        }

        let previous = self.previous.as_ref();
        let raw_frame = ffi::NrdwFrame {
            view_to_clip: frame.view_to_clip,
            view_to_clip_previous: previous.map_or(frame.view_to_clip, |p| p.view_to_clip),
            world_to_view: frame.world_to_view,
            world_to_view_previous: previous.map_or(frame.world_to_view, |p| p.world_to_view),
            motion_vector_scale: frame.motion_vector_scale,
            jitter: frame.jitter,
            jitter_previous: previous.map_or(frame.jitter, |p| p.jitter),
            width: self.size[0],
            height: self.size[1],
            denoising_range: frame.denoising_range,
            frame_index: frame.frame_index,
            history_confidence: inputs.history_confidence.is_some() as u32,
            // The first frame also clears NRD's textures, which hold whatever the driver left in them.
            reset: match (previous.is_some(), frame.reset_history) {
                (false, _) => 2,
                (true, true) => 1,
                (true, false) => 0,
            },
        };
        // SAFETY: the instance is alive and the shim copies the frame before returning.
        if unsafe { ffi::nrdw_set_frame(self.instance.as_ptr(), &raw_frame) } == 0 {
            return Err(NrdError::InvalidSettings("NRD refused the frame settings"));
        }
        self.previous = Some(Previous {
            view_to_clip: frame.view_to_clip,
            world_to_view: frame.world_to_view,
            jitter: frame.jitter,
        });

        let mut params = [0u8; PARAMS_SIZE as usize];
        for (index, value) in self.settings.hit_distance_parameters.iter().enumerate() {
            params[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        params[12..16].copy_from_slice(&frame.denoising_range.to_le_bytes());
        params[16..20].copy_from_slice(&self.size[0].to_le_bytes());
        params[20..24].copy_from_slice(&self.size[1].to_le_bytes());
        queue.write_buffer(&self.params, 0, &params);

        let pass_workgroups = [self.size[0].div_ceil(8), self.size[1].div_ceil(8)];
        let mut steps = Vec::new();

        let textures = &self.textures;
        steps.push(self.own_step(
            device,
            PipelineId::Pack,
            &[
                (inputs.diffuse, false),
                (inputs.specular, false),
                (inputs.normal_roughness, false),
                (inputs.view_depth, false),
                (inputs.motion_vectors, false),
                (&textures.packed_diffuse, true),
                (&textures.packed_specular, true),
                (&textures.packed_normal_roughness, true),
                (&textures.packed_view_z, true),
                (&textures.packed_motion_vectors, true),
            ],
            pass_workgroups,
        )?);

        // SAFETY: the instance is alive. The dispatch list it returns stays valid until the next call, and no
        // other call into NRD is made before the end of this function.
        let dispatch_count = unsafe { ffi::nrdw_begin_dispatches(self.instance.as_ptr()) };
        let dispatch_count = u32::try_from(dispatch_count)
            .map_err(|_| NrdError::InvalidSettings("NRD returned no dispatches"))?;
        self.reserve_constants(device, dispatch_count);

        let shared = &self.pipelines.0;
        let textures = &self.textures;
        let mut constants_slot = 0;
        for index in 0..dispatch_count {
            let mut raw = core::mem::MaybeUninit::<ffi::NrdwDispatch>::uninit();
            // SAFETY: `index` is below the dispatch count, and the shim fills the struct.
            let raw = unsafe {
                ffi::nrdw_dispatch(self.instance.as_ptr(), index, raw.as_mut_ptr());
                raw.assume_init()
            };
            let pipeline = &shared.nrd_pipelines[raw.pipeline as usize];
            let (texture_count, storage_count) = (pipeline.texture_count, pipeline.storage_count);

            // A dispatch lists the textures it reads first, then the ones it writes.
            let mut views = Vec::with_capacity(raw.resource_count as usize);
            for resource_index in 0..raw.resource_count {
                let mut resource = ffi::NrdwResource::default();
                // SAFETY: both indices are in range for the current dispatch list.
                unsafe {
                    ffi::nrdw_dispatch_resource(
                        self.instance.as_ptr(),
                        index,
                        resource_index,
                        &mut resource,
                    );
                };
                let pool_index = resource.pool_index as usize;
                let view = match resource.kind {
                    ffi::NRDW_RESOURCE_MOTION_VECTORS => Some(&textures.packed_motion_vectors),
                    ffi::NRDW_RESOURCE_NORMAL_ROUGHNESS => Some(&textures.packed_normal_roughness),
                    ffi::NRDW_RESOURCE_VIEW_Z => Some(&textures.packed_view_z),
                    ffi::NRDW_RESOURCE_IN_DIFFUSE => Some(&textures.packed_diffuse),
                    ffi::NRDW_RESOURCE_IN_SPECULAR => Some(&textures.packed_specular),
                    ffi::NRDW_RESOURCE_OUT_DIFFUSE => Some(&textures.denoised_diffuse),
                    ffi::NRDW_RESOURCE_OUT_SPECULAR => Some(&textures.denoised_specular),
                    ffi::NRDW_RESOURCE_TRANSIENT_POOL => textures.transient_pool.get(pool_index),
                    ffi::NRDW_RESOURCE_PERMANENT_POOL => textures.permanent_pool.get(pool_index),
                    ffi::NRDW_RESOURCE_IN_DIFFUSE_CONFIDENCE
                    | ffi::NRDW_RESOURCE_IN_SPECULAR_CONFIDENCE => inputs.history_confidence,
                    _ => None,
                };
                let is_storage = resource_index >= texture_count;
                match view {
                    Some(view) if is_storage == (resource.is_storage != 0) => {
                        views.push(view.clone());
                    }
                    _ => {
                        // SAFETY: NRD names every dispatch with a NUL-terminated static string.
                        let name = unsafe { CStr::from_ptr(raw.name) };
                        return Err(NrdError::UnsupportedResource(
                            name.to_string_lossy().into_owned(),
                        ));
                    }
                }
            }
            if views.len() as u32 != texture_count + storage_count {
                return Err(NrdError::UnsupportedLibrary(
                    "a dispatch does not fill its pipeline",
                ));
            }

            let constants_offset = constants_slot * self.constants_stride;
            if raw.constants_size > 0 {
                // SAFETY: NRD keeps `constants_size` bytes of constants for this dispatch until the next call.
                let data = unsafe {
                    core::slice::from_raw_parts(raw.constants, raw.constants_size as usize)
                };
                let mut padded = vec![0u8; data.len().next_multiple_of(4)];
                padded[..data.len()].copy_from_slice(data);
                let (buffer, _) = self.constants.as_ref().unwrap();
                queue.write_buffer(buffer, constants_offset as u64, &padded);
                constants_slot += 1;
            }

            let slots: Vec<(u32, TextureFormat, bool)> = views
                .iter()
                .enumerate()
                .map(|(position, view)| {
                    shared.nrd_slot(position as u32, texture_count, view.texture().format())
                })
                .collect();
            let cached =
                self.pipelines
                    .pipeline(device, PipelineId::Nrd(raw.pipeline), &slots, false)?;
            let bound: Vec<(u32, &TextureView)> = slots
                .iter()
                .zip(&views)
                .map(|((binding, ..), view)| (*binding, view))
                .collect();
            steps.push(Step {
                // SAFETY: NRD names every dispatch with a NUL-terminated static string.
                name: unsafe { CStr::from_ptr(raw.name) },
                resources: self.resources_bind_group(device, &cached.layout, &bound, false),
                pipeline: cached.pipeline,
                constants_offset: Some(constants_offset),
                workgroups: [raw.grid_width, raw.grid_height],
            });
            self.used.insert(raw.pipeline);
        }

        let textures = &self.textures;
        steps.push(self.own_step(
            device,
            PipelineId::Unpack,
            &[
                (&textures.denoised_diffuse, false),
                (&textures.denoised_specular, false),
                (&textures.packed_view_z, false),
                (outputs.diffuse, true),
                (outputs.specular, true),
            ],
            pass_workgroups,
        )?);

        if let Some(timestamps) = &self.timestamps
            && timestamps.capacity < steps.len() as u32 + 1
        {
            self.timestamps = Some(Timestamps::new(device, steps.len() as u32 + 1));
        }
        // One compute pass per dispatch. On the RTX 5070 (NVIDIA 620.73, 2026-11-24, 2560x1440, Full, the GPU
        // shared with a desktop) a frame recorded as one pass took 0.9 to 3.2 ms longer, back to back, than the
        // same dispatches in passes of their own, in 12 of 12 paired rounds; wgpu inserts the same barriers
        // either way. See `vendor/nrd_wgpu/README.md`.
        let queries = self
            .timestamps
            .as_ref()
            .map(|timestamps| &timestamps.queries);
        for (index, step) in steps.iter().enumerate() {
            let index = index as u32;
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("nrd"),
                timestamp_writes: queries.map(|query_set| ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: (index == 0).then_some(0),
                    end_of_pass_write_index: Some(index + 1),
                }),
            });
            pass.set_pipeline(&step.pipeline);
            match step.constants_offset {
                Some(offset) => {
                    let (_, constants) = self.constants.as_ref().unwrap();
                    pass.set_bind_group(shared.layout.resources_set, &step.resources, &[]);
                    pass.set_bind_group(shared.layout.constants_set, constants, &[offset]);
                }
                None => pass.set_bind_group(0, &step.resources, &[]),
            }
            pass.dispatch_workgroups(step.workgroups[0], step.workgroups[1], 1);
        }
        if let Some(timestamps) = &mut self.timestamps {
            let count = steps.len() as u32 + 1;
            encoder.resolve_query_set(&timestamps.queries, 0..count, &timestamps.resolved, 0);
            encoder.copy_buffer_to_buffer(
                &timestamps.resolved,
                0,
                &timestamps.readback,
                0,
                count as u64 * 8,
            );
            timestamps.names = steps.iter().map(|step| step.name).collect();
        }
        Ok(())
    }

    /// Starts or stops timing every dispatch of [`denoise`](Self::denoise) on the GPU, for profiling. Needs
    /// `Features::TIMESTAMP_QUERY`. Each dispatch is in a compute pass of its own whether timing is on or not;
    /// the timestamps are written at the passes' ends.
    pub fn set_timestamps(&mut self, device: &Device, enabled: bool) -> Result<(), NrdError> {
        if !enabled {
            self.timestamps = None;
            return Ok(());
        }
        let missing = Features::TIMESTAMP_QUERY - device.features();
        if !missing.is_empty() {
            return Err(NrdError::MissingFeatures(missing));
        }
        if self.timestamps.is_none() {
            self.timestamps = Some(Timestamps::new(device, 64));
        }
        Ok(())
    }

    /// The time each dispatch of the last recorded frame took on the GPU. Call it after the encoder given to
    /// [`denoise`](Self::denoise) was submitted, and before `denoise` is called again: it waits for the GPU to
    /// finish, so it is for profiling, not for every frame of a game. `None` when timing is off or no frame was
    /// recorded since it was switched on.
    pub fn read_timestamps(&mut self, device: &Device, queue: &Queue) -> Option<Vec<DispatchTime>> {
        let timestamps = self.timestamps.as_ref()?;
        if timestamps.names.is_empty() {
            return None;
        }
        let bytes = (timestamps.names.len() as u64 + 1) * 8;
        let slice = timestamps.readback.slice(..bytes);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::wait_indefinitely()).ok()?;
        let ticks: Vec<u64> = slice
            .get_mapped_range()
            .ok()?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| u64::from_le_bytes(*bytes))
            .collect();
        timestamps.readback.unmap();
        let milliseconds_per_tick = queue.get_timestamp_period() as f64 / 1e6;
        Some(
            timestamps
                .names
                .iter()
                .zip(ticks.windows(2))
                .map(|(name, pair)| DispatchTime {
                    name: name.to_string_lossy().into_owned(),
                    milliseconds: pair[1].wrapping_sub(pair[0]) as f64 * milliseconds_per_tick,
                })
                .collect(),
        )
    }

    /// Builds the pack or the unpack pass: a uniform block at binding 0, then `views` at bindings 1, 2, ...
    fn own_step(
        &self,
        device: &Device,
        id: PipelineId,
        views: &[(&TextureView, bool)],
        workgroups: [u32; 2],
    ) -> Result<Step, NrdError> {
        let slots: Vec<(u32, TextureFormat, bool)> = views
            .iter()
            .enumerate()
            .map(|(index, (view, is_storage))| {
                (index as u32 + 1, view.texture().format(), *is_storage)
            })
            .collect();
        let cached = self.pipelines.pipeline(device, id, &slots, true)?;
        let bound: Vec<(u32, &TextureView)> = views
            .iter()
            .enumerate()
            .map(|(index, (view, _))| (index as u32 + 1, *view))
            .collect();
        Ok(Step {
            name: match id {
                PipelineId::Pack => c"nrd_wgpu pack",
                _ => c"nrd_wgpu unpack",
            },
            resources: self.resources_bind_group(device, &cached.layout, &bound, true),
            pipeline: cached.pipeline,
            constants_offset: None,
            workgroups,
        })
    }

    fn resources_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
        slots: &[(u32, &TextureView)],
        own: bool,
    ) -> BindGroup {
        let last = slots.iter().map(|(binding, _)| *binding).max().unwrap_or(0);
        let entries: Vec<BindGroupEntry> = (0..=last)
            .map(|binding| {
                let resource = match slots.iter().find(|(b, _)| *b == binding) {
                    Some((_, view)) => BindingResource::TextureView(view),
                    None if own && binding == 0 => self.params.as_entire_binding(),
                    None => BindingResource::TextureView(&self.filler),
                };
                BindGroupEntry { binding, resource }
            })
            .collect();
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("nrd_resources"),
            layout,
            entries: &entries,
        })
    }

    /// Makes room for the constants of `dispatches` dispatches. wgpu has no push descriptors, so each dispatch's
    /// constants live in their own slot of one uniform buffer, selected with a dynamic offset.
    fn reserve_constants(&mut self, device: &Device, dispatches: u32) {
        if dispatches <= self.constants_capacity && self.constants.is_some() {
            return;
        }
        let shared = &self.pipelines.0;
        self.constants_capacity = dispatches.max(1);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("nrd_constants"),
            size: self.constants_capacity as u64 * self.constants_stride as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries: Vec<BindGroupEntry> = shared
            .samplers
            .iter()
            .enumerate()
            .map(|(index, sampler)| BindGroupEntry {
                binding: shared.layout.first_sampler_binding + index as u32,
                resource: BindingResource::Sampler(sampler),
            })
            .collect();
        entries.push(BindGroupEntry {
            binding: shared.layout.constants_binding,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: NonZero::new(shared.layout.constants_size as u64),
            }),
        });
        // Bindings a dense layout had to fill, see `dense_layout`.
        let last = entries.iter().map(|entry| entry.binding).max().unwrap_or(0);
        for binding in 0..last {
            if !entries.iter().any(|entry| entry.binding == binding) {
                entries.push(BindGroupEntry {
                    binding,
                    resource: BindingResource::TextureView(&self.filler),
                });
            }
        }
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("nrd_constants"),
            layout: &shared.constants_layout,
            entries: &entries,
        });
        self.constants = Some((buffer, bind_group));
    }
}

fn check_size_and_features(
    device: &Device,
    size: [u32; 2],
    precision: TexturePrecision,
) -> Result<(), NrdError> {
    let missing = Denoiser::required_features(precision) - device.features();
    if !missing.is_empty() {
        return Err(NrdError::MissingFeatures(missing));
    }
    // NRD keeps sizes in 16 bits.
    if size[0] == 0 || size[1] == 0 || size[0] > 0xffff || size[1] > 0xffff {
        return Err(NrdError::InvalidSettings(
            "the size must be 1..=65535 on each axis",
        ));
    }
    Ok(())
}

/// Creates an NRD instance of `kind` and checks that this crate can drive it.
fn create_instance(kind: DenoiserKind) -> Result<(InstanceGuard, ffi::NrdwInfo), NrdError> {
    if cfg!(feature = "mock") {
        return Err(NrdError::UnsupportedLibrary(
            "built with the `mock` feature, without NRD",
        ));
    }
    let raw_kind = match kind {
        DenoiserKind::ReblurDiffuseSpecular => ffi::NRDW_DENOISER_REBLUR_DIFFUSE_SPECULAR,
    };
    let mut info = core::mem::MaybeUninit::<ffi::NrdwInfo>::uninit();
    // SAFETY: `nrdw_create` fills `info` whenever it returns an instance.
    let instance = unsafe { ffi::nrdw_create(raw_kind, info.as_mut_ptr()) };
    let instance = NonNull::new(instance).ok_or(NrdError::CreationFailed)?;
    // From here on the instance is owned by a guard, so that an early return destroys it.
    let guard = InstanceGuard(instance);
    // SAFETY: the instance is not null, so `info` was written.
    let info = unsafe { info.assume_init() };

    if info.normal_roughness_is_10_10_10_2 == 0 {
        return Err(NrdError::UnsupportedLibrary(
            "normals and roughness must be packed as 10:10:10:2",
        ));
    }
    if info.resources_set > 1
        || info.constants_and_samplers_set > 1
        || info.resources_set == info.constants_and_samplers_set
    {
        return Err(NrdError::UnsupportedLibrary("unexpected descriptor sets"));
    }
    Ok((guard, info))
}

/// Runs `f` inside wgpu error scopes for validation, internal and out-of-memory errors, and returns the first error
/// any of them caught. On native wgpu every error is known when the scope is popped, so nothing waits.
fn capture_device_errors<R>(device: &Device, f: impl FnOnce() -> R) -> (R, Option<String>) {
    let out_of_memory = device.push_error_scope(ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(ErrorFilter::Internal);
    let validation = device.push_error_scope(ErrorFilter::Validation);
    let result = f();
    // Scopes are a stack: the last pushed is popped first.
    let errors = [validation.pop(), internal.pop(), out_of_memory.pop()];
    let mut context = Context::from_waker(Waker::noop());
    let error = errors.into_iter().find_map(|future| {
        let mut future = pin!(future);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(error) => error.map(|error| error.to_string()),
            Poll::Pending => None,
        }
    });
    (result, error)
}

/// Destroys an NRD instance unless it is released.
struct InstanceGuard(NonNull<ffi::NrdwInstance>);

impl InstanceGuard {
    fn release(self) -> NonNull<ffi::NrdwInstance> {
        let instance = self.0;
        core::mem::forget(self);
        instance
    }
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        // SAFETY: the instance came from `nrdw_create` and nothing else owns it.
        unsafe { ffi::nrdw_destroy(self.0.as_ptr()) };
    }
}

/// A bind group layout with an entry for **every** binding number from 0 to the highest in `slots`.
///
/// wgpu's Vulkan backend numbers the descriptors of a layout 0, 1, 2... in entry order and rewrites the SPIR-V it
/// generates itself to match. A passthrough shader keeps the numbers it was compiled with, so the two only agree
/// when the layout has no gaps. The gaps get a sampled texture entry that nothing reads.
fn dense_layout(device: &Device, label: &str, slots: &[(u32, BindingType)]) -> BindGroupLayout {
    let last = slots.iter().map(|(binding, _)| *binding).max().unwrap_or(0);
    let entries: Vec<BindGroupLayoutEntry> = (0..=last)
        .map(|binding| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: slots.iter().find(|(b, _)| *b == binding).map_or(
                BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                |(_, ty)| *ty,
            ),
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn passthrough_module(
    device: &Device,
    label: &str,
    entry_point: &str,
    spirv: &[u32],
) -> ShaderModule {
    // SAFETY: the SPIR-V is either NRD's own, compiled by its build for Vulkan, or this crate's two passes,
    // compiled by `build.rs`. wgpu does not validate it; the bind group layouts are built to match it.
    unsafe {
        device.create_shader_module_passthrough(ShaderModuleDescriptorPassthrough {
            label: Some(label),
            entry_points: Cow::Owned(vec![PassthroughShaderEntryPoint {
                name: Cow::Borrowed(entry_point),
                // Only read by the Metal backend.
                workgroup_size: (0, 0, 0),
            }]),
            spirv: Some(Cow::Borrowed(spirv)),
            ..Default::default()
        })
    }
}

fn spirv_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

fn create_texture(
    device: &Device,
    label: &str,
    size: [u32; 2],
    format: TextureFormat,
) -> TextureView {
    device
        .create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn convert_format(format: ffi::NrdwFormat) -> Option<TextureFormat> {
    use TextureFormat::*;
    Some(match (format.channels, format.bits, format.kind) {
        (1, 8, ffi::NRDW_KIND_UNORM) => R8Unorm,
        (1, 8, ffi::NRDW_KIND_SNORM) => R8Snorm,
        (1, 8, ffi::NRDW_KIND_UINT) => R8Uint,
        (1, 8, ffi::NRDW_KIND_SINT) => R8Sint,
        (2, 8, ffi::NRDW_KIND_UNORM) => Rg8Unorm,
        (2, 8, ffi::NRDW_KIND_SNORM) => Rg8Snorm,
        (2, 8, ffi::NRDW_KIND_UINT) => Rg8Uint,
        (2, 8, ffi::NRDW_KIND_SINT) => Rg8Sint,
        (4, 8, ffi::NRDW_KIND_UNORM) => Rgba8Unorm,
        (4, 8, ffi::NRDW_KIND_SNORM) => Rgba8Snorm,
        (4, 8, ffi::NRDW_KIND_UINT) => Rgba8Uint,
        (4, 8, ffi::NRDW_KIND_SINT) => Rgba8Sint,
        (1, 16, ffi::NRDW_KIND_UNORM) => R16Unorm,
        (1, 16, ffi::NRDW_KIND_SNORM) => R16Snorm,
        (1, 16, ffi::NRDW_KIND_UINT) => R16Uint,
        (1, 16, ffi::NRDW_KIND_SINT) => R16Sint,
        (1, 16, ffi::NRDW_KIND_FLOAT) => R16Float,
        (2, 16, ffi::NRDW_KIND_UNORM) => Rg16Unorm,
        (2, 16, ffi::NRDW_KIND_SNORM) => Rg16Snorm,
        (2, 16, ffi::NRDW_KIND_UINT) => Rg16Uint,
        (2, 16, ffi::NRDW_KIND_SINT) => Rg16Sint,
        (2, 16, ffi::NRDW_KIND_FLOAT) => Rg16Float,
        (4, 16, ffi::NRDW_KIND_UNORM) => Rgba16Unorm,
        (4, 16, ffi::NRDW_KIND_SNORM) => Rgba16Snorm,
        (4, 16, ffi::NRDW_KIND_UINT) => Rgba16Uint,
        (4, 16, ffi::NRDW_KIND_SINT) => Rgba16Sint,
        (4, 16, ffi::NRDW_KIND_FLOAT) => Rgba16Float,
        (1, 32, ffi::NRDW_KIND_UINT) => R32Uint,
        (1, 32, ffi::NRDW_KIND_SINT) => R32Sint,
        (1, 32, ffi::NRDW_KIND_FLOAT) => R32Float,
        (2, 32, ffi::NRDW_KIND_UINT) => Rg32Uint,
        (2, 32, ffi::NRDW_KIND_SINT) => Rg32Sint,
        (2, 32, ffi::NRDW_KIND_FLOAT) => Rg32Float,
        (4, 32, ffi::NRDW_KIND_UINT) => Rgba32Uint,
        (4, 32, ffi::NRDW_KIND_SINT) => Rgba32Sint,
        (4, 32, ffi::NRDW_KIND_FLOAT) => Rgba32Float,
        (4, 10, ffi::NRDW_KIND_PACKED_10_10_10_2_UNORM) => Rgb10a2Unorm,
        (3, 11, ffi::NRDW_KIND_PACKED_11_11_10_UFLOAT) => Rg11b10Ufloat,
        _ => return None,
    })
}
