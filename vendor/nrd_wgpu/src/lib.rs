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

use std::{borrow::Cow, collections::HashMap, ffi::CStr, fmt, num::NonZero, ptr::NonNull};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, Extent3d, Features, FilterMode, PassthroughShaderEntryPoint,
    PipelineLayoutDescriptor, Queue, Sampler, SamplerBindingType, SamplerDescriptor, ShaderModule,
    ShaderModuleDescriptorPassthrough, ShaderStages, StorageTextureAccess, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDimension,
};

/// Bytes of the uniform block shared by the pack and unpack passes.
const PARAMS_SIZE: u64 = 32;

/// The format of the two textures of [`DenoiserOutputs`].
pub const OUTPUT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

static PACK_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pack.spv"));
static UNPACK_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unpack.spv"));

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
    /// 16-bit floats, as NRD asks. The denoised image comes out darker than its input, by about 3 % with 30
    /// accumulated frames (measured on a uniform surface, see the README).
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

/// An NRD denoiser for one resolution. A different resolution needs a new `Denoiser`.
pub struct Denoiser {
    instance: NonNull<ffi::NrdwInstance>,
    size: [u32; 2],
    settings: ReblurSettings,
    previous: Option<Previous>,

    entry_point: String,
    resources_set: u32,
    constants_set: u32,
    first_texture_binding: u32,
    first_storage_binding: u32,
    first_sampler_binding: u32,
    constants_binding: u32,

    nrd_pipelines: Vec<NrdPipeline>,
    pack_module: ShaderModule,
    unpack_module: ShaderModule,
    /// One pipeline per shader **and** per set of bound texture formats: wgpu fixes the formats in the layout.
    pipelines: HashMap<(PipelineId, Vec<TextureFormat>), CachedPipeline>,

    constants_layout: BindGroupLayout,
    samplers: Vec<Sampler>,
    constants_size: u32,
    constants_stride: u32,
    constants_capacity: u32,
    constants: Option<(Buffer, BindGroup)>,
    params: Buffer,

    permanent_pool: Vec<TextureView>,
    transient_pool: Vec<TextureView>,
    packed_diffuse: TextureView,
    packed_specular: TextureView,
    packed_normal_roughness: TextureView,
    packed_view_z: TextureView,
    packed_motion_vectors: TextureView,
    denoised_diffuse: TextureView,
    denoised_specular: TextureView,
    /// Bound at the binding numbers a shader leaves unused, see [`dense_layout`].
    filler: TextureView,
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
    /// README); a driver is free not to.
    pub fn required_features(precision: TexturePrecision) -> Features {
        let features =
            Features::PASSTHROUGH_SHADERS | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        match precision {
            TexturePrecision::Half => features,
            TexturePrecision::Full => features | Features::FLOAT32_FILTERABLE,
        }
    }

    /// Creates a denoiser for textures of `size` pixels.
    pub fn new(
        device: &Device,
        kind: DenoiserKind,
        size: [u32; 2],
        precision: TexturePrecision,
    ) -> Result<Self, NrdError> {
        let missing = Self::required_features(precision) - device.features();
        if !missing.is_empty() {
            return Err(NrdError::MissingFeatures(missing));
        }
        // NRD keeps sizes in 16 bits.
        if size[0] == 0 || size[1] == 0 || size[0] > 0xffff || size[1] > 0xffff {
            return Err(NrdError::InvalidSettings(
                "the size must be 1..=65535 on each axis",
            ));
        }

        let raw_kind = match kind {
            DenoiserKind::ReblurDiffuseSpecular => ffi::NRDW_DENOISER_REBLUR_DIFFUSE_SPECULAR,
        };
        let mut info = core::mem::MaybeUninit::<ffi::NrdwInfo>::uninit();
        // SAFETY: `nrdw_create` fills `info` whenever it returns an instance.
        let instance = unsafe { ffi::nrdw_create(raw_kind, info.as_mut_ptr()) };
        let instance = NonNull::new(instance).ok_or(NrdError::CreationFailed)?;
        // SAFETY: the instance is not null, so `info` was written.
        let info = unsafe { info.assume_init() };
        // From here on the instance is owned by a guard, so that an early return destroys it.
        let guard = InstanceGuard(instance);

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
        // SAFETY: NRD returns a NUL-terminated string that lives as long as the library.
        let entry_point = unsafe { CStr::from_ptr(info.entry_point) }
            .to_string_lossy()
            .into_owned();

        let samplers: Vec<Sampler> = (0..info.sampler_count)
            .map(|index| {
                // SAFETY: `index` is below the sampler count NRD reported.
                let linear = unsafe { ffi::nrdw_sampler_is_linear(instance.as_ptr(), index) } != 0;
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

        let mut nrd_pipelines = Vec::new();
        for index in 0..info.pipeline_count {
            let mut raw = core::mem::MaybeUninit::<ffi::NrdwPipeline>::uninit();
            // SAFETY: `index` is below the pipeline count NRD reported, and the shim fills the struct.
            let raw = unsafe {
                ffi::nrdw_pipeline(instance.as_ptr(), index, raw.as_mut_ptr());
                raw.assume_init()
            };
            // SAFETY: NRD embeds each shader as `spirv_bytes` bytes of 4-byte SPIR-V words in static storage,
            // and the name is a NUL-terminated string inside the instance.
            let (spirv, name) = unsafe {
                (
                    core::slice::from_raw_parts(raw.spirv, raw.spirv_bytes as usize / 4),
                    CStr::from_ptr(raw.name).to_string_lossy().into_owned(),
                )
            };
            nrd_pipelines.push(NrdPipeline {
                module: passthrough_module(device, &name, &entry_point, spirv),
                name,
                texture_count: raw.texture_count,
                storage_count: raw.storage_count,
            });
        }

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
        let permanent_pool = pool(true, info.permanent_pool_size)?;
        let transient_pool = pool(false, info.transient_pool_size)?;

        let params = device.create_buffer(&BufferDescriptor {
            label: Some("nrd_pack_params"),
            size: PARAMS_SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let constants_size = info.constants_max_size.next_multiple_of(16);
        let mut denoiser = Self {
            instance: guard.release(),
            size,
            settings: ReblurSettings::default(),
            previous: None,
            entry_point: entry_point.clone(),
            resources_set: info.resources_set,
            constants_set: info.constants_and_samplers_set,
            first_texture_binding: info.first_texture_binding,
            first_storage_binding: info.first_storage_binding,
            first_sampler_binding: info.first_sampler_binding,
            constants_binding: info.constants_binding,
            nrd_pipelines,
            pack_module: passthrough_module(device, "nrd_pack", "main", &spirv_words(PACK_SPIRV)),
            unpack_module: passthrough_module(
                device,
                "nrd_unpack",
                "main",
                &spirv_words(UNPACK_SPIRV),
            ),
            pipelines: HashMap::new(),
            constants_layout,
            samplers,
            constants_size,
            constants_stride: constants_size
                .next_multiple_of(device.limits().min_uniform_buffer_offset_alignment),
            constants_capacity: 0,
            constants: None,
            params,
            permanent_pool,
            transient_pool,
            packed_diffuse: create_texture(
                device,
                "nrd_packed_diffuse",
                size,
                TextureFormat::Rgba16Float,
            ),
            packed_specular: create_texture(
                device,
                "nrd_packed_specular",
                size,
                TextureFormat::Rgba16Float,
            ),
            packed_normal_roughness: create_texture(
                device,
                "nrd_packed_normal_roughness",
                size,
                TextureFormat::Rgb10a2Unorm,
            ),
            packed_view_z: create_texture(
                device,
                "nrd_packed_view_z",
                size,
                TextureFormat::R32Float,
            ),
            packed_motion_vectors: create_texture(
                device,
                "nrd_packed_motion_vectors",
                size,
                TextureFormat::Rg16Float,
            ),
            // NRD also uses its two outputs as scratch space inside a frame, so they follow the precision too.
            denoised_diffuse: create_texture(
                device,
                "nrd_denoised_diffuse",
                size,
                precision.apply(TextureFormat::Rgba16Float),
            ),
            denoised_specular: create_texture(
                device,
                "nrd_denoised_specular",
                size,
                precision.apply(TextureFormat::Rgba16Float),
            ),
            filler: create_texture(device, "nrd_filler", [1, 1], TextureFormat::Rgba8Unorm),
        };
        let settings = denoiser.settings;
        denoiser.set_reblur_settings(&settings)?;
        Ok(denoiser)
    }

    /// The size, in pixels, of every texture this denoiser reads and writes.
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// Bytes of GPU memory in the textures this denoiser owns: NRD's two pools and the packed and denoised
    /// copies around them.
    pub fn texture_bytes(&self) -> u64 {
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
        let mut info = core::mem::MaybeUninit::<ffi::NrdwInfo>::uninit();
        // SAFETY: `nrdw_create` fills `info` whenever it returns an instance, which is destroyed right away.
        unsafe {
            let instance = ffi::nrdw_create(
                ffi::NRDW_DENOISER_REBLUR_DIFFUSE_SPECULAR,
                info.as_mut_ptr(),
            );
            if instance.is_null() {
                return [0; 3];
            }
            ffi::nrdw_destroy(instance);
            let info = info.assume_init();
            [info.version_major, info.version_minor, info.version_build]
        }
    }

    /// NRD's shaders for this denoiser, as `file|macro=value|...`, each with whether a frame has used it yet.
    /// Which ones a frame uses depends on the settings; some are debug views this crate never asks for.
    pub fn shaders(&self) -> impl Iterator<Item = (&str, bool)> {
        self.nrd_pipelines
            .iter()
            .enumerate()
            .map(|(index, pipeline)| {
                let used = self
                    .pipelines
                    .keys()
                    .any(|(id, _)| *id == PipelineId::Nrd(index as u32));
                (pipeline.name.as_str(), used)
            })
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
        if settings
            .hit_distance_parameters
            .iter()
            .any(|p| p.is_nan() || *p <= 0.0)
        {
            return Err(NrdError::InvalidSettings(
                "hit_distance_parameters must be positive",
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
    /// The first frames create compute pipelines on demand (one per NRD shader and set of texture formats).
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

        steps.push(self.own_step(
            device,
            PipelineId::Pack,
            &[
                (inputs.diffuse, false),
                (inputs.specular, false),
                (inputs.normal_roughness, false),
                (inputs.view_depth, false),
                (inputs.motion_vectors, false),
                (&self.packed_diffuse.clone(), true),
                (&self.packed_specular.clone(), true),
                (&self.packed_normal_roughness.clone(), true),
                (&self.packed_view_z.clone(), true),
                (&self.packed_motion_vectors.clone(), true),
            ],
            pass_workgroups,
        )?);

        // SAFETY: the instance is alive. The dispatch list it returns stays valid until the next call, and no
        // other call into NRD is made before the end of this function.
        let dispatch_count = unsafe { ffi::nrdw_begin_dispatches(self.instance.as_ptr()) };
        let dispatch_count = u32::try_from(dispatch_count)
            .map_err(|_| NrdError::InvalidSettings("NRD returned no dispatches"))?;
        self.reserve_constants(device, dispatch_count);

        let mut constants_slot = 0;
        for index in 0..dispatch_count {
            let mut raw = core::mem::MaybeUninit::<ffi::NrdwDispatch>::uninit();
            // SAFETY: `index` is below the dispatch count, and the shim fills the struct.
            let raw = unsafe {
                ffi::nrdw_dispatch(self.instance.as_ptr(), index, raw.as_mut_ptr());
                raw.assume_init()
            };
            let pipeline = &self.nrd_pipelines[raw.pipeline as usize];
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
                    ffi::NRDW_RESOURCE_MOTION_VECTORS => Some(&self.packed_motion_vectors),
                    ffi::NRDW_RESOURCE_NORMAL_ROUGHNESS => Some(&self.packed_normal_roughness),
                    ffi::NRDW_RESOURCE_VIEW_Z => Some(&self.packed_view_z),
                    ffi::NRDW_RESOURCE_IN_DIFFUSE => Some(&self.packed_diffuse),
                    ffi::NRDW_RESOURCE_IN_SPECULAR => Some(&self.packed_specular),
                    ffi::NRDW_RESOURCE_OUT_DIFFUSE => Some(&self.denoised_diffuse),
                    ffi::NRDW_RESOURCE_OUT_SPECULAR => Some(&self.denoised_specular),
                    ffi::NRDW_RESOURCE_TRANSIENT_POOL => self.transient_pool.get(pool_index),
                    ffi::NRDW_RESOURCE_PERMANENT_POOL => self.permanent_pool.get(pool_index),
                    _ => None,
                };
                let is_storage = resource_index >= texture_count;
                match view {
                    Some(view) if is_storage == (resource.is_storage != 0) => {
                        views.push((view.clone(), is_storage));
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

            let slots: Vec<(u32, &TextureView, bool)> = views
                .iter()
                .enumerate()
                .map(|(position, (view, is_storage))| {
                    let position = position as u32;
                    let binding = if *is_storage {
                        self.first_storage_binding + position - texture_count
                    } else {
                        self.first_texture_binding + position
                    };
                    (binding, view, *is_storage)
                })
                .collect();
            let cached = self.pipeline(device, PipelineId::Nrd(raw.pipeline), &slots, false)?;
            steps.push(Step {
                resources: self.resources_bind_group(device, &cached.layout, &slots, false),
                pipeline: cached.pipeline,
                constants_offset: Some(constants_offset),
                workgroups: [raw.grid_width, raw.grid_height],
            });
        }

        steps.push(self.own_step(
            device,
            PipelineId::Unpack,
            &[
                (&self.denoised_diffuse.clone(), false),
                (&self.denoised_specular.clone(), false),
                (&self.packed_view_z.clone(), false),
                (outputs.diffuse, true),
                (outputs.specular, true),
            ],
            pass_workgroups,
        )?);

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("nrd"),
            timestamp_writes: None,
        });
        for step in &steps {
            pass.set_pipeline(&step.pipeline);
            match step.constants_offset {
                Some(offset) => {
                    let (_, constants) = self.constants.as_ref().unwrap();
                    pass.set_bind_group(self.resources_set, &step.resources, &[]);
                    pass.set_bind_group(self.constants_set, constants, &[offset]);
                }
                None => pass.set_bind_group(0, &step.resources, &[]),
            }
            pass.dispatch_workgroups(step.workgroups[0], step.workgroups[1], 1);
        }
        Ok(())
    }

    /// Builds the pack or the unpack pass: a uniform block at binding 0, then `views` at bindings 1, 2, ...
    fn own_step(
        &mut self,
        device: &Device,
        id: PipelineId,
        views: &[(&TextureView, bool)],
        workgroups: [u32; 2],
    ) -> Result<Step, NrdError> {
        let slots: Vec<(u32, &TextureView, bool)> = views
            .iter()
            .enumerate()
            .map(|(index, (view, is_storage))| (index as u32 + 1, *view, *is_storage))
            .collect();
        let cached = self.pipeline(device, id, &slots, true)?;
        Ok(Step {
            resources: self.resources_bind_group(device, &cached.layout, &slots, true),
            pipeline: cached.pipeline,
            constants_offset: None,
            workgroups,
        })
    }

    /// The pipeline of shader `id` for the texture formats of `slots`, created on first use.
    fn pipeline(
        &mut self,
        device: &Device,
        id: PipelineId,
        slots: &[(u32, &TextureView, bool)],
        own: bool,
    ) -> Result<CachedPipeline, NrdError> {
        let formats: Vec<TextureFormat> = slots
            .iter()
            .map(|(_, view, _)| view.texture().format())
            .collect();
        if let Some(cached) = self.pipelines.get(&(id, formats.clone())) {
            return Ok(cached.clone());
        }

        let mut layout_slots = Vec::new();
        if own {
            layout_slots.push((0, Slot::Uniform));
        }
        for ((binding, _, is_storage), format) in slots.iter().zip(&formats) {
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
        let layout = dense_layout(device, "nrd_resources_layout", &entries);

        let (module, entry_point, label) = match id {
            PipelineId::Nrd(index) => {
                let pipeline = &self.nrd_pipelines[index as usize];
                (
                    &pipeline.module,
                    self.entry_point.as_str(),
                    pipeline.name.as_str(),
                )
            }
            PipelineId::Pack => (&self.pack_module, "main", "nrd_pack"),
            PipelineId::Unpack => (&self.unpack_module, "main", "nrd_unpack"),
        };
        let mut sets = [None, None];
        if own {
            sets[0] = Some(&layout);
        } else {
            sets[self.resources_set as usize] = Some(&layout);
            sets[self.constants_set as usize] = Some(&self.constants_layout);
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

        let cached = CachedPipeline { pipeline, layout };
        self.pipelines.insert((id, formats), cached.clone());
        Ok(cached)
    }

    fn resources_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
        slots: &[(u32, &TextureView, bool)],
        own: bool,
    ) -> BindGroup {
        let last = slots
            .iter()
            .map(|(binding, ..)| *binding)
            .max()
            .unwrap_or(0);
        let entries: Vec<BindGroupEntry> = (0..=last)
            .map(|binding| {
                let resource = match slots.iter().find(|(b, ..)| *b == binding) {
                    Some((_, view, _)) => BindingResource::TextureView(view),
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
        self.constants_capacity = dispatches.max(1);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("nrd_constants"),
            size: self.constants_capacity as u64 * self.constants_stride as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries: Vec<BindGroupEntry> = self
            .samplers
            .iter()
            .enumerate()
            .map(|(index, sampler)| BindGroupEntry {
                binding: self.first_sampler_binding + index as u32,
                resource: BindingResource::Sampler(sampler),
            })
            .collect();
        entries.push(BindGroupEntry {
            binding: self.constants_binding,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: NonZero::new(self.constants_size as u64),
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
            layout: &self.constants_layout,
            entries: &entries,
        });
        self.constants = Some((buffer, bind_group));
    }
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
