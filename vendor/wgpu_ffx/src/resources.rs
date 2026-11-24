use wgpu::util::DeviceExt as _;

use crate::{pass::ResourceAccess, FrameKind, FsrDispatchInfo};

/// A pair of textures used for temporal double-buffering.
///
/// One texture holds the current frame's data (write target), while the other
/// holds the previous frame's data (read source). Which physical texture is
/// "current" flips each frame based on [`FrameKind`].
pub(crate) struct DoubleBuffered {
    current_on_even: wgpu::Texture,
    current_on_odd: wgpu::Texture,
}

impl DoubleBuffered {
    fn new(current_on_even: wgpu::Texture, current_on_odd: wgpu::Texture) -> Self {
        Self {
            current_on_even,
            current_on_odd,
        }
    }

    /// The texture being written to this frame.
    pub fn current(&self, kind: FrameKind) -> &wgpu::Texture {
        match kind {
            FrameKind::Even => &self.current_on_even,
            FrameKind::Odd => &self.current_on_odd,
        }
    }

    /// The texture that was written to last frame (read source).
    pub fn previous(&self, kind: FrameKind) -> &wgpu::Texture {
        match kind {
            FrameKind::Even => &self.current_on_odd,
            FrameKind::Odd => &self.current_on_even,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessType {
    Srv,
    Uav,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum FsrResourceName {
    InputColor,
    InputDepth,
    InputMotionVectors,
    InputExposure,
    InputReactiveMask,
    InputTransparencyAndComposition,

    OutputColor,
    DilatedDepth,
    DilatedMotionVectors,
    ReconstructedPreviousDepth,

    Constants,

    /// Current frame's accumulation (write target).
    AccumulationCurrent,
    /// Previous frame's accumulation (read source).
    AccumulationPrevious,
    /// Current frame's luma.
    Luma,
    /// Previous frame's luma.
    PreviousLuma,
    LumaInstability,
    ShadingChange,
    NewLocks,
    /// Current frame's internal upscaled color (write target for Accumulate).
    InternalUpscaledCurrent,
    /// Previous frame's internal upscaled color (read source for Accumulate).
    InternalUpscaledPrevious,
    SpdMips,
    FarthestDepth,
    FarthestDepthMip1,
    /// Current frame's luma history (write target).
    LumaHistoryCurrent,
    /// Previous frame's luma history (read source).
    LumaHistoryPrevious,
    SpdAtomicCount,
    DilatedReactiveMasks,
    Lanczos2Lut,
    FrameInfo,

    SamplerPointClamp,
    SamplerLinearClamp,
}

impl FsrResourceName {
    /// The storage format the shaders declare for this image.
    pub(crate) fn format(&self) -> wgpu::TextureFormat {
        match self {
            FsrResourceName::OutputColor => wgpu::TextureFormat::Rgba16Float,
            FsrResourceName::DilatedDepth => wgpu::TextureFormat::R32Float,
            FsrResourceName::DilatedMotionVectors => wgpu::TextureFormat::Rg16Float,
            FsrResourceName::AccumulationCurrent | FsrResourceName::AccumulationPrevious => {
                wgpu::TextureFormat::R8Unorm
            }
            FsrResourceName::Luma
            | FsrResourceName::PreviousLuma
            | FsrResourceName::LumaInstability
            | FsrResourceName::FarthestDepth
            | FsrResourceName::FarthestDepthMip1 => wgpu::TextureFormat::R16Float,
            FsrResourceName::ShadingChange | FsrResourceName::NewLocks => {
                wgpu::TextureFormat::R8Unorm
            }
            FsrResourceName::InternalUpscaledCurrent
            | FsrResourceName::InternalUpscaledPrevious
            | FsrResourceName::LumaHistoryCurrent
            | FsrResourceName::LumaHistoryPrevious => wgpu::TextureFormat::Rgba16Float,
            FsrResourceName::SpdMips => wgpu::TextureFormat::Rg16Float,
            FsrResourceName::DilatedReactiveMasks => wgpu::TextureFormat::Rgba8Unorm,
            FsrResourceName::FrameInfo => wgpu::TextureFormat::Rgba32Float,
            other => panic!("{other:?} is never bound as a storage image"),
        }
    }

    pub(crate) fn to_bgl_entry(
        self,
        binding: u32,
        access_type: AccessType,
    ) -> wgpu::BindGroupLayoutEntry {
        let ty = match (self, access_type) {
            (FsrResourceName::Constants, _) => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            // The shaders declare both buffers read-write, even where they only read them.
            (FsrResourceName::ReconstructedPreviousDepth | FsrResourceName::SpdAtomicCount, _) => {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            }
            (FsrResourceName::SamplerPointClamp, _) => {
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering)
            }
            (FsrResourceName::SamplerLinearClamp, _) => {
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
            }
            (_, AccessType::Uav) => wgpu::BindingType::StorageTexture {
                access: match self {
                    FsrResourceName::InternalUpscaledCurrent
                    | FsrResourceName::OutputColor
                    | FsrResourceName::DilatedDepth
                    | FsrResourceName::DilatedMotionVectors
                    | FsrResourceName::DilatedReactiveMasks => {
                        wgpu::StorageTextureAccess::WriteOnly
                    }
                    _ => wgpu::StorageTextureAccess::ReadWrite,
                },
                format: self.format(),
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            (_, AccessType::Srv) => wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float {
                    filterable: !matches!(self, FsrResourceName::InputDepth),
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        };
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        }
    }
}

/// Every texture and buffer FSR keeps for one view, sized for its maximum render and upscale sizes.
pub(crate) struct FsrResources {
    pub(crate) constant_buffer: wgpu::Buffer,

    pub(crate) accumulation: DoubleBuffered,
    pub(crate) luma: DoubleBuffered,
    pub(crate) intermediate_fp16x1: wgpu::Texture,
    pub(crate) shading_change: wgpu::Texture,
    pub(crate) new_locks: wgpu::Texture,
    pub(crate) internal_upscaled: DoubleBuffered,
    pub(crate) spd_mips: wgpu::Texture,
    pub(crate) farthest_depth_mip1: wgpu::Texture,
    pub(crate) luma_history: DoubleBuffered,
    pub(crate) spd_atomic_counter: wgpu::Buffer,
    pub(crate) dilated_reactive_masks: wgpu::Texture,
    pub(crate) dilated_depth: wgpu::Texture,
    pub(crate) dilated_motion_vectors: wgpu::Texture,
    pub(crate) reconstructed_previous_depth: wgpu::Buffer,
    pub(crate) lanczos2_lut: wgpu::Texture,
    pub(crate) default_reactivity_mask: wgpu::Texture,
    pub(crate) default_exposure: wgpu::Texture,
    pub(crate) frame_info: wgpu::Texture,

    pub(crate) sampler_point_clamp: wgpu::Sampler,
    pub(crate) sampler_linear_clamp: wgpu::Sampler,
}

impl FsrResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_render_size: [u32; 2],
        max_upscale_size: [u32; 2],
    ) -> Self {
        let extent = |[width, height]: [u32; 2]| wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let half_max_render_size = extent([max_render_size[0] / 2, max_render_size[1] / 2]);

        let constant_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FSR3 Constants"),
            size: size_of::<crate::constants::Constants>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture = |label, size, mip_level_count, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let storage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let render_tex =
            |label, format| texture(label, extent(max_render_size), 1, format, storage);
        let upscale_tex =
            |label, format| texture(label, extent(max_upscale_size), 1, format, storage);
        let half_tex = |label, format| texture(label, half_max_render_size, 1, format, storage);

        // Double-buffered resources: current_on_even written on even frames,
        // current_on_odd written on odd frames.
        let accumulation = DoubleBuffered::new(
            render_tex("FSR3 Accumulation (even)", wgpu::TextureFormat::R8Unorm),
            render_tex("FSR3 Accumulation (odd)", wgpu::TextureFormat::R8Unorm),
        );
        let luma = DoubleBuffered::new(
            render_tex("FSR3 Luma (even)", wgpu::TextureFormat::R16Float),
            render_tex("FSR3 Luma (odd)", wgpu::TextureFormat::R16Float),
        );
        let internal_upscaled = DoubleBuffered::new(
            upscale_tex(
                "FSR3 Internal Upscaled (even)",
                wgpu::TextureFormat::Rgba16Float,
            ),
            upscale_tex(
                "FSR3 Internal Upscaled (odd)",
                wgpu::TextureFormat::Rgba16Float,
            ),
        );
        let luma_history = DoubleBuffered::new(
            render_tex("FSR3 Luma History (even)", wgpu::TextureFormat::Rgba16Float),
            render_tex("FSR3 Luma History (odd)", wgpu::TextureFormat::Rgba16Float),
        );
        let intermediate_fp16x1 =
            render_tex("FSR3 Intermediate FP16x1", wgpu::TextureFormat::R16Float);
        let shading_change = half_tex("FSR3 Shading Change", wgpu::TextureFormat::R8Unorm);
        let new_locks = upscale_tex("FSR3 New Locks", wgpu::TextureFormat::R8Unorm);
        let spd_mips = texture(
            "FSR3 SPD Mips",
            half_max_render_size,
            half_max_render_size.max_mips(wgpu::TextureDimension::D2),
            wgpu::TextureFormat::Rg16Float,
            storage,
        );
        let farthest_depth_mip1 =
            half_tex("FSR3 Farthest Depth Mip1", wgpu::TextureFormat::R16Float);
        let dilated_reactive_masks = render_tex(
            "FSR3 Dilated Reactive Masks",
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let dilated_depth = render_tex("FSR3 Dilated Depth", wgpu::TextureFormat::R32Float);
        let dilated_motion_vectors = render_tex(
            "FSR3 Dilated Motion Vectors",
            wgpu::TextureFormat::Rg16Float,
        );

        // Cleared to zero every frame (see `FsrContext::dispatch`).
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let spd_atomic_counter = buffer("FSR3 SPD Atomic Counter", 4);
        // The shaders index it with the maximum render width as the stride.
        let reconstructed_previous_depth = buffer(
            "FSR3 Reconstructed Previous Depth",
            u64::from(max_render_size[0]) * u64::from(max_render_size[1]) * 4,
        );

        let lanczos2_lut_data = crate::lanczos2::generate_lanczos2_lut();
        let lanczos2_lut = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("FSR3 Lanczos2 LUT"),
                size: extent([lanczos2_lut_data.len() as u32, 1]),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Snorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::default(),
            bytemuck::cast_slice(&lanczos2_lut_data),
        );

        // wgpu zero-initializes both: no reactivity, and an exposure of 0, which the shaders read as 1.
        let default_reactivity_mask = texture(
            "FSR3 Default Reactivity Mask",
            extent([1, 1]),
            1,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let default_exposure = texture(
            "FSR3 Default Exposure",
            extent([1, 1]),
            1,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let frame_info = texture(
            "FSR3 Frame Info",
            extent([1, 1]),
            1,
            wgpu::TextureFormat::Rgba32Float,
            storage | wgpu::TextureUsages::COPY_DST,
        );

        let sampler = |label, filter, mipmap_filter| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: filter,
                min_filter: filter,
                mipmap_filter,
                ..Default::default()
            })
        };

        Self {
            constant_buffer,
            accumulation,
            luma,
            intermediate_fp16x1,
            shading_change,
            new_locks,
            internal_upscaled,
            spd_mips,
            farthest_depth_mip1,
            luma_history,
            spd_atomic_counter,
            dilated_reactive_masks,
            dilated_depth,
            dilated_motion_vectors,
            reconstructed_previous_depth,
            lanczos2_lut,
            default_reactivity_mask,
            default_exposure,
            frame_info,
            sampler_point_clamp: sampler(
                "FSR3 Sampler Point Clamp",
                wgpu::FilterMode::Nearest,
                wgpu::MipmapFilterMode::Nearest,
            ),
            sampler_linear_clamp: sampler(
                "FSR3 Sampler Linear Clamp",
                wgpu::FilterMode::Linear,
                wgpu::MipmapFilterMode::Linear,
            ),
        }
    }

    pub(crate) fn to_binding(
        &self,
        dispatch: &FsrDispatchInfo,
        access: &ResourceAccess,
        kind: FrameKind,
    ) -> OwnedBindingResource {
        let descriptor = wgpu::TextureViewDescriptor {
            label: Some("FSR3 Resource View"),
            base_mip_level: access.mip.unwrap_or(0),
            mip_level_count: access.mip.map(|_| 1),
            ..Default::default()
        };
        let view =
            |texture: &wgpu::Texture| OwnedBindingResource::View(texture.create_view(&descriptor));

        match access.name {
            FsrResourceName::InputColor => view(&dispatch.color),
            FsrResourceName::InputDepth => view(&dispatch.depth),
            FsrResourceName::InputMotionVectors => view(&dispatch.motion_vectors),
            FsrResourceName::InputExposure => view(&self.default_exposure),
            // The transparency-and-composition mask is empty too; the SDK also binds the default mask for it.
            FsrResourceName::InputReactiveMask
            | FsrResourceName::InputTransparencyAndComposition => {
                view(&self.default_reactivity_mask)
            }
            FsrResourceName::OutputColor => view(&dispatch.output),
            FsrResourceName::DilatedDepth => view(&self.dilated_depth),
            FsrResourceName::DilatedMotionVectors => view(&self.dilated_motion_vectors),
            FsrResourceName::ReconstructedPreviousDepth => {
                OwnedBindingResource::Buffer(self.reconstructed_previous_depth.clone())
            }
            FsrResourceName::Constants => {
                OwnedBindingResource::Buffer(self.constant_buffer.clone())
            }
            FsrResourceName::AccumulationCurrent => view(self.accumulation.current(kind)),
            FsrResourceName::AccumulationPrevious => view(self.accumulation.previous(kind)),
            FsrResourceName::Luma => view(self.luma.current(kind)),
            FsrResourceName::PreviousLuma => view(self.luma.previous(kind)),
            FsrResourceName::InternalUpscaledCurrent => view(self.internal_upscaled.current(kind)),
            FsrResourceName::InternalUpscaledPrevious => {
                view(self.internal_upscaled.previous(kind))
            }
            FsrResourceName::LumaHistoryCurrent => view(self.luma_history.current(kind)),
            FsrResourceName::LumaHistoryPrevious => view(self.luma_history.previous(kind)),
            // One R16F texture serves both: the farthest depth is consumed by the luma pyramid before the luma
            // instability pass writes over it.
            FsrResourceName::LumaInstability | FsrResourceName::FarthestDepth => {
                view(&self.intermediate_fp16x1)
            }
            FsrResourceName::ShadingChange => view(&self.shading_change),
            FsrResourceName::NewLocks => view(&self.new_locks),
            FsrResourceName::SpdMips => view(&self.spd_mips),
            FsrResourceName::FarthestDepthMip1 => view(&self.farthest_depth_mip1),
            FsrResourceName::SpdAtomicCount => {
                OwnedBindingResource::Buffer(self.spd_atomic_counter.clone())
            }
            FsrResourceName::DilatedReactiveMasks => view(&self.dilated_reactive_masks),
            FsrResourceName::Lanczos2Lut => view(&self.lanczos2_lut),
            FsrResourceName::FrameInfo => view(&self.frame_info),
            FsrResourceName::SamplerPointClamp => {
                OwnedBindingResource::Sampler(self.sampler_point_clamp.clone())
            }
            FsrResourceName::SamplerLinearClamp => {
                OwnedBindingResource::Sampler(self.sampler_linear_clamp.clone())
            }
        }
    }
}

pub(crate) enum OwnedBindingResource {
    View(wgpu::TextureView),
    Buffer(wgpu::Buffer),
    Sampler(wgpu::Sampler),
}

impl<'a> From<&'a OwnedBindingResource> for wgpu::BindingResource<'a> {
    fn from(value: &'a OwnedBindingResource) -> Self {
        match value {
            OwnedBindingResource::View(v) => wgpu::BindingResource::TextureView(v),
            OwnedBindingResource::Buffer(b) => b.as_entire_binding(),
            OwnedBindingResource::Sampler(s) => wgpu::BindingResource::Sampler(s),
        }
    }
}
