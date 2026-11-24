use std::borrow::Cow;

use crate::{
    resources::{AccessType, FsrResourceName, FsrResources},
    FrameKind, FsrDispatchInfo,
};

pub(crate) struct FsrPass {
    kind: FsrPassKind,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl FsrPass {
    pub fn new(device: &wgpu::Device, kind: FsrPassKind) -> Self {
        let spirv: Vec<u32> = kind
            .spirv()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        // SAFETY: the SPIR-V is compiled by `compile-shaders.sh` for Vulkan from AMD's GLSL, and wgpu does not
        // validate it. The bind group layout below lists the bindings the GLSL declares, in order.
        let shader_module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(kind.label()),
                entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                    name: Cow::Borrowed("main"),
                    // Only read by the Metal backend.
                    workgroup_size: (0, 0, 0),
                }]),
                spirv: Some(Cow::Owned(spirv)),
                ..Default::default()
            })
        };

        let bgl_entries: Vec<_> = kind
            .resources()
            .into_iter()
            .enumerate()
            .map(|(i, access)| access.name.to_bgl_entry(i as u32, access.access_type))
            .collect();

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(kind.label()),
            entries: &bgl_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(kind.label()),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(kind.label()),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            kind,
            pipeline,
            bgl,
        }
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
        resources: &FsrResources,
        info: &FsrDispatchInfo,
        frame_kind: FrameKind,
        [x, y]: [u32; 2],
    ) {
        let bindings: Vec<_> = self
            .kind
            .resources()
            .iter()
            .map(|access| resources.to_binding(info, access, frame_kind))
            .collect();

        let entries: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, binding)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: binding.into(),
            })
            .collect();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.kind.label()),
            layout: &self.bgl,
            entries: &entries,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum FsrPassKind {
    /// A pass which prepares game inputs for later passes
    PrepareInputs,
    /// A pass which generates the luminance mipmap chain for the current frame.
    LumaPyramid,
    /// A pass which generates the shading change detection mipmap chain for the current frame.
    ShadingChangePyramid,
    /// A pass which estimates shading changes for the current frame
    ShadingChange,
    /// A pass which prepares accumulation relevant information
    PrepareReactivity,
    /// A pass which estimates temporal instability of the luminance changes.
    LumaInstability,
    /// A pass which performs upscaling.
    Accumulate,
}

static PREPARE_INPUTS: &[u8] = include_bytes!("../spv/prepare_inputs.spv");
static LUMA_PYRAMID: &[u8] = include_bytes!("../spv/luma_pyramid.spv");
static SHADING_CHANGE_PYRAMID: &[u8] = include_bytes!("../spv/shading_change_pyramid.spv");
static SHADING_CHANGE: &[u8] = include_bytes!("../spv/shading_change.spv");
static PREPARE_REACTIVITY: &[u8] = include_bytes!("../spv/prepare_reactivity.spv");
static LUMA_INSTABILITY: &[u8] = include_bytes!("../spv/luma_instability.spv");
static ACCUMULATE: &[u8] = include_bytes!("../spv/accumulate.spv");

pub(crate) struct ResourceAccess {
    pub name: FsrResourceName,
    pub access_type: AccessType,
    /// The one mip level the view shows, for the SPD mip chain; `None` for the whole texture.
    pub mip: Option<u32>,
}

impl FsrPassKind {
    pub const ALL: [FsrPassKind; 7] = [
        FsrPassKind::PrepareInputs,
        FsrPassKind::LumaPyramid,
        FsrPassKind::ShadingChangePyramid,
        FsrPassKind::ShadingChange,
        FsrPassKind::PrepareReactivity,
        FsrPassKind::LumaInstability,
        FsrPassKind::Accumulate,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            FsrPassKind::PrepareInputs => "FSR3 Prepare Inputs",
            FsrPassKind::LumaPyramid => "FSR3 Luma Pyramid",
            FsrPassKind::ShadingChangePyramid => "FSR3 Shading Change Pyramid",
            FsrPassKind::ShadingChange => "FSR3 Shading Change",
            FsrPassKind::PrepareReactivity => "FSR3 Prepare Reactivity",
            FsrPassKind::LumaInstability => "FSR3 Luma Instability",
            FsrPassKind::Accumulate => "FSR3 Accumulate",
        }
    }

    fn spirv(&self) -> &'static [u8] {
        match self {
            FsrPassKind::PrepareInputs => PREPARE_INPUTS,
            FsrPassKind::LumaPyramid => LUMA_PYRAMID,
            FsrPassKind::ShadingChangePyramid => SHADING_CHANGE_PYRAMID,
            FsrPassKind::ShadingChange => SHADING_CHANGE,
            FsrPassKind::PrepareReactivity => PREPARE_REACTIVITY,
            FsrPassKind::LumaInstability => LUMA_INSTABILITY,
            FsrPassKind::Accumulate => ACCUMULATE,
        }
    }

    /// The pass's bindings, in the order of the `FSR3UPSCALER_BIND_*` numbers in its `.glsl` file.
    pub fn resources(&self) -> Vec<ResourceAccess> {
        use crate::resources::AccessType::*;
        use crate::resources::FsrResourceName::*;

        let access = |name, access_type| ResourceAccess {
            name,
            access_type,
            mip: None,
        };
        let spd_mips = |levels: std::ops::Range<u32>| {
            levels.map(|mip| ResourceAccess {
                name: SpdMips,
                access_type: Uav,
                mip: Some(mip),
            })
        };

        let mut ret = match self {
            // With render-resolution motion vectors, binding 2 is the dilated motion vectors
            // (`FSR3UPSCALER_BIND_SRV_DILATED_MOTION_VECTORS` under `FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS`
            // in the accumulate pass's GLSL). wgpu-ffx 0.1.0 bound the input vectors here: its PR #28.
            FsrPassKind::Accumulate => vec![
                access(InputExposure, Srv),
                access(DilatedReactiveMasks, Srv),
                access(DilatedMotionVectors, Srv),
                access(InternalUpscaledPrevious, Srv),
                access(Lanczos2Lut, Srv),
                access(FarthestDepthMip1, Srv),
                access(Luma, Srv),
                access(LumaInstability, Srv),
                access(InputColor, Srv),
                access(InternalUpscaledCurrent, Uav),
                access(OutputColor, Uav),
                access(NewLocks, Uav),
                access(Constants, Srv),
            ],
            FsrPassKind::LumaInstability => vec![
                access(InputExposure, Srv),
                access(DilatedReactiveMasks, Srv),
                access(DilatedMotionVectors, Srv),
                access(FrameInfo, Srv),
                access(LumaHistoryPrevious, Srv),
                access(FarthestDepthMip1, Srv),
                access(Luma, Srv),
                access(LumaHistoryCurrent, Uav),
                access(LumaInstability, Uav),
                access(Constants, Srv),
            ],
            FsrPassKind::LumaPyramid => {
                let mut v = vec![
                    access(Luma, Srv),
                    access(FarthestDepth, Srv),
                    access(SpdAtomicCount, Uav),
                    access(FrameInfo, Uav),
                ];
                v.extend(spd_mips(0..6));
                v.extend([access(FarthestDepthMip1, Uav), access(Constants, Srv)]);
                v
            }
            FsrPassKind::PrepareInputs => vec![
                access(InputMotionVectors, Srv),
                access(InputDepth, Srv),
                access(InputColor, Srv),
                access(DilatedMotionVectors, Uav),
                access(DilatedDepth, Uav),
                access(ReconstructedPreviousDepth, Uav),
                access(FarthestDepth, Uav),
                access(Luma, Uav),
                access(Constants, Srv),
            ],
            FsrPassKind::PrepareReactivity => vec![
                access(ReconstructedPreviousDepth, Srv),
                access(DilatedMotionVectors, Srv),
                access(DilatedDepth, Srv),
                access(InputReactiveMask, Srv),
                access(InputTransparencyAndComposition, Srv),
                access(AccumulationPrevious, Srv),
                access(ShadingChange, Srv),
                access(Luma, Srv),
                access(InputExposure, Srv),
                access(DilatedReactiveMasks, Uav),
                access(NewLocks, Uav),
                access(AccumulationCurrent, Uav),
                access(Constants, Srv),
            ],
            FsrPassKind::ShadingChange => vec![
                access(SpdMips, Srv),
                access(ShadingChange, Uav),
                access(Constants, Srv),
            ],
            FsrPassKind::ShadingChangePyramid => {
                let mut v = vec![
                    access(Luma, Srv),
                    access(PreviousLuma, Srv),
                    access(DilatedMotionVectors, Srv),
                    access(InputExposure, Srv),
                    access(SpdAtomicCount, Uav),
                ];
                v.extend(spd_mips(0..6));
                v.push(access(Constants, Srv));
                v
            }
        };

        // All passes bind the samplers at the end
        ret.push(access(SamplerPointClamp, Srv));
        ret.push(access(SamplerLinearClamp, Srv));

        ret
    }
}
