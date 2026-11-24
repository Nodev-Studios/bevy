// Wraps NRD's C++ API in the plain C interface of `nrd_shim.h`. Compiled by `build.rs` against the headers of the
// NRD SDK that `tools/fetch-sdks.sh nrd` fetched; none of those headers is part of this repository.
#include "nrd_shim.h"

#include "NRD.h"

#include <cstring>

namespace {

// The single denoiser of an instance is addressed by this identifier.
const nrd::Identifier IDENTIFIER = 1;

NrdwFormat convert_format(nrd::Format format) {
    switch (format) {
        case nrd::Format::R8_UNORM: return {1, 8, NRDW_KIND_UNORM};
        case nrd::Format::R8_SNORM: return {1, 8, NRDW_KIND_SNORM};
        case nrd::Format::R8_UINT: return {1, 8, NRDW_KIND_UINT};
        case nrd::Format::R8_SINT: return {1, 8, NRDW_KIND_SINT};
        case nrd::Format::RG8_UNORM: return {2, 8, NRDW_KIND_UNORM};
        case nrd::Format::RG8_SNORM: return {2, 8, NRDW_KIND_SNORM};
        case nrd::Format::RG8_UINT: return {2, 8, NRDW_KIND_UINT};
        case nrd::Format::RG8_SINT: return {2, 8, NRDW_KIND_SINT};
        case nrd::Format::RGBA8_UNORM: return {4, 8, NRDW_KIND_UNORM};
        case nrd::Format::RGBA8_SNORM: return {4, 8, NRDW_KIND_SNORM};
        case nrd::Format::RGBA8_UINT: return {4, 8, NRDW_KIND_UINT};
        case nrd::Format::RGBA8_SINT: return {4, 8, NRDW_KIND_SINT};
        case nrd::Format::R16_UNORM: return {1, 16, NRDW_KIND_UNORM};
        case nrd::Format::R16_SNORM: return {1, 16, NRDW_KIND_SNORM};
        case nrd::Format::R16_UINT: return {1, 16, NRDW_KIND_UINT};
        case nrd::Format::R16_SINT: return {1, 16, NRDW_KIND_SINT};
        case nrd::Format::R16_SFLOAT: return {1, 16, NRDW_KIND_FLOAT};
        case nrd::Format::RG16_UNORM: return {2, 16, NRDW_KIND_UNORM};
        case nrd::Format::RG16_SNORM: return {2, 16, NRDW_KIND_SNORM};
        case nrd::Format::RG16_UINT: return {2, 16, NRDW_KIND_UINT};
        case nrd::Format::RG16_SINT: return {2, 16, NRDW_KIND_SINT};
        case nrd::Format::RG16_SFLOAT: return {2, 16, NRDW_KIND_FLOAT};
        case nrd::Format::RGBA16_UNORM: return {4, 16, NRDW_KIND_UNORM};
        case nrd::Format::RGBA16_SNORM: return {4, 16, NRDW_KIND_SNORM};
        case nrd::Format::RGBA16_UINT: return {4, 16, NRDW_KIND_UINT};
        case nrd::Format::RGBA16_SINT: return {4, 16, NRDW_KIND_SINT};
        case nrd::Format::RGBA16_SFLOAT: return {4, 16, NRDW_KIND_FLOAT};
        case nrd::Format::R32_UINT: return {1, 32, NRDW_KIND_UINT};
        case nrd::Format::R32_SINT: return {1, 32, NRDW_KIND_SINT};
        case nrd::Format::R32_SFLOAT: return {1, 32, NRDW_KIND_FLOAT};
        case nrd::Format::RG32_UINT: return {2, 32, NRDW_KIND_UINT};
        case nrd::Format::RG32_SINT: return {2, 32, NRDW_KIND_SINT};
        case nrd::Format::RG32_SFLOAT: return {2, 32, NRDW_KIND_FLOAT};
        case nrd::Format::RGBA32_UINT: return {4, 32, NRDW_KIND_UINT};
        case nrd::Format::RGBA32_SINT: return {4, 32, NRDW_KIND_SINT};
        case nrd::Format::RGBA32_SFLOAT: return {4, 32, NRDW_KIND_FLOAT};
        case nrd::Format::R10_G10_B10_A2_UNORM: return {4, 10, NRDW_KIND_PACKED_10_10_10_2_UNORM};
        case nrd::Format::R11_G11_B10_UFLOAT: return {3, 11, NRDW_KIND_PACKED_11_11_10_UFLOAT};
        default: return {0, 0, NRDW_KIND_UNSUPPORTED};
    }
}

uint32_t convert_resource(nrd::ResourceType type) {
    switch (type) {
        case nrd::ResourceType::IN_MV: return NRDW_RESOURCE_MOTION_VECTORS;
        case nrd::ResourceType::IN_NORMAL_ROUGHNESS: return NRDW_RESOURCE_NORMAL_ROUGHNESS;
        case nrd::ResourceType::IN_VIEWZ: return NRDW_RESOURCE_VIEW_Z;
        case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST: return NRDW_RESOURCE_IN_DIFFUSE;
        case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST: return NRDW_RESOURCE_IN_SPECULAR;
        case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST: return NRDW_RESOURCE_OUT_DIFFUSE;
        case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST: return NRDW_RESOURCE_OUT_SPECULAR;
        case nrd::ResourceType::TRANSIENT_POOL: return NRDW_RESOURCE_TRANSIENT_POOL;
        case nrd::ResourceType::PERMANENT_POOL: return NRDW_RESOURCE_PERMANENT_POOL;
        default: return NRDW_RESOURCE_UNSUPPORTED;
    }
}

}

struct NrdwInstance {
    nrd::Instance* instance;
    const nrd::InstanceDesc* desc;
    const nrd::DispatchDesc* dispatches;
    uint32_t dispatch_count;
};

NrdwInstance* nrdw_create(uint32_t denoiser, NrdwInfo* info) {
    nrd::DenoiserDesc denoiser_desc = {};
    denoiser_desc.identifier = IDENTIFIER;
    switch (denoiser) {
        case NRDW_DENOISER_REBLUR_DIFFUSE_SPECULAR:
            denoiser_desc.denoiser = nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR;
            break;
        default:
            return nullptr;
    }

    nrd::InstanceCreationDesc creation_desc = {};
    creation_desc.denoisers = &denoiser_desc;
    creation_desc.denoisersNum = 1;

    nrd::Instance* instance = nullptr;
    if (nrd::CreateInstance(creation_desc, instance) != nrd::Result::SUCCESS || instance == nullptr) {
        return nullptr;
    }

    const nrd::LibraryDesc* library = nrd::GetLibraryDesc();
    const nrd::InstanceDesc* desc = nrd::GetInstanceDesc(*instance);
    const nrd::SPIRVBindingOffsets& offsets = library->spirvBindingOffsets;

    info->version_major = library->versionMajor;
    info->version_minor = library->versionMinor;
    info->version_build = library->versionBuild;
    info->constants_and_samplers_set = desc->constantBufferAndSamplersSpaceIndex;
    info->resources_set = desc->resourcesSpaceIndex;
    info->first_sampler_binding = offsets.samplerOffset + desc->samplersBaseRegisterIndex;
    info->constants_binding = offsets.constantBufferOffset + desc->constantBufferRegisterIndex;
    info->first_texture_binding = offsets.textureOffset + desc->resourcesBaseRegisterIndex;
    info->first_storage_binding = offsets.storageTextureAndBufferOffset + desc->resourcesBaseRegisterIndex;
    info->constants_max_size = desc->constantBufferMaxDataSize;
    info->sampler_count = desc->samplersNum;
    info->pipeline_count = desc->pipelinesNum;
    info->permanent_pool_size = desc->permanentPoolSize;
    info->transient_pool_size = desc->transientPoolSize;
    info->normal_roughness_is_10_10_10_2 = library->normalEncoding == nrd::NormalEncoding::R10_G10_B10_A2_UNORM;
    info->entry_point = desc->shaderEntryPoint;

    return new NrdwInstance{instance, desc, nullptr, 0};
}

void nrdw_destroy(NrdwInstance* instance) {
    nrd::DestroyInstance(*instance->instance);
    delete instance;
}

uint32_t nrdw_sampler_is_linear(const NrdwInstance* instance, uint32_t index) {
    return instance->desc->samplers[index] == nrd::Sampler::LINEAR_CLAMP;
}

void nrdw_pipeline(const NrdwInstance* instance, uint32_t index, NrdwPipeline* pipeline) {
    const nrd::PipelineDesc& desc = instance->desc->pipelines[index];
    pipeline->spirv = static_cast<const uint32_t*>(desc.computeShaderSPIRV.bytecode);
    pipeline->spirv_bytes = desc.computeShaderSPIRV.size;
    pipeline->texture_count = 0;
    pipeline->storage_count = 0;
    for (uint32_t i = 0; i < desc.resourceRangesNum; i++) {
        const nrd::ResourceRangeDesc& range = desc.resourceRanges[i];
        if (range.descriptorType == nrd::DescriptorType::TEXTURE) {
            pipeline->texture_count = range.descriptorsNum;
        } else {
            pipeline->storage_count = range.descriptorsNum;
        }
    }
    pipeline->name = desc.shaderIdentifier;
}

void nrdw_pool_texture(const NrdwInstance* instance, uint32_t permanent, uint32_t index, NrdwPoolTexture* texture) {
    const nrd::TextureDesc& desc = permanent ? instance->desc->permanentPool[index] : instance->desc->transientPool[index];
    texture->format = convert_format(desc.format);
    texture->downsample_factor = desc.downsampleFactor;
}

void nrdw_reblur_defaults(NrdwReblurSettings* settings) {
    const nrd::ReblurSettings defaults = {};
    settings->hit_distance_a = defaults.hitDistanceParameters.A;
    settings->hit_distance_b = defaults.hitDistanceParameters.B;
    settings->hit_distance_c = defaults.hitDistanceParameters.C;
    settings->max_accumulated_frames = defaults.maxAccumulatedFrameNum;
    settings->max_fast_accumulated_frames = defaults.maxFastAccumulatedFrameNum;
    settings->max_stabilized_frames = defaults.maxStabilizedFrameNum;
    settings->diffuse_prepass_blur_radius = defaults.diffusePrepassBlurRadius;
    settings->specular_prepass_blur_radius = defaults.specularPrepassBlurRadius;
    settings->min_blur_radius = defaults.minBlurRadius;
    settings->max_blur_radius = defaults.maxBlurRadius;
    switch (defaults.hitDistanceReconstructionMode) {
        case nrd::HitDistanceReconstructionMode::AREA_3X3: settings->hit_distance_reconstruction = 3; break;
        case nrd::HitDistanceReconstructionMode::AREA_5X5: settings->hit_distance_reconstruction = 5; break;
        default: settings->hit_distance_reconstruction = 0; break;
    }
    settings->anti_firefly = defaults.enableAntiFirefly;
}

uint32_t nrdw_set_reblur_settings(NrdwInstance* instance, const NrdwReblurSettings* settings) {
    nrd::ReblurSettings reblur = {};
    reblur.hitDistanceParameters.A = settings->hit_distance_a;
    reblur.hitDistanceParameters.B = settings->hit_distance_b;
    reblur.hitDistanceParameters.C = settings->hit_distance_c;
    reblur.maxAccumulatedFrameNum = settings->max_accumulated_frames;
    reblur.maxFastAccumulatedFrameNum = settings->max_fast_accumulated_frames;
    reblur.maxStabilizedFrameNum = settings->max_stabilized_frames;
    reblur.diffusePrepassBlurRadius = settings->diffuse_prepass_blur_radius;
    reblur.specularPrepassBlurRadius = settings->specular_prepass_blur_radius;
    reblur.minBlurRadius = settings->min_blur_radius;
    reblur.maxBlurRadius = settings->max_blur_radius;
    switch (settings->hit_distance_reconstruction) {
        case 3: reblur.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3; break;
        case 5: reblur.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_5X5; break;
        default: reblur.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::OFF; break;
    }
    reblur.enableAntiFirefly = settings->anti_firefly != 0;
    return nrd::SetDenoiserSettings(*instance->instance, IDENTIFIER, &reblur) == nrd::Result::SUCCESS;
}

uint32_t nrdw_set_frame(NrdwInstance* instance, const NrdwFrame* frame) {
    nrd::CommonSettings common = {};
    std::memcpy(common.viewToClipMatrix, frame->view_to_clip, sizeof(common.viewToClipMatrix));
    std::memcpy(common.viewToClipMatrixPrev, frame->view_to_clip_previous, sizeof(common.viewToClipMatrixPrev));
    std::memcpy(common.worldToViewMatrix, frame->world_to_view, sizeof(common.worldToViewMatrix));
    std::memcpy(common.worldToViewMatrixPrev, frame->world_to_view_previous, sizeof(common.worldToViewMatrixPrev));
    common.motionVectorScale[0] = frame->motion_vector_scale[0];
    common.motionVectorScale[1] = frame->motion_vector_scale[1];
    common.motionVectorScale[2] = 0.0f;
    common.cameraJitter[0] = frame->jitter[0];
    common.cameraJitter[1] = frame->jitter[1];
    common.cameraJitterPrev[0] = frame->jitter_previous[0];
    common.cameraJitterPrev[1] = frame->jitter_previous[1];
    const uint16_t width = static_cast<uint16_t>(frame->width);
    const uint16_t height = static_cast<uint16_t>(frame->height);
    common.resourceSize[0] = common.resourceSizePrev[0] = common.rectSize[0] = common.rectSizePrev[0] = width;
    common.resourceSize[1] = common.resourceSizePrev[1] = common.rectSize[1] = common.rectSizePrev[1] = height;
    common.denoisingRange = frame->denoising_range;
    common.frameIndex = frame->frame_index;
    switch (frame->reset) {
        case 1: common.accumulationMode = nrd::AccumulationMode::RESTART; break;
        case 2: common.accumulationMode = nrd::AccumulationMode::CLEAR_AND_RESTART; break;
        default: common.accumulationMode = nrd::AccumulationMode::CONTINUE; break;
    }
    return nrd::SetCommonSettings(*instance->instance, common) == nrd::Result::SUCCESS;
}

int32_t nrdw_begin_dispatches(NrdwInstance* instance) {
    const nrd::Identifier identifier = IDENTIFIER;
    instance->dispatches = nullptr;
    instance->dispatch_count = 0;
    const nrd::Result result =
        nrd::GetComputeDispatches(*instance->instance, &identifier, 1, instance->dispatches, instance->dispatch_count);
    if (result != nrd::Result::SUCCESS) {
        instance->dispatch_count = 0;
        return -1;
    }
    return static_cast<int32_t>(instance->dispatch_count);
}

void nrdw_dispatch(const NrdwInstance* instance, uint32_t index, NrdwDispatch* dispatch) {
    const nrd::DispatchDesc& desc = instance->dispatches[index];
    dispatch->name = desc.name;
    dispatch->pipeline = desc.pipelineIndex;
    dispatch->grid_width = desc.gridWidth;
    dispatch->grid_height = desc.gridHeight;
    dispatch->resource_count = desc.resourcesNum;
    dispatch->constants = desc.constantBufferData;
    dispatch->constants_size = desc.constantBufferDataSize;
}

void nrdw_dispatch_resource(const NrdwInstance* instance, uint32_t dispatch, uint32_t index, NrdwResource* resource) {
    const nrd::ResourceDesc& desc = instance->dispatches[dispatch].resources[index];
    resource->kind = convert_resource(desc.type);
    resource->is_storage = desc.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;
    resource->pool_index = desc.indexInPool;
}
