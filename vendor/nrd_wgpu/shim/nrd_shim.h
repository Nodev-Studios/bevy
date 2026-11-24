// The C interface nrd_wgpu's Rust side talks to. NRD's own API is declared `extern "C"` but takes C++
// references and returns pointers into C++ structs, so Rust cannot call it directly; this shim wraps it.
//
// Everything here is nrd_wgpu's own: the struct layouts, the names and the numeric values of the enums are
// chosen by this crate and mirrored by hand in `src/ffi.rs`. NRD's types appear only in `nrd_shim.cpp`, which is
// compiled against the SDK's headers at build time.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NrdwInstance NrdwInstance;

// Which denoiser an instance runs.
enum {
    NRDW_DENOISER_REBLUR_DIFFUSE_SPECULAR = 0,
};

// Texture formats, as a number of channels, bits per channel and numeric kind. `kind` 0 means that the format has
// no equivalent on our side.
enum {
    NRDW_KIND_UNSUPPORTED = 0,
    NRDW_KIND_UNORM = 1,
    NRDW_KIND_SNORM = 2,
    NRDW_KIND_UINT = 3,
    NRDW_KIND_SINT = 4,
    NRDW_KIND_FLOAT = 5,
    // 10 bits for each of three channels and 2 for the fourth, unsigned normalized.
    NRDW_KIND_PACKED_10_10_10_2_UNORM = 6,
    // 11, 11 and 10 bit unsigned floats.
    NRDW_KIND_PACKED_11_11_10_UFLOAT = 7,
};

typedef struct {
    uint32_t channels;
    uint32_t bits;
    uint32_t kind;
} NrdwFormat;

// What a dispatch binds at one of its texture slots.
enum {
    NRDW_RESOURCE_UNSUPPORTED = 0,
    NRDW_RESOURCE_MOTION_VECTORS = 1,
    NRDW_RESOURCE_NORMAL_ROUGHNESS = 2,
    NRDW_RESOURCE_VIEW_Z = 3,
    NRDW_RESOURCE_IN_DIFFUSE = 4,
    NRDW_RESOURCE_IN_SPECULAR = 5,
    NRDW_RESOURCE_OUT_DIFFUSE = 6,
    NRDW_RESOURCE_OUT_SPECULAR = 7,
    NRDW_RESOURCE_TRANSIENT_POOL = 8,
    NRDW_RESOURCE_PERMANENT_POOL = 9,
};

typedef struct {
    uint32_t version_major;
    uint32_t version_minor;
    uint32_t version_build;
    // Descriptor set of the constant buffer and the samplers, and descriptor set of the textures.
    uint32_t constants_and_samplers_set;
    uint32_t resources_set;
    // Binding numbers inside those sets, after the offsets NRD's SPIR-V was compiled with.
    uint32_t first_sampler_binding;
    uint32_t constants_binding;
    uint32_t first_texture_binding;
    uint32_t first_storage_binding;
    uint32_t constants_max_size;
    uint32_t sampler_count;
    uint32_t pipeline_count;
    uint32_t permanent_pool_size;
    uint32_t transient_pool_size;
    // 1 when the library packs normal and roughness into 10:10:10:2 unsigned normalized texels.
    uint32_t normal_roughness_is_10_10_10_2;
    const char* entry_point;
} NrdwInfo;

typedef struct {
    const uint32_t* spirv;
    uint64_t spirv_bytes;
    uint32_t texture_count;
    uint32_t storage_count;
    // "file|macro=value|..." as NRD names the shader permutation.
    const char* name;
} NrdwPipeline;

typedef struct {
    NrdwFormat format;
    uint32_t downsample_factor;
} NrdwPoolTexture;

typedef struct {
    float view_to_clip[16];
    float view_to_clip_previous[16];
    float world_to_view[16];
    float world_to_view_previous[16];
    float motion_vector_scale[2];
    float jitter[2];
    float jitter_previous[2];
    uint32_t width;
    uint32_t height;
    float denoising_range;
    uint32_t frame_index;
    // 0: keep accumulating; 1: drop the history; 2: drop the history and clear NRD's textures.
    uint32_t reset;
} NrdwFrame;

typedef struct {
    float hit_distance_a;
    float hit_distance_b;
    float hit_distance_c;
    uint32_t max_accumulated_frames;
    uint32_t max_fast_accumulated_frames;
    uint32_t max_stabilized_frames;
    float diffuse_prepass_blur_radius;
    float specular_prepass_blur_radius;
    float min_blur_radius;
    float max_blur_radius;
    // 0: off; 3: a 3x3 neighbourhood; 5: a 5x5 neighbourhood.
    uint32_t hit_distance_reconstruction;
    uint32_t anti_firefly;
} NrdwReblurSettings;

typedef struct {
    const char* name;
    uint32_t pipeline;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t resource_count;
    const uint8_t* constants;
    uint32_t constants_size;
} NrdwDispatch;

typedef struct {
    uint32_t kind;
    uint32_t is_storage;
    uint32_t pool_index;
} NrdwResource;

// Returns null when NRD refuses the denoiser.
NrdwInstance* nrdw_create(uint32_t denoiser, NrdwInfo* info);
void nrdw_destroy(NrdwInstance* instance);

// 1 when sampler `index` filters linearly, 0 when it takes the nearest texel. Both clamp to the edge.
uint32_t nrdw_sampler_is_linear(const NrdwInstance* instance, uint32_t index);
void nrdw_pipeline(const NrdwInstance* instance, uint32_t index, NrdwPipeline* pipeline);
void nrdw_pool_texture(const NrdwInstance* instance, uint32_t permanent, uint32_t index, NrdwPoolTexture* texture);

// NRD's own defaults for ReBLUR.
void nrdw_reblur_defaults(NrdwReblurSettings* settings);
// Both return 1 on success.
uint32_t nrdw_set_reblur_settings(NrdwInstance* instance, const NrdwReblurSettings* settings);
uint32_t nrdw_set_frame(NrdwInstance* instance, const NrdwFrame* frame);

// Asks NRD for this frame's dispatches and returns how many there are; -1 on failure. The list is valid until the
// next call.
int32_t nrdw_begin_dispatches(NrdwInstance* instance);
void nrdw_dispatch(const NrdwInstance* instance, uint32_t index, NrdwDispatch* dispatch);
void nrdw_dispatch_resource(const NrdwInstance* instance, uint32_t dispatch, uint32_t index, NrdwResource* resource);

#ifdef __cplusplus
}
#endif
