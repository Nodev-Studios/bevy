// Packs nrd_wgpu's input contract into the textures ReBLUR reads.
//
// The encodings belong to NRD: they are the functions of the SDK's `NRD.hlsli`, included here from the fetched
// SDK when `build.rs` compiles this file to SPIR-V. Nothing of that header is copied into this repository.
#include "NRD.hlsli"

struct Params {
    // ReBLUR's hit distance parameters in xyz, the denoising range in w.
    float4 hit_distance_and_range;
    uint2 size;
    uint2 padding;
};

[[vk::binding(0, 0)]] ConstantBuffer<Params> params;
// rgb: demodulated radiance; a: distance to the first hit of the lobe's ray, 0 where the lobe was not sampled.
[[vk::binding(1, 0)]] Texture2D<float4> contract_diffuse;
[[vk::binding(2, 0)]] Texture2D<float4> contract_specular;
// xyz: world-space normal; w: perceptual roughness.
[[vk::binding(3, 0)]] Texture2D<float4> contract_normal_roughness;
// Linear view-space depth of either sign; 0 or a non-finite value where there is no surface.
[[vk::binding(4, 0)]] Texture2D<float4> contract_view_depth;
// rg: screen-space motion in uv units.
[[vk::binding(5, 0)]] Texture2D<float4> contract_motion_vectors;
[[vk::binding(6, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> packed_diffuse;
[[vk::binding(7, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> packed_specular;
[[vk::binding(8, 0)]] [[vk::image_format("rgb10a2")]] RWTexture2D<float4> packed_normal_roughness;
[[vk::binding(9, 0)]] [[vk::image_format("r32f")]] RWTexture2D<float> packed_view_z;
// ReBLUR binds its motion vector input as a storage image in one pass, so it gets a copy this crate owns rather
// than the caller's texture.
[[vk::binding(10, 0)]] [[vk::image_format("rg16f")]] RWTexture2D<float2> packed_motion_vectors;

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 pixel = id.xy;
    if (any(pixel >= params.size)) {
        return;
    }

    float3 hit_distance_params = params.hit_distance_and_range.xyz;
    float range = params.hit_distance_and_range.w;

    packed_motion_vectors[pixel] = contract_motion_vectors[pixel].xy;

    float depth = abs(contract_view_depth[pixel].x);
    bool surface = !isnan(depth) && depth > 0.0 && depth < range;
    if (!surface) {
        // NRD skips every pixel whose view depth is beyond the denoising range.
        packed_view_z[pixel] = 2.0 * range;
        packed_normal_roughness[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        packed_diffuse[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        packed_specular[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float4 normal_roughness = contract_normal_roughness[pixel];
    float roughness = saturate(normal_roughness.w);
    float4 diffuse = contract_diffuse[pixel];
    float4 specular = contract_specular[pixel];

    // The diffuse lobe is normalized as a fully rough one.
    float diffuse_hit_distance = REBLUR_FrontEnd_GetNormHitDist(diffuse.a, depth, hit_distance_params, 1.0);
    float specular_hit_distance = REBLUR_FrontEnd_GetNormHitDist(specular.a, depth, hit_distance_params, roughness);

    packed_view_z[pixel] = depth;
    packed_normal_roughness[pixel] = NRD_FrontEnd_PackNormalAndRoughness(normal_roughness.xyz, roughness, 0.0);
    packed_diffuse[pixel] = REBLUR_FrontEnd_PackRadianceAndNormHitDist(diffuse.rgb, diffuse_hit_distance, true);
    packed_specular[pixel] = REBLUR_FrontEnd_PackRadianceAndNormHitDist(specular.rgb, specular_hit_distance, true);
}
