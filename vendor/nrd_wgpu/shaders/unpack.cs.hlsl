// Unpacks ReBLUR's two outputs into nrd_wgpu's output contract.
//
// The decoding belongs to NRD: it is a function of the SDK's `NRD.hlsli`, included here from the fetched SDK
// when `build.rs` compiles this file to SPIR-V. Nothing of that header is copied into this repository.
#include "NRD.hlsli"

struct Params {
    // ReBLUR's hit distance parameters in xyz, the denoising range in w.
    float4 hit_distance_and_range;
    uint2 size;
    uint2 padding;
};

[[vk::binding(0, 0)]] ConstantBuffer<Params> params;
[[vk::binding(1, 0)]] Texture2D<float4> denoised_diffuse;
[[vk::binding(2, 0)]] Texture2D<float4> denoised_specular;
[[vk::binding(3, 0)]] Texture2D<float4> packed_view_z;
// rgb: denoised demodulated radiance; a: the lobe's hit distance normalized to 0..1 (an occlusion term).
[[vk::binding(4, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> contract_diffuse;
[[vk::binding(5, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> contract_specular;

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 pixel = id.xy;
    if (any(pixel >= params.size)) {
        return;
    }

    // NRD leaves pixels beyond the denoising range unwritten: give them a defined value.
    if (packed_view_z[pixel].x >= params.hit_distance_and_range.w) {
        contract_diffuse[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        contract_specular[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    contract_diffuse[pixel] = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(denoised_diffuse[pixel]);
    contract_specular[pixel] = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(denoised_specular[pixel]);
}
