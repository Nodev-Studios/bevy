//! The radiance a denoiser is given: split by the lobe of the primary surface that reflects it,
//! divided by that lobe's reflectance, with the length of the ray traced from the primary surface.
//! `bevy_anti_alias::ray_reconstruction::ViewDenoiserInputTextures` documents the encoding.

import bevy_pbr::render::pbr_functions::{calculate_diffuse_color, calculate_F0};
import package::realtime::bindings::{denoiser_diffuse_radiance, denoiser_specular_radiance, view};
import package::realtime::resolve_dlss_rr_textures::env_brdf_approx2;
import package::scene::bindings::ResolvedMaterial;
import package::scene::brdf::{evaluate_diffuse_brdf, evaluate_specular_brdf, F_AB};
import package::scene::sampling::isinf;

enable wgpu_ray_query;

/// Keep in sync with `bevy_anti_alias::ray_reconstruction::DENOISER_MISS_HIT_DISTANCE`.
const DENOISER_MISS_HIT_DISTANCE = 10000.0;

/// Radiance is divided by a reflectance no smaller than this, and multiplied back by the same
/// value, so that a black surface costs range in the stored texture instead of energy.
// to tune: 0.01 keeps a demodulated value within 100 times the lit color it came from.
const MIN_DEMODULATION_ALBEDO = 0.01;

/// Largest finite value of the 16-bit float textures the radiance is stored in.
const MAX_STORED_RADIANCE = 65504.0;

struct DemodulationAlbedos {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

/// What the radiance of each lobe is divided by before denoising, and multiplied by after.
fn demodulation_albedos(material: ResolvedMaterial, world_normal: vec3<f32>, wo: vec3<f32>) -> DemodulationAlbedos {
    let diffuse_albedo = calculate_diffuse_color(material.base_color, material.metallic, 0.0, 0.0);
    let F0 = calculate_F0(material.base_color, material.metallic, vec3(material.reflectance));
    let specular_albedo = env_brdf_approx2(F0, material.roughness, world_normal, wo);
    return DemodulationAlbedos(
        max(diffuse_albedo, vec3(MIN_DEMODULATION_ALBEDO)),
        max(specular_albedo, vec3(MIN_DEMODULATION_ALBEDO)),
    );
}

/// The diffuse lobe's share of the light a surface reflects from `wi`, per channel.
fn diffuse_fraction(wo: vec3<f32>, wi: vec3<f32>, world_normal: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>) -> vec3<f32> {
    let diffuse = evaluate_diffuse_brdf(wo, wi, world_normal, material, F_ab);
    let specular = evaluate_specular_brdf(wo, wi, world_normal, material, F_ab);
    // Where both are 0 the surface reflects nothing from `wi` and the share does not matter.
    return diffuse / max(diffuse + specular, vec3(1e-20));
}

/// `diffuse_fraction` for a direction that `evaluate_and_sample_brdf` sampled with density `pdf`.
fn sampled_direction_diffuse_fraction(wo: vec3<f32>, wi: vec3<f32>, pdf: f32, world_normal: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>) -> vec3<f32> {
    // A mirror sample carries the mirror reflection alone, whatever the diffuse lobe reflects
    // from that direction.
    if isinf(pdf) { return vec3(0.0); }
    return diffuse_fraction(wo, wi, world_normal, material, F_ab);
}

/// Hit distance of the ray traced from the primary surface, in x for the diffuse lobe and in y for
/// the specular lobe: 0 for the lobe the ray was not sampled from.
fn first_bounce_hit_distances(diffuse_selected: bool, missed: bool, ray_t: f32) -> vec2<f32> {
    var hit_distance = DENOISER_MISS_HIT_DISTANCE;
    if !missed { hit_distance = min(ray_t, DENOISER_MISS_HIT_DISTANCE); }
    return select(vec2(0.0, hit_distance), vec2(hit_distance, 0.0), diffuse_selected);
}

/// Writes the final value of a pixel in both radiance textures.
@if(DENOISER_CONTRACT)
fn store_denoiser_radiance(
    pixel_id: vec2<u32>,
    diffuse_radiance: vec3<f32>,
    specular_radiance: vec3<f32>,
    hit_distances: vec2<f32>,
    material: ResolvedMaterial,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
) {
    let albedos = demodulation_albedos(material, world_normal, normalize(view.world_position - world_position));
    let diffuse = clamp(diffuse_radiance * view.exposure / albedos.diffuse, vec3(0.0), vec3(MAX_STORED_RADIANCE));
    let specular = clamp(specular_radiance * view.exposure / albedos.specular, vec3(0.0), vec3(MAX_STORED_RADIANCE));
    textureStore(denoiser_diffuse_radiance, pixel_id, vec4(diffuse, hit_distances.x));
    textureStore(denoiser_specular_radiance, pixel_id, vec4(specular, hit_distances.y));
}

/// With ReSTIR, the initial pass parks the radiance it shaded without a reservoir in the radiance
/// textures, as it parks the sum of both in the view target. It parks it exposed: unexposed, a
/// highlight brighter than the largest 16-bit float would be clipped, as the view target clips it.
@if(DENOISER_CONTRACT)
fn store_unreused_denoiser_radiance(pixel_id: vec2<u32>, diffuse_radiance: vec3<f32>, specular_radiance: vec3<f32>, hit_distances: vec2<f32>) {
    textureStore(denoiser_diffuse_radiance, pixel_id, vec4(diffuse_radiance * view.exposure, hit_distances.x));
    textureStore(denoiser_specular_radiance, pixel_id, vec4(specular_radiance * view.exposure, hit_distances.y));
}

/// With ReSTIR, adds the radiance of the sample the shading pass selected to what the initial pass
/// parked, and writes the final value. `sample_world_position` is where that sample lights the
/// surface from: a point when w is 1, a direction when w is 0. The hit distances stay those of this
/// pixel's own ray: the distance to a reused sample says nothing about what this pixel's lobes see.
@if(DENOISER_CONTRACT)
fn shade_reservoir_into_denoiser_radiance(
    pixel_id: vec2<u32>,
    reused_radiance: vec3<f32>,
    sample_world_position: vec4<f32>,
    material: ResolvedMaterial,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
) {
    var reused_diffuse_radiance = vec3(0.0);
    // An empty reservoir has no sample position to take a direction from.
    if any(reused_radiance != vec3(0.0)) {
        let wo = normalize(view.world_position - world_position);
        let wi = normalize(sample_world_position.xyz - (sample_world_position.w * world_position));
        let F_ab = F_AB(material.perceptual_roughness, max(dot(world_normal, wo), 0.0001));
        reused_diffuse_radiance = reused_radiance * diffuse_fraction(wo, wi, world_normal, material, F_ab);
    }

    // The parked radiance is exposed; `store_denoiser_radiance` exposes what it is given.
    let unreused_diffuse = textureLoad(denoiser_diffuse_radiance, pixel_id);
    let unreused_specular = textureLoad(denoiser_specular_radiance, pixel_id);
    store_denoiser_radiance(
        pixel_id,
        unreused_diffuse.rgb / view.exposure + reused_diffuse_radiance,
        unreused_specular.rgb / view.exposure + reused_radiance - reused_diffuse_radiance,
        vec2(unreused_diffuse.a, unreused_specular.a),
        material,
        world_position,
        world_normal,
    );
}
