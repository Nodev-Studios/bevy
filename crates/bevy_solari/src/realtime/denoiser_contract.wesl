//! The radiance a denoiser is given: split by the lobe of the primary surface that reflects it,
//! divided by that lobe's reflectance, with the length of the ray traced from the primary surface.
//! `bevy_anti_alias::ray_reconstruction::ViewDenoiserInputTextures` documents the encoding.

import bevy_pbr::render::pbr_functions::{calculate_diffuse_color, calculate_F0};
import package::realtime::bindings::{denoiser_diffuse_radiance, denoiser_specular_radiance, Reservoir, view};
import package::realtime::resolve_dlss_rr_textures::env_brdf_approx2;
import package::scene::bindings::{light_sources, RAY_T_MIN, ResolvedMaterial};
import package::scene::brdf::{evaluate_diffuse_brdf, evaluate_specular_brdf, F_AB};
import package::scene::sampling::{calculate_resolved_light_contribution, isinf, NULL_LIGHT_ID, resolve_light_sample};

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
/// textures, as it parks the sum of both in the view target.
@if(DENOISER_CONTRACT)
fn store_unreused_denoiser_radiance(pixel_id: vec2<u32>, diffuse_radiance: vec3<f32>, specular_radiance: vec3<f32>, hit_distances: vec2<f32>) {
    textureStore(denoiser_diffuse_radiance, pixel_id, vec4(diffuse_radiance, hit_distances.x));
    textureStore(denoiser_specular_radiance, pixel_id, vec4(specular_radiance, hit_distances.y));
}

/// With ReSTIR, adds the sample the shading pass selected to what the initial pass parked, and
/// writes the final value. The hit distances stay those of this pixel's own ray: the distance to a
/// reused sample says nothing about what this pixel's lobes see.
@if(DENOISER_CONTRACT)
fn shade_reservoir_into_denoiser_radiance(
    pixel_id: vec2<u32>,
    reservoir: Reservoir,
    selected_sample_brdf_radiance: vec3<f32>,
    material: ResolvedMaterial,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
) {
    let reused_radiance = selected_sample_brdf_radiance * reservoir.unbiased_contribution_weight;
    let reused_diffuse_radiance = reused_radiance * reservoir_diffuse_fraction(reservoir, material, world_position, world_normal);

    let unreused_diffuse = textureLoad(denoiser_diffuse_radiance, pixel_id);
    let unreused_specular = textureLoad(denoiser_specular_radiance, pixel_id);
    store_denoiser_radiance(
        pixel_id,
        unreused_diffuse.rgb + reused_diffuse_radiance,
        unreused_specular.rgb + reused_radiance - reused_diffuse_radiance,
        vec2(unreused_diffuse.a, unreused_specular.a),
        material,
        world_position,
        world_normal,
    );
}

/// `diffuse_fraction` for the direction a reservoir's sample lights the surface from. The sample is
/// resolved a second time here, rather than carried out of `merge_reservoirs`, to leave that
/// function as it is upstream.
@if(DENOISER_CONTRACT)
fn reservoir_diffuse_fraction(reservoir: Reservoir, material: ResolvedMaterial, world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    var wi: vec3<f32>;
    if reservoir.light_sample.light_id != NULL_LIGHT_ID {
        let resolved = resolve_light_sample(reservoir.light_sample, light_sources[reservoir.light_sample.light_id >> 16u]);
        wi = calculate_resolved_light_contribution(resolved, world_position, world_normal).wi;
    } else if any(reservoir.radiance != vec3(0.0)) {
        wi = normalize(reservoir.sample_point_world_position - (world_position + world_normal * RAY_T_MIN));
    } else {
        return vec3(0.0);
    }

    let wo = normalize(view.world_position - world_position);
    let F_ab = F_AB(material.perceptual_roughness, max(dot(world_normal, wo), 0.0001));
    return diffuse_fraction(wo, wi, world_normal, material, F_ab);
}
