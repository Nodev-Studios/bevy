// Bakes the light arriving at voxel centres into Bevy's ambient cubes.
//
// Each thread follows `samples / batches` paths of one voxel. A path starts at the voxel centre in a direction of
// the voxel's rotated Fibonacci set (uniform over the sphere) and then runs the reference path tracer's loop
// (`bevy_solari`'s `pathtracer.wesl`): emission where it hits, next-event estimation of one light per surface,
// BRDF sampling, Russian roulette, no bounce cap, no exposure. Nothing is sampled at the voxel itself, so the
// direct light of analytic lights (which no ray can hit) never enters the volume: the rasterizer draws it.
//
// Side `a` of the ambient cube is E/pi for a surface facing `a`: (4 / N) * sum over paths of max(0, w . a) * L.
//
// The light picked for next-event estimation is chosen by its index in a canonical order (sorted by a key that
// does not depend on when Solari added it), so the same seed picks the same light in every run, and each
// light's share is tallied under that index.

import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance;
import bevy_pbr::render::utils::{rand_f, rand_u, rand_range_u};
import bevy_solari::scene::bindings::{trace_ray, resolve_ray_hit_full, light_sources, light_source_kind, emissive_mesh_triangle_count, LIGHT_SOURCE_KIND_EMISSIVE_MESH, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD};
import bevy_solari::scene::sampling::{LightSample, resolve_light_sample, calculate_resolved_light_contribution, trace_visibility, random_emissive_light_pdf, power_heuristic, sample_environment_map_light};
import bevy_solari::scene::brdf::{evaluate_brdf, evaluate_and_sample_brdf, brdf_pdf, F_AB};

enable wgpu_ray_query;

struct BakeParams {
    voxel_count: u32,
    voxel_global_offset: u32,
    samples: u32,
    batches: u32,
    light_count: u32,
    mode: u32,
    seed: u32,
    _padding: u32,
}

struct VoxelInput {
    position: vec4<f32>,
    // The columns of the voxel's rotation.
    rotation_x: vec4<f32>,
    rotation_y: vec4<f32>,
    rotation_z: vec4<f32>,
}

const MODE_FULL = 0u;
// Gates 1 and 2: only the emission of the first surface hit, nothing reflected.
const MODE_FIRST_HIT_EMISSION = 1u;

@group(1) @binding(0) var<uniform> params: BakeParams;
@group(1) @binding(1) var<storage> voxels: array<VoxelInput>;
@group(1) @binding(2) var<storage> directions: array<vec4<f32>>;
@group(1) @binding(3) var<storage> canonical_lights: array<u32>;
@group(1) @binding(4) var<storage, read_write> faces: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> tallies: array<f32>;
@group(1) @binding(6) var<storage, read_write> first_hits: array<vec2<u32>>;

// Same as `directions.rs`'s `pcg_hash`.
fn pcg_hash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(64, 1, 1)
fn bake(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread = global_id.x;
    let voxel = thread / params.batches;
    if voxel >= params.voxel_count { return; }
    let batch = thread % params.batches;

    let input = voxels[voxel];
    let rotation = mat3x3(input.rotation_x.xyz, input.rotation_y.xyz, input.rotation_z.xyz);
    let origin = input.position.xyz;
    let samples_per_batch = params.samples / params.batches;
    let light_count = params.light_count;
    let tally_base = thread * (light_count + 1u);
    let voxel_seed = pcg_hash(params.seed ^ pcg_hash(params.voxel_global_offset + voxel));

    var sides = array<vec3<f32>, 6>();
    var static_tally = 0.0;
    var back_faces = 0u;
    var misses = 0u;

    for (var k = 0u; k < samples_per_batch; k++) {
        let sample = batch * samples_per_batch + k;
        let direction = normalize(rotation * directions[sample].xyz);
        var rng = pcg_hash(voxel_seed ^ pcg_hash(sample));

        // Weight of this path on each side of the cube, and on all six together (for the tallies).
        let w_positive = 4.0 * max(direction, vec3(0.0));
        let w_negative = 4.0 * max(-direction, vec3(0.0));
        let w_total = 4.0 * (abs(direction.x) + abs(direction.y) + abs(direction.z));

        var radiance = vec3(0.0);
        var ray_origin = origin;
        var ray_direction = direction;
        var ray_t_min = 0.0;
        var throughput = vec3(1.0);
        var p_bounce = 0.0;
        var first = true;
        loop {
            let ray = trace_ray(ray_origin, ray_direction, ray_t_min, RAY_T_MAX, RAY_FLAG_NONE);
            if first {
                if ray.kind == RAY_QUERY_INTERSECTION_NONE {
                    misses += 1u;
                } else if !ray.front_face {
                    back_faces += 1u;
                }
                first = false;
            }

            if ray.kind == RAY_QUERY_INTERSECTION_NONE {
                if params.mode == MODE_FULL {
                    let environment = throughput * sample_environment_map_light(ray_direction);
                    radiance += environment;
                    static_tally += luminance(environment) * w_total;
                }
                break;
            }

            let wo = -ray_direction;
            let ray_hit = resolve_ray_hit_full(ray, wo);
            let NdotV = max(dot(ray_hit.world_normal, wo), 0.0001);
            let F_ab = F_AB(ray_hit.material.perceptual_roughness, NdotV);

            // Emission. The first hit has nothing to share it with (no light was sampled at the voxel).
            var mis_weight = 1.0;
            if p_bounce != 0.0 {
                let p_light = random_emissive_light_pdf(ray_hit, ray.t, NdotV);
                mis_weight = power_heuristic(p_bounce, p_light);
            }
            if dot(ray_hit.world_normal, wo) > 0.0 {
                let emission = mis_weight * throughput * ray_hit.material.emissive;
                radiance += emission;
                static_tally += luminance(emission) * w_total;
            }

            if params.mode == MODE_FIRST_HIT_EMISSION { break; }

            // Next-event estimation of one light, picked uniformly by canonical index
            let is_perfectly_specular = ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999;
            if !is_perfectly_specular && light_count > 0u {
                let canonical = rand_range_u(light_count, &rng);
                let light_id = canonical_lights[canonical];
                let light_source = light_sources[light_id];
                var triangle_id = 0u;
                if light_source_kind(light_source) == LIGHT_SOURCE_KIND_EMISSIVE_MESH {
                    triangle_id = rand_range_u(emissive_mesh_triangle_count(light_source), &rng);
                }
                let seed = rand_u(&rng);
                var resolved = resolve_light_sample(LightSample((light_id << 16u) | triangle_id, seed), light_source);
                resolved.inverse_pdf *= f32(light_count);
                var contribution = calculate_resolved_light_contribution(resolved, ray_hit.world_position, ray_hit.world_normal);
                contribution.radiance *= trace_visibility(ray_hit.world_position, resolved.world_position, RAY_FLAG_NONE);

                mis_weight = 1.0;
                if contribution.brdf_rays_can_hit {
                    let pdf_of_bounce = brdf_pdf(wo, contribution.wi, ray_hit.world_normal, ray_hit.material, F_ab);
                    mis_weight = power_heuristic(1.0 / contribution.inverse_solid_angle_pdf, pdf_of_bounce);
                }
                let brdf = evaluate_brdf(wo, contribution.wi, ray_hit.world_normal, ray_hit.material, F_ab);
                let direct = mis_weight * throughput * contribution.radiance * contribution.inverse_pdf * brdf;
                radiance += direct;
                let tally = luminance(direct) * w_total;
                if light_source_kind(light_source) == LIGHT_SOURCE_KIND_EMISSIVE_MESH {
                    static_tally += tally;
                } else {
                    tallies[tally_base + canonical] += tally;
                }
            }

            // Next bounce
            let next_bounce = evaluate_and_sample_brdf(wo, ray_hit.world_normal, ray_hit.material, F_ab, &rng);
            if next_bounce.pdf == 0.0 { break; }
            ray_direction = next_bounce.wi;
            ray_origin = ray_hit.world_position + (ray_hit.geometric_world_normal * RAY_T_MIN);
            ray_t_min = RAY_T_MIN;
            p_bounce = next_bounce.pdf;
            throughput *= next_bounce.throughput;

            // Russian roulette
            let rr = saturate(luminance(throughput));
            if rand_f(&rng) >= rr { break; }
            throughput /= rr;
        }

        sides[0] += w_positive.x * radiance;
        sides[1] += w_negative.x * radiance;
        sides[2] += w_positive.y * radiance;
        sides[3] += w_negative.y * radiance;
        sides[4] += w_positive.z * radiance;
        sides[5] += w_negative.z * radiance;
    }

    for (var side = 0u; side < 6u; side++) {
        faces[thread * 6u + side] = vec4(sides[side], 0.0);
    }
    tallies[tally_base + light_count] = static_tally;
    first_hits[thread] = vec2(back_faces, misses);
}
