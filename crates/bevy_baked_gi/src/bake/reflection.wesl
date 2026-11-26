// Captures one region's reflection probe: the radiance arriving at the capture origin from each cubemap texel.
//
// Each thread is one texel of level 0 and follows `samples` paths. A path starts at the origin towards the world
// position of a point of the region's box, the point Bevy's parallax correction reads that texel for (see
// `reflection.rs`), jittered within the texel. It then runs the volume kernel's loop (`bake.wesl`, itself the
// reference path tracer's): emission where it hits, next-event estimation of one light per surface picked by
// power / distance², BRDF sampling, Russian roulette, no bounce cap, no exposure, the environment on a miss.
// Unlike the volume, the probe stores what the path sees, so a surface lit directly by a lamp shows lit: that is
// what a glossy floor reflects. The lamps themselves have no geometry, so no ray sees them, and the rasterizer
// still draws their highlights.

import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance;
import bevy_pbr::render::utils::{rand_f, rand_u, rand_range_u};
import bevy_solari::scene::bindings::{trace_ray, resolve_ray_hit_full, light_sources, light_source_kind, emissive_mesh_triangle_count, LIGHT_SOURCE_KIND_EMISSIVE_MESH, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD};
import bevy_solari::scene::sampling::{LightSample, resolve_light_sample, calculate_resolved_light_contribution, trace_visibility, random_emissive_light_pdf, power_heuristic, sample_environment_map_light};
import bevy_solari::scene::brdf::{evaluate_brdf, evaluate_and_sample_brdf, brdf_pdf, F_AB};

enable wgpu_ray_query;

struct ReflectionParams {
    texel_offset: u32,
    texel_count: u32,
    size: u32,
    samples: u32,
    light_count: u32,
    seed: u32,
    probe: u32,
    _padding: u32,
    origin: vec4<f32>,
    // The region box's world-from-unit-box affine transform: three columns and the translation.
    box_x: vec4<f32>,
    box_y: vec4<f32>,
    box_z: vec4<f32>,
    box_translation: vec4<f32>,
}

@group(1) @binding(0) var<uniform> params: ReflectionParams;
@group(1) @binding(1) var<storage> canonical_lights: array<u32>;
// Per canonical light: its position and power, or w < 0 for a light picked at 1 / light_count.
@group(1) @binding(2) var<storage> light_info: array<vec4<f32>>;
// Per texel: the mean radiance, and how many first hits were back faces.
@group(1) @binding(3) var<storage, read_write> texels: array<vec4<f32>>;

// The same constants as `bake.wesl`'s.
const UNIFORM_FLOOR = 0.1;
const MIN_DISTANCE_SQUARED = 0.25;

struct PickedLight {
    canonical: u32,
    probability: f32,
}

fn light_weight(info: vec4<f32>, position: vec3<f32>) -> f32 {
    let offset = info.xyz - position;
    return info.w / max(dot(offset, offset), MIN_DISTANCE_SQUARED);
}

// `bake.wesl`'s light pick.
fn pick_light(position: vec3<f32>, light_count: u32, rng: ptr<function, u32>) -> PickedLight {
    var total = 0.0;
    var positioned = 0u;
    for (var i = 0u; i < light_count; i++) {
        let info = light_info[i];
        if info.w >= 0.0 {
            total += light_weight(info, position);
            positioned += 1u;
        }
    }
    let uniform_probability = 1.0 / f32(light_count);
    let positioned_share = f32(positioned) * uniform_probability;
    let u = rand_f(rng);
    var cumulative = 0.0;
    var picked = PickedLight(0u, uniform_probability);
    for (var i = 0u; i < light_count; i++) {
        let info = light_info[i];
        var probability = uniform_probability;
        if info.w >= 0.0 {
            var importance = 1.0 / f32(positioned);
            if total > 0.0 {
                importance = (1.0 - UNIFORM_FLOOR) * light_weight(info, position) / total + UNIFORM_FLOOR / f32(positioned);
            }
            probability = positioned_share * importance;
        }
        cumulative += probability;
        picked = PickedLight(i, probability);
        if u < cumulative { break; }
    }
    return picked;
}

fn pcg_hash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// `reflection.rs`'s `face_direction`.
fn face_direction(face: u32, s: f32, t: f32) -> vec3<f32> {
    let sc = 2.0 * s - 1.0;
    let tc = 2.0 * t - 1.0;
    switch face {
        case 0u: { return vec3(1.0, -tc, -sc); }
        case 1u: { return vec3(-1.0, -tc, sc); }
        case 2u: { return vec3(sc, 1.0, tc); }
        case 3u: { return vec3(sc, -1.0, -tc); }
        case 4u: { return vec3(sc, -tc, 1.0); }
        default: { return vec3(-sc, -tc, -1.0); }
    }
}

// The radiance a path from `origin` towards `direction` brings back.
fn trace_path(origin: vec3<f32>, direction: vec3<f32>, rng: ptr<function, u32>, back_face: ptr<function, bool>) -> vec3<f32> {
    let light_count = params.light_count;
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
            *back_face = ray.kind != RAY_QUERY_INTERSECTION_NONE && !ray.front_face;
            first = false;
        }
        if ray.kind == RAY_QUERY_INTERSECTION_NONE {
            radiance += throughput * sample_environment_map_light(ray_direction);
            break;
        }

        let wo = -ray_direction;
        let ray_hit = resolve_ray_hit_full(ray, wo);
        let NdotV = max(dot(ray_hit.world_normal, wo), 0.0001);
        let F_ab = F_AB(ray_hit.material.perceptual_roughness, NdotV);

        var mis_weight = 1.0;
        if p_bounce != 0.0 {
            let p_light = random_emissive_light_pdf(ray_hit, ray.t, NdotV);
            mis_weight = power_heuristic(p_bounce, p_light);
        }
        if dot(ray_hit.world_normal, wo) > 0.0 {
            radiance += mis_weight * throughput * ray_hit.material.emissive;
        }

        let is_perfectly_specular = ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999;
        if !is_perfectly_specular && light_count > 0u {
            let picked = pick_light(ray_hit.world_position, light_count, rng);
            let light_id = canonical_lights[picked.canonical];
            let light_source = light_sources[light_id];
            var triangle_id = 0u;
            if light_source_kind(light_source) == LIGHT_SOURCE_KIND_EMISSIVE_MESH {
                triangle_id = rand_range_u(emissive_mesh_triangle_count(light_source), rng);
            }
            let seed = rand_u(rng);
            var resolved = resolve_light_sample(LightSample((light_id << 16u) | triangle_id, seed), light_source);
            resolved.inverse_pdf /= picked.probability;
            var contribution = calculate_resolved_light_contribution(resolved, ray_hit.world_position, ray_hit.world_normal);
            contribution.radiance *= trace_visibility(ray_hit.world_position, resolved.world_position, RAY_FLAG_NONE);

            mis_weight = 1.0;
            if contribution.brdf_rays_can_hit {
                let pdf_of_bounce = brdf_pdf(wo, contribution.wi, ray_hit.world_normal, ray_hit.material, F_ab);
                mis_weight = power_heuristic(1.0 / contribution.inverse_solid_angle_pdf, pdf_of_bounce);
            }
            let brdf = evaluate_brdf(wo, contribution.wi, ray_hit.world_normal, ray_hit.material, F_ab);
            radiance += mis_weight * throughput * contribution.radiance * contribution.inverse_pdf * brdf;
        }

        let next_bounce = evaluate_and_sample_brdf(wo, ray_hit.world_normal, ray_hit.material, F_ab, rng);
        if next_bounce.pdf == 0.0 { break; }
        ray_direction = next_bounce.wi;
        ray_origin = ray_hit.world_position + (ray_hit.geometric_world_normal * RAY_T_MIN);
        ray_t_min = RAY_T_MIN;
        p_bounce = next_bounce.pdf;
        throughput *= next_bounce.throughput;

        let rr = saturate(luminance(throughput));
        if rand_f(rng) >= rr { break; }
        throughput /= rr;
    }
    return radiance;
}

@compute @workgroup_size(64, 1, 1)
fn capture(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let local = global_id.x;
    if local >= params.texel_count { return; }
    let texel = params.texel_offset + local;
    let size = params.size;
    let face = texel / (size * size);
    let in_face = texel % (size * size);
    let row = in_face / size;
    let column = in_face % size;

    let origin = params.origin.xyz;
    let box_from_unit = mat3x3(params.box_x.xyz, params.box_y.xyz, params.box_z.xyz);
    let texel_seed = pcg_hash(params.seed ^ pcg_hash((params.probe << 24u) ^ texel));

    var sum = vec3(0.0);
    var back_faces = 0u;
    for (var k = 0u; k < params.samples; k++) {
        var rng = pcg_hash(texel_seed ^ pcg_hash(k));
        let s = (f32(column) + rand_f(&rng)) / f32(size);
        let t = (f32(row) + rand_f(&rng)) / f32(size);
        let c = face_direction(face, s, t);
        // The unit box's point for this texel under Bevy's parallax correction (z negated), then in the world.
        let p = vec3(c.x, c.y, -c.z);
        let q = 0.5 * p / max(abs(p.x), max(abs(p.y), abs(p.z)));
        let box_point = box_from_unit * q + params.box_translation.xyz;
        let direction = normalize(box_point - origin);
        var back_face = false;
        sum += trace_path(origin, direction, &rng, &back_face);
        if back_face { back_faces += 1u; }
    }
    texels[local] = vec4(sum / f32(params.samples), f32(back_faces));
}
