// The traced light for surfaces the rasterizer still shades forward on a traced view (glass, blended surfaces):
// `bevy_pbr`'s `pbr_functions.wesl` imports this module when the pipeline has `TRACED_FORWARD_LIGHT`
// (`bevy_pbr::TracedForwardLight`). It reads the scene's TLAS for shadow rays and Solari's world cache for the
// light arriving at the fragment, through one bind group of its own (`forward_light.rs`).
//
// Rays here are forced opaque and skip glass (the same as the ladder's world cache, `world_cache_force_opaque`):
// alpha-masked holes (foliage) block them like solid surfaces, which needs no material lookup.

import constants::WORLD_CACHE_SIZE;
import bevy_pbr::render::utils::{rand_f, rand_vec2f};
import bevy_render::maths::orthonormalize;

enable wgpu_ray_query;

@group(constants::TRACED_FORWARD_LIGHT_BIND_GROUP) @binding(0) var tlas: acceleration_structure;
@group(constants::TRACED_FORWARD_LIGHT_BIND_GROUP) @binding(1) var<storage, read_write> world_cache: ForwardWorldCache;
@group(constants::TRACED_FORWARD_LIGHT_BIND_GROUP) @binding(2) var<uniform> constants: ForwardSolariSettings;

/// Keep in step with `SolariLightingSettings` in `realtime/bindings.wesl`.
struct ForwardSolariSettings {
    confidence_weight_cap: f32,
    primary_di_samples: u32,
    secondary_di_samples: u32,
    max_bounces: u32,
    world_cache_max_temporal_samples: f32,
    world_cache_direct_light_sample_count: u32,
    world_cache_max_gi_ray_distance: f32,
    world_cache_max_cell_updates_per_frame: u32,
    world_cache_position_base_cell_size: f32,
    world_cache_position_lod_scale: f32,
    frame_rng: u32,
    reset: u32,
}

struct ForwardWorldCacheGeometryData {
    world_position: vec3<f32>,
    padding_a: u32,
    world_normal: vec3<f32>,
    padding_b: u32,
}

/// Keep in step with `WorldCache` in `realtime/bindings.wesl` (its atomic `life` variant).
struct ForwardWorldCache {
    checksums: array<atomic<u32>, constants::WORLD_CACHE_SIZE>,
    life: array<atomic<u32>, constants::WORLD_CACHE_SIZE>,
    radiance: array<vec4<f32>, constants::WORLD_CACHE_SIZE>,
    geometry_data: array<ForwardWorldCacheGeometryData, constants::WORLD_CACHE_SIZE>,
    luminance_deltas: array<f32, constants::WORLD_CACHE_SIZE>,
    a: array<u32, constants::WORLD_CACHE_SIZE>,
    b: array<u32, constants::WORLD_CACHE_SIZE / 1024u>,
    active_cell_indices: array<u32, constants::WORLD_CACHE_SIZE>,
    active_cells_count: u32,
    update_window_start: u32,
}

/// TLAS instance mask of everything but glass (`scene/bindings.wesl`).
const INSTANCE_MASK_SOLID = 0x01u;
const RAY_T_MIN = 0.001f;
/// Rays start this far off the surface, along the side they leave from (to tune: a few times the depth error of
/// a forward-shaded fragment against the traced triangles at room scale).
const RAY_START_OFFSET = 0.01f;

/// Whether light from `direction` (normalised) reaches `world_position` from as far as `distance`: 1 when
/// nothing solid lies in between, 0 when something does.
fn visibility(world_position: vec3<f32>, world_normal: vec3<f32>, direction: vec3<f32>, distance: f32) -> f32 {
    // Leave from the side of the surface the ray goes to, so a closed mesh does not shadow its own front.
    let side = select(-1.0, 1.0, dot(world_normal, direction) >= 0.0);
    let origin = world_position + side * world_normal * RAY_START_OFFSET;
    let flags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_TERMINATE_ON_FIRST_HIT;
    let ray = RayDesc(flags, INSTANCE_MASK_SOLID, RAY_T_MIN, max(distance - RAY_START_OFFSET, RAY_T_MIN), origin, direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    while rayQueryProceed(&rq) {}
    return f32(rayQueryGetCommittedIntersection(&rq).kind == RAY_QUERY_INTERSECTION_NONE);
}

/// The irradiance at `world_position` facing `world_normal` (direct and indirect, before the view's exposure),
/// from Solari's world cache. A cell the cache lacks is added, as Solari's own queries do, and its light shows from
/// the next frames on; until then this returns 0. The lookups are jittered in the tangent plane, as Solari's are, and
/// averaged.
fn irradiance(world_position: vec3<f32>, world_normal: vec3<f32>, view_position: vec3<f32>, frag_coord: vec2<f32>) -> vec3<f32> {
    var rng = pcg_hash(bitcast<u32>(frag_coord.x) ^ pcg_hash(bitcast<u32>(frag_coord.y) ^ constants.frame_rng));
    // Nothing denoises these fragments after: average a few jittered lookups.
    var sum = vec3(0.0);
    // The cache keys cells by the normal's octant; on a curved surface seen up close that draws seams where the
    // normal crosses an axis plane. Tilting each lookup's normal a little blends the neighbouring octants there.
    for (var i = 0u; i < IRRADIANCE_LOOKUPS; i++) {
        let tilt = (vec3(rand_f(&rng), rand_f(&rng), rand_f(&rng)) * 2.0 - 1.0) * IRRADIANCE_NORMAL_TILT;
        sum += query_world_cache(world_position, normalize(world_normal + tilt), view_position, &rng);
    }
    return sum / f32(IRRADIANCE_LOOKUPS);
}

/// Lookups averaged per fragment (to tune: grain against cost).
const IRRADIANCE_LOOKUPS: u32 = 4u;
/// How far a lookup's normal is tilted, per axis (to tune: wider blends the seams over more of the surface).
const IRRADIANCE_NORMAL_TILT: f32 = 0.3;

// What follows is `realtime/world_cache_query.wesl` with the atomic life buffer and the default lifetime: the keys
// must match it exactly for the cells to be shared.

const WORLD_CACHE_CELL_LIFETIME: u32 = 10u;
const WORLD_CACHE_MAX_SEARCH_STEPS: u32 = 16u;
const WORLD_CACHE_EMPTY_CELL: u32 = 0u;
const WORLD_CACHE_CELL_NOT_FOUND: u32 = 0xFFFFFFFFu;

fn query_world_cache(world_position: vec3<f32>, world_normal: vec3<f32>, view_position: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let cell_size = get_cell_size(world_position, view_position, rng);
    let world_normal_quantized = bitcast<vec3<u32>>(quantize_normal(world_normal));

    // Jitter the query point in the tangent plane, which blurs the cache so it is not so grid-like; only look up
    // cells there, and fall back to the point itself on a miss.
    let TBN = orthonormalize(world_normal);
    let offset = (rand_vec2f(rng) * 2.0 - 1.0) * cell_size * 0.5;
    let jittered_world_position = world_position + offset.x * TBN[0] + offset.y * TBN[1];
    let jittered_cell_size = get_cell_size(jittered_world_position, view_position, rng);
    let jittered_position_quantized = bitcast<vec3<u32>>(quantize_position(jittered_world_position, jittered_cell_size));
    let jittered_key = find_cell_slot(
        compute_key(jittered_position_quantized, world_normal_quantized, jittered_cell_size),
        compute_checksum(jittered_position_quantized, world_normal_quantized, jittered_cell_size),
    );
    if jittered_key != WORLD_CACHE_CELL_NOT_FOUND {
        atomicStore(&world_cache.life[jittered_key], WORLD_CACHE_CELL_LIFETIME);
        return world_cache.radiance[jittered_key].rgb;
    }

    let world_position_quantized = bitcast<vec3<u32>>(quantize_position(world_position, cell_size));
    let base_key = compute_key(world_position_quantized, world_normal_quantized, cell_size);
    let checksum = compute_checksum(world_position_quantized, world_normal_quantized, cell_size);

    let key = find_cell_slot(base_key, checksum);
    if key != WORLD_CACHE_CELL_NOT_FOUND {
        atomicStore(&world_cache.life[key], WORLD_CACHE_CELL_LIFETIME);
        return world_cache.radiance[key].rgb;
    }

    for (var i = 0u; i < WORLD_CACHE_MAX_SEARCH_STEPS; i++) {
        let key = wrap_key(base_key + i);
        let existing_checksum = atomicLoad(&world_cache.checksums[key]);
        if existing_checksum == checksum {
            atomicStore(&world_cache.life[key], WORLD_CACHE_CELL_LIFETIME);
            return world_cache.radiance[key].rgb;
        }
        if existing_checksum != WORLD_CACHE_EMPTY_CELL { continue; }

        let cas = atomicCompareExchangeWeak(&world_cache.checksums[key], WORLD_CACHE_EMPTY_CELL, checksum);
        if cas.exchanged {
            atomicStore(&world_cache.life[key], WORLD_CACHE_CELL_LIFETIME);
            world_cache.geometry_data[key].world_position = world_position;
            world_cache.geometry_data[key].world_normal = world_normal;
            return vec3(0.0);
        }
        if cas.old_value == checksum {
            atomicStore(&world_cache.life[key], WORLD_CACHE_CELL_LIFETIME);
            return world_cache.radiance[key].rgb;
        }
    }

    return vec3(0.0);
}

fn find_cell_slot(base_key: u32, checksum: u32) -> u32 {
    for (var i = 0u; i < WORLD_CACHE_MAX_SEARCH_STEPS; i++) {
        let key = wrap_key(base_key + i);
        if atomicLoad(&world_cache.checksums[key]) == checksum {
            return key;
        }
    }
    return WORLD_CACHE_CELL_NOT_FOUND;
}

fn get_cell_size(world_position: vec3<f32>, view_position: vec3<f32>, rng: ptr<function, u32>) -> f32 {
    let camera_distance = distance(view_position, world_position) / constants.world_cache_position_lod_scale;
    let lod_f = log2(1.0 + camera_distance);
    let lod_fract = fract(lod_f);
    let lod = floor(lod_f) + select(0.0, 1.0, rand_f(rng) < lod_fract * lod_fract * lod_fract);
    return constants.world_cache_position_base_cell_size * exp2(lod);
}

fn quantize_position(world_position: vec3<f32>, quantization_factor: f32) -> vec3<f32> {
    return floor(world_position / quantization_factor + 0.0001);
}

fn quantize_normal(world_normal: vec3<f32>) -> vec3<f32> {
    return floor(world_normal + 0.0001);
}

fn compute_key(world_position: vec3<u32>, world_normal: vec3<u32>, cell_size: f32) -> u32 {
    var key = pcg_hash(world_position.x);
    key = pcg_hash(key + world_position.y);
    key = pcg_hash(key + world_position.z);
    key = pcg_hash(key + world_normal.x);
    key = pcg_hash(key + world_normal.y);
    key = pcg_hash(key + world_normal.z);
    key = pcg_hash(key + bitcast<u32>(cell_size));
    return wrap_key(key);
}

fn compute_checksum(world_position: vec3<u32>, world_normal: vec3<u32>, cell_size: f32) -> u32 {
    var key = jenkins_hash(world_position.x);
    key = jenkins_hash(key + world_position.y);
    key = jenkins_hash(key + world_position.z);
    key = jenkins_hash(key + world_normal.x);
    key = jenkins_hash(key + world_normal.y);
    key = jenkins_hash(key + world_normal.z);
    key = jenkins_hash(key + bitcast<u32>(cell_size));
    return max(key, 1u);
}

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn jenkins_hash(input: u32) -> u32 {
    var a = input;
    a = (a + 0x7ed55d16u) + (a << 12u);
    a = (a ^ 0xc761c23cu) ^ (a >> 19u);
    a = (a + 0x165667b1u) + (a << 5u);
    a = (a + 0xd3a2646cu) ^ (a << 9u);
    a = (a + 0xfd7046c5u) + (a << 3u);
    a = (a ^ 0xb55a4f09u) ^ (a >> 16u);
    return a;
}

fn wrap_key(key: u32) -> u32 {
    return key & (u32(WORLD_CACHE_SIZE) - 1u);
}
