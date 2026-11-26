//! Planar reflections: which pixel lies on a mirror, and what its mirror camera saw there.
//!
//! Bound at group 3 by the passes specialized with `PLANAR_REFLECTIONS` (deferred lighting and screen-space
//! reflections) on a view that reflects at least one mirror this frame. See `planar_reflection/mod.rs`.

const MAX_PLANAR_MIRRORS: u32 = 4u;

struct PlanarMirror {
    // The mirror camera's projection times its view.
    clip_from_world: mat4x4<f32>,
    // `n · x + w = 0`, `n` of unit length, out of the reflecting side.
    plane: vec4<f32>,
    center: vec4<f32>,
    // `(p - center) · dual_u` runs from -1 to 1 across the rectangle; the same for `dual_v`.
    dual_u: vec4<f32>,
    dual_v: vec4<f32>,
    // The part of the texture the mirror camera rendered: (min u, min v, max u, max v).
    uv_rect: vec4<f32>,
}

struct PlanarMirrors {
    mirrors: array<PlanarMirror, MAX_PLANAR_MIRRORS>,
    count: u32,
    normal_distortion: f32,
    _padding: vec4<f32>,
}

@group(3) @binding(0) var<uniform> planar_mirrors: PlanarMirrors;
@group(3) @binding(1) var planar_mirror_texture_0: texture_2d<f32>;
@group(3) @binding(2) var planar_mirror_texture_1: texture_2d<f32>;
@group(3) @binding(3) var planar_mirror_texture_2: texture_2d<f32>;
@group(3) @binding(4) var planar_mirror_texture_3: texture_2d<f32>;
@group(3) @binding(5) var planar_mirror_sampler: sampler;

/// The index of the mirror the point `P` with normal `N` lies on, or -1. `view_distance` is the point's
/// distance from the camera: the depth buffer's error grows with it.
fn planar_mirror_at(P: vec3<f32>, N: vec3<f32>, view_distance: f32) -> i32 {
    // to tune: 5 mm plus 0.2 % of the distance holds a 32-bit reverse-Z depth's reconstruction error at
    // grazing angles, and is far thinner than anything placed against a mirror.
    let tolerance = 0.005 + 0.002 * view_distance;
    for (var i = 0u; i < planar_mirrors.count; i += 1u) {
        let mirror = planar_mirrors.mirrors[i];
        if (abs(dot(mirror.plane.xyz, P) + mirror.plane.w) > tolerance) {
            continue;
        }
        // The reflecting side only, with room for a normal map.
        if (dot(N, mirror.plane.xyz) < 0.5) {
            continue;
        }
        let offset = P - mirror.center.xyz;
        // to tune: a hair past the edge, so the rim of the mirror is not left out by rounding.
        if (abs(dot(offset, mirror.dual_u.xyz)) <= 1.002 && abs(dot(offset, mirror.dual_v.xyz)) <= 1.002) {
            return i32(i);
        }
    }
    return -1;
}

/// What the mirror camera of mirror `index` saw at the point `P`, the light as it reaches the eye before
/// the surface's Fresnel: the texture where `P` projects, moved along the mirror by the bend of the normal `N`.
fn planar_mirror_radiance(index: i32, P: vec3<f32>, N: vec3<f32>, view_distance: f32) -> vec3<f32> {
    let mirror = planar_mirrors.mirrors[index];
    let n = mirror.plane.xyz;
    let bend = N - n * dot(N, n);
    let P_bent = P + bend * (planar_mirrors.normal_distortion * view_distance);
    let clip = mirror.clip_from_world * vec4(P_bent, 1.0);
    // Inside the part of the texture that was rendered, half a texel in.
    let half_texel = 0.5 / vec2<f32>(textureDimensions(planar_mirror_texture_0));
    let uv = clamp(
        clip.xy / clip.w * vec2(0.5, -0.5) + vec2(0.5),
        mirror.uv_rect.xy + half_texel,
        mirror.uv_rect.zw - half_texel,
    );
    switch (index) {
        case 0: {
            return textureSampleLevel(planar_mirror_texture_0, planar_mirror_sampler, uv, 0.0).rgb;
        }
        case 1: {
            return textureSampleLevel(planar_mirror_texture_1, planar_mirror_sampler, uv, 0.0).rgb;
        }
        case 2: {
            return textureSampleLevel(planar_mirror_texture_2, planar_mirror_sampler, uv, 0.0).rgb;
        }
        default: {
            return textureSampleLevel(planar_mirror_texture_3, planar_mirror_sampler, uv, 0.0).rgb;
        }
    }
}
