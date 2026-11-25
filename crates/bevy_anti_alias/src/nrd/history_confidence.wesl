// How far NRD's history can be trusted, estimated from what it showed on the previous frame against what the
// lighting pass traced on this one. `history_confidence.rs` says why it is built this way.
//
// `gather` sums, per tile of TILE_SIZE x TILE_SIZE pixels, the luminance of this frame's noisy lighting and that
// of last frame's denoised lighting where the same surface was, over the pixels whose surface was visible in both
// frames. `resolve` compares those sums over a few tiles around each one and writes the confidence.

/// Keep in sync with `TILE_SIZE` in `history_confidence.rs`.
const TILE_SIZE = 16u;

struct Params {
    /// 0 when last frame's denoised lighting and view depth are not there to compare with.
    history_valid: u32,
    /// Tiles on each side of a tile that `resolve` adds to it.
    radius: u32,
    /// A pixel counts as the same surface as last frame's when the two view depths differ by at most this
    /// fraction of the depth.
    depth_tolerance: f32,
    /// Below this relative change of the lighting the history is kept whole...
    change_kept: f32,
    /// ...and from this one on it is dropped.
    change_dropped: f32,
    /// Luminance per pixel below which a change counts as relative to this value instead, so that noise in dark
    /// tiles does not read as a change of light.
    luminance_floor: f32,
    /// Pixels at or beyond this view depth have no surface.
    sky_depth: f32,
}

@group(0) @binding(0) var noisy_diffuse: texture_2d<f32>;
@group(0) @binding(1) var noisy_specular: texture_2d<f32>;
@group(0) @binding(2) var view_depth: texture_2d<f32>;
@group(0) @binding(3) var motion_vectors: texture_2d<f32>;
@group(0) @binding(4) var previous_diffuse: texture_2d<f32>;
@group(0) @binding(5) var previous_specular: texture_2d<f32>;
@group(0) @binding(6) var previous_view_depth: texture_2d<f32>;
@group(0) @binding(7) var tiles_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(8) var<uniform> params: Params;

var<workgroup> sums: array<vec3<f32>, 64>;

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

/// This frame's and last frame's luminance of one pixel, and 1 in z; zero when the pixel has no surface or its
/// surface was not visible last frame.
fn pixel_luminance(pixel: vec2<u32>, size: vec2<u32>) -> vec3<f32> {
    if any(pixel >= size) { return vec3(0.0); }
    let depth = textureLoad(view_depth, pixel, 0).r;
    if !(depth > 0.0 && depth < params.sky_depth) { return vec3(0.0); }

    // Bevy's motion vectors are this frame's uv minus last frame's.
    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(size);
    let previous_uv = uv - textureLoad(motion_vectors, pixel, 0).xy;
    if any(previous_uv < vec2(0.0)) || any(previous_uv >= vec2(1.0)) { return vec3(0.0); }
    let previous_pixel = vec2<u32>(previous_uv * vec2<f32>(size));
    let previous_depth = textureLoad(previous_view_depth, previous_pixel, 0).r;
    // A disocclusion would read as a change of light; NRD handles those itself.
    if abs(previous_depth - depth) > params.depth_tolerance * depth { return vec3(0.0); }

    let current = luminance(textureLoad(noisy_diffuse, pixel, 0).rgb)
        + luminance(textureLoad(noisy_specular, pixel, 0).rgb);
    let previous = luminance(textureLoad(previous_diffuse, previous_pixel, 0).rgb)
        + luminance(textureLoad(previous_specular, previous_pixel, 0).rgb);
    return vec3(current, previous, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn gather(
    @builtin(local_invocation_index) index: u32,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let size = textureDimensions(view_depth);
    // Each thread takes four pixels 8 apart, so neighbouring threads read neighbouring pixels.
    let origin = group_id.xy * TILE_SIZE + local_id.xy;
    var sum = pixel_luminance(origin, size);
    sum += pixel_luminance(origin + vec2(8u, 0u), size);
    sum += pixel_luminance(origin + vec2(0u, 8u), size);
    sum += pixel_luminance(origin + vec2(8u, 8u), size);

    sums[index] = sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if index < stride {
            sums[index] += sums[index + stride];
        }
        workgroupBarrier();
    }
    if index == 0u {
        textureStore(tiles_out, group_id.xy, vec4(sums[0], 0.0));
    }
}

// `resolve` shares `params` with `gather`.
@group(0) @binding(9) var tiles: texture_2d<f32>;
@group(0) @binding(10) var confidence: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_count = vec2<i32>(textureDimensions(tiles));
    let tile = vec2<i32>(global_id.xy);
    if any(tile >= tile_count) { return; }

    var value = 1.0;
    if params.history_valid != 0u {
        let radius = i32(params.radius);
        var sum = vec3(0.0);
        for (var y = max(tile.y - radius, 0); y <= min(tile.y + radius, tile_count.y - 1); y++) {
            for (var x = max(tile.x - radius, 0); x <= min(tile.x + radius, tile_count.x - 1); x++) {
                sum += textureLoad(tiles, vec2(x, y), 0).xyz;
            }
        }
        let scale = max(max(sum.x, sum.y), max(params.luminance_floor * sum.z, 1e-6));
        let change = abs(sum.x - sum.y) / scale;
        value = 1.0 - smoothstep(params.change_kept, params.change_dropped, change);
    }
    textureStore(confidence, global_id.xy, vec4(value, 0.0, 0.0, 0.0));
}
