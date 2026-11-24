// Copies the top-left render-resolution corner of the motion vector prepass texture, where the
// frame is drawn under an upscaler, into a texture of exactly that size for NRD.

@group(0) @binding(0) var motion_vectors: texture_2d<f32>;
@group(0) @binding(1) var cropped: texture_storage_2d<rg16float, write>;

@compute @workgroup_size(8, 8, 1)
fn crop(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= textureDimensions(cropped)) { return; }
    textureStore(cropped, global_id.xy, textureLoad(motion_vectors, global_id.xy, 0));
}
