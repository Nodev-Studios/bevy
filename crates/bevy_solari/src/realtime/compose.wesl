//! Turns the radiance a denoiser returned back into the lit color of the view: each lobe's
//! radiance times the reflectance it was divided by, plus what the surface emits.

import bevy_render::view::View;
import package::realtime::denoiser_contract::demodulation_albedos;
import package::realtime::gbuffer_utils::gpixel_resolve;

@group(0) @binding(0) var view_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var gbuffer: texture_2d<u32>;
@group(0) @binding(2) var depth_buffer: texture_depth_2d;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var diffuse_radiance: texture_2d<f32>;
@group(0) @binding(5) var specular_radiance: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn compose(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 { return; }

    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip, view.world_position);
    let albedos = demodulation_albedos(surface.material, surface.world_normal, normalize(view.world_position - surface.world_position));

    // The radiance is stored with the view's exposure already applied.
    var pixel_color = textureLoad(diffuse_radiance, global_id.xy, 0).rgb * albedos.diffuse;
    pixel_color += textureLoad(specular_radiance, global_id.xy, 0).rgb * albedos.specular;
    pixel_color += surface.material.emissive * view.exposure;
    textureStore(view_output, global_id.xy, vec4(pixel_color, 1.0));
}
