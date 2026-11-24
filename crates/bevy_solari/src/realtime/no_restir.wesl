import package::realtime::gbuffer_utils::gpixel_resolve;
import package::realtime::initial_path::generate_initial_path;
import package::realtime::bindings::{constants, depth_buffer, gbuffer, view, view_output};
import package::scene::bindings::RAY_T_MAX;
import package::realtime::world_cache_query::{query_world_cache, WORLD_CACHE_CELL_LIFETIME};

@if(DENOISER_CONTRACT)
import package::realtime::denoiser_contract::store_denoiser_radiance;

enable wgpu_ray_query;

@compute @workgroup_size(8, 8, 1)
fn initial_and_shade(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_rng;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 { return; }

    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip, view.world_position);

    let path = generate_initial_path(surface.world_position, surface.world_normal, surface.material, workgroup_id.xy, global_id.xy, &rng);

    var pixel_color = path.radiance;
    pixel_color += surface.material.emissive;
    pixel_color *= view.exposure;
    textureStore(view_output, global_id.xy, vec4(pixel_color, 1.0));

    @if(DENOISER_CONTRACT)
    store_denoiser_radiance(global_id.xy, path.diffuse_radiance, path.radiance - path.diffuse_radiance, path.hit_distances, surface.material, surface.world_position, surface.world_normal);

    @if(VISUALIZE_WORLD_CACHE)
    textureStore(view_output, global_id.xy, vec4(query_world_cache(surface.world_position, surface.world_normal, view.world_position, RAY_T_MAX, WORLD_CACHE_CELL_LIFETIME, &rng) * view.exposure, 1.0));
}
