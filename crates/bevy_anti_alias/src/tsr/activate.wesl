// TSR, pass 2 of 3 (render resolution): how much of the history to distrust because the surface behind the
// pixel changed (depth clip), and whether the pixel's luma keeps flipping, which marks thin detail the
// upscale pass should keep from history.
//
// Ported from https://github.com/SnapdragonStudios/snapdragon-gsr, sgsr/v2/include/glsl_3_pass_cs/
// sgsr2_activate.comp, commit d926f074bcb9d714e179f1ce0fcb9ee2eeb5074e. What differs from it, and why:
// - Depth: SGSR 2 is written for depth that is 0 at the near plane and 1 at the far one. Bevy's is reverse-Z
//   with an infinite far plane, `near / z`, which is exactly `1 - d` of a forward depth with an infinite far
//   plane. So each `1.0 - d` of the original is `d` here, each `min` that picks the nearer depth is a `max`,
//   and the far-plane test `depth < 1 - 1e-5` is `depth > 1e-5`. Differences of two depths keep their size.
//   Keeping Bevy's convention also keeps the half float precise for far surfaces, where `1 - d` loses it.
// - Motion is SGSR 2's clip-space motion, written by `convert.wesl`; its non-Y-up branch applies.
// - Reads at a texel the original gathers from are `textureLoad`s; its `textureGatherOffset`s over the
//   motion-depth texture stay gathers, in the original's order, unrolled because WGSL wants constant offsets.
// - Luma is decoded with `convert.wesl`'s rounded steps.
// - The translucency mask is zero (see `convert.wesl`), so the fourth channel written here holds only the luma
//   flag, as 0 or 1, instead of `floor(mask) + 0.5 * flag`.
//
// Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// SPDX-License-Identifier: BSD-3-Clause

import package::tsr::params::TsrParams;

@group(0) @binding(0) var<uniform> params: TsrParams;
@group(0) @binding(1) var previous_luma_history: texture_2d<u32>;
@group(0) @binding(2) var motion_depth: texture_2d<f32>;
@group(0) @binding(3) var ycocg_color: texture_2d<u32>;
@group(0) @binding(4) var motion_depth_clip: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var luma_history: texture_storage_2d<r32uint, write>;
@group(0) @binding(6) var point_clamp_sampler: sampler;

const EPSILON: f32 = 1.19e-07;

fn decode_color_y(sample32: u32) -> f32 {
    return f32(sample32 >> 21u) * (1.0 / 2047.0);
}

// How well a previous depth explains the current one, after Akeley and Su's minimum triangle separation, as in
// FSR 2's depth clip. `separation` is SGSR 2's `Ksep * Kfov * diagonal`.
fn depth_weight(previous_depth: f32, depth: f32, separation: f32) -> f32 {
    let required = separation * max(previous_depth, depth);
    return clamp(required / (abs(previous_depth - depth) + EPSILON), 0.0, 1.0);
}

@compute
@workgroup_size(8, 8, 1)
fn activate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= params.render_size) {
        return;
    }
    let pos = vec2<i32>(global_id.xy);
    let last = vec2<i32>(params.render_size) - 1;
    let render_size = vec2<f32>(params.render_size);

    let viewport_uv = (vec2<f32>(global_id.xy) + 0.5) * params.render_size_rcp;
    let luma_reference = decode_color_y(textureLoad(ycocg_color, pos, 0).x);

    let mda = textureLoad(motion_depth, pos, 0);
    let motion = mda.xy;
    let depth = mda.z;
    let previous_uv = viewport_uv - 0.5 * motion;

    var depth_clip = 0.0;
    if depth > 1.0e-05 {
        let previous_sample = previous_uv * render_size - 0.5;
        let previous_fraction = previous_sample - floor(previous_sample);
        let one_minus_fraction_x = 1.0 - previous_fraction.x;
        let bilinear_weights = vec4(
            one_minus_fraction_x - one_minus_fraction_x * previous_fraction.y,
            previous_fraction.x - previous_fraction.x * previous_fraction.y,
            one_minus_fraction_x * previous_fraction.y,
            previous_fraction.x * previous_fraction.y,
        );

        let separation = 1.37e-05 * params.camera_fov_angle_hor * length(render_size);
        var weight = 0.0;

        // The original's loop over offsets (-1, -1), (-1, 0), (0, -1), (0, 0), two at a time, unrolled. As
        // there, the second sample of each pair reuses the upper row of the first's gather.
        let gather_0 = textureGather(2, motion_depth, point_clamp_sampler, previous_uv, vec2(-1, -1));
        let upper_0 = max(gather_0.z, gather_0.w);
        var previous_depth = max(max(gather_0.x, gather_0.y), upper_0);
        weight += depth_weight(previous_depth, depth, separation) * bilinear_weights.x;
        let gather_1 = textureGather(2, motion_depth, point_clamp_sampler, previous_uv, vec2(-1, 0)).zw;
        previous_depth = max(max(gather_1.x, gather_1.y), upper_0);
        weight += depth_weight(previous_depth, depth, separation) * bilinear_weights.y;

        let gather_2 = textureGather(2, motion_depth, point_clamp_sampler, previous_uv, vec2(0, -1));
        let upper_2 = max(gather_2.z, gather_2.w);
        previous_depth = max(max(gather_2.x, gather_2.y), upper_2);
        weight += depth_weight(previous_depth, depth, separation) * bilinear_weights.z;
        let gather_3 = textureGather(2, motion_depth, point_clamp_sampler, previous_uv, vec2(0, 0)).zw;
        previous_depth = max(max(gather_3.x, gather_3.y), upper_2);
        weight += depth_weight(previous_depth, depth, separation) * bilinear_weights.w;

        depth_clip = clamp(1.0 - weight, 0.0, 1.0);
    }

    // The texel a gather at `previous_uv` returns in `.w`, with the sampler's clamp to the edge.
    let previous_texel = clamp(vec2<i32>(floor(previous_uv * render_size - 0.5)), vec2(0), last);
    let previous_luma_pack = textureLoad(previous_luma_history, previous_texel, 0).x;
    let previous_luma_diff = vec2(
        unpack2x16float(previous_luma_pack >> 16u).x,
        unpack2x16float(previous_luma_pack & 0xFFFFu).x,
    );

    var history_usable = false;
    if depth_clip + f32(params.reset) < 0.1 {
        history_usable = all(previous_uv >= vec2(0.0)) && all(previous_uv <= vec2(1.0));
    }
    let luma_diff = luma_reference - previous_luma_diff.x;
    var current_luma_diff = vec2(0.0);
    if history_usable {
        current_luma_diff.x = luma_reference;
        if previous_luma_diff.y == 0.0 {
            current_luma_diff.y = luma_diff;
        } else if sign(luma_diff) == sign(previous_luma_diff.y) {
            current_luma_diff.y = sign(luma_diff) * min(abs(previous_luma_diff.y), abs(luma_diff));
        } else {
            current_luma_diff.y = previous_luma_diff.y;
        }
    }

    let luma_flag = select(
        0.0,
        1.0,
        current_luma_diff.x != 0.0 && abs(current_luma_diff.y) != abs(luma_diff),
    );
    let pack = (pack2x16float(vec2(current_luma_diff.x, 0.0)) << 16u)
        | pack2x16float(vec2(current_luma_diff.y, 0.0));
    textureStore(luma_history, pos, vec4(pack));
    textureStore(motion_depth_clip, pos, vec4(motion, depth_clip, luma_flag));
}
