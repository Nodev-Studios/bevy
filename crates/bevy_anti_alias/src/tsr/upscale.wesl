// TSR, pass 3 of 3 (output resolution): upsamples the frame with a Lanczos-like kernel over the 3x3 render
// pixels around each output pixel, clamps the reprojected history to their colour box, blends the two, and
// writes both the new history and the output.
//
// Ported from https://github.com/SnapdragonStudios/snapdragon-gsr, sgsr/v2/include/glsl_3_pass_cs/
// sgsr2_upscale.comp, commit d926f074bcb9d714e179f1ce0fcb9ee2eeb5074e. What differs from it, and why:
// - Motion is SGSR 2's clip-space motion, written by `convert.wesl`; its non-Y-up branch applies.
// - The 3x3 neighbourhood the original gathers in four reads is a loop of `textureLoad`s clamped to the render
//   rectangle; the weights and the order of the box statistics are the original's.
// - Translucency: without a mask (see `convert.wesl`) the original's `alphamask` is zero, and the history
//   weight it carries in the history's alpha (`Wfactor`, the larger of the mask and the previous weight) stays
//   zero from the first frame on. Both are dropped, with every term they scale: `kernelfactor` is the reset
//   flag, `OneMinusWfactor` is 1, and `lerpcontribution` is not reduced by the mask.
// - The luma flag from `activate.wesl` arrives as 0 or 1, not as half of it in a fractional part.
// - Pre-exposure is 1 (see `convert.wesl`), and the output's alpha is 1.
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
@group(0) @binding(1) var previous_history: texture_2d<f32>;
@group(0) @binding(2) var motion_depth_clip: texture_2d<f32>;
@group(0) @binding(3) var ycocg_color: texture_2d<u32>;
@group(0) @binding(4) var history_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var scene_color_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var linear_clamp_sampler: sampler;

const EPSILON: f32 = 1.192e-07;

fn fast_lanczos(base: f32) -> f32 {
    let y = base - 1.0;
    let y2 = y * y;
    let y_temp = 0.75 * y + y2;
    return y_temp * y2;
}

fn decode_color(sample32: u32) -> vec3<f32> {
    let x11 = sample32 >> 21u;
    let y11 = sample32 & (2047u << 10u);
    let z10 = sample32 & 1023u;
    return vec3(
        f32(x11) * (1.0 / 2047.5),
        f32(y11) * 4.76953602e-7 - 0.5,
        f32(z10) * (1.0 / 1023.5) - 0.5,
    );
}

@compute
@workgroup_size(8, 8, 1)
fn upscale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= params.display_size) {
        return;
    }
    let display_size = vec2<f32>(params.display_size);
    let render_size = vec2<f32>(params.render_size);
    let last = vec2<i32>(params.render_size) - 1;
    let reset = f32(params.reset);

    let bias_max_viewport_x_scale = min(display_size.x / render_size.x, 1.99);
    let scale_factor = min(20.0, pow((display_size.x / render_size.x) * (display_size.y / render_size.y), 3.0));
    let jitter = params.jitter_offset;

    let hr_uv = (vec2<f32>(global_id.xy) + 0.5) * params.display_size_rcp;
    let jitter_uv = clamp(hr_uv + jitter * params.render_size_rcp, vec2(0.0), vec2(1.0));
    let input_pos = min(vec2<i32>(jitter_uv * render_size), last);

    let history_value = textureLoad(motion_depth_clip, input_pos, 0).w;
    let mda = textureSampleLevel(motion_depth_clip, linear_clamp_sampler, jitter_uv, 0.0).xyz;
    let motion = mda.xy;
    let depth_factor = mda.z;
    let previous_uv = clamp(hr_uv - 0.5 * motion, vec2(0.0), vec2(1.0));

    var history_color = textureSampleLevel(previous_history, linear_clamp_sampler, previous_uv, 0.0).xyz;

    let bias_max = bias_max_viewport_x_scale - bias_max_viewport_x_scale * reset;
    let bias_min = max(1.0, 0.3 + 0.3 * bias_max);
    let bias_factor = max(0.25 * depth_factor, reset);
    let kernel_bias = 0.5 * mix(bias_max, bias_min, bias_factor);
    let kernel_bias2 = kernel_bias * kernel_bias;
    let motion_viewport_len = length(motion * display_size);
    let curve_bias = mix(-2.0, -3.0, clamp(motion_viewport_len * 0.02, 0.0, 1.0));

    let src_pos = vec2<f32>(input_pos) + 0.5 - jitter;
    let src_output_pos = hr_uv * render_size;
    let src_offset = src_pos - src_output_pos;

    var upsampled = vec4(0.0);
    var box_min = vec3(0.0);
    var box_max = vec3(0.0);
    var box_center = vec3(0.0);
    var box_variance = vec3(0.0);
    var box_weight = 0.0;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let offset = vec2(x, y);
            let sample_color = decode_color(textureLoad(ycocg_color, clamp(input_pos + offset, vec2(0), last), 0).x);
            let base_offset = src_offset + vec2<f32>(offset);
            let base_offset_dot = dot(base_offset, base_offset);
            let weight = fast_lanczos(clamp(base_offset_dot * kernel_bias2, 0.0, 1.0));
            upsampled += vec4(sample_color * weight, weight);
            let sample_box_weight = exp(base_offset_dot * curve_bias);
            if x == -1 && y == -1 {
                box_min = sample_color;
                box_max = sample_color;
            } else {
                box_min = min(box_min, sample_color);
                box_max = max(box_max, sample_color);
            }
            let weighted_sample = sample_color * sample_box_weight;
            box_center += weighted_sample;
            box_variance += sample_color * weighted_sample;
            box_weight += sample_box_weight;
        }
    }

    box_weight = 1.0 / box_weight;
    box_center *= box_weight;
    box_variance *= box_weight;
    box_variance = sqrt(abs(box_variance - box_center * box_center));

    upsampled = vec4(
        clamp(upsampled.xyz / upsampled.w, box_min - vec3(0.05), box_max + vec3(0.05)),
        upsampled.w * (1.0 / 3.0),
    );

    let t_contribute = history_value * clamp(box_variance.x * 10.0, 0.0, 1.0);

    var base_update = 1.0 - depth_factor;
    base_update = min(base_update, mix(base_update, upsampled.w * 10.0, clamp(10.0 * motion_viewport_len, 0.0, 1.0)));
    base_update = min(base_update, mix(base_update, upsampled.w, clamp(motion_viewport_len * 0.05, 0.0, 1.0)));
    var base_alpha = base_update;

    let box_scale = max(depth_factor, clamp(motion_viewport_len * 0.05, 0.0, 1.0));
    let box_size = mix(scale_factor, 1.0, box_scale);
    let scaled_box_variance = box_variance * box_size;
    box_max = min(box_max, box_center + scaled_box_variance);
    box_min = max(box_min, box_center - scaled_box_variance);

    let clamped_color = clamp(history_color, box_min, box_max);
    let outside_box = any(box_min > history_color) || any(history_color > box_max);
    let lerp_contribution = clamp(select(1.0, t_contribute, outside_box), 0.0, 1.0);
    history_color = mix(clamped_color, history_color, lerp_contribution);
    base_alpha = mix(min(base_alpha, 0.1), base_alpha, lerp_contribution);

    let alpha_sum = max(EPSILON, base_alpha + upsampled.w);
    let alpha = clamp(upsampled.w / alpha_sum + reset, 0.0, 1.0);
    let color = mix(history_color, upsampled.xyz, alpha);

    textureStore(history_output, global_id.xy, vec4(color, 0.0));

    // YCoCg to RGB, then SGSR 2's simple tonemap undone.
    let x_z = color.x - color.z;
    let rgb = vec3(x_z + color.y, color.x + color.z, x_z - color.y);
    let comp_max = max(max(rgb.r, rgb.g), rgb.b);
    let scale = 1.0 / ((1.0 + 1.0 / 65504.0) - comp_max);
    textureStore(scene_color_output, global_id.xy, vec4(rgb * scale, 1.0));
}
