// TSR, pass 1 of 3 (render resolution): finds the nearest depth around each pixel, takes its motion, and packs
// its colour as YCoCg in 32 bits.
//
// Ported from https://github.com/SnapdragonStudios/snapdragon-gsr, sgsr/v2/include/glsl_3_pass_cs/
// sgsr2_convert.comp, commit d926f074bcb9d714e179f1ce0fcb9ee2eeb5074e. What differs from it, and why:
// - Depth: Bevy's is reverse-Z with an infinite far plane (1 at the near plane, 0 at infinity), so the nearest
//   depth is the largest one (SGSR 2's README asks for exactly this change).
// - Motion: Bevy writes a motion vector for every drawn pixel, the camera's own motion included, as
//   `uv_now - uv_previous` with uv's y down. SGSR 2 wants clip-space motion in the same direction, scaled
//   to -1..1 (`PrevUV = uv - 0.5 * motion` in its non-Y-up branch), which is twice Bevy's. Its decoding of a
//   16-bit encoded velocity, and its fallback that reprojects depth for pixels without velocity, have nothing
//   to do here.
// - Translucency: SGSR 2 compares the colour before and after the transparent pass to find translucent pixels.
//   Bevy keeps no copy of the frame before its transparent pass, so there is no mask; the fourth channel of
//   the motion-depth texture is left zero.
// - Bevy's input textures are output-sized and only their top-left render rectangle is drawn, so every read is
//   a `textureLoad` clamped to that rectangle instead of SGSR 2's gathers in uv.
// - Quantization: SGSR 2 truncates (`uint(v * 2047.5)`) and decodes by dividing by 2047.5 (1023.5), so the
//   neutral chroma 0.5 comes back as 0.4996 and 0.4993: black turns about (1, 0, 5) / 255 purple after Bevy's
//   tonemapping, which shows in dark scenes. Here each channel is rounded onto steps of 1/2047 (luma), 1/2046
//   and 1/1022 (chroma), which keep 0.5 exact; `activate.wesl` and `upscale.wesl` decode with the same steps.
// - Pre-exposure is 1: Bevy applies the camera's exposure while lighting, so the main texture is already
//   exposed.
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
@group(0) @binding(1) var input_color: texture_2d<f32>;
@group(0) @binding(2) var input_depth: texture_depth_2d;
@group(0) @binding(3) var input_motion_vectors: texture_2d<f32>;
@group(0) @binding(4) var ycocg_color: texture_storage_2d<r32uint, write>;
@group(0) @binding(5) var motion_depth: texture_storage_2d<rgba16float, write>;

@compute
@workgroup_size(8, 8, 1)
fn convert(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= params.render_size) {
        return;
    }
    let pos = vec2<i32>(global_id.xy);
    let last = vec2<i32>(params.render_size) - 1;

    // The 3x3 block SGSR 2 gathers, derived from FSR 2's dilated depth.
    var nearest_z = 0.0;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let depth = textureLoad(input_depth, clamp(pos + vec2(x, y), vec2(0), last), 0);
            nearest_z = max(nearest_z, depth);
        }
    }

    let motion = 2.0 * textureLoad(input_motion_vectors, pos, 0).xy;

    // SGSR 2's simple tonemap, so that the colour fits 11 + 11 + 10 bits.
    var color = textureLoad(input_color, pos, 0).rgb;
    color /= max(max(color.r, color.g), color.b) + 1.0;

    let luma = 0.25 * (color.r + 2.0 * color.g + color.b);
    let co = clamp(0.5 * color.r + 0.5 - 0.5 * color.b, 0.0, 1.0);
    let cg = clamp(luma + co - color.r, 0.0, 1.0);
    // Rounded onto steps that hold 0.5 exactly (see the header).
    let x11 = u32(round(luma * 2047.0));
    let y11 = u32(round(co * 2046.0));
    let z10 = u32(round(cg * 1022.0));

    textureStore(ycocg_color, pos, vec4((x11 << 21u) | (y11 << 10u) | z10));
    textureStore(motion_depth, pos, vec4(motion, nearest_z, 0.0));
}
