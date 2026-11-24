// The uniform every pass of TSR reads: Snapdragon Game Super Resolution 2's `Params` block, without the fields
// its 3-pass compute variant does not read (`clipToPrevClip`, `preExposure`, `cameraNear`, `MinLerpContribution`,
// `bSameCamera`; see `convert.wesl` and `upscale.wesl` for why the first two go).
//
// Ported from https://github.com/SnapdragonStudios/snapdragon-gsr, sgsr/v2/include/glsl_3_pass_cs/, commit
// d926f074bcb9d714e179f1ce0fcb9ee2eeb5074e.
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

struct TsrParams {
    // The rendered rectangle, in pixels. It is the top-left corner of Bevy's full-size main, depth and motion
    // vector textures, and the whole of TSR's own render-size textures.
    render_size: vec2<u32>,
    // The output, in pixels.
    display_size: vec2<u32>,
    render_size_rcp: vec2<f32>,
    display_size_rcp: vec2<f32>,
    // SGSR 2's jitter: the frame's sample sits at the pixel centre minus this, in render pixels, y down. It is
    // the negative of Bevy's `TemporalJitter::offset` (see `prepare.rs`).
    jitter_offset: vec2<f32>,
    // tan(horizontal field of view / 2).
    camera_fov_angle_hor: f32,
    // 1 when the history must be dropped this frame.
    reset: u32,
}
