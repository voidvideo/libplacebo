/*
 * Copyright (C) 2026 Philip Langdale <philipl@overt.org>
 * LGPL-2.1-or-later. See LICENSE for the full license text.
 */
#pragma once

#include <libplacebo/shaders/custom.h>

PL_API_BEGIN

// Packed RGB inputs at the renderer's frame-mixing boundary. Metadata-only
// textures are accepted for prepared shader description. width/height describe
// the visible image, which may be smaller than its texture allocation.
struct pl_fruc_source {
    struct pl_hook_texture frames[2];
    struct pl_hook_texture flow[2]; // forward/backward, two signed 16-bit components
    int width, height;
    float t; // requested position between the two frames, in [0, 1]
};

// FFmpeg FRUC's BT.709 motion proxy. Dispatch into an R8 UNORM target; source
// RGB is not replaced by this proxy during synthesis.
PL_API void pl_shader_fruc_grayscale(pl_shader sh,
    const struct pl_hook_texture *source, int width, int height);

// FFmpeg FRUC synthesis, including six damped Picard iterations and its
// static-region guard. Flow values use NVIDIA's signed S10.5 encoding.
// Dispatch into an RGB(A) target with the source precision.
PL_API void pl_shader_fruc_interpolate(pl_shader sh, const struct pl_fruc_source *source);

PL_API_END
