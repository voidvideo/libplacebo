/*
 * Copyright (C) 2026 Philip Langdale <philipl@overt.org>
 *
 * Ported from FFmpeg fruc_grayscale.comp.glsl and fruc_interpolate.comp.glsl
 * at 9d2d5b1aa39a714ad92cd3dd0253bd3aa7ee8be7.
 * Adaptations: packed RGB mixer inputs, libplacebo shader bindings and output,
 * metadata-only preparation, and separate grayscale dispatches per input.
 * The motion reconstruction and static-region guard retain upstream behavior.
 *
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 2.1, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details. You should have received a copy of the license along with
 * libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>
#include "shaders.h"
#include <libplacebo/shaders/fruc.h>

static bool valid_texture(pl_shader sh, const struct pl_hook_texture *tex,
                          int width, int height, bool flow)
{
    const struct pl_tex_params *p = tex->texture ? &tex->texture->params : &tex->params;
    if (!p->format || !p->sampleable || p->d || p->w < width || p->h < height ||
        (flow ? p->format->type != PL_FMT_SINT || p->format->num_components != 2 ||
                p->format->component_depth[0] != 16 || p->format->component_depth[1] != 16
              : (p->format->type != PL_FMT_FLOAT && p->format->type != PL_FMT_UNORM) ||
                p->format->num_components < 3 || !(p->format->caps & PL_FMT_CAP_LINEAR))) {
        SH_FAIL(sh, "Invalid FRUC %s texture", flow ? "flow" : "RGB");
        return false;
    }
    return true;
}

static ident_t bind_texture(pl_shader sh, const struct pl_hook_texture *tex,
                            bool flow, const pl_rect2df *rect, ident_t *pos)
{
    return sh_bind_metadata(sh, tex->texture, &tex->params, tex->sampler_type,
                            PL_TEX_ADDRESS_CLAMP,
                            flow ? PL_TEX_SAMPLE_NEAREST : PL_TEX_SAMPLE_LINEAR,
                            flow ? "fruc_flow" : "fruc_rgb", rect, pos, NULL);
}

void pl_shader_fruc_grayscale(pl_shader sh, const struct pl_hook_texture *src,
                              int width, int height)
{
    if (width <= 0 || height <= 0 || !src) {
        SH_FAIL(sh, "Invalid FRUC grayscale dimensions or input");
        return;
    }
    if (!valid_texture(sh, src, width, height, false) ||
        !sh_require(sh, PL_SHADER_SIG_NONE, width, height))
        return;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        SH_FAIL(sh, "FRUC requires compute shaders");
        return;
    }
    const pl_rect2df rect = {0, 0, width, height};
    ident_t pos;
    ident_t tex = bind_texture(sh, src, false, &rect, &pos);
    if (!tex)
        return;
    sh_describe(sh, "FRUC grayscale");
    GLSL("vec4 color = vec4(dot(textureLod("$", "$", 0.0).rgb,\n"
         "                      vec3(0.2126, 0.7152, 0.0722)));\n", tex, pos);
}

void pl_shader_fruc_interpolate(pl_shader sh, const struct pl_fruc_source *src)
{
    if (!src || src->width <= 0 || src->height <= 0 ||
        !isfinite(src->t) || src->t < 0.0f || src->t > 1.0f) {
        SH_FAIL(sh, "Invalid FRUC dimensions or interpolation time");
        return;
    }
    for (int i = 0; i < 2; i++) {
        if (!valid_texture(sh, &src->frames[i], src->width, src->height, false) ||
            !valid_texture(sh, &src->flow[i], 1, 1, true))
            return;
    }
    if (!sh_require(sh, PL_SHADER_SIG_NONE, src->width, src->height))
        return;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        SH_FAIL(sh, "FRUC requires compute shaders");
        return;
    }
    ident_t frames[2], flow[2], pos;
    for (int i = 0; i < 2; i++) {
        frames[i] = bind_texture(sh, &src->frames[i], false, NULL, i ? NULL : &pos);
        flow[i] = bind_texture(sh, &src->flow[i], true, NULL, NULL);
        if (!frames[i] || !flow[i])
            return;
    }
    ident_t fn = sh_fresh(sh, "fruc_interpolate");
    GLSLH("vec4 "$"(sampler2D f0, sampler2D f1, isampler2D fw, isampler2D bw,\n"
          "           vec2 base, vec2 size, float t) {\n"
          "    vec2 s0 = base, s1 = base;\n"
          "    for (int k = 0; k < 6; k++) {\n"
          "        vec2 vf = vec2(textureLod(fw, s0, 0.0).xy) / (32.0 * size);\n"
          "        vec2 vb = vec2(textureLod(bw, s1, 0.0).xy) / (32.0 * size);\n"
          "        s0 = mix(s0, base - t * vf, 0.5);\n"
          "        s1 = mix(s1, base - (1.0 - t) * vb, 0.5);\n"
          "    }\n"
          "    vec2 uv0 = size / vec2(textureSize(f0, 0));\n"
          "    vec2 uv1 = size / vec2(textureSize(f1, 0));\n"
          "    vec4 z0 = textureLod(f0, base * uv0, 0.0);\n"
          "    vec4 z1 = textureLod(f1, base * uv1, 0.0);\n"
          "    float l0 = dot(z0.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
          "    float l1 = dot(z1.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
          "    float lz = abs(l0 - l1);\n"
          "    float staticness = exp(-(lz * lz) / (0.03 * 0.03));\n"
          "    float disp = max(length((s0 - base) * size),\n"
          "                     length((s1 - base) * size));\n"
          "    float guard = staticness * smoothstep(5.0, 20.0, disp);\n"
          "    vec4 c0 = textureLod(f0, s0 * uv0, 0.0);\n"
          "    vec4 c1 = textureLod(f1, s1 * uv1, 0.0);\n"
          "    return mix(mix(c0, c1, t), mix(z0, z1, t), guard);\n"
          "}\n", fn);
    sh_describe(sh, "FRUC interpolation");
    GLSL("vec4 color = "$"("$", "$", "$", "$", "$", vec2(%d, %d), "$ ");\n",
         fn, frames[0], frames[1], flow[0], flow[1], pos,
         src->width, src->height, SH_FLOAT_DYN(src->t));
}
