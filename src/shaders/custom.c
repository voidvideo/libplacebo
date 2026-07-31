/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "shaders.h"

#include <libplacebo/shaders/custom.h>

bool pl_shader_custom(pl_shader sh, const struct pl_custom_shader *params)
{
    if (params->compute) {
        int bw = PL_DEF(params->compute_group_size[0], 16);
        int bh = PL_DEF(params->compute_group_size[1], 16);
        bool flex = !params->compute_group_size[0] ||
                    !params->compute_group_size[1];
        if (!sh_try_compute(sh, bw, bh, flex, params->compute_shmem))
            return false;
    }

    if (!sh_require(sh, params->input, params->output_w, params->output_h))
        return false;

    sh->output = params->output;

    for (int i = 0; i < params->num_variables; i++) {
        struct pl_shader_var sv = params->variables[i];
        GLSLP("#define %s "$"\n", sv.var.name, sh_var(sh, sv));
    }

    for (int i = 0; i < params->num_descriptors; i++) {
        struct pl_shader_desc sd = params->descriptors[i];
        GLSLP("#define %s "$"\n", sd.desc.name, sh_desc(sh, sd));
    }

    for (int i = 0; i < params->num_vertex_attribs; i++) {
        struct pl_shader_va sva = params->vertex_attribs[i];
        GLSLP("#define %s "$"\n", sva.attr.name, sh_attr(sh, sva));
    }

    for (int i = 0; i < params->num_constants; i++) {
        struct pl_shader_const sc = params->constants[i];
        GLSLP("#define %s "$"\n", sc.name, sh_const(sh, sc));
    }

    // `static_text` routes the three source strings through
    // `pl_str_builder_const_str`, which records the POINTER rather than copying
    // the text. The pass signature is a hash of what was recorded, so this is
    // what stops the lookup cost scaling with the size of the shader. See the
    // contract on `pl_custom_shader.static_text` -- it is the caller's promise
    // that address identity is text identity.
    if (params->prelude) {
        GLSLP("// pl_shader_custom prelude: \n");
        if (params->static_text) {
            sh_append_const_str(sh, SH_BUF_PRELUDE, params->prelude);
            GLSLP("\n");
        } else {
            GLSLP("%s\n", params->prelude);
        }
    }
    if (params->header) {
        GLSLH("// pl_shader_custom header: \n");
        if (params->static_text) {
            sh_append_const_str(sh, SH_BUF_HEADER, params->header);
            GLSLH("\n");
        } else {
            GLSLH("%s\n", params->header);
        }
    }

    if (params->description)
        sh_describef(sh, "%s", params->description);

    if (params->body) {
        const char *output_decl = "";
        if (params->output != params->input) {
            switch (params->output) {
            case PL_SHADER_SIG_NONE: break;
            case PL_SHADER_SIG_COLOR:
                output_decl = "vec4 color = vec4(0.0);";
                break;

            case PL_SHADER_SIG_SAMPLER:
                pl_unreachable();
            }
        }

        if (params->static_text) {
            GLSL("// pl_shader_custom \n"
                 "%s                  \n"
                 "{                   \n",
                 output_decl);
            sh_append_const_str(sh, SH_BUF_BODY, params->body);
            GLSL("\n}                 \n");
        } else {
            GLSL("// pl_shader_custom \n"
                 "%s                  \n"
                 "{                   \n"
                 "%s                  \n"
                 "}                   \n",
                 output_decl, params->body);
        }
    }

    return true;
}
