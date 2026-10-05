#include "shaders.h"
#include "utils.h"
#include <libplacebo/dummy.h>
#include <libplacebo/shaders/dithering.h>
#include <libplacebo/shaders/sampling.h>

static pl_tex (*real_create)(pl_gpu, const struct pl_tex_params *);
static bool (*real_upload)(pl_gpu, const struct pl_tex_transfer_params *);
static int creates, uploads, fills;
static bool fail_create;

static pl_tex count_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    creates++;
    return fail_create ? NULL : real_create(gpu, params);
}

static bool count_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    uploads++;
    return real_upload(gpu, params);
}

struct collected {
    pl_shader_obj lut[8];
    int count;
};

static void collect(void *priv, pl_shader_obj lut)
{
    struct collected *c = priv;
    REQUIRE(c->count < PL_ARRAY_SIZE(c->lut));
    c->lut[c->count++] = lut;
}

static pl_shader shader(pl_gpu gpu, bool describe, struct collected *c)
{
    return pl_shader_alloc(gpu->log, pl_shader_params(
        .gpu = gpu, .description_only = describe,
        .describe_lut = collect, .describe_priv = c,
    ));
}

static void fill(void *data, const struct sh_lut_params *params)
{
    fills++;
    for (int i = 0; i < params->width; i++)
        ((float *) data)[i] = i + *(float *) params->priv;
}

static void compare(pl_shader a, pl_shader b)
{
    const struct pl_shader_res *ra = pl_shader_finalize(a);
    const struct pl_shader_res *rb = pl_shader_finalize(b);
    REQUIRE(ra && rb);
    REQUIRE(!strcmp(ra->glsl, rb->glsl));
    REQUIRE(ra->num_descriptors == rb->num_descriptors);
    for (int i = 0; i < ra->num_descriptors; i++) {
        const struct pl_shader_desc *da = &ra->descriptors[i];
        const struct pl_shader_desc *db = &rb->descriptors[i];
        REQUIRE(da->desc.type == db->desc.type);
        REQUIRE(da->binding.sample_mode == db->binding.sample_mode);
        const struct pl_tex_params *ta = da->binding.object
            ? &((pl_tex) da->binding.object)->params : da->texture;
        const struct pl_tex_params *tb = db->binding.object
            ? &((pl_tex) db->binding.object)->params : db->texture;
        REQUIRE(ta && tb);
        REQUIRE(ta->w == tb->w && ta->h == tb->h && ta->d == tb->d);
        REQUIRE(ta->format == tb->format);
    }
}

static void test_lut(pl_gpu gpu, enum sh_lut_type type, bool dynamic)
{
    float base = 0.25f;
    struct collected c = {0};
    pl_shader_obj active = NULL, candidate = NULL;
    struct sh_lut_params params = {
        .object = &active, .var_type = PL_VAR_FLOAT, .lut_type = type,
        .width = 16, .comps = 1, .signature = 41, .fill = fill, .priv = &base,
        .dynamic = dynamic,
    };
    pl_shader real = shader(gpu, false, NULL);
    REQUIRE(sh_lut(real, &params));
    const int before_create = creates, before_upload = uploads;
    pl_shader desc = shader(gpu, true, &c);
    params.object = &candidate;
    REQUIRE(sh_lut(desc, &params));
    REQUIRE(creates == before_create && uploads == before_upload);
    REQUIRE(c.count == 1 && c.lut[0] == candidate);
    compare(real, desc);
    const struct pl_shader_res *rr = pl_shader_finalize(real);
    const struct pl_shader_res *rd = pl_shader_finalize(desc);
    if (type == SH_LUT_TEXTURE) {
        REQUIRE(rr->num_descriptors == 1 && rd->num_descriptors == 1);
        REQUIRE(!rd->descriptors[0].binding.object);
        REQUIRE(!rd->descriptors[0].texture->initial_data);
    }

    pl_shader premature = shader(gpu, false, NULL);
    REQUIRE(!sh_lut(premature, &params));
    REQUIRE(pl_shader_is_failed(premature));
    REQUIRE(creates == before_create && uploads == before_upload);
    pl_shader_free(&premature);

    // Mutation of caller data cannot change what was staged. Failed staging
    // also leaves the active texture and its contents untouched.
    base = 1000.0f;
    if (type == SH_LUT_TEXTURE) {
        fail_create = true;
        REQUIRE(!sh_lut_stage(candidate));
        fail_create = false;
    }
    REQUIRE(sh_lut_stage(candidate));
    const int staged_create = creates, staged_upload = uploads, staged_fill = fills;
    REQUIRE(sh_lut_stage(candidate));
    REQUIRE(creates == staged_create && uploads == staged_upload);
    pl_shader execute = shader(gpu, false, NULL);
    REQUIRE(sh_lut(execute, &params));
    compare(real, execute);
    if (type == SH_LUT_TEXTURE) {
        pl_tex old = rr->descriptors[0].binding.object;
        pl_tex staged = pl_shader_finalize(execute)->descriptors[0].binding.object;
        REQUIRE(old != staged);
        REQUIRE(!memcmp(pl_tex_dummy_data(old), pl_tex_dummy_data(staged),
                        old->params.w * old->params.format->texel_size));
    }

    // Matching candidate preflight is read-only after readiness.
    pl_shader preflight = shader(gpu, true, &c);
    REQUIRE(sh_lut(preflight, &params));
    compare(desc, preflight);
    REQUIRE(creates == staged_create && uploads == staged_upload && fills == staged_fill);

    // Invalid metadata rejected before the staged-update guard must not poison
    // the existing candidate's error latch or prevent its original execution.
    pl_shader invalid_format = shader(gpu, true, &c);
    params.fmt = pl_find_fmt(gpu, PL_FMT_UINT, 1, 32, 32, PL_FMT_CAP_SAMPLEABLE);
    REQUIRE(params.fmt && params.fmt->type == PL_FMT_UINT);
    REQUIRE(!sh_lut(invalid_format, &params));
    params.fmt = NULL;
    pl_shader restored = shader(gpu, true, &c);
    REQUIRE(sh_lut(restored, &params));
    compare(desc, restored);
    pl_shader restored_execute = shader(gpu, false, NULL);
    REQUIRE(sh_lut(restored_execute, &params));
    compare(real, restored_execute);
    REQUIRE(sh_lut_stage(candidate));
    REQUIRE(creates == staged_create && uploads == staged_upload && fills == staged_fill);
    pl_shader_free(&restored_execute);
    pl_shader_free(&restored);
    pl_shader_free(&invalid_format);

    // A changed prepared resource is rejected instead of being rewritten.
    pl_shader changed = shader(gpu, true, &c);
    params.signature++;
    REQUIRE(!sh_lut(changed, &params));
    REQUIRE(pl_shader_is_failed(changed));
    REQUIRE(creates == staged_create && uploads == staged_upload && fills == staged_fill);

    // Ordinary live shader objects cannot accidentally become candidate state.
    pl_shader live = shader(gpu, true, &c);
    params.object = &active;
    REQUIRE(!sh_lut(live, &params));
    REQUIRE(pl_shader_is_failed(live));
    REQUIRE(creates == staged_create && uploads == staged_upload && fills == staged_fill);

    pl_shader_free(&live);
    pl_shader_free(&changed);
    pl_shader_free(&preflight);
    pl_shader_free(&execute);
    pl_shader_free(&desc);
    pl_shader_free(&real);
    pl_shader_obj_destroy(&candidate);
    pl_shader_obj_destroy(&active);
}

static void emit_default(pl_shader sh, pl_tex src, pl_shader_obj *state, bool ewa)
{
    if (ewa) {
        REQUIRE(pl_shader_sample_polar(sh, pl_sample_src(
            .tex = src, .new_w = 48, .new_h = 40,
        ), pl_sample_filter_params(.filter = pl_filter_ewa_lanczos, .lut = state)));
    } else {
        pl_shader_dither(sh, 8, state, NULL);
    }
}

static void test_defaults(pl_gpu gpu, bool ewa)
{
    pl_tex src = pl_tex_create(gpu, pl_tex_params(
        .w = 32, .h = 32, .format = pl_find_named_fmt(gpu, "rgba8"), .sampleable = true,
    ));
    REQUIRE(src);
    pl_shader_obj active = NULL, candidate = NULL;
    struct collected c = {0};
    pl_shader real = shader(gpu, false, NULL);
    emit_default(real, src, &active, ewa);
    const int before_create = creates, before_upload = uploads;
    pl_shader desc = shader(gpu, true, &c);
    emit_default(desc, src, &candidate, ewa);
    REQUIRE(c.count == 1);
    REQUIRE(creates == before_create && uploads == before_upload);
    compare(real, desc);
    REQUIRE(sh_lut_stage(c.lut[0]));
    const int staged_create = creates, staged_upload = uploads;
    pl_shader execute = shader(gpu, false, NULL);
    emit_default(execute, src, &candidate, ewa);
    compare(real, execute);
    pl_shader preflight = shader(gpu, true, &c);
    emit_default(preflight, src, &candidate, ewa);
    compare(desc, preflight);
    REQUIRE(creates == staged_create && uploads == staged_upload);
    pl_shader_free(&preflight);
    pl_shader_free(&execute);
    pl_shader_free(&desc);
    pl_shader_free(&real);
    pl_shader_obj_destroy(&active);
    pl_shader_obj_destroy(&candidate);
    pl_tex_destroy(gpu, &src);
}

int main(void)
{
    pl_log log = pl_test_logger();
    pl_gpu gpu = pl_gpu_dummy_create(log, NULL);
    struct pl_gpu_fns *fns = PL_PRIV(gpu);
    real_create = fns->tex_create;
    real_upload = fns->tex_upload;
    fns->tex_create = count_create;
    fns->tex_upload = count_upload;
    test_lut(gpu, SH_LUT_TEXTURE, false);
    test_lut(gpu, SH_LUT_TEXTURE, true);
    test_lut(gpu, SH_LUT_UNIFORM, false);
    test_lut(gpu, SH_LUT_LITERAL, false);
    test_defaults(gpu, false);
    test_defaults(gpu, true);
    pl_gpu_dummy_destroy(&gpu);
    pl_log_destroy(&log);
}
