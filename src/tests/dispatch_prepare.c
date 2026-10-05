#include "vulkan/gpu.h"
#include <libplacebo/vulkan.h>
#include <libplacebo/shaders/custom.h>
#include "pass_prepare_helpers.h"

static _Thread_local bool describe_only, forbid_compile;
static struct pl_gpu_fns original;
static struct spirv_compiler compiler;
static const struct spirv_compiler *original_compiler;
static PFN_vkCreateComputePipelines original_compute;
static PFN_vkCreateGraphicsPipelines original_graphics;
static atomic_int translations, pipelines, runs, destroys, live_buffers, live_timers;
static struct prepare_latch compile_latch;
static bool fail_run;

static pl_str observe_compile(pl_spirv spirv, void *alloc,
    struct pl_glsl_version version, enum glsl_shader_stage stage, const char *source)
{
    REQUIRE(!describe_only && !forbid_compile);
    atomic_fetch_add(&translations, 1);
    latch_block(&compile_latch);
    return original_compiler->compile(spirv, alloc, version, stage, source);
}

static VKAPI_ATTR VkResult VKAPI_CALL observe_compute(VkDevice device,
    VkPipelineCache cache, uint32_t count, const VkComputePipelineCreateInfo *info,
    const VkAllocationCallbacks *alloc, VkPipeline *result)
{
    REQUIRE(!describe_only && !forbid_compile);
    atomic_fetch_add(&pipelines, count);
    return original_compute(device, cache, count, info, alloc, result);
}

static VKAPI_ATTR VkResult VKAPI_CALL observe_graphics(VkDevice device,
    VkPipelineCache cache, uint32_t count, const VkGraphicsPipelineCreateInfo *info,
    const VkAllocationCallbacks *alloc, VkPipeline *result)
{
    REQUIRE(!describe_only && !forbid_compile);
    atomic_fetch_add(&pipelines, count);
    return original_graphics(device, cache, count, info, alloc, result);
}

static pl_buf observe_buf_create(pl_gpu gpu, const struct pl_buf_params *params)
{
    REQUIRE(!describe_only);
    pl_buf result = original.buf_create(gpu, params);
    if (result)
        atomic_fetch_add(&live_buffers, 1);
    return result;
}
static pl_tex observe_tex_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    REQUIRE(!describe_only);
    return original.tex_create(gpu, params);
}
static pl_timer observe_timer_create(pl_gpu gpu)
{
    REQUIRE(!describe_only);
    pl_timer result = original.timer_create(gpu);
    if (result)
        atomic_fetch_add(&live_timers, 1);
    return result;
}
static void observe_buf_destroy(pl_gpu gpu, pl_buf buffer)
{
    REQUIRE(!describe_only);
    atomic_fetch_sub(&live_buffers, 1);
    original.buf_destroy(gpu, buffer);
}
static void observe_timer_destroy(pl_gpu gpu, pl_timer timer)
{
    REQUIRE(!describe_only);
    atomic_fetch_sub(&live_timers, 1);
    original.timer_destroy(gpu, timer);
}
static void observe_buf_write(pl_gpu gpu, pl_buf buf, size_t offset,
                              const void *data, size_t size)
{
    REQUIRE(!describe_only);
    original.buf_write(gpu, buf, offset, data, size);
}
static pl_pass observe_pass_create(pl_gpu gpu, const struct pl_pass_params *params)
{
    REQUIRE(!describe_only && !forbid_compile);
    return original.pass_create(gpu, params);
}
static pl_pass observe_pass_prepare(pl_gpu gpu, const struct pl_pass_params *params,
                                    enum pl_pass_prepare_phase *phase)
{
    REQUIRE(!describe_only && !forbid_compile);
    return original.pass_create_prepared(gpu, params, phase);
}
static void observe_pass_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    REQUIRE(!describe_only);
    atomic_fetch_add(&runs, 1);
    original.pass_run(gpu, params);
}
static bool observe_prepared_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    REQUIRE(!describe_only);
    atomic_fetch_add(&runs, 1);
    return !fail_run && original.pass_run_prepared(gpu, params);
}
static void observe_pass_destroy(pl_gpu gpu, pl_pass pass)
{
    REQUIRE(!describe_only);
    atomic_fetch_add(&destroys, 1);
    original.pass_destroy(gpu, pass);
}

static pl_dispatch_preparation prepare(pl_dispatch_description *desc)
{
    pl_dispatch_preparation request = NULL;
    forbid_compile = true;
    REQUIRE(pl_dispatch_prepare_submit(desc, &request) == PL_PASS_PREPARE_ACCEPTED);
    forbid_compile = false;
    REQUIRE(!*desc && request);
    return request;
}

static enum pl_pass_prepare_state await_dispatch(pl_dispatch_preparation request)
{
    pl_clock_t start = pl_clock_now();
    enum pl_pass_prepare_state state;
    while ((state = pl_dispatch_prepare_poll(request)) == PL_PASS_PREPARE_PENDING) {
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 10.0);
        pl_thread_sleep(0.001);
    }
    return state;
}

static pl_dispatch_prepared take(pl_dispatch_preparation *request)
{
    REQUIRE(await_dispatch(*request) == PL_PASS_PREPARE_READY);
    pl_dispatch_prepared result = NULL;
    REQUIRE(pl_dispatch_prepare_take(request, &result) == PL_PASS_PREPARE_READY);
    REQUIRE(!*request && result);
    return result;
}

static void compare_descriptions(pl_dispatch_description a, pl_dispatch_description b)
{
    const struct pl_pass_params *x = pl_dispatch_description_params(a);
    const struct pl_pass_params *y = pl_dispatch_description_params(b);
    REQUIRE(x->type == y->type);
    REQUIRE(!strcmp(x->glsl_shader, y->glsl_shader));
    REQUIRE(!x->vertex_shader == !y->vertex_shader);
    if (x->vertex_shader)
        REQUIRE(!strcmp(x->vertex_shader, y->vertex_shader));
    REQUIRE(x->num_descriptors == y->num_descriptors);
    REQUIRE(x->num_constants == y->num_constants);
    REQUIRE(x->target_format == y->target_format);
    REQUIRE(x->vertex_stride == y->vertex_stride);
    REQUIRE(x->load_target == y->load_target);
}

static pl_shader sampled(pl_dispatch dp, pl_tex tex,
                         const struct pl_tex_params *metadata,
                         enum pl_tex_sample_mode mode, float gain)
{
    pl_shader sh = pl_dispatch_begin(dp);
    struct pl_shader_desc desc = {
        .desc = { .name = "source", .type = PL_DESC_SAMPLED_TEX },
        .binding = { .object = tex, .sample_mode = mode },
        .texture = metadata, .sampler_type = PL_SAMPLER_NORMAL,
    };
    struct pl_shader_var var = {
        .var = pl_var_float("gain"), .data = &gain, .dynamic = true,
    };
    REQUIRE(pl_shader_custom(sh, &(struct pl_custom_shader) {
        .body = "color = texture(source, vec2(0.5)) * gain;",
        .output = PL_SHADER_SIG_COLOR,
        .descriptors = &desc, .num_descriptors = 1,
        .variables = &var, .num_variables = 1,
    }));
    return sh;
}

static pl_shader constant_color(pl_dispatch dp, const char *body)
{
    pl_shader sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_custom(sh, &(struct pl_custom_shader) {
        .body = body, .output = PL_SHADER_SIG_COLOR,
    }));
    return sh;
}

static void read_pixels(pl_gpu gpu, pl_tex tex, uint8_t out[64])
{
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = tex, .ptr = out)));
}

static void test_sampled(pl_gpu gpu, pl_dispatch dp, pl_tex reference, pl_tex output)
{
    const uint8_t source_data[] = {0,0,0,255, 64,0,0,255, 128,0,0,255, 255,0,0,255};
    struct pl_tex_params src_meta = {
        .w = 2, .h = 2, .format = reference->params.format, .sampleable = true,
    };
    struct pl_tex_params target_meta = output->params;
    struct pl_tex_params allocated = src_meta;
    allocated.initial_data = source_data;
    pl_tex source = pl_tex_create(gpu, &allocated);
    REQUIRE(source);
    pl_dispatch_description desc = NULL, bound = NULL;
    int before_compile = atomic_load(&translations), before_pipeline = atomic_load(&pipelines);
    describe_only = forbid_compile = true;
    pl_shader sh = sampled(dp, NULL, &src_meta, PL_TEX_SAMPLE_LINEAR, 1.0f);
    memset(&src_meta, 0, sizeof(src_meta)); // shader owns metadata already
    REQUIRE(pl_dispatch_describe_finish(dp, pl_dispatch_params(.shader = &sh),
                                        &target_meta, &desc) == PL_DISPATCH_OK);
    REQUIRE(!sh);
    sh = sampled(dp, source, NULL, PL_TEX_SAMPLE_LINEAR, 1.0f);
    REQUIRE(pl_dispatch_describe_finish(dp,
        pl_dispatch_params(.shader = &sh, .target = output), NULL, &bound) == PL_DISPATCH_OK);
    describe_only = forbid_compile = false;
    REQUIRE(atomic_load(&translations) == before_compile);
    REQUIRE(atomic_load(&pipelines) == before_pipeline);
    compare_descriptions(desc, bound);
    pl_dispatch_description_destroy(&bound);
    pl_dispatch_preparation request = prepare(&desc);
    pl_dispatch_prepared pass = take(&request);

    // Sampler mode and runtime uniform values can change without a new pass.
    for (int mode = 0; mode < 2; mode++) {
        float gain = mode ? 0.5f : 1.0f;
        enum pl_tex_sample_mode sampling = mode ? PL_TEX_SAMPLE_NEAREST : PL_TEX_SAMPLE_LINEAR;
        sh = sampled(dp, source, NULL, sampling, gain);
        REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = reference)));
        uint8_t expected[64], actual[64];
        read_pixels(gpu, reference, expected);
        REQUIRE(abs((int) expected[0] - (mode ? 128 : 112)) <= 1);
        before_compile = atomic_load(&translations);
        before_pipeline = atomic_load(&pipelines);
        forbid_compile = true;
        sh = sampled(dp, source, NULL, sampling, gain);
        REQUIRE(pl_dispatch_finish_prepared(dp, pass,
            pl_dispatch_params(.shader = &sh, .target = output)) == PL_DISPATCH_OK);
        forbid_compile = false;
        read_pixels(gpu, output, actual);
        REQUIRE_MEMEQ(actual, expected, sizeof(actual));
        REQUIRE(atomic_load(&translations) == before_compile);
        REQUIRE(atomic_load(&pipelines) == before_pipeline);
    }
    pl_dispatch_prepared_destroy(&pass);
    pl_tex_destroy(gpu, &source);
}

static pl_shader compute_shader(pl_dispatch dp, pl_buf output,
    const struct pl_buf_params *metadata, uint32_t runtime, uint32_t specialization)
{
    pl_shader sh = pl_dispatch_begin(dp);
    struct pl_buffer_var value = { .var = pl_var_uint("value") };
    value.layout = pl_std430_layout(0, &value.var);
    struct pl_shader_desc desc = {
        .desc = { .name = "output", .type = PL_DESC_BUF_STORAGE,
                  .access = PL_DESC_ACCESS_WRITEONLY },
        .binding.object = output, .buffer = metadata,
        .buffer_vars = &value, .num_buffer_vars = 1,
    };
    float padding[256] = { [255] = 2 };
    struct pl_shader_var vars[] = {
        { .var = pl_var_uint("runtime"), .data = &runtime, .dynamic = true },
        { .var = pl_var_float("padding"), .data = padding, .dynamic = true },
    };
    vars[1].var.dim_a = PL_ARRAY_SIZE(padding); // force entry-owned UBO storage
    struct pl_shader_const spec = {
        .name = "specialization", .type = PL_VAR_UINT,
        .data = &specialization, .compile_time = true,
    };
    REQUIRE(pl_shader_custom(sh, &(struct pl_custom_shader) {
        .body = "value = runtime + specialization + uint(padding[255]);",
        .compute = true, .compute_group_size = {1, 1},
        .descriptors = &desc, .num_descriptors = 1,
        .variables = vars, .num_variables = PL_ARRAY_SIZE(vars),
        .constants = &spec, .num_constants = 1,
    }));
    return sh;
}

static pl_dispatch_description describe_compute(pl_dispatch dp,
    const struct pl_buf_params *metadata, uint32_t spec)
{
    describe_only = forbid_compile = true;
    pl_shader sh = compute_shader(dp, NULL, metadata, 5, spec);
    pl_dispatch_description desc = NULL;
    REQUIRE(pl_dispatch_describe_compute(dp,
        pl_dispatch_compute_params(.shader = &sh, .dispatch_size = {1,1,1}),
        &desc) == PL_DISPATCH_OK);
    REQUIRE(!sh);
    describe_only = forbid_compile = false;
    const struct pl_pass_params *params = pl_dispatch_description_params(desc);
    REQUIRE(params->type == PL_PASS_COMPUTE && params->num_constants == 1);
    bool has_ubo = false;
    for (int i = 0; i < params->num_descriptors; i++)
        has_ubo |= params->descriptors[i].type == PL_DESC_BUF_UNIFORM;
    REQUIRE(has_ubo);
    return desc;
}

static enum pl_dispatch_result run_compute(pl_dispatch dp, pl_dispatch_prepared pass,
    pl_buf output, uint32_t runtime, uint32_t specialization)
{
    forbid_compile = true;
    pl_shader sh = compute_shader(dp, output, NULL, runtime, specialization);
    enum pl_dispatch_result result = pl_dispatch_compute_prepared(dp, pass,
        pl_dispatch_compute_params(.shader = &sh, .dispatch_size = {1,1,1}));
    REQUIRE(!sh);
    forbid_compile = false;
    return result;
}

static void test_compute(pl_gpu gpu, pl_dispatch dp)
{
    struct pl_buf_params metadata = {
        .size = sizeof(uint32_t), .storable = true, .host_readable = true,
    };
    pl_buf output = pl_buf_create(gpu, &metadata);
    REQUIRE(output);
    pl_dispatch_description desc = describe_compute(dp, &metadata, 7);
    latch_arm(&compile_latch);
    pl_dispatch_preparation request = prepare(&desc);
    latch_entered(&compile_latch);
    REQUIRE(pl_dispatch_prepare_poll(request) == PL_PASS_PREPARE_PENDING);
    pl_dispatch_prepared pass = NULL;
    REQUIRE(pl_dispatch_prepare_take(&request, &pass) == PL_PASS_PREPARE_PENDING);
    REQUIRE(request && !pass);
    int before_runs = atomic_load(&runs);
    REQUIRE(run_compute(dp, NULL, output, 5, 7) == PL_DISPATCH_NOT_READY);
    REQUIRE(atomic_load(&runs) == before_runs);
    latch_release(&compile_latch);
    pass = take(&request);
    int before_compile = atomic_load(&translations), before_pipeline = atomic_load(&pipelines);
    for (uint32_t runtime = 5; runtime <= 9; runtime += 4) {
        REQUIRE(run_compute(dp, pass, output, runtime, 7) == PL_DISPATCH_OK);
        uint32_t value = 0;
        REQUIRE(pl_buf_read(gpu, output, 0, &value, sizeof(value)));
        REQUIRE(value == runtime + 9);
    }
    before_runs = atomic_load(&runs);
    describe_only = true; // mismatches must not even upload uniforms
    REQUIRE(run_compute(dp, pass, output, 5, 8) == PL_DISPATCH_NOT_READY);
    REQUIRE(atomic_load(&runs) == before_runs);
    describe_only = false;
    pl_dispatch other = pl_dispatch_create(gpu->log, gpu);
    REQUIRE(run_compute(other, pass, output, 5, 7) == PL_DISPATCH_INVALID);
    REQUIRE(atomic_load(&runs) == before_runs);
    pl_dispatch_destroy(&other);
    // Runtime validation must preserve INVALID, without reaching the backend.
    before_runs = atomic_load(&runs);
    forbid_compile = true;
    pl_shader invalid = compute_shader(dp, output, NULL, 5, 7);
    enum pl_dispatch_result invalid_result = pl_dispatch_compute_prepared(dp, pass,
        pl_dispatch_compute_params(.shader = &invalid, .dispatch_size = {-1,1,1}));
    forbid_compile = false;
    REQUIRE(!invalid);
    REQUIRE(atomic_load(&runs) == before_runs);
    REQUIRE_CMP(invalid_result, ==, PL_DISPATCH_INVALID, "d");
    fail_run = true;
    REQUIRE(run_compute(dp, pass, output, 5, 7) == PL_DISPATCH_FAILED);
    fail_run = false;
    REQUIRE(atomic_load(&translations) == before_compile);
    REQUIRE(atomic_load(&pipelines) == before_pipeline);
    pl_dispatch_prepared_destroy(&pass);
    pl_buf_destroy(gpu, &output);
}

static void test_cancel_and_retry(pl_dispatch dp)
{
    const struct pl_buf_params metadata = { .size = sizeof(uint32_t), .storable = true };
    int buffers = atomic_load(&live_buffers), timers = atomic_load(&live_timers);
    pl_dispatch_preparation requests[128] = {0};
    pl_dispatch_description rejected = describe_compute(dp, &metadata, 7);
    latch_arm(&compile_latch);
    requests[0] = prepare(&rejected);
    latch_entered(&compile_latch);
    int count = 1;
    for (; count < PL_ARRAY_SIZE(requests); count++) {
        rejected = describe_compute(dp, &metadata, 7);
        pl_dispatch_description identity = rejected;
        forbid_compile = true;
        enum pl_pass_prepare_result result = pl_dispatch_prepare_submit(&rejected, &requests[count]);
        forbid_compile = false;
        if (result == PL_PASS_PREPARE_CAPACITY) {
            REQUIRE(rejected == identity && !requests[count]);
            break;
        }
        REQUIRE(result == PL_PASS_PREPARE_ACCEPTED && !rejected);
    }
    REQUIRE(count < PL_ARRAY_SIZE(requests));
    // Rejected admission must release its staged resources. Cancellation and
    // release free every entry-owned UBO/timer while driver work stays blocked.
    REQUIRE(atomic_load(&live_buffers) == buffers + count);
    REQUIRE(atomic_load(&live_timers) == timers + count);
    for (int i = 0; i < count; i++) {
        pl_dispatch_prepare_cancel(requests[i]);
        REQUIRE(pl_dispatch_prepare_poll(requests[i]) == PL_PASS_PREPARE_CANCELLED);
        pl_dispatch_prepare_release(&requests[i]);
        REQUIRE(!requests[i]);
    }
    REQUIRE(atomic_load(&live_buffers) == buffers);
    REQUIRE(atomic_load(&live_timers) == timers);
    latch_release(&compile_latch);
    pl_clock_t start = pl_clock_now();
    pl_dispatch_preparation retry = NULL;
    for (;;) {
        forbid_compile = true;
        enum pl_pass_prepare_result result = pl_dispatch_prepare_submit(&rejected, &retry);
        forbid_compile = false;
        if (result == PL_PASS_PREPARE_ACCEPTED)
            break;
        REQUIRE(result == PL_PASS_PREPARE_CAPACITY && rejected && !retry);
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 10.0);
        pl_thread_sleep(0.001);
    }
    REQUIRE(!rejected);
    pl_dispatch_prepared pass = take(&retry);
    pl_dispatch_prepared_destroy(&pass);
    REQUIRE(atomic_load(&live_buffers) == buffers);
    REQUIRE(atomic_load(&live_timers) == timers);
}

static void test_gpu_less_vertex(pl_gpu gpu, pl_dispatch dp, pl_tex output)
{
    // Explicit vertex formats permit this legacy path without shader.gpu.
    pl_shader sh = pl_shader_alloc(gpu->log, NULL);
    REQUIRE(pl_shader_custom(sh, &(struct pl_custom_shader) {
        .body = "color = vec4(0.25, 0, 0, 1);", .output = PL_SHADER_SIG_COLOR,
    }));
    const float vertices[] = {-1,-1, 3,-1, -1,3};
    struct pl_vertex_attrib attribute = {
        .name = "position", .fmt = pl_find_named_fmt(gpu, "rg32f"),
    };
    REQUIRE(pl_dispatch_vertex(dp, pl_dispatch_vertex_params(
        .shader = &sh, .target = output,
        .vertex_attribs = &attribute, .num_vertex_attribs = 1,
        .vertex_stride = sizeof(float[2]), .vertex_coords = PL_COORDS_NORMALIZED,
        .vertex_type = PL_PRIM_TRIANGLE_LIST, .vertex_count = 3,
        .vertex_data = vertices,
    )));
    REQUIRE(!sh);
    uint8_t pixels[64];
    read_pixels(gpu, output, pixels);
    REQUIRE(pixels[0] == 64 && pixels[1] == 0 && pixels[2] == 0 && pixels[3] == 255);
}

static void test_vertex_and_gc(pl_gpu gpu, pl_dispatch dp, pl_tex reference, pl_tex output)
{
    const char body[] = "color = vec4(0.25, 0.5, 0.75, 1);";
    struct pl_vertex_attrib attrib = {
        .name = "position", .fmt = pl_find_named_fmt(gpu, "rg32f"),
    };
    float vertices[] = {0,0, 8,0, 0,8};
    struct pl_dispatch_vertex_params params = {
        .vertex_attribs = &attrib, .num_vertex_attribs = 1,
        .vertex_stride = 2 * sizeof(float), .vertex_type = PL_PRIM_TRIANGLE_LIST,
        .vertex_coords = PL_COORDS_ABSOLUTE, .vertex_count = 3,
        .scissors = {1,1,3,3},
    };
    pl_dispatch_description desc = NULL;
    describe_only = forbid_compile = true;
    pl_shader sh = constant_color(dp, body);
    params.shader = &sh;
    REQUIRE(pl_dispatch_describe_vertex(dp, &params, &output->params, &desc) == PL_DISPATCH_OK);
    describe_only = forbid_compile = false;
    pl_dispatch_preparation request = prepare(&desc);
    pl_dispatch_prepared pass = take(&request);
    params.vertex_data = vertices;
    params.target = reference;
    sh = constant_color(dp, body);
    pl_tex_clear(gpu, reference, (float[4]) {0});
    REQUIRE(pl_dispatch_vertex(dp, &params));
    uint8_t expected[64], actual[64];
    read_pixels(gpu, reference, expected);
    REQUIRE(expected[0] == 0 && expected[(1*4+1)*4] == 64);
    for (int after_gc = 0; after_gc < 2; after_gc++) {
        params.target = output;
        pl_tex_clear(gpu, output, (float[4]) {0});
        int before_compile = atomic_load(&translations), before_pipeline = atomic_load(&pipelines);
        forbid_compile = true;
        sh = constant_color(dp, body);
        REQUIRE(pl_dispatch_vertex_prepared(dp, pass, &params) == PL_DISPATCH_OK);
        forbid_compile = false;
        read_pixels(gpu, output, actual);
        REQUIRE_MEMEQ(actual, expected, sizeof(actual));
        REQUIRE(atomic_load(&translations) == before_compile);
        REQUIRE(atomic_load(&pipelines) == before_pipeline);
        if (after_gc)
            break;
        // Fill the ordinary dispatch cache, age its entries, then verify actual
        // eviction occurred. The separately owned prepared entry must survive.
        for (int i = 0; i < 110; i++) {
            char unique[96];
            snprintf(unique, sizeof(unique), "color = vec4(%d.0 / 128.0, 0, 0, 1);", i);
            sh = constant_color(dp, unique);
            REQUIRE(pl_dispatch_finish(dp,
                pl_dispatch_params(.shader = &sh, .target = reference)));
        }
        int before_destroy = atomic_load(&destroys);
        for (int frame = 0; frame < 12; frame++)
            pl_dispatch_reset_frame(dp);
        // Trigger cache pressure after entries have aged.
        for (int i = 110; i < 220; i++) {
            char unique[96];
            snprintf(unique, sizeof(unique), "color = vec4(%d.0 / 256.0, 0, 0, 1);", i);
            sh = constant_color(dp, unique);
            REQUIRE(pl_dispatch_finish(dp,
                pl_dispatch_params(.shader = &sh, .target = reference)));
        }
        pl_dispatch_reset_frame(dp);
        REQUIRE(atomic_load(&destroys) > before_destroy);
    }
    pl_dispatch_prepared_destroy(&pass);
}

static pl_shader image_compute(pl_dispatch dp)
{
    pl_shader sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_custom(sh, &(struct pl_custom_shader) {
        .body = "color = vec4(0.25, 0.5, 0.75, 1);",
        .output = PL_SHADER_SIG_COLOR,
        .compute = true, .compute_group_size = {2, 2},
    }));
    return sh;
}

static void test_finish_compute(pl_gpu gpu, pl_dispatch dp)
{
    struct pl_tex_params metadata = {
        .w = 4, .h = 4, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .storable = true, .host_readable = true, .blit_dst = true,
    };
    pl_tex reference = pl_tex_create(gpu, &metadata), output = pl_tex_create(gpu, &metadata);
    REQUIRE(reference && output);
    pl_rect2d rect = {1, 1, 3, 3};
    pl_dispatch_description desc = NULL, bound = NULL;
    describe_only = forbid_compile = true;
    pl_shader sh = image_compute(dp);
    REQUIRE(pl_dispatch_describe_finish(dp,
        pl_dispatch_params(.shader = &sh, .rect = rect), &metadata, &desc) == PL_DISPATCH_OK);
    sh = image_compute(dp);
    REQUIRE(pl_dispatch_describe_finish(dp,
        pl_dispatch_params(.shader = &sh, .rect = rect, .target = output), NULL,
        &bound) == PL_DISPATCH_OK);
    describe_only = forbid_compile = false;
    compare_descriptions(desc, bound);
    const struct pl_pass_params *normalized = pl_dispatch_description_params(desc);
    REQUIRE(normalized->type == PL_PASS_COMPUTE);
    REQUIRE(normalized->num_descriptors == 1);
    REQUIRE(normalized->descriptors[0].type == PL_DESC_STORAGE_IMG);
    pl_dispatch_description_destroy(&bound);
    pl_dispatch_preparation request = prepare(&desc);
    pl_dispatch_prepared pass = take(&request);
    pl_tex_clear(gpu, reference, (float[4]){0});
    pl_tex_clear(gpu, output, (float[4]){0});
    sh = image_compute(dp);
    REQUIRE(pl_dispatch_finish(dp,
        pl_dispatch_params(.shader = &sh, .rect = rect, .target = reference)));
    int before_compile = atomic_load(&translations), before_pipeline = atomic_load(&pipelines);
    forbid_compile = true;
    sh = image_compute(dp);
    REQUIRE(pl_dispatch_finish_prepared(dp, pass,
        pl_dispatch_params(.shader = &sh, .rect = rect, .target = output)) == PL_DISPATCH_OK);
    forbid_compile = false;
    uint8_t expected[64], actual[64];
    read_pixels(gpu, reference, expected);
    read_pixels(gpu, output, actual);
    REQUIRE(expected[0] == 0 && expected[(1*4+1)*4] == 64);
    REQUIRE_MEMEQ(actual, expected, sizeof(actual));
    REQUIRE(atomic_load(&translations) == before_compile);
    REQUIRE(atomic_load(&pipelines) == before_pipeline);
    pl_dispatch_prepared_destroy(&pass);
    pl_tex_destroy(gpu, &reference);
    pl_tex_destroy(gpu, &output);
}

static pl_shader static_color(pl_dispatch dp, const char *text)
{
    pl_shader sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_custom(sh, &(struct pl_custom_shader) {
        .body = text, .output = PL_SHADER_SIG_COLOR, .static_text = true,
    }));
    return sh;
}

static void test_static_text(pl_gpu gpu, pl_dispatch dp, pl_tex output)
{
    static const char first[] = "color = vec4(0.5, 0, 0, 1);";
    static const char second[] = "color = vec4(0.5, 0, 0, 1);";
    REQUIRE(&first[0] != &second[0]);
    describe_only = forbid_compile = true;
    pl_shader sh = static_color(dp, first);
    pl_dispatch_description desc = NULL;
    REQUIRE(pl_dispatch_describe_finish(dp, pl_dispatch_params(.shader = &sh),
                                        &output->params, &desc) == PL_DISPATCH_OK);
    describe_only = forbid_compile = false;
    pl_dispatch_preparation request = prepare(&desc);
    pl_dispatch_prepared pass = take(&request);
    int before_compile = atomic_load(&translations), before_pipeline = atomic_load(&pipelines);
    forbid_compile = true;
    sh = static_color(dp, second);
    REQUIRE(pl_dispatch_finish_prepared(dp, pass,
        pl_dispatch_params(.shader = &sh, .target = output)) == PL_DISPATCH_OK);
    forbid_compile = false;
    uint8_t actual[64];
    read_pixels(gpu, output, actual);
    REQUIRE(abs((int) actual[0] - 128) <= 1);
    REQUIRE(actual[1] == 0 && actual[2] == 0 && actual[3] == 255);
    REQUIRE(atomic_load(&translations) == before_compile);
    REQUIRE(atomic_load(&pipelines) == before_pipeline);
    pl_dispatch_prepared_destroy(&pass);
}

static void test_failure(pl_dispatch dp, pl_tex target)
{
    describe_only = forbid_compile = true;
    pl_shader sh = constant_color(dp, "this is not GLSL;");
    pl_dispatch_description desc = NULL;
    REQUIRE(pl_dispatch_describe_finish(dp, pl_dispatch_params(.shader = &sh),
                                        &target->params, &desc) == PL_DISPATCH_OK);
    describe_only = forbid_compile = false;
    pl_dispatch_preparation request = prepare(&desc);
    REQUIRE(await_dispatch(request) == PL_PASS_PREPARE_FAILED);
    REQUIRE(pl_dispatch_prepare_failure_phase(request) == PL_PASS_PREPARE_PHASE_TRANSLATION);
    REQUIRE(pl_dispatch_prepare_error(request) && *pl_dispatch_prepare_error(request));
    pl_dispatch_prepared pass = NULL;
    REQUIRE(pl_dispatch_prepare_take(&request, &pass) == PL_PASS_PREPARE_FAILED);
    REQUIRE(request && !pass);
    pl_dispatch_prepare_release(&request);
}

int main(void)
{
    latch_init(&compile_latch);
    pl_log log = pl_test_logger();
    pl_vulkan vk = pl_vulkan_create(log, pl_vulkan_params(.allow_software = true));
    if (!vk)
        return SKIP;
    pl_gpu gpu = vk->gpu;
    struct pl_vk *priv = PL_PRIV(gpu);
    original = priv->impl;
    priv->impl.buf_create = observe_buf_create;
    priv->impl.tex_create = observe_tex_create;
    priv->impl.buf_destroy = observe_buf_destroy;
    priv->impl.timer_destroy = observe_timer_destroy;
    priv->impl.timer_create = observe_timer_create;
    priv->impl.buf_write = observe_buf_write;
    priv->impl.pass_create = observe_pass_create;
    priv->impl.pass_create_prepared = observe_pass_prepare;
    priv->impl.pass_run = observe_pass_run;
    priv->impl.pass_run_prepared = observe_prepared_run;
    priv->impl.pass_destroy = observe_pass_destroy;
    original_compiler = priv->spirv->impl;
    compiler = *original_compiler;
    compiler.compile = observe_compile;
    ((struct pl_spirv_t *) priv->spirv)->impl = &compiler;
    original_compute = priv->vk->CreateComputePipelines;
    original_graphics = priv->vk->CreateGraphicsPipelines;
    priv->vk->CreateComputePipelines = observe_compute;
    priv->vk->CreateGraphicsPipelines = observe_graphics;
    pl_dispatch dp = pl_dispatch_create(log, gpu);
    struct pl_tex_params params = {
        .w = 4, .h = 4, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true, .blit_dst = true,
    };
    pl_tex reference = pl_tex_create(gpu, &params), output = pl_tex_create(gpu, &params);
    REQUIRE(reference && output);
    test_sampled(gpu, dp, reference, output);
    test_compute(gpu, dp);
    test_cancel_and_retry(dp);
    test_finish_compute(gpu, dp);
    test_static_text(gpu, dp, output);
    test_vertex_and_gc(gpu, dp, reference, output);
    test_failure(dp, output);
    test_gpu_less_vertex(gpu, dp, output);
    pl_dispatch_destroy(&dp);
    pl_tex_destroy(gpu, &reference);
    pl_tex_destroy(gpu, &output);
    pl_pass_prepare_uninit(gpu);
    priv->vk->CreateComputePipelines = original_compute;
    priv->vk->CreateGraphicsPipelines = original_graphics;
    ((struct pl_spirv_t *) priv->spirv)->impl = original_compiler;
    pl_vulkan_destroy(&vk);
    pl_log_destroy(&log);
    latch_destroy(&compile_latch);
}
