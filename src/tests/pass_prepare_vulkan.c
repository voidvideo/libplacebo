#include "vulkan/gpu.h"
#include <libplacebo/vulkan.h>
#include "pass_prepare_helpers.h"

static _Thread_local bool caller_thread;
static struct prepare_latch translation, pipeline;
static struct spirv_compiler compiler;
static const struct spirv_compiler *original_compiler;
static PFN_vkCreateComputePipelines original_compute;
static PFN_vkCreateGraphicsPipelines original_graphics;
static atomic_int translations, pipelines;
static bool fail_pipeline;

static pl_str observed_compile(pl_spirv spirv, void *alloc,
                                struct pl_glsl_version version,
                                enum glsl_shader_stage stage, const char *source)
{
    REQUIRE(!caller_thread);
    atomic_fetch_add(&translations, 1);
    latch_block(&translation);
    return original_compiler->compile(spirv, alloc, version, stage, source);
}

static VKAPI_ATTR VkResult VKAPI_CALL observed_pipeline(
    VkDevice device, VkPipelineCache cache, uint32_t count,
    const VkComputePipelineCreateInfo *info, const VkAllocationCallbacks *alloc,
    VkPipeline *result)
{
    REQUIRE(!caller_thread);
    atomic_fetch_add(&pipelines, count);
    latch_block(&pipeline);
    if (fail_pipeline)
        return VK_ERROR_UNKNOWN;
    return original_compute(device, cache, count, info, alloc, result);
}

static VKAPI_ATTR VkResult VKAPI_CALL observed_graphics(
    VkDevice device, VkPipelineCache cache, uint32_t count,
    const VkGraphicsPipelineCreateInfo *info, const VkAllocationCallbacks *alloc,
    VkPipeline *result)
{
    REQUIRE(!caller_thread);
    atomic_fetch_add(&pipelines, count);
    return original_graphics(device, cache, count, info, alloc, result);
}

static void test_raster(pl_gpu gpu)
{
    pl_fmt format = pl_find_named_fmt(gpu, "rgba8");
    pl_fmt vertex_format = pl_find_named_fmt(gpu, "rg32f");
    REQUIRE(format && vertex_format);
    pl_tex target = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = format,
        .renderable = true, .host_readable = true,
    ));
    REQUIRE(target);
    float value = 0.25f;
    struct pl_constant spec = { .type = PL_VAR_FLOAT, .id = 0 };
    struct pl_vertex_attrib attrib = { .name = "position", .fmt = vertex_format };
    pl_pass_preparation request = NULL;
    REQUIRE(pl_pass_prepare_submit(gpu, pl_pass_params(
        .type = PL_PASS_RASTER,
        .glsl_shader = "#version 450\n"
                       "layout(constant_id=0) const float red = 0.5;\n"
                       "layout(location=0) out vec4 color;\n"
                       "void main() { color = vec4(red, 0, 0, 1); }\n",
        .vertex_shader = "#version 450\n"
                         "layout(location=0) in vec2 position;\n"
                         "void main() { gl_Position = vec4(position, 0, 1); }\n",
        .num_constants = 1, .constants = &spec, .constant_data = &value,
        .num_vertex_attribs = 1, .vertex_attribs = &attrib,
        .vertex_stride = 2 * sizeof(float),
        .vertex_type = PL_PRIM_TRIANGLE_LIST, .target_format = format,
    ), &request) == PL_PASS_PREPARE_ACCEPTED);
    pl_prepared_pass pass = take_prepared(&request);
    int before_translation = atomic_load(&translations);
    int before_pipeline = atomic_load(&pipelines);
    const float vertices[] = {-1, -1, 3, -1, -1, 3};
    // Host vertices exercise Vulkan's recursive VBO conversion checked path.
    struct pl_pass_run_params run = {
        .vertex_data = vertices, .vertex_count = 3, .target = target,
    };
    REQUIRE(pl_prepared_pass_run(pass, &run) == PL_PREPARED_PASS_RUN_OK);
    uint8_t pixels[4 * 4 * 4];
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = target, .ptr = pixels)));
    for (int i = 0; i < sizeof(pixels); i += 4) {
        REQUIRE(pixels[i] == 64);
        REQUIRE(pixels[i+1] == 0 && pixels[i+2] == 0 && pixels[i+3] == 255);
    }
    value = 0.75f;
    run.constant_data = &value;
    REQUIRE(pl_prepared_pass_run(pass, &run) == PL_PREPARED_PASS_RUN_VARIANT_MISMATCH);
    REQUIRE(atomic_load(&translations) == before_translation);
    REQUIRE(atomic_load(&pipelines) == before_pipeline);
    pl_prepared_pass_destroy(&pass);
    pl_tex_destroy(gpu, &target);
}

#ifdef VK_KHR_cooperative_matrix
static void test_cooperative_matrix_pass(pl_vulkan vk)
{
    const struct pl_vulkan_cooperative_matrix *matrix = NULL;
    for (int i = 0; i < vk->num_cooperative_matrices; i++) {
        const struct pl_vulkan_cooperative_matrix *candidate =
            &vk->cooperative_matrices[i];
        if (candidate->a_type == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            candidate->b_type == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            candidate->c_type == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            candidate->result_type == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            candidate->scope == VK_SCOPE_SUBGROUP_KHR &&
            !candidate->saturating_accumulation &&
            (candidate->stages & VK_SHADER_STAGE_COMPUTE_BIT))
        {
            matrix = candidate;
            break;
        }
    }

    if (!matrix)
        return;

    char source[4096];
    int len = snprintf(source, sizeof(source),
        "#version 450\n"
        "#extension GL_EXT_shader_16bit_storage : require\n"
        "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
        "#extension GL_KHR_cooperative_matrix : require\n"
        "#extension GL_KHR_memory_scope_semantics : require\n"
        "#pragma use_vulkan_memory_model\n"
        "layout(local_size_x = %u) in;\n"
        "layout(std430, binding = 0) buffer Output { float result[]; };\n"
        "shared float16_t a_data[%u];\n"
        "shared float16_t b_data[%u];\n"
        "shared float c_data[%u];\n"
        "void main() {\n"
        "  coopmat<float16_t, gl_ScopeSubgroup, %u, %u, gl_MatrixUseA> a;\n"
        "  coopmat<float16_t, gl_ScopeSubgroup, %u, %u, gl_MatrixUseB> b;\n"
        "  coopmat<float, gl_ScopeSubgroup, %u, %u, gl_MatrixUseAccumulator> c =\n"
        "      coopmat<float, gl_ScopeSubgroup, %u, %u, gl_MatrixUseAccumulator>(0.0);\n"
        "  coopMatLoad(a, a_data, 0, %u, gl_CooperativeMatrixLayoutRowMajor);\n"
        "  coopMatLoad(b, b_data, 0, %u, gl_CooperativeMatrixLayoutRowMajor);\n"
        "  c = coopMatMulAdd(a, b, c);\n"
        "  coopMatStore(c, c_data, 0, %u, gl_CooperativeMatrixLayoutRowMajor);\n"
        "  barrier();\n"
        "  for (uint i = gl_LocalInvocationIndex; i < %u; i += %u)\n"
        "    result[i] = c_data[i];\n"
        "}\n",
        vk->gpu->glsl.subgroup_size,
        matrix->m * matrix->k, matrix->k * matrix->n, matrix->m * matrix->n,
        matrix->m, matrix->k, matrix->k, matrix->n,
        matrix->m, matrix->n, matrix->m, matrix->n,
        matrix->k, matrix->n, matrix->n,
        matrix->m * matrix->n, vk->gpu->glsl.subgroup_size);
    REQUIRE(len > 0 && len < sizeof(source));

    struct pl_desc output = {
        .name = "Output",
        .type = PL_DESC_BUF_STORAGE,
        .access = PL_DESC_ACCESS_WRITEONLY,
    };
    pl_pass_preparation request = NULL;
    REQUIRE(pl_pass_prepare_submit(vk->gpu, pl_pass_params(
        .type = PL_PASS_COMPUTE,
        .glsl_shader = source,
        .num_descriptors = 1,
        .descriptors = &output,
    ), &request) == PL_PASS_PREPARE_ACCEPTED);
    pl_prepared_pass pass = take_prepared(&request);
    REQUIRE(pass);
    pl_prepared_pass_destroy(&pass);
}
#endif

static const char source[] =
    "#version 450\n"
    "layout(local_size_x = 1) in;\n"
    "layout(constant_id = 0) const uint value = 11u;\n"
    "layout(std430, binding = 0) buffer Output { uint result; };\n"
    "void main() { result = value; }\n";

static pl_pass_preparation submit_compute_ex(pl_gpu gpu, uint32_t value,
                                              bool use_defaults)
{
    // Every caller-owned input goes out of scope immediately after submission.
    char *owned_source = strdup(source);
    struct pl_desc desc = {
        .name = "Output", .type = PL_DESC_BUF_STORAGE,
        .access = PL_DESC_ACCESS_WRITEONLY,
    };
    struct pl_constant spec = { .type = PL_VAR_UINT, .id = 0 };
    pl_pass_preparation request = NULL;
    REQUIRE(pl_pass_prepare_submit(gpu, pl_pass_params(
        .type = PL_PASS_COMPUTE, .glsl_shader = owned_source,
        .num_descriptors = 1, .descriptors = &desc,
        .num_constants = 1, .constants = &spec,
        .constant_data = use_defaults ? NULL : &value,
    ), &request) == PL_PASS_PREPARE_ACCEPTED);
    memset(owned_source, 'x', strlen(owned_source));
    free(owned_source);
    return request;
}

static pl_pass_preparation submit_compute(pl_gpu gpu, uint32_t value)
{
    return submit_compute_ex(gpu, value, false);
}

static void execute_and_check(pl_gpu gpu, pl_prepared_pass pass, pl_buf buffer,
                              uint32_t expected)
{
    int before_translation = atomic_load(&translations);
    int before_pipeline = atomic_load(&pipelines);
    struct pl_desc_binding binding = { .object = buffer };
    REQUIRE(pl_prepared_pass_run(pass, pl_pass_run_params(
        .desc_bindings = &binding, .compute_groups = {1, 1, 1}
    )) == PL_PREPARED_PASS_RUN_OK);
    uint32_t result = 0;
    REQUIRE(pl_buf_read(gpu, buffer, 0, &result, sizeof(result)));
    REQUIRE_CMP(result, ==, expected, PRIu32);
    REQUIRE(atomic_load(&translations) == before_translation);
    REQUIRE(atomic_load(&pipelines) == before_pipeline);

    // A new specialization is rejected before submission or driver creation.
    uint32_t changed = expected + 1;
    REQUIRE(pl_prepared_pass_run(pass, pl_pass_run_params(
        .desc_bindings = &binding, .compute_groups = {1, 1, 1},
        .constant_data = &changed,
    )) == PL_PREPARED_PASS_RUN_VARIANT_MISMATCH);
    REQUIRE(atomic_load(&translations) == before_translation);
    REQUIRE(atomic_load(&pipelines) == before_pipeline);
    REQUIRE(pl_buf_read(gpu, buffer, 0, &result, sizeof(result)));
    REQUIRE_CMP(result, ==, expected, PRIu32);
}

int main(void)
{
    caller_thread = true;
    latch_init(&translation);
    latch_init(&pipeline);
    pl_log log = pl_test_logger();
    pl_vulkan vk = pl_vulkan_create(log, pl_vulkan_params(
        .allow_software = true,
        .instance_params = pl_vk_inst_params(.debug = true),
    ));
    if (!vk) {
        pl_log_destroy(&log);
        return SKIP;
    }
    pl_gpu gpu = vk->gpu;
    REQUIRE(pl_pass_prepare_supported(gpu));
    struct pl_vk *priv = PL_PRIV(gpu);
    original_compiler = priv->spirv->impl;
    compiler = *original_compiler;
    compiler.compile = observed_compile;
    ((struct pl_spirv_t *) priv->spirv)->impl = &compiler;
    original_compute = priv->vk->CreateComputePipelines;
    priv->vk->CreateComputePipelines = observed_pipeline;
    original_graphics = priv->vk->CreateGraphicsPipelines;
    priv->vk->CreateGraphicsPipelines = observed_graphics;

#ifdef VK_KHR_cooperative_matrix
    test_cooperative_matrix_pass(vk);
#endif

    pl_buf buffer = pl_buf_create(gpu, pl_buf_params(
        .size = sizeof(uint32_t), .storable = true, .host_readable = true,
    ));
    REQUIRE(buffer);
    pl_pass_preparation request = submit_compute(gpu, 23);
    pl_prepared_pass active = take_prepared(&request);
    REQUIRE(atomic_load(&translations) > 0);
    REQUIRE(atomic_load(&pipelines) > 0);
    execute_and_check(gpu, active, buffer, 23);

    // No initial specialization bytes means the GLSL default is frozen. Its
    // first execution must use 11 without compiling, and even an explicitly
    // supplied value equal to that default is a different preparation request.
    request = submit_compute_ex(gpu, 0, true);
    pl_prepared_pass defaults = take_prepared(&request);
    REQUIRE(!pl_prepared_pass_params(defaults)->constant_data);
    execute_and_check(gpu, defaults, buffer, 11);
    int before_translation = atomic_load(&translations);
    int before_pipeline = atomic_load(&pipelines);
    uint32_t explicit_default = 11;
    struct pl_desc_binding binding = { .object = buffer };
    REQUIRE(pl_prepared_pass_run(defaults, pl_pass_run_params(
        .desc_bindings = &binding, .compute_groups = {1, 1, 1},
        .constant_data = &explicit_default,
    )) == PL_PREPARED_PASS_RUN_VARIANT_MISMATCH);
    REQUIRE(atomic_load(&translations) == before_translation);
    REQUIRE(atomic_load(&pipelines) == before_pipeline);
    pl_prepared_pass_destroy(&defaults);

    struct prepare_latch *phases[] = { &translation, &pipeline };
    for (int i = 0; i < PL_ARRAY_SIZE(phases); i++) {
        latch_arm(phases[i]);
        request = submit_compute(gpu, 31 + i);
        latch_entered(phases[i]);
        REQUIRE(pl_pass_prepare_poll(request) == PL_PASS_PREPARE_PENDING);
        // Actual GPU execution must complete while the replacement's actual
        // compiler/driver entry point is held, independently for both phases.
        execute_and_check(gpu, active, buffer, 23);
        latch_release(phases[i]);
        pl_prepared_pass candidate = take_prepared(&request);
        execute_and_check(gpu, candidate, buffer, 31 + i);
        pl_prepared_pass_destroy(&candidate);
    }

    REQUIRE(pl_pass_prepare_submit(gpu, pl_pass_params(
        .type = PL_PASS_COMPUTE, .glsl_shader = "not GLSL",
    ), &request) == PL_PASS_PREPARE_ACCEPTED);
    REQUIRE(await_prepared(request) == PL_PASS_PREPARE_FAILED);
    REQUIRE(pl_pass_prepare_failure_phase(request) == PL_PASS_PREPARE_PHASE_TRANSLATION);
    REQUIRE(pl_pass_prepare_error(request) && *pl_pass_prepare_error(request));
    pl_pass_prepare_release(&request);
    fail_pipeline = true;
    request = submit_compute(gpu, 47);
    REQUIRE(await_prepared(request) == PL_PASS_PREPARE_FAILED);
    REQUIRE(pl_pass_prepare_failure_phase(request) == PL_PASS_PREPARE_PHASE_PIPELINE);
    REQUIRE(pl_pass_prepare_error(request) && *pl_pass_prepare_error(request));
    pl_pass_prepare_release(&request);
    fail_pipeline = false;
    execute_and_check(gpu, active, buffer, 23);

    test_raster(gpu);
    pl_prepared_pass_destroy(&active);
    pl_pass_prepare_uninit(gpu);
    priv->vk->CreateComputePipelines = original_compute;
    priv->vk->CreateGraphicsPipelines = original_graphics;
    ((struct pl_spirv_t *) priv->spirv)->impl = original_compiler;
    pl_buf_destroy(gpu, &buffer);
    pl_vulkan_destroy(&vk);
    pl_log_destroy(&log);
    latch_destroy(&translation);
    latch_destroy(&pipeline);
}
