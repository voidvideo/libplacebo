#include <vulkan/vulkan.h>

#include "gpu_tests.h"
#include "vulkan/command.h"
#include "vulkan/gpu.h"

#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

// Exercise producer-complete imports without requiring a capture device. The
// producer uses the same GPU allocation but explicitly hands ownership away;
// this proves buffer data and ownership transitions, not V4L2 driver interop.
static void vulkan_buffer_import_tests(pl_vulkan vk,
                                      enum pl_handle_type handle_type,
                                      uint32_t external_qf)
{
    pl_gpu gpu = vk->gpu;
    if (handle_type != PL_HANDLE_DMA_BUF) {
        printf("skipping buffer import 0x%x: allocator has no implementation\n",
               handle_type);
        return;
    }
    if (!(gpu->export_caps.buf & handle_type) ||
        !(gpu->import_caps.buf & handle_type)) {
        printf("skipping buffer import 0x%x: device capability unavailable\n",
               handle_type);
        return;
    }

    if (external_qf == VK_QUEUE_FAMILY_FOREIGN_EXT) {
        bool enabled = false;
        for (int i = 0; i < vk->num_extensions; i++)
            enabled |= strcmp(vk->extensions[i], VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME) == 0;
        if (!enabled) {
            printf("skipping FOREIGN ownership: extension not enabled\n");
            return;
        }
    }

    printf("testing externally owned buffer 0x%x, owner 0x%x\n",
           handle_type, external_qf);
    uint32_t expected[256], actual[256];
    pl_buf source = pl_buf_create(gpu, pl_buf_params(
        .size = sizeof(expected),
        .storable = true,
        .host_writable = true,
        .export_handle = handle_type,
    ));
    REQUIRE(source);
    struct pl_buf_vk *source_vk = PL_PRIV(source);
    REQUIRE_CMP(source_vk->external_qf, ==, VK_QUEUE_FAMILY_EXTERNAL, "u");
    source_vk->external_qf = external_qf;

    struct pl_buf_params params = *pl_buf_params(
        .size = sizeof(expected),
        .storable = true,
        .import_handle = handle_type,
        .shared_mem = source->shared_mem,
    );
    pl_buf imported = pl_vulkan_buf_import(gpu, &params, external_qf);
    REQUIRE(imported);
    struct pl_buf_vk *imported_vk = PL_PRIV(imported);
    REQUIRE(imported_vk->exported); // first use must acquire, too
    REQUIRE_CMP(imported_vk->external_qf, ==, external_qf, "u");

    pl_buf readback = pl_buf_create(gpu, pl_buf_params(
        .size = sizeof(expected),
        .host_readable = true,
    ));
    REQUIRE(readback);
    for (int cycle = 0; cycle < 4; cycle++) {
        for (int i = 0; i < PL_ARRAY_SIZE(expected); i++)
            expected[i] = 0x12345678u ^ (cycle * 333 + i * 113);

        pl_buf_write(gpu, source, 0, expected, sizeof(expected));
        REQUIRE(pl_buf_export(gpu, source));
        pl_gpu_finish(gpu); // synchronous producer-complete test fixture only
        REQUIRE(!pl_buf_poll(gpu, source, 0));

        // A partial first read still acquires ownership for the full buffer.
        pl_buf_copy(gpu, readback, 16, imported, 16, 100);
        REQUIRE(!imported_vk->exported);
        pl_buf_copy(gpu, readback, 0, imported, 0, sizeof(expected));
        REQUIRE(pl_buf_export(gpu, imported));
        REQUIRE(pl_buf_export(gpu, imported)); // harmless repeated export
        REQUIRE(imported_vk->exported);
        pl_gpu_finish(gpu);
        REQUIRE(!pl_buf_poll(gpu, imported, 0));
        REQUIRE(pl_buf_read(gpu, readback, 0, actual, sizeof(actual)));
        REQUIRE(memcmp(actual, expected, sizeof(actual)) == 0);
    }

    pl_buf_destroy(gpu, &readback);
    pl_buf_destroy(gpu, &imported);
    pl_buf_destroy(gpu, &source);
    REQUIRE(!pl_gpu_is_failed(gpu));
}

static void vulkan_buffer_import_rejections(pl_vulkan vk)
{
    pl_gpu gpu = vk->gpu;
    struct pl_buf_params params = *pl_buf_params(.size = 64);
    REQUIRE(!pl_vulkan_buf_import(gpu, &params, VK_QUEUE_FAMILY_EXTERNAL));
    params.import_handle = PL_HANDLE_HOST_PTR;
    REQUIRE(!pl_vulkan_buf_import(gpu, &params, VK_QUEUE_FAMILY_EXTERNAL));
    params.import_handle = PL_HANDLE_FD;
    REQUIRE(!pl_vulkan_buf_import(gpu, &params, VK_QUEUE_FAMILY_EXTERNAL));
    params.import_handle = PL_HANDLE_DMA_BUF;
    REQUIRE(!pl_vulkan_buf_import(gpu, &params, VK_QUEUE_FAMILY_IGNORED));
    params.initial_data = &params;
    REQUIRE(!pl_vulkan_buf_import(gpu, &params, VK_QUEUE_FAMILY_EXTERNAL));
    params.initial_data = NULL;
    params.export_handle = PL_HANDLE_FD;
    REQUIRE(!pl_vulkan_buf_import(gpu, &params, VK_QUEUE_FAMILY_EXTERNAL));
    params.export_handle = 0;
    params.import_handle = PL_HANDLE_FD | PL_HANDLE_DMA_BUF;
    REQUIRE(!pl_vulkan_buf_import(gpu, &params, VK_QUEUE_FAMILY_EXTERNAL));
    REQUIRE(!pl_vulkan_buf_import(NULL, &params, VK_QUEUE_FAMILY_EXTERNAL));
    REQUIRE(!pl_vulkan_buf_import(gpu, NULL, VK_QUEUE_FAMILY_EXTERNAL));
}

static void vulkan_interop_tests(pl_vulkan pl_vk,
                                 enum pl_handle_type handle_type,
                                 bool buffer_only)
{
    pl_gpu gpu = pl_vk->gpu;
    printf("testing vulkan interop for handle type 0x%x\n", handle_type);
    vulkan_buffer_import_tests(pl_vk, handle_type, VK_QUEUE_FAMILY_EXTERNAL);
    vulkan_buffer_import_tests(pl_vk, handle_type, VK_QUEUE_FAMILY_FOREIGN_EXT);
    if (buffer_only)
        return;

    if (gpu->export_caps.buf & handle_type) {
        pl_buf buf = pl_buf_create(gpu, pl_buf_params(
            .size = 1024,
            .export_handle = handle_type,
        ));

        REQUIRE(buf);
        REQUIRE_HANDLE(buf->shared_mem, handle_type);
        REQUIRE_CMP(buf->shared_mem.size, >=, buf->params.size, "zu");
        REQUIRE(pl_buf_export(gpu, buf));
        pl_buf_destroy(gpu, &buf);
    }

    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 0, 0, PL_FMT_CAP_BLITTABLE);
    if (!fmt)
        return;

    if (handle_type == PL_HANDLE_DMA_BUF && !fmt->num_modifiers)
        return;

    // Test interop API
    if (gpu->export_caps.tex & handle_type) {
        VkSemaphore sem = pl_vulkan_sem_create(gpu, pl_vulkan_sem_params(
            .type           = VK_SEMAPHORE_TYPE_TIMELINE,
            .initial_value  = 0,
        ));

        pl_tex tex = pl_tex_create(gpu, pl_tex_params(
            .w              = 32,
            .h              = 32,
            .format         = fmt,
            .blit_dst       = true,
            .export_handle  = handle_type,
        ));

        REQUIRE(sem);
        REQUIRE(tex);

        REQUIRE(pl_vulkan_hold_ex(gpu, pl_vulkan_hold_params(
            .tex            = tex,
            .layout         = VK_IMAGE_LAYOUT_GENERAL,
            .qf             = VK_QUEUE_FAMILY_EXTERNAL,
            .semaphore      = { sem, 1 },
        )));

        pl_vulkan_release_ex(gpu, pl_vulkan_release_params(
            .tex            = tex,
            .layout         = VK_IMAGE_LAYOUT_GENERAL,
            .qf             = VK_QUEUE_FAMILY_EXTERNAL,
            .semaphore      = { sem, 1 },
        ));

        pl_tex_clear(gpu, tex, (float[4]){0});
        pl_gpu_finish(gpu);
        REQUIRE(!pl_tex_poll(gpu, tex, 0));

        pl_vulkan_sem_destroy(gpu, &sem);
        pl_tex_destroy(gpu, &tex);
    }
}

static void vulkan_swapchain_tests(pl_vulkan vk, VkSurfaceKHR surf)
{
    if (!surf)
        return;

    printf("testing vulkan swapchain\n");
    pl_gpu gpu = vk->gpu;
    pl_swapchain sw;
    sw = pl_vulkan_create_swapchain(vk, pl_vulkan_swapchain_params(
        .surface = surf,
    ));
    REQUIRE(sw);

    // Attempt actually initializing the swapchain
    int w = 640, h = 480;
    REQUIRE(pl_swapchain_resize(sw, &w, &h));

    for (int i = 0; i < 10; i++) {
        struct pl_swapchain_frame frame;
        REQUIRE(pl_swapchain_start_frame(sw, &frame));
        if (frame.fbo->params.blit_dst)
            pl_tex_clear(gpu, frame.fbo, (float[4]){0});

        // TODO: test this with an actual pl_renderer instance
        struct pl_frame target;
        pl_frame_from_swapchain(&target, &frame);

        REQUIRE(pl_swapchain_submit_frame(sw));
        pl_swapchain_swap_buffers(sw);

        // Try resizing the swapchain in the middle of rendering
        if (i == 5) {
            w = 320;
            h = 240;
            REQUIRE(pl_swapchain_resize(sw, &w, &h));
        }
    }

    pl_swapchain_destroy(&sw);
}

int main(int argc, char **argv)
{
    const bool buffer_only = argc == 2 && !strcmp(argv[1], "--buffer-interop");
    pl_log log = pl_test_logger();
    pl_vk_inst inst = pl_vk_inst_create(log, pl_vk_inst_params(
        .debug = true,
        .debug_extra = true,
        .get_proc_addr = vkGetInstanceProcAddr,
        .opt_extensions = (const char *[]){
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
        },
        .num_opt_extensions = 2,
    ));

    if (!inst)
        return SKIP;

    PL_VK_LOAD_FUN(inst->instance, EnumeratePhysicalDevices, inst->get_proc_addr);
    PL_VK_LOAD_FUN(inst->instance, GetPhysicalDeviceProperties, inst->get_proc_addr);

    uint32_t num = 0;
    EnumeratePhysicalDevices(inst->instance, &num, NULL);
    if (!num)
        return SKIP;

    VkPhysicalDevice *devices = calloc(num, sizeof(*devices));
    if (!devices)
        return 1;
    EnumeratePhysicalDevices(inst->instance, &num, devices);

    VkSurfaceKHR surf = VK_NULL_HANDLE;

    PL_VK_LOAD_FUN(inst->instance, CreateHeadlessSurfaceEXT, inst->get_proc_addr);
    if (CreateHeadlessSurfaceEXT && !buffer_only) {
        VkHeadlessSurfaceCreateInfoEXT info = {
            .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
        };

        VkResult res = CreateHeadlessSurfaceEXT(inst->instance, &info, NULL, &surf);
        REQUIRE_CMP(res, ==, VK_SUCCESS, "u");
    }

    // Make sure choosing any device works
    VkPhysicalDevice dev;
    dev = pl_vulkan_choose_device(log, pl_vulkan_device_params(
        .instance = inst->instance,
        .get_proc_addr = inst->get_proc_addr,
        .allow_software = true,
        .surface = surf,
    ));
    if (!dev)
        return SKIP;

    // Test all attached devices
    for (int i = 0; i < num; i++) {
        VkPhysicalDeviceProperties props = {0};
        GetPhysicalDeviceProperties(devices[i], &props);
#ifndef CI_ALLOW_SW
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !buffer_only) {
            printf("Skipping device %d: %s\n", i, props.deviceName);
            continue;
        }
#endif
        printf("Testing device %d: %s\n", i, props.deviceName);

        // Make sure we can choose this device by name
        dev = pl_vulkan_choose_device(log, pl_vulkan_device_params(
            .instance = inst->instance,
            .get_proc_addr = inst->get_proc_addr,
            .device_name = props.deviceName,
        ));
        REQUIRE_CMP(dev, ==, devices[i], "p");

        struct pl_vulkan_params params = *pl_vulkan_params(
            .instance = inst->instance,
            .get_proc_addr = inst->get_proc_addr,
            .device = devices[i],
            .queue_count = 8, // test inter-queue stuff
            .surface = surf,
        );

        pl_vulkan vk = pl_vulkan_create(log, &params);
        if (!vk)
            continue;

        vulkan_buffer_import_rejections(vk);
        if (!buffer_only) {
            gpu_shader_tests(vk->gpu);
            vulkan_swapchain_tests(vk, surf);
        }

        // Print heap statistics
        pl_vk_print_heap(vk->gpu, PL_LOG_DEBUG);

        // Test importing this context via the vulkan interop API
        pl_vulkan vk2 = pl_vulkan_import(log, pl_vulkan_import_params(
            .instance = vk->instance,
            .get_proc_addr = inst->get_proc_addr,
            .phys_device = vk->phys_device,
            .device = vk->device,

            .extensions = vk->extensions,
            .num_extensions = vk->num_extensions,
            .features = vk->features,
            .queue_graphics = vk->queue_graphics,
            .queue_compute = vk->queue_compute,
            .queue_transfer = vk->queue_transfer,
        ));
        REQUIRE(vk2);
        pl_vulkan_destroy(&vk2);

        // Run these tests last because they disable some validation layers
#ifdef PL_HAVE_UNIX
        vulkan_interop_tests(vk, PL_HANDLE_FD, buffer_only);
        vulkan_interop_tests(vk, PL_HANDLE_DMA_BUF, buffer_only);
#endif
#ifdef PL_HAVE_WIN32
        vulkan_interop_tests(vk, PL_HANDLE_WIN32, buffer_only);
        vulkan_interop_tests(vk, PL_HANDLE_WIN32_KMT, buffer_only);
#endif
        if (!buffer_only)
            gpu_interop_tests(vk->gpu);
        pl_vulkan_destroy(&vk);

        // Re-run the same export/import tests with async queues disabled
        params.async_compute = false;
        params.async_transfer = false;
        vk = pl_vulkan_create(log, &params);
        REQUIRE(vk); // it succeeded the first time

#ifdef PL_HAVE_UNIX
        vulkan_interop_tests(vk, PL_HANDLE_FD, buffer_only);
        vulkan_interop_tests(vk, PL_HANDLE_DMA_BUF, buffer_only);
#endif
#ifdef PL_HAVE_WIN32
        vulkan_interop_tests(vk, PL_HANDLE_WIN32, buffer_only);
        vulkan_interop_tests(vk, PL_HANDLE_WIN32_KMT, buffer_only);
#endif
        if (!buffer_only)
            gpu_interop_tests(vk->gpu);
        pl_vulkan_destroy(&vk);

        // Reduce log spam after first tested device
        pl_log_level_update(log, PL_LOG_INFO);
    }

    if (surf)
        vkDestroySurfaceKHR(inst->instance, surf, NULL);
    pl_vk_inst_destroy(&inst);
    pl_log_destroy(&log);
    free(devices);
}
