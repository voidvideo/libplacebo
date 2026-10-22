// Exercise cancellation on a real visible X11 swapchain. The deterministic
// companion test covers failures without relying on driver fault injection.
#define VK_USE_PLATFORM_XLIB_KHR
#include <X11/Xlib.h>
#include <libplacebo/vulkan.h>
#include "utils.h"

int main(void)
{
    Display *display = XOpenDisplay(NULL);
    if (!display) return SKIP;
    Window window = XCreateSimpleWindow(display, DefaultRootWindow(display),
                                        0, 0, 64, 64, 0, 0, 0);
    XStoreName(display, window, "libplacebo cancellation test");
    XMapWindow(display, window);
    XSync(display, False);
    pl_log log = pl_test_logger();
    pl_vk_inst inst = pl_vk_inst_create(log, pl_vk_inst_params(
        .get_proc_addr = vkGetInstanceProcAddr,
        .extensions = (const char *[]) {
            VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
        },
        .num_extensions = 2,
    ));
    REQUIRE(inst);
    PFN_vkCreateXlibSurfaceKHR create_surface = (PFN_vkCreateXlibSurfaceKHR)
        inst->get_proc_addr(inst->instance, "vkCreateXlibSurfaceKHR");
    PFN_vkDestroySurfaceKHR destroy_surface = (PFN_vkDestroySurfaceKHR)
        inst->get_proc_addr(inst->instance, "vkDestroySurfaceKHR");
    REQUIRE(create_surface && destroy_surface);
    VkSurfaceKHR surface;
    VkXlibSurfaceCreateInfoKHR info = {
        .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
        .dpy = display, .window = window,
    };
    REQUIRE(create_surface(inst->instance, &info, NULL, &surface) == VK_SUCCESS);
    pl_vulkan vk = pl_vulkan_create(log, pl_vulkan_params(
        .instance = inst->instance, .get_proc_addr = inst->get_proc_addr,
        .surface = surface,
    ));
    REQUIRE(vk);
    pl_swapchain sw = pl_vulkan_create_swapchain(vk, pl_vulkan_swapchain_params(.surface = surface));
    REQUIRE(sw);
    int width = 64, height = 64;
    REQUIRE(pl_swapchain_resize(sw, &width, &height));
    bool supported = pl_swapchain_can_cancel_frame(sw);
    if (supported) {
        REQUIRE(pl_swapchain_cancel_frame(sw) == PL_SWAPCHAIN_CANCEL_NO_FRAME);
        for (int i = 0; i < 24; i++) {
            struct pl_swapchain_frame frame;
            REQUIRE(pl_swapchain_start_frame(sw, &frame));
            // Cover cancellation both before and after GPU work was queued.
            if (i % 2) pl_tex_clear(vk->gpu, frame.fbo, (float[4]){0});
            REQUIRE(pl_swapchain_cancel_frame(sw) == PL_SWAPCHAIN_CANCELLED);
            REQUIRE(pl_swapchain_cancel_frame(sw) == PL_SWAPCHAIN_CANCEL_NO_FRAME);
            struct pl_swapchain_present_feedback feedback;
            REQUIRE(pl_swapchain_poll_present_feedback(sw, &feedback, 1) == 0);
            REQUIRE(pl_swapchain_resize(sw, &width, &height));
        }
        // Cancellation must not poison subsequent normal presentation.
        for (int i = 0; i < 3; i++) {
            struct pl_swapchain_frame frame;
            REQUIRE(pl_swapchain_start_frame(sw, &frame));
            pl_tex_clear(vk->gpu, frame.fbo, (float[4]){0.25f, 0.25f, 0.25f, 1});
            REQUIRE(pl_swapchain_submit_frame(sw));
            pl_swapchain_swap_buffers(sw);
        }
        struct pl_swapchain_frame frame;
        REQUIRE(pl_swapchain_start_frame(sw, &frame));
        REQUIRE(pl_swapchain_cancel_frame(sw) == PL_SWAPCHAIN_CANCELLED);
        REQUIRE(!pl_gpu_is_failed(vk->gpu));
        puts("PASS: native acquire/cancel, queued GPU work, repeated cancellation, resize, present, destroy");
    }
    pl_swapchain_destroy(&sw);
    pl_vulkan_destroy(&vk);
    destroy_surface(inst->instance, surface, NULL);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
    pl_vk_inst_destroy(&inst);
    pl_log_destroy(&log);
    return supported ? 0 : SKIP;
}
