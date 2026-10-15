#include "utils.h"
#include "vulkan/command.h"
#include "vulkan/swapchain_internal.h"

static int locked = -1, waited, unlocked;
static VkSurfaceCapabilitiesKHR surface_caps;
static VkResult caps_result = VK_SUCCESS;

static void lock_queue(void *ctx, uint32_t family, uint32_t index)
{
    REQUIRE(ctx == &waited);
    REQUIRE(family == 7 && index < 2);
    REQUIRE(locked == -1);
    locked = index;
}

static void unlock_queue(void *ctx, uint32_t family, uint32_t index)
{
    REQUIRE(ctx == &waited && family == 7);
    REQUIRE(locked == (int) index);
    locked = -1;
    unlocked++;
}

static VKAPI_ATTR VkResult VKAPI_CALL wait_idle(VkQueue queue)
{
    // Reject the precise API misuse reported by the Windows validation layer.
    REQUIRE(locked >= 0);
    REQUIRE(queue == (VkQueue) (uintptr_t) (locked + 1));
    waited++;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL get_caps(VkPhysicalDevice device,
        VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR *out)
{
    REQUIRE(surface == (VkSurfaceKHR) (uintptr_t) 3);
    *out = surface_caps;
    return caps_result;
}

int main(void)
{
    VkQueue queues[] = {(VkQueue) (uintptr_t) 1, (VkQueue) (uintptr_t) 2};
    struct vk_cmdpool pool = {.qf = 7, .queues = queues, .num_queues = 2};
    struct vk_ctx vk = {
        .pool_graphics = &pool, .queue_ctx = &waited,
        .lock_queue = lock_queue, .unlock_queue = unlock_queue,
        .QueueWaitIdle = wait_idle,
        .GetPhysicalDeviceSurfaceCapabilitiesKHR = get_caps,
    };
    vk_swapchain_wait_idle(&vk);
    REQUIRE(waited == 2 && unlocked == 2 && locked == -1);

    const VkSurfaceKHR surface = (VkSurfaceKHR) (uintptr_t) 3;
    // Reproduce a captured Windows resize: the earlier sample was 1376x908,
    // while the surface has moved to 1377x919 before creation.
    VkSwapchainCreateInfoKHR info = {.imageExtent = {1376, 908}};
    surface_caps = (VkSurfaceCapabilitiesKHR) {
        .currentExtent = {1377, 919}, .minImageExtent = {1377, 919},
        .maxImageExtent = {1377, 919},
    };
    REQUIRE(vk_swapchain_refresh_extent(&vk, surface, &info));
    REQUIRE(info.imageExtent.width == 1377 && info.imageExtent.height == 919);

    surface_caps.currentExtent = (VkExtent2D) {UINT32_MAX, UINT32_MAX};
    surface_caps.minImageExtent = (VkExtent2D) {64, 64};
    surface_caps.maxImageExtent = (VkExtent2D) {1024, 768};
    info.imageExtent = (VkExtent2D) {1920, 32};
    REQUIRE(vk_swapchain_refresh_extent(&vk, surface, &info));
    REQUIRE(info.imageExtent.width == 1024 && info.imageExtent.height == 64);

    VkExtent2D previous = info.imageExtent;
    surface_caps = (VkSurfaceCapabilitiesKHR) {0};
    REQUIRE(!vk_swapchain_refresh_extent(&vk, surface, &info));
    REQUIRE(info.imageExtent.width == previous.width && info.imageExtent.height == previous.height);
    caps_result = VK_ERROR_SURFACE_LOST_KHR;
    REQUIRE(!vk_swapchain_refresh_extent(&vk, surface, &info));
    REQUIRE(info.imageExtent.width == previous.width && info.imageExtent.height == previous.height);
    return 0;
}
