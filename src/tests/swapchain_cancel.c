// Compile the production Vulkan swapchain operations with mocked GPU work and
// native calls. Renaming public backend entry points avoids duplicate exports.
#define pl_vulkan_create_swapchain test_create_swapchain
#define pl_vulkan_swapchain_set_present_trace test_set_present_trace
#define pl_vulkan_swapchain_request_exclusive test_request_exclusive
#define pl_vulkan_swapchain_set_exclusive_target test_set_exclusive_target
#define pl_vulkan_swapchain_exclusive_status test_exclusive_status
#define pl_vulkan_swapchain_suboptimal test_suboptimal
#define pl_vulkan_hold_ex test_hold
#define pl_gpu_finish test_finish
#include "../vulkan/swapchain.c"
#include "utils.h"

static bool fail_create, fail_hold, fail_gpu;
static VkResult release_result;
static int holds, finishes, releases, destroys;
static bool finished;
static struct vk_ctx context;

bool test_hold(pl_gpu gpu, const struct pl_vulkan_hold_params *params)
{
    holds++;
    REQUIRE(params->semaphore.sem == (VkSemaphore) 42);
    REQUIRE(params->layout == VK_IMAGE_LAYOUT_GENERAL);
    return !fail_hold;
}
void test_finish(pl_gpu gpu)
{
    finishes++;
    finished = true;
    context.failed = fail_gpu;
}
static VKAPI_ATTR VkResult VKAPI_CALL create_sem(VkDevice dev,
    const VkSemaphoreCreateInfo *info, const VkAllocationCallbacks *alloc, VkSemaphore *out)
{
    if (fail_create) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *out = (VkSemaphore) 42;
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL destroy_sem(VkDevice dev, VkSemaphore sem,
    const VkAllocationCallbacks *alloc)
{
    REQUIRE(finished);
    destroys++;
}
static VKAPI_ATTR VkResult VKAPI_CALL release_images(VkDevice dev,
    const VkReleaseSwapchainImagesInfoKHR *info)
{
    REQUIRE(finished);
    REQUIRE(info->imageIndexCount == 1 && info->pImageIndices[0] == 0);
    releases++;
    return release_result;
}

static void run_case(bool ext, int failure)
{
    holds = finishes = releases = destroys = 0;
    finished = false;
    fail_create = failure == 1;
    fail_hold = failure == 2;
    fail_gpu = failure == 3;
    release_result = failure == 4 ? VK_ERROR_SURFACE_LOST_KHR : VK_SUCCESS;
    context = (struct vk_ctx) {
        .CreateSemaphore = create_sem,
        .DestroySemaphore = destroy_sem,
        .ReleaseSwapchainImagesKHR = ext ? NULL : release_images,
        .ReleaseSwapchainImagesEXT = ext ? release_images : NULL,
    };
    struct pl_swapchain_t *sw = pl_zalloc_obj(NULL, sw, struct priv);
    struct priv *p = PL_PRIV(sw);
    p->impl = vulkan_swapchain;
    p->vk = &context;
    p->has_swapchain_maintenance1 = true;
    pl_mutex_init(&p->lock);
    atomic_init(&p->frame_state, FRAME_IDLE);
    struct vk_swapchain current = {.last_imgidx = 0};
    pl_tex images[] = {NULL};
    current.images.elem = images;
    current.images.num = 1;
    p->current = &current;
    REQUIRE(pl_swapchain_can_cancel_frame(sw));
    REQUIRE(pl_swapchain_cancel_frame(sw) == PL_SWAPCHAIN_CANCEL_NO_FRAME);
    pl_mutex_lock(&p->lock);
    atomic_store(&p->frame_state, FRAME_ACQUIRED);
    const enum pl_swapchain_cancel_result result = pl_swapchain_cancel_frame(sw);
    REQUIRE(result == (failure ? PL_SWAPCHAIN_CANCEL_FAILED : PL_SWAPCHAIN_CANCELLED));
    REQUIRE(atomic_load(&p->frame_state) == (failure ? FRAME_CANCEL_FAILED : FRAME_IDLE));
    // This blocks and fails the test timeout if cancellation retained the lock.
    pl_mutex_lock(&p->lock);
    pl_mutex_unlock(&p->lock);
    REQUIRE(holds == !fail_create && finishes == !fail_create && destroys == !fail_create);
    REQUIRE(releases == (!fail_create && !fail_hold && !fail_gpu));
    REQUIRE(current.last_imgidx == (failure ? 0 : -1));
    REQUIRE(pl_swapchain_cancel_frame(sw) ==
        (failure ? PL_SWAPCHAIN_CANCEL_FAILED : PL_SWAPCHAIN_CANCEL_NO_FRAME));
    if (failure) {
        struct pl_swapchain_frame frame;
        REQUIRE(!pl_swapchain_start_frame(sw, &frame));
    }
    pl_mutex_destroy(&p->lock);
    pl_free(sw);
}

int main(void)
{
    for (int ext = 0; ext < 2; ext++)
        for (int failure = 0; failure <= 4; failure++)
            run_case(ext, failure);

    struct pl_swapchain_t *sw = pl_zalloc_obj(NULL, sw, struct priv);
    struct priv *p = PL_PRIV(sw);
    p->vk = &context;
    REQUIRE(!pl_swapchain_can_cancel_frame(sw));
    REQUIRE(pl_swapchain_cancel_frame(sw) == PL_SWAPCHAIN_CANCEL_UNSUPPORTED);
    p->impl = vulkan_swapchain;
    REQUIRE(!pl_swapchain_can_cancel_frame(sw));
    REQUIRE(pl_swapchain_cancel_frame(sw) == PL_SWAPCHAIN_CANCEL_UNSUPPORTED);
    p->has_swapchain_maintenance1 = true;
    context.ReleaseSwapchainImagesKHR = NULL;
    context.ReleaseSwapchainImagesEXT = NULL;
    REQUIRE(!pl_swapchain_can_cancel_frame(sw));
    REQUIRE(pl_swapchain_cancel_frame(sw) == PL_SWAPCHAIN_CANCEL_UNSUPPORTED);
    pl_free(sw);
}
