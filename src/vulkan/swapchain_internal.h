#pragma once
#include "common.h"

// Retirement must synchronize with all users of each graphics queue.
void vk_swapchain_wait_idle(struct vk_ctx *vk);

// Refresh dimensions before creating a Win32 swapchain. On failure info stays
// unchanged, and the caller must not retire its existing swapchain.
bool vk_swapchain_refresh_extent(struct vk_ctx *vk, VkSurfaceKHR surface,
                                 VkSwapchainCreateInfoKHR *info);
