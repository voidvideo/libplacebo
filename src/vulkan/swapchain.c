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
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "common.h"
#include "command.h"
#include "formats.h"
#include "utils.h"
#include "gpu.h"
#include "present_feedback.h"
#include "swapchain.h"
#include "pl_thread.h"

struct vk_swapchain {
    VkSwapchainKHR swapchain;
    // state of the images:
    PL_ARRAY(VkImage) vkimages;     // VkImages waiting to be wrapped
    PL_ARRAY(pl_tex) images;        // pl_tex wrappers for the VkImages
    PL_ARRAY(VkSemaphore) sems_in;  // pool of semaphores used to acquire images
    PL_ARRAY(VkSemaphore) sems_out; // pool of semaphores used to present images
    PL_ARRAY(VkFence) fences_out;   // pool of fences for presented images
    int idx_sems_in;                // index of next free semaphore to acquire
    int last_imgidx;                // the image index last acquired (for submit)
};

struct priv {
    struct pl_sw_fns impl;

    pl_mutex lock;
    struct vk_ctx *vk;
    VkSurfaceKHR surf;
    PL_ARRAY(VkSurfaceFormatKHR) formats;

    // current swapchain and metadata:
    struct pl_vulkan_swapchain_params params;
    VkSwapchainCreateInfoKHR protoInfo; // partially filled-in prototype
    struct vk_swapchain *current;
    PL_ARRAY(struct vk_swapchain*) retired;
#ifdef PL_HAVE_WIN32
    HMONITOR exclusive_monitor;
    bool exclusive_managed;
    bool exclusive_acquired;
    VkResult exclusive_result;
    atomic_uint_fast64_t exclusive_snapshot;
#endif
    uint32_t queue_families[3];
    int cur_width, cur_height;
    int swapchain_depth;
    pl_rc_t frames_in_flight;       // number of frames currently queued
    bool suboptimal;                // true once VK_SUBOPTIMAL_KHR is returned
    bool needs_recreate;            // swapchain needs to be recreated
    bool has_swapchain_maintenance1;
    bool feedback_device_supported;
    bool feedback_surface_supported;
    enum pl_swapchain_present_clock feedback_clock;
    uint64_t feedback_clock_id;
    struct vk_present_feedback_state feedback;
    struct pl_color_repr color_repr;
    struct pl_color_space color_space;
    struct pl_hdr_metadata hdr_metadata;
};

static const struct pl_sw_fns vulkan_swapchain;

#ifdef PL_HAVE_WIN32
// Publish result and confirmed acquisition together. Status readers must never
// wait on the lock held across start_frame/render/submit or on driver calls.
static void publish_exclusive(struct priv *p)
{
    uint64_t snapshot = ((uint64_t) (uint32_t) p->exclusive_result << 1) |
                        (uint64_t) p->exclusive_acquired;
    atomic_store_explicit(&p->exclusive_snapshot, snapshot, memory_order_release);
}
#endif

static bool map_color_space(VkColorSpaceKHR space, struct pl_color_space *out)
{
    switch (space) {
    case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_709,
            .transfer  = PL_COLOR_TRC_SRGB,
        };
        return true;
    case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_709,
            .transfer  = PL_COLOR_TRC_BT_1886,
        };
        return true;
    case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_DISPLAY_P3,
            // Actually there are some controversy about Display P3's TRC curve,
            // just like sRGB
            .transfer  = PL_COLOR_TRC_SRGB,
        };
        return true;
    case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_DISPLAY_P3,
            .transfer  = PL_COLOR_TRC_LINEAR,
        };
        return true;
    case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
        // This color space is using XYZ color system than RGB
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_DCI_P3,
            .transfer  = PL_COLOR_TRC_ST428,
        };
        return true;
    case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_709,
            .transfer  = PL_COLOR_TRC_SCRGB,
        };
        return true;
    case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
        // TODO
        return false;
    case VK_COLOR_SPACE_BT709_LINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_709,
            .transfer  = PL_COLOR_TRC_LINEAR,
        };
        return true;
    case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_2020,
            .transfer  = PL_COLOR_TRC_LINEAR,
        };
        return true;
    case VK_COLOR_SPACE_HDR10_ST2084_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_2020,
            .transfer  = PL_COLOR_TRC_PQ,
        };
        return true;
    case VK_COLOR_SPACE_DOLBYVISION_EXT:
        // Unlikely to ever be implemented
        return false;
    case VK_COLOR_SPACE_HDR10_HLG_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_2020,
            .transfer  = PL_COLOR_TRC_HLG,
        };
        return true;
    case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_ADOBE,
            .transfer  = PL_COLOR_TRC_LINEAR,
        };
        return true;
    case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT:
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_ADOBE,
            .transfer  = PL_COLOR_TRC_GAMMA22,
        };
        return true;
    case VK_COLOR_SPACE_PASS_THROUGH_EXT:
        // Platform specific output color space, map to unknown
        *out = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_UNKNOWN,
            .transfer  = PL_COLOR_TRC_UNKNOWN,
        };
        return true;

#ifdef VK_AMD_display_native_hdr
    case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD:
        // TODO
        return false;
#endif

    default: return false;
    }
}

static inline int quant_score(int depth, int requested)
{
    pl_assert(depth >= 0 && depth <= 16);
    pl_assert(requested >= 0 && requested <= 16);
    return requested <= depth ? 0 : 17 * (16 - depth) + requested - depth;
}

static bool pick_surf_format(pl_swapchain sw, const struct pl_color_space *hint)
{
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    pl_gpu gpu = sw->gpu;

    int best_score = -1000, best_id;
    bool wide_gamut = pl_color_primaries_is_wide_gamut(hint->primaries);
    bool prefer_hdr = pl_color_transfer_is_hdr(hint->transfer);

    for (int i = 0; i < p->formats.num; i++) {
        // Color space / format whitelist
        struct pl_color_space space;
        if (!map_color_space(p->formats.elem[i].colorSpace, &space))
            continue;

        bool disable10 = !pl_color_transfer_is_hdr(space.transfer) &&
                         p->params.disable_10bit_sdr;

        // Make sure we can wrap this format to a meaningful, valid pl_fmt
        for (int n = 0; n < gpu->num_formats; n++) {
            pl_fmt plfmt = gpu->formats[n];
            const struct vk_format **pvkfmt = PL_PRIV(plfmt);
            if ((*pvkfmt)->tfmt != p->formats.elem[i].format)
                continue;

            enum pl_fmt_caps render_caps = 0;
            render_caps |= PL_FMT_CAP_RENDERABLE;
            render_caps |= PL_FMT_CAP_BLITTABLE;
            if ((plfmt->caps & render_caps) != render_caps)
                continue;

            // format valid, use it if it has a higher score
            int score = 0;
            switch (plfmt->component_depth[0]) {
                case 8:
                    if (pl_color_transfer_is_hdr(space.transfer))
                        score += 10;
                    else if (space.transfer == PL_COLOR_TRC_LINEAR ||
                             space.transfer == PL_COLOR_TRC_SCRGB)
                        continue; // avoid 8-bit linear formats
                    else if (space.transfer == PL_COLOR_TRC_UNKNOWN)
                        score += 10;
                    else if (disable10)
                        score += 30;
                    else
                        score += 20;
                    break;
                case 10:
                    if (pl_color_transfer_is_hdr(space.transfer))
                        score += 30;
                    else if (space.transfer == PL_COLOR_TRC_LINEAR ||
                             space.transfer == PL_COLOR_TRC_SCRGB)
                        continue; // avoid 10-bit linear formats
                    else if (space.transfer == PL_COLOR_TRC_UNKNOWN)
                        score += 20;
                    else if (disable10)
                        score += 20;
                    else
                        score += 30;
                    break;
                case 16:
                    if (pl_color_transfer_is_hdr(space.transfer))
                        score += 20;
                    else if (space.transfer == PL_COLOR_TRC_LINEAR ||
                             space.transfer == PL_COLOR_TRC_SCRGB)
                        score += 30;
                    else if (space.transfer == PL_COLOR_TRC_UNKNOWN)
                        score += 30;
                    else if (disable10)
                        score += 10;
                    else
                        score += 10;
                    break;
                default: // skip any other format
                    continue;
            }
            int alpha_depth = plfmt->num_components < 4 ? 0 : PL_MIN(plfmt->component_depth[3], 16);
            int err = quant_score(plfmt->component_depth[0], PL_MIN(p->params.color_bits, 16)) +
                      quant_score(alpha_depth, PL_MIN(p->params.alpha_bits, 16));
            // Reset score if we don't meet the requested bit depth
            if (err)
                score = -err;
#ifdef __APPLE__
            // On Apple hardware, only these formats allow direct-to-display
            // rendering, so give them a slight score boost to tie-break against
            // other formats
            switch (p->formats.elem[i].format) {
                case VK_FORMAT_B8G8R8A8_UNORM:
                case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
                case VK_FORMAT_R16G16B16A16_SFLOAT:
                    score += 5;
                    break;
                default:
                    break;
            }
#else
            // On other platforms, it doesn't matter in theory but some drivers
            // or hardware may have limited support for BGR formats, so prefer
            // RGB instead.
            if (pl_fmt_is_ordered(plfmt))
                score += 5;
#endif
            if (pl_color_primaries_is_wide_gamut(space.primaries) == wide_gamut)
                score += 1000;
            if (space.primaries == hint->primaries)
                score += 2000;
            if (pl_color_transfer_is_hdr(space.transfer) == prefer_hdr)
                score += 10000;
            if (space.transfer == hint->transfer)
                score += 20000;
            else if (space.transfer == PL_COLOR_TRC_UNKNOWN)
                continue; // allow unknown (PASS_THROUGH_EXT) only if requested

            switch (plfmt->type) {
            case PL_FMT_UNKNOWN: break;
            case PL_FMT_UINT: break;
            case PL_FMT_SINT: break;
            case PL_FMT_UNORM: score += 3; break;
            case PL_FMT_SNORM: score += 2; break;
            case PL_FMT_FLOAT: score += 1; break;
            case PL_FMT_TYPE_COUNT: pl_unreachable();
            };

            if (score > best_score) {
                best_score = score;
                best_id = i;
                break;
            }
        }
    }

    if (best_score == -1000) {
        PL_ERR(vk, "Failed picking any valid, renderable surface format!");
        return false;
    }

    VkSurfaceFormatKHR new_sfmt = p->formats.elem[best_id];
    if (p->protoInfo.imageFormat != new_sfmt.format ||
        p->protoInfo.imageColorSpace != new_sfmt.colorSpace)
    {
        PL_INFO(vk, "Picked surface configuration %d: %s + %s", best_id,
                vk_fmt_name(new_sfmt.format),
                vk_csp_name(new_sfmt.colorSpace));

        p->protoInfo.imageFormat = new_sfmt.format;
        p->protoInfo.imageColorSpace = new_sfmt.colorSpace;
        p->needs_recreate = true;
    }

    return true;
}

static void set_hdr_metadata(struct priv *p, const struct pl_hdr_metadata *metadata)
{
    struct vk_ctx *vk = p->vk;
    if (!vk->SetHdrMetadataEXT)
        return;

    // Whitelist only values that we support signalling metadata for
    struct pl_hdr_metadata fix = {
        .prim     = metadata->prim,
        .min_luma = metadata->min_luma,
        .max_luma = metadata->max_luma,
        .max_cll  = metadata->max_cll,
        .max_fall = metadata->max_fall,
    };

    // Ignore no-op changes
    if (pl_hdr_metadata_equal(&fix, &p->hdr_metadata))
        return;

    // Remember the metadata so we can re-apply it after swapchain recreation
    p->hdr_metadata = fix;

    // Ignore HDR metadata requests for SDR swapchains
    if (!pl_color_transfer_is_hdr(p->color_space.transfer))
        return;

    if (!p->current)
        return;

    vk->SetHdrMetadataEXT(vk->dev, 1, &p->current->swapchain, &(VkHdrMetadataEXT) {
        .sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
        .displayPrimaryRed   = { fix.prim.red.x,   fix.prim.red.y },
        .displayPrimaryGreen = { fix.prim.green.x, fix.prim.green.y },
        .displayPrimaryBlue  = { fix.prim.blue.x,  fix.prim.blue.y },
        .whitePoint = { fix.prim.white.x, fix.prim.white.y },
        .maxLuminance = fix.max_luma,
        .minLuminance = fix.min_luma,
        .maxContentLightLevel = fix.max_cll,
        .maxFrameAverageLightLevel = fix.max_fall,
    });

    // Keep track of applied HDR colorimetry metadata
    p->color_space.hdr = p->hdr_metadata;
}

pl_swapchain pl_vulkan_create_swapchain(pl_vulkan plvk,
                              const struct pl_vulkan_swapchain_params *params)
{
    struct vk_ctx *vk = PL_PRIV(plvk);
    pl_gpu gpu = plvk->gpu;

    if (!vk->CreateSwapchainKHR) {
        PL_ERR(gpu, VK_KHR_SWAPCHAIN_EXTENSION_NAME " not enabled!");
        return NULL;
    }

    struct pl_swapchain_t *sw = pl_zalloc_obj(NULL, sw, struct priv);
    sw->log = vk->log;
    sw->gpu = gpu;

    const VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR *sw_maint_features =
    vk_find_struct(vk->features.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR);
    const VkPhysicalDevicePresentTimingFeaturesEXT *timing_features =
    vk_find_struct(vk->features.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT);
    const VkPhysicalDevicePresentId2FeaturesKHR *present_id_features =
    vk_find_struct(vk->features.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR);

    struct priv *p = PL_PRIV(sw);
    pl_mutex_init(&p->lock);
    p->impl = vulkan_swapchain;
    p->params = *params;
    p->vk = vk;
    p->surf = params->surface;
    p->swapchain_depth = PL_DEF(params->swapchain_depth, 3);
    p->has_swapchain_maintenance1 = sw_maint_features && sw_maint_features->swapchainMaintenance1;
    p->feedback_device_supported = timing_features && timing_features->presentTiming &&
        present_id_features && present_id_features->presentId2 &&
        vk->SetSwapchainPresentTimingQueueSizeEXT &&
        vk->GetSwapchainTimeDomainPropertiesEXT && vk->GetPastPresentationTimingEXT &&
        vk->GetPhysicalDeviceSurfaceCapabilities2KHR;
    pl_assert(p->swapchain_depth > 0);
    atomic_init(&p->frames_in_flight, 0);
#ifdef PL_HAVE_WIN32
    p->exclusive_result = VK_NOT_READY;
    atomic_init(&p->exclusive_snapshot, (uint64_t) VK_NOT_READY << 1);
#endif
    p->protoInfo = (VkSwapchainCreateInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = p->surf,
        .imageArrayLayers = 1, // non-stereoscopic
        .imageSharingMode = vk->pools.num > 1 ? VK_SHARING_MODE_CONCURRENT
                                              : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = vk->pools.num,
        .pQueueFamilyIndices = p->queue_families,
        .minImageCount = p->swapchain_depth + 1, // +1 for the FB
        .presentMode = params->present_mode,
        .clipped = true,
    };

    pl_assert(vk->pools.num <= PL_ARRAY_SIZE(p->queue_families));
    for (int i = 0; i < vk->pools.num; i++)
        p->queue_families[i] = vk->pools.elem[i]->qf;

    // These fields will be updated by `vk_sw_recreate`
    p->color_space = pl_color_space_unknown;
    p->color_repr = (struct pl_color_repr) {
        .sys    = PL_COLOR_SYSTEM_RGB,
        .levels = PL_COLOR_LEVELS_FULL,
        .alpha  = PL_ALPHA_UNKNOWN,
    };

    // Make sure the swapchain present mode is supported
    VkPresentModeKHR *modes = NULL;
    uint32_t num_modes = 0;
    VK(vk->GetPhysicalDeviceSurfacePresentModesKHR(vk->physd, p->surf, &num_modes, NULL));
    modes = pl_calloc_ptr(NULL, num_modes, modes);
    VK(vk->GetPhysicalDeviceSurfacePresentModesKHR(vk->physd, p->surf, &num_modes, modes));

    bool supported = false;
    for (int i = 0; i < num_modes; i++)
        supported |= (modes[i] == p->protoInfo.presentMode);
    pl_free_ptr(&modes);

    if (!supported) {
        PL_WARN(vk, "Requested swap mode unsupported by this device, falling "
                "back to VK_PRESENT_MODE_FIFO_KHR");
        p->protoInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    }

    // Enumerate the supported surface color spaces
    uint32_t num_formats = 0;
    VK(vk->GetPhysicalDeviceSurfaceFormatsKHR(vk->physd, p->surf, &num_formats, NULL));
    PL_ARRAY_RESIZE(sw, p->formats, num_formats);
    VK(vk->GetPhysicalDeviceSurfaceFormatsKHR(vk->physd, p->surf, &num_formats, p->formats.elem));
    p->formats.num = num_formats;

    PL_INFO(gpu, "Available surface configurations:");
    for (int i = 0; i < p->formats.num; i++) {
        PL_INFO(gpu, "    %d: %-40s %s", i,
                vk_fmt_name(p->formats.elem[i].format),
                vk_csp_name(p->formats.elem[i].colorSpace));
    }

    // Ensure there exists at least some valid renderable surface format
    struct pl_color_space hint = pl_color_space_srgb;
    if (!pick_surf_format(sw, &hint))
        goto error;

    return sw;

error:
    pl_free(modes);
    pl_free(sw);
    return NULL;
}

static bool swapchain_destroy(pl_swapchain sw, struct vk_swapchain **ptr,
                                uint64_t timeout)
{
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    struct vk_swapchain *vk_sw = *ptr;

    if (!vk_sw)
        return true;

    if (vk_sw->fences_out.num > 0) {
        VkResult res = vk->WaitForFences(vk->dev, vk_sw->fences_out.num,
                                         vk_sw->fences_out.elem, VK_TRUE, timeout);
        if (res == VK_NOT_READY || res == VK_TIMEOUT)
            return false;
        for (int i = 0; i < vk_sw->fences_out.num; i++)
            vk->DestroyFence(vk->dev, vk_sw->fences_out.elem[i], PL_VK_ALLOC);
    } else {
        // Vulkan without VK_KHR_swapchain_maintenance1 offers no way to know
        // when a queue presentation command is done using these resources,
        // leading to undefined behavior when destroying resources tied to the
        // swapchain. Use an extra `vkQueueWaitIdle` on all of the queues we may
        // have oustanding presentation calls on, to mitigate this risk.
        for (int i = 0; i < vk->pool_graphics->num_queues; i++)
            vk->QueueWaitIdle(vk->pool_graphics->queues[i]);
    }

    for (int i = 0; i < vk_sw->images.num; i++)
        pl_tex_destroy(sw->gpu, &vk_sw->images.elem[i]);
    for (int i = 0; i < vk_sw->sems_in.num; i++)
        vk->DestroySemaphore(vk->dev, vk_sw->sems_in.elem[i], PL_VK_ALLOC);
    for (int i = 0; i < vk_sw->sems_out.num; i++)
        vk->DestroySemaphore(vk->dev, vk_sw->sems_out.elem[i], PL_VK_ALLOC);

    vk->DestroySwapchainKHR(vk->dev, vk_sw->swapchain, PL_VK_ALLOC);
    pl_free_ptr(ptr);
    return true;
}

static void cleanup_retired_swapchains(pl_swapchain sw, uint64_t timeout)
{
    struct priv *p = PL_PRIV(sw);
    for (int i = 0; i < p->retired.num; i++) {
        if (swapchain_destroy(sw, &p->retired.elem[i], timeout)) {
            PL_ARRAY_REMOVE_AT(p->retired, i);
            i--;
        }
    }
}

// Called with the swapchain lock held (or during final destruction).
static void release_exclusive(struct priv *p)
{
#ifdef PL_HAVE_WIN32
    if (p->exclusive_acquired && p->current && p->current->swapchain) {
        VkResult res = p->vk->ReleaseFullScreenExclusiveModeEXT(
            p->vk->dev, p->current->swapchain);
        if (res != VK_SUCCESS)
            PL_WARN(p->vk, "[Exclusive] Release failed: %s", vk_res_str(res));
        p->exclusive_acquired = false;
        p->exclusive_result = res;
        publish_exclusive(p);
    }
#else
    (void) p;
#endif
}

#ifdef PL_HAVE_WIN32
static bool acquire_exclusive(struct priv *p)
{
    if (!p->exclusive_monitor || p->exclusive_acquired)
        return true;
    if (!p->current || !p->current->swapchain)
        return false;

    VkResult previous = p->exclusive_result;
    p->exclusive_result = p->vk->AcquireFullScreenExclusiveModeEXT(
        p->vk->dev, p->current->swapchain);
    p->exclusive_acquired = p->exclusive_result == VK_SUCCESS;
    publish_exclusive(p);
    if (p->exclusive_acquired)
        PL_INFO(p->vk, "[Exclusive] Vulkan fullscreen acquisition confirmed");
    else if (previous != p->exclusive_result)
        PL_ERR(p->vk, "[Exclusive] Acquisition failed: %s",
               vk_res_str(p->exclusive_result));
    return p->exclusive_acquired;
}

static void exclusive_lost(struct priv *p)
{
    p->exclusive_acquired = false;
    p->exclusive_result = VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT;
    publish_exclusive(p);
    PL_WARN(p->vk, "[Exclusive] Vulkan fullscreen mode lost; presentation suspended until reacquired");
}
#endif

static void vk_sw_destroy(pl_swapchain sw)
{
    pl_gpu gpu = sw->gpu;
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;

    pl_gpu_flush(gpu);
    vk_wait_idle(vk);

    release_exclusive(p);
    cleanup_retired_swapchains(sw, UINT64_MAX);
    swapchain_destroy(sw, &p->current, UINT64_MAX);

    vk_present_feedback_uninit(&p->feedback);
    pl_mutex_destroy(&p->lock);
    pl_free((void *) sw);
}

static int vk_sw_latency(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    return p->swapchain_depth;
}

static bool update_swapchain_info(struct priv *p, VkSwapchainCreateInfoKHR *info,
                                  int w, int h)
{
    struct vk_ctx *vk = p->vk;

    // Query the supported capabilities and update this struct as needed
    VkSurfaceCapabilitiesKHR caps = {0};
    p->feedback_surface_supported = false;
    info->flags &= ~(VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT |
                     VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR);
    if (!vk->GetPhysicalDeviceSurfaceCapabilities2KHR) {
        VK(vk->GetPhysicalDeviceSurfaceCapabilitiesKHR(vk->physd, p->surf, &caps));
        goto caps_ready;
    }

    VkSurfacePresentModeKHR present_mode = {
        .sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_KHR,
        .presentMode = p->protoInfo.presentMode,
    };
    VkPhysicalDeviceSurfaceInfo2KHR surface_info = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
        .pNext = &present_mode,
        .surface = p->surf,
    };
#ifdef VK_EXT_full_screen_exclusive
    VkSurfaceFullScreenExclusiveInfoEXT fsinfo = {
        .sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT,
#ifdef PL_HAVE_WIN32
        .fullScreenExclusive = p->exclusive_monitor
            ? VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT
            : p->exclusive_managed ? VK_FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT
                                   : VK_FULL_SCREEN_EXCLUSIVE_ALLOWED_EXT,
#else
        .fullScreenExclusive = VK_FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT,
#endif
    };
    if (vk->AcquireFullScreenExclusiveModeEXT)
        vk_link_struct(&surface_info, &fsinfo);
#ifdef PL_HAVE_WIN32
    VkSurfaceFullScreenExclusiveWin32InfoEXT monitor = {
        .sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT,
        .hmonitor = p->exclusive_monitor,
    };
    if (p->exclusive_monitor)
        vk_link_struct(&surface_info, &monitor);
#endif
#endif

#ifdef PL_HAVE_WIN32
    VkSurfaceCapabilitiesFullScreenExclusiveEXT exclusive_support = {
        .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_FULL_SCREEN_EXCLUSIVE_EXT,
    };
#endif
    VkPresentTimingSurfaceCapabilitiesEXT timing_support = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT,
    };
    VkSurfaceCapabilitiesPresentId2KHR present_id_support = {
        .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR,
    };
    VkSurfaceCapabilities2KHR surface_caps = {
        .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR,
#ifdef PL_HAVE_WIN32
        .pNext = p->exclusive_monitor ? &exclusive_support : NULL,
#endif
    };
    if (p->feedback_device_supported) {
        vk_link_struct(&surface_caps, &timing_support);
        vk_link_struct(&surface_caps, &present_id_support);
    }
    VkResult caps_res = vk->GetPhysicalDeviceSurfaceCapabilities2KHR(
        vk->physd, &surface_info, &surface_caps);
#ifdef PL_HAVE_WIN32
    if (p->exclusive_monitor &&
        (caps_res != VK_SUCCESS || !exclusive_support.fullScreenExclusiveSupported))
    {
        p->exclusive_result = caps_res != VK_SUCCESS
            ? caps_res : VK_ERROR_FEATURE_NOT_PRESENT;
        publish_exclusive(p);
        PL_ERR(vk, "[Exclusive] Target surface/monitor does not support exclusive fullscreen: %s",
               vk_res_str(p->exclusive_result));
        return false;
    }
#endif
    if (caps_res != VK_SUCCESS) {
        PL_ERR(vk, "Failed querying surface capabilities: %s", vk_res_str(caps_res));
        return false;
    }
    caps = surface_caps.surfaceCapabilities;
    p->feedback_surface_supported = p->feedback_device_supported &&
        timing_support.presentTimingSupported && present_id_support.presentId2Supported &&
        (timing_support.presentStageQueries &
         VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT);
    if (p->feedback_surface_supported) {
        info->flags |= VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT |
                       VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR;
    }

caps_ready:
    // Check for hidden/invisible window
    if (!caps.currentExtent.width || !caps.currentExtent.height) {
        PL_DEBUG(vk, "maxImageExtent reported as 0x0, hidden window? skipping");
        return false;
    }

    // Sorted by preference
    static const struct { VkCompositeAlphaFlagsKHR vk_mode;
                          enum pl_alpha_mode pl_mode;
                        } alphaModes[] = {
        {VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, PL_ALPHA_INDEPENDENT},
        {VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,  PL_ALPHA_PREMULTIPLIED},
        {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,          PL_ALPHA_UNKNOWN},
        {VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,         PL_ALPHA_UNKNOWN},
    };

    for (int i = 0; i < PL_ARRAY_SIZE(alphaModes); i++) {
        if (caps.supportedCompositeAlpha & alphaModes[i].vk_mode) {
            info->compositeAlpha = alphaModes[i].vk_mode;
            p->color_repr.alpha = alphaModes[i].pl_mode;
            PL_DEBUG(vk, "Requested alpha compositing mode: %s",
                     vk_alpha_mode(info->compositeAlpha));
            break;
        }
    }

    if (!info->compositeAlpha) {
        PL_ERR(vk, "Failed picking alpha compositing mode (caps: 0x%x)",
               caps.supportedCompositeAlpha);
        goto error;
    }

    // Note: We could probably also allow picking a surface transform that
    // flips the framebuffer and set `pl_swapchain_frame.flipped`, but this
    // doesn't appear to be necessary for any vulkan implementations.
    static const VkSurfaceTransformFlagsKHR rotModes[] = {
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR,
    };

    for (int i = 0; i < PL_ARRAY_SIZE(rotModes); i++) {
        if (caps.supportedTransforms & rotModes[i]) {
            info->preTransform = rotModes[i];
            PL_DEBUG(vk, "Requested surface transform: %s",
                     vk_surface_transform(info->preTransform));
            break;
        }
    }

    if (!info->preTransform) {
        PL_ERR(vk, "Failed picking surface transform mode (caps: 0x%x)",
               caps.supportedTransforms);
        goto error;
    }

    // Image count as required
    PL_DEBUG(vk, "Requested image count: %d (min %d max %d)",
             (int) info->minImageCount, (int) caps.minImageCount,
             (int) caps.maxImageCount);

    info->minImageCount = PL_MAX(info->minImageCount, caps.minImageCount);
    if (caps.maxImageCount)
        info->minImageCount = PL_MIN(info->minImageCount, caps.maxImageCount);

    PL_DEBUG(vk, "Requested image size: %dx%d (min %dx%d < cur %dx%d < max %dx%d)",
             w, h, caps.minImageExtent.width, caps.minImageExtent.height,
             caps.currentExtent.width, caps.currentExtent.height,
             caps.maxImageExtent.width, caps.maxImageExtent.height);

    // Default the requested size based on the reported extent
    if (caps.currentExtent.width != 0xFFFFFFFF)
        w = PL_DEF(w, caps.currentExtent.width);
    if (caps.currentExtent.height != 0xFFFFFFFF)
        h = PL_DEF(h, caps.currentExtent.height);

    // Otherwise, re-use the existing size if available
    w = PL_DEF(w, info->imageExtent.width);
    h = PL_DEF(h, info->imageExtent.height);

    if (!w || !h) {
        PL_ERR(vk, "Failed resizing swapchain: unknown size?");
        goto error;
    }

    // Clamp the extent based on the supported limits
    w = PL_CLAMP(w, caps.minImageExtent.width,  caps.maxImageExtent.width);
    h = PL_CLAMP(h, caps.minImageExtent.height, caps.maxImageExtent.height);
    info->imageExtent = (VkExtent2D) { w, h };

    // We just request whatever makes sense, and let the pl_vk decide what
    // pl_tex_params that translates to. That said, we still need to intersect
    // the swapchain usage flags with the format usage flags
    VkImageUsageFlags req_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageUsageFlags opt_flags = VK_IMAGE_USAGE_STORAGE_BIT;

    info->imageUsage = caps.supportedUsageFlags & (req_flags | opt_flags);
    VkFormatProperties fmtprop = {0};
    vk->GetPhysicalDeviceFormatProperties(vk->physd, info->imageFormat, &fmtprop);

#define CHECK(usage, feature) \
    if (!((fmtprop.optimalTilingFeatures & VK_FORMAT_FEATURE_##feature##_BIT))) \
        info->imageUsage &= ~VK_IMAGE_USAGE_##usage##_BIT

    CHECK(COLOR_ATTACHMENT, COLOR_ATTACHMENT);
    CHECK(TRANSFER_DST, TRANSFER_DST);
    CHECK(STORAGE, STORAGE_IMAGE);

    if ((info->imageUsage & req_flags) != req_flags) {
        PL_ERR(vk, "The swapchain doesn't support rendering and blitting!");
        goto error;
    }

    return true;

error:
    return false;
}

static bool vk_sw_image_init(pl_swapchain sw, int idx)
{
    pl_gpu gpu = sw->gpu;
    struct priv *p = PL_PRIV(sw);
    struct vk_swapchain *current = p->current;

    if (current->images.elem[idx])
        return true;

    pl_assert(current->vkimages.elem[idx] != VK_NULL_HANDLE);

    char *name = pl_asprintf(NULL, "swapchain #%d", idx);
    pl_tex tex = pl_vulkan_wrap(gpu, pl_vulkan_wrap_params(
        .image = current->vkimages.elem[idx],
        .width = p->protoInfo.imageExtent.width,
        .height = p->protoInfo.imageExtent.height,
        .format = p->protoInfo.imageFormat,
        .usage = p->protoInfo.imageUsage,
        .debug_tag = name,
    ));

    if (!tex) {
        pl_free(name);
        goto error;
    }
    pl_steal((void *) tex, name);

    current->images.elem[idx] = tex;
    current->vkimages.elem[idx] = VK_NULL_HANDLE;

    int bits = 0;
    // The channel with the most bits is probably the most authoritative about
    // the actual color information (consider e.g. a2bgr10). Slight downside
    // in that it results in rounding r/b for e.g. rgb565, but we don't pick
    // surfaces with fewer than 8 bits anyway, so let's not care for now.
    pl_fmt fmt = current->images.elem[idx]->params.format;
    for (int i = 0; i < fmt->num_components; i++)
        bits = PL_MAX(bits, fmt->component_depth[i]);
    p->color_repr.bits.sample_depth = bits;
    p->color_repr.bits.color_depth = bits;

    return true;

error:
    PL_ERR(sw, "Failed wrapping swapchain image!");
    return false;
}

static bool vk_sw_recreate(pl_swapchain sw, int w, int h)
{
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    struct vk_swapchain *current = p->current;

    uint32_t num_images = 0;
    char name[32];

    if (!update_swapchain_info(p, &p->protoInfo, w, h))
        return false;

    vk_present_feedback_reset(&p->feedback);

    VkSwapchainCreateInfoKHR sinfo = p->protoInfo;

    VkSwapchainPresentModesCreateInfoKHR pminfo = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR,
        .presentModeCount = 1,
        .pPresentModes = &p->protoInfo.presentMode,
    };
    if (p->has_swapchain_maintenance1)
        vk_link_struct(&sinfo, &pminfo);

#ifdef VK_EXT_full_screen_exclusive
    VkSurfaceFullScreenExclusiveInfoEXT fsinfo = {
        .sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT,
#ifdef PL_HAVE_WIN32
        .fullScreenExclusive = p->exclusive_monitor
            ? VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT
            : p->exclusive_managed ? VK_FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT
                                   : VK_FULL_SCREEN_EXCLUSIVE_ALLOWED_EXT,
#else
        .fullScreenExclusive = VK_FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT,
#endif
    };
    if (vk->AcquireFullScreenExclusiveModeEXT)
        vk_link_struct(&sinfo, &fsinfo);
#ifdef PL_HAVE_WIN32
    VkSurfaceFullScreenExclusiveWin32InfoEXT monitor = {
        .sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT,
        .hmonitor = p->exclusive_monitor,
    };
    if (p->exclusive_monitor)
        vk_link_struct(&sinfo, &monitor);
    else if (vk->AcquireFullScreenExclusiveModeEXT)
        PL_INFO(sw, "[Fullscreen] Vulkan exclusive policy: %s", p->exclusive_managed
                ? "disallowed (windowed or inactive)"
                : "allowed (driver-managed; acquisition not confirmed)");
    else
        PL_WARN(sw, "[Fullscreen] VK_EXT_full_screen_exclusive unavailable; using driver default behavior");
#endif
#endif

    p->suboptimal = false;
    p->needs_recreate = false;
    p->cur_width = sinfo.imageExtent.width;
    p->cur_height = sinfo.imageExtent.height;

    PL_DEBUG(sw, "(Re)creating swapchain of size %dx%d",
             sinfo.imageExtent.width,
             sinfo.imageExtent.height);

    // Calling `vkCreateSwapchainKHR` puts sinfo.oldSwapchain into a retired
    // state whether the call succeeds or not, so we always need to garbage
    // collect it afterwards - asynchronously as it may still be in use
    if (current) {
        PL_ARRAY_APPEND(sw, p->retired, current);
        sinfo.oldSwapchain = current->swapchain;
    } else {
        sinfo.oldSwapchain = VK_NULL_HANDLE;
    }
    p->current = current = pl_zalloc_ptr(NULL, p->current);
    current->last_imgidx = -1;
    VkResult res = vk->CreateSwapchainKHR(vk->dev, &sinfo, PL_VK_ALLOC, &current->swapchain);
#ifdef PL_HAVE_WIN32
    // A successful replacement inherits exclusive access from oldSwapchain.
    // Do not acquire it a second time. A failed replacement retires the old
    // chain, so clear the confirmed state.
    if (p->exclusive_monitor && res != VK_SUCCESS) {
        p->exclusive_acquired = false;
        p->exclusive_result = res;
        publish_exclusive(p);
    } else if (p->exclusive_acquired) {
        p->exclusive_result = VK_SUCCESS;
        publish_exclusive(p);
    }
#endif
    PL_VK_ASSERT(res, "vk->CreateSwapchainKHR(...)");

    if (p->feedback_surface_supported) {
        uint32_t queue_size = PL_MAX(8, p->swapchain_depth * 4);
        res = vk->SetSwapchainPresentTimingQueueSizeEXT(vk->dev, current->swapchain,
                                                        queue_size);
        if (res != VK_SUCCESS) {
            PL_WARN(vk, "Failed setting presentation timing queue size: %s",
                    vk_res_str(res));
            p->feedback_surface_supported = false;
        }
    }

    p->feedback_clock = PL_SWAPCHAIN_PRESENT_CLOCK_UNKNOWN;
    p->feedback_clock_id = 0;
    if (p->feedback_surface_supported) {
        VkSwapchainTimeDomainPropertiesEXT props = {
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT,
        };
        uint64_t counter = 0;
        res = vk->GetSwapchainTimeDomainPropertiesEXT(vk->dev, current->swapchain,
                                                       &props, &counter);
        if (res == VK_SUCCESS && props.timeDomainCount) {
            void *tmp = pl_tmp(NULL);
            VkTimeDomainKHR *domains = pl_calloc_ptr(tmp, props.timeDomainCount, domains);
            uint64_t *ids = pl_calloc_ptr(tmp, props.timeDomainCount, ids);
            props.pTimeDomains = domains;
            props.pTimeDomainIds = ids;
            res = vk->GetSwapchainTimeDomainPropertiesEXT(vk->dev, current->swapchain,
                                                           &props, &counter);
#ifdef PL_HAVE_WIN32
            const enum pl_swapchain_present_clock preferred =
                PL_SWAPCHAIN_PRESENT_CLOCK_PERFORMANCE_COUNTER;
#else
            const enum pl_swapchain_present_clock preferred =
                PL_SWAPCHAIN_PRESENT_CLOCK_MONOTONIC;
#endif
            if (res == VK_SUCCESS || res == VK_INCOMPLETE) {
                p->feedback_clock = vk_present_feedback_pick_clock(
                    domains, ids, props.timeDomainCount, preferred,
                    &p->feedback_clock_id);
            }
            pl_free(tmp);
        }
        if (p->feedback_clock == PL_SWAPCHAIN_PRESENT_CLOCK_UNKNOWN) {
            PL_WARN(vk, "No presentation timing clock is available");
            p->feedback_surface_supported = false;
        }
    }

    // Get the new swapchain images
    VK(vk->GetSwapchainImagesKHR(vk->dev, current->swapchain, &num_images, NULL));
    PL_ARRAY_RESIZE(current, current->vkimages, num_images);
    VK(vk->GetSwapchainImagesKHR(vk->dev, current->swapchain, &num_images, current->vkimages.elem));

    static const VkSemaphoreCreateInfo seminfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    static const VkFenceCreateInfo fenceinfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    pl_assert(num_images > 0);

    PL_ARRAY_CLEAR(current, current->images, num_images);

    // Without swapchain_maintenance1, we cannot use a fence to know when an
    // acquisition semaphore is safe to reuse. We allocate an extra "spare"
    // to ensure we always have one available for vkAcquireNextImageKHR while
    // the others are potentially still in flight.
    PL_ARRAY_CLEAR(current, current->sems_in, num_images + !p->has_swapchain_maintenance1);
    for (int i = 0; i < current->sems_in.num; i++) {
        VK(vk->CreateSemaphore(vk->dev, &seminfo, PL_VK_ALLOC, &current->sems_in.elem[i]));
        snprintf(name, sizeof(name), "swapchain in #%d", i);
        PL_VK_NAME(SEMAPHORE, current->sems_in.elem[i], name);
    }

    PL_ARRAY_CLEAR(current, current->sems_out, num_images);
    for (int i = 0; i < current->sems_out.num; i++) {
        VK(vk->CreateSemaphore(vk->dev, &seminfo, PL_VK_ALLOC, &current->sems_out.elem[i]));
        snprintf(name, sizeof(name), "swapchain out #%d", i);
        PL_VK_NAME(SEMAPHORE, current->sems_out.elem[i], name);
    }

    for (int i = 0; i < num_images && p->has_swapchain_maintenance1; i++) {
        VkFence fence;
        VK(vk->CreateFence(vk->dev, &fenceinfo, PL_VK_ALLOC, &fence));
        snprintf(name, sizeof(name), "present fence #%d", i);
        PL_VK_NAME(FENCE, fence, name);
        PL_ARRAY_APPEND(current, current->fences_out, fence);
    }

    // Note: `p->color_space.hdr` is (re-)applied by `set_hdr_metadata`
    map_color_space(sinfo.imageColorSpace, &p->color_space);

    // To convert to XYZ color system for VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT
    if (p->color_space.transfer == PL_COLOR_TRC_ST428) {
        p->color_repr.sys = PL_COLOR_SYSTEM_XYZ;
    } else {
        p->color_repr.sys = PL_COLOR_SYSTEM_RGB;
    }

    // Forcibly re-apply HDR metadata, bypassing the no-op check
    struct pl_hdr_metadata metadata = p->hdr_metadata;
    p->hdr_metadata = pl_hdr_metadata_empty;
    set_hdr_metadata(p, &metadata);

#ifdef PL_HAVE_WIN32
    if (!acquire_exclusive(p))
        goto error;
#endif
    return true;

error:
    PL_ERR(vk, "Failed (re)creating swapchain!");
    release_exclusive(p);
#ifdef PL_HAVE_WIN32
    if (p->exclusive_monitor && (p->exclusive_result == VK_SUCCESS ||
                                 p->exclusive_result == VK_NOT_READY)) {
        p->exclusive_result = VK_ERROR_INITIALIZATION_FAILED;
        publish_exclusive(p);
    }
#endif
    swapchain_destroy(sw, &p->current, UINT64_MAX);
    p->cur_width = p->cur_height = 0;
    return false;
}

static bool vk_sw_start_frame(pl_swapchain sw,
                              struct pl_swapchain_frame *out_frame)
{
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    pl_mutex_lock(&p->lock);

    bool recreate = !p->current || p->needs_recreate;
    if (p->suboptimal && !p->params.allow_suboptimal)
        recreate = true;

    if (recreate && !vk_sw_recreate(sw, 0, 0)) {
        pl_mutex_unlock(&p->lock);
        return false;
    }

#ifdef PL_HAVE_WIN32
    // Never continue ordinary presentation after exclusive access was lost.
    // Retrying on the same live swapchain is explicitly allowed by Vulkan.
    if (!acquire_exclusive(p)) {
        pl_mutex_unlock(&p->lock);
        return false;
    }
#endif

    for (int attempts = 0; attempts < 2; attempts++) {
        VkSemaphore sem_in = p->current->sems_in.elem[p->current->idx_sems_in];
        p->current->idx_sems_in = (p->current->idx_sems_in + 1) % p->current->sems_in.num;
        PL_TRACE(vk, "vkAcquireNextImageKHR signals 0x%"PRIx64, (uint64_t) sem_in);

        uint32_t imgidx = 0;
        VkResult res = vk->AcquireNextImageKHR(vk->dev, p->current->swapchain, UINT64_MAX,
                                               sem_in, VK_NULL_HANDLE, &imgidx);

        switch (res) {
        case VK_SUBOPTIMAL_KHR:
            p->suboptimal = true;
            // fall through
        case VK_SUCCESS:
            p->current->last_imgidx = imgidx;
            if (p->current->fences_out.num > 0) {
                VkFence *pfence = &p->current->fences_out.elem[imgidx];
                vk->WaitForFences(vk->dev, 1, pfence, VK_TRUE, UINT64_MAX);
                vk->ResetFences(vk->dev, 1, pfence);
            }
            if (!vk_sw_image_init(sw, imgidx))
                goto error;
            pl_vulkan_release_ex(sw->gpu, pl_vulkan_release_params(
                .tex        = p->current->images.elem[imgidx],
                .layout     = VK_IMAGE_LAYOUT_UNDEFINED,
                .qf         = VK_QUEUE_FAMILY_IGNORED,
                .semaphore  = { sem_in },
            ));
            *out_frame = (struct pl_swapchain_frame) {
                .fbo = p->current->images.elem[imgidx],
                .flipped = false,
                .color_repr = p->color_repr,
                .color_space = p->color_space,
            };
            // keep lock held
            return true;

#ifdef PL_HAVE_WIN32
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
            exclusive_lost(p);
            pl_mutex_unlock(&p->lock);
            return false;
#endif

        case VK_ERROR_OUT_OF_DATE_KHR: {
            // In these cases try recreating the swapchain
            if (!vk_sw_recreate(sw, 0, 0)) {
                pl_mutex_unlock(&p->lock);
                return false;
            }
            continue;
        }

        default:
            PL_ERR(vk, "Failed acquiring swapchain image: %s", vk_res_str(res));
            goto error;
        }
    }

error:
    // If we've exhausted the number of attempts to recreate the swapchain,
    // just give up silently and let the user retry some time later.
    pl_mutex_unlock(&p->lock);
    return false;
}

static void present_cb(struct priv *p, void *arg)
{
    (void) pl_rc_deref(&p->frames_in_flight);
}

VK_CB_FUNC_DEF(present_cb);

static enum pl_swapchain_submit_result vk_sw_submit_frame_ex(
    pl_swapchain sw, const struct pl_swapchain_submit_params *params)
{
    pl_gpu gpu = sw->gpu;
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    struct vk_swapchain *current = p->current;
    pl_assert(current);
    pl_assert(current->last_imgidx >= 0);
    uint32_t idx = current->last_imgidx;
    VkSemaphore sem_out = current->sems_out.elem[idx];
    current->last_imgidx = -1;

    pl_clock_t t0 = vk->trace_present ? pl_clock_now() : 0;
    bool held = pl_vulkan_hold_ex(gpu, pl_vulkan_hold_params(
        .tex        = current->images.elem[idx],
        .layout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .qf         = VK_QUEUE_FAMILY_IGNORED,
        .semaphore  = { sem_out },
    ));
    if (!held) {
        PL_ERR(gpu, "Failed holding swapchain image for presentation");
        pl_mutex_unlock(&p->lock);
        return PL_SWAPCHAIN_SUBMIT_FAILED;
    }

    pl_clock_t t1 = vk->trace_present ? pl_clock_now() : 0;
    struct vk_cmd *cmd = CMD_BEGIN(GRAPHICS);
    pl_clock_t t2 = vk->trace_present ? pl_clock_now() : 0;
    if (!cmd) {
        pl_mutex_unlock(&p->lock);
        return PL_SWAPCHAIN_SUBMIT_FAILED;
    }

    pl_rc_ref(&p->frames_in_flight);
    vk_cmd_callback(cmd, VK_CB_FUNC(present_cb), p, NULL);
    int qidx = cmd->qindex;
    if (!CMD_SUBMIT(&cmd)) {
        pl_mutex_unlock(&p->lock);
        return PL_SWAPCHAIN_SUBMIT_FAILED;
    }
    pl_clock_t t3 = vk->trace_present ? pl_clock_now() : 0;
    struct vk_cmdpool *pool = vk->pool_graphics;
    VkQueue queue = pool->queues[qidx];

    vk_rotate_queues(p->vk);
    vk_malloc_garbage_collect(vk->ma);
    cleanup_retired_swapchains(sw, 0);

    pl_clock_t t4 = vk->trace_present ? pl_clock_now() : 0;
    VkSwapchainPresentFenceInfoKHR fenceInfo = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
        .swapchainCount = 1,
    };

    VkPresentInfoKHR pinfo = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &sem_out,
        .swapchainCount = 1,
        .pSwapchains = &current->swapchain,
        .pImageIndices = &idx,
    };

    if (current->fences_out.num > 0) {
        VkFence *pfence = &current->fences_out.elem[idx];
        fenceInfo.pFences = pfence;
        vk_link_struct(&pinfo, &fenceInfo);
    }

    bool tracked = p->feedback_surface_supported && params &&
        params->feedback_stages == PL_SWAPCHAIN_PRESENT_STAGE_FIRST_PIXEL_OUT;
    uint64_t present_id = 0;
    VkPresentId2KHR present_id_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR,
        .swapchainCount = 1,
        .pPresentIds = &present_id,
    };
    VkPresentTimingInfoEXT timing_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT,
        .timeDomainId = p->feedback_clock_id,
        .presentStageQueries = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT,
        .targetTimeDomainPresentStage =
            p->feedback_clock == PL_SWAPCHAIN_PRESENT_CLOCK_PRESENT_STAGE_LOCAL
                ? VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT : 0,
    };
    VkPresentTimingsInfoEXT timings_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT,
        .swapchainCount = 1,
        .pTimingInfos = &timing_info,
    };
    if (tracked) {
        present_id = vk_present_feedback_track((void *) sw, &p->feedback,
                                               params->token);
        vk_link_struct(&pinfo, &present_id_info);
        vk_link_struct(&pinfo, &timings_info);
    }

    PL_TRACE(vk, "vkQueuePresentKHR waits on 0x%"PRIx64, (uint64_t) sem_out);
    pl_clock_t t5 = vk->trace_present ? pl_clock_now() : 0;
    vk->lock_queue(vk->queue_ctx, pool->qf, qidx);
    pl_clock_t t6 = vk->trace_present ? pl_clock_now() : 0;
    VkResult res = vk->QueuePresentKHR(queue, &pinfo);
    if (res == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT && tracked) {
        vk_present_feedback_discard(&p->feedback, present_id);
        tracked = false;
        pinfo.pNext = current->fences_out.num > 0 ? &fenceInfo : NULL;
        fenceInfo.pNext = NULL;
        res = vk->QueuePresentKHR(queue, &pinfo);
    }
    pl_clock_t t7 = vk->trace_present ? pl_clock_now() : 0;
    vk->unlock_queue(vk->queue_ctx, pool->qf, qidx);
    if (vk->trace_present) {
        PL_INFO(vk, "[PresentTrace] present start=%"PRIu64" image=%u size=%dx%d mode=%d "
                "hold_ms=%.3f cmd_begin_ms=%.3f cmd_submit_ms=%.3f maintenance_ms=%.3f "
                "setup_ms=%.3f lock_ms=%.3f driver_ms=%.3f total_ms=%.3f result=%d",
                (uint64_t) t0, idx, p->cur_width, p->cur_height, (int) p->protoInfo.presentMode,
                pl_clock_diff(t1, t0) * 1e3, pl_clock_diff(t2, t1) * 1e3,
                pl_clock_diff(t3, t2) * 1e3, pl_clock_diff(t4, t3) * 1e3,
                pl_clock_diff(t5, t4) * 1e3, pl_clock_diff(t6, t5) * 1e3,
                pl_clock_diff(t7, t6) * 1e3, pl_clock_diff(t7, t0) * 1e3, (int) res);
    }
#ifdef PL_HAVE_WIN32
    if (res == VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT) {
        if (tracked)
            vk_present_feedback_discard(&p->feedback, present_id);
        exclusive_lost(p);
        pl_mutex_unlock(&p->lock);
        return PL_SWAPCHAIN_SUBMIT_FAILED;
    }
#endif

    switch (res) {
    case VK_SUBOPTIMAL_KHR:
        p->suboptimal = true;
        // fall through
    case VK_SUCCESS:
        pl_mutex_unlock(&p->lock);
        return tracked ? PL_SWAPCHAIN_SUBMITTED_TRACKED
                       : PL_SWAPCHAIN_SUBMITTED_UNTRACKED;

    case VK_ERROR_OUT_OF_DATE_KHR:
        // We can silently ignore this error, since the next start_frame will
        // recreate the swapchain automatically.
        if (tracked)
            vk_present_feedback_discard(&p->feedback, present_id);
        pl_mutex_unlock(&p->lock);
        return PL_SWAPCHAIN_SUBMITTED_UNTRACKED;

    default:
        if (tracked)
            vk_present_feedback_discard(&p->feedback, present_id);
        PL_ERR(vk, "Failed presenting to queue %p: %s", (void *) queue,
               vk_res_str(res));
        pl_mutex_unlock(&p->lock);
        return PL_SWAPCHAIN_SUBMIT_FAILED;
    }
}

static bool vk_sw_submit_frame(pl_swapchain sw)
{
    return vk_sw_submit_frame_ex(sw, NULL) != PL_SWAPCHAIN_SUBMIT_FAILED;
}

static uint32_t vk_sw_get_present_feedback_capabilities(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    pl_mutex_lock(&p->lock);
    uint32_t stages = p->feedback_surface_supported
        ? PL_SWAPCHAIN_PRESENT_STAGE_FIRST_PIXEL_OUT : 0;
    pl_mutex_unlock(&p->lock);
    return stages;
}

static int vk_sw_poll_present_feedback(
    pl_swapchain sw, struct pl_swapchain_present_feedback *out_feedback,
    int max_feedback)
{
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    int written = 0;

    pl_mutex_lock(&p->lock);
    uint64_t token;
    while (written < max_feedback &&
           vk_present_feedback_pop_discarded(&p->feedback, &token))
    {
        out_feedback[written++] = (struct pl_swapchain_present_feedback) {
            .token = token,
            .status = PL_SWAPCHAIN_PRESENT_FEEDBACK_DISCARDED,
        };
    }

    if (written == max_feedback || !p->feedback_surface_supported || !p->current)
        goto done;

    VkPastPresentationTimingInfoEXT query = {
        .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT,
        .swapchain = p->current->swapchain,
    };
    VkPastPresentationTimingPropertiesEXT properties = {
        .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT,
    };
    VkResult res = vk->GetPastPresentationTimingEXT(vk->dev, &query, &properties);
    if (res != VK_SUCCESS || !properties.presentationTimingCount)
        goto done;

    void *tmp = pl_tmp(NULL);
    uint32_t count = PL_MIN(properties.presentationTimingCount,
                            (uint32_t) (max_feedback - written));
    VkPastPresentationTimingEXT *timings = pl_calloc_ptr(tmp, count, timings);
    VkPresentStageTimeEXT *stages = pl_calloc_ptr(tmp, count, stages);
    for (uint32_t i = 0; i < count; i++) {
        timings[i] = (VkPastPresentationTimingEXT) {
            .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT,
            .presentStageCount = 1,
            .pPresentStages = &stages[i],
        };
    }
    properties.presentationTimingCount = count;
    properties.pPresentationTimings = timings;
    res = vk->GetPastPresentationTimingEXT(vk->dev, &query, &properties);
    if (res != VK_SUCCESS && res != VK_INCOMPLETE) {
        PL_WARN(vk, "Failed polling presentation timing: %s", vk_res_str(res));
        pl_free(tmp);
        goto done;
    }

    for (uint32_t i = 0; i < properties.presentationTimingCount; i++) {
        if (!timings[i].reportComplete ||
            !vk_present_feedback_complete(&p->feedback, timings[i].presentId, &token))
            continue;

        bool first_pixel_out = false;
        uint64_t first_pixel_out_time = 0;
        for (uint32_t n = 0; n < timings[i].presentStageCount; n++) {
            if ((timings[i].pPresentStages[n].stage &
                 VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT) &&
                timings[i].pPresentStages[n].time != 0)
            {
                first_pixel_out = true;
                first_pixel_out_time = timings[i].pPresentStages[n].time;
                break;
            }
        }

        out_feedback[written++] = (struct pl_swapchain_present_feedback) {
            .token = token,
            .completed_stages = first_pixel_out
                ? PL_SWAPCHAIN_PRESENT_STAGE_FIRST_PIXEL_OUT : 0,
            .status = first_pixel_out ? PL_SWAPCHAIN_PRESENT_FEEDBACK_COMPLETED
                                      : PL_SWAPCHAIN_PRESENT_FEEDBACK_DISCARDED,
            .timestamp = first_pixel_out_time,
            .clock = vk_present_feedback_map_clock(timings[i].timeDomain),
            .clock_id = timings[i].timeDomainId,
        };
    }
    pl_free(tmp);

done:
    pl_mutex_unlock(&p->lock);
    return written;
}

static void vk_sw_swap_buffers(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);

    pl_mutex_lock(&p->lock);
    while (pl_rc_count(&p->frames_in_flight) >= p->swapchain_depth) {
        pl_mutex_unlock(&p->lock); // don't hold mutex while blocking
        vk_poll_commands(p->vk, UINT64_MAX);
        pl_mutex_lock(&p->lock);
    }
    pl_mutex_unlock(&p->lock);
}

static bool vk_sw_resize(pl_swapchain sw, int *width, int *height)
{
    struct priv *p = PL_PRIV(sw);
    bool ok = true;

    pl_mutex_lock(&p->lock);

    bool width_changed = *width && *width != p->cur_width,
         height_changed = *height && *height != p->cur_height;

    if (p->suboptimal || p->needs_recreate || width_changed || height_changed)
        ok = vk_sw_recreate(sw, *width, *height);

    *width = p->cur_width;
    *height = p->cur_height;

    pl_mutex_unlock(&p->lock);
    return ok;
}

static void vk_sw_colorspace_hint(pl_swapchain sw, const struct pl_color_space *csp)
{
    struct priv *p = PL_PRIV(sw);
    pl_mutex_lock(&p->lock);

    // This should never fail if the swapchain already exists
    bool ok = pick_surf_format(sw, csp);

    if (p->protoInfo.imageColorSpace == VK_COLOR_SPACE_PASS_THROUGH_EXT) {
        // Don't try to apply anything for VK_COLOR_SPACE_PASS_THROUGH_EXT and
        // also immediately cleanup the retired swapchain. This is needed
        // because the driver might hold Wayland color surface parented to that
        // swapchain.
        if (p->needs_recreate) {
            vk_sw_recreate(sw, 0, 0);
            cleanup_retired_swapchains(sw, UINT64_MAX);
        }
    } else {
        set_hdr_metadata(p, &csp->hdr);
    }
    pl_assert(ok);

    pl_mutex_unlock(&p->lock);
}

VkResult pl_vulkan_swapchain_request_exclusive(pl_swapchain sw, void *native_monitor)
{
#ifdef PL_HAVE_WIN32
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    pl_mutex_lock(&p->lock);
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    if (p->current || !native_monitor)
        goto done;
    result = VK_ERROR_EXTENSION_NOT_PRESENT;
    if (!vk->AcquireFullScreenExclusiveModeEXT ||
        !vk->ReleaseFullScreenExclusiveModeEXT ||
        !vk->GetPhysicalDeviceSurfaceCapabilities2KHR)
        goto done;
    p->exclusive_monitor = (HMONITOR) native_monitor;
    p->exclusive_result = VK_NOT_READY;
    publish_exclusive(p);
    p->needs_recreate = true;
    result = VK_SUCCESS;
done:
    pl_mutex_unlock(&p->lock);
    return result;
#else
    (void) sw;
    (void) native_monitor;
    return VK_ERROR_EXTENSION_NOT_PRESENT;
#endif
}

VkResult pl_vulkan_swapchain_set_exclusive_target(pl_swapchain sw,
                                                 void *native_monitor)
{
#ifdef PL_HAVE_WIN32
    struct priv *p = PL_PRIV(sw);
    struct vk_ctx *vk = p->vk;
    pl_mutex_lock(&p->lock);
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    if (p->current && p->current->last_imgidx >= 0)
        goto done;
    result = VK_ERROR_EXTENSION_NOT_PRESENT;
    if (!vk->AcquireFullScreenExclusiveModeEXT ||
        !vk->ReleaseFullScreenExclusiveModeEXT ||
        !vk->GetPhysicalDeviceSurfaceCapabilities2KHR)
        goto done;
    result = VK_SUCCESS;
    if (p->exclusive_managed &&
        p->exclusive_monitor == (HMONITOR) native_monitor)
        goto done;

    release_exclusive(p);
    p->exclusive_managed = true;
    p->exclusive_monitor = (HMONITOR) native_monitor;
    p->exclusive_result = VK_NOT_READY;
    publish_exclusive(p);
    p->needs_recreate = true;
done:
    pl_mutex_unlock(&p->lock);
    return result;
#else
    (void) sw;
    (void) native_monitor;
    return VK_ERROR_EXTENSION_NOT_PRESENT;
#endif
}

// Reports actual acquisition and the last acquisition/query/create/loss result.
// Does not acquire or infer exclusivity from window flags. A successful config
// initially reports VK_NOT_READY and acquired=false. Uses a nonblocking atomic
// snapshot; the caller must keep sw alive for the duration of this call.
PL_API VkResult pl_vulkan_swapchain_exclusive_status(pl_swapchain sw, bool *acquired)
{
    *acquired = false;
#ifdef PL_HAVE_WIN32
    struct priv *p = PL_PRIV(sw);
    uint64_t snapshot = atomic_load_explicit(&p->exclusive_snapshot,
                                             memory_order_acquire);
    *acquired = snapshot & 1;
    return (VkResult) (int32_t) (snapshot >> 1);
#else
    (void) sw;
    return VK_ERROR_EXTENSION_NOT_PRESENT;
#endif
}

bool pl_vulkan_swapchain_suboptimal(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    return p->suboptimal;
}

static const struct pl_sw_fns vulkan_swapchain = {
    .destroy            = vk_sw_destroy,
    .latency            = vk_sw_latency,
    .resize             = vk_sw_resize,
    .colorspace_hint    = vk_sw_colorspace_hint,
    .start_frame        = vk_sw_start_frame,
    .submit_frame       = vk_sw_submit_frame,
    .get_present_feedback_capabilities = vk_sw_get_present_feedback_capabilities,
    .submit_frame_ex    = vk_sw_submit_frame_ex,
    .poll_present_feedback = vk_sw_poll_present_feedback,
    .swap_buffers       = vk_sw_swap_buffers,
};
