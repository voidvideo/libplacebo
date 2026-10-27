/*
 * NVIDIA optical-flow backend for the FFmpeg FRUC port.
 * Copyright (C) 2026 Philip Langdale <philipl@overt.org>
 * Based on FFmpeg vf_fruc_vulkan.c at
 * 9d2d5b1aa39a714ad92cd3dd0253bd3aa7ee8be7.
 * LGPL-2.1-or-later; see LICENSE.
 *
 * Uses libplacebo command tracking and concurrent images in place of FFmpeg's
 * execution pools and maintenance9 ownership transfers.
 */
#include "gpu.h"
#include "../fruc.h"

struct fruc_vk {
    struct pl_fruc pub;
    struct vk_ctx *vk;
    struct vk_cmdpool *pool;
    VkOpticalFlowSessionNV session;
    VkImage images[4];
    VkImageView views[4];
    VkDeviceMemory memory[4];
};

static struct vk_cmdpool *optical_pool(pl_gpu gpu)
{
    pl_vulkan vulkan = pl_vulkan_get(gpu);
    if (!vulkan) return NULL;
    struct vk_ctx *vk = PL_PRIV(vulkan);
    const VkPhysicalDeviceOpticalFlowFeaturesNV *features =
        (const void *) vk_find_struct(&vk->features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV);
    if (!features || !features->opticalFlow || !vk->CreateOpticalFlowSessionNV)
        return NULL;
    for (int i = 0; i < vk->pools.num; i++)
        if (vk->pools.elem[i]->props.queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV)
            return vk->pools.elem[i];
    return NULL;
}

static bool format_supported(struct vk_ctx *vk, VkOpticalFlowUsageFlagsNV usage, VkFormat format)
{
    VkOpticalFlowImageFormatInfoNV info = {
        .sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV, .usage = usage,
    };
    uint32_t count = 0;
    if (vk->GetPhysicalDeviceOpticalFlowImageFormatsNV(vk->physd, &info, &count, NULL) != VK_SUCCESS || !count)
        return false;
    VkOpticalFlowImageFormatPropertiesNV *formats = pl_calloc_ptr(NULL, count, formats);
    for (uint32_t i = 0; i < count; i++)
        formats[i].sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES_NV;
    bool found = false;
    if (vk->GetPhysicalDeviceOpticalFlowImageFormatsNV(vk->physd, &info, &count, formats) == VK_SUCCESS)
        for (uint32_t i = 0; i < count; i++) found |= formats[i].format == format;
    pl_free(formats);
    return found;
}

static int describe(pl_gpu gpu, int width, int height,
                     struct pl_hook_texture gray[2], struct pl_hook_texture flow[2])
{
    struct vk_cmdpool *pool = optical_pool(gpu);
    if (!pool || width <= 0 || height <= 0) return 0;
    struct vk_ctx *vk = pool->vk;
    VkPhysicalDeviceOpticalFlowPropertiesNV props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES_NV,
    };
    VkPhysicalDeviceProperties2 query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &props,
    };
    vk->GetPhysicalDeviceProperties2(vk->physd, &query);
    if (!props.bidirectionalFlowSupported || width < props.minWidth || height < props.minHeight ||
        width > props.maxWidth || height > props.maxHeight ||
        !format_supported(vk, VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV, VK_FORMAT_R8_UNORM) ||
        !format_supported(vk, VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV, VK_FORMAT_R16G16_SFIXED5_NV))
        return 0;
    int grid = 0;
    for (int bit = 1; bit <= 8; bit *= 2)
        if (props.supportedOutputGridSizes & bit) { grid = bit; break; }
    if (!grid) return 0;
    pl_fmt gray_fmt = pl_find_named_fmt(gpu, "r8");
    pl_fmt flow_fmt = NULL;
    for (int i = 0; i < gpu->num_formats; i++) {
        struct pl_fmt_vk *f = PL_PRIV(gpu->formats[i]);
        if (f->vk_fmt->tfmt == VK_FORMAT_R16G16_SINT) flow_fmt = gpu->formats[i];
    }
    const enum pl_fmt_caps gray_caps = PL_FMT_CAP_STORABLE | PL_FMT_CAP_SAMPLEABLE |
                                      PL_FMT_CAP_RENDERABLE;
    if (!gray_fmt || !flow_fmt || (gray_fmt->caps & gray_caps) != gray_caps ||
        !(flow_fmt->caps & PL_FMT_CAP_SAMPLEABLE)) return 0;
    for (int i = 0; i < 2; i++) {
        gray[i] = (struct pl_hook_texture) { .params = {
            .w = width, .h = height, .format = gray_fmt,
            .sampleable = true, .storable = true, .renderable = true,
        }};
        flow[i] = (struct pl_hook_texture) { .params = {
            .w = (width + grid - 1) / grid, .h = (height + grid - 1) / grid,
            .format = flow_fmt, .sampleable = true,
        }};
    }
    return grid;
}

bool pl_fruc_describe(pl_gpu gpu, int width, int height,
                      struct pl_hook_texture gray[2], struct pl_hook_texture flow[2])
{
    return describe(gpu, width, height, gray, flow) != 0;
}

static void free_resources(void *priv, void *arg)
{
    struct fruc_vk *f = arg;
    struct vk_ctx *vk = f->vk;
    if (f->session) vk->DestroyOpticalFlowSessionNV(vk->dev, f->session, PL_VK_ALLOC);
    for (int i = 0; i < 4; i++) {
        if (f->views[i]) vk->DestroyImageView(vk->dev, f->views[i], PL_VK_ALLOC);
        if (f->images[i]) vk->DestroyImage(vk->dev, f->images[i], PL_VK_ALLOC);
        if (f->memory[i]) vk->FreeMemory(vk->dev, f->memory[i], PL_VK_ALLOC);
    }
    pl_free(f);
}

void pl_fruc_destroy(pl_gpu gpu, struct pl_fruc **ptr)
{
    if (!ptr || !*ptr) return;
    struct fruc_vk *f = (void *) *ptr;
    *ptr = NULL;
    for (int i = 0; i < 2; i++) {
        pl_tex_destroy(gpu, &f->pub.gray[i]);
        pl_tex_destroy(gpu, &f->pub.flow[i]);
    }
    vk_gpu_idle_callback(gpu, free_resources, NULL, f);
}

static bool create_image(pl_gpu gpu, struct fruc_vk *f, int i, const struct pl_tex_params *p)
{
    struct vk_ctx *vk = f->vk;
    const bool flow = i >= 2;
    VkFormat format = flow ? VK_FORMAT_R16G16_SFIXED5_NV : VK_FORMAT_R8_UNORM;
    VkImageUsageFlags shader_usage = VK_IMAGE_USAGE_SAMPLED_BIT |
        (flow ? 0 : VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    VkOpticalFlowImageFormatInfoNV of = {
        .sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV,
        .usage = flow ? VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV : VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV,
    };
    uint32_t *families = pl_alloc(NULL, vk->pools.num * sizeof(*families));
    for (int j = 0; j < vk->pools.num; j++) families[j] = vk->pools.elem[j]->qf;
    VkImageCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &of,
        .flags = flow ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0,
        .imageType = VK_IMAGE_TYPE_2D, .format = format,
        .extent = {p->w, p->h, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = shader_usage,
        .sharingMode = vk->pools.num > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = vk->pools.num, .pQueueFamilyIndices = families,
    };
    VkResult result = vk->CreateImage(vk->dev, &info, PL_VK_ALLOC, &f->images[i]);
    pl_free(families);
    if (result != VK_SUCCESS) return false;
    VkImageMemoryRequirementsInfo2 requirements = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .image = f->images[i],
    };
    VkMemoryRequirements2 req2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vk->GetImageMemoryRequirements2(vk->dev, &requirements, &req2);
    VkMemoryRequirements req = req2.memoryRequirements;
    VkPhysicalDeviceMemoryProperties props;
    vk->GetPhysicalDeviceMemoryProperties(vk->physd, &props);
    uint32_t type = UINT32_MAX;
    for (uint32_t j = 0; j < props.memoryTypeCount; j++)
        if ((req.memoryTypeBits & (1u << j)) &&
            (props.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = j; break;
        }
    if (type == UINT32_MAX) return false;
    VkMemoryAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = type,
    };
    if (vk->AllocateMemory(vk->dev, &alloc, PL_VK_ALLOC, &f->memory[i]) != VK_SUCCESS ||
        vk->BindImageMemory(vk->dev, f->images[i], f->memory[i], 0) != VK_SUCCESS) return false;
    VkImageViewUsageCreateInfo usage = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = shader_usage,
    };
    VkImageViewCreateInfo view = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &usage,
        .image = f->images[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1},
    };
    if (vk->CreateImageView(vk->dev, &view, PL_VK_ALLOC, &f->views[i]) != VK_SUCCESS) return false;
    pl_tex *tex = flow ? &f->pub.flow[i-2] : &f->pub.gray[i];
    *tex = pl_vulkan_wrap(gpu, pl_vulkan_wrap_params(
        .image = f->images[i], .width = p->w, .height = p->h,
        .format = flow ? VK_FORMAT_R16G16_SINT : format, .usage = shader_usage));
    if (!*tex) return false;
    pl_vulkan_release_ex(gpu, pl_vulkan_release_params(
        .tex = *tex, .layout = VK_IMAGE_LAYOUT_UNDEFINED, .qf = VK_QUEUE_FAMILY_IGNORED));
    return true;
}

struct pl_fruc *pl_fruc_create(pl_gpu gpu, int width, int height)
{
    struct pl_hook_texture gray[2], flow[2];
    int grid = describe(gpu, width, height, gray, flow);
    if (!grid) {
        PL_ERR(gpu, "FRUC requires enabled NVIDIA Vulkan optical flow and supported image dimensions/formats");
        return NULL;
    }
    struct fruc_vk *f = pl_zalloc_ptr(NULL, f);
    f->pool = optical_pool(gpu);
    f->vk = f->pool->vk;
    f->pub.width = width;
    f->pub.height = height;
    struct vk_ctx *vk = f->vk;
    VkOpticalFlowSessionCreateInfoNV create = {
        .sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV,
        .width = width, .height = height, .imageFormat = VK_FORMAT_R8_UNORM,
        .flowVectorFormat = VK_FORMAT_R16G16_SFIXED5_NV,
        .outputGridSize = grid, .hintGridSize = VK_OPTICAL_FLOW_GRID_SIZE_UNKNOWN_NV,
        .performanceLevel = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_SLOW_NV,
        .flags = VK_OPTICAL_FLOW_SESSION_CREATE_BOTH_DIRECTIONS_BIT_NV,
    };
    if (vk->CreateOpticalFlowSessionNV(vk->dev, &create, PL_VK_ALLOC, &f->session) != VK_SUCCESS)
        goto fail;
    const VkOpticalFlowSessionBindingPointNV points[4] = {
        VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV,
        VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV,
        VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV,
        VK_OPTICAL_FLOW_SESSION_BINDING_POINT_BACKWARD_FLOW_VECTOR_NV,
    };
    for (int i = 0; i < 4; i++) {
        if (!create_image(gpu, f, i, i < 2 ? &gray[i].params : &flow[i-2].params)) goto fail;
        if (vk->BindOpticalFlowSessionImageNV(vk->dev, f->session, points[i], f->views[i],
                                             VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS) goto fail;
    }
    return &f->pub;
fail:
    PL_ERR(gpu, "Failed creating FRUC optical-flow resources");
    struct pl_fruc *pub = &f->pub;
    pl_fruc_destroy(gpu, &pub);
    return NULL;
}

bool pl_fruc_execute(pl_gpu gpu, struct pl_fruc *pub)
{
    struct fruc_vk *f = (void *) pub;
    if (!f) return false;
    // Submit pending grayscale work before optical flow waits for it. The
    // texture dependency tracker supplies cross-queue timeline semaphores.
    pl_gpu_flush(gpu);
    struct vk_cmd *cmd = vk_cmd_begin(f->pool, PL_DEBUG_TAG);
    if (!cmd) return false;
    for (int i = 0; i < 2; i++) {
        vk_tex_barrier(gpu, cmd, pub->gray[i], VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV,
                       VK_ACCESS_2_OPTICAL_FLOW_READ_BIT_NV, VK_IMAGE_LAYOUT_GENERAL,
                       VK_QUEUE_FAMILY_IGNORED);
        vk_tex_barrier(gpu, cmd, pub->flow[i], VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV,
                       VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV, VK_IMAGE_LAYOUT_GENERAL,
                       VK_QUEUE_FAMILY_IGNORED);
    }
    VkOpticalFlowExecuteInfoNV execute = {.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV};
    f->vk->CmdOpticalFlowExecuteNV(cmd->buf, f->session, &execute);
    return vk_cmd_submit(&cmd);
}
