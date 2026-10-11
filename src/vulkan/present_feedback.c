#include "present_feedback.h"

enum pl_swapchain_present_clock
vk_present_feedback_map_clock(VkTimeDomainKHR domain)
{
    switch (domain) {
    case VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR:
        return PL_SWAPCHAIN_PRESENT_CLOCK_MONOTONIC;
    case VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR:
        return PL_SWAPCHAIN_PRESENT_CLOCK_MONOTONIC_RAW;
    case VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR:
        return PL_SWAPCHAIN_PRESENT_CLOCK_PERFORMANCE_COUNTER;
    case VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT:
        return PL_SWAPCHAIN_PRESENT_CLOCK_PRESENT_STAGE_LOCAL;
    case VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT:
        return PL_SWAPCHAIN_PRESENT_CLOCK_SWAPCHAIN_LOCAL;
    case VK_TIME_DOMAIN_DEVICE_KHR:
        return PL_SWAPCHAIN_PRESENT_CLOCK_DEVICE;
    default:
        return PL_SWAPCHAIN_PRESENT_CLOCK_UNKNOWN;
    }
}

enum pl_swapchain_present_clock
vk_present_feedback_pick_clock(const VkTimeDomainKHR *domains,
                               const uint64_t *domain_ids, int num_domains,
                               enum pl_swapchain_present_clock preferred,
                               uint64_t *out_domain_id)
{
    static const enum pl_swapchain_present_clock fallbacks[] = {
        PL_SWAPCHAIN_PRESENT_CLOCK_SWAPCHAIN_LOCAL,
        PL_SWAPCHAIN_PRESENT_CLOCK_PRESENT_STAGE_LOCAL,
        PL_SWAPCHAIN_PRESENT_CLOCK_MONOTONIC,
        PL_SWAPCHAIN_PRESENT_CLOCK_MONOTONIC_RAW,
        PL_SWAPCHAIN_PRESENT_CLOCK_PERFORMANCE_COUNTER,
        PL_SWAPCHAIN_PRESENT_CLOCK_DEVICE,
    };
    if (out_domain_id)
        *out_domain_id = 0;
    if (!domains || !domain_ids || num_domains <= 0)
        return PL_SWAPCHAIN_PRESENT_CLOCK_UNKNOWN;

    for (int pass = -1; pass < (int) PL_ARRAY_SIZE(fallbacks); pass++) {
        enum pl_swapchain_present_clock wanted =
            pass < 0 ? preferred : fallbacks[pass];
        if (pass >= 0 && wanted == preferred)
            continue;
        for (int i = 0; i < num_domains; i++) {
            if (vk_present_feedback_map_clock(domains[i]) != wanted)
                continue;
            if (out_domain_id)
                *out_domain_id = domain_ids[i];
            return wanted;
        }
    }
    return PL_SWAPCHAIN_PRESENT_CLOCK_UNKNOWN;
}

uint64_t vk_present_feedback_track(void *parent,
                                   struct vk_present_feedback_state *state,
                                   uint64_t token)
{
    if (!state->next_present_id)
        state->next_present_id = 1;

    uint64_t present_id = state->next_present_id++;
    PL_ARRAY_APPEND(parent, state->pending, (struct vk_present_feedback_entry) {
        .present_id = present_id,
        .token = token,
    });
    return present_id;
}

bool vk_present_feedback_complete(struct vk_present_feedback_state *state,
                                  uint64_t present_id, uint64_t *out_token)
{
    for (int i = 0; i < state->pending.num; i++) {
        if (state->pending.elem[i].present_id != present_id)
            continue;

        if (out_token)
            *out_token = state->pending.elem[i].token;
        PL_ARRAY_REMOVE_AT(state->pending, i);
        return true;
    }

    return false;
}

bool vk_present_feedback_discard(struct vk_present_feedback_state *state,
                                 uint64_t present_id)
{
    return vk_present_feedback_complete(state, present_id, NULL);
}

bool vk_present_feedback_pop_discarded(struct vk_present_feedback_state *state,
                                       uint64_t *out_token)
{
    if (!state->discarded.num)
        return false;

    if (out_token)
        *out_token = state->discarded.elem[0];
    PL_ARRAY_REMOVE_AT(state->discarded, 0);
    return true;
}

void vk_present_feedback_reset(struct vk_present_feedback_state *state)
{
    for (int i = 0; i < state->pending.num; i++)
        PL_ARRAY_APPEND(NULL, state->discarded, state->pending.elem[i].token);
    state->next_present_id = 1;
    state->pending.num = 0;
}

void vk_present_feedback_uninit(struct vk_present_feedback_state *state)
{
    pl_free_ptr(&state->pending.elem);
    pl_free_ptr(&state->discarded.elem);
    *state = (struct vk_present_feedback_state) {0};
}
