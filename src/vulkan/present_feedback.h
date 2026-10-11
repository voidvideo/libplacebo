#pragma once

#include "common.h"

struct vk_present_feedback_entry {
    uint64_t present_id;
    uint64_t token;
};

struct vk_present_feedback_state {
    uint64_t next_present_id;
    PL_ARRAY(struct vk_present_feedback_entry) pending;
    PL_ARRAY(uint64_t) discarded;
};

enum pl_swapchain_present_clock
vk_present_feedback_map_clock(VkTimeDomainKHR domain);
enum pl_swapchain_present_clock
vk_present_feedback_pick_clock(const VkTimeDomainKHR *domains,
                               const uint64_t *domain_ids, int num_domains,
                               enum pl_swapchain_present_clock preferred,
                               uint64_t *out_domain_id);

uint64_t vk_present_feedback_track(void *parent,
                                   struct vk_present_feedback_state *state,
                                   uint64_t token);
bool vk_present_feedback_complete(struct vk_present_feedback_state *state,
                                  uint64_t present_id, uint64_t *out_token);
bool vk_present_feedback_discard(struct vk_present_feedback_state *state,
                                 uint64_t present_id);
bool vk_present_feedback_pop_discarded(struct vk_present_feedback_state *state,
                                       uint64_t *out_token);
void vk_present_feedback_reset(struct vk_present_feedback_state *state);
void vk_present_feedback_uninit(struct vk_present_feedback_state *state);
