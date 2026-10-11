#include "present_feedback.h"

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
