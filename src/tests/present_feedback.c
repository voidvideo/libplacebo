#include "vulkan/present_feedback.h"

#include "utils.h"

int main(void)
{
    struct vk_present_feedback_state state = {0};

    uint64_t first = vk_present_feedback_track(NULL, &state, 41);
    uint64_t second = vk_present_feedback_track(NULL, &state, 99);
    REQUIRE_CMP((unsigned long long) first, >, 0, "llu");
    REQUIRE_CMP((unsigned long long) second, >,
                (unsigned long long) first, "llu");

    uint64_t token = 0;
    REQUIRE(!vk_present_feedback_complete(&state, 123456, &token));
    REQUIRE(vk_present_feedback_complete(&state, second, &token));
    REQUIRE_CMP((unsigned long long) token, ==, 99, "llu");
    REQUIRE(!vk_present_feedback_complete(&state, second, &token));

    REQUIRE(vk_present_feedback_discard(&state, first));
    REQUIRE(!vk_present_feedback_complete(&state, first, &token));

    uint64_t before_reset = vk_present_feedback_track(NULL, &state, 7);
    vk_present_feedback_reset(&state);
    REQUIRE(!vk_present_feedback_complete(&state, before_reset, &token));
    REQUIRE(vk_present_feedback_pop_discarded(&state, &token));
    REQUIRE_CMP((unsigned long long) token, ==, 7, "llu");
    REQUIRE(!vk_present_feedback_pop_discarded(&state, &token));
    REQUIRE_CMP((unsigned long long) vk_present_feedback_track(NULL, &state, 8),
                ==, 1, "llu");

    vk_present_feedback_uninit(&state);
}
