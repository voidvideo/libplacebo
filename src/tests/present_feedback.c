#include "vulkan/present_feedback.h"

#include "utils.h"

int main(void)
{
    const VkTimeDomainKHR domains[] = {
        VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT,
        VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR,
        VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR,
    };
    const uint64_t domain_ids[] = {11, 22, 33};
    uint64_t selected_id = 0;
    REQUIRE_CMP(vk_present_feedback_pick_clock(domains, domain_ids,
                                               PL_ARRAY_SIZE(domains),
                                               PL_SWAPCHAIN_PRESENT_CLOCK_MONOTONIC,
                                               &selected_id),
                ==, PL_SWAPCHAIN_PRESENT_CLOCK_MONOTONIC, "d");
    REQUIRE_CMP((unsigned long long) selected_id, ==, 22, "llu");
    REQUIRE_CMP(vk_present_feedback_map_clock(VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR),
                ==, PL_SWAPCHAIN_PRESENT_CLOCK_PERFORMANCE_COUNTER, "d");
    REQUIRE_CMP(vk_present_feedback_map_clock(VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT),
                ==, PL_SWAPCHAIN_PRESENT_CLOCK_SWAPCHAIN_LOCAL, "d");

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
