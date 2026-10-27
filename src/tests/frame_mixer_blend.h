#pragma once
#include <libplacebo/frame_mixer.h>
struct blend_stats {
    unsigned submissions, polls, releases, instances;
    uint64_t content[2], allocation[2], epoch;
    bool hold, fail;
};
struct blend_config {
    enum pl_mixer_color_domain domain;
    float gain;
    struct blend_stats *stats;
};
