#pragma once

#include <libplacebo/shaders/fruc.h>

// Internal NVIDIA optical-flow resources. Compute shaders and hardware optical
// flow share these images through the GPU backend's command dependency tracker.
struct pl_fruc {
    pl_tex gray[2], flow[2];
    int width, height;
};

bool pl_fruc_describe(pl_gpu gpu, int width, int height,
                      struct pl_hook_texture gray[2], struct pl_hook_texture flow[2]);
struct pl_fruc *pl_fruc_create(pl_gpu gpu, int width, int height);
bool pl_fruc_execute(pl_gpu gpu, struct pl_fruc *fruc);
void pl_fruc_destroy(pl_gpu gpu, struct pl_fruc **fruc);
