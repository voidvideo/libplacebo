#include "utils.h"
#include <assert.h>
#include <libplacebo/utils/frame_queue.h>

static bool map(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src,
                struct pl_frame *out)
{
    return true;
}

// Compare selection at EOF against the ordinary queue, not a second EOF policy.
static void compare(bool declared, int mode, bool interlaced, int count,
                    double origin, double target, bool eof)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue ordinary = pl_queue_create(&gpu), prepared = pl_queue_create(&gpu);
    const float duration = 0.0625f;
    const float interval = interlaced ? duration / 2 : duration;
    for (int i = 0; i < count; i++) {
        struct pl_source_frame source = {
            .pts = origin + i * duration, .duration = duration,
            .first_field = interlaced ? PL_FIELD_TOP : PL_FIELD_NONE, .map = map,
        };
        pl_queue_push(ordinary, &source);
        pl_queue_push(prepared, &source);
    }
    if (eof) {
        pl_queue_push(ordinary, NULL);
        pl_queue_push(prepared, NULL);
    }
    struct pl_queue_params params = {
        .pts = target, .radius = mode < 2 ? 0 : mode - 1,
        .vsync_duration = mode == 0 ? interval : interval / 2,
        .interpolation_threshold = mode == 0 ? 0.001f : 0,
    };
    pl_queue_plan plan = NULL;
    enum pl_queue_plan_status status;
    if (declared) {
        struct pl_queue_plan_cadence cadence = {
            .source_duration = interval, .output_duration = params.vsync_duration,
        };
        assert(pl_queue_plan_prepare(prepared, &params, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
        status = pl_queue_plan_set_target(plan, target);
    } else {
        status = pl_queue_plan_inspect(prepared, &params, &plan);
    }
    struct pl_frame_mix a = {0}, b = {0};
    enum pl_queue_status reference = pl_queue_update(ordinary, &a, &params);
    if (reference == PL_QUEUE_EOF) {
        assert(status == PL_QUEUE_PLAN_EOF);
        if (plan) {
            assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_EOF);
            assert(pl_queue_plan_consume(plan, &b) == PL_QUEUE_PLAN_EOF);
        }
    } else {
        assert(reference == PL_QUEUE_OK || reference == PL_QUEUE_MORE);
        assert(status == PL_QUEUE_PLAN_READY);
        assert(pl_queue_plan_consume(plan, &b) == PL_QUEUE_PLAN_READY);
        assert(a.num_frames == b.num_frames);
        assert(fabs(a.vsync_duration - b.vsync_duration) < 1e-6);
        for (int i = 0; i < a.num_frames; i++) {
            assert(a.signatures[i] == b.signatures[i]);
            assert(a.frames[i]->field == b.frames[i]->field);
            assert(fabs(a.timestamps[i] - b.timestamps[i]) < 1e-6);
        }
    }
    pl_queue_plan_destroy(&plan);
    pl_queue_destroy(&prepared);
    pl_queue_destroy(&ordinary);
}

int main(void)
{
    for (int declared = 0; declared < 2; declared++) {
        for (int mode = 0; mode < 4; mode++) {
            for (int interlaced = 0; interlaced < 2; interlaced++) {
                for (int count = 1; count <= 3; count++) {
                    const double end = 2.0 + count * 0.0625;
                    compare(declared, mode, interlaced, count, 2, nextafter(end, -INFINITY), true);
                    compare(declared, mode, interlaced, count, 2, end, true);
                    compare(declared, mode, interlaced, count, 2, nextafter(end, INFINITY), true);
                    compare(declared, mode, interlaced, count, 2, end + 1, true);
                }
                compare(declared, mode, interlaced, 0, 2, 2, true);
            }
            // Ordinary libplacebo holds a sole progressive image at PTS zero.
            compare(declared, mode, false, 1, 0, 4, true);
            // Absent EOF, a late target can still use the available progressive mix.
            compare(declared, mode, false, 1, 2, 2.25, false);
        }
    }
    return 0;
}
