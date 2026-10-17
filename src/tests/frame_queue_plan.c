#include "utils.h"
#include <assert.h>
#include <libplacebo/utils/frame_queue.h>

struct sample { int mapped, discarded, unmapped; bool fail; };
static bool map(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src, struct pl_frame *out)
{
    struct sample *sample = src->frame_data;
    sample->mapped++;
    out->user_data = sample;
    return !sample->fail;
}
static void discard(const struct pl_source_frame *src)
{
    ((struct sample *) src->frame_data)->discarded++;
}
static void unmap(pl_gpu gpu, struct pl_frame *frame, const struct pl_source_frame *src)
{
    ((struct sample *) src->frame_data)->unmapped++;
}
static void push(pl_queue queue, struct sample *sample, double pts, float duration, enum pl_field field)
{
    pl_queue_push(queue, &(struct pl_source_frame) {
        .pts = pts, .duration = duration, .first_field = field,
        .frame_data = sample, .map = map, .discard = discard, .unmap = unmap,
    });
}
static void pending_publication(void)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue queue = pl_queue_create(&gpu);
    struct sample samples[6] = {0};
    struct pl_queue_params params = {.pts = 2.02, .vsync_duration = 1.0f / 60};
    pl_queue_plan plan = NULL;
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_EMPTY);
    assert(!plan);
    push(queue, &samples[0], 2.0, .04, PL_FIELD_NONE);
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_PENDING);
    const struct pl_queue_plan_info initial = *pl_queue_plan_get_info(plan);
    assert(initial.mode == PL_QUEUE_PLAN_OVERSAMPLE);
    const float fps = pl_queue_estimate_fps(queue), vps = pl_queue_estimate_vps(queue);
    for (int i = 0; i < 20; i++) {
        assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_PENDING);
        assert(pl_queue_estimate_fps(queue) == fps);
        assert(pl_queue_estimate_vps(queue) == vps);
        assert(pl_queue_num_frames(queue) == 1);
        assert(samples[0].mapped == 0 && samples[0].discarded == 0);
    }
    // Changed publication estimates cannot change this plan's normalization.
    push(queue, &samples[1], 2.05, .04, PL_FIELD_NONE);
    assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_READY);
    assert(pl_queue_plan_get_info(plan)->frame_duration == initial.frame_duration);
    assert(pl_queue_plan_get_info(plan)->pts == initial.pts);
    int count;
    const struct pl_queue_plan_sample *selected = pl_queue_plan_get_samples(plan, &count);
    assert(count == 2 && selected[0].pts == 2.0 && selected[1].pts == 2.05);
    struct pl_queue_plan_sample expected[8];
    assert(count <= PL_ARRAY_SIZE(expected));
    memcpy(expected, selected, count * sizeof(expected[0]));
    selected = expected;
    struct pl_frame_mix mix;
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
    assert(mix.num_frames == count);
    for (int i = 0; i < count; i++) {
        assert(mix.signatures[i] == selected[i].signature);
        assert(fabs(mix.timestamps[i] - (selected[i].pts - initial.pts) / initial.frame_duration) < 1e-6);
        assert(samples[i].mapped == 1);
    }
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_STALE);
    pl_queue_plan_destroy(&plan);
    pl_queue_destroy(&queue);
}
static void exact_nearest(void)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue queue = pl_queue_create(&gpu);
    struct sample sample = {0};
    push(queue, &sample, 2.0, .04, PL_FIELD_NONE);
    struct pl_queue_params params = {.pts = 2.0, .vsync_duration = .04, .interpolation_threshold = .001};
    pl_queue_plan plan;
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_READY);
    assert(pl_queue_plan_get_info(plan)->mode == PL_QUEUE_PLAN_NEAREST);
    struct pl_frame_mix mix;
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
    assert(mix.num_frames == 1 && sample.mapped == 1);
    pl_queue_plan_destroy(&plan);
    pl_queue_destroy(&queue);
}
static void differential(float radius, float threshold, float vsync)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue a = pl_queue_create(&gpu), b = pl_queue_create(&gpu);
    struct sample sa[80] = {0}, sb[80] = {0};
    for (int i = 0; i < 80; i++) {
        push(a, &sa[i], 2 + i * .04, .04, PL_FIELD_NONE);
        push(b, &sb[i], 2 + i * .04, .04, PL_FIELD_NONE);
    }
    for (int j = 0; j < 30; j++) {
        struct pl_queue_params params = {.pts = 2.2 + j * vsync, .radius = radius,
            .vsync_duration = vsync, .interpolation_threshold = threshold};
        pl_queue_plan plan;
        assert(pl_queue_plan_inspect(a, &params, &plan) == PL_QUEUE_PLAN_READY);
        const struct pl_queue_plan_info before = *pl_queue_plan_get_info(plan);
        for (int k = 0; k < 10; k++) {
            assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_READY);
            assert(pl_queue_plan_get_info(plan)->policy_revision == before.policy_revision);
        }
        struct pl_frame_mix ma, mb;
        assert(pl_queue_plan_consume(plan, &ma) == PL_QUEUE_PLAN_READY);
        assert(pl_queue_update(b, &mb, &params) == PL_QUEUE_OK);
        assert(ma.num_frames == mb.num_frames);
        assert(fabs(ma.vsync_duration - mb.vsync_duration) < 1e-6);
        for (int k = 0; k < ma.num_frames; k++) {
            assert(ma.signatures[k] == mb.signatures[k]);
            assert(fabs(ma.timestamps[k] - mb.timestamps[k]) < 1e-6);
        }
        assert(pl_queue_num_frames(a) == pl_queue_num_frames(b));
        assert(pl_queue_estimate_vps(a) == pl_queue_estimate_vps(b));
        pl_queue_plan_destroy(&plan);
    }
    pl_queue_destroy(&a);
    pl_queue_destroy(&b);
}
static void interlaced(void)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue queue = pl_queue_create(&gpu);
    struct sample samples[3] = {0};
    push(queue, &samples[0], 0, .04, PL_FIELD_TOP);
    struct pl_queue_params params = {.pts = .02, .vsync_duration = .02, .interpolation_threshold = .001};
    pl_queue_plan plan;
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_PENDING);
    int count;
    const struct pl_queue_plan_sample *selected = pl_queue_plan_get_samples(plan, &count);
    assert(count == 1 && selected[0].field == PL_FIELD_BOTTOM && !selected[0].complete);
    assert(fabs(selected[0].pts - .02) < 1e-8);
    assert(selected[0].signature != selected[0].primary_signature);
    push(queue, &samples[1], .04, .04, PL_FIELD_TOP);
    assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_READY);
    selected = pl_queue_plan_get_samples(plan, &count);
    assert(count == 1 && selected[0].complete && selected[0].has_next);
    const struct pl_queue_plan_sample expected = selected[0];
    selected = &expected;
    struct pl_frame_mix mix;
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
    assert(mix.num_frames == 1 && mix.frames[0]->field == PL_FIELD_BOTTOM);
    assert(mix.frames[0]->next && samples[0].mapped == 1 && samples[1].mapped == 1);
    assert(mix.signatures[0] == selected[0].signature);
    pl_queue_plan_destroy(&plan);
    pl_queue_destroy(&queue);
}
static void invalidation_and_failure(void)
{
    const struct pl_gpu_t gpu = {0};
    for (int test = 0; test < 6; test++) {
        pl_queue queue = pl_queue_create(&gpu);
        struct sample samples[4] = {0};
        push(queue, &samples[0], 2, .04, PL_FIELD_NONE);
        struct pl_queue_params params = {.pts = 2.02, .vsync_duration = .02};
        pl_queue_plan plan;
        assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_PENDING);
        if (test == 0) pl_queue_reset(queue);
        if (test == 1) pl_queue_advance(queue, 2);
        if (test == 2) push(queue, &samples[1], 1.99, .04, PL_FIELD_NONE);
        if (test == 3) push(queue, &samples[1], 2.04, .04, PL_FIELD_TOP);
        if (test < 4) assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_STALE);
        if (test == 4) {
            pl_queue_push(queue, NULL);
            assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_EOF);
        }
        if (test == 5) {
            samples[0].fail = true;
            push(queue, &samples[1], 2.04, .04, PL_FIELD_NONE);
            struct pl_frame_mix mix;
            assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_ERROR);
            assert(!mix.num_frames && samples[0].mapped == 1);
            assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_STALE);
        }
        pl_queue_plan_destroy(&plan);
        pl_queue_destroy(&queue);
    }
}
static void threshold_and_abandon(void)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue queue = pl_queue_create(&gpu);
    struct sample samples[20] = {0};
    for (int i = 0; i < 20; i++) push(queue, &samples[i], 2 + i * .04, .04, PL_FIELD_NONE);
    struct pl_queue_params params = {.pts = 2, .vsync_duration = .04, .interpolation_threshold = .001};
    pl_queue_plan plan;
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_READY);
    struct pl_frame_mix mix;
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY); // installs hold counter
    pl_queue_plan_destroy(&plan);
    params.pts += .044;
    // Inspecting and discarding this proposed next transition does not advance it.
    for (int i = 0; i < 20; i++) {
        assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_READY);
        assert(pl_queue_plan_get_info(plan)->mode == PL_QUEUE_PLAN_NEAREST);
        pl_queue_plan_destroy(&plan);
    }
    for (int i = 0; i < 6; i++) {
        assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_READY);
        assert(pl_queue_plan_get_info(plan)->mode ==
            (i < 5 ? PL_QUEUE_PLAN_NEAREST : PL_QUEUE_PLAN_OVERSAMPLE));
        assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
        pl_queue_plan_destroy(&plan);
        params.pts += .044;
    }
    pl_queue_destroy(&queue);
}
static bool push_block(pl_queue queue, struct sample *sample, double pts)
{
    const struct pl_source_frame src = {.pts = pts, .duration = .04,
        .frame_data = sample, .map = map, .discard = discard, .unmap = unmap};
    return pl_queue_push_block(queue, 1, &src);
}
static void explicit_publication_demand(void)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue queue = pl_queue_create(&gpu);
    struct sample samples[10] = {0};
    assert(push_block(queue, &samples[0], 2));
    assert(push_block(queue, &samples[1], 2.04));
    struct pl_queue_params params = {.pts = 2.12, .radius = 2, .vsync_duration = 1.0f / 60};
    pl_queue_plan plan;
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_PENDING);
    for (int i = 2; i <= 5; i++) {
        assert(!push_block(queue, &samples[i], 2 + i * .04));
        assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_PENDING);
        assert(pl_queue_plan_request(plan) == PL_QUEUE_PLAN_PENDING);
        assert(push_block(queue, &samples[i], 2 + i * .04));
    }
    assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_READY);
    assert(pl_queue_plan_request(plan) == PL_QUEUE_PLAN_READY);
    assert(!push_block(queue, &samples[6], 2.24));
    pl_queue_plan_destroy(&plan);
    // A destroyed pending plan must not leave its publication permission open.
    params.pts = 2.3;
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_PENDING);
    assert(pl_queue_plan_request(plan) == PL_QUEUE_PLAN_PENDING);
    pl_queue_plan_destroy(&plan);
    assert(!push_block(queue, &samples[6], 2.24));
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_PENDING);
    assert(pl_queue_plan_request(plan) == PL_QUEUE_PLAN_PENDING);
    // A competing selection invalidates both the plan and its admission.
    struct pl_frame frame;
    assert(!pl_queue_peek_mapped(queue, 999, &frame));
    assert(pl_queue_plan_request(plan) == PL_QUEUE_PLAN_STALE);
    assert(!push_block(queue, &samples[6], 2.24));
    pl_queue_plan_destroy(&plan);
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_PENDING);
    assert(pl_queue_plan_request(plan) == PL_QUEUE_PLAN_PENDING);
    pl_queue_push(queue, NULL);
    assert(pl_queue_plan_request(plan) == PL_QUEUE_PLAN_EOF);
    assert(pl_queue_num_frames(queue) == 6);
    pl_queue_plan_destroy(&plan);
    pl_queue_destroy(&queue);
}
static void bootstrap_and_adjusted_target(void)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue queue = pl_queue_create(&gpu);
    struct sample samples[3] = {0};
    push(queue, &samples[0], 2, 0, PL_FIELD_NONE);
    struct pl_queue_params params = {.pts = 2.0399, .vsync_duration = .02, .drift_compensation = .001};
    pl_queue_plan plan;
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_BOOTSTRAP);
    assert(!plan);
    push(queue, &samples[1], 2.04, 0, PL_FIELD_NONE);
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_PENDING);
    const struct pl_queue_plan_info *info = pl_queue_plan_get_info(plan);
    assert(info->requested_pts == params.pts && info->pts == 2.04);
    assert(pl_queue_pts_offset(queue) == 0);
    push(queue, &samples[2], 2.08, 0, PL_FIELD_NONE);
    struct pl_frame_mix mix;
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
    assert(fabs(pl_queue_pts_offset(queue) - .0001) < 1e-12);
    pl_queue_plan_destroy(&plan);
    pl_queue_destroy(&queue);
}

static void interlaced_temporal_and_bootstrap(void)
{
    const struct pl_gpu_t gpu = {0};
    for (int radius = 0; radius <= 2; radius++) {
        pl_queue queue = pl_queue_create(&gpu);
        struct sample samples[3] = {0};
        push(queue, &samples[0], 0, .04, PL_FIELD_TOP);
        struct pl_queue_params params = {.pts = .01, .radius = radius, .vsync_duration = .01};
        pl_queue_plan plan;
        assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_PENDING);
        assert(pl_queue_plan_get_info(plan)->mode ==
            (radius ? PL_QUEUE_PLAN_TEMPORAL : PL_QUEUE_PLAN_OVERSAMPLE));
        push(queue, &samples[1], .04, .04, PL_FIELD_TOP);
        if (radius == 2) {
            // The selected .04 first field still lacks its future reference.
            assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_PENDING);
            push(queue, &samples[2], .08, .04, PL_FIELD_TOP);
        }
        assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_READY);
        int count;
        const struct pl_queue_plan_sample *members = pl_queue_plan_get_samples(plan, &count);
        assert(count >= 2);
        struct pl_queue_plan_sample expected[8];
        assert(count <= PL_ARRAY_SIZE(expected));
        memcpy(expected, members, count * sizeof(expected[0]));
        members = expected;
        struct pl_frame_mix mix;
        assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
        assert(mix.num_frames == count);
        for (int i = 0; i < count; i++) {
            assert(members[i].complete && mix.signatures[i] == members[i].signature);
            assert(mix.frames[i]->field == members[i].field);
        }
        pl_queue_plan_destroy(&plan);
        pl_queue_destroy(&queue);
    }
    pl_queue queue = pl_queue_create(&gpu);
    struct sample samples[2] = {0};
    push(queue, &samples[0], 0, 0, PL_FIELD_TOP);
    struct pl_queue_params params = {.pts = .02, .vsync_duration = .02, .interpolation_threshold = .001};
    pl_queue_plan plan;
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_BOOTSTRAP);
    assert(!plan);
    push(queue, &samples[1], .04, 0, PL_FIELD_TOP);
    assert(pl_queue_plan_inspect(queue, &params, &plan) == PL_QUEUE_PLAN_READY);
    int count;
    const struct pl_queue_plan_sample *members = pl_queue_plan_get_samples(plan, &count);
    assert(count == 1 && members[0].pts == .02 && members[0].field == PL_FIELD_BOTTOM);
    pl_queue_plan_destroy(&plan);
    pl_queue_destroy(&queue);
}

static void declared_cadence(void)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue queue = pl_queue_create(&gpu);
    struct sample samples[80] = {0};
    struct pl_queue_params settings = {.radius = 1, .interpolation_threshold = .001};
    struct pl_queue_plan_cadence cadence = {.source_duration = 1.0f / 24, .output_duration = 1.0f / 60};
    pl_queue_plan plan;
    // An empty queue does not erase known cadence or require an ordinary update.
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_PREPARED);
    assert(pl_queue_plan_request(plan) == PL_QUEUE_PLAN_PREPARED);
    struct pl_frame_mix mix;
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_PREPARED);
    const struct pl_queue_plan_info *info = pl_queue_plan_get_info(plan);
    assert(!info->target_set && info->declared_cadence);
    assert(info->mode == PL_QUEUE_PLAN_TEMPORAL);
    assert(fabs(info->support_radius - cadence.source_duration) < 1e-8);
    assert(pl_queue_plan_set_target(plan, 2.2) == PL_QUEUE_PLAN_PENDING);
    assert(pl_queue_plan_set_target(plan, 2.3) == PL_QUEUE_PLAN_ERROR);
    assert(info->pts == 2.2);
    for (int i = 0; i < 80; i++) push(queue, &samples[i], 2 + i / 24.0, 1.0f / 24, PL_FIELD_NONE);
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
    assert(fabs(mix.vsync_duration - .4) < 1e-7);
    assert(pl_queue_estimate_vps(queue) == 0); // no target-delta cadence estimator
    pl_queue_plan_destroy(&plan);
    // Deliberately irregular latency-adjusted targets cannot change the supplied
    // ratio/kernel support. These are not interpreted as display intervals.
    const double targets[] = {2.21, 2.3, 2.301, 2.6};
    for (int i = 0; i < PL_ARRAY_SIZE(targets); i++) {
        assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
        info = pl_queue_plan_get_info(plan);
        assert(info->mode == PL_QUEUE_PLAN_TEMPORAL);
        assert(info->frame_duration == cadence.source_duration);
        assert(info->vsync_duration == cadence.output_duration);
        assert(pl_queue_plan_set_target(plan, targets[i]) == PL_QUEUE_PLAN_READY);
        int count;
        const struct pl_queue_plan_sample *members = pl_queue_plan_get_samples(plan, &count);
        struct pl_queue_plan_sample expected[16];
        assert(count <= PL_ARRAY_SIZE(expected));
        memcpy(expected, members, count * sizeof(expected[0]));
        assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
        assert(mix.num_frames == count);
        for (int j = 0; j < count; j++) assert(mix.signatures[j] == expected[j].signature);
        assert(pl_queue_estimate_vps(queue) == 0);
        pl_queue_plan_destroy(&plan);
    }
    // 60 -> 24 and explicit source-rate conversion preserve dimensional meaning.
    cadence.source_duration = 1.0f / 60;
    for (int rate = 1; rate <= 2; rate++) {
        cadence.output_duration = rate / 24.0f;
        assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
        info = pl_queue_plan_get_info(plan);
        assert(fabs(info->support_radius - cadence.output_duration) < 1e-8);
        assert(info->mode == PL_QUEUE_PLAN_TEMPORAL);
        pl_queue_plan_destroy(&plan);
    }
    pl_queue_destroy(&queue);
}
static void declared_invalid_and_threshold(void)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue queue = pl_queue_create(&gpu);
    struct sample samples[30] = {0};
    for (int i = 0; i < 30; i++) push(queue, &samples[i], 2 + i * .04, .04, PL_FIELD_NONE);
    struct pl_queue_params settings = {.interpolation_threshold = .001};
    struct pl_queue_plan_cadence cadence = {.source_duration = .04, .output_duration = .04};
    pl_queue_plan plan;
    assert(pl_queue_plan_prepare(queue, &settings, NULL, &plan) == PL_QUEUE_PLAN_BOOTSTRAP && !plan);
    cadence.source_duration = 0;
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_BOOTSTRAP && !plan);
    cadence.source_duration = NAN;
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_ERROR && !plan);
    cadence.source_duration = .04;
    cadence.output_duration = -1;
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_ERROR && !plan);
    cadence.output_duration = .04;
    settings.drift_compensation = .001;
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_ERROR && !plan);
    settings.drift_compensation = 0;
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    assert(pl_queue_plan_get_info(plan)->mode == PL_QUEUE_PLAN_NEAREST);
    assert(pl_queue_plan_set_target(plan, 2) == PL_QUEUE_PLAN_READY);
    struct pl_frame_mix mix;
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
    pl_queue_plan_destroy(&plan);
    cadence.output_duration = .044;
    for (int i = 0; i < 6; i++) {
        // Abandoned preparations and pending checks do not spend hysteresis.
        for (int probe = 0; probe < 4; probe++) {
            assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
            assert(pl_queue_plan_get_info(plan)->mode ==
                (i < 5 ? PL_QUEUE_PLAN_NEAREST : PL_QUEUE_PLAN_OVERSAMPLE));
            assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_PREPARED);
            pl_queue_plan_destroy(&plan);
        }
        assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
        assert(pl_queue_plan_set_target(plan, 2.04 + i * .04) == PL_QUEUE_PLAN_READY);
        assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
        pl_queue_plan_destroy(&plan);
    }
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    pl_queue_reset(queue);
    assert(pl_queue_plan_set_target(plan, 0) == PL_QUEUE_PLAN_STALE);
    pl_queue_plan_destroy(&plan);
    pl_queue_destroy(&queue);
}

static void caller_selection_intents(void)
{
    const struct pl_gpu_t gpu = {0};
    pl_queue queue = pl_queue_create(&gpu);
    struct sample samples[8] = {0};
    struct pl_queue_params settings = {0};
    struct pl_queue_plan_cadence cadence = {
        .source_duration = .04, .output_duration = 1.0f / 60,
    };
    push(queue, &samples[0], 1, .04, PL_FIELD_NONE);
    push(queue, &samples[1], 1.04, .04, PL_FIELD_NONE);
    pl_queue_plan plan;
    struct pl_frame_mix mix;
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    assert(pl_queue_plan_get_info(plan)->mode == PL_QUEUE_PLAN_OVERSAMPLE);
    pl_queue_plan_destroy(&plan);
    cadence.intent = PL_QUEUE_PLAN_INTENT_SINGLE;
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    assert(pl_queue_plan_get_info(plan)->mode == PL_QUEUE_PLAN_NEAREST);
    assert(pl_queue_plan_set_target(plan, 1.03) == PL_QUEUE_PLAN_READY);
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
    assert(mix.num_frames == 1 && mix.frames[0]->user_data == &samples[1]);
    pl_queue_plan_destroy(&plan);
    // Explicit image steps declare no physical host refresh interval.
    cadence.output_duration = 0;
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    assert(pl_queue_plan_get_info(plan)->vsync_duration == 0);
    assert(pl_queue_plan_set_target(plan, 1.04) == PL_QUEUE_PLAN_READY);
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY && mix.num_frames == 1);
    pl_queue_plan_destroy(&plan);
    cadence.output_duration = 1.0f / 60;
    cadence.intent = PL_QUEUE_PLAN_INTENT_LAST_DUE;
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    assert(pl_queue_plan_set_target(plan, 9) == PL_QUEUE_PLAN_READY); // no new age gate
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
    assert(mix.num_frames == 1 && mix.frames[0]->user_data == &samples[1]);
    pl_queue_plan_destroy(&plan);
    pl_queue_reset(queue);
    memset(samples, 0, sizeof(samples));
    // Freeze the actual prefix, not a mandatory maximum-depth requirement.
    cadence.intent = PL_QUEUE_PLAN_INTENT_PREFIX;
    cadence.max_entries = 8;
    push(queue, &samples[0], 1, .04, PL_FIELD_NONE);
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    assert(pl_queue_plan_set_target(plan, 1) == PL_QUEUE_PLAN_READY);
    push(queue, &samples[1], 1.04, .04, PL_FIELD_NONE);
    assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_READY);
    int count;
    pl_queue_plan_get_samples(plan, &count);
    assert(count == 1 && samples[0].mapped == 0);
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY && mix.num_frames == 1);
    pl_queue_plan_destroy(&plan);
    // A failure in any frozen member returns no partial mix or queue advance.
    samples[1].fail = true;
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    assert(pl_queue_plan_set_target(plan, 1) == PL_QUEUE_PLAN_READY);
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_ERROR && mix.num_frames == 0);
    assert(pl_queue_num_frames(queue) == 2);
    pl_queue_plan_destroy(&plan);
    pl_queue_reset(queue);
    memset(samples, 0, sizeof(samples));
    // Internal field identities and future deinterlacing references come from
    // the same complete-support resolver used by ordinary prepared plans.
    cadence.source_duration = .02;
    push(queue, &samples[0], 1, .04, PL_FIELD_TOP);
    push(queue, &samples[1], 1.04, .04, PL_FIELD_TOP);
    assert(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    assert(pl_queue_plan_set_target(plan, 1.01) == PL_QUEUE_PLAN_PENDING);
    const struct pl_queue_plan_sample *members = pl_queue_plan_get_samples(plan, &count);
    assert(count == 4 && members[0].field == PL_FIELD_TOP && members[1].field == PL_FIELD_BOTTOM);
    assert(fabs(members[1].pts - 1.02) < 1e-7);
    assert(members[0].signature != members[1].signature);
    assert(members[0].primary_signature == members[1].primary_signature);
    uint64_t expected[4];
    for (int i = 0; i < 4; i++) expected[i] = members[i].signature;
    assert(!members[3].complete && samples[0].mapped == 0);
    push(queue, &samples[2], 1.08, .04, PL_FIELD_TOP);
    assert(pl_queue_plan_check(plan) == PL_QUEUE_PLAN_READY);
    members = pl_queue_plan_get_samples(plan, &count);
    assert(count == 4 && members[3].complete && members[3].has_next);
    for (int i = 0; i < 4; i++) assert(members[i].signature == expected[i]);
    assert(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY && mix.num_frames == 4);
    for (int i = 0; i < 4; i++) assert(mix.signatures[i] == expected[i]);
    assert(mix.frames[3]->next && mix.frames[3]->next->user_data == &samples[2]);
    pl_queue_plan_destroy(&plan);
    pl_queue_destroy(&queue);
}

int main(void)
{
    caller_selection_intents();
    declared_cadence();
    declared_invalid_and_threshold();
    explicit_publication_demand();
    bootstrap_and_adjusted_target();
    pending_publication();
    exact_nearest();
    differential(0, .001, .04);
    differential(0, 0, 1.0f / 60);
    differential(1, 0, 1.0f / 60);
    differential(2, 0, 1.0f / 60);
    interlaced();
    interlaced_temporal_and_bootstrap();
    invalidation_and_failure();
    threshold_and_abandon();
    return 0;
}
