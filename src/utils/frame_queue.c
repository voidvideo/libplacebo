/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <errno.h>
#include <math.h>

#include "common.h"
#include "log.h"
#include "pl_thread.h"

#include <libplacebo/utils/frame_queue.h>

struct cache_entry {
    pl_tex tex[4];
};

struct entry {
    pl_rc_t rc;
    double pts;
    struct cache_entry cache;
    struct pl_source_frame src;
    struct pl_frame frame;
    uint64_t signature;
    bool mapped;
    bool ok;

    // for interlaced frames
    enum pl_field field;
    struct entry *primary;
    struct entry *prev, *next;
    bool defer_second_field;
    bool dirty;
};

// Hard limits for vsync timing validity
#define MIN_FPS 10
#define MAX_FPS 400

// Limits for FPS estimation state
#define MAX_SAMPLES 32
#define MIN_SAMPLES 4

// Stickiness to prevent `interpolation_threshold` oscillation
#define THRESHOLD_MAX_RATIO 0.3
#define THRESHOLD_FRAMES 5

// Maximum number of not-yet-mapped frames to allow queueing in advance
#define PREFETCH_FRAMES 2

struct pool {
    float samples[MAX_SAMPLES];
    float estimate;
    float sum;
    int idx;
    int num;
    int total;
};

struct pl_queue_t {
    pl_gpu gpu;
    pl_log log;

    // For multi-threading, we use two locks. The `lock_weak` guards the queue
    // state itself. The `lock_strong` has a bigger scope and should be held
    // for the duration of any functions that expect the queue state to
    // remain more or less valid (with the exception of adding new members).
    //
    // In particular, `pl_queue_reset` and `pl_queue_update` will take
    // the strong lock, while `pl_queue_push_*` will only take the weak
    // lock.
    pl_mutex lock_strong;
    pl_mutex lock_weak;
    pl_cond wakeup;

    // Frame queue and state
    PL_ARRAY(struct entry *) queue;
    uint64_t signature;
    uint64_t plan_epoch, policy_revision;
    int threshold_frames;
    bool want_frame;
    pl_queue_plan demand_plan;
    bool eof;

    // Average vsync/frame fps estimation state
    struct pool vps, fps;
    float reported_vps;
    float reported_fps;
    double prev_pts;
    double pts_offset;

    // Storage for temporary arrays
    PL_ARRAY(uint64_t) tmp_sig;
    PL_ARRAY(float) tmp_ts;
    PL_ARRAY(const struct pl_frame *) tmp_frame;

    // Queue of GPU objects to reuse
    PL_ARRAY(struct cache_entry) cache;
};

static void clear_plan_demand(pl_queue p)
{
    if (p->demand_plan) {
        p->demand_plan = NULL;
        p->want_frame = false;
    }
}

pl_queue pl_queue_create(pl_gpu gpu)
{
    pl_queue p = pl_alloc_ptr(NULL, p);
    *p = (struct pl_queue_t) {
        .gpu = gpu,
        .log = gpu->log,
    };

    pl_mutex_init(&p->lock_strong);
    pl_mutex_init(&p->lock_weak);
    int ret = pl_cond_init(&p->wakeup);
    if (ret) {
        PL_ERR(p, "Failed to init conditional variable: %d", ret);
        return NULL;
    }
    return p;
}

static void recycle_cache(pl_queue p, struct cache_entry *cache, bool recycle)
{
    bool has_textures = false;
    for (int i = 0; i < PL_ARRAY_SIZE(cache->tex); i++) {
        if (!cache->tex[i])
            continue;

        has_textures = true;
        if (recycle) {
            pl_tex_invalidate(p->gpu, cache->tex[i]);
        } else {
            pl_tex_destroy(p->gpu, &cache->tex[i]);
        }
    }

    if (recycle && has_textures)
        PL_ARRAY_APPEND(p, p->cache, *cache);

    memset(cache, 0, sizeof(*cache)); // sanity
}

static void entry_deref(pl_queue p, struct entry **pentry, bool recycle)
{
    struct entry *entry = *pentry;
    *pentry = NULL;
    if (!entry || !pl_rc_deref(&entry->rc))
        return;

    if (!entry->mapped && entry->src.discard) {
        PL_TRACE(p, "Discarding unused frame id %"PRIu64" with PTS %f",
                 entry->signature, entry->src.pts);
        entry->src.discard(&entry->src);
    }

    if (entry->mapped && entry->ok && entry->src.unmap) {
        PL_TRACE(p, "Unmapping frame id %"PRIu64" with PTS %f",
                 entry->signature, entry->src.pts);
        entry->src.unmap(p->gpu, &entry->frame, &entry->src);
    }

    recycle_cache(p, &entry->cache, recycle);
    pl_free(entry);
}

static struct entry *entry_ref(struct entry *entry)
{
    pl_rc_ref(&entry->rc);
    return entry;
}

static void entry_cull(pl_queue p, struct entry *entry, bool recycle)
{
    // Forcibly clean up references to prev/next frames, even if `entry` has
    // remaining refs pointing at it. This is to prevent cyclic references.
    entry_deref(p, &entry->primary, recycle);
    entry_deref(p, &entry->prev, recycle);
    entry_deref(p, &entry->next, recycle);
    entry_deref(p, &entry, recycle);
}

void pl_queue_destroy(pl_queue *queue)
{
    pl_queue p = *queue;
    if (!p)
        return;

    for (int n = 0; n < p->queue.num; n++)
        entry_cull(p, p->queue.elem[n], false);
    for (int n = 0; n < p->cache.num; n++) {
        for (int i = 0; i < PL_ARRAY_SIZE(p->cache.elem[n].tex); i++)
            pl_tex_destroy(p->gpu, &p->cache.elem[n].tex[i]);
    }

    pl_cond_destroy(&p->wakeup);
    pl_mutex_destroy(&p->lock_weak);
    pl_mutex_destroy(&p->lock_strong);
    pl_free(p);
    *queue = NULL;
}

void pl_queue_reset(pl_queue p)
{
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);

    for (int i = 0; i < p->queue.num; i++)
        entry_cull(p, p->queue.elem[i], false);

    *p = (struct pl_queue_t) {
        .gpu = p->gpu,
        .log = p->log,

        .plan_epoch = p->plan_epoch + 1,
        .policy_revision = p->policy_revision + 1,

        // Reuse lock objects
        .lock_strong = p->lock_strong,
        .lock_weak = p->lock_weak,
        .wakeup = p->wakeup,

        // Explicitly preserve allocations
        .queue.elem = p->queue.elem,
        .tmp_sig.elem = p->tmp_sig.elem,
        .tmp_ts.elem = p->tmp_ts.elem,
        .tmp_frame.elem = p->tmp_frame.elem,

        // Reuse GPU object cache entirely
        .cache = p->cache,
    };

    pl_cond_signal(&p->wakeup);
    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
}

static inline float delta(float old, float new)
{
    return fabsf((new - old) / PL_MIN(new, old));
}

static inline void default_estimate(struct pool *pool, float val)
{
    if (!pool->estimate && isnormal(val) && val > 0.0)
        pool->estimate = val;
}

static inline void update_estimate(struct pool *pool, float cur)
{
    if (pool->num) {
        static const float max_delta = 0.3;
        if (delta(pool->sum / pool->num, cur) > max_delta) {
            pool->sum = 0.0;
            pool->num = pool->idx = 0;
        }
    }

    if (pool->num++ == MAX_SAMPLES) {
        pool->sum -= pool->samples[pool->idx];
        pool->num--;
    }

    pool->sum += pool->samples[pool->idx] = cur;
    pool->idx = (pool->idx + 1) % MAX_SAMPLES;
    pool->total++;

    if (pool->total < MIN_SAMPLES || pool->num >= MIN_SAMPLES)
        pool->estimate = pool->sum / pool->num;
}

static void queue_push(pl_queue p, const struct pl_source_frame *src)
{
    if (p->eof && !src)
        return; // ignore duplicate EOF

    if (p->eof && src) {
        PL_INFO(p, "Received frame after EOF signaled... discarding frame!");
        if (src->discard)
            src->discard(src);
        return;
    }

    pl_cond_signal(&p->wakeup);

    if (!src) {
        PL_TRACE(p, "Received EOF, draining frame queue...");
        p->eof = true;
        p->want_frame = false;
        p->demand_plan = NULL;

        const struct entry *last = p->queue.num ? p->queue.elem[p->queue.num - 1] : NULL;
        if (last && last->defer_second_field) {
            PL_WARN(p, "Received EOF while waiting for second field of "
                    "interlaced frame with PTS %f, discarding last field!",
                    last->pts);
        }
        return;
    }

    // Waiting plans permit ordinary publication, but not a contradictory source
    // order/layout. Native image formats are opaque here; owners reset on change.
    if (p->queue.num) {
        const struct entry *last = p->queue.elem[p->queue.num - 1];
        const struct entry *primary = PL_DEF(last->primary, last);
        if (src->first_field != primary->src.first_field)
            p->plan_epoch++;
    }

    // Update FPS estimates if possible/reasonable
    default_estimate(&p->fps, src->first_field ? src->duration / 2 : src->duration);
    if (p->queue.num) {
        struct entry *last = p->queue.elem[p->queue.num - 1];
        double last_pts = last->pts;
        float delta = src->pts - last_pts;
        if (src->first_field != PL_FIELD_NONE && last->defer_second_field)
            delta /= 2;
        if (delta <= 0.0f) {
            p->plan_epoch++;
            PL_DEBUG(p, "Non monotonically increasing PTS %f -> %f", last_pts, src->pts);
        } else if (p->fps.estimate && delta > 10.0 * p->fps.estimate) {
            p->plan_epoch++;
            PL_DEBUG(p, "Discontinuous source PTS jump %f -> %f", last_pts, src->pts);
        } else {
            update_estimate(&p->fps, delta);
        }
    } else if (src->pts != 0) {
        PL_DEBUG(p, "First frame received with non-zero PTS %f", src->pts);
    }

    struct entry *entry = pl_alloc_ptr(NULL, entry);
    *entry = (struct entry) {
        .signature = p->signature++,
        .pts = src->pts,
        .src = *src,
    };
    pl_rc_init(&entry->rc);
    PL_ARRAY_POP(p->cache, &entry->cache);
    PL_TRACE(p, "Added new frame id %"PRIu64" with PTS %f",
             entry->signature, entry->pts);

    // Insert new entry into the correct spot in the queue, sorted by PTS
    for (int i = p->queue.num;; i--) {
        if (i == 0 || p->queue.elem[i - 1]->pts <= entry->pts) {
            if (src->first_field == PL_FIELD_NONE) {
                // Progressive
                PL_ARRAY_INSERT_AT(p, p->queue, i, entry);
                break;
            } else {
                // Interlaced
                struct entry *prev = i > 0 ? p->queue.elem[i - 1] : NULL;
                struct entry *next = i < p->queue.num ? p->queue.elem[i] : NULL;
                if (prev && prev->defer_second_field) {
                    struct entry *prev2 = pl_zalloc_ptr(NULL, prev2);
                    pl_rc_init(&prev2->rc);
                    prev2->pts = (prev->pts + entry->pts) / 2;
                    prev2->field = pl_field_other(prev->field);
                    prev2->primary = entry_ref(prev);
                    prev2->signature = p->signature++;
                    PL_ARRAY_INSERT_AT(p, p->queue, i, prev2);
                    PL_TRACE(p, "Added deferred second field id %"PRIu64" with "
                             " PTS %f for previous frame with PTS %f",
                             prev2->signature, prev2->pts, prev->pts);

                    prev->defer_second_field = false;
                    prev = prev2;
                    i++;
                }

                struct entry *entry2 = pl_zalloc_ptr(NULL, entry2);
                pl_rc_init(&entry2->rc);
                entry->field = src->first_field;
                if (next) {
                    entry2->pts = (entry->pts + next->pts) / 2;
                } else if (src->duration) {
                    entry2->pts = entry->pts + src->duration / 2;
                } else if (p->fps.estimate) {
                    entry2->pts = entry->pts + p->fps.estimate;
                } else {
                    PL_DEBUG(p, "Frame with PTS %f specified as interlaced, but "
                             "no FPS information known yet! Deferring insertion "
                             "of second field.", src->pts);
                    entry->defer_second_field = true;
                    PL_ARRAY_INSERT_AT(p, p->queue, i, entry);
                    pl_free(entry2);
                    break;
                }

                entry2->field = pl_field_other(entry->field);
                entry2->primary = entry_ref(entry);
                entry2->signature = p->signature++;

                PL_TRACE(p, "Added second field id %"PRIu64" with PTS %f",
                         entry2->signature, entry2->pts);

                // Link previous/next frames
                if (prev) {
                    entry->prev = entry_ref(PL_DEF(prev->primary, prev));
                    entry2->prev = entry_ref(PL_DEF(prev->primary, prev));
                    // Retroactively re-link the previous frames that should
                    // be referencing this frame
                    for (int j = i - 1; j >= 0; --j) {
                        struct entry *e = p->queue.elem[j];
                        if (e != prev && e != prev->primary)
                            break;
                        entry_deref(p, &e->next, true);
                        e->next = entry_ref(entry);
                        if (e->dirty) { // reset signature to signal change
                            e->signature = p->signature++;
                            e->dirty = false;
                        }
                    }
                }

                if (next) {
                    entry->next = entry_ref(PL_DEF(next->primary, next));
                    entry2->next = entry_ref(PL_DEF(next->primary, next));
                    for (int j = i; j < p->queue.num; j++) {
                        struct entry *e = p->queue.elem[j];
                        if (e != next && e != next->primary)
                            break;
                        entry_deref(p, &e->prev, true);
                        e->prev = entry_ref(entry);
                        if (e->dirty) {
                            e->signature = p->signature++;
                            e->dirty = false;
                        }
                    }
                }

                PL_ARRAY_INSERT_AT(p, p->queue, i, entry);
                PL_ARRAY_INSERT_AT(p, p->queue, i+1, entry2);
                break;
            }
        }
    }

    p->want_frame = false;
    p->demand_plan = NULL;
}

void pl_queue_push(pl_queue p, const struct pl_source_frame *frame)
{
    pl_mutex_lock(&p->lock_weak);
    queue_push(p, frame);
    pl_mutex_unlock(&p->lock_weak);
}

static inline bool entry_mapped(struct entry *entry)
{
    return entry->mapped || (entry->primary && entry->primary->mapped);
}

static bool queue_has_room(pl_queue p)
{
    if (p->want_frame)
        return true;

    int wanted_frames = PREFETCH_FRAMES;
    if (p->fps.estimate && p->vps.estimate && p->vps.estimate <= 1.0f / MIN_FPS)
        wanted_frames += ceilf(p->vps.estimate / p->fps.estimate) - 1;

    // Examine the queue tail
    for (int i = p->queue.num - 1; i >= 0; i--) {
        if (entry_mapped(p->queue.elem[i]))
            return true;
        if (p->queue.num - i >= wanted_frames)
            return false;
    }

    return true;
}

bool pl_queue_push_block(pl_queue p, uint64_t timeout,
                         const struct pl_source_frame *frame)
{
    pl_mutex_lock(&p->lock_weak);
    if (!timeout || !frame || p->eof)
        goto skip_blocking;

    while (!queue_has_room(p) && !p->eof) {
        if (pl_cond_timedwait(&p->wakeup, &p->lock_weak, timeout) == ETIMEDOUT) {
            pl_mutex_unlock(&p->lock_weak);
            return false;
        }
    }

skip_blocking:

    queue_push(p, frame);
    pl_mutex_unlock(&p->lock_weak);
    return true;
}

static void report_estimates(pl_queue p)
{
    if (p->fps.total >= MIN_SAMPLES && p->vps.total >= MIN_SAMPLES) {
        if (p->reported_fps && p->reported_vps) {
            // Only re-report the estimates if they've changed considerably
            // from the previously reported values
            static const float report_delta = 0.3f;
            float delta_fps = delta(p->reported_fps, p->fps.estimate);
            float delta_vps = delta(p->reported_vps, p->vps.estimate);
            if (delta_fps < report_delta && delta_vps < report_delta)
                return;
        }

        PL_INFO(p, "Estimated source FPS: %.3f, display FPS: %.3f",
                1.0 / p->fps.estimate, 1.0 / p->vps.estimate);

        p->reported_fps = p->fps.estimate;
        p->reported_vps = p->vps.estimate;
    }
}

// note: may add more than one frame, since it releases the lock
static enum pl_queue_status get_frame(pl_queue p, const struct pl_queue_params *params)
{
    if (p->eof)
        return PL_QUEUE_EOF;

    if (!params->get_frame) {
        if (!params->timeout)
            return PL_QUEUE_MORE;

        p->want_frame = true;
        pl_cond_signal(&p->wakeup);

        while (p->want_frame) {
            if (pl_cond_timedwait(&p->wakeup, &p->lock_weak, params->timeout) == ETIMEDOUT)
                return PL_QUEUE_MORE;
        }

        return p->eof ? PL_QUEUE_EOF : PL_QUEUE_OK;
    }

    // Don't hold the weak mutex while calling into `get_frame`, to allow
    // `pl_queue_push` to run concurrently while we're waiting for frames
    pl_mutex_unlock(&p->lock_weak);

    struct pl_source_frame src;
    enum pl_queue_status ret;
    switch ((ret = params->get_frame(&src, params))) {
    case PL_QUEUE_OK:
        pl_queue_push(p, &src);
        break;
    case PL_QUEUE_EOF:
        pl_queue_push(p, NULL);
        break;
    case PL_QUEUE_MORE:
    case PL_QUEUE_ERR:
        break;
    }

    pl_mutex_lock(&p->lock_weak);
    return ret;
}

static inline bool map_frame(pl_queue p, struct entry *entry)
{
    if (!entry->mapped) {
        PL_TRACE(p, "Mapping frame id %"PRIu64" with PTS %f",
                 entry->signature, entry->pts);
        entry->mapped = true;
        entry->ok = entry->src.map(p->gpu, entry->cache.tex,
                                   &entry->src, &entry->frame);
        if (!entry->ok)
            PL_ERR(p, "Failed mapping frame id %"PRIu64" with PTS %f",
                   entry->signature, entry->pts);
    }

    return entry->ok;
}

static bool map_entry(pl_queue p, struct entry *entry)
{
    bool ok = map_frame(p, entry->primary ? entry->primary : entry);
    if (entry->prev)
        ok &= map_frame(p, entry->prev);
    if (entry->next)
        ok &= map_frame(p, entry->next);
    if (!ok)
        return false;

    if (entry->primary)
        entry->frame = entry->primary->frame;

    if (entry->field) {
        entry->frame.field = entry->field;
        entry->frame.first_field = PL_DEF(entry->primary, entry)->src.first_field;
        entry->frame.prev = entry->prev ? &entry->prev->frame : NULL;
        entry->frame.next = entry->next ? &entry->next->frame : NULL;
        entry->dirty = true;
    }

    return true;
}

static bool entry_complete(pl_queue p, struct entry *entry)
{
    pl_assert(!(entry->next && entry->defer_second_field));
    return entry->field ? (entry->next || p->eof) : true;
}

// Advance the queue as needed to make sure idx 0 is the last frame before
// `pts`, and idx 1 is the first frame after `pts`; exceptions apply if this
// is the first or last frame.
//
// On EOF, continues to return PL_QUEUE_OK until idx 0 is no longer legal under
// ZOH semantics.
static enum pl_queue_status advance(pl_queue p, double pts,
                                    const struct pl_queue_params *params)
{
retry: ;
    // Cull all frames except the last frame before `pts`
    int culled = 0;
    for (int i = 1; i < p->queue.num; i++) {
        if (p->queue.elem[i]->pts <= pts) {
            entry_cull(p, p->queue.elem[i - 1], true);
            culled++;
        }
    }
    PL_ARRAY_REMOVE_RANGE(p->queue, 0, culled);

    // Keep adding new frames until we find one in the future, or EOF
    enum pl_queue_status ret = PL_QUEUE_OK;
    if (!p->queue.num || p->queue.elem[p->queue.num - 1]->pts <= pts) {
        switch ((ret = get_frame(p, params))) {
        case PL_QUEUE_ERR:
        case PL_QUEUE_MORE:
            return ret;
        case PL_QUEUE_EOF:
            if (!p->queue.num)
                return ret;
            goto done;
        case PL_QUEUE_OK:
            goto retry;
        }
    }

    pl_assert(p->queue.num);
    if (!entry_complete(p, p->queue.elem[PL_MIN(p->queue.num - 1, 1)])) {
        switch (get_frame(p, params)) {
        case PL_QUEUE_ERR:
            return PL_QUEUE_ERR;
        case PL_QUEUE_MORE:
            ret = PL_QUEUE_MORE;
            // fall through
        case PL_QUEUE_EOF:
        case PL_QUEUE_OK:
            goto done;
        }
    }

done:
    if (p->eof && p->queue.num == 1) {
        if (p->queue.elem[0]->pts == 0.0 || !p->fps.estimate) {
            // If the last frame has PTS 0.0, or we have no FPS estimate, then
            // this is probably a single-frame file, in which case we want to
            // extend the ZOH to infinity, rather than returning. Not a perfect
            // heuristic, but w/e
            return PL_QUEUE_OK;
        }

        // Last frame is held for an extra `p->fps.estimate` duration,
        // afterwards this function just returns EOF.
        if (pts < p->queue.elem[0]->pts + p->fps.estimate) {
            ret = PL_QUEUE_OK;
        } else {
            entry_cull(p, p->queue.elem[0], true);
            p->queue.num = 0;
            return PL_QUEUE_EOF;
        }
    }

    pl_assert(p->queue.num);
    return ret;
}

static inline enum pl_queue_status point(pl_queue p, struct pl_frame_mix *mix,
                                         const struct pl_queue_params *params)
{
    if (!p->queue.num) {
        *mix = (struct pl_frame_mix) {0};
        return PL_QUEUE_MORE;
    }

    // Find closest frame (nearest neighbour semantics)
    struct entry *entry = p->queue.elem[0];
    if (entry->pts > params->pts) { // first frame not visible yet
        *mix = (struct pl_frame_mix) {0};
        return PL_QUEUE_OK;
    }

    double best = fabs(entry->pts - params->pts);
    for (int i = 1; i < p->queue.num; i++) {
        double dist = fabs(p->queue.elem[i]->pts - params->pts);
        if (dist < best) {
            entry = p->queue.elem[i];
            best = dist;
            continue;
        } else {
            break;
        }
    }

    if (!map_entry(p, entry))
        return PL_QUEUE_ERR;

    // Return a mix containing only this single frame
    p->tmp_sig.num = p->tmp_ts.num = p->tmp_frame.num = 0;
    PL_ARRAY_APPEND(p, p->tmp_sig, entry->signature);
    PL_ARRAY_APPEND(p, p->tmp_frame, &entry->frame);
    PL_ARRAY_APPEND(p, p->tmp_ts, 0.0);
    *mix = (struct pl_frame_mix) {
        .num_frames = 1,
        .frames = p->tmp_frame.elem,
        .signatures = p->tmp_sig.elem,
        .timestamps = p->tmp_ts.elem,
        .vsync_duration = 1.0,
    };

    PL_TRACE(p, "Showing single frame id %"PRIu64" with PTS %f for target PTS %f",
             entry->signature, entry->pts, params->pts);

    report_estimates(p);
    return entry_complete(p, entry) ? PL_QUEUE_OK : PL_QUEUE_MORE;
}

// Present a single frame as appropriate for `pts`
static enum pl_queue_status nearest(pl_queue p, struct pl_frame_mix *mix,
                                    const struct pl_queue_params *params)
{
    enum pl_queue_status ret;
    switch ((ret = advance(p, params->pts, params))) {
    case PL_QUEUE_ERR:
    case PL_QUEUE_EOF:
        return ret;
    case PL_QUEUE_OK:
    case PL_QUEUE_MORE:
        if (mix && point(p, mix, params) == PL_QUEUE_ERR)
            return PL_QUEUE_ERR;
        return ret;
    }

    pl_unreachable();
}

// Special case of `interpolate` for radius = 0, in which case we need exactly
// the previous frame and the following frame
static enum pl_queue_status oversample(pl_queue p, struct pl_frame_mix *mix,
                                       const struct pl_queue_params *params)
{
    enum pl_queue_status ret;
    switch ((ret = advance(p, params->pts, params))) {
    case PL_QUEUE_ERR:
    case PL_QUEUE_EOF:
        return ret;
    case PL_QUEUE_OK:
        break;
    case PL_QUEUE_MORE:
        if (!p->queue.num) {
            if (mix)
                *mix = (struct pl_frame_mix) {0};
            return ret;
        }
        break;
    }

    if (!mix)
        return PL_QUEUE_OK;

    // Can't oversample with only a single frame, fall back to point sampling
    if (p->queue.num < 2 || p->queue.elem[0]->pts > params->pts) {
        if (point(p, mix, params) == PL_QUEUE_ERR)
            return PL_QUEUE_ERR;
        return ret;
    }

    struct entry *entries[2] = { p->queue.elem[0], p->queue.elem[1] };
    pl_assert(entries[0]->pts <= params->pts);
    pl_assert(entries[1]->pts >= params->pts);

    // Returning a mix containing both of these two frames
    p->tmp_sig.num = p->tmp_ts.num = p->tmp_frame.num = 0;
    for (int i = 0; i < 2; i++) {
        if (!map_entry(p, entries[i]))
            return PL_QUEUE_ERR;
        float ts = (entries[i]->pts - params->pts) / p->fps.estimate;
        PL_ARRAY_APPEND(p, p->tmp_sig, entries[i]->signature);
        PL_ARRAY_APPEND(p, p->tmp_frame, &entries[i]->frame);
        PL_ARRAY_APPEND(p, p->tmp_ts, ts);
    }

    *mix = (struct pl_frame_mix) {
        .num_frames = 2,
        .frames = p->tmp_frame.elem,
        .signatures = p->tmp_sig.elem,
        .timestamps = p->tmp_ts.elem,
        .vsync_duration = p->vps.estimate / p->fps.estimate,
    };

    PL_TRACE(p, "Oversampling 2 frames for target PTS %f:", params->pts);
    for (int i = 0; i < mix->num_frames; i++)
        PL_TRACE(p, "    id %"PRIu64" ts %f", mix->signatures[i], mix->timestamps[i]);

    report_estimates(p);
    return ret;
}

// Shared effective-mode transition. Inspection runs this on copied state;
// ordinary update and plan consumption commit exactly one transition.
static enum pl_queue_plan_mode selection_mode(float fps, float vps, int threshold,
    const struct pl_queue_params *params, int *next_threshold)
{
    *next_threshold = threshold;
    if (!fps)
        return PL_QUEUE_PLAN_NEAREST;
    float ratio = fabs(fps / vps - 1.0);
    if (ratio <= params->interpolation_threshold) {
        *next_threshold = THRESHOLD_FRAMES + 1;
        return PL_QUEUE_PLAN_NEAREST;
    } else if (ratio < THRESHOLD_MAX_RATIO && threshold > 1) {
        *next_threshold = threshold - 1;
        return PL_QUEUE_PLAN_NEAREST;
    }
    *next_threshold = 0;
    return params->radius ? PL_QUEUE_PLAN_TEMPORAL : PL_QUEUE_PLAN_OVERSAMPLE;
}

// Present a mixture of frames, relative to the vsync ratio
static enum pl_queue_status interpolate(pl_queue p, struct pl_frame_mix *mix,
                                        const struct pl_queue_params *params)
{
    int next_threshold;
    enum pl_queue_plan_mode mode = selection_mode(p->fps.estimate, p->vps.estimate,
        p->threshold_frames, params, &next_threshold);
    if (p->fps.estimate) {
        float ratio = fabs(p->fps.estimate / p->vps.estimate - 1.0);
        if (ratio <= params->interpolation_threshold && !p->threshold_frames) {
            PL_INFO(p, "Detected fps ratio %.4f below threshold %.4f, "
                    "disabling interpolation", ratio, params->interpolation_threshold);
        } else if (!next_threshold && p->threshold_frames) {
            PL_INFO(p, "Detected fps ratio %.4f exceeds threshold %.4f, "
                    "re-enabling interpolation", ratio, params->interpolation_threshold);
        }
    }
    p->threshold_frames = next_threshold;
    if (mode == PL_QUEUE_PLAN_NEAREST)
        return nearest(p, mix, params);
    if (mode == PL_QUEUE_PLAN_OVERSAMPLE)
        return oversample(p, mix, params);

    pl_assert(p->fps.estimate && p->vps.estimate);
    float radius = params->radius * fmaxf(1.0f, p->vps.estimate / p->fps.estimate);
    double min_pts = params->pts - radius * p->fps.estimate,
           max_pts = params->pts + radius * p->fps.estimate;

    enum pl_queue_status ret;
    switch ((ret = advance(p, min_pts, params))) {
    case PL_QUEUE_ERR:
    case PL_QUEUE_EOF:
        return ret;
    case PL_QUEUE_MORE:
        goto done;
    case PL_QUEUE_OK:
        break;
    }

    pl_assert(p->queue.num);

    // Don't show first frame until it's visible under ZOH semantics
    double first_pts = p->queue.elem[0]->pts;
    if (first_pts > params->pts)
        return nearest(p, mix, params);

    // Keep adding new frames until we've covered the range we care about
    while (p->queue.elem[p->queue.num - 1]->pts < max_pts) {
        switch ((ret = get_frame(p, params))) {
        case PL_QUEUE_ERR:
            return ret;
        case PL_QUEUE_MORE:
            goto done;
        case PL_QUEUE_EOF:;
            // Don't forward EOF until we've held the last frame for the
            // desired ZOH hold duration
            double last_pts = p->queue.elem[p->queue.num - 1]->pts;
            if (last_pts && params->pts >= last_pts + p->fps.estimate)
                return ret;
            ret = PL_QUEUE_OK;
            goto done;
        case PL_QUEUE_OK:
            continue;
        }
    }

    // Ensure the last frame is complete (for interlaced sources)
    int last_idx = p->queue.num - 1;
    while (last_idx && p->queue.elem[last_idx]->pts > max_pts)
        last_idx--;

    if (!entry_complete(p, p->queue.elem[last_idx])) {
        switch ((ret = get_frame(p, params))) {
        case PL_QUEUE_MORE:
        case PL_QUEUE_OK:
            break;
        case PL_QUEUE_ERR:
        case PL_QUEUE_EOF:
            return ret;
        }
    }

done: ;

    if (!mix)
        return PL_QUEUE_OK;

    // Construct a mix object representing the current queue state, starting at
    // the last frame before `min_pts` to make sure there's a fallback frame
    // available for ZOH semantics.
    p->tmp_sig.num = p->tmp_ts.num = p->tmp_frame.num = 0;
    for (int i = 0; i < p->queue.num; i++) {
        struct entry *entry = p->queue.elem[i];
        if (entry->pts > max_pts)
            break;
        if (!map_entry(p, entry))
            return PL_QUEUE_ERR;
        float ts = (entry->pts - params->pts) / p->fps.estimate;
        PL_ARRAY_APPEND(p, p->tmp_sig, entry->signature);
        PL_ARRAY_APPEND(p, p->tmp_frame, &entry->frame);
        PL_ARRAY_APPEND(p, p->tmp_ts, ts);
    }

    *mix = (struct pl_frame_mix) {
        .num_frames = p->tmp_frame.num,
        .frames = p->tmp_frame.elem,
        .signatures = p->tmp_sig.elem,
        .timestamps = p->tmp_ts.elem,
        .vsync_duration = p->vps.estimate / p->fps.estimate,
    };

    PL_TRACE(p, "Showing mix of %d frames for target PTS %f:",
             mix->num_frames, params->pts);
    for (int i = 0; i < mix->num_frames; i++)
        PL_TRACE(p, "    id %"PRIu64" ts %f", mix->signatures[i], mix->timestamps[i]);

    report_estimates(p);
    return ret;
}

static bool prefill(pl_queue p, const struct pl_queue_params *params)
{
    int min_frames = 2 * ceilf(params->radius);
    if (p->fps.estimate && p->vps.estimate && p->vps.estimate <= 1.0f / MIN_FPS)
        min_frames *= ceilf(p->vps.estimate / p->fps.estimate);
    min_frames = PL_MAX(min_frames, PREFETCH_FRAMES);

    while (p->queue.num < min_frames) {
        switch (get_frame(p, params)) {
        case PL_QUEUE_ERR:
            return false;
        case PL_QUEUE_EOF:
        case PL_QUEUE_MORE:
            return true;
        case PL_QUEUE_OK:
            continue;
        }
    }

    // In the most likely case, the first few frames will all be required. So
    // force-map them all to initialize GPU state on initial rendering. This is
    // better than the alternative of missing the cache later, when timing is
    // more relevant.
    for (int i = 0; i < min_frames; i++) {
        if (!map_entry(p, p->queue.elem[i]))
            return false;
    }

    return true;
}

struct selection_timing {
    struct pool vps;
    double pts, pts_offset;
};

// Compute the existing target/vsync transition without changing the queue.
static bool selection_timing(pl_queue p, const struct pl_queue_params *params,
                             struct selection_timing *out)
{
    *out = (struct selection_timing) {
        .vps = p->vps, .pts = params->pts, .pts_offset = p->pts_offset,
    };
    default_estimate(&out->vps, params->vsync_duration);
    float delta = params->pts - p->prev_pts;
    if (delta < 0.0f) {
        if (p->queue.num && p->queue.elem[0]->pts > params->pts)
            return false;
    } else if (delta > 1.0f) {
        out->pts_offset = 0.0;
    } else if (delta > 0) {
        update_estimate(&out->vps, params->pts - p->prev_pts);
    }
    if (params->drift_compensation > 0.0f) {
        out->pts += out->pts_offset;
        for (int i = 0; i < p->queue.num; i++) {
            if (fabs(p->queue.elem[i]->pts - out->pts) < params->drift_compensation) {
                out->pts_offset = p->queue.elem[i]->pts - params->pts;
                out->pts = p->queue.elem[i]->pts;
                break;
            }
        }
    }
    return true;
}

enum pl_queue_status pl_queue_update(pl_queue p, struct pl_frame_mix *out_mix,
                                     const struct pl_queue_params *params)
{
    struct pl_queue_params fixed;
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);
    p->policy_revision++;
    clear_plan_demand(p);
    struct selection_timing timing;
    if (!selection_timing(p, params, &timing)) {
        PL_ERR(p, "Requested PTS %f is lower than the oldest frame "
               "PTS %f. This is not supported, PTS must be monotonically "
               "increasing! Please use `pl_queue_reset` to reset the frame "
               "queue on discontinuous PTS jumps.", params->pts, p->queue.elem[0]->pts);
        pl_mutex_unlock(&p->lock_weak);
        pl_mutex_unlock(&p->lock_strong);
        return PL_QUEUE_ERR;
    }
    if (params->pts - p->prev_pts > 1.0)
        PL_TRACE(p, "Discontinuous target PTS jump %f -> %f, ignoring...",
                 p->prev_pts, params->pts);
    p->vps = timing.vps;
    p->prev_pts = params->pts;
    p->pts_offset = timing.pts_offset;
    fixed = *params;
    fixed.pts = timing.pts;
    params = &fixed;

    // As a special case, prefill the queue if this is the first frame
    if (!params->pts && !p->queue.num) {
        if (!prefill(p, params)) {
            pl_mutex_unlock(&p->lock_weak);
            pl_mutex_unlock(&p->lock_strong);
            return PL_QUEUE_ERR;
        }
    }

    // Ignore unrealistically high or low FPS, common near start of playback
    static const float max_vsync = 1.0 / MIN_FPS;
    static const float min_vsync = 1.0 / MAX_FPS;
    bool estimation_ok = p->vps.estimate > min_vsync && p->vps.estimate < max_vsync;
    enum pl_queue_status ret;

    if (estimation_ok || params->vsync_duration > 0) {
        // We know the vsync duration, so construct an interpolation mix
        ret = interpolate(p, out_mix, params);
    } else {
        // We don't know the vsync duration (yet), so just point-sample
        ret = nearest(p, out_mix, params);
    }

    if (ret == PL_QUEUE_ERR)
        PL_ERR(p, "Failed updating frame queue for target PTS %f!", params->pts);

    pl_cond_signal(&p->wakeup);
    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
    return ret;
}

float pl_queue_estimate_fps(pl_queue p)
{
    pl_mutex_lock(&p->lock_weak);
    float estimate = p->fps.estimate;
    pl_mutex_unlock(&p->lock_weak);
    return estimate ? 1.0f / estimate : 0.0f;
}

float pl_queue_estimate_vps(pl_queue p)
{
    pl_mutex_lock(&p->lock_weak);
    float estimate = p->vps.estimate;
    pl_mutex_unlock(&p->lock_weak);
    return estimate ? 1.0f / estimate : 0.0f;
}

int pl_queue_num_frames(pl_queue p)
{
    pl_mutex_lock(&p->lock_weak);
    int count = p->queue.num;
    pl_mutex_unlock(&p->lock_weak);
    return count;
}

double pl_queue_pts_offset(pl_queue p)
{
    pl_mutex_lock(&p->lock_weak);
    double offset = p->pts_offset;
    pl_mutex_unlock(&p->lock_weak);
    return offset;
}

bool pl_queue_peek(pl_queue p, int idx, struct pl_source_frame *out)
{
    pl_mutex_lock(&p->lock_weak);
    bool ok = idx >= 0 && idx < p->queue.num;
    if (ok)
        *out = p->queue.elem[idx]->src;
    pl_mutex_unlock(&p->lock_weak);
    return ok;
}

bool pl_queue_peek_mapped(pl_queue p, int idx, struct pl_frame *out)
{
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);
    p->policy_revision++;
    clear_plan_demand(p);

    bool ok = false;
    if (idx >= 0 && idx < p->queue.num) {
        struct entry *entry = p->queue.elem[idx];
        if (map_entry(p, entry)) {
            *out = entry->frame;
            ok = true;
        }
    }

    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
    return ok;
}

void pl_queue_advance(pl_queue p, double pts)
{
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);
    p->policy_revision++;
    clear_plan_demand(p);

    int culled = 0;
    for (int i = 1; i < p->queue.num; i++) {
        if (p->queue.elem[i]->pts <= pts) {
            entry_cull(p, p->queue.elem[i - 1], true);
            culled++;
        }
    }
    PL_ARRAY_REMOVE_RANGE(p->queue, 0, culled);

    // pl_queue_push_block waits for this condition whenever the prefetch
    // window is full. Custom cadence drivers use pl_queue_advance instead of
    // pl_queue_update, so advancing must provide the same producer wakeup as
    // the normal update path after it frees queue capacity.
    if (culled)
        pl_cond_signal(&p->wakeup);

    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
}

struct pl_queue_plan_t {
    pl_queue queue;
    struct pl_queue_plan_info info;
    struct selection_timing timing;
    int next_threshold;
    enum pl_queue_plan_intent intent;
    int max_entries;
    PL_ARRAY(uint64_t) members;
    bool consumed;
    int first_index, last_index;
    PL_ARRAY(struct pl_queue_plan_sample) samples;
};

static enum pl_queue_plan_status plan_missing(pl_queue p)
{
    return p->eof ? PL_QUEUE_PLAN_EOF : PL_QUEUE_PLAN_PENDING;
}

// Both check and consume use this exact resolver while holding the queue locks.
// It never evaluates timing/mode policy, changes a target, maps, or retires data.
static enum pl_queue_plan_status plan_check_locked(pl_queue_plan plan)
{
    pl_queue p = plan->queue;
    const struct pl_queue_plan_info *info = &plan->info;
    plan->samples.num = 0;
    if (plan->consumed || info->epoch != p->plan_epoch ||
        info->policy_revision != p->policy_revision)
        return PL_QUEUE_PLAN_STALE;
    if (!info->target_set)
        return PL_QUEUE_PLAN_PREPARED;
    if (!p->queue.num)
        return plan_missing(p);

    double start = info->mode == PL_QUEUE_PLAN_TEMPORAL ? info->min_pts : info->pts;
    int first = 0;
    while (first + 1 < p->queue.num && p->queue.elem[first + 1]->pts <= start)
        first++;
    // Preserve initial visibility. A future first sample cannot be presented.
    if (p->queue.elem[first]->pts > info->pts)
        return plan_missing(p);

    int last = first;
    bool complete = true;
    if (info->mode == PL_QUEUE_PLAN_LAST_DUE || info->mode == PL_QUEUE_PLAN_PREFIX) {
        if (!plan->members.num) {
            last = info->mode == PL_QUEUE_PLAN_PREFIX ?
                first + PL_MIN(plan->max_entries, p->queue.num - first) - 1 : first;
            for (int i = first; i <= last; i++)
                PL_ARRAY_APPEND(plan, plan->members, p->queue.elem[i]->signature);
        } else {
            last = first + plan->members.num - 1;
            if (last >= p->queue.num)
                return PL_QUEUE_PLAN_STALE;
            for (int i = first; i <= last; i++) {
                if (p->queue.elem[i]->signature != plan->members.elem[i - first])
                    return PL_QUEUE_PLAN_STALE;
            }
        }
    } else if (info->mode == PL_QUEUE_PLAN_TEMPORAL) {
        while (last + 1 < p->queue.num && p->queue.elem[last + 1]->pts <= info->max_pts)
            last++;
        complete = p->queue.elem[p->queue.num - 1]->pts >= info->max_pts;
    } else if (info->mode == PL_QUEUE_PLAN_OVERSAMPLE) {
        if (last + 1 < p->queue.num)
            last++;
        else
            complete = false;
    } else {
        if (first + 1 < p->queue.num) {
            if (fabs(p->queue.elem[first + 1]->pts - info->pts) <
                fabs(p->queue.elem[first]->pts - info->pts))
                first++;
            last = first;
        } else {
            // Nearest's terminal hold selects the actual last sample. It is
            // valid only for the existing ZOH lifetime, not an added frame.
            complete = p->queue.elem[first]->pts == info->pts ||
                (p->eof && (p->queue.elem[first]->pts == 0.0 ||
                 info->pts < p->queue.elem[first]->pts + info->frame_duration));
        }
    }

    plan->first_index = first;
    plan->last_index = last;
    for (int i = first; i <= last; i++) {
        const struct entry *entry = p->queue.elem[i];
        struct pl_queue_plan_sample sample = {
            .signature = entry->signature,
            .pts = entry->pts,
            .field = entry->field,
            .primary_signature = PL_DEF(entry->primary, entry)->signature,
            .has_prev = !!entry->prev,
            .has_next = !!entry->next,
            .prev_signature = entry->prev ? entry->prev->signature : 0,
            .next_signature = entry->next ? entry->next->signature : 0,
            .complete = entry_complete(p, p->queue.elem[i]),
        };
        complete &= sample.complete;
        PL_ARRAY_APPEND(plan, plan->samples, sample);
    }
    return complete ? PL_QUEUE_PLAN_READY : plan_missing(p);
}

enum pl_queue_plan_status pl_queue_plan_inspect(pl_queue p,
    const struct pl_queue_params *params, pl_queue_plan *out_plan)
{
    if (!out_plan)
        return PL_QUEUE_PLAN_ERROR;
    *out_plan = NULL;
    if (!p || !params || !isfinite(params->pts) || !isfinite(params->radius) ||
        params->radius < 0 || !isfinite(params->vsync_duration) || params->vsync_duration < 0 ||
        !isfinite(params->drift_compensation) || params->drift_compensation < 0 ||
        !isfinite(params->interpolation_threshold) || params->interpolation_threshold < 0)
        return PL_QUEUE_PLAN_ERROR;
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);
    enum pl_queue_plan_status status;
    struct selection_timing timing;
    if (!p->queue.num) {
        status = p->eof ? PL_QUEUE_PLAN_EOF : PL_QUEUE_PLAN_EMPTY;
        goto done;
    }
    if (!selection_timing(p, params, &timing)) {
        status = PL_QUEUE_PLAN_ERROR;
        goto done;
    }
    // Pushes provide FPS evidence; explicit vsync duration supplies initial VPS.
    // No ordinary update or successful render is required for bootstrapping.
    if (!isnormal(p->fps.estimate) || p->fps.estimate <= 0 ||
        !isnormal(timing.vps.estimate) || timing.vps.estimate <= 0) {
        status = p->eof ? PL_QUEUE_PLAN_EOF : PL_QUEUE_PLAN_BOOTSTRAP;
        goto done;
    }
    pl_queue_plan plan = pl_zalloc_ptr(NULL, plan);
    plan->queue = p;
    plan->timing = timing;
    plan->next_threshold = p->threshold_frames;
    bool estimation_ok = timing.vps.estimate > 1.0f / MAX_FPS &&
                         timing.vps.estimate < 1.0f / MIN_FPS;
    enum pl_queue_plan_mode mode = PL_QUEUE_PLAN_NEAREST;
    if (estimation_ok || params->vsync_duration > 0)
        mode = selection_mode(p->fps.estimate, timing.vps.estimate,
                              p->threshold_frames, params, &plan->next_threshold);
    float radius = params->radius * fmaxf(1.0f, timing.vps.estimate / p->fps.estimate);
    plan->info = (struct pl_queue_plan_info) {
        .epoch = p->plan_epoch, .policy_revision = p->policy_revision,
        .mode = mode, .target_set = true,
        .support_radius = mode == PL_QUEUE_PLAN_TEMPORAL ? radius * p->fps.estimate : 0,
        .requested_pts = params->pts, .pts = timing.pts,
        .frame_duration = p->fps.estimate, .vsync_duration = timing.vps.estimate,
        .min_pts = timing.pts - radius * p->fps.estimate,
        .max_pts = timing.pts + radius * p->fps.estimate,
    };
    if (!isfinite(plan->info.min_pts) || !isfinite(plan->info.max_pts)) {
        pl_free(plan);
        status = PL_QUEUE_PLAN_ERROR;
        goto done;
    }
    status = plan_check_locked(plan);
    *out_plan = plan;
done:
    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
    return status;
}

enum pl_queue_plan_status pl_queue_plan_prepare(pl_queue p,
    const struct pl_queue_params *settings, const struct pl_queue_plan_cadence *cadence,
    pl_queue_plan *out_plan)
{
    if (!out_plan)
        return PL_QUEUE_PLAN_ERROR;
    *out_plan = NULL;
    if (!p || !settings || !isfinite(settings->radius) || settings->radius < 0 ||
        !isfinite(settings->interpolation_threshold) || settings->interpolation_threshold < 0 ||
        settings->drift_compensation != 0)
        return PL_QUEUE_PLAN_ERROR;
    if (!cadence)
        return PL_QUEUE_PLAN_BOOTSTRAP;
    if (!isfinite(cadence->source_duration) || cadence->source_duration < 0 ||
        !isfinite(cadence->output_duration) || cadence->output_duration < 0)
        return PL_QUEUE_PLAN_ERROR;
    if (cadence->intent < PL_QUEUE_PLAN_INTENT_DEFAULT || cadence->intent > PL_QUEUE_PLAN_INTENT_PREFIX ||
        (cadence->intent == PL_QUEUE_PLAN_INTENT_PREFIX && cadence->max_entries <= 0))
        return PL_QUEUE_PLAN_ERROR;
    if (!cadence->source_duration ||
        (!cadence->output_duration && cadence->intent != PL_QUEUE_PLAN_INTENT_SINGLE))
        return PL_QUEUE_PLAN_BOOTSTRAP;

    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);
    pl_queue_plan plan = pl_zalloc_ptr(NULL, plan);
    plan->queue = p;
    plan->intent = cadence->intent;
    plan->max_entries = cadence->max_entries;
    plan->next_threshold = p->threshold_frames;
    enum pl_queue_plan_mode mode = PL_QUEUE_PLAN_NEAREST;
    switch (cadence->intent) {
    case PL_QUEUE_PLAN_INTENT_DEFAULT:
        mode = selection_mode(cadence->source_duration, cadence->output_duration,
            p->threshold_frames, settings, &plan->next_threshold);
        break;
    case PL_QUEUE_PLAN_INTENT_SINGLE: break;
    case PL_QUEUE_PLAN_INTENT_LAST_DUE: mode = PL_QUEUE_PLAN_LAST_DUE; break;
    case PL_QUEUE_PLAN_INTENT_PREFIX: mode = PL_QUEUE_PLAN_PREFIX; break;
    }
    float normalized_radius = settings->radius *
        fmaxf(1.0f, cadence->output_duration / cadence->source_duration);
    double radius = mode == PL_QUEUE_PLAN_TEMPORAL ?
        normalized_radius * cadence->source_duration : 0;
    plan->info = (struct pl_queue_plan_info) {
        .epoch = p->plan_epoch, .policy_revision = p->policy_revision,
        .mode = mode, .declared_cadence = true, .support_radius = radius,
        .frame_duration = cadence->source_duration,
        .vsync_duration = cadence->output_duration,
    };
    enum pl_queue_plan_status status = PL_QUEUE_PLAN_PREPARED;
    if (!isfinite(radius)) {
        status = PL_QUEUE_PLAN_ERROR;
        pl_free(plan);
    } else {
        *out_plan = plan;
    }
    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
    return status;
}

enum pl_queue_plan_status pl_queue_plan_set_target(pl_queue_plan plan, double pts)
{
    if (!plan || !isfinite(pts))
        return PL_QUEUE_PLAN_ERROR;
    pl_queue p = plan->queue;
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);
    enum pl_queue_plan_status status = PL_QUEUE_PLAN_ERROR;
    if (plan->consumed || plan->info.epoch != p->plan_epoch ||
        plan->info.policy_revision != p->policy_revision) {
        status = PL_QUEUE_PLAN_STALE;
    } else if (!plan->info.target_set) {
        double min_pts = pts - plan->info.support_radius;
        double max_pts = pts + plan->info.support_radius;
        if (isfinite(min_pts) && isfinite(max_pts)) {
            plan->info.target_set = true;
            plan->info.requested_pts = plan->info.pts = pts;
            plan->info.min_pts = min_pts;
            plan->info.max_pts = max_pts;
            status = plan_check_locked(plan);
        }
    }
    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
    return status;
}

const struct pl_queue_plan_info *pl_queue_plan_get_info(pl_queue_plan plan)
{
    return plan ? &plan->info : NULL;
}

enum pl_queue_plan_status pl_queue_plan_check(pl_queue_plan plan)
{
    if (!plan)
        return PL_QUEUE_PLAN_ERROR;
    pl_queue p = plan->queue;
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);
    enum pl_queue_plan_status status = plan_check_locked(plan);
    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
    return status;
}

enum pl_queue_plan_status pl_queue_plan_request(pl_queue_plan plan)
{
    if (!plan)
        return PL_QUEUE_PLAN_ERROR;
    pl_queue p = plan->queue;
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);
    enum pl_queue_plan_status status = plan_check_locked(plan);
    if (status == PL_QUEUE_PLAN_PENDING) {
        p->demand_plan = plan;
        p->want_frame = true;
        pl_cond_signal(&p->wakeup);
    } else if (p->demand_plan == plan) {
        clear_plan_demand(p);
    }
    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
    return status;
}

const struct pl_queue_plan_sample *pl_queue_plan_get_samples(pl_queue_plan plan, int *count)
{
    *count = plan ? plan->samples.num : 0;
    return plan ? plan->samples.elem : NULL;
}

enum pl_queue_plan_status pl_queue_plan_consume(pl_queue_plan plan, struct pl_frame_mix *mix)
{
    if (!plan || !mix)
        return PL_QUEUE_PLAN_ERROR;
    *mix = (struct pl_frame_mix) {0};
    pl_queue p = plan->queue;
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);
    enum pl_queue_plan_status status = plan_check_locked(plan);
    if (status != PL_QUEUE_PLAN_READY)
        goto done;

    // From here this is a single terminal selection attempt, including failure.
    plan->consumed = true;
    p->policy_revision++;
    clear_plan_demand(p);
    p->tmp_sig.num = p->tmp_ts.num = p->tmp_frame.num = 0;
    for (int i = plan->first_index; i <= plan->last_index; i++) {
        struct entry *entry = p->queue.elem[i];
        if (!map_entry(p, entry)) {
            status = PL_QUEUE_PLAN_ERROR;
            goto done;
        }
        PL_ARRAY_APPEND(p, p->tmp_sig, entry->signature);
        PL_ARRAY_APPEND(p, p->tmp_frame, &entry->frame);
        float ts = (plan->info.mode == PL_QUEUE_PLAN_NEAREST ||
                    plan->info.mode == PL_QUEUE_PLAN_LAST_DUE) ? 0.0f :
            (entry->pts - plan->info.pts) / plan->info.frame_duration;
        PL_ARRAY_APPEND(p, p->tmp_ts, ts);
    }
    if (!plan->info.declared_cadence) {
        p->vps = plan->timing.vps;
        p->pts_offset = plan->timing.pts_offset;
    }
    p->prev_pts = plan->info.requested_pts;
    if (plan->intent == PL_QUEUE_PLAN_INTENT_DEFAULT)
        p->threshold_frames = plan->next_threshold;
    // Nearest can select the future bracket: preserve the preceding entry as
    // ordinary advance does, so the following target has identical ownership.
    double start = plan->info.mode == PL_QUEUE_PLAN_TEMPORAL ? plan->info.min_pts : plan->info.pts;
    int culled = 0;
    while (culled + 1 < p->queue.num && p->queue.elem[culled + 1]->pts <= start) {
        entry_cull(p, p->queue.elem[culled], true);
        culled++;
    }
    PL_ARRAY_REMOVE_RANGE(p->queue, 0, culled);
    *mix = (struct pl_frame_mix) {
        .num_frames = p->tmp_frame.num,
        .frames = p->tmp_frame.elem,
        .signatures = p->tmp_sig.elem,
        .timestamps = p->tmp_ts.elem,
        .vsync_duration = plan->info.mode == PL_QUEUE_PLAN_NEAREST ? 1.0f :
            plan->info.vsync_duration / plan->info.frame_duration,
    };
    report_estimates(p);
done:
    pl_cond_signal(&p->wakeup);
    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
    return status;
}

void pl_queue_plan_destroy(pl_queue_plan *plan)
{
    if (plan && *plan) {
        pl_queue p = (*plan)->queue;
        pl_mutex_lock(&p->lock_strong);
        pl_mutex_lock(&p->lock_weak);
        if (p->demand_plan == *plan)
            clear_plan_demand(p);
        pl_mutex_unlock(&p->lock_weak);
        pl_mutex_unlock(&p->lock_strong);
        pl_free(*plan);
        *plan = NULL;
    }
}
