/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 2.1, or (at your option) any later version.
 */

#include "gpu.h"
#include "pl_thread.h"

// Bounds queued work and unclaimed results alike. Admission never waits for a
// slot: callers retire obsolete requests and explicitly retry rejected work.
#define PREPARE_CAPACITY 64

struct pl_prepared_pass_t {
    pl_gpu gpu;
    struct pl_pass_params params;
    pl_pass pass;
};

struct pl_pass_preparation_t {
    struct pl_pass_prepare_service *service;
    struct pl_pass_preparation_t *next;
    pl_prepared_pass prepared;
    enum pl_pass_prepare_state state;
    enum pl_pass_prepare_phase phase;
    bool released;
};

struct pl_pass_prepare_service {
    pl_gpu gpu;
    pl_mutex lock;
    pl_cond wake;
    pl_thread worker;
    pl_pass_preparation requests;
    unsigned count;
    bool stopping;
};

static void unlink_request(struct pl_pass_prepare_service *s,
                           pl_pass_preparation r)
{
    pl_pass_preparation *link = &s->requests;
    while (*link != r)
        link = &(*link)->next;
    *link = r->next;
    s->count--;
}

static PL_THREAD_VOID prepare_worker(void *arg)
{
    struct pl_pass_prepare_service *s = arg;
    const struct pl_gpu_fns *impl = PL_PRIV(s->gpu);
    pl_mutex_lock(&s->lock);
    for (;;) {
        pl_pass_preparation r;
        for (r = s->requests; r; r = r->next) {
            if (s->stopping) {
                r->state = PL_PASS_PREPARE_CANCELLED;
                r->released = true;
            }
            if (r->state == PL_PASS_PREPARE_PENDING || r->released ||
                (r->state == PL_PASS_PREPARE_CANCELLED && r->prepared))
                break;
        }

        if (!r) {
            if (s->stopping)
                break;
            pl_cond_wait(&s->wake, &s->lock);
            continue;
        }

        if (r->state != PL_PASS_PREPARE_PENDING) {
            pl_prepared_pass prepared = r->prepared;
            r->prepared = NULL;
            bool release = r->released;
            if (release)
                unlink_request(s, r);
            pl_mutex_unlock(&s->lock);
            // All abandoned GPU results are destroyed by this worker, never by
            // cancellation or public handle release on the rendering thread.
            pl_prepared_pass_destroy(&prepared);
            if (release)
                pl_free(r);
            pl_mutex_lock(&s->lock);
            continue;
        }

        pl_prepared_pass prepared = r->prepared;
        pl_mutex_unlock(&s->lock);
        enum pl_pass_prepare_phase phase = PL_PASS_PREPARE_PHASE_RESOURCES;
        // Neither the service lock nor a dispatch/recording/cache lock is held
        // here. The Vulkan callback includes translation and complete pipeline
        // creation, using only the job's immutable, independently owned inputs.
        prepared->pass = impl->pass_create_prepared(s->gpu, &prepared->params,
                                                   &phase);
        pl_mutex_lock(&s->lock);
        if (r->state == PL_PASS_PREPARE_PENDING) {
            r->state = prepared->pass ? PL_PASS_PREPARE_READY : PL_PASS_PREPARE_FAILED;
            r->phase = prepared->pass ? PL_PASS_PREPARE_PHASE_NONE : phase;
        }
        // Cancellation may have won while the driver was running. The next
        // iteration retires that result without publishing it as ready.
    }
    pl_mutex_unlock(&s->lock);
    PL_THREAD_RETURN();
}

bool pl_pass_prepare_supported(pl_gpu gpu)
{
    if (!gpu)
        return false;
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    return gpu->limits.thread_safe && impl->pass_create_prepared &&
           impl->pass_run_prepared;
}

void pl_pass_prepare_init(pl_gpu gpu)
{
    if (!pl_pass_prepare_supported(gpu))
        return;
    struct pl_gpu_fns *impl = PL_PRIV(gpu);
    if (impl->prepare)
        return;
    struct pl_pass_prepare_service *s = pl_zalloc_ptr(NULL, s);
    s->gpu = gpu;
    pl_mutex_init(&s->lock);
    if (pl_cond_init(&s->wake)) {
        pl_mutex_destroy(&s->lock);
        pl_free(s);
        return;
    }
    if (pl_thread_create(&s->worker, prepare_worker, s)) {
        pl_cond_destroy(&s->wake);
        pl_mutex_destroy(&s->lock);
        pl_free(s);
        return;
    }
    impl->prepare = s;
}

void pl_pass_prepare_uninit(pl_gpu gpu)
{
    struct pl_gpu_fns *impl = PL_PRIV(gpu);
    struct pl_pass_prepare_service *s = impl->prepare;
    if (!s)
        return;
    pl_mutex_lock(&s->lock);
    s->stopping = true;
    pl_cond_signal(&s->wake);
    pl_mutex_unlock(&s->lock);
    // This is the only worker join. The caller is explicitly destroying the
    // GPU; requests/cache/backend resources are still alive until it finishes.
    pl_thread_join(s->worker);
    pl_cond_destroy(&s->wake);
    pl_mutex_destroy(&s->lock);
    pl_free(s);
    impl->prepare = NULL;
}

enum pl_pass_prepare_result pl_pass_prepare_submit(
    pl_gpu gpu, const struct pl_pass_params *params, pl_pass_preparation *out)
{
    if (!out)
        return PL_PASS_PREPARE_INVALID;
    *out = NULL;
    if (!gpu || !pl_pass_params_valid(gpu, params))
        return PL_PASS_PREPARE_INVALID;
    if (!pl_pass_prepare_supported(gpu))
        return PL_PASS_PREPARE_UNSUPPORTED;
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    struct pl_pass_prepare_service *s = impl->prepare;
    if (!s)
        return PL_PASS_PREPARE_UNAVAILABLE;

    pl_mutex_lock(&s->lock);
    if (s->stopping || s->count >= PREPARE_CAPACITY) {
        enum pl_pass_prepare_result result = s->stopping
            ? PL_PASS_PREPARE_UNAVAILABLE : PL_PASS_PREPARE_CAPACITY;
        pl_mutex_unlock(&s->lock);
        return result;
    }
    s->count++;
    pl_mutex_unlock(&s->lock);

    // Copy outside the publication lock. Parent allocations are independent
    // roots, never part of an active renderer or GPU allocation tree.
    pl_pass_preparation r = pl_zalloc_ptr(NULL, r);
    pl_prepared_pass prepared = pl_zalloc_ptr(NULL, prepared);
    prepared->gpu = gpu;
    prepared->params = pl_pass_params_copy(prepared, params);
    if (params->constant_data && params->num_constants) {
        size_t size = 0;
        for (int i = 0; i < params->num_constants; i++) {
            const struct pl_constant *c = &params->constants[i];
            size = PL_MAX(size, c->offset + pl_var_type_size(c->type));
        }
        prepared->params.constant_data = pl_memdup(prepared, params->constant_data, size);
    }
    r->service = s;
    r->prepared = prepared;
    r->state = PL_PASS_PREPARE_PENDING;

    pl_mutex_lock(&s->lock);
    pl_pass_preparation *tail = &s->requests;
    while (*tail)
        tail = &(*tail)->next;
    *tail = r;
    *out = r;
    pl_cond_signal(&s->wake);
    pl_mutex_unlock(&s->lock);
    return PL_PASS_PREPARE_ACCEPTED;
}

enum pl_pass_prepare_state pl_pass_prepare_poll(pl_pass_preparation r)
{
    struct pl_pass_prepare_service *s = r->service;
    pl_mutex_lock(&s->lock);
    enum pl_pass_prepare_state state = r->state;
    pl_mutex_unlock(&s->lock);
    return state;
}

enum pl_pass_prepare_phase pl_pass_prepare_failure_phase(pl_pass_preparation r)
{
    struct pl_pass_prepare_service *s = r->service;
    pl_mutex_lock(&s->lock);
    enum pl_pass_prepare_phase phase = r->state == PL_PASS_PREPARE_FAILED
        ? r->phase : PL_PASS_PREPARE_PHASE_NONE;
    pl_mutex_unlock(&s->lock);
    return phase;
}

const char *pl_pass_prepare_error(pl_pass_preparation r)
{
    switch (pl_pass_prepare_failure_phase(r)) {
    case PL_PASS_PREPARE_PHASE_RESOURCES:
        return "Pass resource creation failed; see GPU log";
    case PL_PASS_PREPARE_PHASE_TRANSLATION:
        return "GLSL to SPIR-V translation failed; see GPU log";
    case PL_PASS_PREPARE_PHASE_PIPELINE:
        return "Shader module or pipeline creation failed; see GPU log";
    case PL_PASS_PREPARE_PHASE_NONE:
        return NULL;
    }
    pl_unreachable();
}

enum pl_pass_prepare_state pl_pass_prepare_take(
    pl_pass_preparation *request, pl_prepared_pass *out)
{
    pl_pass_preparation r = *request;
    struct pl_pass_prepare_service *s = r->service;
    *out = NULL;
    pl_mutex_lock(&s->lock);
    enum pl_pass_prepare_state state = r->state;
    if (state == PL_PASS_PREPARE_READY) {
        *out = r->prepared;
        r->prepared = NULL;
        unlink_request(s, r);
        *request = NULL;
    }
    pl_mutex_unlock(&s->lock);
    if (state == PL_PASS_PREPARE_READY)
        pl_free(r);
    return state;
}

void pl_pass_prepare_cancel(pl_pass_preparation r)
{
    struct pl_pass_prepare_service *s = r->service;
    pl_mutex_lock(&s->lock);
    r->state = PL_PASS_PREPARE_CANCELLED;
    pl_cond_signal(&s->wake);
    pl_mutex_unlock(&s->lock);
}

void pl_pass_prepare_release(pl_pass_preparation *request)
{
    pl_pass_preparation r = *request;
    if (!r)
        return;
    *request = NULL;
    struct pl_pass_prepare_service *s = r->service;
    pl_mutex_lock(&s->lock);
    r->state = PL_PASS_PREPARE_CANCELLED;
    r->released = true;
    pl_cond_signal(&s->wake);
    pl_mutex_unlock(&s->lock);
}

const struct pl_pass_params *pl_prepared_pass_params(pl_prepared_pass pass)
{
    return &pass->params;
}

void pl_prepared_pass_destroy(pl_prepared_pass *pass)
{
    if (!*pass)
        return;
    pl_prepared_pass p = *pass;
    pl_pass_destroy(p->gpu, &p->pass);
    pl_free(p);
    *pass = NULL;
}

enum pl_prepared_pass_run_result pl_prepared_pass_run(
    pl_prepared_pass pass, const struct pl_pass_run_params *params)
{
    if (!pass || !params || params->pass)
        return PL_PREPARED_PASS_RUN_INVALID;
    if (params->constant_data && pass->params.num_constants) {
        if (!pass->params.constant_data)
            return PL_PREPARED_PASS_RUN_VARIANT_MISMATCH;
        // Ignore layout padding; identity is the value of each constant.
        for (int i = 0; i < pass->params.num_constants; i++) {
            const struct pl_constant *c = &pass->params.constants[i];
            if (memcmp((const char *) params->constant_data + c->offset,
                       (const char *) pass->params.constant_data + c->offset,
                       pl_var_type_size(c->type)))
                return PL_PREPARED_PASS_RUN_VARIANT_MISMATCH;
        }
    }
    struct pl_pass_run_params run = *params;
    run.pass = pass->pass;
    run.constant_data = NULL;
    return pl_pass_run_checked(pass->gpu, &run, true);
}
