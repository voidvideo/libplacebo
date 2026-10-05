#pragma once

#include "utils.h"
#include "pl_thread.h"

// A blocked worker is released only by the test owner. Public operations must
// finish while it is blocked, so a hidden wait is a deterministic test timeout.
struct prepare_latch {
    pl_mutex lock;
    pl_cond cond;
    bool armed, entered, released;
};

static void latch_init(struct prepare_latch *l)
{
    pl_mutex_init(&l->lock);
    REQUIRE(!pl_cond_init(&l->cond));
}

static void latch_arm(struct prepare_latch *l)
{
    pl_mutex_lock(&l->lock);
    l->armed = true;
    l->entered = l->released = false;
    pl_mutex_unlock(&l->lock);
}

static void latch_block(struct prepare_latch *l)
{
    pl_mutex_lock(&l->lock);
    if (l->armed) {
        l->entered = true;
        pl_cond_broadcast(&l->cond);
        while (!l->released)
            pl_cond_wait(&l->cond, &l->lock);
        l->armed = false;
    }
    pl_mutex_unlock(&l->lock);
}

static void latch_entered(struct prepare_latch *l)
{
    pl_mutex_lock(&l->lock);
    while (!l->entered)
        REQUIRE(!pl_cond_timedwait(&l->cond, &l->lock, 5000000000ULL));
    pl_mutex_unlock(&l->lock);
}

static void latch_release(struct prepare_latch *l)
{
    pl_mutex_lock(&l->lock);
    l->released = true;
    pl_cond_broadcast(&l->cond);
    pl_mutex_unlock(&l->lock);
}

static void latch_destroy(struct prepare_latch *l)
{
    pl_cond_destroy(&l->cond);
    pl_mutex_destroy(&l->lock);
}

static enum pl_pass_prepare_state await_prepared(pl_pass_preparation request)
{
    pl_clock_t start = pl_clock_now();
    enum pl_pass_prepare_state state;
    while ((state = pl_pass_prepare_poll(request)) == PL_PASS_PREPARE_PENDING) {
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 10.0);
        pl_thread_sleep(0.001);
    }
    return state;
}

static pl_prepared_pass take_prepared(pl_pass_preparation *request)
{
    REQUIRE(await_prepared(*request) == PL_PASS_PREPARE_READY);
    pl_prepared_pass pass = NULL;
    REQUIRE(pl_pass_prepare_take(request, &pass) == PL_PASS_PREPARE_READY);
    REQUIRE(!*request);
    REQUIRE(pass);
    return pass;
}
