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

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <libplacebo/frame_mixer.h>

struct entry {
    struct pl_mixer_descriptor desc;
    unsigned instances;
    struct entry *next;
};
struct pl_mixer_registry_t { struct entry *head; };
struct pl_mixer_t {
    struct entry *entry;
    void *instance;
    pl_gpu gpu;
    struct pl_mixer_requirements req;
    unsigned jobs, owners;
};
struct pl_mixer_job_t {
    pl_mixer mixer;
    struct pl_mixer_job_params params;
    void *ticket;
    enum pl_mixer_status status;
    struct pl_mixer_frame frames[];
};

pl_mixer_registry pl_mixer_registry_create(void)
{
    return calloc(1, sizeof(struct pl_mixer_registry_t));
}
enum pl_mixer_status pl_mixer_registry_destroy(pl_mixer_registry *r)
{
    if (!r || !*r) return PL_MIXER_INVALID;
    if ((*r)->head) return PL_MIXER_BUSY;
    free(*r); *r = NULL;
    return PL_MIXER_OK;
}
static struct entry **find(pl_mixer_registry r, const char *id)
{
    struct entry **p = &r->head;
    while (*p && strcmp((*p)->desc.id, id)) p = &(*p)->next;
    return p;
}
enum pl_mixer_status pl_mixer_register(pl_mixer_registry r,
                                      const struct pl_mixer_descriptor *d)
{
    if (!r || !d || d->abi_version != PL_MIXER_ABI_VERSION ||
        d->struct_size < sizeof(*d)) return PL_MIXER_UNSUPPORTED;
    if (!d->id || !*d->id || !d->name || !d->retain || !d->release ||
        !d->negotiate || !d->create || !d->submit || !d->poll ||
        !d->release_job || !d->destroy) return PL_MIXER_INVALID;
    struct entry **p = find(r, d->id);
    if (*p) return PL_MIXER_INVALID;
    struct entry *e = calloc(1, sizeof(*e));
    if (!e) return PL_MIXER_FAILED;
    e->desc = *d;
    e->desc.struct_size = sizeof(e->desc);
    e->desc.retain(e->desc.priv);
    *p = e;
    return PL_MIXER_OK;
}
enum pl_mixer_status pl_mixer_unregister(pl_mixer_registry r, const char *id)
{
    if (!r || !id) return PL_MIXER_INVALID;
    struct entry **p = find(r, id), *e = *p;
    if (!e) return PL_MIXER_INVALID;
    if (e->instances) return PL_MIXER_BUSY;
    *p = e->next;
    e->desc.release(e->desc.priv);
    free(e);
    return PL_MIXER_OK;
}
int pl_mixer_registry_count(pl_mixer_registry r)
{
    int count = 0;
    if (r) for (struct entry *e = r->head; e; e = e->next) count++;
    return count;
}
const struct pl_mixer_descriptor *pl_mixer_registry_get(pl_mixer_registry r, int i)
{
    if (!r || i < 0) return NULL;
    struct entry *e = r->head;
    while (e && i--) e = e->next;
    return e ? &e->desc : NULL;
}
enum pl_mixer_status pl_mixer_create(pl_mixer_registry r, const char *id,
    pl_gpu gpu, const struct pl_mixer_config *config, pl_mixer *out)
{
    if (!out) return PL_MIXER_INVALID;
    *out = NULL;
    if (!r || !id || !gpu || !config || (config->size && !config->data))
        return PL_MIXER_INVALID;
    struct entry *e = *find(r, id);
    if (!e) return PL_MIXER_UNSUPPORTED;
    pl_mixer m = calloc(1, sizeof(*m));
    if (!m) return PL_MIXER_FAILED;
    m->entry = e; m->gpu = gpu;
    enum pl_mixer_status s = e->desc.negotiate(e->desc.priv, gpu, config, &m->req);
    if (s != PL_MIXER_OK) goto fail;
    const struct pl_mixer_requirements *q = &m->req;
    bool known_format = false;
    for (int i = 0; i < gpu->num_formats; i++)
        known_format |= gpu->formats[i] == q->format;
    if (!known_format || !isfinite(q->radius) || q->radius < 0 || !q->max_frames ||
        (q->domain == PL_MIXER_COLOR_LINEAR && q->format->type != PL_FMT_FLOAT) ||
        !q->max_in_flight ||
        (q->domain != PL_MIXER_COLOR_COMPLETED && q->domain != PL_MIXER_COLOR_LINEAR) ||
        (q->backend != PL_MIXER_BACKEND_VULKAN && q->backend != PL_MIXER_BACKEND_CUDA) ||
        (q->export_handle && ((q->export_handle & (q->export_handle - 1)) ||
          !(gpu->export_caps.tex & q->export_handle))) ||
        (q->backend == PL_MIXER_BACKEND_CUDA && !q->export_handle)) {
        s = PL_MIXER_UNSUPPORTED; goto fail;
    }
    s = e->desc.create(e->desc.priv, gpu, config, q, &m->instance);
    if (s != PL_MIXER_OK) goto fail;
    e->instances++;
    *out = m;
    return PL_MIXER_OK;
fail:
    free(m);
    return s == PL_MIXER_PENDING ? PL_MIXER_INVALID : s;
}
const struct pl_mixer_requirements *pl_mixer_get_requirements(pl_mixer m)
{
    return m ? &m->req : NULL;
}
enum pl_mixer_status pl_mixer_destroy(pl_mixer *m)
{
    if (!m || !*m) return PL_MIXER_INVALID;
    if ((*m)->jobs || (*m)->owners) return PL_MIXER_BUSY;
    (*m)->entry->desc.destroy((*m)->instance);
    (*m)->entry->instances--;
    free(*m); *m = NULL;
    return PL_MIXER_OK;
}
static bool texture_valid(pl_mixer m, pl_tex t)
{
    return t && t->params.format == m->req.format &&
        t->params.export_handle == m->req.export_handle &&
        t->params.w > 0 && t->params.h > 0 && !t->params.d;
}
enum pl_mixer_status pl_mixer_submit(pl_mixer m,
    const struct pl_mixer_job_params *p, pl_mixer_job *out)
{
    if (!out) return PL_MIXER_INVALID;
    *out = NULL;
    if (!m || !p || !p->retain || !p->release || !p->frames ||
        !isfinite(p->pts) || !p->num_frames ||
        p->num_frames > m->req.max_frames || !texture_valid(m, p->output) ||
        (m->req.domain == PL_MIXER_COLOR_LINEAR && p->color.transfer != PL_COLOR_TRC_LINEAR))
        return PL_MIXER_INVALID;
    if (m->jobs >= m->req.max_in_flight) return PL_MIXER_BUSY;
    for (uint32_t i = 0; i < p->num_frames; i++) {
        const struct pl_mixer_frame *f = &p->frames[i];
        if (!texture_valid(m, f->texture) || f->texture == p->output ||
            !isfinite(f->pts) || (i && f->pts <= p->frames[i-1].pts))
            return PL_MIXER_INVALID;
    }
    size_t bytes = (size_t) p->num_frames * sizeof(*p->frames);
    if (bytes / sizeof(*p->frames) != p->num_frames ||
        bytes > SIZE_MAX - sizeof(struct pl_mixer_job_t)) return PL_MIXER_INVALID;
    pl_mixer_job j = calloc(1, sizeof(*j) + bytes);
    if (!j) return PL_MIXER_FAILED;
    j->mixer = m; j->params = *p;
    memcpy(j->frames, p->frames, bytes);
    j->params.frames = j->frames;
    p->retain(p->owner);
    j->status = m->entry->desc.submit(m->instance, &j->params, &j->ticket);
    if (j->status != PL_MIXER_OK && j->status != PL_MIXER_PENDING) {
        enum pl_mixer_status s = j->status;
        p->release(p->owner); free(j); return s;
    }
    m->jobs++;
    *out = j;
    return j->status;
}
enum pl_mixer_status pl_mixer_poll(pl_mixer_job j)
{
    if (!j) return PL_MIXER_INVALID;
    if (j->status == PL_MIXER_PENDING) {
        enum pl_mixer_status s = j->mixer->entry->desc.poll(j->mixer->instance, j->ticket);
        // An invalid callback result cannot prove GPU completion. Keep the lease.
        if (s != PL_MIXER_OK && s != PL_MIXER_FAILED && s != PL_MIXER_PENDING)
            return PL_MIXER_INVALID;
        j->status = s;
    }
    return j->status;
}
enum pl_mixer_status pl_mixer_job_destroy(pl_mixer_job *j)
{
    if (!j || !*j) return PL_MIXER_INVALID;
    if ((*j)->status == PL_MIXER_PENDING) return PL_MIXER_BUSY;
    pl_mixer m = (*j)->mixer;
    m->entry->desc.release_job(m->instance, (*j)->ticket);
    (*j)->params.release((*j)->params.owner);
    m->jobs--;
    free(*j); *j = NULL;
    return PL_MIXER_OK;
}

pl_gpu pl_mixer_get_gpu(pl_mixer m) { return m ? m->gpu : NULL; }
void pl_mixer_retain(pl_mixer m) { if (m) m->owners++; }
void pl_mixer_release(pl_mixer m) { if (m && m->owners) m->owners--; }
