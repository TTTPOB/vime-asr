/* Thin ownership bridge; the Python service serializes context use. */
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct {
    qwen_live_audio_t audio;
    size_t tokens;
    double first_token_at;
    void (*on_piece)(const char *);
} vime_live;

static void count_token(const char *piece, void *userdata) {
    vime_live *live = userdata;
    if (!live->tokens) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        live->first_token_at = ts.tv_sec + ts.tv_nsec / 1e9;
    }
    live->tokens++;
    if (live->on_piece) live->on_piece(piece);
}

void *vime_load(const char *dir) {
    qwen_verbose = 0;
    qwen_monitor = 0;
    qwen_set_threads(4);
    qwen_ctx_t *ctx = qwen_load(dir);
    if (ctx) {
        ctx->past_text_conditioning = 1;
        const char *unfixed = getenv("VIME_STREAM_UNFIXED_CHUNKS");
        ctx->stream_unfixed_chunks = unfixed ? atoi(unfixed) : 0;
    }
    return ctx;
}
void vime_model_free(void *ctx) { qwen_free(ctx); }

void *vime_live_new(void) {
    vime_live *live = calloc(1, sizeof(*live));
    if (!live) return NULL;
    if (pthread_mutex_init(&live->audio.mutex, NULL)) { free(live); return NULL; }
    if (pthread_cond_init(&live->audio.cond, NULL)) {
        pthread_mutex_destroy(&live->audio.mutex); free(live); return NULL;
    }
    return live;
}

int vime_append(void *handle, const float *samples, int n) {
    vime_live *live = handle;
    qwen_live_audio_t *a = &live->audio;
    if (n < 0) return -1;
    pthread_mutex_lock(&a->mutex);
    if (a->eof) { pthread_mutex_unlock(&a->mutex); return -1; }
    int64_t need = a->n_samples + n;
    if ((uint64_t)need > SIZE_MAX / sizeof(float)) {
        pthread_mutex_unlock(&a->mutex); return -1;
    }
    if (need > a->capacity) {
        int64_t cap = need > 32000 ? need : 32000;
        float *p = realloc(a->samples, (size_t)cap * sizeof(float));
        if (!p) { pthread_mutex_unlock(&a->mutex); return -1; }
        a->samples = p;
        a->capacity = cap;
    }
    /* n_samples is the unconsumed tail, not a global sample index.
       Never reset sample_offset: the consumer advances it under this lock. */
    if (n) memcpy(a->samples + a->n_samples, samples, (size_t)n * sizeof(float));
    a->n_samples = need;
    pthread_cond_signal(&a->cond);
    pthread_mutex_unlock(&a->mutex);
    return 0;
}
void vime_eof(void *handle) {
    qwen_live_audio_t *a = &((vime_live *)handle)->audio;
    pthread_mutex_lock(&a->mutex);
    a->eof = 1;
    pthread_cond_broadcast(&a->cond);
    pthread_mutex_unlock(&a->mutex);
}
void *vime_run_stream(void *ctx, void *handle, void (*on_piece)(const char *)) {
    vime_live *live = handle;
    live->on_piece = on_piece;
    qwen_set_token_callback(ctx, count_token, live);
    char *text = qwen_transcribe_stream_live(ctx, &live->audio);
    qwen_set_token_callback(ctx, NULL, NULL);
    live->on_piece = NULL;
    return text;
}
size_t vime_tokens(void *handle) { return ((vime_live *)handle)->tokens; }
double vime_first_token_at(void *handle) { return ((vime_live *)handle)->first_token_at; }
void vime_text_free(void *text) { free(text); }
/* Only call after vime_run returns and all producer operations finish. */
void vime_live_free(void *handle) {
    vime_live *live = handle;
    free(live->audio.samples);
    pthread_cond_destroy(&live->audio.cond);
    pthread_mutex_destroy(&live->audio.mutex);
    free(live);
}
