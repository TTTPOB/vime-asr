/* Pinned GGML Q4_K body / Q8_0 head: explicit offline export, read-only service. */
#include "vime_ar_weights.h"
#include "ggml-cpu.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint16_t *src;
    size_t elements, row_bytes, activation_bytes, bytes;
    const struct ggml_type_traits_cpu *weight_traits;
    ggml_from_float_t quantize_activation;
    int in_dim, out_dim;
    enum ggml_type type;
    unsigned char *quant;
} ar_entry;
static ar_entry *entries;
static int n_entries;

/* Native-endian x86_64 weights, with pinned GGML packing and fixed tensor order. */
typedef struct {
    char magic[8];
    uint32_t version, body_type, layers, dim, qdim, kvdim, inter, vocab;
} weights_header;
_Static_assert(sizeof(weights_header) == 40, "Unexpected AR weights header layout");

void vime_ar_free(void) {
    for (int i = 0; i < n_entries; i++) free(entries[i].quant);
    free(entries);
    entries = NULL;
    n_entries = 0;
}

/* Source pointers and CPU traits are rebuilt from the current context, never read from disk. */
static int register_weight(const uint16_t *src, int in_dim, int out_dim,
                           enum ggml_type type) {
    const struct ggml_type_traits_cpu *wt = ggml_get_type_traits_cpu(type);
    const struct ggml_type_traits_cpu *xt = ggml_get_type_traits_cpu(wt->vec_dot_type);
    if (!wt->vec_dot || wt->nrows != 1 || !xt->from_float ||
        in_dim % ggml_blck_size(type) || in_dim % ggml_blck_size(wt->vec_dot_type)) {
        fprintf(stderr, "Unsupported pinned GGML AR CPU traits\n");
        return -1;
    }
    size_t row_bytes = ggml_row_size(type, in_dim);
    entries[n_entries++] = (ar_entry){
        .src = src, .elements = (size_t)in_dim * out_dim,
        .row_bytes = row_bytes, .activation_bytes = ggml_row_size(wt->vec_dot_type, in_dim),
        .bytes = row_bytes * out_dim, .weight_traits = wt,
        .quantize_activation = xt->from_float, .in_dim = in_dim, .out_dim = out_dim,
        .type = type
    };
    return 0;
}

static int register_model(qwen_ctx_t *ctx, enum ggml_type body_type) {
    ggml_cpu_init();
    const qwen_config_t *c = &ctx->config;
    entries = calloc((size_t)c->dec_layers * 6 + 1, sizeof(*entries));
    if (!entries) return -1;
    int dim = c->dec_hidden, qdim = c->dec_heads*c->dec_head_dim;
    int kvdim = c->dec_kv_heads*c->dec_head_dim, inter = c->dec_intermediate;
    for (int i = 0; i < c->dec_layers; i++) {
        qwen_dec_layer_t *l = &ctx->decoder.layers[i];
        if (register_weight(l->wq_weight_bf16, dim, qdim, body_type) ||
            register_weight(l->wk_weight_bf16, dim, kvdim, body_type) ||
            register_weight(l->wv_weight_bf16, dim, kvdim, body_type) ||
            register_weight(l->wo_weight_bf16, qdim, dim, body_type) ||
            register_weight(l->gate_up_fused_bf16, dim, 2*inter, body_type) ||
            register_weight(l->down_weight_bf16, inter, dim, body_type)) return -1;
    }
    return register_weight(ctx->decoder.tok_embeddings_bf16, dim, c->vocab_size,
                           GGML_TYPE_Q8_0);
}

static weights_header model_header(const qwen_ctx_t *ctx, enum ggml_type body_type) {
    const qwen_config_t *c = &ctx->config;
    weights_header h = {0};
    memcpy(h.magic, "VIMEARW1", sizeof(h.magic));
    h.version = 1;
    h.body_type = body_type;
    h.layers = c->dec_layers;
    h.dim = c->dec_hidden;
    h.qdim = c->dec_heads*c->dec_head_dim;
    h.kvdim = c->dec_kv_heads*c->dec_head_dim;
    h.inter = c->dec_intermediate;
    h.vocab = c->vocab_size;
    return h;
}

static size_t weights_bytes(void) {
    size_t bytes = sizeof(weights_header);
    for (int i = 0; i < n_entries; i++) bytes += entries[i].bytes;
    return bytes;
}

/* No quantization, repair, fallback, source stamp or writer in this service path. */
int vime_ar_init(qwen_ctx_t *ctx) {
    const char *path = getenv("VIME_AR_WEIGHTS");
    if (!path || !*path) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open AR weights %s: %s\n", path, strerror(errno));
        return -1;
    }
    weights_header actual;
    int ok = fread(&actual, sizeof(actual), 1, f) == 1;
    if (!ok) fprintf(stderr, "Cannot read AR weights header: %s\n", path);
    if (ok && actual.body_type != GGML_TYPE_Q4_K && actual.body_type != GGML_TYPE_Q8_0) {
        fprintf(stderr, "Unsupported AR weights body type %u: %s\n", actual.body_type, path);
        ok = 0;
    }
    if (ok) {
        weights_header expected = model_header(ctx, (enum ggml_type)actual.body_type);
        if (memcmp(&actual, &expected, sizeof(actual))) {
            fprintf(stderr, "AR weights format/shape does not match model: %s\n", path);
            ok = 0;
        }
    }
    if (ok && register_model(ctx, (enum ggml_type)actual.body_type)) ok = 0;
    for (int i = 0; i < n_entries && ok; i++) {
        ar_entry *e = &entries[i];
        e->quant = malloc(e->bytes);
        if (!e->quant) {
            fprintf(stderr, "AR weights allocation failed: %s\n", strerror(errno));
            ok = 0;
        } else if (fread(e->quant, 1, e->bytes, f) != e->bytes) {
            fprintf(stderr, "Cannot read AR weights tensor %d: %s\n", i, path);
            ok = 0;
        }
    }
    fclose(f);
    if (!ok) { vime_ar_free(); return -1; }
    fprintf(stderr, "AR weights loaded %s (body=%u, %zu bytes)\n",
            path, actual.body_type, weights_bytes());
    return 0;
}

/* The reference quantizer is used only by the explicit offline export API. */
static int quantize_entry(ar_entry *e) {
    e->quant = malloc(e->bytes);
    float *row = malloc((size_t)e->in_dim * sizeof(float));
    if (!e->quant || !row) {
        fprintf(stderr, "AR export allocation failed: %s\n", strerror(errno));
        free(row);
        return -1;
    }
    for (int r = 0; r < e->out_dim; r++) {
        for (int k = 0; k < e->in_dim; k++) {
            uint32_t bits = (uint32_t)e->src[(size_t)r * e->in_dim + k] << 16;
            memcpy(row + k, &bits, sizeof(bits));
        }
        ggml_quantize_chunk(e->type, row, e->quant + (size_t)r * e->row_bytes,
                            0, 1, e->in_dim, NULL);
    }
    free(row);
    return 0;
}

/* Direct ordinary-file output. Deploy it only after the exporter exits successfully. */
int vime_write_ar_weights(void *opaque, const char *output) {
    qwen_ctx_t *ctx = opaque;
    if (n_entries) {
        fprintf(stderr, "AR export requires a BF16 context with VIME_AR_WEIGHTS unset\n");
        return -1;
    }
    const char *body = getenv("VIME_AR_BODY");
    enum ggml_type body_type = GGML_TYPE_Q4_K;
    if (body && strcmp(body, "q4_k")) {
        if (!strcmp(body, "q8_0")) body_type = GGML_TYPE_Q8_0;
        else {
            fprintf(stderr, "VIME_AR_BODY must be q4_k or q8_0 for offline export\n");
            return -1;
        }
    }
    if (register_model(ctx, body_type)) { vime_ar_free(); return -1; }
    FILE *f = fopen(output, "wb");
    if (!f) {
        fprintf(stderr, "Cannot write AR weights %s: %s\n", output, strerror(errno));
        vime_ar_free();
        return -1;
    }
    weights_header header = model_header(ctx, body_type);
    errno = 0;
    int ok = fwrite(&header, sizeof(header), 1, f) == 1;
    for (int i = 0; i < n_entries && ok; i++) {
        ar_entry *e = &entries[i];
        if (quantize_entry(e)) ok = 0;
        else ok = fwrite(e->quant, 1, e->bytes, f) == e->bytes;
        free(e->quant);
        e->quant = NULL;
    }
    int saved_errno = errno;
    if (fclose(f)) { ok = 0; saved_errno = errno; }
    if (!ok) fprintf(stderr, "AR export failed %s: %s\n", output,
                     strerror(saved_errno ? saved_errno : EIO));
    else fprintf(stderr, "AR weights exported %s (%zu bytes)\n", output, weights_bytes());
    vime_ar_free();
    return ok ? 0 : -1;
}

/* Workers only read initialized entries and never allocate or initialize traits. */
static const ar_entry *lookup(const uint16_t *w, int in_dim, int out_dim, size_t *offset) {
    uintptr_t p = (uintptr_t)w;
    for (int i = 0; i < n_entries; i++) {
        const ar_entry *e = &entries[i];
        uintptr_t begin = (uintptr_t)e->src, end = begin + e->elements*sizeof(uint16_t);
        if (e->in_dim == in_dim && p >= begin && p < end) {
            size_t elements = (p-begin)/sizeof(uint16_t);
            if (elements % in_dim || elements+(size_t)out_dim*in_dim > e->elements) return NULL;
            *offset = (elements/in_dim)*e->row_bytes;
            return e;
        }
    }
    return NULL;
}

int vime_ar_matvec(float *y, const float *x, const uint16_t *w, const float *bias,
                   int in_dim, int out_dim) {
    size_t offset;
    const ar_entry *e = lookup(w, in_dim, out_dim, &offset);
    if (!e) return 0;
    unsigned char qx[e->activation_bytes];
    e->quantize_activation(x, qx, in_dim);
    for (int r = 0; r < out_dim; r++) {
        float sum;
        e->weight_traits->vec_dot(in_dim, &sum, 0,
            e->quant + offset + (size_t)r*e->row_bytes, 0, qx, 0, 1);
        y[r] = sum + (bias ? bias[r] : 0.0f);
    }
    return 1;
}

int vime_ar_argmax(const float *x, const uint16_t *w, int in_dim, int start,
                   int end, int *best, float *best_val) {
    size_t offset;
    const ar_entry *e = lookup(w, in_dim, end, &offset);
    if (!e) return 0;
    unsigned char qx[e->activation_bytes];
    e->quantize_activation(x, qx, in_dim);
    *best = start;
    *best_val = -1e30f;
    for (int r = start; r < end; r++) {
        float sum;
        e->weight_traits->vec_dot(in_dim, &sum, 0,
            e->quant + offset + (size_t)r*e->row_bytes, 0, qx, 0, 1);
        if (sum > *best_val) { *best_val = sum; *best = r; }
    }
    return 1;
}
