/* Offline AR weight export and read-only loading for one serialized model. */
#ifndef VIME_AR_WEIGHTS_H
#define VIME_AR_WEIGHTS_H
#include "qwen_asr.h"
int vime_ar_init(qwen_ctx_t *ctx);
void vime_ar_free(void);
int vime_ar_matvec(float *y, const float *x, const uint16_t *w, const float *bias,
                   int in_dim, int out_dim);
int vime_ar_argmax(const float *x, const uint16_t *w, int in_dim, int start,
                   int end, int *best, float *best_val);
/* Call explicitly on a BF16 context; the daemon never invokes this writer. */
int vime_write_ar_weights(void *ctx, const char *output);
#endif
