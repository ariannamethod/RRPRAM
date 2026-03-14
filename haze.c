/*
 * haze.c — Haze: Hybrid Attention Entropy System
 * PostGPT: post-transformer hybrid attention language model
 *
 * Three attention modes:
 *   RRPRAM  — attn = x @ wr          (positional pattern recognition)
 *   Content — attn = (x@wq)(x@wk)^T  (semantic similarity, classic QKV)
 *   Hybrid  — learned gate alpha between RRPRAM and Content
 *
 * "What comes after you understand GPT and ask:
 *  what if we didn't compute QK^T for everything?"
 *
 * Character-level (byte vocab 256). Train + generate. Zero dependencies.
 *   cc haze.c -O2 -lm -o haze
 *   ./haze --train data.txt --mode hybrid --depth 4
 *   ./haze --generate --load haze.bin --sampling entropy
 *
 * Sampling strategies: basic, top_k, top_p, entropy, mirostat
 *
 * By Arianna Method. 2026.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ===== config ===== */

#define VOCAB    256
#define MAX_BLK  16
#define MAX_DIM  512
#define MAX_CTX  128

typedef enum { MODE_RRPRAM, MODE_CONTENT, MODE_HYBRID } HeadMode;
typedef enum { SAMP_BASIC, SAMP_TOP_K, SAMP_TOP_P, SAMP_ENTROPY, SAMP_MIROSTAT } Sampling;

typedef struct {
    int T;       /* context length */
    int E;       /* embedding dim */
    int H;       /* number of heads */
    int D;       /* head dim = E/H */
    int B;       /* number of blocks */
    int M;       /* MLP hidden dim */
    HeadMode mode;
} Cfg;

static Cfg cfg_from_depth(int depth, HeadMode mode) {
    Cfg c;
    c.T = (depth >= 8) ? 64 : 32;
    c.E = depth * 16;
    c.H = (depth < 4) ? 2 : 4;
    c.D = c.E / c.H;
    c.B = depth;
    c.M = c.E * 2;
    c.mode = mode;
    return c;
}

/* ===== math ===== */

static void matmul(float *C, const float *A, const float *B, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
}

static void matmul_atb(float *C, const float *A, const float *B, int m, int k, int n) {
    /* C[m,n] = A^T @ B  where A is [k,m] */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[p*m+i] * B[p*n+j];
            C[i*n+j] = s;
        }
}

static void matmul_abt(float *C, const float *A, const float *B, int m, int k, int n) {
    /* C[m,n] = A @ B^T  where B is [n,k] */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
}

static void row_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    float inv = 1.0f / (s + 1e-10f);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static float gelu_f(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)));
}

static float gelu_grad(float x) {
    float k = 0.7978845608f, c = 0.044715f;
    float inner = k * (x + c * x*x*x);
    float t = tanhf(inner);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t*t) * k * (1.0f + 3.0f*c*x*x);
}

static float randn(void) {
    float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ===== layer norm ===== */

static void layernorm_fwd(float *out, const float *x, const float *g, const float *b,
                          int T, int E) {
    for (int t = 0; t < T; t++) {
        float mean = 0;
        for (int e = 0; e < E; e++) mean += x[t*E+e];
        mean /= E;
        float var = 0;
        for (int e = 0; e < E; e++) { float d = x[t*E+e] - mean; var += d*d; }
        var /= E;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int e = 0; e < E; e++)
            out[t*E+e] = g[e] * (x[t*E+e] - mean) * inv + b[e];
    }
}

static void layernorm_bwd(float *dx, float *dg, float *db,
                          const float *dout, const float *x, const float *g,
                          int T, int E) {
    for (int t = 0; t < T; t++) {
        float mean = 0;
        for (int e = 0; e < E; e++) mean += x[t*E+e];
        mean /= E;
        float var = 0;
        for (int e = 0; e < E; e++) { float d = x[t*E+e] - mean; var += d*d; }
        var /= E;
        float inv = 1.0f / sqrtf(var + 1e-5f);

        float xhat[MAX_DIM], dxh[MAX_DIM];
        float m1 = 0, m2 = 0;
        for (int e = 0; e < E; e++) {
            xhat[e] = (x[t*E+e] - mean) * inv;
            dxh[e] = dout[t*E+e] * g[e];
            dg[e] += dout[t*E+e] * xhat[e];
            db[e] += dout[t*E+e];
            m1 += dxh[e];
            m2 += dxh[e] * xhat[e];
        }
        m1 /= E; m2 /= E;
        for (int e = 0; e < E; e++)
            dx[t*E+e] = inv * (dxh[e] - m1 - xhat[e] * m2);
    }
}

/* ===== sampling strategies ===== */

static float entropy_bits(const float *probs, int n) {
    float h = 0;
    for (int i = 0; i < n; i++)
        if (probs[i] > 1e-10f)
            h -= probs[i] * log2f(probs[i]);
    return h;
}

static int sample_basic(float *logits, int n, float temp) {
    if (temp <= 0) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < n; i++) logits[i] /= temp;
    row_softmax(logits, n);
    float r = (float)rand() / RAND_MAX;
    float cum = 0;
    for (int i = 0; i < n; i++) { cum += logits[i]; if (cum >= r) return i; }
    return n - 1;
}

static int sample_top_k(float *logits, int n, int k, float temp) {
    if (temp <= 0) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    /* find k-th largest value */
    float *sorted = malloc(n * sizeof(float));
    memcpy(sorted, logits, n * sizeof(float));
    /* partial sort: find threshold */
    for (int i = 0; i < k && i < n; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (sorted[j] > sorted[best]) best = j;
        float tmp = sorted[i]; sorted[i] = sorted[best]; sorted[best] = tmp;
    }
    float threshold = (k < n) ? sorted[k-1] : sorted[n-1];
    free(sorted);

    /* mask below threshold */
    for (int i = 0; i < n; i++)
        if (logits[i] < threshold) logits[i] = -1e9f;
    for (int i = 0; i < n; i++) logits[i] /= temp;
    row_softmax(logits, n);

    float r = (float)rand() / RAND_MAX;
    float cum = 0;
    for (int i = 0; i < n; i++) { cum += logits[i]; if (cum >= r) return i; }
    return n - 1;
}

static int sample_top_p(float *logits, int n, float p, float temp) {
    if (temp <= 0) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < n; i++) logits[i] /= temp;
    float probs[VOCAB];
    memcpy(probs, logits, n * sizeof(float));
    row_softmax(probs, n);

    /* sort indices by probability descending */
    int *idx = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < n-1; i++)
        for (int j = i+1; j < n; j++)
            if (probs[idx[j]] > probs[idx[i]]) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }

    /* find cutoff */
    float cum = 0;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cum += probs[idx[i]];
        if (cum >= p) { cutoff = i + 1; break; }
    }

    /* mask and renormalize */
    float mask[VOCAB];
    memset(mask, 0, n * sizeof(float));
    for (int i = 0; i < cutoff; i++) mask[idx[i]] = probs[idx[i]];
    float total = 0;
    for (int i = 0; i < n; i++) total += mask[i];
    float inv = 1.0f / (total + 1e-10f);

    float r = (float)rand() / RAND_MAX;
    cum = 0;
    int result = n - 1;
    for (int i = 0; i < n; i++) {
        cum += mask[i] * inv;
        if (cum >= r) { result = i; break; }
    }
    free(idx);
    return result;
}

static float entropy_temperature(float *logits, int n,
                                  float target_entropy, float min_temp, float max_temp) {
    float probs[VOCAB];
    memcpy(probs, logits, n * sizeof(float));
    row_softmax(probs, n);
    float h = entropy_bits(probs, n);
    if (h < 1e-6f) return min_temp;
    float ratio = target_entropy / h;
    float temp = powf(ratio, 0.5f);
    if (temp < min_temp) temp = min_temp;
    if (temp > max_temp) temp = max_temp;
    return temp;
}

static int sample_entropy(float *logits, int n, float target_entropy,
                           float top_p, float min_temp, float max_temp) {
    float temp = entropy_temperature(logits, n, target_entropy, min_temp, max_temp);
    float work[VOCAB];
    memcpy(work, logits, n * sizeof(float));
    return sample_top_p(work, n, top_p, temp);
}

static int sample_mirostat(float *logits, int n, float target_entropy,
                            float tau, float *mu) {
    float probs[VOCAB];
    memcpy(probs, logits, n * sizeof(float));
    row_softmax(probs, n);

    /* sort descending */
    int *idx = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < n-1; i++)
        for (int j = i+1; j < n; j++)
            if (probs[idx[j]] > probs[idx[i]]) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }

    /* find k: tokens with surprise <= mu */
    int k = 0;
    for (int i = 0; i < n; i++) {
        float surprise = -log2f(probs[idx[i]] + 1e-10f);
        if (surprise <= *mu) k = i + 1;
    }
    if (k == 0) k = 1;

    /* sample from top-k */
    float total = 0;
    for (int i = 0; i < k; i++) total += probs[idx[i]];
    float r = (float)rand() / RAND_MAX * total;
    float cum = 0;
    int chosen = idx[0];
    for (int i = 0; i < k; i++) {
        cum += probs[idx[i]];
        if (cum >= r) { chosen = idx[i]; break; }
    }

    /* update mu */
    float observed = -log2f(probs[chosen] + 1e-10f);
    *mu -= tau * (observed - target_entropy);

    free(idx);
    return chosen;
}

/* ===== model ===== */

/*
 * Weight layout varies by mode:
 *   RRPRAM:  per head — wv[E,D] + wr[E,T]
 *   Content: per head — wq[E,D] + wk[E,D] + wv[E,D]
 *   Hybrid:  per head — wq[E,D] + wk[E,D] + wv[E,D] + wr[E,T] + gate[1]
 */

typedef struct {
    float *tok_emb;   /* [V, E] */
    float *pos_emb;   /* [T, E] */
    float *ln1_g[MAX_BLK], *ln1_b[MAX_BLK];
    /* attention weights: flat buffer per block, layout depends on mode */
    float *attn_w[MAX_BLK];
    float *wo[MAX_BLK];     /* [E, E] */
    float *ln2_g[MAX_BLK], *ln2_b[MAX_BLK];
    float *w1[MAX_BLK], *b1[MAX_BLK];
    float *w2[MAX_BLK], *b2[MAX_BLK];
    float *fln_g, *fln_b;
    float *out_w;           /* [E, V] */
} Ptrs;

/* per-head attention weight size */
static int head_weight_size(Cfg *c) {
    switch (c->mode) {
        case MODE_RRPRAM:  return c->E * c->D + c->E * c->T;           /* wv + wr */
        case MODE_CONTENT: return 3 * c->E * c->D;                     /* wq + wk + wv */
        case MODE_HYBRID:  return 3 * c->E * c->D + c->E * c->T + 1;  /* wq + wk + wv + wr + gate */
    }
    return 0;
}

static int model_size(Cfg *c) {
    int s = VOCAB * c->E + c->T * c->E;
    int hw = head_weight_size(c);
    for (int b = 0; b < c->B; b++) {
        s += 2 * c->E;           /* ln1 */
        s += c->H * hw;          /* attention heads */
        s += c->E * c->E;        /* wo */
        s += 2 * c->E;           /* ln2 */
        s += c->E * c->M + c->M; /* w1, b1 */
        s += c->M * c->E + c->E; /* w2, b2 */
    }
    s += 2 * c->E;
    s += c->E * VOCAB;
    return s;
}

static void assign_ptrs(Ptrs *p, float *base, Cfg *c) {
    float *q = base;
    int hw = head_weight_size(c);
    p->tok_emb = q; q += VOCAB * c->E;
    p->pos_emb = q; q += c->T * c->E;
    for (int b = 0; b < c->B; b++) {
        p->ln1_g[b] = q; q += c->E;
        p->ln1_b[b] = q; q += c->E;
        p->attn_w[b] = q; q += c->H * hw;
        p->wo[b] = q; q += c->E * c->E;
        p->ln2_g[b] = q; q += c->E;
        p->ln2_b[b] = q; q += c->E;
        p->w1[b] = q; q += c->E * c->M;
        p->b1[b] = q; q += c->M;
        p->w2[b] = q; q += c->M * c->E;
        p->b2[b] = q; q += c->E;
    }
    p->fln_g = q; q += c->E;
    p->fln_b = q; q += c->E;
    p->out_w = q;
}

/* head weight accessors */
static float *head_wv(Ptrs *p, Cfg *c, int blk, int h) {
    int hw = head_weight_size(c);
    float *base = p->attn_w[blk] + h * hw;
    switch (c->mode) {
        case MODE_RRPRAM:  return base;                    /* wv is first */
        case MODE_CONTENT: return base + 2 * c->E * c->D; /* after wq, wk */
        case MODE_HYBRID:  return base + 2 * c->E * c->D; /* after wq, wk */
    }
    return NULL;
}

static float *head_wr(Ptrs *p, Cfg *c, int blk, int h) {
    /* only for RRPRAM and Hybrid */
    int hw = head_weight_size(c);
    float *base = p->attn_w[blk] + h * hw;
    switch (c->mode) {
        case MODE_RRPRAM:  return base + c->E * c->D;                   /* after wv */
        case MODE_HYBRID:  return base + 3 * c->E * c->D;               /* after wq, wk, wv */
        default: return NULL;
    }
}

static float *head_wq(Ptrs *p, Cfg *c, int blk, int h) {
    /* only for Content and Hybrid */
    int hw = head_weight_size(c);
    float *base = p->attn_w[blk] + h * hw;
    return base; /* wq is first for content/hybrid */
}

static float *head_wk(Ptrs *p, Cfg *c, int blk, int h) {
    int hw = head_weight_size(c);
    float *base = p->attn_w[blk] + h * hw;
    return base + c->E * c->D; /* after wq */
}

static float *head_gate(Ptrs *p, Cfg *c, int blk, int h) {
    /* only for Hybrid */
    int hw = head_weight_size(c);
    float *base = p->attn_w[blk] + h * hw;
    return base + 3 * c->E * c->D + c->E * c->T; /* last element */
}

typedef struct {
    Cfg cfg;
    int n_params;
    float *data, *grad, *adam_m, *adam_v;
    Ptrs w, g;
} Model;

static void model_init(Model *m, int depth, HeadMode mode) {
    m->cfg = cfg_from_depth(depth, mode);
    m->n_params = model_size(&m->cfg);
    m->data   = calloc(m->n_params, sizeof(float));
    m->grad   = calloc(m->n_params, sizeof(float));
    m->adam_m  = calloc(m->n_params, sizeof(float));
    m->adam_v  = calloc(m->n_params, sizeof(float));
    assign_ptrs(&m->w, m->data, &m->cfg);
    assign_ptrs(&m->g, m->grad, &m->cfg);

    Cfg *c = &m->cfg;
    float scale;

    /* Xavier init */
    scale = sqrtf(2.0f / VOCAB);
    for (int i = 0; i < VOCAB * c->E; i++) m->w.tok_emb[i] = randn() * scale;
    scale = sqrtf(2.0f / c->T);
    for (int i = 0; i < c->T * c->E; i++) m->w.pos_emb[i] = randn() * scale;

    for (int b = 0; b < c->B; b++) {
        for (int e = 0; e < c->E; e++) { m->w.ln1_g[b][e] = 1.0f; m->w.ln2_g[b][e] = 1.0f; }

        scale = sqrtf(2.0f / c->E);
        int hw = head_weight_size(c);
        for (int h = 0; h < c->H; h++) {
            float *base = m->w.attn_w[b] + h * hw;
            int attn_floats = hw;
            if (c->mode == MODE_HYBRID) attn_floats -= 1; /* don't randomize gate */
            for (int i = 0; i < attn_floats; i++) base[i] = randn() * scale;
            if (c->mode == MODE_HYBRID) {
                /* init gate to 0.5 */
                *head_gate(&m->w, c, b, h) = 0.5f;
            }
        }

        for (int i = 0; i < c->E * c->E; i++) m->w.wo[b][i] = randn() * scale / sqrtf(c->B);
        for (int i = 0; i < c->E * c->M; i++) m->w.w1[b][i] = randn() * scale;
        scale = sqrtf(2.0f / c->M);
        for (int i = 0; i < c->M * c->E; i++) m->w.w2[b][i] = randn() * scale / sqrtf(c->B);
    }
    for (int e = 0; e < c->E; e++) m->w.fln_g[e] = 1.0f;
    scale = sqrtf(2.0f / c->E);
    for (int i = 0; i < c->E * VOCAB; i++) m->w.out_w[i] = randn() * scale;

    const char *mode_str[] = {"rrpram", "content", "hybrid"};
    printf("[haze] mode=%s depth=%d params=%d (%.1fK)\n",
           mode_str[mode], depth, m->n_params, m->n_params/1000.0f);
    printf("[haze] T=%d E=%d H=%d D=%d B=%d M=%d\n", c->T, c->E, c->H, c->D, c->B, c->M);
}

/* ===== activation cache ===== */

typedef struct {
    float *x;        /* [(B+1)*T*E] */
    float *ln1;      /* [B*T*E] */
    /* attention intermediates — we allocate max needed across modes */
    float *v;        /* [B*H*T*D] */
    float *q;        /* [B*H*T*D] (content/hybrid only) */
    float *k;        /* [B*H*T*D] (content/hybrid only) */
    float *attn_r;   /* [B*H*T*T] RRPRAM attention */
    float *attn_c;   /* [B*H*T*T] Content attention (hybrid only) */
    float *concat;   /* [B*T*E] */
    float *r1;       /* [B*T*E] */
    float *ln2;      /* [B*T*E] */
    float *mlp_pre;  /* [B*T*M] */
    float *fln;      /* [T*E] */
    float *logits;   /* [T*V] */
    float *probs;    /* [T*V] */
} Acts;

static void acts_alloc(Acts *a, Cfg *c) {
    int TE = c->T * c->E, TT = c->T * c->T, TD = c->T * c->D, TM = c->T * c->M;
    a->x       = calloc((c->B+1) * TE, sizeof(float));
    a->ln1     = calloc(c->B * TE, sizeof(float));
    a->v       = calloc(c->B * c->H * TD, sizeof(float));
    a->q       = (c->mode != MODE_RRPRAM) ? calloc(c->B * c->H * TD, sizeof(float)) : NULL;
    a->k       = (c->mode != MODE_RRPRAM) ? calloc(c->B * c->H * TD, sizeof(float)) : NULL;
    a->attn_r  = (c->mode != MODE_CONTENT) ? calloc(c->B * c->H * TT, sizeof(float)) : NULL;
    a->attn_c  = (c->mode == MODE_HYBRID) ? calloc(c->B * c->H * TT, sizeof(float)) : NULL;
    a->concat  = calloc(c->B * TE, sizeof(float));
    a->r1      = calloc(c->B * TE, sizeof(float));
    a->ln2     = calloc(c->B * TE, sizeof(float));
    a->mlp_pre = calloc(c->B * TM, sizeof(float));
    a->fln     = calloc(TE, sizeof(float));
    a->logits  = calloc(c->T * VOCAB, sizeof(float));
    a->probs   = calloc(c->T * VOCAB, sizeof(float));
}

static void acts_free(Acts *a) {
    free(a->x); free(a->ln1); free(a->v);
    free(a->q); free(a->k);
    free(a->attn_r); free(a->attn_c);
    free(a->concat); free(a->r1); free(a->ln2); free(a->mlp_pre);
    free(a->fln); free(a->logits); free(a->probs);
}

/* ===== forward ===== */

static void causal_mask_softmax(float *attn, int T) {
    for (int i = 0; i < T; i++) {
        for (int j = i+1; j < T; j++) attn[i*T+j] = -1e9f;
        row_softmax(attn + i*T, T);
    }
}

static float forward(Model *m, Acts *a, const int *tokens, const int *targets) {
    Cfg *c = &m->cfg;
    int T = c->T, E = c->E, H = c->H, D = c->D, B = c->B, M = c->M;
    Ptrs *w = &m->w;
    float *head_buf = calloc(T * D, sizeof(float));
    float scale = 1.0f / sqrtf((float)D);

    /* embedding */
    float *x0 = a->x;
    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++)
            x0[t*E+e] = w->tok_emb[tokens[t]*E+e] + w->pos_emb[t*E+e];

    for (int b = 0; b < B; b++) {
        float *xin  = a->x + b * T * E;
        float *xout = a->x + (b+1) * T * E;
        float *ln1  = a->ln1 + b * T * E;
        float *cat  = a->concat + b * T * E;
        float *r1   = a->r1 + b * T * E;
        float *ln2  = a->ln2 + b * T * E;
        float *mpre = a->mlp_pre + b * T * M;

        layernorm_fwd(ln1, xin, w->ln1_g[b], w->ln1_b[b], T, E);

        memset(cat, 0, T * E * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *v_h = a->v + (b*H+h) * T * D;
            float *wv_h = head_wv(w, c, b, h);
            matmul(v_h, ln1, wv_h, T, E, D);

            if (c->mode == MODE_RRPRAM) {
                /* RRPRAM: attn = ln1 @ wr */
                float *wr_h = head_wr(w, c, b, h);
                float *at_h = a->attn_r + (b*H+h) * T * T;
                matmul(at_h, ln1, wr_h, T, E, T);
                causal_mask_softmax(at_h, T);
                matmul(head_buf, at_h, v_h, T, T, D);

            } else if (c->mode == MODE_CONTENT) {
                /* Content: attn = (q @ k^T) / sqrt(d) */
                float *wq_h = head_wq(w, c, b, h);
                float *wk_h = head_wk(w, c, b, h);
                float *q_h = a->q + (b*H+h) * T * D;
                float *k_h = a->k + (b*H+h) * T * D;
                matmul(q_h, ln1, wq_h, T, E, D);
                matmul(k_h, ln1, wk_h, T, E, D);
                /* We reuse attn_r for content mode's attention storage */
                float *at_h = calloc(T * T, sizeof(float));
                matmul_abt(at_h, q_h, k_h, T, D, T);
                for (int i = 0; i < T*T; i++) at_h[i] *= scale;
                causal_mask_softmax(at_h, T);
                matmul(head_buf, at_h, v_h, T, T, D);
                free(at_h);

            } else { /* MODE_HYBRID */
                /* Both paths */
                float *wr_h = head_wr(w, c, b, h);
                float *wq_h = head_wq(w, c, b, h);
                float *wk_h = head_wk(w, c, b, h);
                float *q_h = a->q + (b*H+h) * T * D;
                float *k_h = a->k + (b*H+h) * T * D;
                float *at_r = a->attn_r + (b*H+h) * T * T;
                float *at_c = a->attn_c + (b*H+h) * T * T;
                float alpha = *head_gate(w, c, b, h);

                /* RRPRAM path */
                matmul(at_r, ln1, wr_h, T, E, T);
                causal_mask_softmax(at_r, T);
                float *out_r = calloc(T * D, sizeof(float));
                matmul(out_r, at_r, v_h, T, T, D);

                /* Content path */
                matmul(q_h, ln1, wq_h, T, E, D);
                matmul(k_h, ln1, wk_h, T, E, D);
                matmul_abt(at_c, q_h, k_h, T, D, T);
                for (int i = 0; i < T*T; i++) at_c[i] *= scale;
                causal_mask_softmax(at_c, T);
                float *out_c = calloc(T * D, sizeof(float));
                matmul(out_c, at_c, v_h, T, T, D);

                /* gated combination */
                for (int i = 0; i < T*D; i++)
                    head_buf[i] = alpha * out_r[i] + (1.0f - alpha) * out_c[i];

                free(out_r);
                free(out_c);
            }

            /* scatter into concat */
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    cat[t*E + h*D + d] = head_buf[t*D + d];
        }

        /* output projection + residual */
        float *wo_buf = calloc(T * E, sizeof(float));
        matmul(wo_buf, cat, w->wo[b], T, E, E);
        for (int i = 0; i < T*E; i++) r1[i] = xin[i] + wo_buf[i];
        free(wo_buf);

        /* LN2 + MLP */
        layernorm_fwd(ln2, r1, w->ln2_g[b], w->ln2_b[b], T, E);
        matmul(mpre, ln2, w->w1[b], T, E, M);
        for (int t = 0; t < T; t++)
            for (int j = 0; j < M; j++)
                mpre[t*M+j] += w->b1[b][j];

        float *mlp_act = calloc(T * M, sizeof(float));
        for (int i = 0; i < T*M; i++) mlp_act[i] = gelu_f(mpre[i]);

        float *mlp_out = calloc(T * E, sizeof(float));
        matmul(mlp_out, mlp_act, w->w2[b], T, M, E);
        for (int t = 0; t < T; t++)
            for (int e = 0; e < E; e++)
                xout[t*E+e] = r1[t*E+e] + mlp_out[t*E+e] + w->b2[b][e];

        free(mlp_act);
        free(mlp_out);
    }

    /* final LN + output projection */
    float *xfinal = a->x + B * T * E;
    layernorm_fwd(a->fln, xfinal, w->fln_g, w->fln_b, T, E);
    matmul(a->logits, a->fln, w->out_w, T, E, VOCAB);

    /* loss */
    float loss = 0;
    for (int t = 0; t < T; t++) {
        memcpy(a->probs + t*VOCAB, a->logits + t*VOCAB, VOCAB * sizeof(float));
        row_softmax(a->probs + t*VOCAB, VOCAB);
        if (targets) {
            float p = a->probs[t*VOCAB + targets[t]];
            loss -= logf(p > 1e-10f ? p : 1e-10f);
        }
    }
    free(head_buf);
    return targets ? loss / T : 0;
}

/* ===== backward ===== */

static void backward(Model *m, Acts *a, const int *tokens, const int *targets) {
    Cfg *c = &m->cfg;
    int T = c->T, E = c->E, H = c->H, D = c->D, B = c->B, M = c->M;
    Ptrs *w = &m->w, *g = &m->g;
    float scale = 1.0f / sqrtf((float)D);

    memset(m->grad, 0, m->n_params * sizeof(float));

    float *dx = calloc(T * E, sizeof(float));
    float *d_ln = calloc(T * E, sizeof(float));
    float *d_cat = calloc(T * E, sizeof(float));
    float *d_head = calloc(T * D, sizeof(float));
    float *d_pat = calloc(T * T, sizeof(float));
    float *d_raw = calloc(T * T, sizeof(float));
    float *d_v = calloc(T * D, sizeof(float));
    float *d_act = calloc(T * M, sizeof(float));
    float *d_pre = calloc(T * M, sizeof(float));

    /* d_logits */
    float *d_logits = calloc(T * VOCAB, sizeof(float));
    for (int t = 0; t < T; t++)
        for (int v = 0; v < VOCAB; v++)
            d_logits[t*VOCAB+v] = (a->probs[t*VOCAB+v] - (v == targets[t] ? 1.0f : 0.0f)) / T;

    /* output projection backward */
    float *d_fln = calloc(T * E, sizeof(float));
    matmul_abt(d_fln, d_logits, w->out_w, T, VOCAB, E);
    matmul_atb(g->out_w, a->fln, d_logits, E, T, VOCAB);

    /* final LN backward */
    float *xfinal = a->x + B * T * E;
    layernorm_bwd(dx, g->fln_g, g->fln_b, d_fln, xfinal, w->fln_g, T, E);

    for (int b = B-1; b >= 0; b--) {
        float *xin  = a->x + b * T * E;
        float *ln1  = a->ln1 + b * T * E;
        float *cat  = a->concat + b * T * E;
        float *r1   = a->r1 + b * T * E;
        float *ln2  = a->ln2 + b * T * E;
        float *mpre = a->mlp_pre + b * T * M;

        /* MLP backward */
        for (int e = 0; e < E; e++) {
            float s = 0;
            for (int t = 0; t < T; t++) s += dx[t*E+e];
            g->b2[b][e] += s;
        }
        matmul_abt(d_act, dx, w->w2[b], T, E, M);

        float *mlp_act = calloc(T * M, sizeof(float));
        for (int i = 0; i < T*M; i++) mlp_act[i] = gelu_f(mpre[i]);
        matmul_atb(g->w2[b], mlp_act, dx, M, T, E);

        for (int i = 0; i < T*M; i++)
            d_pre[i] = d_act[i] * gelu_grad(mpre[i]);

        for (int j = 0; j < M; j++) {
            float s = 0;
            for (int t = 0; t < T; t++) s += d_pre[t*M+j];
            g->b1[b][j] += s;
        }
        matmul_atb(g->w1[b], ln2, d_pre, E, T, M);
        matmul_abt(d_ln, d_pre, w->w1[b], T, M, E);
        free(mlp_act);

        float *d_r1 = calloc(T * E, sizeof(float));
        layernorm_bwd(d_r1, g->ln2_g[b], g->ln2_b[b], d_ln, r1, w->ln2_g[b], T, E);
        for (int i = 0; i < T*E; i++) d_r1[i] += dx[i];

        /* output projection backward */
        matmul_abt(d_cat, d_r1, w->wo[b], T, E, E);
        matmul_atb(g->wo[b], cat, d_r1, E, T, E);

        /* attention backward */
        memset(d_ln, 0, T * E * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *wv_h = head_wv(w, c, b, h);
            float *v_h  = a->v + (b*H+h) * T * D;
            float *g_wv = head_wv(g, c, b, h);

            /* extract d_head */
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    d_head[t*D+d] = d_cat[t*E + h*D + d];

            if (c->mode == MODE_RRPRAM) {
                float *wr_h = head_wr(w, c, b, h);
                float *at_h = a->attn_r + (b*H+h) * T * T;
                float *g_wr = head_wr(g, c, b, h);

                /* d_pat = d_head @ v^T */
                matmul_abt(d_pat, d_head, v_h, T, D, T);
                /* d_v = attn^T @ d_head */
                matmul_atb(d_v, at_h, d_head, T, T, D);

                /* softmax backward */
                for (int i = 0; i < T; i++) {
                    float dot = 0;
                    for (int j = 0; j <= i; j++) dot += at_h[i*T+j] * d_pat[i*T+j];
                    for (int j = 0; j <= i; j++)
                        d_raw[i*T+j] = at_h[i*T+j] * (d_pat[i*T+j] - dot);
                    for (int j = i+1; j < T; j++) d_raw[i*T+j] = 0;
                }

                matmul_atb(g_wr, ln1, d_raw, E, T, T);
                float *d_ln1_wr = calloc(T * E, sizeof(float));
                matmul_abt(d_ln1_wr, d_raw, wr_h, T, T, E);

                matmul_atb(g_wv, ln1, d_v, E, T, D);
                float *d_ln1_wv = calloc(T * E, sizeof(float));
                matmul_abt(d_ln1_wv, d_v, wv_h, T, D, E);

                for (int i = 0; i < T*E; i++) d_ln[i] += d_ln1_wr[i] + d_ln1_wv[i];
                free(d_ln1_wr); free(d_ln1_wv);

            } else if (c->mode == MODE_CONTENT) {
                float *wq_h = head_wq(w, c, b, h);
                float *wk_h = head_wk(w, c, b, h);
                float *q_h = a->q + (b*H+h) * T * D;
                float *k_h = a->k + (b*H+h) * T * D;
                float *g_wq = head_wq(g, c, b, h);
                float *g_wk = head_wk(g, c, b, h);

                /* recompute attention for backward */
                float *at_h = calloc(T * T, sizeof(float));
                matmul_abt(at_h, q_h, k_h, T, D, T);
                for (int i = 0; i < T*T; i++) at_h[i] *= scale;
                causal_mask_softmax(at_h, T);

                matmul_abt(d_pat, d_head, v_h, T, D, T);
                matmul_atb(d_v, at_h, d_head, T, T, D);

                /* softmax backward */
                for (int i = 0; i < T; i++) {
                    float dot = 0;
                    for (int j = 0; j <= i; j++) dot += at_h[i*T+j] * d_pat[i*T+j];
                    for (int j = 0; j <= i; j++)
                        d_raw[i*T+j] = at_h[i*T+j] * (d_pat[i*T+j] - dot);
                    for (int j = i+1; j < T; j++) d_raw[i*T+j] = 0;
                }

                /* scale backward */
                for (int i = 0; i < T*T; i++) d_raw[i] *= scale;

                /* QK^T backward: d_q = d_raw @ k, d_k = d_raw^T @ q */
                float *d_q = calloc(T * D, sizeof(float));
                float *d_k = calloc(T * D, sizeof(float));
                matmul(d_q, d_raw, k_h, T, T, D);
                matmul_atb(d_k, d_raw, q_h, T, T, D);

                /* weight gradients */
                matmul_atb(g_wq, ln1, d_q, E, T, D);
                matmul_atb(g_wk, ln1, d_k, E, T, D);
                matmul_atb(g_wv, ln1, d_v, E, T, D);

                /* d_ln1 */
                float *d_ln1_q = calloc(T * E, sizeof(float));
                float *d_ln1_k = calloc(T * E, sizeof(float));
                float *d_ln1_v = calloc(T * E, sizeof(float));
                matmul_abt(d_ln1_q, d_q, wq_h, T, D, E);
                matmul_abt(d_ln1_k, d_k, wk_h, T, D, E);
                matmul_abt(d_ln1_v, d_v, wv_h, T, D, E);

                for (int i = 0; i < T*E; i++)
                    d_ln[i] += d_ln1_q[i] + d_ln1_k[i] + d_ln1_v[i];

                free(at_h); free(d_q); free(d_k);
                free(d_ln1_q); free(d_ln1_k); free(d_ln1_v);

            } else { /* MODE_HYBRID */
                float alpha = *head_gate(w, c, b, h);
                float *wr_h = head_wr(w, c, b, h);
                float *wq_h = head_wq(w, c, b, h);
                float *wk_h = head_wk(w, c, b, h);
                float *q_h = a->q + (b*H+h) * T * D;
                float *k_h = a->k + (b*H+h) * T * D;
                float *at_r = a->attn_r + (b*H+h) * T * T;
                float *at_c = a->attn_c + (b*H+h) * T * T;
                float *g_wr = head_wr(g, c, b, h);
                float *g_wq = head_wq(g, c, b, h);
                float *g_wk = head_wk(g, c, b, h);

                /* recompute head outputs for gate gradient */
                float *out_r = calloc(T * D, sizeof(float));
                float *out_c = calloc(T * D, sizeof(float));
                matmul(out_r, at_r, v_h, T, T, D);
                matmul(out_c, at_c, v_h, T, T, D);

                /* gate gradient: d_gate = sum(d_head * (out_r - out_c)) */
                float d_gate = 0;
                for (int i = 0; i < T*D; i++)
                    d_gate += d_head[i] * (out_r[i] - out_c[i]);
                *head_gate(g, c, b, h) += d_gate;

                /* d_head splits: d_out_r = alpha * d_head, d_out_c = (1-alpha) * d_head */
                float *d_out_r = calloc(T * D, sizeof(float));
                float *d_out_c = calloc(T * D, sizeof(float));
                for (int i = 0; i < T*D; i++) {
                    d_out_r[i] = alpha * d_head[i];
                    d_out_c[i] = (1.0f - alpha) * d_head[i];
                }

                /* --- RRPRAM path backward --- */
                matmul_abt(d_pat, d_out_r, v_h, T, D, T);
                float *d_v_r = calloc(T * D, sizeof(float));
                matmul_atb(d_v_r, at_r, d_out_r, T, T, D);

                for (int i = 0; i < T; i++) {
                    float dot = 0;
                    for (int j = 0; j <= i; j++) dot += at_r[i*T+j] * d_pat[i*T+j];
                    for (int j = 0; j <= i; j++)
                        d_raw[i*T+j] = at_r[i*T+j] * (d_pat[i*T+j] - dot);
                    for (int j = i+1; j < T; j++) d_raw[i*T+j] = 0;
                }
                matmul_atb(g_wr, ln1, d_raw, E, T, T);

                float *d_ln1_wr = calloc(T * E, sizeof(float));
                matmul_abt(d_ln1_wr, d_raw, wr_h, T, T, E);

                /* --- Content path backward --- */
                float *d_pat_c = calloc(T * T, sizeof(float));
                float *d_raw_c = calloc(T * T, sizeof(float));
                matmul_abt(d_pat_c, d_out_c, v_h, T, D, T);
                float *d_v_c = calloc(T * D, sizeof(float));
                matmul_atb(d_v_c, at_c, d_out_c, T, T, D);

                for (int i = 0; i < T; i++) {
                    float dot = 0;
                    for (int j = 0; j <= i; j++) dot += at_c[i*T+j] * d_pat_c[i*T+j];
                    for (int j = 0; j <= i; j++)
                        d_raw_c[i*T+j] = at_c[i*T+j] * (d_pat_c[i*T+j] - dot);
                    for (int j = i+1; j < T; j++) d_raw_c[i*T+j] = 0;
                }
                for (int i = 0; i < T*T; i++) d_raw_c[i] *= scale;

                float *d_q = calloc(T * D, sizeof(float));
                float *d_k = calloc(T * D, sizeof(float));
                matmul(d_q, d_raw_c, k_h, T, T, D);
                matmul_atb(d_k, d_raw_c, q_h, T, T, D);

                matmul_atb(g_wq, ln1, d_q, E, T, D);
                matmul_atb(g_wk, ln1, d_k, E, T, D);

                /* combined d_v from both paths */
                for (int i = 0; i < T*D; i++) d_v[i] = d_v_r[i] + d_v_c[i];
                matmul_atb(g_wv, ln1, d_v, E, T, D);

                /* d_ln1 from all paths */
                float *d_ln1_q = calloc(T * E, sizeof(float));
                float *d_ln1_k = calloc(T * E, sizeof(float));
                float *d_ln1_v = calloc(T * E, sizeof(float));
                matmul_abt(d_ln1_q, d_q, wq_h, T, D, E);
                matmul_abt(d_ln1_k, d_k, wk_h, T, D, E);
                matmul_abt(d_ln1_v, d_v, wv_h, T, D, E);

                for (int i = 0; i < T*E; i++)
                    d_ln[i] += d_ln1_wr[i] + d_ln1_q[i] + d_ln1_k[i] + d_ln1_v[i];

                free(out_r); free(out_c); free(d_out_r); free(d_out_c);
                free(d_v_r); free(d_v_c); free(d_ln1_wr);
                free(d_pat_c); free(d_raw_c); free(d_q); free(d_k);
                free(d_ln1_q); free(d_ln1_k); free(d_ln1_v);
            }
        }

        /* LN1 backward */
        float *d_xin = calloc(T * E, sizeof(float));
        layernorm_bwd(d_xin, g->ln1_g[b], g->ln1_b[b], d_ln, xin, w->ln1_g[b], T, E);

        for (int i = 0; i < T*E; i++) dx[i] = d_r1[i] + d_xin[i];
        free(d_r1);
        free(d_xin);
    }

    /* embedding backward */
    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++) {
            g->tok_emb[tokens[t]*E+e] += dx[t*E+e];
            g->pos_emb[t*E+e] += dx[t*E+e];
        }

    free(dx); free(d_ln); free(d_cat); free(d_head);
    free(d_pat); free(d_raw); free(d_v);
    free(d_act); free(d_pre); free(d_logits); free(d_fln);
}

/* ===== adam ===== */

static void adam_update(Model *m, float lr, int step) {
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, step);
    float bc2 = 1.0f - powf(b2, step);
    for (int i = 0; i < m->n_params; i++) {
        float g = m->grad[i];
        m->adam_m[i] = b1 * m->adam_m[i] + (1-b1) * g;
        m->adam_v[i] = b2 * m->adam_v[i] + (1-b2) * g * g;
        float mhat = m->adam_m[i] / bc1;
        float vhat = m->adam_v[i] / bc2;
        m->data[i] -= lr * mhat / (sqrtf(vhat) + eps);
    }
}

/* ===== data ===== */

typedef struct {
    unsigned char *bytes;
    int len;
} Data;

static Data load_data(const char *path) {
    Data d = {0};
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    d.len = ftell(f);
    fseek(f, 0, SEEK_SET);
    d.bytes = malloc(d.len);
    fread(d.bytes, 1, d.len, f);
    fclose(f);
    printf("[haze] loaded %s: %d bytes (%.1fKB)\n", path, d.len, d.len/1024.0f);
    return d;
}

/* ===== training ===== */

static void train(Model *m, Data *data, int max_steps, float lr) {
    Acts a;
    acts_alloc(&a, &m->cfg);
    int T = m->cfg.T;
    int *tokens = malloc(T * sizeof(int));
    int *targets = malloc(T * sizeof(int));

    const char *mode_str[] = {"rrpram", "content", "hybrid"};
    printf("[haze] training (%s): %d steps, lr=%.1e\n", mode_str[m->cfg.mode], max_steps, lr);
    clock_t t0 = clock();

    for (int step = 1; step <= max_steps; step++) {
        int offset = rand() % (data->len - T - 1);
        for (int t = 0; t < T; t++) {
            tokens[t]  = data->bytes[offset + t];
            targets[t] = data->bytes[offset + t + 1];
        }

        float loss = forward(m, &a, tokens, targets);
        backward(m, &a, tokens, targets);
        adam_update(m, lr, step);

        if (step % 100 == 0 || step == 1) {
            float elapsed = (float)(clock() - t0) / CLOCKS_PER_SEC;
            float steps_sec = step / elapsed;
            printf("  step %5d/%d  loss=%.4f  %.1f steps/s\n",
                   step, max_steps, loss, steps_sec);
        }
    }

    acts_free(&a);
    free(tokens);
    free(targets);
}

/* ===== generation ===== */

static void generate(Model *m, const char *seed, int n_chars, Sampling samp,
                      float temperature, int top_k_val, float top_p_val,
                      float target_entropy) {
    Acts a;
    acts_alloc(&a, &m->cfg);
    int T = m->cfg.T;
    int *ctx = calloc(T, sizeof(int));
    float mirostat_mu = target_entropy * 2.0f;

    int seed_len = strlen(seed);
    if (seed_len > T) seed_len = T;
    for (int i = 0; i < seed_len; i++)
        ctx[T - seed_len + i] = (unsigned char)seed[i];

    printf("%s", seed);
    float total_entropy = 0;
    for (int i = 0; i < n_chars; i++) {
        forward(m, &a, ctx, NULL);

        float logits[VOCAB];
        memcpy(logits, a.logits + (T-1) * VOCAB, VOCAB * sizeof(float));

        /* track entropy */
        float probs_tmp[VOCAB];
        memcpy(probs_tmp, logits, VOCAB * sizeof(float));
        row_softmax(probs_tmp, VOCAB);
        total_entropy += entropy_bits(probs_tmp, VOCAB);

        int next;
        switch (samp) {
            case SAMP_TOP_K:
                next = sample_top_k(logits, VOCAB, top_k_val, temperature);
                break;
            case SAMP_TOP_P:
                next = sample_top_p(logits, VOCAB, top_p_val, temperature);
                break;
            case SAMP_ENTROPY:
                next = sample_entropy(logits, VOCAB, target_entropy, top_p_val, 0.3f, 2.0f);
                break;
            case SAMP_MIROSTAT:
                next = sample_mirostat(logits, VOCAB, target_entropy, 0.1f, &mirostat_mu);
                break;
            default:
                next = sample_basic(logits, VOCAB, temperature);
                break;
        }

        putchar(next);
        memmove(ctx, ctx+1, (T-1) * sizeof(int));
        ctx[T-1] = next;
    }
    printf("\n\n[haze] avg entropy: %.2f bits\n", total_entropy / n_chars);

    acts_free(&a);
    free(ctx);
}

/* ===== save/load ===== */

static void model_save(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    fwrite(&m->cfg, sizeof(Cfg), 1, f);
    fwrite(m->data, sizeof(float), m->n_params, f);
    fclose(f);
    printf("[haze] saved %s (%d params, %.1fKB)\n",
           path, m->n_params, m->n_params * 4.0f / 1024);
}

static void model_load(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    Cfg loaded_cfg;
    fread(&loaded_cfg, sizeof(Cfg), 1, f);

    int depth = loaded_cfg.B;
    model_init(m, depth, loaded_cfg.mode);
    m->cfg = loaded_cfg;
    m->n_params = model_size(&m->cfg);
    assign_ptrs(&m->w, m->data, &m->cfg);
    assign_ptrs(&m->g, m->grad, &m->cfg);

    fread(m->data, sizeof(float), m->n_params, f);
    fclose(f);
    printf("[haze] loaded %s\n", path);
}

/* ===== main ===== */

static HeadMode parse_mode(const char *s) {
    if (!strcmp(s, "rrpram") || !strcmp(s, "reweight")) return MODE_RRPRAM;
    if (!strcmp(s, "content")) return MODE_CONTENT;
    if (!strcmp(s, "hybrid")) return MODE_HYBRID;
    fprintf(stderr, "unknown mode: %s (use rrpram/content/hybrid)\n", s);
    exit(1);
}

static Sampling parse_sampling(const char *s) {
    if (!strcmp(s, "basic"))    return SAMP_BASIC;
    if (!strcmp(s, "top_k"))    return SAMP_TOP_K;
    if (!strcmp(s, "top_p"))    return SAMP_TOP_P;
    if (!strcmp(s, "entropy"))  return SAMP_ENTROPY;
    if (!strcmp(s, "mirostat")) return SAMP_MIROSTAT;
    fprintf(stderr, "unknown sampling: %s\n", s);
    exit(1);
}

int main(int argc, char **argv) {
    int depth = 4;
    int max_steps = 5000;
    float lr = 3e-4f;
    float temperature = 0.8f;
    int gen_chars = 500;
    int top_k_val = 40;
    float top_p_val = 0.9f;
    float target_entropy = 3.0f;
    HeadMode mode = MODE_HYBRID;
    Sampling samp = SAMP_ENTROPY;
    const char *train_file = NULL;
    const char *load_file = NULL;
    const char *save_file = "haze.bin";
    const char *seed = "The ";
    int do_generate = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--train") && i+1 < argc) train_file = argv[++i];
        else if (!strcmp(argv[i], "--load") && i+1 < argc) load_file = argv[++i];
        else if (!strcmp(argv[i], "--save") && i+1 < argc) save_file = argv[++i];
        else if (!strcmp(argv[i], "--depth") && i+1 < argc) depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) max_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc) lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc) temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "--chars") && i+1 < argc) gen_chars = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i+1 < argc) mode = parse_mode(argv[++i]);
        else if (!strcmp(argv[i], "--sampling") && i+1 < argc) samp = parse_sampling(argv[++i]);
        else if (!strcmp(argv[i], "--top_k") && i+1 < argc) top_k_val = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top_p") && i+1 < argc) top_p_val = atof(argv[++i]);
        else if (!strcmp(argv[i], "--target_entropy") && i+1 < argc) target_entropy = atof(argv[++i]);
        else if (!strcmp(argv[i], "--generate")) do_generate = 1;
        else { fprintf(stderr, "unknown: %s\n", argv[i]); return 1; }
    }

    srand(time(NULL));
    Model m = {0};

    if (load_file) {
        model_load(&m, load_file);
    } else {
        model_init(&m, depth, mode);
    }

    if (train_file) {
        Data data = load_data(train_file);
        train(&m, &data, max_steps, lr);
        model_save(&m, save_file);
        free(data.bytes);
        printf("\n--- sample (entropy) ---\n");
        generate(&m, seed, gen_chars, SAMP_ENTROPY, temperature, top_k_val, top_p_val, target_entropy);
    }

    if (do_generate) {
        generate(&m, seed, gen_chars, samp, temperature, top_k_val, top_p_val, target_entropy);
    }

    free(m.data); free(m.grad); free(m.adam_m); free(m.adam_v);
    return 0;
}
