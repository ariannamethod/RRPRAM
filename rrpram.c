/*
 * rrpram.c — RRPRAM Transformer
 * Janus Architecture  
 * Recursive Resonant Pattern Recognition Attention Mechanism
 *
 * Standard attention: attn[i,j] = (x@wq)_i · (x@wk)_j  (bilinear, semantic)
 * RRPRAM attention:   attn[i,j] = x_i · wr[:,j]         (linear, positional)
 *
 * RRPRAM sees PATTERNS, not MEANING.
 * Like a child who recognizes rhythm before understanding words.
 *
 * Character-level transformer. Train + generate. Zero dependencies.
 *   cc rrpram.c -O2 -lm -o rrpram
 *   ./rrpram --train data.txt --depth 4
 *   ./rrpram --generate --load rrpram.bin
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

typedef struct {
    int T;  /* context length */
    int E;  /* embedding dim */
    int H;  /* number of heads */
    int D;  /* head dim = E/H */
    int B;  /* number of blocks */
    int M;  /* MLP hidden dim */
} Cfg;

static Cfg cfg_from_depth(int depth) {
    Cfg c;
    c.T = (depth >= 8) ? 64 : 32;
    c.E = depth * 16;
    c.H = (depth < 4) ? 2 : 4;
    c.D = c.E / c.H;
    c.B = depth;
    c.M = c.E * 2;
    return c;
}

/* ===== math ===== */

static void matmul(float *C, const float *A, const float *B, int m, int k, int n) {
    /* C[m,n] = A[m,k] @ B[k,n] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++)
                s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
    }
}

static void matmul_atb(float *C, const float *A, const float *B, int m, int k, int n) {
    /* C[m,n] = A^T[m,k] @ B[k,n]  where A is [k,m] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++)
                s += A[p*m+i] * B[p*n+j];
            C[i*n+j] = s;
        }
    }
}

static void matmul_abt(float *C, const float *A, const float *B, int m, int k, int n) {
    /* C[m,n] = A[m,k] @ B^T[n,k]  where B is [n,k] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++)
                s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
    }
}

static void row_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
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
    /* Box-Muller */
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

/* ===== model ===== */

typedef struct {
    float *tok_emb;   /* [V, E] */
    float *pos_emb;   /* [T, E] */
    float *ln1_g[MAX_BLK], *ln1_b[MAX_BLK];  /* [E] each */
    float *wv[MAX_BLK];    /* [H*E*D] = H heads of [E,D] */
    float *wr[MAX_BLK];    /* [H*E*T] = H heads of [E,T] — RRPRAM */
    float *wo[MAX_BLK];    /* [E, E] */
    float *ln2_g[MAX_BLK], *ln2_b[MAX_BLK];
    float *w1[MAX_BLK], *b1[MAX_BLK];  /* [E,M], [M] */
    float *w2[MAX_BLK], *b2[MAX_BLK];  /* [M,E], [E] */
    float *fln_g, *fln_b;              /* [E] */
    float *out_w;                       /* [E, V] */
} Ptrs;

typedef struct {
    Cfg cfg;
    int n_params;
    float *data, *grad, *adam_m, *adam_v;
    Ptrs w, g;  /* weight ptrs and grad ptrs */
} Model;

static int model_size(Cfg *c) {
    int s = VOCAB * c->E + c->T * c->E;  /* embeddings */
    for (int b = 0; b < c->B; b++) {
        s += 2 * c->E;                          /* ln1 */
        s += c->H * c->E * c->D;                /* wv */
        s += c->H * c->E * c->T;                /* wr (RRPRAM!) */
        s += c->E * c->E;                       /* wo */
        s += 2 * c->E;                          /* ln2 */
        s += c->E * c->M + c->M;                /* w1, b1 */
        s += c->M * c->E + c->E;                /* w2, b2 */
    }
    s += 2 * c->E;              /* final ln */
    s += c->E * VOCAB;          /* output proj */
    return s;
}

static void assign_ptrs(Ptrs *p, float *base, Cfg *c) {
    float *q = base;
    p->tok_emb = q; q += VOCAB * c->E;
    p->pos_emb = q; q += c->T * c->E;
    for (int b = 0; b < c->B; b++) {
        p->ln1_g[b] = q; q += c->E;
        p->ln1_b[b] = q; q += c->E;
        p->wv[b] = q; q += c->H * c->E * c->D;
        p->wr[b] = q; q += c->H * c->E * c->T;
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
    p->out_w = q; q += c->E * VOCAB;
}

static void model_init(Model *m, int depth) {
    m->cfg = cfg_from_depth(depth);
    m->n_params = model_size(&m->cfg);
    m->data   = calloc(m->n_params, sizeof(float));
    m->grad   = calloc(m->n_params, sizeof(float));
    m->adam_m  = calloc(m->n_params, sizeof(float));
    m->adam_v  = calloc(m->n_params, sizeof(float));
    assign_ptrs(&m->w, m->data, &m->cfg);
    assign_ptrs(&m->g, m->grad, &m->cfg);

    Cfg *c = &m->cfg;
    /* Xavier init for projections */
    float scale;
    scale = sqrtf(2.0f / VOCAB);
    for (int i = 0; i < VOCAB * c->E; i++) m->w.tok_emb[i] = randn() * scale;
    scale = sqrtf(2.0f / c->T);
    for (int i = 0; i < c->T * c->E; i++) m->w.pos_emb[i] = randn() * scale;

    for (int b = 0; b < c->B; b++) {
        for (int e = 0; e < c->E; e++) { m->w.ln1_g[b][e] = 1.0f; m->w.ln2_g[b][e] = 1.0f; }
        scale = sqrtf(2.0f / c->E);
        for (int i = 0; i < c->H * c->E * c->D; i++) m->w.wv[b][i] = randn() * scale;
        for (int i = 0; i < c->H * c->E * c->T; i++) m->w.wr[b][i] = randn() * scale;
        for (int i = 0; i < c->E * c->E; i++) m->w.wo[b][i] = randn() * scale / sqrtf(c->B);
        for (int i = 0; i < c->E * c->M; i++) m->w.w1[b][i] = randn() * scale;
        scale = sqrtf(2.0f / c->M);
        for (int i = 0; i < c->M * c->E; i++) m->w.w2[b][i] = randn() * scale / sqrtf(c->B);
    }
    for (int e = 0; e < c->E; e++) m->w.fln_g[e] = 1.0f;
    scale = sqrtf(2.0f / c->E);
    for (int i = 0; i < c->E * VOCAB; i++) m->w.out_w[i] = randn() * scale;

    printf("[rrpram] depth=%d params=%d (%.1fK)\n", depth, m->n_params, m->n_params/1000.0f);
    printf("[rrpram] T=%d E=%d H=%d D=%d B=%d M=%d\n", c->T, c->E, c->H, c->D, c->B, c->M);
}

/* ===== activation cache ===== */

typedef struct {
    float *x;        /* [(B+1)*T*E] block inputs + final */
    float *ln1;      /* [B*T*E] */
    float *v;        /* [B*H*T*D] */
    float *attn;     /* [B*H*T*T] */
    float *concat;   /* [B*T*E] */
    float *r1;       /* [B*T*E]  first residual */
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
    a->attn    = calloc(c->B * c->H * TT, sizeof(float));
    a->concat  = calloc(c->B * TE, sizeof(float));
    a->r1      = calloc(c->B * TE, sizeof(float));
    a->ln2     = calloc(c->B * TE, sizeof(float));
    a->mlp_pre = calloc(c->B * TM, sizeof(float));
    a->fln     = calloc(TE, sizeof(float));
    a->logits  = calloc(c->T * VOCAB, sizeof(float));
    a->probs   = calloc(c->T * VOCAB, sizeof(float));
}

static void acts_free(Acts *a) {
    free(a->x); free(a->ln1); free(a->v); free(a->attn);
    free(a->concat); free(a->r1); free(a->ln2); free(a->mlp_pre);
    free(a->fln); free(a->logits); free(a->probs);
}

/* ===== forward ===== */

static float forward(Model *m, Acts *a, const int *tokens, const int *targets) {
    Cfg *c = &m->cfg;
    int T = c->T, E = c->E, H = c->H, D = c->D, B = c->B, M = c->M;
    Ptrs *w = &m->w;
    float *head_buf = (float *)calloc(T * D, sizeof(float));

    /* embedding */
    float *x0 = a->x;  /* first block input */
    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++)
            x0[t*E+e] = w->tok_emb[tokens[t]*E+e] + w->pos_emb[t*E+e];

    /* transformer blocks */
    for (int b = 0; b < B; b++) {
        float *xin  = a->x + b * T * E;
        float *xout = a->x + (b+1) * T * E;
        float *ln1  = a->ln1 + b * T * E;
        float *cat  = a->concat + b * T * E;
        float *r1   = a->r1 + b * T * E;
        float *ln2  = a->ln2 + b * T * E;
        float *mpre = a->mlp_pre + b * T * M;

        /* LN1 */
        layernorm_fwd(ln1, xin, w->ln1_g[b], w->ln1_b[b], T, E);

        /* multi-head RRPRAM attention */
        memset(cat, 0, T * E * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *wv_h = w->wv[b] + h * E * D;
            float *wr_h = w->wr[b] + h * E * T;
            float *v_h  = a->v + (b*H+h) * T * D;
            float *at_h = a->attn + (b*H+h) * T * T;

            /* v = ln1 @ wv_h  [T,D] */
            matmul(v_h, ln1, wv_h, T, E, D);

            /* raw = ln1 @ wr_h  [T,T]  ← THE RRPRAM INNOVATION */
            matmul(at_h, ln1, wr_h, T, E, T);

            /* causal mask + softmax */
            for (int i = 0; i < T; i++) {
                for (int j = i+1; j < T; j++) at_h[i*T+j] = -1e9f;
                row_softmax(at_h + i*T, T);
            }

            /* out = attn @ v  [T,D] */
            matmul(head_buf, at_h, v_h, T, T, D);

            /* scatter into concat */
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    cat[t*E + h*D + d] = head_buf[t*D + d];
        }

        /* output projection: r1 = xin + cat @ wo */
        float *wo_buf = (float *)calloc(T * E, sizeof(float));
        matmul(wo_buf, cat, w->wo[b], T, E, E);
        for (int i = 0; i < T*E; i++) r1[i] = xin[i] + wo_buf[i];
        free(wo_buf);

        /* LN2 */
        layernorm_fwd(ln2, r1, w->ln2_g[b], w->ln2_b[b], T, E);

        /* MLP: gelu(ln2 @ w1 + b1) @ w2 + b2 */
        matmul(mpre, ln2, w->w1[b], T, E, M);
        for (int t = 0; t < T; t++)
            for (int j = 0; j < M; j++)
                mpre[t*M+j] += w->b1[b][j];

        float *mlp_act = (float *)calloc(T * M, sizeof(float));
        for (int i = 0; i < T*M; i++) mlp_act[i] = gelu_f(mpre[i]);

        float *mlp_out = (float *)calloc(T * E, sizeof(float));
        matmul(mlp_out, mlp_act, w->w2[b], T, M, E);
        for (int t = 0; t < T; t++)
            for (int e = 0; e < E; e++)
                xout[t*E+e] = r1[t*E+e] + mlp_out[t*E+e] + w->b2[b][e];

        free(mlp_act);
        free(mlp_out);
    }

    /* final layer norm + output projection */
    float *xfinal = a->x + B * T * E;
    layernorm_fwd(a->fln, xfinal, w->fln_g, w->fln_b, T, E);
    matmul(a->logits, a->fln, w->out_w, T, E, VOCAB);

    /* softmax + cross-entropy loss */
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

    memset(m->grad, 0, m->n_params * sizeof(float));

    /* scratch buffers */
    float *dx = calloc(T * E, sizeof(float));  /* gradient flowing backward */
    float *d_ln = calloc(T * E, sizeof(float));
    float *d_cat = calloc(T * E, sizeof(float));
    float *d_head = calloc(T * D, sizeof(float));
    float *d_pat = calloc(T * T, sizeof(float));
    float *d_raw = calloc(T * T, sizeof(float));
    float *d_v = calloc(T * D, sizeof(float));
    float *d_act = calloc(T * M, sizeof(float));
    float *d_pre = calloc(T * M, sizeof(float));

    /* d_logits = (probs - onehot) / T */
    float *d_logits = calloc(T * VOCAB, sizeof(float));
    for (int t = 0; t < T; t++)
        for (int v = 0; v < VOCAB; v++)
            d_logits[t*VOCAB+v] = (a->probs[t*VOCAB+v] - (v == targets[t] ? 1.0f : 0.0f)) / T;

    /* output projection backward */
    /* d_fln = d_logits @ out_w^T  [T,V]@[V,E] = [T,E] */
    float *d_fln = calloc(T * E, sizeof(float));
    matmul_abt(d_fln, d_logits, w->out_w, T, VOCAB, E);
    /* d_out_w = fln^T @ d_logits  [E,T]@[T,V] = [E,V] */
    matmul_atb(g->out_w, a->fln, d_logits, E, T, VOCAB);

    /* final layernorm backward */
    float *xfinal = a->x + B * T * E;
    layernorm_bwd(dx, g->fln_g, g->fln_b, d_fln, xfinal, w->fln_g, T, E);

    /* blocks backward */
    for (int b = B-1; b >= 0; b--) {
        float *xin  = a->x + b * T * E;
        float *ln1  = a->ln1 + b * T * E;
        float *cat  = a->concat + b * T * E;
        float *r1   = a->r1 + b * T * E;
        float *ln2  = a->ln2 + b * T * E;
        float *mpre = a->mlp_pre + b * T * M;

        /* dx is d_x[b+1], gradient at output of this block */
        /* second residual: xout = r1 + mlp_out */
        /* d_r1 = dx, d_mlp_out = dx */

        /* MLP backward */
        /* mlp_out = gelu(mpre) @ w2 + b2 */
        /* d_b2 */
        for (int e = 0; e < E; e++) {
            float s = 0;
            for (int t = 0; t < T; t++) s += dx[t*E+e];
            g->b2[b][e] += s;
        }
        /* d_act = dx @ w2^T  [T,E]@[E,M] = [T,M] */
        matmul_abt(d_act, dx, w->w2[b], T, E, M);

        /* recompute mlp_act from mpre */
        float *mlp_act = calloc(T * M, sizeof(float));
        for (int i = 0; i < T*M; i++) mlp_act[i] = gelu_f(mpre[i]);

        /* d_w2 = act^T @ dx  [M,T]@[T,E] = [M,E] */
        matmul_atb(g->w2[b], mlp_act, dx, M, T, E);

        /* d_pre = d_act * gelu'(mpre) */
        for (int i = 0; i < T*M; i++)
            d_pre[i] = d_act[i] * gelu_grad(mpre[i]);

        /* d_b1 */
        for (int j = 0; j < M; j++) {
            float s = 0;
            for (int t = 0; t < T; t++) s += d_pre[t*M+j];
            g->b1[b][j] += s;
        }

        /* d_w1 = ln2^T @ d_pre  [E,M] */
        matmul_atb(g->w1[b], ln2, d_pre, E, T, M);

        /* d_ln2 = d_pre @ w1^T  [T,E] */
        matmul_abt(d_ln, d_pre, w->w1[b], T, M, E);

        free(mlp_act);

        /* LN2 backward: gradient goes into d_r1_from_ln2 */
        float *d_r1 = calloc(T * E, sizeof(float));
        layernorm_bwd(d_r1, g->ln2_g[b], g->ln2_b[b], d_ln, r1, w->ln2_g[b], T, E);

        /* add residual gradient: d_r1 += dx */
        for (int i = 0; i < T*E; i++) d_r1[i] += dx[i];

        /* first residual: r1 = xin + cat @ wo */
        /* d_xin = d_r1, d_wo_out = d_r1 */

        /* output projection backward */
        /* d_cat = d_r1 @ wo^T  [T,E] */
        matmul_abt(d_cat, d_r1, w->wo[b], T, E, E);
        /* d_wo = cat^T @ d_r1  [E,E] */
        matmul_atb(g->wo[b], cat, d_r1, E, T, E);

        /* multi-head RRPRAM attention backward */
        memset(d_ln, 0, T * E * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *wv_h = w->wv[b] + h * E * D;
            float *wr_h = w->wr[b] + h * E * T;
            float *v_h  = a->v + (b*H+h) * T * D;
            float *at_h = a->attn + (b*H+h) * T * T;
            float *g_wv = g->wv[b] + h * E * D;
            float *g_wr = g->wr[b] + h * E * T;

            /* extract d_head from d_cat */
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    d_head[t*D+d] = d_cat[t*E + h*D + d];

            /* out = attn @ v, so: */
            /* d_pat = d_head @ v^T  [T,T] */
            matmul_abt(d_pat, d_head, v_h, T, D, T);
            /* d_v = attn^T @ d_head  [T,D] */
            matmul_atb(d_v, at_h, d_head, T, T, D);

            /* softmax backward (per row, causal) */
            for (int i = 0; i < T; i++) {
                float dot = 0;
                for (int j = 0; j <= i; j++) dot += at_h[i*T+j] * d_pat[i*T+j];
                for (int j = 0; j <= i; j++)
                    d_raw[i*T+j] = at_h[i*T+j] * (d_pat[i*T+j] - dot);
                for (int j = i+1; j < T; j++)
                    d_raw[i*T+j] = 0;
            }

            /* RRPRAM backward: raw = ln1 @ wr */
            /* d_wr += ln1^T @ d_raw  [E,T] */
            matmul_atb(g_wr, ln1, d_raw, E, T, T);
            /* Wait — need to ACCUMULATE, not overwrite */
            /* Redo: compute into scratch and add */
            {
                float *tmp = calloc(E * T, sizeof(float));
                matmul_atb(tmp, ln1, d_raw, E, T, T);
                for (int i = 0; i < E*T; i++) g_wr[i] += tmp[i];
                /* Oops, matmul_atb already wrote to g_wr. Need to fix. */
                /* Actually, assign_ptrs points g_wr into m->grad which was zeroed.
                   First head writes, subsequent heads need to accumulate.
                   Since we process heads sequentially and g_wr points to different
                   per-head slices, this is fine — each head has its own g_wr. */
                free(tmp);
            }
            /* Actually, each head h has its own g_wr slice, so overwrite is correct. */
            /* Re-do the matmul_atb to write directly: */
            matmul_atb(g_wr, ln1, d_raw, E, T, T);

            /* d_ln1 from wr path = d_raw @ wr^T  [T,E] */
            float *d_ln1_wr = calloc(T * E, sizeof(float));
            matmul_abt(d_ln1_wr, d_raw, wr_h, T, T, E);

            /* value backward: v = ln1 @ wv */
            /* d_wv = ln1^T @ d_v  [E,D] */
            matmul_atb(g_wv, ln1, d_v, E, T, D);

            /* d_ln1 from wv path = d_v @ wv^T  [T,E] */
            float *d_ln1_wv = calloc(T * E, sizeof(float));
            matmul_abt(d_ln1_wv, d_v, wv_h, T, D, E);

            /* accumulate into d_ln */
            for (int i = 0; i < T*E; i++)
                d_ln[i] += d_ln1_wr[i] + d_ln1_wv[i];

            free(d_ln1_wr);
            free(d_ln1_wv);
        }

        /* LN1 backward */
        float *d_xin = calloc(T * E, sizeof(float));
        layernorm_bwd(d_xin, g->ln1_g[b], g->ln1_b[b], d_ln, xin, w->ln1_g[b], T, E);

        /* residual: d_x[b] = d_r1 + d_xin */
        for (int i = 0; i < T*E; i++)
            dx[i] = d_r1[i] + d_xin[i];

        free(d_r1);
        free(d_xin);
    }

    /* embedding backward */
    for (int t = 0; t < T; t++) {
        for (int e = 0; e < E; e++) {
            g->tok_emb[tokens[t]*E+e] += dx[t*E+e];
            g->pos_emb[t*E+e] += dx[t*E+e];
        }
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
    printf("[rrpram] loaded %s: %d bytes (%.1fKB)\n", path, d.len, d.len/1024.0f);
    return d;
}

/* ===== training ===== */

static void train(Model *m, Data *data, int max_steps, float lr) {
    Acts a;
    acts_alloc(&a, &m->cfg);
    int T = m->cfg.T;
    int *tokens = malloc(T * sizeof(int));
    int *targets = malloc(T * sizeof(int));

    printf("[rrpram] training: %d steps, lr=%.1e\n", max_steps, lr);
    clock_t t0 = clock();

    for (int step = 1; step <= max_steps; step++) {
        /* random subsequence */
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

static void generate(Model *m, const char *seed, int n_chars, float temperature) {
    Acts a;
    acts_alloc(&a, &m->cfg);
    int T = m->cfg.T;
    int *ctx = calloc(T, sizeof(int));

    /* fill context with seed */
    int seed_len = strlen(seed);
    if (seed_len > T) seed_len = T;
    for (int i = 0; i < seed_len; i++)
        ctx[T - seed_len + i] = (unsigned char)seed[i];

    printf("%s", seed);
    for (int i = 0; i < n_chars; i++) {
        forward(m, &a, ctx, NULL);

        /* sample from last position */
        float *logits = a.logits + (T-1) * VOCAB;
        if (temperature != 1.0f)
            for (int v = 0; v < VOCAB; v++) logits[v] /= temperature;
        row_softmax(logits, VOCAB);

        /* sample */
        float r = (float)rand() / RAND_MAX;
        float cum = 0;
        int next = 0;
        for (int v = 0; v < VOCAB; v++) {
            cum += logits[v];
            if (cum >= r) { next = v; break; }
        }

        putchar(next);

        /* slide context */
        memmove(ctx, ctx+1, (T-1) * sizeof(int));
        ctx[T-1] = next;
    }
    printf("\n");

    acts_free(&a);
    free(ctx);
}

/* ===== save/load ===== */

static void model_save(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    fwrite(&m->cfg, sizeof(Cfg), 1, f);
    fwrite(m->data, sizeof(float), m->n_params, f);
    fclose(f);
    printf("[rrpram] saved %s (%d params, %.1fKB)\n",
           path, m->n_params, m->n_params * 4.0f / 1024);
}

static void model_load(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    Cfg loaded_cfg;
    fread(&loaded_cfg, sizeof(Cfg), 1, f);

    /* reinit with loaded config */
    int depth = loaded_cfg.B;
    model_init(m, depth);
    /* overwrite cfg in case T differs */
    m->cfg = loaded_cfg;
    m->n_params = model_size(&m->cfg);
    assign_ptrs(&m->w, m->data, &m->cfg);
    assign_ptrs(&m->g, m->grad, &m->cfg);

    fread(m->data, sizeof(float), m->n_params, f);
    fclose(f);
    printf("[rrpram] loaded %s\n", path);
}

/* ===== main ===== */

int main(int argc, char **argv) {
    int depth = 4;
    int max_steps = 5000;
    float lr = 3e-4f;
    float temperature = 0.8f;
    int gen_chars = 500;
    const char *train_file = NULL;
    const char *load_file = NULL;
    const char *save_file = "rrpram.bin";
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
        else if (!strcmp(argv[i], "--generate")) do_generate = 1;
        else { fprintf(stderr, "unknown: %s\n", argv[i]); return 1; }
    }

    srand(time(NULL));
    Model m = {0};

    if (load_file) {
        model_load(&m, load_file);
    } else {
        model_init(&m, depth);
    }

    if (train_file) {
        Data data = load_data(train_file);
        train(&m, &data, max_steps, lr);
        model_save(&m, save_file);
        free(data.bytes);
        /* generate sample after training */
        printf("\n--- sample ---\n");
        generate(&m, seed, gen_chars, temperature);
    }

    if (do_generate) {
        generate(&m, seed, gen_chars, temperature);
    }

    free(m.data); free(m.grad); free(m.adam_m); free(m.adam_v);
    return 0;
}
