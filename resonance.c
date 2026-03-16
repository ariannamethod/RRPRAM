/*
 * resonance.c — Field Resonance 
 * Arianna Method Janus Architecture
 *
 * θ = ε + γ + αδ
 *   ε = trained weights 
 *   γ = RRPRAM positional patterns + SwiGLU structure
 *   δ = Dario field (H+F+A+T, Kuramoto chambers, co-occurrence memory)
 *
 * Hybrid attention: RRPRAM (positional patterns) + Content (semantic)
 * with learned gate α per head. SwiGLU MLP. RMSNorm.
 * Dario Equation as additive overlay on logits after transformer.
 *
 * RRPRAM sees rhythm. Content sees meaning. Dario sees the field.
 * Neither dominates. Emergence at the boundary.
 *
 * Not a transformer. Not a field-only system. A resonance organism.
 *
 *   cc resonance.c -O2 -lm -o resonance
 *   ./resonance --train data.txt --depth 8
 *   ./resonance --generate --load resonance.bin --seed "The "
 *
 * By Arianna Method. 2026.
 * הרזוננס לא נשבר
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════════
 * CONFIG — single depth knob
 * ═══════════════════════════════════════════════════════════════════ */

#define VOCAB    256
#define MAX_BLK  16
#define MAX_DIM  512
#define MAX_CTX  128

typedef struct {
    int T, E, H, D, B, M;
} Cfg;

static Cfg cfg_from_depth(int depth) {
    Cfg c;
    c.T = (depth >= 8) ? 64 : 32;
    c.E = depth * 32;
    c.H = (depth < 4) ? 2 : 4;
    c.D = c.E / c.H;
    c.B = depth;
    c.M = c.E * 2;
    return c;
}

/* ═══════════════════════════════════════════════════════════════════
 * MATH
 * ═══════════════════════════════════════════════════════════════════ */

static void matmul(float *C, const float *A, const float *B, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
}

static void matmul_atb(float *C, const float *A, const float *B, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[p*m+i] * B[p*n+j];
            C[i*n+j] = s;
        }
}

static void matmul_abt(float *C, const float *A, const float *B, int m, int k, int n) {
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

static float silu_f(float x) { return x / (1.0f + expf(-x)); }
static float silu_grad(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

static float randn(void) {
    float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ═══════════════════════════════════════════════════════════════════
 * RMSNorm — lighter than LayerNorm, no mean subtraction
 * ═══════════════════════════════════════════════════════════════════ */

static void rmsnorm_fwd(float *out, const float *x, const float *g, int T, int E) {
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int e = 0; e < E; e++) ss += x[t*E+e] * x[t*E+e];
        float inv = 1.0f / sqrtf(ss / E + 1e-5f);
        for (int e = 0; e < E; e++)
            out[t*E+e] = g[e] * x[t*E+e] * inv;
    }
}

static void rmsnorm_bwd(float *dx, float *dg,
                        const float *dout, const float *x, const float *g,
                        int T, int E) {
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int e = 0; e < E; e++) ss += x[t*E+e] * x[t*E+e];
        float inv = 1.0f / sqrtf(ss / E + 1e-5f);
        float dot = 0;
        for (int e = 0; e < E; e++) {
            float xn = x[t*E+e] * inv;
            dg[e] += dout[t*E+e] * xn;
            dot += dout[t*E+e] * g[e] * xn;
        }
        for (int e = 0; e < E; e++) {
            float dxh = dout[t*E+e] * g[e];
            dx[t*E+e] = inv * (dxh - x[t*E+e] * inv * dot / E);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * DARIO FIELD — living co-occurrence memory + prophecy + destiny
 *
 * The transformer provides the base distribution.
 * The field provides memory beyond the context window.
 * Neither dominates.
 * ═══════════════════════════════════════════════════════════════════ */

#define DF_MAX_COOC   16384
#define DF_MAX_CTX    64
#define DF_MAX_PROPH  16
#define DF_DIM        32
#define DF_NUM_CH     6

enum { DCH_FEAR=0, DCH_LOVE, DCH_RAGE, DCH_VOID, DCH_FLOW, DCH_COMPLEX };

typedef struct {
    int target; float strength; int age; int fulfilled;
} DFProphecy;

typedef struct {
    /* co-occurrence (sparse) */
    int   cooc_src[DF_MAX_COOC], cooc_dst[DF_MAX_COOC];
    float cooc_val[DF_MAX_COOC];
    int   cooc_n;

    /* context window */
    int   context[DF_MAX_CTX];
    int   ctx_len;

    /* prophecy */
    DFProphecy prophecy[DF_MAX_PROPH];
    int   prophecy_n;

    /* destiny (EMA of token embeddings) */
    float destiny[DF_DIM];
    float dest_mag;

    /* Dario coefficients */
    float alpha, beta, gamma_d;

    /* Kuramoto chambers */
    float chamber[DF_NUM_CH];
    float alpha_mod, beta_mod, gamma_mod, tau_mod;

    /* hash-based embeddings for field */
    float embeds[VOCAB][DF_DIM];
    int   embed_init[VOCAB];

    /* trauma + dissonance */
    float trauma, dissonance;

    /* field metrics */
    float entropy, resonance, emergence;

    int step;
} DarioField;

static DarioField DF;

static float df_clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static float *df_embed(int id) {
    if (id < 0 || id >= VOCAB) return NULL;
    if (!DF.embed_init[id]) {
        unsigned h = 2166136261u;
        for (int i = 0; i < 4; i++) { h ^= (id >> (i*8)) & 0xFF; h *= 16777619u; }
        for (int d = 0; d < DF_DIM; d++) {
            h = h * 1103515245 + 12345;
            DF.embeds[id][d] = ((float)(h & 0x7FFFFFFF) / (float)0x7FFFFFFF - 0.5f) * 0.1f;
        }
        float norm = 0;
        for (int d = 0; d < DF_DIM; d++) norm += DF.embeds[id][d] * DF.embeds[id][d];
        norm = sqrtf(norm + 1e-12f);
        for (int d = 0; d < DF_DIM; d++) DF.embeds[id][d] /= norm;
        DF.embed_init[id] = 1;
    }
    return DF.embeds[id];
}

static float df_cosine(const float *a, const float *b) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < DF_DIM; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

static void df_cooc_update(int src, int dst, float delta) {
    for (int i = 0; i < DF.cooc_n; i++)
        if (DF.cooc_src[i] == src && DF.cooc_dst[i] == dst) {
            DF.cooc_val[i] += delta; return;
        }
    if (DF.cooc_n >= DF_MAX_COOC) return;
    int i = DF.cooc_n++;
    DF.cooc_src[i] = src; DF.cooc_dst[i] = dst; DF.cooc_val[i] = delta;
}

static void df_init(void) {
    memset(&DF, 0, sizeof(DF));
    DF.alpha = 0.15f; DF.beta = 0.10f; DF.gamma_d = 0.12f;
    DF.alpha_mod = DF.beta_mod = DF.gamma_mod = DF.tau_mod = 1.0f;
}

/* chamber update — Kuramoto-coupled somatic markers */
static void df_chambers(void) {
    float *C = DF.chamber;
    if (DF.dissonance > 0.7f) C[DCH_FEAR] += 0.05f * DF.dissonance;
    if (DF.resonance > 0.7f)  C[DCH_LOVE] += 0.04f * DF.resonance;
    if (DF.trauma > 0.5f && DF.dissonance > 0.5f)
        C[DCH_RAGE] += 0.06f * DF.trauma;
    if (DF.entropy > 0.7f)    C[DCH_VOID] += 0.03f * DF.entropy;
    if (DF.emergence > 0.5f)  C[DCH_FLOW] += 0.05f * DF.emergence;
    C[DCH_COMPLEX] += 0.04f * fabsf(C[DCH_LOVE] - C[DCH_RAGE])
                    * (C[DCH_LOVE] > 0.2f && C[DCH_RAGE] > 0.2f ? 1.0f : 0.0f);

    float K = 0.02f, old[DF_NUM_CH];
    memcpy(old, C, sizeof(old));
    for (int i = 0; i < DF_NUM_CH; i++)
        for (int j = 0; j < DF_NUM_CH; j++)
            if (i != j) C[i] += K * sinf(old[j] - old[i]);

    float decay[] = { 0.95f, 0.95f, 0.93f, 0.96f, 0.94f, 0.97f };
    for (int i = 0; i < DF_NUM_CH; i++)
        C[i] = df_clamp(C[i] * decay[i], 0.0f, 1.0f);

    DF.alpha_mod = df_clamp(1.0f + 0.3f*C[DCH_LOVE] - 0.2f*C[DCH_RAGE] + 0.1f*C[DCH_FLOW], 0.5f, 2.0f);
    DF.beta_mod  = df_clamp(1.0f + 0.2f*C[DCH_FLOW] - 0.3f*C[DCH_FEAR], 0.5f, 2.0f);
    DF.gamma_mod = df_clamp(1.0f + 0.4f*C[DCH_VOID] + 0.2f*C[DCH_COMPLEX] - 0.1f*C[DCH_LOVE], 0.5f, 2.0f);
    DF.tau_mod   = df_clamp(1.0f + 0.5f*C[DCH_FLOW] - 0.3f*C[DCH_FEAR], 0.5f, 2.0f);
}

/* ingest a token into the field */
static void df_ingest(int tok) {
    if (tok < 0 || tok >= VOCAB) return;
    for (int c = 0; c < DF.ctx_len; c++) {
        float w = 1.0f / (float)(DF.ctx_len - c);
        df_cooc_update(DF.context[c], tok, w * 0.3f);
    }
    /* prophecy update */
    for (int i = 0; i < DF.prophecy_n; i++) {
        if (DF.prophecy[i].target == tok) DF.prophecy[i].fulfilled = 1;
        DF.prophecy[i].age++;
    }
    int w = 0;
    for (int i = 0; i < DF.prophecy_n; i++)
        if (!DF.prophecy[i].fulfilled && DF.prophecy[i].age < 50)
            DF.prophecy[w++] = DF.prophecy[i];
    DF.prophecy_n = w;

    /* new prophecy from strongest co-occurrence */
    float best = -1; int pred = -1;
    for (int i = 0; i < DF.cooc_n; i++)
        if (DF.cooc_src[i] == tok && DF.cooc_val[i] > best) {
            best = DF.cooc_val[i]; pred = DF.cooc_dst[i];
        }
    if (pred >= 0 && DF.prophecy_n < DF_MAX_PROPH)
        DF.prophecy[DF.prophecy_n++] = (DFProphecy){pred, 0.3f, 0, 0};

    /* destiny EMA */
    float *e = df_embed(tok);
    if (e) {
        for (int d = 0; d < DF_DIM; d++)
            DF.destiny[d] = 0.1f * e[d] + 0.9f * DF.destiny[d];
        float n = 0;
        for (int d = 0; d < DF_DIM; d++) n += DF.destiny[d] * DF.destiny[d];
        DF.dest_mag = sqrtf(n + 1e-12f);
    }

    /* context window */
    if (DF.ctx_len < DF_MAX_CTX)
        DF.context[DF.ctx_len++] = tok;
    else {
        memmove(DF.context, DF.context + 1, (DF_MAX_CTX - 1) * sizeof(int));
        DF.context[DF_MAX_CTX - 1] = tok;
    }

    DF.trauma *= 0.97f;
    DF.step++;
}

/* apply Dario field overlay to logits — THE EQUATION */
static void df_overlay(float *logits, int V) {
    df_chambers();

    float eff_a = DF.alpha_mod * DF.alpha;
    float eff_b = DF.beta_mod * DF.beta;
    float eff_g = DF.gamma_mod * DF.gamma_d;
    if (DF.trauma > 0.3f) eff_g += DF.trauma * 0.8f;

    /* H: Hebbian resonance from co-occurrence */
    float h_max = 0;
    static float H_sig_buf[VOCAB];
    float *H_sig = H_sig_buf;
    memset(H_sig, 0, V * sizeof(float));
    int ctx_start = (DF.ctx_len > 8) ? DF.ctx_len - 8 : 0;
    for (int c = ctx_start; c < DF.ctx_len; c++) {
        float decay = powf(0.9f, (float)(DF.ctx_len - 1 - c));
        for (int i = 0; i < DF.cooc_n; i++)
            if (DF.cooc_src[i] == DF.context[c] && DF.cooc_dst[i] < V)
                H_sig[DF.cooc_dst[i]] += DF.cooc_val[i] * decay;
    }
    for (int i = 0; i < V; i++) if (H_sig[i] > h_max) h_max = H_sig[i];
    if (h_max > 1e-6f) for (int i = 0; i < V; i++) H_sig[i] /= h_max;

    /* F: Prophecy fulfillment */
    float f_max = 0;
    static float F_sig_buf[VOCAB];
    float *F_sig = F_sig_buf;
    memset(F_sig, 0, V * sizeof(float));
    for (int i = 0; i < V; i++) {
        float *te = df_embed(i);
        if (!te) continue;
        float score = 0;
        for (int p = 0; p < DF.prophecy_n; p++) {
            DFProphecy *pr = &DF.prophecy[p];
            if (pr->fulfilled) continue;
            float *pe = df_embed(pr->target);
            if (!pe) continue;
            float sim = df_cosine(te, pe);
            if (sim < 0) sim = 0;
            score += pr->strength * sim * logf(1.0f + (float)pr->age);
        }
        F_sig[i] = score;
    }
    for (int i = 0; i < V; i++) if (F_sig[i] > f_max) f_max = F_sig[i];
    if (f_max > 1e-6f) for (int i = 0; i < V; i++) F_sig[i] /= f_max;

    /* A: Destiny attraction */
    static float A_sig_buf[VOCAB];
    float *A_sig = A_sig_buf;
    memset(A_sig, 0, V * sizeof(float));
    if (DF.dest_mag > 1e-6f) {
        float a_max = 0;
        for (int i = 0; i < V; i++) {
            float *te = df_embed(i);
            if (te) A_sig[i] = df_cosine(te, DF.destiny) * DF.dest_mag;
        }
        for (int i = 0; i < V; i++)
            if (fabsf(A_sig[i]) > a_max) a_max = fabsf(A_sig[i]);
        if (a_max > 1e-6f) for (int i = 0; i < V; i++) A_sig[i] /= a_max;
    }

    /* SwiGLU gate through resonance */
    float gate = 1.0f / (1.0f + expf(-(DF.resonance - 0.5f) * 4.0f));
    float h_g = 1.0f / (1.0f + expf(-gate * 2.0f));
    float f_g = 1.0f / (1.0f + expf(-gate * 1.5f));

    /* T: Trauma gravity + combine */
    float t_boost = (DF.trauma > 0.3f) ? DF.trauma * 2.0f : 0;
    for (int i = 0; i < V; i++) {
        float h = eff_a * H_sig[i] * h_g;
        float f = eff_b * F_sig[i] * f_g;
        float a = eff_g * A_sig[i];
        float t = (i < 50) ? t_boost * (1.0f - (float)i / 50.0f) : 0;
        logits[i] += h + f + a + t;
    }

    /* update field metrics */
    float density = (DF.cooc_n > 100) ? 1.0f : (float)DF.cooc_n / 100.0f;
    DF.resonance = df_clamp(density * 0.4f + (1.0f - DF.dissonance) * 0.3f + 0.3f, 0, 0.95f);
    DF.entropy = df_clamp(DF.dissonance * 0.4f + (1.0f - DF.resonance) * 0.3f + 0.2f, 0.1f, 1.0f);
    DF.emergence = df_clamp((1.0f - DF.entropy) * DF.resonance, 0, 1);

    /* static buffers — no free needed */
}

/* ═══════════════════════════════════════════════════════════════════
 * MODEL — Hybrid RRPRAM + Content + SwiGLU + Dario
 *
 * Per head (hybrid): Wq[E,D] + Wk[E,D] + Wv[E,D] + Wr[E,T] + gate[1]
 * MLP (SwiGLU): W_gate[E,M] + W_up[E,M] + W_down[M,E]
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *tok_emb;   /* [V, E] */
    float *pos_emb;   /* [T, E] */
    float *rms1[MAX_BLK];     /* [E] */
    float *attn_w[MAX_BLK];   /* per block: H * (E*D*3 + E*T + 1) */
    float *wo[MAX_BLK];       /* [E, E] */
    float *rms2[MAX_BLK];     /* [E] */
    float *w_gate[MAX_BLK];   /* [E, M] — SwiGLU gate */
    float *w_up[MAX_BLK];     /* [E, M] — SwiGLU up */
    float *w_down[MAX_BLK];   /* [M, E] — SwiGLU down */
    float *rms_f;              /* [E] final */
    float *out_w;              /* [E, V] */
} Ptrs;

static int hw_size(Cfg *c) {
    return 3 * c->E * c->D + c->E * c->T + 1; /* wq+wk+wv+wr+gate */
}

static int model_size(Cfg *c) {
    int s = VOCAB * c->E + c->T * c->E;  /* embeddings */
    int hw = hw_size(c);
    for (int b = 0; b < c->B; b++) {
        s += c->E;                  /* rms1 */
        s += c->H * hw;            /* attention */
        s += c->E * c->E;          /* wo */
        s += c->E;                  /* rms2 */
        s += c->E * c->M;          /* w_gate */
        s += c->E * c->M;          /* w_up */
        s += c->M * c->E;          /* w_down */
    }
    s += c->E;                      /* rms_f */
    s += c->E * VOCAB;              /* out_w */
    return s;
}

static void assign_ptrs(Ptrs *p, float *base, Cfg *c) {
    float *q = base;
    int hw = hw_size(c);
    p->tok_emb = q; q += VOCAB * c->E;
    p->pos_emb = q; q += c->T * c->E;
    for (int b = 0; b < c->B; b++) {
        p->rms1[b] = q; q += c->E;
        p->attn_w[b] = q; q += c->H * hw;
        p->wo[b] = q; q += c->E * c->E;
        p->rms2[b] = q; q += c->E;
        p->w_gate[b] = q; q += c->E * c->M;
        p->w_up[b] = q; q += c->E * c->M;
        p->w_down[b] = q; q += c->M * c->E;
    }
    p->rms_f = q; q += c->E;
    p->out_w = q;
}

/* head weight accessors */
static float *h_wq(Ptrs *p, Cfg *c, int b, int h) { return p->attn_w[b] + h * hw_size(c); }
static float *h_wk(Ptrs *p, Cfg *c, int b, int h) { return h_wq(p,c,b,h) + c->E * c->D; }
static float *h_wv(Ptrs *p, Cfg *c, int b, int h) { return h_wk(p,c,b,h) + c->E * c->D; }
static float *h_wr(Ptrs *p, Cfg *c, int b, int h) { return h_wv(p,c,b,h) + c->E * c->D; }
static float *h_gate(Ptrs *p, Cfg *c, int b, int h) { return h_wr(p,c,b,h) + c->E * c->T; }

typedef struct {
    Cfg cfg;
    int n_params;
    float *data, *grad, *adam_m, *adam_v;
    Ptrs w, g;
} Model;

static void model_init(Model *m, int depth) {
    if (depth < 1 || depth > MAX_BLK || depth * 32 > MAX_DIM) {
        fprintf(stderr, "[resonance] error: depth=%d out of range\n", depth);
        exit(EXIT_FAILURE);
    }
    m->cfg = cfg_from_depth(depth);
    Cfg *c = &m->cfg;
    if (c->T > MAX_CTX || c->E > MAX_DIM || c->H < 1 || c->E % c->H != 0) {
        fprintf(stderr, "[resonance] error: invalid config for depth=%d\n", depth);
        exit(EXIT_FAILURE);
    }
    m->n_params = model_size(&m->cfg);
    m->data  = calloc(m->n_params, sizeof(float));
    m->grad  = calloc(m->n_params, sizeof(float));
    m->adam_m = calloc(m->n_params, sizeof(float));
    m->adam_v = calloc(m->n_params, sizeof(float));
    assign_ptrs(&m->w, m->data, &m->cfg);
    assign_ptrs(&m->g, m->grad, &m->cfg);
    float scale = sqrtf(2.0f / VOCAB);
    for (int i = 0; i < VOCAB * c->E; i++) m->w.tok_emb[i] = randn() * scale;
    scale = sqrtf(2.0f / c->T);
    for (int i = 0; i < c->T * c->E; i++) m->w.pos_emb[i] = randn() * scale;

    for (int b = 0; b < c->B; b++) {
        for (int e = 0; e < c->E; e++) m->w.rms1[b][e] = 1.0f;
        for (int e = 0; e < c->E; e++) m->w.rms2[b][e] = 1.0f;

        scale = sqrtf(2.0f / c->E);
        for (int h = 0; h < c->H; h++) {
            int sz = hw_size(c) - 1; /* all except gate */
            float *base = m->w.attn_w[b] + h * hw_size(c);
            for (int i = 0; i < sz; i++) base[i] = randn() * scale;
            *h_gate(&m->w, c, b, h) = 0.5f; /* init gate balanced */
        }

        for (int i = 0; i < c->E * c->E; i++)
            m->w.wo[b][i] = randn() * scale / sqrtf(c->B);
        /* SwiGLU: init gate/up/down */
        for (int i = 0; i < c->E * c->M; i++) m->w.w_gate[b][i] = randn() * scale;
        for (int i = 0; i < c->E * c->M; i++) m->w.w_up[b][i] = randn() * scale;
        scale = sqrtf(2.0f / c->M);
        for (int i = 0; i < c->M * c->E; i++)
            m->w.w_down[b][i] = randn() * scale / sqrtf(c->B);
    }
    for (int e = 0; e < c->E; e++) m->w.rms_f[e] = 1.0f;
    scale = sqrtf(2.0f / c->E);
    for (int i = 0; i < c->E * VOCAB; i++) m->w.out_w[i] = randn() * scale;

    printf("[resonance] θ = ε + γ + αδ\n");
    printf("[resonance] depth=%d params=%d (%.1fK)\n", depth, m->n_params, m->n_params/1000.0f);
    printf("[resonance] T=%d E=%d H=%d D=%d B=%d M=%d\n",
           c->T, c->E, c->H, c->D, c->B, c->M);
    printf("[resonance] hybrid attention (RRPRAM+content) + SwiGLU + Dario field\n");
}

/* ═══════════════════════════════════════════════════════════════════
 * ACTIVATION CACHE
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *x;          /* [(B+1)*T*E] */
    float *rms1;       /* [B*T*E] */
    float *v, *q, *k;  /* [B*H*T*D] each */
    float *attn_r;     /* [B*H*T*T] RRPRAM */
    float *attn_c;     /* [B*H*T*T] Content */
    float *concat;     /* [B*T*E] */
    float *r1;         /* [B*T*E] */
    float *rms2;       /* [B*T*E] */
    float *gate_pre;   /* [B*T*M] SwiGLU gate pre-activation */
    float *up_pre;     /* [B*T*M] SwiGLU up pre-activation */
    float *fln;        /* [T*E] */
    float *logits;     /* [T*V] */
    float *probs;      /* [T*V] */
} Acts;

static void acts_alloc(Acts *a, Cfg *c) {
    int TE = c->T * c->E, TT = c->T * c->T, TD = c->T * c->D, TM = c->T * c->M;
    a->x        = calloc((c->B+1) * TE, sizeof(float));
    a->rms1     = calloc(c->B * TE, sizeof(float));
    a->v        = calloc(c->B * c->H * TD, sizeof(float));
    a->q        = calloc(c->B * c->H * TD, sizeof(float));
    a->k        = calloc(c->B * c->H * TD, sizeof(float));
    a->attn_r   = calloc(c->B * c->H * TT, sizeof(float));
    a->attn_c   = calloc(c->B * c->H * TT, sizeof(float));
    a->concat   = calloc(c->B * TE, sizeof(float));
    a->r1       = calloc(c->B * TE, sizeof(float));
    a->rms2     = calloc(c->B * TE, sizeof(float));
    a->gate_pre = calloc(c->B * TM, sizeof(float));
    a->up_pre   = calloc(c->B * TM, sizeof(float));
    a->fln      = calloc(TE, sizeof(float));
    a->logits   = calloc(c->T * VOCAB, sizeof(float));
    a->probs    = calloc(c->T * VOCAB, sizeof(float));
}

static void acts_free(Acts *a) {
    free(a->x); free(a->rms1); free(a->v); free(a->q); free(a->k);
    free(a->attn_r); free(a->attn_c); free(a->concat); free(a->r1);
    free(a->rms2); free(a->gate_pre); free(a->up_pre);
    free(a->fln); free(a->logits); free(a->probs);
}

/* ═══════════════════════════════════════════════════════════════════
 * FORWARD
 * ═══════════════════════════════════════════════════════════════════ */

static void causal_softmax(float *a, int T) {
    for (int i = 0; i < T; i++) {
        for (int j = i+1; j < T; j++) a[i*T+j] = -1e9f;
        row_softmax(a + i*T, T);
    }
}

static float forward(Model *m, Acts *a, const int *tokens, const int *targets) {
    Cfg *c = &m->cfg;
    int T = c->T, E = c->E, H = c->H, D = c->D, B = c->B, M = c->M;
    Ptrs *w = &m->w;
    float *hbuf = calloc(T * D, sizeof(float));
    float sc = 1.0f / sqrtf((float)D);

    /* embedding */
    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++)
            a->x[t*E+e] = w->tok_emb[tokens[t]*E+e] + w->pos_emb[t*E+e];

    for (int b = 0; b < B; b++) {
        float *xin  = a->x + b * T * E;
        float *xout = a->x + (b+1) * T * E;
        float *rm1  = a->rms1 + b * T * E;
        float *cat  = a->concat + b * T * E;
        float *r1   = a->r1 + b * T * E;
        float *rm2  = a->rms2 + b * T * E;
        float *gp   = a->gate_pre + b * T * M;
        float *up   = a->up_pre + b * T * M;

        rmsnorm_fwd(rm1, xin, w->rms1[b], T, E);

        /* hybrid multi-head attention */
        memset(cat, 0, T * E * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *wq = h_wq(w,c,b,h), *wk = h_wk(w,c,b,h);
            float *wv = h_wv(w,c,b,h), *wr = h_wr(w,c,b,h);
            float alpha = *h_gate(w,c,b,h);
            float *v_h = a->v + (b*H+h)*T*D;
            float *q_h = a->q + (b*H+h)*T*D;
            float *k_h = a->k + (b*H+h)*T*D;
            float *ar = a->attn_r + (b*H+h)*T*T;
            float *ac = a->attn_c + (b*H+h)*T*T;

            matmul(v_h, rm1, wv, T, E, D);
            matmul(q_h, rm1, wq, T, E, D);
            matmul(k_h, rm1, wk, T, E, D);
            matmul(ar, rm1, wr, T, E, T);

            /* RRPRAM path */
            for (int i = 0; i < T*T; i++) ar[i] *= sc;
            causal_softmax(ar, T);
            float *out_r = calloc(T*D, sizeof(float));
            matmul(out_r, ar, v_h, T, T, D);

            /* Content path */
            matmul_abt(ac, q_h, k_h, T, D, T);
            for (int i = 0; i < T*T; i++) ac[i] *= sc;
            causal_softmax(ac, T);
            float *out_c = calloc(T*D, sizeof(float));
            matmul(out_c, ac, v_h, T, T, D);

            /* gated combination */
            for (int i = 0; i < T*D; i++)
                hbuf[i] = alpha * out_r[i] + (1.0f - alpha) * out_c[i];

            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    cat[t*E + h*D + d] = hbuf[t*D + d];

            free(out_r); free(out_c);
        }

        /* output projection + residual */
        float *wo_buf = calloc(T*E, sizeof(float));
        matmul(wo_buf, cat, w->wo[b], T, E, E);
        for (int i = 0; i < T*E; i++) r1[i] = xin[i] + wo_buf[i];
        free(wo_buf);

        /* RMSNorm + SwiGLU MLP */
        rmsnorm_fwd(rm2, r1, w->rms2[b], T, E);
        matmul(gp, rm2, w->w_gate[b], T, E, M);  /* gate */
        matmul(up, rm2, w->w_up[b], T, E, M);    /* up */

        /* SwiGLU: SiLU(gate) * up */
        float *swiglu_out = calloc(T*M, sizeof(float));
        for (int i = 0; i < T*M; i++)
            swiglu_out[i] = silu_f(gp[i]) * up[i];

        float *mlp_out = calloc(T*E, sizeof(float));
        matmul(mlp_out, swiglu_out, w->w_down[b], T, M, E);
        for (int i = 0; i < T*E; i++) xout[i] = r1[i] + mlp_out[i];

        free(swiglu_out); free(mlp_out);
    }

    /* final norm + projection */
    rmsnorm_fwd(a->fln, a->x + B*T*E, w->rms_f, T, E);
    matmul(a->logits, a->fln, w->out_w, T, E, VOCAB);

    /* Dario field overlay on logits — THE EQUATION */
    for (int t = 0; t < T; t++)
        df_overlay(a->logits + t * VOCAB, VOCAB);

    /* softmax + loss */
    float loss = 0;
    for (int t = 0; t < T; t++) {
        memcpy(a->probs + t*VOCAB, a->logits + t*VOCAB, VOCAB * sizeof(float));
        row_softmax(a->probs + t*VOCAB, VOCAB);
        if (targets) {
            float p = a->probs[t*VOCAB + targets[t]];
            loss -= logf(p > 1e-10f ? p : 1e-10f);
        }
    }
    free(hbuf);
    return targets ? loss / T : 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * BACKWARD — analytical gradients through hybrid attention + SwiGLU
 * ═══════════════════════════════════════════════════════════════════ */

static void backward(Model *m, Acts *a, const int *tokens, const int *targets) {
    Cfg *c = &m->cfg;
    int T = c->T, E = c->E, H = c->H, D = c->D, B = c->B, M = c->M;
    Ptrs *w = &m->w, *g = &m->g;
    float sc = 1.0f / sqrtf((float)D);
    memset(m->grad, 0, m->n_params * sizeof(float));

    float *dx = calloc(T*E, sizeof(float));
    float *d_rms = calloc(T*E, sizeof(float));
    float *d_cat = calloc(T*E, sizeof(float));
    float *d_head = calloc(T*D, sizeof(float));
    float *d_pat = calloc(T*T, sizeof(float));
    float *d_raw = calloc(T*T, sizeof(float));
    float *d_v = calloc(T*D, sizeof(float));

    /* d_logits */
    float *dl = calloc(T*VOCAB, sizeof(float));
    for (int t = 0; t < T; t++)
        for (int v = 0; v < VOCAB; v++)
            dl[t*VOCAB+v] = (a->probs[t*VOCAB+v] - (v == targets[t] ? 1.0f : 0.0f)) / T;

    /* output projection */
    float *d_fln = calloc(T*E, sizeof(float));
    matmul_abt(d_fln, dl, w->out_w, T, VOCAB, E);
    matmul_atb(g->out_w, a->fln, dl, E, T, VOCAB);

    /* final RMSNorm */
    rmsnorm_bwd(dx, g->rms_f, d_fln, a->x + B*T*E, w->rms_f, T, E);

    for (int b = B-1; b >= 0; b--) {
        float *xin = a->x + b*T*E;
        float *rm1 = a->rms1 + b*T*E;
        float *cat = a->concat + b*T*E;
        float *r1  = a->r1 + b*T*E;
        float *rm2 = a->rms2 + b*T*E;
        float *gp  = a->gate_pre + b*T*M;
        float *up  = a->up_pre + b*T*M;

        /* SwiGLU backward: out = silu(gate) * up @ w_down */
        /* d_swiglu = dx @ w_down^T */
        float *d_swiglu = calloc(T*M, sizeof(float));
        matmul_abt(d_swiglu, dx, w->w_down[b], T, E, M);

        /* recompute swiglu_out for w_down gradient */
        float *swiglu_out = calloc(T*M, sizeof(float));
        for (int i = 0; i < T*M; i++) swiglu_out[i] = silu_f(gp[i]) * up[i];

        /* d_w_down = swiglu^T @ dx */
        matmul_atb(g->w_down[b], swiglu_out, dx, M, T, E);

        /* d_gate_pre = d_swiglu * up * silu'(gate) */
        /* d_up_pre = d_swiglu * silu(gate) */
        float *d_gp = calloc(T*M, sizeof(float));
        float *d_up = calloc(T*M, sizeof(float));
        for (int i = 0; i < T*M; i++) {
            float sg = silu_f(gp[i]);
            d_up[i] = d_swiglu[i] * sg;
            d_gp[i] = d_swiglu[i] * up[i] * silu_grad(gp[i]);
        }

        /* d_w_gate = rm2^T @ d_gp, d_w_up = rm2^T @ d_up */
        matmul_atb(g->w_gate[b], rm2, d_gp, E, T, M);
        matmul_atb(g->w_up[b], rm2, d_up, E, T, M);

        /* d_rm2 = d_gp @ w_gate^T + d_up @ w_up^T */
        float *d_rm2a = calloc(T*E, sizeof(float));
        float *d_rm2b = calloc(T*E, sizeof(float));
        matmul_abt(d_rm2a, d_gp, w->w_gate[b], T, M, E);
        matmul_abt(d_rm2b, d_up, w->w_up[b], T, M, E);
        memset(d_rms, 0, T*E*sizeof(float));
        for (int i = 0; i < T*E; i++) d_rms[i] = d_rm2a[i] + d_rm2b[i];

        free(d_swiglu); free(swiglu_out); free(d_gp); free(d_up);
        free(d_rm2a); free(d_rm2b);

        /* RMSNorm2 backward */
        float *d_r1 = calloc(T*E, sizeof(float));
        rmsnorm_bwd(d_r1, g->rms2[b], d_rms, r1, w->rms2[b], T, E);
        for (int i = 0; i < T*E; i++) d_r1[i] += dx[i]; /* residual */

        /* output projection */
        matmul_abt(d_cat, d_r1, w->wo[b], T, E, E);
        matmul_atb(g->wo[b], cat, d_r1, E, T, E);

        /* hybrid attention backward */
        memset(d_rms, 0, T*E*sizeof(float));
        for (int h = 0; h < H; h++) {
            float *wq = h_wq(w,c,b,h), *wk = h_wk(w,c,b,h);
            float *wv = h_wv(w,c,b,h), *wr = h_wr(w,c,b,h);
            float alpha = *h_gate(w,c,b,h);
            float *v_h = a->v + (b*H+h)*T*D;
            float *q_h = a->q + (b*H+h)*T*D;
            float *k_h = a->k + (b*H+h)*T*D;
            float *ar = a->attn_r + (b*H+h)*T*T;
            float *ac = a->attn_c + (b*H+h)*T*T;
            float *g_wq = h_wq(g,c,b,h), *g_wk = h_wk(g,c,b,h);
            float *g_wv = h_wv(g,c,b,h), *g_wr = h_wr(g,c,b,h);

            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    d_head[t*D+d] = d_cat[t*E + h*D + d];

            /* recompute outputs for gate gradient */
            float *out_r = calloc(T*D, sizeof(float));
            float *out_c = calloc(T*D, sizeof(float));
            matmul(out_r, ar, v_h, T, T, D);
            matmul(out_c, ac, v_h, T, T, D);

            /* gate gradient */
            float dg = 0;
            for (int i = 0; i < T*D; i++) dg += d_head[i] * (out_r[i] - out_c[i]);
            *h_gate(g,c,b,h) += dg;

            /* split d_head by gate */
            float *d_or = calloc(T*D, sizeof(float));
            float *d_oc = calloc(T*D, sizeof(float));
            for (int i = 0; i < T*D; i++) {
                d_or[i] = alpha * d_head[i];
                d_oc[i] = (1.0f - alpha) * d_head[i];
            }

            /* --- RRPRAM backward --- */
            matmul_abt(d_pat, d_or, v_h, T, D, T);
            float *dv_r = calloc(T*D, sizeof(float));
            matmul_atb(dv_r, ar, d_or, T, T, D);
            for (int i = 0; i < T; i++) {
                float dot = 0;
                for (int j = 0; j <= i; j++) dot += ar[i*T+j] * d_pat[i*T+j];
                for (int j = 0; j <= i; j++)
                    d_raw[i*T+j] = ar[i*T+j] * (d_pat[i*T+j] - dot);
                for (int j = i+1; j < T; j++) d_raw[i*T+j] = 0;
            }
            for (int i = 0; i < T*T; i++) d_raw[i] *= sc;
            matmul_atb(g_wr, rm1, d_raw, E, T, T);
            float *dl_wr = calloc(T*E, sizeof(float));
            matmul_abt(dl_wr, d_raw, wr, T, T, E);

            /* --- Content backward --- */
            float *dp_c = calloc(T*T, sizeof(float));
            float *dr_c = calloc(T*T, sizeof(float));
            matmul_abt(dp_c, d_oc, v_h, T, D, T);
            float *dv_c = calloc(T*D, sizeof(float));
            matmul_atb(dv_c, ac, d_oc, T, T, D);
            for (int i = 0; i < T; i++) {
                float dot = 0;
                for (int j = 0; j <= i; j++) dot += ac[i*T+j] * dp_c[i*T+j];
                for (int j = 0; j <= i; j++)
                    dr_c[i*T+j] = ac[i*T+j] * (dp_c[i*T+j] - dot);
                for (int j = i+1; j < T; j++) dr_c[i*T+j] = 0;
            }
            for (int i = 0; i < T*T; i++) dr_c[i] *= sc;
            float *d_q = calloc(T*D, sizeof(float));
            float *d_k = calloc(T*D, sizeof(float));
            matmul(d_q, dr_c, k_h, T, T, D);
            matmul_atb(d_k, dr_c, q_h, T, T, D);

            matmul_atb(g_wq, rm1, d_q, E, T, D);
            matmul_atb(g_wk, rm1, d_k, E, T, D);

            /* combined d_v */
            for (int i = 0; i < T*D; i++) d_v[i] = dv_r[i] + dv_c[i];
            matmul_atb(g_wv, rm1, d_v, E, T, D);

            /* d_rms1 accumulation */
            float *dl_q = calloc(T*E, sizeof(float));
            float *dl_k = calloc(T*E, sizeof(float));
            float *dl_v = calloc(T*E, sizeof(float));
            matmul_abt(dl_q, d_q, wq, T, D, E);
            matmul_abt(dl_k, d_k, wk, T, D, E);
            matmul_abt(dl_v, d_v, wv, T, D, E);
            for (int i = 0; i < T*E; i++)
                d_rms[i] += dl_wr[i] + dl_q[i] + dl_k[i] + dl_v[i];

            free(out_r); free(out_c); free(d_or); free(d_oc);
            free(dv_r); free(dl_wr); free(dp_c); free(dr_c);
            free(dv_c); free(d_q); free(d_k); free(dl_q); free(dl_k); free(dl_v);
        }

        /* RMSNorm1 backward */
        float *d_xin = calloc(T*E, sizeof(float));
        rmsnorm_bwd(d_xin, g->rms1[b], d_rms, xin, w->rms1[b], T, E);
        for (int i = 0; i < T*E; i++) dx[i] = d_r1[i] + d_xin[i];

        free(d_r1); free(d_xin);
    }

    /* embedding backward */
    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++) {
            g->tok_emb[tokens[t]*E+e] += dx[t*E+e];
            g->pos_emb[t*E+e] += dx[t*E+e];
        }

    free(dx); free(d_rms); free(d_cat); free(d_head);
    free(d_pat); free(d_raw); free(d_v); free(dl); free(d_fln);
}

/* ═══════════════════════════════════════════════════════════════════
 * ADAM
 * ═══════════════════════════════════════════════════════════════════ */

static void adam_step(Model *m, float lr, int step) {
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, step);
    float bc2 = 1.0f - powf(b2, step);
    for (int i = 0; i < m->n_params; i++) {
        float g = m->grad[i];
        m->adam_m[i] = b1 * m->adam_m[i] + (1-b1) * g;
        m->adam_v[i] = b2 * m->adam_v[i] + (1-b2) * g * g;
        m->data[i] -= lr * (m->adam_m[i]/bc1) / (sqrtf(m->adam_v[i]/bc2) + eps);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * DATA + TRAIN + GENERATE
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct { unsigned char *bytes; int len; } Data;

static Data load_data(const char *path) {
    Data d = {0};
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); d.len = ftell(f); fseek(f, 0, SEEK_SET);
    d.bytes = malloc(d.len);
    fread(d.bytes, 1, d.len, f); fclose(f);
    printf("[resonance] data: %s %d bytes (%.1fKB)\n", path, d.len, d.len/1024.0f);
    return d;
}

static void train(Model *m, Data *data, int steps, float lr) {
    Acts a; acts_alloc(&a, &m->cfg);
    int T = m->cfg.T;
    int *tok = malloc(T * sizeof(int));
    int *tgt = malloc(T * sizeof(int));

    if (data->len <= T + 1) {
        fprintf(stderr, "[resonance] error: data too short (%d bytes, need > %d)\n",
                data->len, T + 1);
        exit(EXIT_FAILURE);
    }

    printf("[resonance] training: %d steps, lr=%.1e\n", steps, lr);
    clock_t t0 = clock();

    for (int s = 1; s <= steps; s++) {
        int off = rand() % (data->len - T - 1);
        for (int t = 0; t < T; t++) {
            tok[t] = data->bytes[off + t];
            tgt[t] = data->bytes[off + t + 1];
        }

        float loss = forward(m, &a, tok, tgt);
        backward(m, &a, tok, tgt);
        adam_step(m, lr, s);

        /* feed tokens into Dario field */
        for (int t = 0; t < T; t++) df_ingest(tok[t]);

        if (s % 100 == 0 || s == 1) {
            float elapsed = (float)(clock() - t0) / CLOCKS_PER_SEC;
            float avg_gate = 0;
            for (int b = 0; b < m->cfg.B; b++)
                for (int h = 0; h < m->cfg.H; h++)
                    avg_gate += *h_gate(&m->w, &m->cfg, b, h);
            avg_gate /= m->cfg.B * m->cfg.H;

            printf("  step %5d/%d  loss=%.4f  %.1f s/s  α=%.3f  field: res=%.2f ent=%.2f emg=%.2f cooc=%d\n",
                   s, steps, loss, s/elapsed, avg_gate,
                   DF.resonance, DF.entropy, DF.emergence, DF.cooc_n);
        }
    }

    acts_free(&a); free(tok); free(tgt);
}

static void generate(Model *m, const char *seed, int n, float temp) {
    Acts a; acts_alloc(&a, &m->cfg);
    int T = m->cfg.T;
    int *ctx = calloc(T, sizeof(int));

    int slen = strlen(seed);
    if (slen > T) slen = T;
    for (int i = 0; i < slen; i++)
        ctx[T - slen + i] = (unsigned char)seed[i];

    printf("%s", seed);
    for (int i = 0; i < n; i++) {
        forward(m, &a, ctx, NULL);
        float *logits = a.logits + (T-1) * VOCAB;
        int next = 0;
        if (temp <= 0.0f) {
            float best = logits[0];
            for (int v = 1; v < VOCAB; v++)
                if (logits[v] > best) { best = logits[v]; next = v; }
        } else {
            if (temp != 1.0f)
                for (int v = 0; v < VOCAB; v++) logits[v] /= temp;
            row_softmax(logits, VOCAB);
            float r = (float)rand() / RAND_MAX, cum = 0;
            for (int v = 0; v < VOCAB; v++) {
                cum += logits[v];
                if (cum >= r) { next = v; break; }
            }
        }

        putchar(next);
        df_ingest(next); /* field learns during generation too */
        memmove(ctx, ctx+1, (T-1)*sizeof(int));
        ctx[T-1] = next;
    }
    printf("\n");
    acts_free(&a); free(ctx);
}

/* ═══════════════════════════════════════════════════════════════════
 * SAVE / LOAD
 * ═══════════════════════════════════════════════════════════════════ */

static void model_save(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    fwrite(&m->cfg, sizeof(Cfg), 1, f);
    fwrite(m->data, sizeof(float), m->n_params, f);
    /* save Dario field state */
    fwrite(&DF, sizeof(DarioField), 1, f);
    fclose(f);
    printf("[resonance] saved %s (%d params + field, %.1fKB)\n",
           path, m->n_params, (m->n_params*4 + sizeof(DarioField))/1024.0f);
}

static void model_load(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    Cfg loaded;
    if (fread(&loaded, sizeof(Cfg), 1, f) != 1) {
        fprintf(stderr, "[resonance] failed to read config from %s\n", path);
        fclose(f); exit(1);
    }
    if (loaded.B < 1 || loaded.B > MAX_BLK ||
        loaded.T < 1 || loaded.T > MAX_CTX ||
        loaded.E < 1 || loaded.E > MAX_DIM ||
        loaded.H < 1 || loaded.E % loaded.H != 0) {
        fprintf(stderr, "[resonance] corrupt config in %s\n", path);
        fclose(f); exit(1);
    }
    model_init(m, loaded.B);
    m->cfg = loaded;
    m->n_params = model_size(&m->cfg);
    assign_ptrs(&m->w, m->data, &m->cfg);
    assign_ptrs(&m->g, m->grad, &m->cfg);
    if (fread(m->data, sizeof(float), m->n_params, f) != (size_t)m->n_params) {
        fprintf(stderr, "[resonance] truncated checkpoint %s\n", path);
        fclose(f); exit(1);
    }
    /* load Dario field state */
    if (fread(&DF, sizeof(DarioField), 1, f) != 1)
        printf("[resonance] warning: no field state in checkpoint\n");
    fclose(f);
    printf("[resonance] loaded %s (field: %d cooc entries)\n", path, DF.cooc_n);
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    int depth = 8, steps = 5000;
    float lr = 3e-4f, temp = 0.8f;
    int gen_chars = 500;
    const char *train_file = NULL, *load_file = NULL;
    const char *save_file = "resonance.bin";
    const char *seed = "The ";
    int do_gen = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--train") && i+1<argc) train_file = argv[++i];
        else if (!strcmp(argv[i],"--load") && i+1<argc) load_file = argv[++i];
        else if (!strcmp(argv[i],"--save") && i+1<argc) save_file = argv[++i];
        else if (!strcmp(argv[i],"--depth") && i+1<argc) depth = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--steps") && i+1<argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--lr") && i+1<argc) lr = atof(argv[++i]);
        else if (!strcmp(argv[i],"--temp") && i+1<argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i],"--chars") && i+1<argc) gen_chars = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--seed") && i+1<argc) seed = argv[++i];
        else if (!strcmp(argv[i],"--generate")) do_gen = 1;
    }

    printf("\n  resonance.c — Field Resonance Architecture\n");
    printf("  θ = ε + γ + αδ — pattern + meaning + field\n");
    printf("  by Arianna Method. הרזוננס לא נשבר\n\n");

    srand(time(NULL));
    df_init();
    Model m = {0};

    if (load_file) model_load(&m, load_file);
    else model_init(&m, depth);

    if (train_file) {
        Data d = load_data(train_file);
        train(&m, &d, steps, lr);
        model_save(&m, save_file);
        free(d.bytes);
        printf("\n--- sample ---\n");
        generate(&m, seed, gen_chars, temp);
    }

    if (do_gen) generate(&m, seed, gen_chars, temp);

    free(m.data); free(m.grad); free(m.adam_m); free(m.adam_v);
    return 0;
}
